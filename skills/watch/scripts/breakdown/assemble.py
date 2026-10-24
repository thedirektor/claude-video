"""Assemble every measurement into one breakdown (JSON + brief + contact sheet).

``build`` runs the measurement modules on the video's media and reconciles them
(shots vs transitions), ``brief_md`` renders the brief for a building agent,
``store`` writes ``breakdown.json`` / ``breakdown.md`` / ``contact.png`` next to
the video in the store, pins its media and indexes the brief for search.
Nothing here invents: a value that was not measured is the string
``"not determined"`` with ``confidence: 0``; LLM text lives only under keys
named ``interpretation``.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from config import memory_paths
from memory import index, store as mstore

from . import audio as audio_mod
from . import camera, look, media, shots as shots_mod, transitions, typography

SCHEMA = "watch.breakdown/1"
NOT_DETERMINED = "not determined"
METHODS = ("measured", "ocr", "transcript", "interpretation")
CONTRACT = ("> **Build contract:** follow only what this breakdown states. Anything marked _not determined_ "
            "stays an explicit placeholder in what you build. Never fill gaps with taste or assumptions.")
FORMATS = ("otio", "fcpxml", "edl")
NOTE_KIND = "breakdown"
NOTE_CHARS = 2000
MAX_TILES = 48
EPS = 1e-6
BRIEF_MIN_TEXT_CONF = 0.5  # OCR tracks below this are listed in breakdown.json only
SAMPLING_NOTE = f"sampled every {typography.STEP_S:g} s; shorter or stylised text may be missed"
BLIND_SPOT_NOTE = ("dissolves between two moving shots can go undetected and jump cuts within one framing "
                   "may be merged: cuts and shots are what was measured")
HOLD_MIN_S = 0.5  # a black / white hold this long between two fades is its own shot (0.7.0 ruling)
FOLD_MAX_S = 1.0  # a fade between two cuts this close is one dip, not a shot


# --------------------------------------------------------------------------- helpers

def _ev(t0: float, t1: float, method: str, conf: float, **extra) -> dict:
    return {"t_start": round(float(t0), 6), "t_end": round(float(t1), 6), "method": method,
            "confidence": round(max(0.0, min(1.0, float(conf))), 3), **extra}


def _nd_value(duration: float, **extra) -> dict:
    return {"value": NOT_DETERMINED, "evidence": _ev(0.0, duration or 0.0, "measured", 0.0), **extra}


def _add_note(notes: list[str], text: str) -> None:
    if text and text not in notes:
        notes.append(text)


# --------------------------------------------------------------------------- reconcile

def _fade_spans(entries: list[dict]) -> tuple[list[tuple[float, float, dict | None]], set[int]]:
    """Dip / fade spans from classify's entries.

    A dip is fade_to_X followed by fade_from_X; it gets one composite span and
    consumes both entries. Returns (spans (lo, hi, composite-or-None), consumed ids)."""
    spans: list[tuple[float, float, dict | None]] = []
    consumed: set[int] = set()
    fades = [e for e in entries if e["kind"].startswith("fade_")]
    used: set[int] = set()
    for i, e in enumerate(fades):
        if not e["kind"].startswith("fade_to_") or id(e) in used:
            continue
        color = e["kind"][len("fade_to_"):]
        for f in fades[i + 1:]:
            if f["kind"] == f"fade_from_{color}" and id(f) not in used:
                if f["evidence"]["t_start"] - e["evidence"]["t_end"] >= HOLD_MIN_S:
                    break  # a real hold: two fades around a shot of their own (see _retime_holds)
                lo, hi = e["evidence"]["t_start"], f["evidence"]["t_end"]
                comp = {"t": round((e["evidence"]["t_end"] + f["evidence"]["t_start"]) / 2, 6),
                        "kind": f"dip_{color}", "duration": round(hi - lo, 6),
                        "evidence": _ev(lo, hi, "measured",
                                        min(e["evidence"]["confidence"], f["evidence"]["confidence"])),
                        "components": [e, f]}
                spans.append((lo, hi, comp))
                used.update((id(e), id(f)))
                consumed.update((id(e), id(f)))
                break
    for e in fades:
        if id(e) not in used:
            spans.append((e["evidence"]["t_start"], e["evidence"]["t_end"], None))
    return spans, consumed


def _retime_holds(raw_shots: list[dict], entries: list[dict]) -> tuple[list[dict], list[dict]]:
    """A black / white hold of >= HOLD_MIN_S between a fade-to and a fade-from is a shot an NLE
    timeline needs (0.7.0 black-hold ruling). Its boundaries go to where the fade-to ends and the
    fade-from starts (shot detection puts them at the ramp centres), carrying those fades as their
    transitions; boundaries inside the dip are dropped. Dips without a hold are left to the fold."""
    pairs = []
    fades = sorted((e for e in entries if e["kind"].startswith("fade_")), key=lambda e: e["evidence"]["t_start"])
    used: set[int] = set()
    for e in fades:
        if not e["kind"].startswith("fade_to_") or id(e) in used:
            continue
        color = e["kind"][len("fade_to_"):]
        f = next((g for g in fades if g["kind"] == f"fade_from_{color}" and id(g) not in used
                  and g["evidence"]["t_start"] >= e["evidence"]["t_end"]), None)
        if f is not None and f["evidence"]["t_start"] - e["evidence"]["t_end"] >= HOLD_MIN_S:
            pairs.append((e, f))
            used.update((id(e), id(f)))
    if not pairs or not raw_shots:
        return raw_shots, entries
    bounds = []
    for sh in raw_shots:
        b = sh.get("boundary")
        if b:
            bounds.append({"kind": b["kind"], "t": b["t"], "frames": b["frames"],
                           "t_start": b["evidence"]["t_start"], "t_end": b["evidence"]["t_end"],
                           "confidence": b["evidence"]["confidence"]})
    new_entries = list(entries)
    for e, f in pairs:
        lo, hi = e["evidence"]["t_start"], f["evidence"]["t_end"]
        bounds = [b for b in bounds if not (lo - EPS <= b["t"] <= hi + EPS)]
        for fade, t in ((e, e["evidence"]["t_end"]), (f, f["evidence"]["t_start"])):
            bounds.append({"kind": "fade", "t": t, "frames": 0, "t_start": fade["evidence"]["t_start"],
                           "t_end": fade["evidence"]["t_end"], "confidence": fade["evidence"]["confidence"]})
            new_entries[new_entries.index(fade)] = {**fade, "t": round(t, 6)}
    bounds.sort(key=lambda b: b["t"])
    return shots_mod.build_shots(bounds, raw_shots[-1]["end"]), new_entries


def reconcile(raw_shots: list[dict], entries: list[dict]) -> tuple[list[dict], list[dict]]:
    """Merge shots whose boundary ``classify`` absorbed into a fade / dip.

    Every remaining boundary carries exactly one transition entry (a dip keeps
    ONE boundary with a composite ``dip_<color>`` entry). Returns
    (shots with ``transition``, fade entries not attached to a boundary)."""
    raw_shots, entries = _retime_holds(raw_shots, entries)
    spans, consumed = _fade_spans(entries)
    by_t = {round(e["t"], 6): e for e in entries if id(e) not in consumed}
    dip_kept: set[int] = set()
    kept: list[dict] = []
    attached: set[int] = set()
    for sh in raw_shots:
        b = sh.get("boundary")
        if not b:
            kept.append({**sh, "transition": None, "_conf": [sh["evidence"]["confidence"]]})
            continue
        entry = by_t.get(round(b["t"], 6))
        if entry is None:
            hit = next((k for k, (lo, hi, _) in enumerate(spans) if lo - EPS <= b["t"] <= hi + EPS), None)
            if hit is not None:
                comp = spans[hit][2]
                if comp is not None and hit not in dip_kept:
                    dip_kept.add(hit)
                    entry = comp
                    attached.update(id(c) for c in comp["components"])
                else:
                    kept[-1]["end"] = sh["end"]
                    kept[-1]["_conf"].append(sh["evidence"]["confidence"])
                    continue
            else:  # unmatched and not on a fade: keep it, flagged, never dropped silently
                entry = {"t": b["t"], "kind": NOT_DETERMINED, "duration": 0.0,
                         "evidence": _ev(b["evidence"]["t_start"], b["evidence"]["t_end"], "measured", 0.0)}
        else:
            attached.add(id(entry))
        if entry.get("components") and kept and kept[-1]["start"] < entry["t"] < sh["end"]:
            # the dip's t is where the shot starts: the boundary sat at the ramp centre
            kept[-1]["end"] = entry["t"]
            sh = {**sh, "start": entry["t"], "boundary": {"kind": "dip", "t": entry["t"], "frames": 0,
                                                          "evidence": entry["evidence"]}}
        kept.append({**sh, "transition": entry, "_conf": [sh["evidence"]["confidence"]]})
    loose_now = [e for e in entries if e["kind"].startswith("fade_") and id(e) not in consumed
                 and id(e) not in attached]
    i = 1
    while i < len(kept) - 1:
        mid, nxt = kept[i], kept[i + 1]
        fade = next((f for f in loose_now if f["evidence"]["t_start"] < nxt["start"] + EPS
                     and f["evidence"]["t_end"] > mid["start"] - EPS), None)
        if (mid["transition"] and nxt["transition"] and nxt["start"] - mid["start"] <= FOLD_MAX_S
                and fade is not None):
            color = fade["kind"].rsplit("_", 1)[1]
            cut = round((mid["start"] + nxt["start"]) / 2, 6)
            lo = min(mid["transition"]["evidence"]["t_start"], fade["evidence"]["t_start"])
            hi = max(nxt["transition"]["evidence"]["t_end"], fade["evidence"]["t_end"])
            comp = {"t": cut, "kind": f"dip_{color}", "duration": round(hi - lo, 6),
                    "evidence": _ev(lo, hi, "measured", min(0.85, fade["evidence"]["confidence"])),
                    "components": [mid["transition"], nxt["transition"], fade]}
            kept[i - 1]["end"] = cut
            kept[i - 1]["_conf"] += mid["_conf"]
            nxt["start"] = cut
            nxt["transition"] = comp
            nxt["boundary"] = {"kind": "dip", "t": cut, "frames": 0, "evidence": comp["evidence"]}
            nxt["_conf"] += mid["_conf"]
            attached.add(id(fade))
            loose_now.remove(fade)
            del kept[i]
        else:
            i += 1
    out = []
    for i, sh in enumerate(kept):
        start, end = sh["start"], sh["end"]
        conf = min(sh.pop("_conf"))
        sh.update({"index": i, "start": round(start, 6), "end": round(end, 6),
                   "duration": round(end - start, 6),
                   "evidence": _ev(start, end, "measured", conf)})
        out.append(sh)
    loose = [e for e in entries if e["kind"].startswith("fade_") and id(e) not in consumed
             and id(e) not in attached]
    return out, loose


# --------------------------------------------------------------------------- sections

def _active_aspect(width: int, height: int, looks: list[dict], duration: float) -> dict:
    measured = [s["look"] for s in looks if s.get("look") and s["look"].get("brightness") != NOT_DETERMINED]
    if not width or not height or not measured:
        return {"value": NOT_DETERMINED, "evidence": _ev(0.0, duration, "measured", 0.0)}
    bars = {k: sorted((lk.get("letterbox") or {}).get(k, 0.0) for lk in measured) for k in
            ("top", "bottom", "left", "right")}
    med = {k: v[len(v) // 2] for k, v in bars.items()}
    w = width * (1 - med["left"] - med["right"])
    h = height * (1 - med["top"] - med["bottom"])
    if w <= 0 or h <= 0:
        return {"value": NOT_DETERMINED, "evidence": _ev(0.0, duration, "measured", 0.0)}
    conf = min(lk["evidence"]["confidence"] for lk in measured)
    return {"value": round(w / h, 3), "width": int(round(w)), "height": int(round(h)), "bars": med,
            "evidence": _ev(0.0, duration, "measured", conf)}


def _overlap(a0: float, a1: float, spans: list[dict]) -> float:
    return sum(max(0.0, min(a1, s["end"]) - max(a0, s["start"])) for s in spans)


def _music_section(m: dict, a: dict, speech: list[dict], duration: float) -> dict:
    """Music presence per window from audio periodicity (measured), with speech overlap."""
    wins = audio_mod.music_windows(m["path"], duration) if m.get("path") and m.get("has_audio") else None
    if not wins:
        return {"present": NOT_DETERMINED, "segments": NOT_DETERMINED,
                "evidence": _ev(0.0, duration, "measured", 0.0)}
    segs: list[dict] = []
    for w in wins:
        if segs and segs[-1]["state"] == w["state"]:
            segs[-1]["t_end"] = w["t_end"]
            segs[-1]["_s"].append(w["strength"])
        else:
            segs.append({"t_start": w["t_start"], "t_end": w["t_end"], "state": w["state"], "_s": [w["strength"]]})
    for s in segs:
        st = s.pop("_s")
        conf = min(0.9, sum(st) / len(st)) if s["state"] == "music" else (0.9 if s["state"] == "silent" else 0.0)
        s["evidence"] = _ev(s["t_start"], s["t_end"], "measured", conf)
        s["overlaps_speech_s"] = round(_overlap(s["t_start"], s["t_end"], speech), 3) if speech else NOT_DETERMINED
    states = {s["state"] for s in segs}
    present = True if "music" in states else (False if states == {"silent"} else NOT_DETERMINED)
    confs = [s["evidence"]["confidence"] for s in segs if s["state"] == "music"]
    return {"present": present, "segments": segs,
            "evidence": _ev(0.0, duration, "measured", (sum(confs) / len(confs)) if confs else
                            (0.9 if present is False else 0.0))}


def _audio_section(m: dict, bd_shots: list[dict], duration: float, segments: list[dict],
                   notes: list[str]) -> dict:
    if m.get("path") and m.get("has_audio"):
        a = audio_mod.analyze(m["path"], duration)
    else:
        why = "no media (stored frames only)" if not m.get("path") else "no audio stream"
        a = {"tempo_bpm": NOT_DETERMINED, "beats": [], "loudness": None,
             "evidence": _ev(0.0, duration, "measured", 0.0),
             "loudness_evidence": _ev(0.0, duration, "measured", 0.0),
             "notes": [f"{why}; tempo, beats and loudness not determined"]}
    for n in a.get("notes") or []:
        _add_note(notes, n)
    beats = a.get("beats") or []
    cob = audio_mod.cuts_on_beat(bd_shots, beats, m.get("fps") or 0.0,
                                 tempo_conf=a["evidence"]["confidence"], alt_ratio=a.get("tempo_alt_ratio"))
    if isinstance(cob, str):
        cob = {"share": NOT_DETERMINED, "evidence": _ev(0.0, duration, "measured", 0.0)}
    a["cuts_on_beat"] = cob
    speech = [{"t_start": s["start"], "t_end": s["end"],
               "evidence": _ev(s["start"], s["end"], "transcript", 0.8)} for s in segments]
    segments = [{"start": s["start"], "end": s["end"]} for s in segments]
    music = _music_section(m, a, segments, duration)
    seg = {"speech": speech if speech else NOT_DETERMINED, "music": music,
           "evidence": _ev(0.0, duration, "transcript" if speech else "measured",
                           0.8 if speech else music["evidence"]["confidence"])}
    a["segments"] = seg
    return a


def _typography_section(m: dict, duration: float, notes: list[str]) -> dict:
    if not m.get("path"):
        _add_note(notes, "no media (stored frames only); typography not determined")
        return {"tracks": NOT_DETERMINED, "evidence": _ev(0.0, duration, "ocr", 0.0)}
    tracks, tnotes = typography.text_tracks(m["path"], duration)
    for n in tnotes:
        _add_note(notes, n)
    if tnotes:
        return {"tracks": NOT_DETERMINED, "evidence": _ev(0.0, duration, "ocr", 0.0)}
    conf = max((t["evidence"]["confidence"] for t in tracks), default=0.5)
    return {"tracks": tracks, "sampling": SAMPLING_NOTE,
            "evidence": _ev(0.0, duration, "ocr", conf, sampling=SAMPLING_NOTE)}


def _rhythm_section(shot_list: list[dict]) -> dict:
    r = shots_mod.rhythm(shot_list)
    for k in ("fastest_window", "slowest_window"):
        w = r.get(k)
        if isinstance(w, dict) and "evidence" not in w:
            w["evidence"] = _ev(w["t_start"], w["t_end"], "measured", r["evidence"]["confidence"])
    return r


def _interpret_payload(bd: dict) -> tuple[str, str]:
    facts = [{"t": [s["start"], s["end"]], "transition": (s.get("transition") or {}).get("kind"),
              "camera": s["camera"]["type"],
              "look": {k: s["look"].get(k) for k in ("brightness", "temperature")}}
             for s in bd["shots"]][:60]
    tracks = bd["typography"]["tracks"]
    system = ("You describe a video from MEASURED facts only. Write at most 5 short sentences about the "
              "visual style. Do not add anything not supported by the facts. Plain text.")
    user = json.dumps({"duration": bd["video"]["duration"], "shots": facts,
                       "text_on_screen": [t["text"] for t in tracks][:20] if isinstance(tracks, list) else []})
    return system, user


INTERPRET_MAX_TOKENS = 400


def estimate_interpret_usd(system: str, user: str) -> float:
    """Worst-case price of the one call: payload size / 4 tokens in, the output cap out,
    at the real price of the most expensive provider in the configured chain."""
    import cost
    from memory import llm
    from config import setting

    models = {"headroom": setting("WATCH_LLM_MODEL", "gemini-3.1-flash-lite"),
              "gemini": setting("WATCH_LLM_GEMINI_MODEL", "gemini-3.1-flash-lite"),
              "openrouter": setting("WATCH_LLM_OPENROUTER_MODEL", "google/gemini-2.5-flash-lite")}
    tin = (len(system) + len(user)) / 4.0
    chain = llm.chain() or ["gemini"]
    return max(cost._tokens_usd(f"{p}:{models[p]}", tin=tin, tout=INTERPRET_MAX_TOKENS) for p in chain)


def _interpret(bd: dict, complete, allow_cost: bool = False) -> dict | None:
    import cost
    import spend
    from memory import llm

    complete = complete or llm.complete
    system, user = _interpret_payload(bd)
    est, ceiling = estimate_interpret_usd(system, user), cost.ceiling()
    if not allow_cost and ceiling > 0 and est > ceiling:
        raise cost.CostCeilingExceeded(
            f"estimated interpretation cost ${est:.4f} exceeds WATCH_COST_CEILING_USD ${ceiling:.2f} - "
            "re-run with --allow-cost (or raise the ceiling).")
    reply = complete(system, user, max_tokens=INTERPRET_MAX_TOKENS)
    if reply is None or not (getattr(reply, "text", "") or "").strip():
        return None
    spend.record(reply.provider, "breakdown_interpretation", price_key=f"{reply.provider}:{reply.model}",
                 tokens_in=reply.tokens_in, tokens_out=reply.tokens_out)
    return {"text": reply.text.strip(), "made_from": "measured fields (no frames)",
            "provider": reply.provider, "model": reply.model,
            "evidence": _ev(0.0, bd["video"]["duration"] or 0.0, "interpretation", 0.3)}


# --------------------------------------------------------------------------- build

def build(conn, video_key: str, *, interpret: bool = False, store_dir=None, complete=None, fetch=None,
          allow_cost: bool = False) -> dict:
    row = conn.execute("SELECT title, url, duration FROM videos WHERE video_key = ?", (video_key,)).fetchone()
    title, url, row_dur = (row["title"], row["url"], row["duration"]) if row else (None, None, 0.0)
    m = media.ensure_media(conn, video_key, store_dir, fetch=fetch)
    if m.get("path"):
        media.pin(Path(store_dir) if store_dir is not None else Path(memory_paths()["store_dir"]), video_key)
    duration = float(m.get("duration") or row_dur or 0.0)
    fps = float(m.get("fps") or 0.0)
    notes: list[str] = []
    has_media = bool(m.get("path"))
    if m.get("fps_note"):
        _add_note(notes, m["fps_note"])
    if m.get("media_note"):
        _add_note(notes, m["media_note"])

    raw: list[dict] = []
    entries: list[dict] = []
    series: dict = {}
    if has_media and fps > 0:
        series = shots_mod.frame_series(m["path"], fps)
        raw = shots_mod.detect_shots(series, duration)
        if raw:
            entries = transitions.classify(series, raw)
            _add_note(notes, BLIND_SPOT_NOTE)
            if series.get("merged_same_picture"):
                _add_note(notes, f"{series['merged_same_picture']} cut candidate(s) merged as same-picture "
                                 "(a step inside one framing, not an edit)")
        else:
            _add_note(notes, "shot detection produced no shots; cuts not determined")
    else:
        _add_note(notes, "media not available (stored frames only); low resolution: cuts and shots not determined")
    shot_list, loose_fades = reconcile(raw, entries) if raw else ([], [])

    for sh in shot_list:
        _add_note(notes, (sh.get("boundary") or {}).get("note", ""))
        sh["camera"] = camera.camera_motion(m["path"], sh, crop=series.get("crop"))
        sh["look"] = look.shot_look(m["path"], sh)
        for n in (sh["camera"].get("notes") or []) + (sh["look"].get("notes") or []):
            _add_note(notes, n)

    width, height = int(m.get("width") or 0), int(m.get("height") or 0)
    video = {"key": video_key, "title": title, "url": url, "duration": duration, "fps": fps,
             "fps_rational": m.get("timeline_fps_rational") or m.get("fps_rational") or "0/1",
             "fps_measured_rational": m.get("fps_rational") or "0/1", "width": width, "height": height,
             "source": m.get("source"),
             "active_aspect": _active_aspect(width, height, shot_list, duration),
             "evidence": _ev(0.0, duration, "measured",
                             (0.45 if m.get("fps_offstandard") else 0.95) if has_media else 0.3)}

    segs = [dict(r) for r in conn.execute(
        'SELECT start, "end" AS end FROM segments WHERE video_key = ? ORDER BY start', (video_key,))]
    segs = [s for s in segs if s["start"] is not None and s["end"] is not None]

    bd = {
        "schema": SCHEMA,
        "contract": CONTRACT,
        "video": video,
        "low_resolution": bool(m.get("low_resolution")),
        "cuts": ({"count": sum(1 for s in shot_list[1:] if s["transition"]["kind"] == "cut"),
                  "boundaries": len(shot_list) - 1,
                  "evidence": _ev(0.0, duration, "measured",
                                  min(s["evidence"]["confidence"] for s in shot_list))}
                 if shot_list else _nd_value(duration)),
        "shots": shot_list,
        "fades": loose_fades,
        "rhythm": _rhythm_section(shot_list),
    }
    bd["typography"] = _typography_section(m, duration, notes)
    bd["audio"] = _audio_section(m, shot_list, duration, segs, notes)
    bd["notes"] = notes
    version = _evidence_version(conn, video_key)
    if interpret:
        interp = _interpret(bd, complete, allow_cost)
        if interp:
            bd["interpretation"] = {**interp, "evidence_version": version, "stale": False}
        else:
            _add_note(notes, "interpretation requested but no LLM reply was available")
    if "interpretation" not in bd:
        kept = _stored_interpretation(conn, video_key, store_dir)
        if kept:
            bd["interpretation"] = {**kept, "stale": kept.get("evidence_version") != version}
            _add_note(notes, "interpretation kept from an earlier run (not regenerated); measured fields may "
                             "have changed since")
    return bd


def _evidence_version(conn, video_key: str) -> int:
    row = conn.execute("SELECT evidence_version FROM videos WHERE video_key = ?", (video_key,)).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def stored_path(conn, video_key: str, store_dir=None) -> Path:
    root = Path(store_dir) if store_dir is not None else Path(memory_paths()["store_dir"])
    return breakdown_dir(conn, root, video_key) / "breakdown.json"


def read_stored(conn, video_key: str, store_dir=None) -> dict | None:
    """The stored breakdown.json, or None."""
    try:
        return json.loads(stored_path(conn, video_key, store_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _stored_interpretation(conn, video_key: str, store_dir=None) -> dict | None:
    old = read_stored(conn, video_key, store_dir)
    interp = (old or {}).get("interpretation")
    return interp if isinstance(interp, dict) and (interp.get("text") or "").strip() else None


# --------------------------------------------------------------------------- brief

def _t(v) -> str:
    return "_not determined_" if v == NOT_DETERMINED or v is None else f"{float(v):.2f}s"


def _v(v) -> str:
    return "_not determined_" if v == NOT_DETERMINED or v is None else str(v)


def _md_nd(text: str) -> str:
    return text.replace("_not determined_", "not determined").replace("not determined", "_not determined_")


def _lc(ev: dict | None) -> str:
    """Visible low-confidence marker; the brief must not hide how sure a number is."""
    c = (ev or {}).get("confidence")
    return f" (low confidence {c:.2f})" if isinstance(c, (int, float)) and c < 0.5 else ""


def _span(ev: dict) -> str:
    return f"{ev['t_start']:.2f}-{ev['t_end']:.2f}s"


def _trans(tr: dict | None) -> str:
    if not tr:
        return "start"
    kind = _v(tr["kind"])
    if tr["duration"]:
        return f"{kind} ({_span(tr['evidence'])})" + _lc(tr["evidence"])
    return f"{kind} at {_t(tr['t'])}" + _lc(tr["evidence"])


def _bars(lb) -> str:
    if not lb:
        return ""
    if not isinstance(lb, dict):  # the shot's look was not determined
        return f" · letterbox {_v(lb)}"
    return " · letterbox " + " ".join(f"{k} {lb.get(k, 0):.1%}" for k in ("top", "bottom", "left", "right"))


def brief_md(bd: dict) -> str:
    v = bd["video"]
    out = [CONTRACT, "", f"# Breakdown: {v.get('title') or v['key']}", "",
           "Full numbers (box tracks, beat times, per-element evidence) are in `breakdown.json` next to this file.",
           ""]
    out.append(f"- video: `{v['key']}`" + (f" · {v['url']}" if v.get("url") else ""))
    out.append(f"- duration: {_t(v['duration'])} · fps: {v['fps_rational']} ({v['fps']:.3f}) · "
               f"size: {v['width']}x{v['height']}")
    aa = v["active_aspect"]
    out.append("- active picture aspect: " + (_v(aa["value"]) if aa["value"] == NOT_DETERMINED
                                              else f"{aa['value']} ({aa['width']}x{aa['height']} px)")
               + _lc(aa["evidence"]))
    if bd.get("low_resolution"):
        out.append("- **low resolution:** timing is approximate (media not at full quality)")
    out += ["", "## Shots", "",
            "Look scales: brightness, contrast, saturation are 0-1.", ""]
    if not bd["shots"]:
        out.append(f"Cuts and shots: {_v(NOT_DETERMINED)}")
    for s in bd["shots"]:
        pal = s["look"]["palette"]
        pal_s = _v(pal) if pal == NOT_DETERMINED else ", ".join(f"{p['hex']} {p['share']:.0%}" for p in pal)
        out.append(f"{s['index'] + 1}. {_t(s['start'])} - {_t(s['end'])} ({_t(s['duration'])}) "
                   f"· in: {_trans(s.get('transition'))}"
                   f" · camera: {_v(s['camera']['type'])}{_lc(s['camera']['evidence'])} · palette: {pal_s}"
                   f" · brightness {_v(s['look']['brightness'])} · contrast {_v(s['look']['contrast'])}"
                   f" · saturation {_v(s['look']['saturation'])} · temperature {_v(s['look']['temperature'])}"
                   + _bars(s["look"].get("letterbox")) + _lc(s["look"]["evidence"]))
    if bd.get("fades"):
        out += ["", "Fades at clip edges / not tied to a cut: "
                + "; ".join(f"{f['kind']} {_span(f['evidence'])}" + _lc(f["evidence"]) for f in bd["fades"])]
    r = bd["rhythm"]
    out += ["", "## Rhythm", "",
            f"- shots: {r['count']} · mean {_t(r['mean'])} · median {_t(r['median'])} · "
            f"min {_t(r['min'])} · max {_t(r['max'])}"]
    for k in ("fastest_window", "slowest_window"):
        w = r[k]
        out.append(f"- {k.replace('_', ' ')}: " + (_v(w) if w == NOT_DETERMINED
                                                   else f"{_t(w['t_start'])} - {_t(w['t_end'])} (mean shot {_t(w['mean'])})"))
    out += ["", "## Typography", "", "Box is x, y, w, h as 0-1 fractions of the frame.",
            f"Text {SAMPLING_NOTE}.", ""]
    tracks = bd["typography"]["tracks"]
    if tracks == NOT_DETERMINED:
        out.append(_v(NOT_DETERMINED))
    elif not tracks:
        out.append(f"No on-screen text found in the samples ({SAMPLING_NOTE}).")
    shown = [t for t in tracks if t["evidence"]["confidence"] >= BRIEF_MIN_TEXT_CONF] if isinstance(tracks, list) else []
    hidden = len(tracks) - len(shown) if isinstance(tracks, list) else 0
    if isinstance(tracks, list) and tracks and not shown:
        out.append(f"No confidently read on-screen text ({SAMPLING_NOTE}).")
    if hidden:
        out.append(f"{hidden} low-confidence OCR reading(s) left out of this brief (likely texture or fragments, "
                   "not text to rebuild); they are listed in breakdown.json.")
    for tr in shown:
        lc = _lc(tr["evidence"])
        out.append(f"- \"{tr['text']}\" {_t(tr['t_in'])} - {_t(tr['t_out'])} · box {tr['box']} · "
                   f"height {tr['height_rel']} of frame · colour {_v(tr['color_hex'])} · "
                   f"entrance {_v(tr['entrance'])}{lc} · exit {_v(tr['exit'])}{lc} · font {_v(tr['font_family'])}")
    a = bd["audio"]
    out += ["", "## Audio", "",
            f"- tempo: {_v(a['tempo_bpm'])}" + (" BPM" if a["tempo_bpm"] != NOT_DETERMINED else "")
            + (f" (alternative reading {a['tempo_alt']} BPM; half/double time is ambiguous)" if a.get("tempo_alt") else "")
            + _lc(a["evidence"]),
            (f"- beats: {len(a['beats'])}; first {min(16, len(a['beats']))} at "
             + ", ".join(f"{b:.2f}s" for b in a["beats"][:16])
             + (" (full list in breakdown.json)" if len(a["beats"]) > 16 else ""))
            if a["beats"] else f"- beats: {_v(NOT_DETERMINED)}"]
    for n in a.get("notes") or []:
        out.append(f"- tempo note: {_md_nd(n)}")
    cob = a["cuts_on_beat"]
    out.append("- cuts on beat (+-2 frames): " + (_v(cob["share"]) if cob["share"] == NOT_DETERMINED
                                                  else f"{cob['share']:.0%}")
               + (f" on the primary grid, {cob['share_alt']:.0%} on the other half/double-time grid "
                  "(tempo octave ambiguous)" if "share_alt" in cob else "") + _lc(cob["evidence"]))
    sg = a["segments"]
    out.append("- voice (speech or sung lyrics, from the transcript): " + (_v(sg["speech"]) if sg["speech"] == NOT_DETERMINED
                               else ", ".join(f"{_t(x['t_start'])}-{_t(x['t_end'])}" for x in sg["speech"][:40]))
               + _lc(sg["evidence"]))
    mu = sg["music"]
    out.append("- music present: " + _v(mu["present"]) + _lc(mu["evidence"]))
    if isinstance(mu["segments"], list):
        for x in mu["segments"]:
            ov = x["overlaps_speech_s"]
            out.append(f"  - {_t(x['t_start'])} - {_t(x['t_end'])}: {_v(x['state'])}"
                       + ("" if ov == NOT_DETERMINED else f" · overlaps speech {ov:.1f}s") + _lc(x["evidence"]))
    if bd["notes"]:
        out += ["", "## Notes", ""] + [f"- {_md_nd(n)}" for n in bd["notes"]]
    if bd.get("interpretation"):
        i = bd["interpretation"]
        out += ["", "## Interpretation (AI-written, not measured)", "",
                f"> made from: {i['made_from']}"
                + (f" · made on evidence version {i['evidence_version']}"
                   + (" (kept from an earlier run; the measured fields may have changed since)"
                      if i.get("stale") else "") if i.get("evidence_version") is not None else ""),
                "", i["text"]]
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- store

def breakdown_dir(conn, root: Path, video_key: str) -> Path:
    r = conn.execute("SELECT store_path FROM videos WHERE video_key = ?", (video_key,)).fetchone()
    vdir = Path(root) / r[0] if r and r[0] else mstore.video_dir(root, video_key)
    return vdir / "breakdown"


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)


def _write_contact(conn, m: dict, bd: dict, root: Path, out: Path) -> None:
    import exports

    with tempfile.TemporaryDirectory(prefix="watch-contact-") as tmp:
        tmp = Path(tmp)
        if m.get("path") and bd["shots"]:
            tiles = []
            for s in bd["shots"][:MAX_TILES]:
                png = tmp / f"s{s['index']:03d}.png"
                mid = (s["start"] + s["end"]) / 2
                _run(["ffmpeg", "-v", "error", "-y", "-ss", f"{mid:.3f}", "-i", m["path"],
                      "-frames:v", "1", str(png)])
                tiles.append({"t": s["start"], "path": str(png)})
            d = {"chapters": [{"frame": f} for f in tiles], "frames": []}
        else:
            frames = [{"t": t, "path": str(Path(root) / sp)} for t, sp in conn.execute(
                "SELECT t, store_path FROM frames WHERE video_key = ? AND store_path IS NOT NULL ORDER BY t",
                (bd["video"]["key"],))]
            d = {"chapters": [], "frames": frames}
        exports.write_contact(d, tmp, out, max_tiles=MAX_TILES)


def store(conn, video_key: str, bd: dict, store_dir=None) -> Path:
    """Write breakdown.json / .md / contact.png, pin media, index the brief. Returns the directory."""
    root = Path(store_dir) if store_dir is not None else Path(memory_paths()["store_dir"])
    out_dir = breakdown_dir(conn, root, video_key)
    out_dir.mkdir(parents=True, exist_ok=True)
    brief = brief_md(bd)
    (out_dir / "breakdown.json").write_text(json.dumps(bd, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "breakdown.md").write_text(brief, encoding="utf-8")
    try:
        m = media.ensure_media(conn, video_key, root, fetch=lambda *_: None)  # never re-download here
        _write_contact(conn, m, bd, root, out_dir / "contact.png")
    except Exception as exc:  # noqa: BLE001 - a missing contact sheet never fails the breakdown
        print(f"[watch] breakdown: contact sheet skipped ({type(exc).__name__}: {exc})", file=sys.stderr)
    media.pin(root, video_key)
    text = brief[:NOTE_CHARS]
    with index._tx(conn):
        for (nid,) in conn.execute("SELECT id FROM notes WHERE video_key = ? AND kind = ?",
                                   (video_key, NOTE_KIND)).fetchall():
            conn.execute("DELETE FROM fts WHERE src_table = 'notes' AND src_id = ?", (nid,))
            conn.execute("DELETE FROM embeddings WHERE src_table = 'notes' AND src_id = ?", (nid,))
            conn.execute("DELETE FROM notes WHERE id = ?", (nid,))
        cur = conn.execute("INSERT INTO notes (video_key, run_id, kind, t_start, t_end, text) "
                           "VALUES (?, NULL, ?, NULL, NULL, ?)", (video_key, NOTE_KIND, text))
        index._fts_add(conn, "note", video_key, "notes", cur.lastrowid, None, text)
    index.bump_evidence(conn, video_key)
    return out_dir


# --------------------------------------------------------------------------- timelines

def write_timelines(bd: dict, out_dir: Path, formats: list[str]) -> list[Path]:
    """Write ``timeline.<fmt>`` for each of ``formats`` into ``out_dir`` (OTIO / FCPXML / EDL).

    Returns no files when the breakdown has no determined shots (see ``interchange.skip_reason``)."""
    from . import interchange

    return interchange.write_timelines(bd, out_dir, formats)
