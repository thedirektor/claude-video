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


def reconcile(raw_shots: list[dict], entries: list[dict]) -> tuple[list[dict], list[dict]]:
    """Merge shots whose boundary ``classify`` absorbed into a fade / dip.

    Every remaining boundary carries exactly one transition entry (a dip keeps
    ONE boundary with a composite ``dip_<color>`` entry). Returns
    (shots with ``transition``, fade entries not attached to a boundary)."""
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
        kept.append({**sh, "transition": entry, "_conf": [sh["evidence"]["confidence"]]})
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
    cob = audio_mod.cuts_on_beat(bd_shots, beats, m.get("fps") or 0.0)
    if isinstance(cob, str):
        cob = {"share": NOT_DETERMINED, "evidence": _ev(0.0, duration, "measured", 0.0)}
    a["cuts_on_beat"] = cob
    speech = [{"t_start": s["start"], "t_end": s["end"],
               "evidence": _ev(s["start"], s["end"], "transcript", 0.8)} for s in segments]
    tempo_known = a.get("tempo_bpm") != NOT_DETERMINED
    music = ({"present": True, "evidence": _ev(a["evidence"]["t_start"], a["evidence"]["t_end"], "measured",
                                              a["evidence"]["confidence"])}
             if tempo_known else {"present": NOT_DETERMINED, "evidence": _ev(0.0, duration, "measured", 0.0)})
    seg = {"speech": speech if speech else NOT_DETERMINED, "music": music,
           "evidence": _ev(0.0, duration, "transcript" if speech else "measured",
                           0.8 if speech else (music["evidence"]["confidence"] if tempo_known else 0.0))}
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
    return {"tracks": tracks, "evidence": _ev(0.0, duration, "ocr", conf)}


def _rhythm_section(shot_list: list[dict]) -> dict:
    r = shots_mod.rhythm(shot_list)
    for k in ("fastest_window", "slowest_window"):
        w = r.get(k)
        if isinstance(w, dict) and "evidence" not in w:
            w["evidence"] = _ev(w["t_start"], w["t_end"], "measured", r["evidence"]["confidence"])
    return r


def _interpret(bd: dict, complete) -> dict | None:
    import spend
    from memory import llm

    complete = complete or llm.complete
    mids = [round((s["start"] + s["end"]) / 2, 3) for s in bd["shots"]] or [0.0]
    facts = [{"t": [s["start"], s["end"]], "transition": (s.get("transition") or {}).get("kind"),
              "camera": s["camera"]["type"],
              "look": {k: s["look"].get(k) for k in ("brightness", "temperature")}}
             for s in bd["shots"]][:60]
    tracks = bd["typography"]["tracks"]
    system = ("You describe a video from MEASURED facts only. Write at most 5 short sentences about the "
              "visual style. Do not add anything not supported by the facts. Plain text.")
    user = json.dumps({"duration": bd["video"]["duration"], "shots": facts,
                       "text_on_screen": [t["text"] for t in tracks][:20] if isinstance(tracks, list) else []})
    reply = complete(system, user, max_tokens=400)
    if reply is None or not (getattr(reply, "text", "") or "").strip():
        return None
    spend.record(reply.provider, "breakdown_interpretation", price_key=f"{reply.provider}:{reply.model}",
                 tokens_in=reply.tokens_in, tokens_out=reply.tokens_out)
    t1 = bd["video"]["duration"] or 0.0
    return {"text": reply.text.strip(), "frame_times": mids, "inputs": "measured breakdown fields (text only)",
            "provider": reply.provider, "model": reply.model,
            "evidence": _ev(0.0, t1, "interpretation", 0.3)}


# --------------------------------------------------------------------------- build

def build(conn, video_key: str, *, interpret: bool = False, store_dir=None, complete=None, fetch=None) -> dict:
    row = conn.execute("SELECT title, url, duration FROM videos WHERE video_key = ?", (video_key,)).fetchone()
    title, url, row_dur = (row["title"], row["url"], row["duration"]) if row else (None, None, 0.0)
    m = media.ensure_media(conn, video_key, store_dir, fetch=fetch)
    duration = float(m.get("duration") or row_dur or 0.0)
    fps = float(m.get("fps") or 0.0)
    notes: list[str] = []
    has_media = bool(m.get("path"))

    raw: list[dict] = []
    entries: list[dict] = []
    if has_media and fps > 0:
        series = shots_mod.frame_series(m["path"], fps)
        raw = shots_mod.detect_shots(series, duration)
        if raw:
            entries = transitions.classify(series, raw)
        else:
            _add_note(notes, "shot detection produced no shots; cuts not determined")
    else:
        _add_note(notes, "media not available (stored frames only); low resolution: cuts and shots not determined")
    shot_list, loose_fades = reconcile(raw, entries) if raw else ([], [])

    for sh in shot_list:
        sh["camera"] = camera.camera_motion(m["path"], sh)
        sh["look"] = look.shot_look(m["path"], sh)
        for n in (sh["camera"].get("notes") or []) + (sh["look"].get("notes") or []):
            _add_note(notes, n)

    width, height = int(m.get("width") or 0), int(m.get("height") or 0)
    video = {"key": video_key, "title": title, "url": url, "duration": duration, "fps": fps,
             "fps_rational": m.get("fps_rational") or "0/1", "width": width, "height": height,
             "source": m.get("source"),
             "active_aspect": _active_aspect(width, height, shot_list, duration),
             "evidence": _ev(0.0, duration, "measured", 0.95 if has_media else 0.3)}

    segs = [dict(r) for r in conn.execute(
        'SELECT start, "end" AS end FROM segments WHERE video_key = ? ORDER BY start', (video_key,))]
    segs = [s for s in segs if s["start"] is not None and s["end"] is not None]

    bd = {
        "schema": SCHEMA,
        "video": video,
        "low_resolution": bool(m.get("low_resolution")),
        "cuts": ({"count": len(shot_list) - 1,
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
    if interpret:
        interp = _interpret(bd, complete)
        if interp:
            bd["interpretation"] = interp
        else:
            _add_note(notes, "interpretation requested but no LLM reply was available")
    return bd


# --------------------------------------------------------------------------- brief

def _t(v) -> str:
    return "_not determined_" if v == NOT_DETERMINED or v is None else f"{float(v):.2f}s"


def _v(v) -> str:
    return "_not determined_" if v == NOT_DETERMINED or v is None else str(v)


def _md_nd(text: str) -> str:
    return text.replace("_not determined_", "not determined").replace("not determined", "_not determined_")


def brief_md(bd: dict) -> str:
    v = bd["video"]
    out = [CONTRACT, "", f"# Breakdown: {v.get('title') or v['key']}", ""]
    out.append(f"- video: `{v['key']}`" + (f" · {v['url']}" if v.get("url") else ""))
    out.append(f"- duration: {_t(v['duration'])} · fps: {v['fps_rational']} ({v['fps']:.3f}) · "
               f"size: {v['width']}x{v['height']}")
    aa = v["active_aspect"]
    out.append("- active picture aspect: " + (_v(aa["value"]) if aa["value"] == NOT_DETERMINED
                                              else f"{aa['value']} ({aa['width']}x{aa['height']})"))
    if bd.get("low_resolution"):
        out.append("- **low resolution:** timing is approximate (media not at full quality)")
    out += ["", "## Shots", ""]
    if not bd["shots"]:
        out.append(f"Cuts and shots: {_v(NOT_DETERMINED)}")
    for s in bd["shots"]:
        tr = s.get("transition")
        pal = s["look"]["palette"]
        pal_s = _v(pal) if pal == NOT_DETERMINED else ", ".join(f"{p['hex']} {p['share']:.0%}" for p in pal)
        lb = s["look"].get("letterbox")
        out.append(f"{s['index'] + 1}. {_t(s['start'])} - {_t(s['end'])} ({_t(s['duration'])}) "
                   f"· in: {_v(tr['kind']) if tr else 'start'}"
                   + (f" ({_t(tr['duration'])})" if tr and tr["duration"] else "")
                   + f" · camera: {_v(s['camera']['type'])} · palette: {pal_s}"
                   f" · brightness {_v(s['look']['brightness'])} · contrast {_v(s['look']['contrast'])}"
                   f" · saturation {_v(s['look']['saturation'])} · temperature {_v(s['look']['temperature'])}"
                   + (f" · letterbox {lb}" if lb else ""))
    if bd.get("fades"):
        out += ["", "Fades at clip edges / not tied to a cut: "
                + "; ".join(f"{f['kind']} at {_t(f['t'])} ({_t(f['duration'])})" for f in bd["fades"])]
    r = bd["rhythm"]
    out += ["", "## Rhythm", "",
            f"- shots: {r['count']} · mean {_t(r['mean'])} · median {_t(r['median'])} · "
            f"min {_t(r['min'])} · max {_t(r['max'])}"]
    for k in ("fastest_window", "slowest_window"):
        w = r[k]
        out.append(f"- {k.replace('_', ' ')}: " + (_v(w) if w == NOT_DETERMINED
                                                   else f"{_t(w['t_start'])} - {_t(w['t_end'])} (mean shot {_t(w['mean'])})"))
    out += ["", "## Typography", ""]
    tracks = bd["typography"]["tracks"]
    if tracks == NOT_DETERMINED:
        out.append(_v(NOT_DETERMINED))
    elif not tracks:
        out.append("No on-screen text found.")
    for tr in tracks if isinstance(tracks, list) else []:
        out.append(f"- \"{tr['text']}\" {_t(tr['t_in'])} - {_t(tr['t_out'])} · box {tr['box']} · "
                   f"height {tr['height_rel']} of frame · colour {_v(tr['color_hex'])} · "
                   f"entrance {_v(tr['entrance'])} · exit {_v(tr['exit'])} · font {_v(tr['font_family'])}")
    a = bd["audio"]
    out += ["", "## Audio", "",
            f"- tempo: {_v(a['tempo_bpm'])}" + (" BPM" if a["tempo_bpm"] != NOT_DETERMINED else "")
            + (f" (alternative {a['tempo_alt']})" if a.get("tempo_alt") else ""),
            f"- beats: {len(a['beats'])}" if a["beats"] else f"- beats: {_v(NOT_DETERMINED)}"]
    cob = a["cuts_on_beat"]
    out.append("- cuts on beat (+-2 frames): " + (_v(cob["share"]) if cob["share"] == NOT_DETERMINED
                                                  else f"{cob['share']:.0%}"))
    sg = a["segments"]
    out.append("- speech: " + (_v(sg["speech"]) if sg["speech"] == NOT_DETERMINED
                               else ", ".join(f"{_t(x['t_start'])}-{_t(x['t_end'])}" for x in sg["speech"][:40])))
    out.append("- music present: " + _v(sg["music"]["present"]))
    if bd["notes"]:
        out += ["", "## Notes", ""] + [f"- {_md_nd(n)}" for n in bd["notes"]]
    if bd.get("interpretation"):
        i = bd["interpretation"]
        out += ["", "## Interpretation (AI-written, not measured)", "",
                f"> made from times {i['frame_times']}", "", i["text"]]
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
        m = media.ensure_media(conn, video_key, root)
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


# --------------------------------------------------------------------------- timelines (Task 7)

def write_timelines(bd: dict, out_dir: Path, formats: list[str]) -> list[Path]:
    """Hook for Task 7: write ``timeline.<fmt>`` for each of ``formats`` into ``out_dir``.

    Not implemented yet: returns no files."""
    return []
