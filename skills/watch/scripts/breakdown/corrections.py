"""User corrections to a breakdown: kept next to it, applied on every rebuild, exportable as eval labels.

``<video>/breakdown/corrections.json`` is a list of entries::

    {"kind": "production", "t": 12.3, "label": "camera", "note": "...", "at": "<iso time>"}
    {"kind": "add_cut", "t": 2.2, ...}       {"kind": "remove_cut", "t": 1.5, ...}
    {"kind": "cuts_verified", ...}           (every cut of the stored breakdown was checked)

A production correction names a time inside the shot, so it survives a rebuild that moves the cuts.
Corrections never change thresholds: they replace the measured value of that video only, and the
measured value stays in the breakdown under ``measured`` (evidence method ``correction``). Exported
labels let ``watch.py evaluate`` measure whether a threshold change would have got them right.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from . import shots as shots_mod

KINDS = ("production", "add_cut", "remove_cut", "cuts_verified")
LABELS = ("still", "still_moved", "synthetic", "camera", "ai_generated", "ai_video")
CUT_SNAP_S = 0.25  # a remove_cut takes out the detected boundary nearest its time within this
FILE = "corrections.json"


class CorrectionError(ValueError):
    pass


def path_for(conn, root: Path, video_key: str) -> Path:
    from . import assemble  # noqa: PLC0415 - assemble imports this module

    return assemble.breakdown_dir(conn, root, video_key) / FILE


def load(p: Path) -> list[dict]:
    try:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [e for e in data if isinstance(e, dict) and e.get("kind") in KINDS] if isinstance(data, list) else []


def save(p: Path, entries: list[dict]) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(p)


def make(kind: str, t: float | None = None, label: str | None = None, note: str | None = None) -> dict:
    if kind not in KINDS:
        raise CorrectionError(f"kind must be one of {', '.join(KINDS)}")
    e: dict = {"kind": kind, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if kind != "cuts_verified":
        if t is None or not isinstance(t, (int, float)) or t < 0:
            raise CorrectionError(f"{kind} needs a time t >= 0 (seconds)")
        e["t"] = round(float(t), 3)
    if kind == "production":
        if label not in LABELS:
            raise CorrectionError(f"label must be one of {', '.join(LABELS)}")
        e["label"] = label
    if note:
        e["note"] = str(note)[:500]
    return e


# --------------------------------------------------------------------------- applying

def apply_cuts(raw_shots: list[dict], entries: list[dict], duration: float, fps: float) -> list[dict]:
    """Rebuild the raw shot list with added / removed cuts (before transitions and per-shot measures)."""
    adds = [e["t"] for e in entries if e["kind"] == "add_cut" and 0 < e["t"] < duration]
    removes = [e["t"] for e in entries if e["kind"] == "remove_cut"]
    if not adds and not removes:
        return raw_shots
    bounds = []
    for s in raw_shots[1:]:
        b = s.get("boundary") or {}
        ev = b.get("evidence") or {}
        bounds.append({"kind": b.get("kind", "cut"), "t": b.get("t", s["start"]), "frames": b.get("frames", 1),
                       "t_start": ev.get("t_start", s["start"]), "t_end": ev.get("t_end", s["start"]),
                       "confidence": ev.get("confidence", 0.5), **({"note": b["note"]} if b.get("note") else {}),
                       **({"jump": True} if b.get("jump") else {})})
    for r in removes:
        near = [b for b in bounds if abs(b["t"] - r) <= CUT_SNAP_S]
        if near:
            bounds.remove(min(near, key=lambda b: abs(b["t"] - r)))
    frame = 1.0 / fps if fps else 0.04
    for a in adds:
        if any(abs(b["t"] - a) <= frame * 1.5 for b in bounds):
            continue  # already there
        bounds.append({"kind": "cut", "t": a, "frames": 1, "t_start": max(0.0, a - frame), "t_end": a,
                       "confidence": 1.0, "corrected": True})
    bounds.sort(key=lambda b: b["t"])
    out = shots_mod.build_shots(bounds, duration)
    for s, b in zip(out[1:], bounds):
        if b.get("corrected"):
            s["boundary"]["corrected"] = True
            s["boundary"]["evidence"]["method"] = "correction"
    return out


def apply_production(shot_list: list[dict], entries: list[dict]) -> int:
    """Replace the production label of the shot holding each correction's time. Returns how many applied."""
    n = 0
    for e in entries:
        if e["kind"] != "production":
            continue
        for s in shot_list:
            last = s is shot_list[-1]
            if s["start"] <= e["t"] < s["end"] or (last and e["t"] == s["end"]):
                p = s.get("production") or {}
                measured = p.get("measured") or {k: p.get(k) for k in ("label", "candidates", "evidence")}
                s["production"] = {**p, "label": e["label"], "measured": measured,
                                   "candidates": [{"label": e["label"], "confidence": 1.0}],
                                   "evidence": {"t_start": s["start"], "t_end": s["end"], "method": "correction",
                                                "confidence": 1.0},
                                   "notes": (p.get("notes") or []) + [
                                       "corrected by the user" + (f": {e['note']}" if e.get("note") else "")]}
                n += 1
                break
    return n


def summary(entries: list[dict]) -> dict:
    out = {k: sum(1 for e in entries if e["kind"] == k) for k in KINDS}
    out["total"] = len(entries)
    return out


# --------------------------------------------------------------------------- export as eval labels

def export_labels(conn, root: Path, out: Path) -> dict:
    """One labels file (``watch.py evaluate`` format) from every video with corrections.

    Production spans are the corrected shots of the stored breakdown; cuts are exported only for a
    video whose cuts were all verified (``cuts_verified``), since a partial cut list would score every
    unlisted detection as a false positive."""
    from . import assemble, media  # noqa: PLC0415

    videos = []
    for row in conn.execute("SELECT video_key FROM videos ORDER BY video_key").fetchall():
        key = row[0]
        cp = path_for(conn, root, key)
        entries = load(cp) if cp.is_file() else []
        if not entries:
            continue
        bd = assemble.read_stored(conn, key, root)
        m = media.ensure_media(conn, key, root, fetch=lambda *_: None)
        if not bd or not m.get("path"):
            continue
        v: dict = {"id": key, "path": str(Path(m["path"]).resolve()), "production": []}
        for s in bd.get("shots") or []:
            if (s.get("production") or {}).get("evidence", {}).get("method") == "correction":
                v["production"].append([s["start"], s["end"], s["production"]["label"]])
        if any(e["kind"] == "cuts_verified" for e in entries):
            v["cuts"] = [s["boundary"]["t"] for s in (bd.get("shots") or [])[1:] if s.get("boundary")]
        if v["production"] or "cuts" in v:
            videos.append(v)
    data = {"tolerance_frames": 2, "_source": "watch.py correct --export-labels", "videos": videos}
    Path(out).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"videos": len(videos), "path": str(out)}
