"""Editor interchange built from a breakdown: OTIO, FCPXML 1.10, CMX3600 EDL.

Every format references the ORIGINAL media with source ranges (one clip per
shot), so an editor relinks the file and sees the reference under each
placeholder. Timeline rate = source rate (rational, e.g. ``30000/1001``), start
timecode ``01:00:00:00`` (drop-frame for 29.97 / 59.94). All arithmetic is on
integer frames derived once from the shot times; nothing accumulates floats.

Mapping is honest (spec section 3): a measured dissolve becomes the format's
dissolve; a dip / fade (no native equivalent here) and anything
``"not determined"`` become a cut plus a marker saying so. No transition type
is ever invented.
"""
from __future__ import annotations

import bisect
import json
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path

NOT_DETERMINED = "not determined"
START_TC = "01:00:00:00"
MIN_DISSOLVE_FRAMES = 2
CROSS_DISSOLVE_UID = "FxPlug:4731E73A-8DAC-4113-9A30-AE85B1761265"
BEAT_FILTER = "beat"


# --------------------------------------------------------------------------- timecode

class Timecode:
    """Frame <-> timecode for a rational rate. Drop-frame for 30000/1001 and 60000/1001."""

    def __init__(self, num: int, den: int):
        self.num, self.den = int(num), int(den)
        self.nominal = max(1, int(round(self.num / self.den)))
        self.drop = (2 if self.nominal == 30 else 4 if self.nominal == 60 else 0) if self.den == 1001 else 0
        self.sep = ";" if self.drop else ":"

    def frames_of(self, seconds: float) -> int:
        x = Fraction(float(seconds)).limit_denominator(10 ** 6) * self.num / self.den
        return int((x * 2 + 1) // 2)  # round half up, exact

    def format(self, frames: int) -> str:
        f = int(frames)
        if self.drop:
            per10 = self.nominal * 600 - 9 * self.drop
            per_min = self.nominal * 60 - self.drop
            d, m = divmod(f, per10)
            f += 9 * self.drop * d + (self.drop * ((m - self.drop) // per_min) if m >= self.drop else 0)
        ff = f % self.nominal
        s = f // self.nominal
        return f"{s // 3600:02d}:{s // 60 % 60:02d}:{s % 60:02d}{self.sep}{ff:02d}"

    def parse(self, tc: str) -> int:
        h, m, rest = tc.split(":", 2)
        s, ff = rest.replace(";", ":").split(":")
        h, m, s, ff = int(h), int(m), int(s), int(ff)
        total_min = h * 60 + m
        f = (total_min * 60 + s) * self.nominal + ff
        if self.drop:
            f -= self.drop * (total_min - total_min // 10)
        return f

    def start_frames(self) -> int:
        return self.parse(START_TC)

    def seconds_str(self, frames: int) -> str:
        """FCPXML rational time: ``frames * den / num`` seconds."""
        return "0s" if frames == 0 else f"{frames * self.den}/{self.num}s"


# --------------------------------------------------------------------------- plan

def skip_reason(bd: dict) -> str | None:
    """Why no timeline can be written, or None. Timelines need measured shots and a frame rate."""
    shots = bd.get("shots")
    if not isinstance(shots, list) or not shots:
        return ("no shots determined (low-resolution or frames-only breakdown); "
                "timelines need the original media - not written")
    fr = (bd.get("video") or {}).get("fps_rational") or "0/1"
    try:
        num, den = (int(x) for x in str(fr).split("/"))
    except ValueError:
        num = den = 0
    if num <= 0 or den <= 0:
        return "no frame rate determined; timelines not written"
    for s in shots:
        if not all(isinstance(s.get(k), (int, float)) for k in ("start", "end")):
            return "shot times not determined; timelines not written"
    return None


def _rate(bd: dict) -> Timecode:
    num, den = (int(x) for x in bd["video"]["fps_rational"].split("/"))
    return Timecode(num, den)


def _media_name(bd: dict, media_path) -> str:
    if media_path:
        return Path(str(media_path)).name or "media"
    return (bd["video"].get("title") or bd["video"].get("key") or "media")


def _media_uri(bd: dict, media_path) -> str:
    if media_path:
        p = str(media_path)
        if "://" in p:
            return p
        return Path(p).absolute().as_uri()
    return bd["video"].get("url") or f"missing://{bd['video'].get('key', 'media')}"


def _plan(bd: dict, media_path) -> dict:
    tc = _rate(bd)
    shots = bd["shots"]
    n = len(shots)
    b = [0]
    for s in shots[1:]:
        b.append(max(tc.frames_of(s["start"]), b[-1] + 1))
    b.append(max(tc.frames_of(shots[-1]["end"]), b[-1] + 1))
    clips = []
    for i, s in enumerate(shots):
        c = {"i": i, "name": f"Shot {i + 1:03d}", "start": b[i], "end": b[i + 1], "h": 0, "tail": 0,
             "markers": [], "transition": None}
        tr = s.get("transition") if i else None
        if i:
            kind = (tr or {}).get("kind", NOT_DETERMINED) if tr else NOT_DETERMINED
            dur = (tr or {}).get("duration") or 0.0
            dur = dur if isinstance(dur, (int, float)) else 0.0
            c["transition"] = kind
            if kind == "dissolve" and dur > 0:
                d = tc.frames_of(dur)
                h = min(d // 2, (b[i] - b[i - 1]) // 2)
                tail = min(d - d // 2, (b[i + 1] - b[i]) // 2)
                if h + tail >= MIN_DISSOLVE_FRAMES:
                    c["h"], c["tail"] = h, tail
                else:
                    _tmark(c, tc, b[i], "dissolve too short to place (represented as a cut)", dur)
            elif kind != "cut":
                if kind == NOT_DETERMINED or not isinstance(kind, str):
                    _tmark(c, tc, b[i], "transition not determined", dur)
                else:
                    _tmark(c, tc, b[i], f"{kind} {dur:.2f}s (no native equivalent; represented as a cut)", dur)
                    c["markers"][-1]["name"] = kind
        clips.append(c)
    starts = b[:-1]

    def owner(frame: int) -> dict:
        return clips[max(0, min(n - 1, bisect.bisect_right(starts, frame) - 1))]

    def add(frame: int, name: str, comment: str, color: str, kind: str, dur_frames: int = 1) -> None:
        c = owner(frame)
        f = max(c["start"], min(frame, c["end"] - 1))
        c["markers"].append({"frame": f, "dur": max(1, min(dur_frames, c["end"] - f)), "name": name,
                             "comment": comment, "color": color, "kind": kind})

    for f in bd.get("fades") or []:
        ev = f.get("evidence") or {}
        t0, t1 = ev.get("t_start", f.get("t", 0.0)), ev.get("t_end", f.get("t", 0.0))
        add(tc.frames_of(t0), f["kind"], f"{f['kind']} {t0:.2f}-{t1:.2f}s (not a cut; no native equivalent)",
            "RED", "fade", tc.frames_of(t1) - tc.frames_of(t0))
    tracks = (bd.get("typography") or {}).get("tracks")
    for t in tracks if isinstance(tracks, list) else []:
        t_in, t_out = t.get("t_in"), t.get("t_out")
        if not isinstance(t_in, (int, float)):
            continue
        t_out = t_out if isinstance(t_out, (int, float)) else t_in
        add(tc.frames_of(t_in), f"text: {t['text']}", f"text \"{t['text']}\" {t_in:.2f}-{t_out:.2f}s",
            "BLUE", "text", tc.frames_of(t_out) - tc.frames_of(t_in))
    beats = (bd.get("audio") or {}).get("beats")
    for k, t in enumerate(beats if isinstance(beats, list) else []):
        add(tc.frames_of(t), BEAT_FILTER, f"beat {k + 1} at {t:.3f}s", "GREEN", "beat")
    for note in bd.get("notes") or []:
        add(0, "note", str(note), "YELLOW", "note")
    for c in clips:
        c["markers"].sort(key=lambda m: (m["frame"], m["kind"]))

    total = b[-1]
    media_frames = max(total, tc.frames_of(bd["video"].get("duration") or 0.0))
    return {"tc": tc, "clips": clips, "total": total, "media_frames": media_frames,
            "start": tc.start_frames(), "media_name": _media_name(bd, media_path),
            "uri": _media_uri(bd, media_path), "path": str(media_path) if media_path else None}


def _tmark(c: dict, tc: Timecode, frame: int, text: str, dur: float) -> None:
    c["markers"].append({"frame": frame, "dur": 1, "name": text.split(" (")[0], "comment": text,
                         "color": "RED", "kind": "transition"})


# --------------------------------------------------------------------------- OTIO

def _rt(value: int, rate: float) -> dict:
    return {"OTIO_SCHEMA": "RationalTime.1", "rate": rate, "value": value}


def _tr(start: int, dur: int, rate: float) -> dict:
    return {"OTIO_SCHEMA": "TimeRange.1", "duration": _rt(dur, rate), "start_time": _rt(start, rate)}


def to_otio(bd: dict, media_path) -> dict:
    """OTIO JSON (a ``Timeline.1`` dict): one video track of ``Clip.2`` with an
    ``ExternalReference.1`` + ``source_range``; ``Transition.1`` for dissolves."""
    p = _plan(bd, media_path)
    tc = p["tc"]
    rate = tc.num / tc.den
    ref = {"OTIO_SCHEMA": "ExternalReference.1", "name": p["media_name"], "target_url": p["uri"],
           "available_range": _tr(0, p["media_frames"], rate), "metadata": {}}
    children = []
    for c in p["clips"]:
        if c["h"] + c["tail"]:
            children.append({"OTIO_SCHEMA": "Transition.1", "name": "Cross Dissolve",
                             "transition_type": "SMPTE_Dissolve", "in_offset": _rt(c["h"], rate),
                             "out_offset": _rt(c["tail"], rate), "metadata": {}})
        children.append({
            "OTIO_SCHEMA": "Clip.2", "name": c["name"], "enabled": True, "effects": [],
            "source_range": _tr(c["start"], c["end"] - c["start"], rate),
            "media_references": {"DEFAULT_MEDIA": dict(ref)}, "active_media_reference_key": "DEFAULT_MEDIA",
            "markers": [{"OTIO_SCHEMA": "Marker.2", "name": m["name"], "color": m["color"],
                         "comment": m["comment"], "marked_range": _tr(m["frame"], m["dur"], rate),
                         "metadata": {"watch": {"kind": m["kind"]}}} for m in c["markers"]],
            "metadata": {"watch": {"shot": c["i"], "media_file": p["media_name"]}}})
    track = {"OTIO_SCHEMA": "Track.1", "name": "Video 1", "kind": "Video", "enabled": True, "children": children,
             "source_range": None, "effects": [], "markers": [], "metadata": {}}
    return {"OTIO_SCHEMA": "Timeline.1", "name": bd["video"].get("title") or bd["video"].get("key") or "breakdown",
            "global_start_time": _rt(p["start"], rate), "metadata": {"watch": {"schema": bd.get("schema")}},
            "tracks": {"OTIO_SCHEMA": "Stack.1", "name": "tracks", "enabled": True, "children": [track],
                       "source_range": None, "effects": [], "markers": [], "metadata": {}}}


# --------------------------------------------------------------------------- FCPXML

def to_fcpxml(bd: dict, media_path) -> str:
    """FCPXML 1.10: one ``asset``, a ``spine`` of ``asset-clip`` (one per shot),
    ``transition`` for dissolves, ``marker`` for beats / text / notes."""
    p = _plan(bd, media_path)
    tc = p["tc"]
    S = tc.seconds_str
    v = bd["video"]
    root = ET.Element("fcpxml", version="1.10")
    res = ET.SubElement(root, "resources")
    fmt = {"id": "r1", "frameDuration": f"{tc.den}/{tc.num}s"}
    if v.get("width") and v.get("height"):
        fmt.update(width=str(int(v["width"])), height=str(int(v["height"])))
    ET.SubElement(res, "format", fmt)
    asset = ET.SubElement(res, "asset", {"id": "r2", "name": p["media_name"], "start": "0s",
                                         "duration": S(p["media_frames"]), "hasVideo": "1", "format": "r1"})
    ET.SubElement(asset, "media-rep", {"kind": "original-media", "src": p["uri"]})
    if any(c["h"] + c["tail"] for c in p["clips"]):
        ET.SubElement(res, "effect", {"id": "r3", "name": "Cross Dissolve", "uid": CROSS_DISSOLVE_UID})
    lib = ET.SubElement(root, "library")
    ev = ET.SubElement(lib, "event", name=v.get("title") or "breakdown")
    proj = ET.SubElement(ev, "project", name=v.get("title") or "breakdown")
    seq = ET.SubElement(proj, "sequence", {"format": "r1", "duration": S(p["total"]), "tcStart": S(p["start"]),
                                           "tcFormat": "DF" if tc.drop else "NDF"})
    spine = ET.SubElement(seq, "spine")
    for c in p["clips"]:
        if c["h"] + c["tail"]:
            t = ET.SubElement(spine, "transition", {"name": "Cross Dissolve",
                                                    "offset": S(p["start"] + c["start"] - c["h"]),
                                                    "duration": S(c["h"] + c["tail"])})
            ET.SubElement(t, "filter-video", {"ref": "r3", "name": "Cross Dissolve"})
        ac = ET.SubElement(spine, "asset-clip", {
            "ref": "r2", "name": c["name"], "offset": S(p["start"] + c["start"]),
            "start": S(c["start"]), "duration": S(c["end"] - c["start"]), "format": "r1",
            "tcFormat": "DF" if tc.drop else "NDF"})
        for m in c["markers"]:
            ET.SubElement(ac, "marker", {"start": S(m["frame"]), "duration": S(m["dur"]),
                                         "value": f"{m['name']}" if m["kind"] in ("beat", "text") else m["comment"],
                                         **({"note": m["comment"]} if m["kind"] in ("beat", "text") else {})})
    ET.indent(root)
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>\n'
            + ET.tostring(root, encoding="unicode") + "\n")


# --------------------------------------------------------------------------- EDL

def _edl_title(bd: dict) -> str:
    t = bd["video"].get("title") or bd["video"].get("key") or "breakdown"
    return " ".join(str(t).split())[:60]


def to_edl(bd: dict, media_path) -> str:
    """CMX3600 EDL: one numbered event per shot (``C`` cut, ``D nnn`` dissolve),
    ``* FROM CLIP NAME:`` comments, drop-frame TC (``;``) for 29.97 / 59.94."""
    p = _plan(bd, media_path)
    tc = p["tc"]
    T = lambda f: tc.format(p["start"] + f)  # noqa: E731 - record timecode
    S = tc.format  # source timecode (media TC starts at 0)
    notes = [m for c in p["clips"] for m in c["markers"] if m["kind"] == "note"]
    beats = sum(1 for c in p["clips"] for m in c["markers"] if m["kind"] == "beat")
    out = [f"TITLE: {_edl_title(bd)}", f"FCM: {'DROP FRAME' if tc.drop else 'NON-DROP FRAME'}", ""]
    # strict parsers (OTIO's cmx_3600) reject comment lines before the first event,
    # so the file-level remarks ride along with event 1's comments
    head = ["* EDL has no markers: text / transition notes follow as LOC comments"
            + (f"; {beats} beat markers are in the OTIO / FCPXML files only" if beats else "")]
    head += [f"* NOTE: {m['comment']}" for m in notes]
    clips = p["clips"]
    for k, c in enumerate(clips):
        rec_in = c["start"] - c["h"]
        rec_out = clips[k + 1]["start"] - clips[k + 1]["h"] if k + 1 < len(clips) else c["end"]
        src_in, src_out = rec_in, rec_out
        n = c["i"] + 1
        rows = []
        if c["h"] + c["tail"]:
            rows.append(f"{n:03d}  AX       V     C        {S(src_in)} {S(src_in)} {T(rec_in)} {T(rec_in)}")
            rows.append(f"{n:03d}  AX       V     D    {c['h'] + c['tail']:03d} "
                        f"{S(src_in)} {S(src_out)} {T(rec_in)} {T(rec_out)}")
        else:
            rows.append(f"{n:03d}  AX       V     C        {S(src_in)} {S(src_out)} {T(rec_in)} {T(rec_out)}")
        out += rows
        out.append(f"* FROM CLIP NAME: {p['media_name']}")
        if k == 0:
            out += head
        if p["path"]:
            out.append(f"* SOURCE FILE: {p['path']}")
        out.append(f"* COMMENT: {c['name']} (source {S(c['start'])} - {S(c['end'])})")
        if c["h"] + c["tail"]:
            out.append(f"* COMMENT: dissolve {c['h'] + c['tail']} frames centred on the cut at {T(c['start'])}")
        for m in c["markers"]:
            if m["kind"] in ("text", "transition", "fade"):
                out.append(f"* LOC: {T(m['frame'])} {m['color']:<7} {m['comment']}")
        out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------- files

WRITERS = {"otio": lambda bd, mp: json.dumps(to_otio(bd, mp), indent=2, ensure_ascii=False) + "\n",
           "fcpxml": to_fcpxml, "edl": to_edl}


def find_media(out_dir: Path) -> Path | None:
    """The stored original (``<vdir>/media.*``) next to a ``<vdir>/breakdown`` directory."""
    found = sorted(Path(out_dir).parent.glob("media.*"))
    return found[0] if found else None


def write_timelines(bd: dict, out_dir: Path, formats: list[str], media_path=None) -> list[Path]:
    """Write ``timeline.<fmt>`` for each format; empty list when ``skip_reason`` applies."""
    if skip_reason(bd):
        return []
    media_path = media_path or find_media(out_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        path = out_dir / f"timeline.{fmt}"
        path.write_text(WRITERS[fmt](bd, media_path), encoding="utf-8")
        written.append(path)
    return written
