"""Subtitle files from a breakdown: what was said (transcript) and what was shown (on-screen text).

* ``captions.srt``  - the transcript segments (speech or sung lyrics), as timed in the video memory.
* ``onscreen.srt``  - OCR text tracks (confidence >= ``MIN_TEXT_CONF``), one cue per track.
* ``onscreen.ass``  - the same tracks placed where they were measured: ``\\pos`` from the box, font
  size from the measured line height, primary colour from the measured text colour; plus the
  transcript in a ``Speech`` style. Only measured values are written: the font family is not
  determined (the default font is a placeholder), entrance / exit animations are noted in the
  event's Name field ("in:fade out:slide_left") and never rendered as invented effects.
"""
from __future__ import annotations

from pathlib import Path

NOT_DETERMINED = "not determined"
MIN_TEXT_CONF = 0.5  # same bar as the brief: weaker OCR stays in breakdown.json only
SIZE_FROM_BOX = 1.25  # font size (ASS, px at PlayRes) per pixel of measured line-box height
FORMATS = ("srt", "ass")


def _srt_time(t: float) -> str:
    ms = int(round(max(0.0, t) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _ass_time(t: float) -> str:
    cs = int(round(max(0.0, t) * 100))
    h, cs = divmod(cs, 360_000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def _clean(text: str) -> str:
    return " ".join(str(text or "").split())


def speech_cues(segments: list[dict]) -> list[dict]:
    out = []
    for s in segments:
        text = _clean(s.get("text"))
        if text and s.get("start") is not None and s.get("end") is not None and s["end"] > s["start"]:
            out.append({"t_in": float(s["start"]), "t_out": float(s["end"]), "text": text})
    return out


def text_cues(bd: dict) -> list[dict]:
    tracks = (bd.get("typography") or {}).get("tracks")
    if not isinstance(tracks, list):
        return []
    return [t for t in tracks if _clean(t.get("text")) and t.get("t_out", 0) > t.get("t_in", 0)
            and (t.get("evidence") or {}).get("confidence", 0) >= MIN_TEXT_CONF]


def srt(cues: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(sorted(cues, key=lambda c: (c["t_in"], c["t_out"])), 1):
        blocks.append(f"{i}\n{_srt_time(c['t_in'])} --> {_srt_time(c['t_out'])}\n{_clean(c['text'])}\n")
    return "\n".join(blocks)


def _ass_colour(hex_str) -> str | None:
    if not isinstance(hex_str, str) or len(hex_str) != 7 or not hex_str.startswith("#"):
        return None
    try:
        r, g, b = (int(hex_str[i:i + 2], 16) for i in (1, 3, 5))
    except ValueError:
        return None
    return f"&H{b:02X}{g:02X}{r:02X}&"


def _ass_escape(text: str) -> str:
    return _clean(text).replace("\\", "\\\\").replace("{", "(").replace("}", ")")


def ass(bd: dict, speech: list[dict]) -> str:
    v = bd.get("video") or {}
    w, h = int(v.get("width") or 0) or 1920, int(v.get("height") or 0) or 1080
    speech_size = max(16, round(h * 0.05))
    lines = [
        "[Script Info]",
        f"; from watch breakdown of {v.get('key', '')}: positions, sizes and colours measured; font family "
        "not determined (placeholder); animations noted in Name, not rendered",
        "ScriptType: v4.00+", f"PlayResX: {w}", f"PlayResY: {h}", "WrapStyle: 2", "ScaledBorderAndShadow: yes", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding",
        f"Style: Speech,Arial,{speech_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,0,2,"
        f"40,40,{round(h * 0.06)},1",
        f"Style: OnScreen,Arial,{speech_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,"
        "7,0,0,0,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for c in sorted(speech, key=lambda c: c["t_in"]):
        lines.append(f"Dialogue: 0,{_ass_time(c['t_in'])},{_ass_time(c['t_out'])},Speech,,0,0,0,,"
                     f"{_ass_escape(c['text'])}")
    for t in text_cues(bd):
        box = t.get("box") or [0, 0, 0, 0]
        x, y = round(box[0] * w), round(box[1] * h)
        size = max(8, round(box[3] * h * SIZE_FROM_BOX)) if box[3] else speech_size
        tags = [f"\\an7\\pos({x},{y})", f"\\fs{size}"]
        col = _ass_colour(t.get("color_hex"))
        if col:
            tags.append(f"\\1c{col}")
        anim = []
        for k, tag in (("entrance", "in"), ("exit", "out")):
            if t.get(k) and t[k] != NOT_DETERMINED:
                anim.append(f"{tag}:{t[k]}")
        lines.append(f"Dialogue: 1,{_ass_time(t['t_in'])},{_ass_time(t['t_out'])},OnScreen,{' '.join(anim)},0,0,0,,"
                     "{" + "".join(tags) + "}" + _ass_escape(t["text"]))
    return "\n".join(lines) + "\n"


def write(bd: dict, segments: list[dict], out_dir: Path, formats: list[str]) -> list[Path]:
    """Write the requested subtitle files; a file with nothing to say is not written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    speech = speech_cues(segments)
    shown = text_cues(bd)
    written = []
    if "srt" in formats:
        for name, cues in (("captions.srt", speech), ("onscreen.srt", shown)):
            if cues:
                (out_dir / name).write_text(srt(cues), encoding="utf-8")
                written.append(out_dir / name)
    if "ass" in formats and (speech or shown):
        (out_dir / "onscreen.ass").write_text(ass(bd, speech), encoding="utf-8")
        written.append(out_dir / "onscreen.ass")
    return written
