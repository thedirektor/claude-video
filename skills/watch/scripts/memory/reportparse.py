"""Parse a saved watch ``report.md`` back into evidence (used by the Paperless backfill).

Inverse of what ``watch.py`` writes: the header bullets, the ``## Frames`` lines
(with the optional ``[speech]`` tag, ``— Immich: …/photos/<id>`` link and
``— OCR: …`` tail) and the fenced transcript block (plain ``[MM:SS]`` /
``[H:MM:SS]`` or diarized ``[Speaker X] (M:SS-M:SS)``; minutes may exceed 59).
Pure stdlib, never raises on odd input.
"""
from __future__ import annotations

import re

LAST_SEGMENT_SECONDS = 5.0

_SOURCE = re.compile(r"\*\*Source:\*\*\s*(\S+)")
_TITLE = re.compile(r"^\s*-\s*\*\*Title:\*\*\s*(.+?)\s*$", re.M)
_DURATION = re.compile(r"\*\*Duration:\*\*[^\n]*?\(([\d.]+)s\)")
_TSOURCE = re.compile(r"^_Source: (.*?)(?:\. Filtered to .*:_|\._)\s*$", re.M)
_FRAME = re.compile(r"^- `[^`]*` \(t=((?:\d+:)?\d+:\d{2})(?:,[^)]*)?\)(.*)$")
_IMMICH = re.compile(r"— Immich: (\S+)")
_PLAIN = re.compile(r"^\[(\d+):(\d{2})(?::(\d{2}))?\]\s*(.*)$")
_DIARIZED = re.compile(r"^\[([^\]]+)\]\s*\((\d+):(\d{2})-(\d+):(\d{2})\)\s*(.*)$")
_OCR_SEP = " — OCR: "


def _clock(text: str) -> float:
    parts = [int(p) for p in text.split(":")]
    secs = 0
    for p in parts:
        secs = secs * 60 + p
    return float(secs)


def _section(text: str, heading: str) -> str:
    m = re.search(rf"^## {re.escape(heading)}\s*$", text, re.M)
    if not m:
        return ""
    rest = text[m.end():]
    nxt = re.search(r"^## ", rest, re.M)
    return rest[: nxt.start()] if nxt else rest


def _frames(block: str) -> list[dict]:
    out = []
    for line in block.splitlines():
        m = _FRAME.match(line.rstrip())
        if not m:
            continue
        tail = m.group(2)
        head, sep, ocr = tail.partition(_OCR_SEP)
        asset = None
        im = _IMMICH.search(head)
        if im:
            asset = im.group(1).rstrip("/").rsplit("/", 1)[-1] or None
        out.append({"t": _clock(m.group(1)), "ocr": ocr.strip() if sep else "", "immich_asset_id": asset})
    return out


def _fenced(block: str) -> list[str]:
    lines, inside, got = [], False, []
    for line in block.splitlines():
        if line.strip().startswith("```"):
            if inside:
                break
            inside = True
            continue
        if inside:
            got.append(line)
    lines = got
    return lines


def _segments(lines: list[str]) -> list[dict]:
    segs: list[dict] = []
    plain: list[int] = []  # indices of segments whose end is still open
    for line in lines:
        line = line.rstrip()
        m = _DIARIZED.match(line)
        if m:
            segs.append({
                "start": _clock(f"{m.group(2)}:{m.group(3)}"),
                "end": _clock(f"{m.group(4)}:{m.group(5)}"),
                "speaker": m.group(1),
                "text": m.group(6).strip(),
            })
            continue
        m = _PLAIN.match(line)
        if m:
            clock = f"{m.group(1)}:{m.group(2)}" if m.group(3) is None else f"{m.group(1)}:{m.group(2)}:{m.group(3)}"
            segs.append({"start": _clock(clock), "end": None, "text": m.group(4).strip()})
            plain.append(len(segs) - 1)
    for n, i in enumerate(plain):
        if n + 1 < len(plain):
            segs[i]["end"] = segs[plain[n + 1]]["start"]
        else:
            segs[i]["end"] = segs[i]["start"] + LAST_SEGMENT_SECONDS
    return [s for s in segs if s["text"]]


def parse_report(markdown: str) -> dict:
    text = markdown or ""
    src = _SOURCE.search(text)
    title = _TITLE.search(text)
    dur = _DURATION.search(text)
    transcript = _section(text, "Transcript")
    tsrc = _TSOURCE.search(transcript)
    return {
        "source": src.group(1) if src else None,
        "title": title.group(1) if title else None,
        "transcript_source": tsrc.group(1).strip() if tsrc else None,
        "duration": float(dur.group(1)) if dur else None,
        "segments": _segments(_fenced(transcript)),
        "frames": _frames(_section(text, "Frames")),
    }
