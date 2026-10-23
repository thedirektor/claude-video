"""Speaker-aware transcript chunking for the video-memory index (stdlib only)."""
from __future__ import annotations


def chunk_segments(
    segments: list[dict], min_s: float = 20.0, max_s: float = 40.0, turn_min_s: float = 8.0
) -> list[dict]:
    """Group segments into ~min_s..max_s windows that respect speaker turns.

    Before adding segment ``s`` to a non-empty chunk the chunk is closed if
    ``dur >= min_s``, ``dur + len(s) > max_s``, or the speaker changes after at
    least ``turn_min_s``. A multi-speaker chunk has ``speaker=None``; a segment
    longer than ``max_s`` becomes its own chunk.
    """
    out: list[dict] = []
    cur: dict | None = None
    speakers: set = set()

    def close() -> None:
        nonlocal cur, speakers
        if cur is not None:
            cur["speaker"] = next(iter(speakers)) if len(speakers) == 1 else None
            cur["text"] = " ".join(cur.pop("_parts"))
            out.append(cur)
        cur, speakers = None, set()

    for s in segments:
        start, end = s["start"], s["end"]
        sp = s.get("speaker")
        if cur is not None:
            dur = cur["t_end"] - cur["t_start"]
            cur_speaker = next(iter(speakers)) if len(speakers) == 1 else None
            if dur >= min_s or dur + (end - start) > max_s or (sp != cur_speaker and dur >= turn_min_s):
                close()
        if cur is None:
            cur = {"t_start": start, "t_end": end, "speaker": None, "text": "", "_parts": []}
        cur["t_end"] = end
        cur["_parts"].append(s["text"])
        speakers.add(sp)
    close()
    return out
