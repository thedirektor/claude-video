#!/usr/bin/env python3
"""Two-pass sampling: compute speech windows from a transcript, then pick
timestamps so the frame budget concentrates where the VO/dialogue happens.

Why this exists: a 2-minute promo with a 1-minute voiceover wants dense
coverage during the VO (where supers and key visuals land) and sparse
coverage during the music-only tail. Uniform fps or scene-detect alone can
miss the alignment between spoken content and on-screen evidence.
"""
from __future__ import annotations

import sys
from typing import Iterable

# Force UTF-8 on stdout/stderr so non-ASCII output doesn't crash on Windows.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


DEFAULT_GAP_THRESHOLD = 2.0   # seconds; segments closer than this merge into one window
DEFAULT_SPEECH_SHARE = 0.7    # 70% of frame budget goes to speech windows


def compute_speech_windows(
    segments: list[dict],
    gap_threshold: float = DEFAULT_GAP_THRESHOLD,
    range_start: float | None = None,
    range_end: float | None = None,
) -> list[tuple[float, float]]:
    """Group adjacent transcript segments into contiguous speech windows.

    Segments closer than `gap_threshold` seconds merge into a single window.
    Optionally clipped to [range_start, range_end] when --start/--end is set.
    """
    if not segments:
        return []

    sorted_segs = sorted(
        ({"start": float(s["start"]), "end": float(s["end"])} for s in segments),
        key=lambda s: s["start"],
    )

    windows: list[tuple[float, float]] = []
    cur_start = sorted_segs[0]["start"]
    cur_end = sorted_segs[0]["end"]
    for seg in sorted_segs[1:]:
        if seg["start"] - cur_end <= gap_threshold:
            cur_end = max(cur_end, seg["end"])
        else:
            windows.append((cur_start, cur_end))
            cur_start, cur_end = seg["start"], seg["end"]
    windows.append((cur_start, cur_end))

    if range_start is not None or range_end is not None:
        lo = range_start if range_start is not None else float("-inf")
        hi = range_end if range_end is not None else float("inf")
        clipped: list[tuple[float, float]] = []
        for s, e in windows:
            cs = max(s, lo)
            ce = min(e, hi)
            if ce > cs:
                clipped.append((cs, ce))
        windows = clipped

    return windows


def _scene_midpoints_in(
    window: tuple[float, float],
    scenes: list[tuple[float, float]],
) -> list[float]:
    s, e = window
    out: list[float] = []
    for ss, se in scenes:
        mid = (ss + se) / 2.0
        if s <= mid <= e:
            out.append(mid)
    return out


def _evenly_spaced(start: float, end: float, count: int) -> list[float]:
    """Return `count` timestamps evenly distributed inside (start, end).

    Boundaries are nudged inward so we don't sample exactly at the edge of a
    cut — keeps the same anti-blur intent as scene midpoints.
    """
    if count <= 0 or end <= start:
        return []
    if count == 1:
        return [(start + end) / 2.0]
    span = end - start
    step = span / (count + 1)
    return [start + step * (i + 1) for i in range(count)]


def _pick_in_window(
    window: tuple[float, float],
    scenes: list[tuple[float, float]],
    budget: int,
) -> list[float]:
    """Pick `budget` timestamps inside the window.

    Strategy: prefer scene midpoints that land inside the window; if there
    aren't enough, fill the gap with evenly-spaced timestamps. If scene
    detection wasn't run (empty `scenes`), use evenly-spaced only.
    """
    if budget <= 0:
        return []

    s, e = window
    if e <= s:
        return []

    midpoints = _scene_midpoints_in(window, scenes) if scenes else []
    midpoints.sort()

    if len(midpoints) >= budget:
        if budget == 1:
            return [midpoints[len(midpoints) // 2]]
        n = len(midpoints)
        step = (n - 1) / (budget - 1)
        idx = sorted({int(round(i * step)) for i in range(budget)})
        return [midpoints[i] for i in idx]

    extras_needed = budget - len(midpoints)
    extras = _evenly_spaced(s, e, extras_needed + len(midpoints) + 1)
    # Greedily remove extras that are too close to existing midpoints (within
    # 5% of window span) so we don't double-sample near the same moment.
    tolerance = max(0.5, (e - s) * 0.05)
    kept: list[float] = []
    for ts in extras:
        if all(abs(ts - m) > tolerance for m in midpoints):
            kept.append(ts)
        if len(kept) == extras_needed:
            break
    if len(kept) < extras_needed:
        # Fall back: top up with raw evenly-spaced ignoring tolerance.
        kept = _evenly_spaced(s, e, extras_needed)
    return sorted(midpoints + kept)


def two_pass_sample(
    range_start: float,
    range_end: float,
    speech_windows: list[tuple[float, float]],
    scenes: list[tuple[float, float]],
    max_frames: int,
    speech_share: float = DEFAULT_SPEECH_SHARE,
) -> dict:
    """Pick timestamps using the 70/30 speech vs non-speech split.

    Returns:
      {
        "timestamps": [t, ...],            # sorted, deduped, capped to max_frames
        "speech_count": N,                 # frames inside a speech window
        "non_speech_count": M,             # frames outside speech windows
        "speech_total_seconds": float,
        "non_speech_total_seconds": float,
        "gaps": [(s, e), ...],             # the non-speech ranges we sampled from
      }
    """
    if max_frames <= 0 or range_end <= range_start:
        return {
            "timestamps": [],
            "speech_count": 0,
            "non_speech_count": 0,
            "speech_total_seconds": 0.0,
            "non_speech_total_seconds": 0.0,
            "gaps": [],
        }

    full_total = range_end - range_start
    speech_total = sum(e - s for s, e in speech_windows)
    speech_total = min(speech_total, full_total)

    # Build gap windows: [range_start, first_window_start], between windows,
    # and [last_window_end, range_end]. These are the "non-speech" sections.
    gaps: list[tuple[float, float]] = []
    cursor = range_start
    for ws, we in sorted(speech_windows):
        if ws > cursor:
            gaps.append((cursor, ws))
        cursor = max(cursor, we)
    if cursor < range_end:
        gaps.append((cursor, range_end))
    non_speech_total = sum(e - s for s, e in gaps)

    if speech_total <= 0:
        # No speech (or windows fell entirely outside range): uniform fallback.
        ts = _evenly_spaced(range_start, range_end, max_frames)
        return {
            "timestamps": ts,
            "speech_count": 0,
            "non_speech_count": len(ts),
            "speech_total_seconds": 0.0,
            "non_speech_total_seconds": non_speech_total,
            "gaps": gaps,
        }

    if non_speech_total <= 0:
        # All time is speech — give it the whole budget.
        speech_budget = max_frames
        non_speech_budget = 0
    else:
        speech_budget = int(round(max_frames * speech_share))
        non_speech_budget = max_frames - speech_budget
        # Make sure neither side is starved when the other has time to sample.
        if speech_budget == 0 and speech_total > 0:
            speech_budget = 1
            non_speech_budget = max_frames - 1
        if non_speech_budget == 0 and non_speech_total > 0 and max_frames > 1:
            non_speech_budget = 1
            speech_budget = max_frames - 1

    speech_timestamps: list[float] = []
    if speech_budget > 0 and speech_total > 0:
        # Distribute speech budget across windows in proportion to duration,
        # min 1 frame per window so short windows aren't dropped.
        per_window: list[int] = []
        for s, e in speech_windows:
            share = (e - s) / speech_total
            per_window.append(max(1, int(round(speech_budget * share))))
        # Reconcile rounding/min-1 inflation back down to speech_budget.
        diff = sum(per_window) - speech_budget
        while diff > 0:
            # Trim from the largest allocations first (but never below 1).
            i = max(range(len(per_window)), key=lambda i: per_window[i])
            if per_window[i] <= 1:
                break
            per_window[i] -= 1
            diff -= 1
        while diff < 0:
            i = max(range(len(speech_windows)), key=lambda i: speech_windows[i][1] - speech_windows[i][0])
            per_window[i] += 1
            diff += 1
        for window, budget in zip(speech_windows, per_window):
            speech_timestamps.extend(_pick_in_window(window, scenes, budget))

    non_speech_timestamps: list[float] = []
    if non_speech_budget > 0 and gaps:
        per_gap: list[int] = []
        for s, e in gaps:
            share = (e - s) / non_speech_total
            per_gap.append(max(1, int(round(non_speech_budget * share))))
        diff = sum(per_gap) - non_speech_budget
        while diff > 0:
            i = max(range(len(per_gap)), key=lambda i: per_gap[i])
            if per_gap[i] <= 1:
                break
            per_gap[i] -= 1
            diff -= 1
        while diff < 0:
            i = max(range(len(gaps)), key=lambda i: gaps[i][1] - gaps[i][0])
            per_gap[i] += 1
            diff += 1
        for gap, budget in zip(gaps, per_gap):
            non_speech_timestamps.extend(_pick_in_window(gap, scenes, budget))

    combined = sorted(set(round(t, 3) for t in speech_timestamps + non_speech_timestamps))

    if len(combined) > max_frames:
        # Last-resort uniform thinning (rare — only when min-1 inflation pushed us over).
        n = len(combined)
        step = (n - 1) / (max_frames - 1) if max_frames > 1 else 0
        idx = sorted({int(round(i * step)) for i in range(max_frames)})
        combined = [combined[i] for i in idx]

    speech_set = [(s, e) for s, e in speech_windows]
    speech_count = sum(1 for t in combined if any(s <= t <= e for s, e in speech_set))
    non_speech_count = len(combined) - speech_count

    return {
        "timestamps": combined,
        "speech_count": speech_count,
        "non_speech_count": non_speech_count,
        "speech_total_seconds": speech_total,
        "non_speech_total_seconds": non_speech_total,
        "gaps": gaps,
    }


def format_windows(
    windows: Iterable[tuple[float, float]],
    speech_total: float | None = None,
    full_duration: float | None = None,
) -> str:
    """Format speech windows for the report header line."""
    parts: list[str] = []
    for s, e in windows:
        parts.append(f"{_fmt(s)}-{_fmt(e)}")
    if not parts:
        return "(none)"
    out = ", ".join(parts)
    if speech_total is not None and full_duration is not None and full_duration > 0:
        out += f" (covering {speech_total:.0f}s of {full_duration:.0f}s)"
    return out


def _fmt(seconds: float) -> str:
    total = int(round(seconds))
    m, s = divmod(total, 60)
    return f"{m}:{s:02d}"
