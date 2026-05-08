#!/usr/bin/env python3
"""Scene-aware frame selection via PySceneDetect.

Why mid-scene? Cut boundaries are noisy — fades, motion blur, half-rendered
text. Sampling the midpoint of each detected scene biases toward stabilized
content (per @MrMiguelChaves). Token cost stays the same, evidence quality
improves.
"""
from __future__ import annotations

import sys

# Force UTF-8 on stdout/stderr so non-ASCII output doesn't crash on Windows.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


DEFAULT_THRESHOLD = 27.0


def detect_scenes(
    video_path: str,
    threshold: float = DEFAULT_THRESHOLD,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> list[tuple[float, float]]:
    """Run ContentDetector and return [(scene_start_s, scene_end_s), ...].

    Empty list means no cuts were found (single-scene video, e.g. a static
    podcast camera). The caller should fall back to fps-based extraction.
    """
    try:
        from scenedetect import ContentDetector, detect  # type: ignore
    except ImportError:
        print(
            "[scenes] PySceneDetect is not installed — falling back to fps extraction. "
            "Install with: pip install scenedetect[opencv]",
            file=sys.stderr,
        )
        return []

    kwargs: dict = {}
    if start_seconds is not None:
        kwargs["start_time"] = float(start_seconds)
    if end_seconds is not None:
        kwargs["end_time"] = float(end_seconds)

    try:
        scene_list = detect(
            video_path,
            ContentDetector(threshold=threshold),
            **kwargs,
        )
    except Exception as exc:
        print(f"[scenes] PySceneDetect failed: {exc} — falling back to fps", file=sys.stderr)
        return []

    return [(s.get_seconds(), e.get_seconds()) for s, e in scene_list]


def pick_midpoints(
    scenes: list[tuple[float, float]],
    max_frames: int,
) -> list[float]:
    """Pick one timestamp per scene (midpoint), capped at max_frames.

    If we have more scenes than the budget allows, sample evenly across the
    list rather than truncating — preserves coverage of the whole video.
    """
    if not scenes or max_frames <= 0:
        return []

    midpoints = [(start + end) / 2.0 for start, end in scenes]

    if len(midpoints) <= max_frames:
        return midpoints

    if max_frames == 1:
        return [midpoints[len(midpoints) // 2]]

    n = len(midpoints)
    step = (n - 1) / (max_frames - 1)
    indices = sorted({int(round(i * step)) for i in range(max_frames)})
    # Defensive: rounding collisions can drop us below max_frames; backfill.
    if len(indices) < max_frames:
        for i in range(n):
            if i not in indices:
                indices.append(i)
                if len(indices) == max_frames:
                    break
        indices.sort()
    return [midpoints[i] for i in indices]


if __name__ == "__main__":
    import json

    if len(sys.argv) < 2:
        print("usage: scenes.py <video> [--threshold 27.0] [--start S] [--end S]", file=sys.stderr)
        raise SystemExit(2)

    video = sys.argv[1]
    threshold = DEFAULT_THRESHOLD
    start_s: float | None = None
    end_s: float | None = None
    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == "--threshold":
            threshold = float(args[i + 1]); i += 2
        elif args[i] == "--start":
            start_s = float(args[i + 1]); i += 2
        elif args[i] == "--end":
            end_s = float(args[i + 1]); i += 2
        else:
            i += 1

    scenes = detect_scenes(video, threshold=threshold, start_seconds=start_s, end_seconds=end_s)
    out = {
        "scene_count": len(scenes),
        "scenes": [{"start": s, "end": e, "midpoint": (s + e) / 2.0} for s, e in scenes],
    }
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
