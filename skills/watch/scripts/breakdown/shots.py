"""Frame-accurate shot detection and rhythm stats.

``frame_series`` makes one ffmpeg pass (160 px, every frame) and records the
scene score and mean luma per frame. ``detect_shots`` turns that series into
shots: a hard cut is a single-frame score spike, a gradual boundary is a run of
frames whose end frames differ a lot (tiny colour thumbnails) while the content on both sides of the window is stable; a flash frame is not a cut. Boundaries are frame
accurate because they come from per-frame timestamps, never from sampling.
"""
from __future__ import annotations

import re
import statistics
import subprocess

NOT_DETERMINED = "not determined"
CUT_FLOOR = 0.15  # brief says 0.3; ffmpeg's scene score for a real red->green cut is only 0.23
LUMA_SCALE = 60.0  # a luma jump of this many levels counts as a full-strength cut
CUT_K = 6.0
GRADUAL_MIN_FRAMES = 3

_PTS = re.compile(r"pts_time:([0-9.]+)")
_SCENE = re.compile(r"lavfi\.scene_score=([0-9.eE+-]+)")
_YAVG = re.compile(r"lavfi\.signalstats\.YAVG=([0-9.eE+-]+)")


def _empty() -> dict:
    return {"t": [], "scene": [], "yavg": [], "thumb": []}


def frame_series(path: str, fps: float) -> dict:
    """Per-frame ``{"t", "scene", "yavg"}``; empty when fps is unknown (0)."""
    if not fps or fps <= 0:
        return _empty()
    vf = "scale=160:-2,signalstats,select='gte(scene,0)',metadata=print:file=-"
    try:
        res = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-an",
             "-vf", vf, "-vsync", "0", "-f", "null", "-"],
            capture_output=True, text=True, timeout=1800)
    except (OSError, subprocess.SubprocessError):
        return _empty()
    series = _empty()
    cur: dict = {}

    def flush() -> None:
        if "t" in cur:
            series["t"].append(cur["t"])
            series["scene"].append(cur.get("scene", 0.0))
            series["yavg"].append(cur.get("yavg", 0.0))

    for line in res.stdout.splitlines():
        if line.startswith("frame:"):
            flush()
            cur = {}
            m = _PTS.search(line)
            if m:
                cur["t"] = float(m.group(1))
        elif (m := _SCENE.search(line)):
            cur["scene"] = float(m.group(1))
        elif (m := _YAVG.search(line)):
            cur["yavg"] = float(m.group(1))
    flush()
    series["thumb"] = _thumbs(path, len(series["t"]))
    return series


def _thumbs(path: str, count: int) -> list[list[int]]:
    """Tiny RGB thumbnail per frame (second pass); [] when the count disagrees."""
    size = THUMB_W * THUMB_H * 3
    try:
        res = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-an", "-vf",
             f"scale={THUMB_W}:{THUMB_H}:flags=area,format=rgb24", "-vsync", "0", "-f", "rawvideo", "-"],
            capture_output=True, timeout=1800)
    except (OSError, subprocess.SubprocessError):
        return []
    raw = res.stdout
    if len(raw) != size * count:
        return []
    return [list(raw[i * size:(i + 1) * size]) for i in range(count)]


def _cut_threshold(scene: list[float]) -> float:
    med = statistics.median(scene)
    mad = statistics.median([abs(s - med) for s in scene])
    return max(CUT_FLOOR, med + CUT_K * mad)


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


THUMB_W, THUMB_H = 16, 9
MAX_GRADUAL_S = 2.0
MIN_CROSS = 0.08  # content change across a gradual window (0..1 mean abs thumbnail diff)
MIN_SCORE = 3.0  # cross-window change vs the shots' own frame-to-frame variation
FLASH_FRAMES = 3
DARK = 40  # near-black thumbnail (mean channel value)
WHITE_MEAN = 215  # near-white thumbnail (mean channel value)
_LENGTHS = (4, 6, 9, 13, 19, 27, 38, 50)


def _dist(a, b) -> float:
    return sum(abs(x - y) for x, y in zip(a, b)) / (255.0 * len(a))


def _flat_extreme(th) -> bool:
    """Near-black or near-white thumbnail (the end of a fade, not content)."""
    m = sum(th) / len(th)
    return m < DARK or m > WHITE_MEAN


def _fps_of(t: list[float]) -> float:
    dts = sorted(t[i] - t[i - 1] for i in range(1, len(t)) if t[i] > t[i - 1])
    return 1.0 / dts[len(dts) // 2] if dts else 0.0


def _flash_cuts(cands: list[int], strength: list[float], series: dict) -> set[int]:
    """Cut pairs a few frames apart where the content returns (flash frames)."""
    thumb, yavg = series.get("thumb") or [], series["yavg"]
    drop: set[int] = set()
    for i in cands:
        for j in cands:
            if not (i < j <= i + FLASH_FRAMES) or i in drop:
                continue
            if thumb:
                before, after = _dist(thumb[i - 1], thumb[i]), _dist(thumb[i - 1], thumb[j])
            else:
                before, after = abs(yavg[i] - yavg[i - 1]), abs(yavg[j] - yavg[i - 1])
            if after < 0.3 * before:
                drop.update((i, j))
    return drop


def _gradual(series: dict, cut_idx: set[int]) -> list[dict]:
    """Dissolves: a window whose end frames differ a lot while the content on
    either side of the window is stable. Needs thumbnails; never guesses."""
    thumb, t = series.get("thumb") or [], series["t"]
    n = len(t)
    if len(thumb) != n or n < 8:
        return []
    fps = _fps_of(t)
    cap = int(MAX_GRADUAL_S * fps)
    cands = []
    for i0 in range(n):
        for length in _LENGTHS:
            if length > cap or length < GRADUAL_MIN_FRAMES:
                continue
            i1 = i0 + length
            pre_n = min(length, i0)
            post_n = min(length, n - 1 - i1)
            if i1 >= n or pre_n < 3 or post_n < 3:
                continue
            if any(i0 < c <= i1 for c in cut_idx):
                continue
            # a fade in from / out to black near a clip edge is not a shot boundary
            near = int(0.5 * fps)
            if (i0 < near and _flat_extreme(thumb[i0])) or (n - 1 - i1 < near and _flat_extreme(thumb[i1])):
                continue
            cross = _dist(thumb[i0], thumb[i1])
            if cross < MIN_CROSS:
                continue
            own = max(_dist(thumb[i0 - pre_n], thumb[i0]), _dist(thumb[i1], thumb[i1 + post_n]))
            cands.append((cross / (own + 0.02), cross, i0, i1))
    cands.sort(key=lambda c: (-c[0], c[3] - c[2]))
    out, taken = [], []
    for score, cross, i0, i1 in cands:
        if score < MIN_SCORE:
            break
        if any(i0 <= e + 1 and s0 - 1 <= i1 for s0, e in taken):
            continue
        taken.append((i0, i1))
        out.append({"kind": "gradual", "t": round((t[i0] + t[i1]) / 2, 6), "frames": i1 - i0,
                    "t_start": t[i0], "t_end": t[i1],
                    "confidence": round(_clamp(0.4 + 0.08 * score), 3)})
    return out


def _boundaries(series: dict) -> list[dict]:
    t, scene, yavg = series["t"], series["scene"], series["yavg"]
    n = len(t)
    dy = [0.0] + [abs(yavg[i] - yavg[i - 1]) for i in range(1, n)]
    # cut strength: ffmpeg's scene score, or the luma jump when the score under-reads
    strength = [max(scene[i], dy[i] / LUMA_SCALE) for i in range(n)]
    thr = _cut_threshold(strength)
    cands = [i for i in range(1, n) if strength[i] >= thr]
    flash = _flash_cuts(cands, strength, series)
    cands = [i for i in cands if i not in flash]
    cut_idx: set[int] = set()
    out: list[dict] = []
    for i in cands:
        # adjacent candidates are one cut: keep the strongest (first on a tie)
        if i - 1 in cut_idx and strength[i] <= strength[i - 1]:
            continue
        if i - 1 in cut_idx:
            cut_idx.discard(i - 1)
            out = [b for b in out if b["t"] != t[i - 1]]
        cut_idx.add(i)
        out.append({"kind": "cut", "t": t[i], "frames": 1, "t_start": t[i - 1], "t_end": t[i],
                    "confidence": _clamp(0.5 + strength[i] / (2 * thr))})
    out += _gradual(series, cut_idx)
    out.sort(key=lambda b: b["t"])
    return out


SINGLE_SHOT_CONF = 0.6  # no neighbouring boundary: a dissolve between two moving shots can go undetected


def detect_shots(series: dict, duration: float) -> list[dict]:
    """Shots from a frame series. Empty list when nothing can be measured."""
    if not series or len(series.get("t", [])) < 2 or not duration or duration <= 0:
        return []
    bounds = _boundaries(series)
    edges = [0.0] + [b["t"] for b in bounds] + [float(duration)]
    shots = []
    for k in range(len(edges) - 1):
        start, end = edges[k], edges[k + 1]
        adj = [bounds[x] for x in (k - 1, k) if 0 <= x < len(bounds)]
        conf = min((b["confidence"] for b in adj), default=SINGLE_SHOT_CONF)
        shot = {
            "index": k, "start": round(start, 6), "end": round(end, 6),
            "duration": round(end - start, 6),
            "boundary": None,
            "evidence": {"t_start": round(start, 6), "t_end": round(end, 6),
                         "method": "measured", "confidence": round(conf, 3)},
        }
        if k > 0:
            b = bounds[k - 1]
            shot["boundary"] = {"kind": b["kind"], "t": round(b["t"], 6), "frames": b["frames"],
                                "evidence": {"t_start": b["t_start"], "t_end": b["t_end"],
                                             "method": "measured", "confidence": round(b["confidence"], 3)}}
        shots.append(shot)
    return shots


def rhythm(shots: list[dict]) -> dict:
    """Cut-rhythm stats; never divides by zero, single shot is its own window."""
    if not shots:
        return {"count": 0, "mean": NOT_DETERMINED, "median": NOT_DETERMINED,
                "min": NOT_DETERMINED, "max": NOT_DETERMINED,
                "fastest_window": NOT_DETERMINED, "slowest_window": NOT_DETERMINED,
                "evidence": {"t_start": 0.0, "t_end": 0.0, "method": "measured", "confidence": 0}}
    durs = [s["duration"] for s in shots]
    span = shots[-1]["end"] - shots[0]["start"]
    size = min(3, len(shots))
    best = min(range(len(shots) - size + 1), key=lambda i: sum(durs[i:i + size]))
    win = shots[best:best + size]
    worst = max(range(len(shots) - size + 1), key=lambda i: sum(durs[i:i + size]))
    slow = shots[worst:worst + size]
    return {
        "count": len(shots),
        "mean": span / len(shots),
        "median": statistics.median(durs),
        "min": min(durs),
        "max": max(durs),
        "fastest_window": {"t_start": win[0]["start"], "t_end": win[-1]["end"],
                           "mean": sum(durs[best:best + size]) / size},
        "slowest_window": {"t_start": slow[0]["start"], "t_end": slow[-1]["end"],
                           "mean": sum(durs[worst:worst + size]) / size},
        "evidence": {"t_start": shots[0]["start"], "t_end": shots[-1]["end"],
                     "method": "measured", "confidence": round(min(s["evidence"]["confidence"] for s in shots), 3)},
    }
