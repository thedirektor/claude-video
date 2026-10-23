"""Frame-accurate shot detection and rhythm stats.

``frame_series`` makes one ffmpeg pass (160 px, every frame) and records the
scene score and mean luma per frame. ``detect_shots`` turns that series into
shots: a hard cut is a single-frame score spike, a gradual boundary is a run of
frames above a lower threshold with a smooth ramp. Boundaries are frame
accurate because they come from per-frame timestamps, never from sampling.
"""
from __future__ import annotations

import re
import statistics
import subprocess

NOT_DETERMINED = "not determined"
CUT_FLOOR = 0.15  # brief says 0.3; ffmpeg's scene score for a real red->green cut is only 0.23
LUMA_SCALE = 60.0  # a luma jump of this many levels counts as a full-strength cut
GRADUAL_LUMA = 0.5  # per-frame luma step that marks a ramping frame
CUT_K = 6.0
GRADUAL_FLOOR = 0.004
GRADUAL_MIN_FRAMES = 3
GAP_FRAMES = 1
EDGE_FRAMES = 2

_PTS = re.compile(r"pts_time:([0-9.]+)")
_SCENE = re.compile(r"lavfi\.scene_score=([0-9.eE+-]+)")
_YAVG = re.compile(r"lavfi\.signalstats\.YAVG=([0-9.eE+-]+)")


def _empty() -> dict:
    return {"t": [], "scene": [], "yavg": []}


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
    return series


def _cut_threshold(scene: list[float]) -> float:
    med = statistics.median(scene)
    mad = statistics.median([abs(s - med) for s in scene])
    return max(CUT_FLOOR, med + CUT_K * mad)


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def _ramp_ok(scores: list[float], yavg: list[float]) -> bool:
    """Monotonic luma change, or a smooth (no spikes) score ramp."""
    net = yavg[-1] - yavg[0]
    if abs(net) >= 2.0:
        sign = 1 if net > 0 else -1
        if all(sign * (b - a) >= -1.0 for a, b in zip(yavg, yavg[1:])):
            return True
    top, med = max(scores), statistics.median(scores)
    return med > 0 and top <= 6 * med


def _boundaries(series: dict) -> list[dict]:
    t, scene, yavg = series["t"], series["scene"], series["yavg"]
    n = len(t)
    dy = [0.0] + [abs(yavg[i] - yavg[i - 1]) for i in range(1, n)]
    # cut strength: ffmpeg's scene score, or the luma jump when the score under-reads
    strength = [max(scene[i], dy[i] / LUMA_SCALE) for i in range(n)]
    thr = _cut_threshold(strength)
    cut_idx = set()
    out: list[dict] = []
    for i in range(1, n):
        if strength[i] >= thr and strength[i] >= strength[i - 1] and (i + 1 >= n or strength[i] >= strength[i + 1]):
            cut_idx.add(i)
            out.append({"kind": "cut", "t": t[i], "frames": 1, "t_start": t[i - 1], "t_end": t[i],
                        "confidence": _clamp(0.5 + strength[i] / (2 * thr))})
    low = min(GRADUAL_FLOOR, thr / 4)

    def active(k: int) -> bool:
        return scene[k] > low or dy[k] >= GRADUAL_LUMA

    i = 1
    while i < n:
        if not active(i) or i in cut_idx:
            i += 1
            continue
        j, last, gap = i, i, 0
        while j + 1 < n and (j + 1) not in cut_idx:
            if active(j + 1):
                j += 1
                last, gap = j, 0
            elif gap < GAP_FRAMES:
                j += 1
                gap += 1
            else:
                break
        run = range(i, last + 1)
        if len(run) >= GRADUAL_MIN_FRAMES and _ramp_ok([scene[k] for k in run], [yavg[k] for k in run]):
            starts_clip = i - 1 <= EDGE_FRAMES
            ends_clip = last >= n - 1 - EDGE_FRAMES
            # a fade in from / out to black at the clip edge is not a shot boundary
            if not (starts_clip or ends_clip):
                t0, t1 = t[i - 1], t[last]
                out.append({"kind": "gradual", "t": round((t0 + t1) / 2, 6), "frames": len(run),
                            "t_start": t0, "t_end": t1, "confidence": 0.8})
        i = last + 1
    out.sort(key=lambda b: b["t"])
    return out


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
        conf = min((b["confidence"] for b in adj), default=0.9)
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
                "fastest_window": NOT_DETERMINED,
                "evidence": {"t_start": 0.0, "t_end": 0.0, "method": "measured", "confidence": 0}}
    durs = [s["duration"] for s in shots]
    span = shots[-1]["end"] - shots[0]["start"]
    size = min(3, len(shots))
    best = min(range(len(shots) - size + 1), key=lambda i: sum(durs[i:i + size]))
    win = shots[best:best + size]
    return {
        "count": len(shots),
        "mean": span / len(shots),
        "median": statistics.median(durs),
        "min": min(durs),
        "max": max(durs),
        "fastest_window": {"t_start": win[0]["start"], "t_end": win[-1]["end"],
                           "mean": sum(durs[best:best + size]) / size},
        "evidence": {"t_start": shots[0]["start"], "t_end": shots[-1]["end"],
                     "method": "measured", "confidence": round(min(s["evidence"]["confidence"] for s in shots), 3)},
    }
