"""Transition classification: cut / dissolve / fade to or from black or white.

Works only from the per-frame series (mean luma + tiny thumbnails). A gradual
boundary is a dissolve only when every in-between thumbnail is a blend of the
two end frames; a fade only when one end is near black / white. Anything else
is ``"not determined"`` with confidence 0 -- never a guess.
"""
from __future__ import annotations

import bisect

NOT_DETERMINED = "not determined"
BLACK = 18.0  # limited-range black is luma 16
WHITE = 233.0  # limited-range white is luma 235
EDGE_MIN_FRAMES = 3
EDGE_MIN_DROP = 20.0
BLEND_TOL = 0.06  # mean abs thumbnail residual (0..1) allowed against a straight blend


def _ev(t0: float, t1: float, conf: float) -> dict:
    return {"t_start": round(t0, 6), "t_end": round(t1, 6), "method": "measured",
            "confidence": round(max(0.0, min(1.0, conf)), 3)}


def _entry(t, kind, duration, t0, t1, conf) -> dict:
    return {"t": round(t, 6), "kind": kind, "duration": round(duration, 6), "evidence": _ev(t0, t1, conf)}


def _luma_kind(y: float) -> str | None:
    return "black" if y <= BLACK else "white" if y >= WHITE else None


def _blend_residual(thumb: list, i0: int, i1: int) -> float | None:
    """Worst-case-ish mean residual of in-between frames against the end-frame blend."""
    a, b = thumb[i0], thumb[i1]
    d = [y - x for x, y in zip(a, b)]
    dd = sum(v * v for v in d)
    if dd == 0 or i1 - i0 < 2:
        return None
    res = []
    for i in range(i0 + 1, i1):
        f = thumb[i]
        alpha = max(0.0, min(1.0, sum((fv - x) * dv for fv, x, dv in zip(f, a, d)) / dd))
        res.append(sum(abs(fv - (x + alpha * dv)) for fv, x, dv in zip(f, a, d)) / (255.0 * len(f)))
    return sum(res) / len(res)


def _classify_gradual(series: dict, b: dict) -> dict:
    t, yavg, thumb = series["t"], series["yavg"], series.get("thumb") or []
    ev = b["evidence"]
    t0, t1 = ev["t_start"], ev["t_end"]
    dur = t1 - t0
    i0 = max(0, bisect.bisect_left(t, t0 - 1e-9))
    i1 = min(len(t) - 1, bisect.bisect_left(t, t1 - 1e-9))
    k0, k1 = _luma_kind(yavg[i0]), _luma_kind(yavg[i1])
    if k1 and not k0:
        return _entry(b["t"], f"fade_to_{k1}", dur, t0, t1, 0.85)
    if k0 and not k1:
        return _entry(b["t"], f"fade_from_{k0}", dur, t0, t1, 0.85)
    if k0 or k1 or len(thumb) != len(t):
        return _entry(b["t"], NOT_DETERMINED, dur, t0, t1, 0)
    resid = _blend_residual(thumb, i0, i1)
    if resid is not None and resid <= BLEND_TOL:
        return _entry(b["t"], "dissolve", dur, t0, t1, 1 - resid / BLEND_TOL * 0.4)
    return _entry(b["t"], NOT_DETERMINED, dur, t0, t1, 0)


def _edge_fades(series: dict) -> list[dict]:
    """Fade from black/white at the clip start, to black/white at the clip end."""
    t, y = series["t"], series["yavg"]
    n = len(t)
    out: list[dict] = []
    if n < EDGE_MIN_FRAMES + 1:
        return out
    k = _luma_kind(y[0])
    if k:  # walk forward while luma keeps moving away from the extreme
        j = 0
        sign = 1 if k == "black" else -1
        while j + 2 < n and sign * (y[min(j + 2, n - 1)] - y[j]) > 0.3:
            j += 1
        while j + 1 < n and sign * (y[j + 1] - y[j]) > 0.3:
            j += 1
        if j >= EDGE_MIN_FRAMES and abs(y[j] - y[0]) >= EDGE_MIN_DROP:
            out.append(_entry((t[0] + t[j]) / 2, f"fade_from_{k}", t[j] - t[0], t[0], t[j], 0.85))
    k = _luma_kind(y[-1])
    if k:
        j = n - 1
        sign = 1 if k == "black" else -1
        while j - 2 >= 0 and sign * (y[j - 2] - y[j]) > 0.3:
            j -= 1
        while j - 1 >= 0 and sign * (y[j - 1] - y[j]) > 0.3:
            j -= 1
        if n - 1 - j >= EDGE_MIN_FRAMES and abs(y[j] - y[-1]) >= EDGE_MIN_DROP:
            out.append(_entry((t[j] + t[-1]) / 2, f"fade_to_{k}", t[-1] - t[j], t[j], t[-1], 0.85))
    return out


def classify(series: dict, shots: list[dict]) -> list[dict]:
    """One entry per boundary, plus clip-edge fades. Sorted by time."""
    if not series or len(series.get("t", [])) < 2:
        return []
    out: list[dict] = []
    for shot in shots or []:
        b = shot.get("boundary")
        if not b:
            continue
        if b["kind"] == "cut":
            out.append(_entry(b["t"], "cut", 0.0, b["evidence"]["t_start"], b["evidence"]["t_end"],
                              b["evidence"]["confidence"]))
        elif b["kind"] == "gradual":
            out.append(_classify_gradual(series, b))
        else:
            ev = b["evidence"]
            out.append(_entry(b["t"], NOT_DETERMINED, 0.0, ev["t_start"], ev["t_end"], 0))
    out += _edge_fades(series)
    out.sort(key=lambda e: e["t"])
    return out
