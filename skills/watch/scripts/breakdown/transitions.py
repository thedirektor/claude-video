"""Transition classification: cut / dissolve / fade to or from black or white.

Works only from the per-frame series (mean luma + tiny thumbnails). A gradual
boundary is a dissolve only when every in-between thumbnail is a blend of the
two end frames; a fade only when one end is near black / white. Anything else
is ``"not determined"`` with confidence 0 -- never a guess.
"""
from __future__ import annotations

import bisect
import statistics

NOT_DETERMINED = "not determined"
DARK_Y = 28.0  # a fade need not reach legal black (16); real fades end near 20
LIGHT_Y = 220.0
UNIFORM_STD = 10.0  # thumbnail std below this reads as a blank (faded) frame
TREND_STEP = 0.3
MIN_FRAMES = 3
MIN_DROP = 15.0
BLEND_TOL = 0.06  # mean abs thumbnail residual (0..1) allowed against a straight blend


def _ev(t0: float, t1: float, conf: float) -> dict:
    return {"t_start": round(t0, 6), "t_end": round(t1, 6), "method": "measured",
            "confidence": round(max(0.0, min(1.0, conf)), 3)}


def _entry(t, kind, duration, t0, t1, conf) -> dict:
    return {"t": round(t, 6), "kind": kind, "duration": round(duration, 6), "evidence": _ev(t0, t1, conf)}


def _extreme(series: dict, i: int) -> str | None:
    """"black"/"white" when frame i is a blank near-black / near-white frame."""
    y = series["yavg"][i]
    k = "black" if y <= DARK_Y else "white" if y >= LIGHT_Y else None
    if k is None:
        return None
    thumb = series.get("thumb") or []
    if len(thumb) == len(series["t"]) and statistics.pstdev(thumb[i]) >= UNIFORM_STD:
        return None
    return k


def _blend_residual(thumb: list, i0: int, i1: int) -> float | None:
    """Mean residual of in-between frames against the end-frame blend."""
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


def _walk(y: list[float], i: int, step: int, sign: int) -> int:
    """Follow luma monotonically away from an extreme (2-frame lookahead rides over plateaus)."""
    n = len(y)
    while True:
        j2, j1 = i + 2 * step, i + step
        if 0 <= j2 < n and sign * (y[j2] - y[i]) > TREND_STEP and sign * (y[j1] - y[i]) > -TREND_STEP:
            i = j1
        elif 0 <= j1 < n and sign * (y[j1] - y[i]) > TREND_STEP:
            i = j1
        else:
            return i


def _fades(series: dict) -> tuple[list[dict], list[tuple[float, float]]]:
    """Fades at clip edges and dips through black / white inside the clip.

    A dip is two entries: fade_to_<c> at its start and fade_from_<c> at its end.
    Returns the entries and the time spans of the dips (they replace any
    cut / gradual boundary inside them).
    """
    t, y = series["t"], series["yavg"]
    n = len(t)
    kinds = [_extreme(series, i) for i in range(n)]
    out: list[dict] = []
    spans: list[tuple[float, float]] = []
    i = 0
    while i < n:
        k = kinds[i]
        if k is None:
            i += 1
            continue
        j = i
        while j + 1 < n and kinds[j + 1] == k:
            j += 1
        sign = 1 if k == "black" else -1
        ext = min(y[i:j + 1]) if k == "black" else max(y[i:j + 1])
        a = _walk(y, i, -1, sign) if i > 0 else i
        b = _walk(y, j, 1, sign) if j < n - 1 else j
        to_ok = i > 0 and i - a >= MIN_FRAMES and sign * (y[a] - ext) >= MIN_DROP
        from_ok = j < n - 1 and b - j >= MIN_FRAMES and sign * (y[b] - ext) >= MIN_DROP
        if to_ok:
            out.append(_entry((t[a] + t[i]) / 2, f"fade_to_{k}", t[i] - t[a], t[a], t[i], 0.85))
        if from_ok:
            out.append(_entry((t[j] + t[b]) / 2, f"fade_from_{k}", t[b] - t[j], t[j], t[b], 0.85))
        if to_ok:
            spans.append((t[a], t[i]))
        if from_ok:
            spans.append((t[j], t[b]))
        if to_ok and from_ok:
            spans.append((t[a], t[b]))
        i = j + 1
    return out, spans


def _classify_gradual(series: dict, b: dict) -> dict:
    t, thumb = series["t"], series.get("thumb") or []
    ev = b["evidence"]
    t0, t1 = ev["t_start"], ev["t_end"]
    dur = t1 - t0
    i0 = max(0, bisect.bisect_left(t, t0 - 1e-9))
    i1 = min(len(t) - 1, bisect.bisect_left(t, t1 - 1e-9))
    k0, k1 = _extreme(series, i0), _extreme(series, i1)
    if k1 and not k0:
        return _entry(b["t"], f"fade_to_{k1}", dur, t0, t1, 0.8)
    if k0 and not k1:
        return _entry(b["t"], f"fade_from_{k0}", dur, t0, t1, 0.8)
    if k0 or k1 or len(thumb) != len(t):
        return _entry(b["t"], NOT_DETERMINED, dur, t0, t1, 0)
    resid = _blend_residual(thumb, i0, i1)
    if resid is not None and resid <= BLEND_TOL:
        return _entry(b["t"], "dissolve", dur, t0, t1, 1 - resid / BLEND_TOL * 0.4)
    return _entry(b["t"], NOT_DETERMINED, dur, t0, t1, 0)


def classify(series: dict, shots: list[dict]) -> list[dict]:
    """One entry per boundary, plus clip-edge fades and dips. Sorted by time."""
    if not series or len(series.get("t", [])) < 2:
        return []
    fades, spans = _fades(series)
    out: list[dict] = list(fades)
    for shot in shots or []:
        b = shot.get("boundary")
        if not b:
            continue
        ev = b["evidence"]
        if any(ev["t_start"] <= hi + 1e-9 and ev["t_end"] >= lo - 1e-9 for lo, hi in spans):
            continue  # a boundary on a fade / dip is that fade itself
        if b["kind"] == "cut":
            out.append(_entry(b["t"], "cut", 0.0, ev["t_start"], ev["t_end"], ev["confidence"]))
        elif b["kind"] == "gradual":
            out.append(_classify_gradual(series, b))
        else:
            out.append(_entry(b["t"], NOT_DETERMINED, 0.0, ev["t_start"], ev["t_end"], 0))
    out.sort(key=lambda e: e["t"])
    return out
