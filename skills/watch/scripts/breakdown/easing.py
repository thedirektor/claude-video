"""Godot 4 ``Tween`` TRANS x EASE curves and a fitter for sampled motion.

The curves are the standard Penner equations as Godot 4 implements them
(BACK overshoot constant 1.70158; ``OUT_IN`` is the out-curve over the first
half and the in-curve over the second). ``fit`` picks the (trans, ease) pair
that best explains normalised samples, also fits a CSS ``cubic-bezier`` and
emits Unity ``AnimationCurve`` keys. Pure python (no numpy), so it works
wherever the plugin runs.

A fit is never forced: when no curve explains the samples within
``MAX_RMS`` the Godot pair is the string ``"not determined"``, and when the
runner-up is practically as good the result is flagged ``ambiguous`` with a
lower confidence and the runner-up listed.
"""
from __future__ import annotations

import math

NOT_DETERMINED = "not determined"
MAX_RMS = 0.06  # normalised RMS above which a curve is not trusted
AMBIG_ABS = 0.002  # runner-up closer than this (absolute RMS) is indistinguishable
AMBIG_REL = 0.35  # ...or closer than this share of the best RMS
MIN_SAMPLES = 4

TRANS = ["LINEAR", "SINE", "QUINT", "QUART", "QUAD", "EXPO", "ELASTIC", "CUBIC", "CIRC", "BOUNCE", "BACK", "SPRING"]
EASES = ["IN", "OUT", "IN_OUT", "OUT_IN"]

_C1 = 1.70158
_C2 = _C1 * 1.525
_C3 = _C1 + 1.0
_C4 = 2.0 * math.pi / 3.0
_C5 = 2.0 * math.pi / 4.5


def _pow_in(p):
    return lambda t: t ** p


def _sine_in(t):
    return 1.0 - math.cos(t * math.pi / 2.0)


def _expo_in(t):
    return 0.0 if t <= 0.0 else 2.0 ** (10.0 * t - 10.0)


def _circ_in(t):
    return 1.0 - math.sqrt(max(0.0, 1.0 - t * t))


def _back_in(t):
    return _C3 * t ** 3 - _C1 * t ** 2


def _elastic_in(t):
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return -(2.0 ** (10.0 * t - 10.0)) * math.sin((10.0 * t - 10.75) * _C4)


def _bounce_out(t):
    n1, d1 = 7.5625, 2.75
    if t < 1.0 / d1:
        return n1 * t * t
    if t < 2.0 / d1:
        t -= 1.5 / d1
        return n1 * t * t + 0.75
    if t < 2.5 / d1:
        t -= 2.25 / d1
        return n1 * t * t + 0.9375
    t -= 2.625 / d1
    return n1 * t * t + 0.984375


def _bounce_in(t):
    return 1.0 - _bounce_out(1.0 - t)


def _spring_out(t):
    t = min(1.0, max(0.0, t))
    s = 1.0 - t
    return (math.sin(t * math.pi * (0.2 + 2.5 * t ** 3)) * s ** 2.2 + t) * (1.0 + 1.2 * s)


def _spring_in(t):
    return 1.0 - _spring_out(1.0 - t)


_IN = {
    "LINEAR": lambda t: t,
    "SINE": _sine_in,
    "QUINT": _pow_in(5),
    "QUART": _pow_in(4),
    "QUAD": _pow_in(2),
    "EXPO": _expo_in,
    "ELASTIC": _elastic_in,
    "CUBIC": _pow_in(3),
    "CIRC": _circ_in,
    "BOUNCE": _bounce_in,
    "BACK": _back_in,
    "SPRING": _spring_in,
}


def _clamp01(t: float) -> float:
    return 0.0 if t < 0.0 else 1.0 if t > 1.0 else t


def _build(trans: str) -> dict:
    fin = _IN[trans]

    def f_in(t):
        t = _clamp01(t)
        return fin(t)

    def f_out(t):
        t = _clamp01(t)
        return 1.0 - fin(1.0 - t)

    def f_in_out(t):
        t = _clamp01(t)
        return fin(2.0 * t) / 2.0 if t < 0.5 else 1.0 - fin(2.0 - 2.0 * t) / 2.0

    def f_out_in(t):
        t = _clamp01(t)
        return (1.0 - fin(1.0 - 2.0 * t)) / 2.0 if t < 0.5 else fin(2.0 * t - 1.0) / 2.0 + 0.5

    out = {"IN": f_in, "OUT": f_out, "IN_OUT": f_in_out, "OUT_IN": f_out_in}
    if trans == "BACK":
        def back_in_out(t):
            t = _clamp01(t)
            if t < 0.5:
                return ((2.0 * t) ** 2 * ((_C2 + 1.0) * 2.0 * t - _C2)) / 2.0
            return ((2.0 * t - 2.0) ** 2 * ((_C2 + 1.0) * (2.0 * t - 2.0) + _C2) + 2.0) / 2.0
        out["IN_OUT"] = back_in_out
    if trans == "ELASTIC":
        def elastic_in_out(t):
            t = _clamp01(t)
            if t <= 0.0:
                return 0.0
            if t >= 1.0:
                return 1.0
            if t < 0.5:
                return -(2.0 ** (20.0 * t - 10.0) * math.sin((20.0 * t - 11.125) * _C5)) / 2.0
            return (2.0 ** (-20.0 * t + 10.0) * math.sin((20.0 * t - 11.125) * _C5)) / 2.0 + 1.0
        out["IN_OUT"] = elastic_in_out
    return out


CURVES: dict[str, dict] = {tr: _build(tr) for tr in TRANS}

# preference when two curves explain the data equally well: simpler first
_COMPLEXITY = {"LINEAR": 0, "QUAD": 1, "CUBIC": 1, "SINE": 1, "QUART": 2, "QUINT": 2, "EXPO": 2, "CIRC": 2,
               "BACK": 3, "BOUNCE": 4, "ELASTIC": 4, "SPRING": 4}
_TIE = 0.0004


def curve(trans: str, ease: str):
    return CURVES[trans][ease]


def _candidates() -> list[tuple[str, str]]:
    out = [("LINEAR", "IN_OUT")]  # LINEAR ignores ease; Godot's default ease is IN_OUT
    for tr in TRANS:
        if tr == "LINEAR":
            continue
        for ea in EASES:
            out.append((tr, ea))
    return out


CANDIDATES = _candidates()


def _rms(f, xs, ys) -> float:
    acc = 0.0
    for x, y in zip(xs, ys):
        d = f(x) - y
        acc += d * d
    return math.sqrt(acc / len(xs)) if xs else float("inf")


def normalise(t, v, t0=None, t1=None, v0=None, v1=None):
    """Samples -> (xs, ys) on the unit square, or None when degenerate."""
    t0 = t[0] if t0 is None else t0
    t1 = t[-1] if t1 is None else t1
    v0 = v[0] if v0 is None else v0
    v1 = v[-1] if v1 is None else v1
    if t1 - t0 <= 1e-9 or abs(v1 - v0) <= 1e-9:
        return None
    xs = [_clamp01((ti - t0) / (t1 - t0)) for ti in t]
    ys = [(vi - v0) / (v1 - v0) for vi in v]
    return xs, ys


def rank(xs, ys) -> list[dict]:
    """Every candidate with its RMS, best first (ties broken towards simpler curves)."""
    res = [{"trans": tr, "ease": ea, "rms": _rms(CURVES[tr][ea], xs, ys)} for tr, ea in CANDIDATES]
    res.sort(key=lambda r: r["rms"] + _TIE * _COMPLEXITY[r["trans"]])
    return res


# ---- cubic-bezier (CSS) ---------------------------------------------------

def _bezier_table(p, n=48):
    x1, y1, x2, y2 = p
    tx, ty = [], []
    for i in range(n + 1):
        s = i / n
        u = 1.0 - s
        tx.append(3 * u * u * s * x1 + 3 * u * s * s * x2 + s ** 3)
        ty.append(3 * u * u * s * y1 + 3 * u * s * s * y2 + s ** 3)
    return tx, ty


def _bezier_rms(p, xs, ys) -> float:
    tx, ty = _bezier_table(p)
    n = len(tx) - 1
    acc = 0.0
    j = 0
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    for i in order:
        x = xs[i]
        while j < n - 1 and tx[j + 1] < x:
            j += 1
        x0, x1 = tx[j], tx[j + 1]
        w = 0.0 if x1 - x0 <= 1e-12 else min(1.0, max(0.0, (x - x0) / (x1 - x0)))
        y = ty[j] + (ty[j + 1] - ty[j]) * w
        d = y - ys[i]
        acc += d * d
    return math.sqrt(acc / len(xs))


def fit_bezier(xs, ys) -> tuple[list[float], float]:
    starts = [(0.215, 0.61, 0.355, 1.0), (0.55, 0.055, 0.675, 0.19), (0.25, 0.1, 0.25, 1.0), (0.42, 0.0, 1.0, 1.0), (0.0, 0.0, 0.58, 1.0), (0.42, 0.0, 0.58, 1.0),
              (0.33, 1.0, 0.68, 1.0), (0.33, 0.0, 0.67, 1.0), (0.0, 0.0, 1.0, 1.0)]
    lo, hi = (0.0, -1.0, 0.0, -1.0), (1.0, 2.0, 1.0, 2.0)
    best_p, best_e = None, float("inf")
    for start in starts:
        p = list(start)
        e = _bezier_rms(p, xs, ys)
        step = 0.2
        while step > 0.002:
            improved = False
            for k in range(4):
                for sgn in (1, -1):
                    q = list(p)
                    q[k] = min(hi[k], max(lo[k], q[k] + sgn * step))
                    eq = _bezier_rms(q, xs, ys)
                    if eq < e - 1e-9:
                        p, e, improved = q, eq, True
            if not improved:
                step /= 2.0
        if e < best_e:
            best_p, best_e = p, e
    return [round(c, 3) for c in best_p], best_e


# ---- Unity keys -----------------------------------------------------------

def unity_keys(f, n: int = 5) -> list[dict]:
    """Hermite keys (normalised time/value, tangents = value per unit time) of curve ``f``."""
    h = 1e-3
    keys = []
    for i in range(n):
        t = i / (n - 1)
        a, b = max(0.0, t - h), min(1.0, t + h)
        slope = (f(b) - f(a)) / (b - a)
        keys.append({"time": round(t, 4), "value": round(f(t), 5),
                     "inTangent": round(slope, 4), "outTangent": round(slope, 4)})
    return keys


def _keys_from_samples(xs, ys, n: int = 5) -> list[dict]:
    pts = sorted(zip(xs, ys))

    def at(x):
        for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
            if xa <= x <= xb:
                return ya if xb == xa else ya + (yb - ya) * (x - xa) / (xb - xa)
        return pts[-1][1]
    return unity_keys(at, n)


# ---- the fit --------------------------------------------------------------

def _undetermined(note: str, rms: float | None = None, n: int = 0) -> dict:
    return {"godot": NOT_DETERMINED, "bezier": NOT_DETERMINED, "unity_keys": NOT_DETERMINED,
            "error_rms": rms if rms is not None else NOT_DETERMINED, "confidence": 0.0,
            "runner_up": None, "ambiguous": False, "samples": n, "notes": [note]}


def _assemble(ranked: list[dict], xs, ys) -> dict:
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    n = len(xs)
    bez, bez_err = fit_bezier(xs, ys)
    out: dict = {"samples": n, "notes": []}
    if best["rms"] > MAX_RMS:
        out["godot"] = NOT_DETERMINED
        out["unity_keys"] = NOT_DETERMINED if bez_err > MAX_RMS else _keys_from_samples(xs, ys)
        out["confidence"] = 0.0
        out["runner_up"] = None
        out["ambiguous"] = False
        out["notes"].append(f"no Godot TRANS x EASE curve fits (best RMS {best['rms']:.3f} > {MAX_RMS})")
    else:
        out["godot"] = {"trans": best["trans"], "ease": best["ease"]}
        out["unity_keys"] = unity_keys(CURVES[best["trans"]][best["ease"]])
        ambiguous = bool(second) and (second["rms"] - best["rms"]) < max(AMBIG_ABS, AMBIG_REL * best["rms"])
        conf = max(0.0, 1.0 - best["rms"] / MAX_RMS)
        if ambiguous:
            conf *= 0.55
            out["notes"].append(f"{best['trans']}/{best['ease']} is nearly indistinguishable from "
                                f"{second['trans']}/{second['ease']}")
        if n < 8:
            conf *= 0.7
        out["confidence"] = round(conf, 3)
        out["ambiguous"] = ambiguous
        out["runner_up"] = ({"trans": second["trans"], "ease": second["ease"], "error_rms": round(second["rms"], 5)}
                            if second else None)
    out["error_rms"] = round(best["rms"], 5)
    if bez_err <= MAX_RMS:
        out["bezier"] = bez
        out["bezier_error_rms"] = round(bez_err, 5)
    else:
        out["bezier"] = NOT_DETERMINED
        out["bezier_error_rms"] = round(bez_err, 5)
    return out


def fit(t, v, t0=None, t1=None, v0=None, v1=None) -> dict:
    """Fit samples ``(t, v)`` (any units) to the Godot curve set, a CSS bezier and Unity keys.

    The motion is taken as running from ``(t0, v0)`` to ``(t1, v1)`` (default the
    first and last sample); samples outside ``[t0, t1]`` count as the plateau.
    """
    t, v = list(t), list(v)
    if len(t) != len(v) or len(t) < MIN_SAMPLES:
        return _undetermined("too few samples to fit a curve", n=len(t))
    nm = normalise(t, v, t0, t1, v0, v1)
    if nm is None:
        return _undetermined("no net change between the endpoints", n=len(t))
    xs, ys = nm
    return _assemble(rank(xs, ys), xs, ys)


def fit_bounds(t, v, s_cands, e_cands, top: int = 12) -> dict:
    """Like ``fit`` but also searches the exact start/end of the move.

    ``t, v`` span the move plus the rest plateaus on either side (``v[0]`` and
    ``v[-1]`` are the plateau values). The move's start is searched in
    ``s_cands`` and its end in ``e_cands``; the result carries ``t_start`` and
    ``t_end``. Ease-out tails make the end invisible to a velocity threshold,
    so the end is the one that makes the whole curve fit best.
    """
    t, v = list(t), list(v)
    if len(t) != len(v) or len(t) < MIN_SAMPLES:
        return _undetermined("too few samples to fit a curve", n=len(t))
    v0, v1 = v[0], v[-1]
    s0, e0 = s_cands[0], e_cands[0]
    if normalise(t, v, s0, e0, v0, v1) is None:
        return _undetermined("no net change between the endpoints", n=len(t))
    # shortlist from several plausible ends: a curve far off at the first end can be right at a later one
    picks = sorted({0, len(e_cands) // 3, (2 * len(e_cands)) // 3, len(e_cands) - 1})
    shortlist = []
    for pi in picks:
        xs0, ys0 = normalise(t, v, s0, e_cands[pi], v0, v1)
        for r in rank(xs0, ys0)[:top]:
            if (r["trans"], r["ease"]) not in shortlist:
                shortlist.append((r["trans"], r["ease"]))
    rows = []
    for tr, ea in shortlist:
        f = CURVES[tr][ea]
        best = (float("inf"), s0, e0)
        for s in s_cands:
            for e in e_cands:
                if e - s <= 1e-6:
                    continue
                xs, ys = normalise(t, v, s, e, v0, v1)
                r = _rms(f, xs, ys)
                if r < best[0]:
                    best = (r, s, e)
        rows.append({"trans": tr, "ease": ea, "rms": best[0], "s": best[1], "e": best[2]})
    rows.sort(key=lambda r: r["rms"] + _TIE * _COMPLEXITY[r["trans"]])
    top_row = rows[0]
    xs, ys = normalise(t, v, top_row["s"], top_row["e"], v0, v1)
    res = _assemble([{"trans": r["trans"], "ease": r["ease"], "rms": r["rms"]} for r in rows], xs, ys)
    res["t_start"], res["t_end"] = top_row["s"], top_row["e"]
    return res
