"""Audio facts: tempo, beat times, loudness curve, share of cuts on the beat.

Decode is one ffmpeg pass (mono, 11025 Hz, s16). Tempo comes from the
autocorrelation of a spectral-flux onset envelope, scored as a comb over
several multiples of each candidate lag so half/double-time candidates lose to
the true period. A weak periodicity (noise, speech, ambience) is reported as
"not determined", never as a confident BPM. Missing numpy or ffmpeg, or a file
with no audio stream, also gives "not determined" plus a note; nothing raises.
"""
from __future__ import annotations

import math
import subprocess

NOT_DETERMINED = "not determined"
SR = 11025
N_FFT = 256
HOP = 64
BPM_MIN, BPM_MAX = 60.0, 200.0
MIN_STRENGTH = 0.35  # comb-averaged normalised autocorrelation needed to call a tempo
MIN_SECONDS = 3.0
CHUNK_FRAMES = 8192
PRIOR_BPM = 120.0
OFFBEAT_RATIO = 0.6
AMBIG_LOW = 0.35  # off/on ratio between this and OFFBEAT_RATIO cannot separate the octaves
AMBIG_BPM = (80.0, 100.0)  # where a double-time reading (160-200) is also musically plausible
AMBIG_CONF = 0.5
SUPPORT_REL = 0.3
STABILITY_MIN_S = 4.0
STABILITY_TOL = 0.06
LOUD_WINDOW = 0.5
LOUD_FLOOR = -70.0
TOLERANCE_FRAMES = 2


def _np():
    import numpy as np  # noqa: PLC0415
    return np


def _ev(t0: float, t1: float, conf: float, **extra) -> dict:
    ev = {"t_start": round(t0, 6), "t_end": round(t1, 6), "method": "measured",
          "confidence": round(max(0.0, min(1.0, conf)), 3)}
    ev.update(extra)
    return ev


def _nd(duration: float, notes: list[str], loudness: list | None = None) -> dict:
    loudness = loudness or []
    return {"tempo_bpm": NOT_DETERMINED, "beats": [], "loudness": loudness,
            "evidence": _ev(0.0, duration or 0.0, 0.0), "loudness_evidence": _loud_ev(loudness, duration),
            "notes": notes}


def _loud_ev(loud: list, duration: float) -> dict:
    if not loud:
        return _ev(0.0, duration or 0.0, 0.0)
    return _ev(loud[0]["t"], loud[-1]["t"] + LOUD_WINDOW, 0.9)


def _decode(path: str):
    """Mono float samples in [-1, 1], or (None, note)."""
    np = _np()
    try:
        res = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-vn",
             "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
            capture_output=True, timeout=1800)
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"audio decode failed: {type(exc).__name__}"
    raw = res.stdout
    if res.returncode != 0 or len(raw) < 2 * N_FFT:
        return None, "no audio stream"
    return np.frombuffer(raw[: len(raw) // 2 * 2], dtype="<i2").astype("float32") / 32768.0, ""


def _loudness(x, np) -> list[dict]:
    n = int(LOUD_WINDOW * SR)
    out = []
    for i in range(0, len(x), n):
        w = x[i:i + n]
        if len(w) < n // 2:
            break
        ms = float(np.mean(w.astype("float64") ** 2))
        lufs = LOUD_FLOOR if ms <= 1e-12 else max(LOUD_FLOOR, -0.691 + 10 * math.log10(ms))
        out.append({"t": round(i / SR, 3), "lufs_approx": round(lufs, 2)})
    return out


def _onset_envelope(x, np):
    pad = np.concatenate([np.zeros(N_FFT, dtype=x.dtype), x])  # lets an onset at t=0 register
    n = 1 + (len(pad) - N_FFT) // HOP
    if n < 4:
        return None
    win = np.hanning(N_FFT)
    cols = np.arange(N_FFT)[None, :]
    parts = []
    prev = None
    for c0 in range(0, n, CHUNK_FRAMES):  # bounded memory: never hold the whole STFT
        c1 = min(n, c0 + CHUNK_FRAMES)
        idx = cols + HOP * np.arange(c0, c1)[:, None]
        mag = np.log1p(30.0 * np.abs(np.fft.rfft(pad[idx] * win, axis=1)))
        if prev is not None:
            mag = np.vstack([prev, mag])
        if len(mag) > 1:
            parts.append(np.maximum(mag[1:] - mag[:-1], 0).sum(axis=1))
        prev = mag[-1:]
    flux = np.concatenate(parts)
    # flux j compares frames j and j+1; frame j+1 starts at (j+1)*HOP-N_FFT in source time
    t0 = (HOP - N_FFT / 2) / SR
    return flux, t0, HOP / SR


def _autocorr(env, np):
    """Overlap-normalised autocorrelation: ac[k] is a mean product, so long lags are not penalised."""
    e = env - env.mean()
    n = len(e)
    size = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(e, size)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    if ac[0] <= 0:
        return None
    ac = ac / (n - np.arange(n))
    return ac / ac[0]


def _prior(bpm: float) -> float:
    """Log-Gaussian tempo prior (centre 120 BPM, sigma one octave) to break octave ties."""
    return math.exp(-0.5 * (math.log2(bpm / PRIOR_BPM)) ** 2)


def _comb_tempo(env, hop_s: float, np):
    ac = _autocorr(env, np)
    if ac is None:
        return None
    n = len(ac)
    scores = []
    for bpm in np.arange(BPM_MIN, BPM_MAX + 0.01, 0.5):
        lag = 60.0 / bpm / hop_s
        vals = []
        for k in (1, 2, 3, 4):
            c = lag * k
            if c > 0.75 * n:
                break
            r = int(round(c))
            vals.append(max(ac[r - 1:r + 2]))
        scores.append((float(np.mean(vals)) if len(vals) >= 2 else 0.0, float(bpm)))
    top = max(s for s, _ in scores)
    if top <= 0:
        return None
    cands = [(s, b) for s, b in scores if s >= top * 0.9]
    s_best, b_best = max(cands, key=lambda sb: sb[0] * _prior(sb[1]))
    return b_best, s_best


def _slot_peaks(e, start: float, period: float, win: int, np):
    n = len(e)
    out = []
    for p in np.arange(start, n, period):
        c = int(round(p))
        lo, hi = max(0, c - win), min(n, c + win + 1)
        out.append(float(e[lo:hi].max()) if hi > lo else 0.0)
    return out


def _best_phase(e, period: float, np):
    n = len(e)
    best_phase, best_sum = 0.0, -1.0
    for ph in np.arange(0.0, period, 0.5):
        pos = np.arange(ph, n, period)
        sm = float(e[np.round(pos).astype(int).clip(0, n - 1)].sum())
        if sm > best_sum:
            best_phase, best_sum = ph, sm
    return best_phase


def _resolve_octave(env, hop_s: float, bpm: float, np):
    """Half-time guard. Returns ``(bpm, alt)``: ``alt`` is the other octave when the evidence
    cannot separate them (slots midway between beats carry onsets of middling strength)."""
    if bpm * 2 > BPM_MAX:
        return bpm, None
    e = np.maximum(env - np.median(env), 0)
    period = 60.0 / bpm / hop_s
    win = max(1, int(round(period * 0.08)))
    ph = _best_phase(e, period, np)
    on = _slot_peaks(e, ph, period, win, np)
    off = _slot_peaks(e, ph + period / 2, period, win, np)
    m = min(len(on), len(off))
    if m < 4:
        return bpm, None
    on_m, off_m = float(np.median(on[:m])), float(np.median(off[:m]))
    if on_m <= 0:
        return bpm, None
    ratio = off_m / on_m
    if ratio >= OFFBEAT_RATIO:
        return bpm * 2, None
    if ratio >= AMBIG_LOW and AMBIG_BPM[0] <= bpm <= AMBIG_BPM[1]:
        return bpm, bpm * 2
    return bpm, None


def _fold_score(e, per: float, np):
    """Best phase and strength of ``e`` folded at a (fractional) period, half-frame phase bins."""
    nb = int(per * 2) + 2
    bins = (np.mod(np.arange(len(e)), per) * 2).astype(int)
    sums = np.bincount(bins, weights=e, minlength=nb)
    sm = sums + np.roll(sums, 1) + np.roll(sums, -1)
    j = int(np.argmax(sm))
    return float(sm[j]), j / 2.0


def _refine_period(e, per0: float, np):
    """Two-pass fractional search around the comb tempo: the 0.5 BPM grid alone drifts by
    hundreds of ms over minutes. Returns ``(period_frames, phase_frames)``."""
    best = (-1.0, per0, 0.0)
    for rel in np.arange(-0.012, 0.01201, 0.0005):
        sc, ph = _fold_score(e, per0 * (1 + rel), np)
        if sc > best[0]:
            best = (sc, per0 * (1 + rel), ph)
    centre = best[1]
    for rel in np.arange(-0.0005, 0.00051, 0.00005):
        per = centre * (1 + rel)
        sc, ph = _fold_score(e, per, np)
        if sc > best[0]:
            best = (sc, per, ph)
    return best[1], best[2]


def _snap(e, positions, win: int, np):
    n = len(e)
    out = []
    for p in positions:
        c = int(round(p))
        lo, hi = max(0, c - win), min(n, c + win + 1)
        if hi > lo:
            j = lo + int(np.argmax(e[lo:hi]))
            out.append((j, float(e[j])))
        else:
            out.append((c, 0.0))
    return out


def _fit(ks, ts, np):
    a, b = np.polyfit(np.array(ks, dtype=float), np.array(ts, dtype=float), 1)
    return a, b


def _beat_grid(env, t0: float, hop_s: float, bpm: float, np):
    n = len(env)
    e = np.maximum(env - np.median(env), 0)
    period, phase = _refine_period(e, 60.0 / bpm / hop_s, np)
    win = max(1, int(round(period * 0.15)))
    a, b = period, phase
    ks = []
    for _ in range(2):  # re-snap against the fitted line so the fit cannot lock on a drifted grid
        count = int((n - b) / a) + 1
        peaks = _snap(e, b + a * np.arange(max(count, 0)), win, np)
        if not peaks:
            return None
        # support is relative to the typical slot, not the loudest hit: weaker true beats stay
        thr = SUPPORT_REL * float(np.median([v for _, v in peaks]))
        ks = [k for k, (_, v) in enumerate(peaks) if v >= thr and v > 0]
        if len(ks) < 3:
            return None
        a, b = _fit(ks, [peaks[k][0] for k in ks], np)
        if a <= 0:
            return None
    refined_bpm = 60.0 / (a * hop_s)
    beats = [round(max(0.0, float(t0 + (b + a * k) * hop_s)), 3) for k in range(ks[0], ks[-1] + 1)]
    return float(refined_bpm), beats


def _stable(env, hop_s: float, np) -> str:
    """"" when the tempo holds across sub-windows, else a note describing the disagreement.
    Raw readings are compared with octaves folded: 70 and 140 are the same pulse."""
    n = len(env)
    if n * hop_s < 2 * STABILITY_MIN_S:
        return ""
    found = []
    for a, b in ((0.0, 0.5), (0.5, 1.0), (0.25, 0.75)):
        got = _comb_tempo(env[int(a * n):int(b * n)], hop_s, np)
        if got is not None and got[1] >= MIN_STRENGTH:
            found.append(got[0])
    if len(found) < 2:
        return ""
    lo, hi = min(found), max(found)
    x = math.log2(hi / lo)
    if abs(x - round(x)) > math.log2(1 + STABILITY_TOL):
        return f"tempo changes within the clip ({lo:.0f} to {hi:.0f} BPM): no single tempo"
    return ""


def analyze(path: str, duration: float = 0.0) -> dict:
    """``{"tempo_bpm", "beats", "loudness", "evidence", "notes"}``; never raises."""
    try:
        np = _np()
    except ImportError:
        return _nd(duration, ["numpy not installed: tempo, beats and loudness not determined"])
    try:
        x, note = _decode(path)
        if x is None:
            return _nd(duration, [note])
        dur = float(duration) if duration else len(x) / SR
        loud = _loudness(x, np)
        if len(x) / SR < MIN_SECONDS or float(np.max(np.abs(x))) < 1e-3:
            return _nd(dur, ["audio too short or silent for tempo"], loud)
        got = _onset_envelope(x, np)
        if got is None:
            return _nd(dur, ["audio too short for tempo"], loud)
        env, t0, hop_s = got
        tempo = _comb_tempo(env, hop_s, np)
        if tempo is None or tempo[1] < MIN_STRENGTH:
            strength = 0.0 if tempo is None else tempo[1]
            return _nd(dur, [f"no steady pulse found (periodicity {strength:.2f} < {MIN_STRENGTH})"], loud)
        change = _stable(env, hop_s, np)
        if change:
            return _nd(dur, [change], loud)
        bpm0, alt = _resolve_octave(env, hop_s, tempo[0], np)
        grid = _beat_grid(env, t0, hop_s, bpm0, np)
        if grid is None:
            return _nd(dur, ["tempo candidate not supported by repeated onsets"], loud)
        bpm, beats = grid
        if not (BPM_MIN - 2 <= bpm <= BPM_MAX + 2):
            return _nd(dur, ["refined tempo outside the 60-200 BPM range"], loud)
        conf = min(0.95, tempo[1])
        notes: list[str] = []
        extra: dict = {}
        if alt:
            conf = min(conf, AMBIG_CONF)
            notes.append("half/double-time ambiguous")
            extra["tempo_alt"] = round(alt * bpm / bpm0, 1)
        return {"tempo_bpm": round(bpm, 1), "beats": beats, "loudness": loud, **extra,
                "evidence": _ev(beats[0], beats[-1], conf, periodicity=round(tempo[1], 3)),
                "loudness_evidence": _loud_ev(loud, dur),
                "notes": notes}
    except Exception as exc:  # noqa: BLE001 - a command must never raise out of an analysis
        return _nd(duration, [f"audio analysis failed: {type(exc).__name__}"])


def cuts_on_beat(shots: list[dict], beats: list[float], fps: float):
    """Share of hard cuts within +-2 source frames of a beat, or "not determined"."""
    if not beats or not fps or fps <= 0:
        return NOT_DETERMINED
    cuts = [s["boundary"]["t"] for s in shots or []
            if s.get("boundary") and s["boundary"].get("kind") == "cut"]
    if not cuts:
        return NOT_DETERMINED
    tol = TOLERANCE_FRAMES / fps + 1e-6
    hits = sum(1 for c in cuts if min(abs(c - b) for b in beats) <= tol)
    n = len(cuts)
    return {"share": round(hits / n, 3), "tolerance_frames": TOLERANCE_FRAMES,
            "evidence": _ev(min(cuts), max(cuts), 0.9 * min(1.0, n / 4), n_cuts=n, n_on_beat=hits)}
