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
    return {"tempo_bpm": NOT_DETERMINED, "beats": [], "loudness": loudness or [],
            "evidence": _ev(0.0, duration or 0.0, 0.0), "notes": notes}


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
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
    mag = np.abs(np.fft.rfft(pad[idx] * np.hanning(N_FFT), axis=1))
    mag = np.log1p(30.0 * mag)
    flux = np.maximum(mag[1:] - mag[:-1], 0).sum(axis=1)
    # flux j compares frames j and j+1; frame j+1 starts at (j+1)*HOP-N_FFT in source time
    t0 = (HOP - N_FFT / 2) / SR
    return flux, t0, HOP / SR


def _comb_tempo(env, hop_s: float, np):
    e = env - env.mean()
    if not np.any(e):
        return None
    n = len(e)
    size = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(e, size)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    if ac[0] <= 0:
        return None
    ac = ac / ac[0]
    best = None
    scores = []
    for bpm in np.arange(BPM_MIN, BPM_MAX + 0.01, 0.5):
        lag = 60.0 / bpm / hop_s
        vals = []
        for k in (1, 2, 3, 4):
            c = lag * k
            if c + 2 >= n:
                break
            lo, hi = int(round(c)) - 1, int(round(c)) + 2
            vals.append(max(ac[lo:hi]))
        if len(vals) < 2:
            scores.append((0.0, bpm))
            continue
        scores.append((float(np.mean(vals)), bpm))
    top = max(s for s, _ in scores)
    if top <= 0:
        return None
    # tie-break toward the faster tempo: a double-time reading scores like the true one
    # only when it truly repeats at that rate, while half-time loses the odd multiples.
    cands = [(s, b) for s, b in scores if s >= top * 0.97]
    best = max(cands, key=lambda sb: sb[1])
    return best[1], best[0]


def _beat_grid(env, t0: float, hop_s: float, bpm: float, np):
    period = 60.0 / bpm / hop_s  # in frames
    n = len(env)
    e = env - np.median(env)
    e = np.maximum(e, 0)
    thr = float(e.max()) * 0.25 if e.max() > 0 else 0.0
    best_phase, best_sum = 0.0, -1.0
    for ph in np.arange(0.0, period, 0.5):
        pos = np.arange(ph, n, period)
        s = float(e[np.round(pos).astype(int).clip(0, n - 1)].sum())
        if s > best_sum:
            best_phase, best_sum = ph, s
    pos = np.arange(best_phase, n, period)
    # snap each grid slot to the strongest nearby onset; keep slots that have real support
    win = max(1, int(round(period * 0.15)))
    ks, ts = [], []
    for k, p in enumerate(pos):
        c = int(round(p))
        lo, hi = max(0, c - win), min(n, c + win + 1)
        if hi <= lo:
            continue
        j = lo + int(np.argmax(e[lo:hi]))
        if e[j] >= thr:
            ks.append(k)
            ts.append(j)
    if len(ks) < 3:
        return None
    # least-squares line over matched onsets refines the period and removes frame quantisation
    a, b = np.polyfit(np.array(ks, dtype=float), np.array(ts, dtype=float), 1)
    if a <= 0:
        return None
    refined_bpm = 60.0 / (a * hop_s)
    beats = [round(max(0.0, float(t0 + (b + a * k) * hop_s)), 3) for k in ks]
    return float(refined_bpm), beats


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
        grid = _beat_grid(env, t0, hop_s, tempo[0], np)
        if grid is None:
            return _nd(dur, ["tempo candidate not supported by repeated onsets"], loud)
        bpm, beats = grid
        if not (BPM_MIN - 2 <= bpm <= BPM_MAX + 2):
            return _nd(dur, ["refined tempo outside the 60-200 BPM range"], loud)
        conf = min(0.95, tempo[1])
        return {"tempo_bpm": round(bpm, 1), "beats": beats, "loudness": loud,
                "evidence": _ev(beats[0], beats[-1], conf, periodicity=round(tempo[1], 3)),
                "notes": []}
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
