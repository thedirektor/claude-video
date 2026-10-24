"""Audio facts: tempo, beat times, loudness curve, share of cuts on the beat.

Decode is one ffmpeg pass (mono, 11025 Hz, s16). Tempo comes from the
autocorrelation of a spectral-flux onset envelope, scored as a comb over
several multiples of each candidate lag so half/double-time candidates lose to
the true period. Which octave (T or 2T) is the beat is judged per frequency band
(uncompressed low/mid/high flux) plus the comb periodicity of both octaves; when they do
not agree the tempo is reported with ``tempo_alt`` and confidence <= 0.5, never as a
confident wrong octave. A weak periodicity (noise, speech, ambience) is reported as
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
LOW_BAND_MIN_STRENGTH = 0.6  # the kick band alone must be this periodic to stand in for the full band
MIN_SECONDS = 3.0
CHUNK_FRAMES = 8192
PRIOR_BPM = 120.0
# Octave decision (lo vs 2*lo, both inside 60-200 BPM). r = off-slot / on-slot onset strength
# of one frequency band on the lo grid; the off-slots are the beats only the faster octave claims.
FAST_PULSE_R = 0.8  # low (kick) band: off-beats carry the same pulse -> faster can be confident
FAST_MIN_R = 0.3  # ...and no band has nearly empty off-beats
PRIMARY_R = 0.5  # ambiguous: pulse-band r at or above this makes the faster octave the primary reading
BAND_MIN_LEVEL = 3.0  # slot medians below this many times the band mean: the band carries no pulse
MIX_R_CHANGE = 2.0  # mean off/on of a band must change by this factor between halves for a real section change
MIX_R_DELTA = 0.3
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
    err = (res.stderr or b"").decode("utf-8", "replace")
    if res.returncode != 0:
        if "does not contain any stream" in err or "matches no streams" in err:
            return None, "no audio stream"
        return None, f"audio decode failed (ffmpeg exit {res.returncode})"
    if len(raw) < 2 * N_FFT:
        return None, "no audio stream"  # decoded fine but (nearly) nothing came out
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


BANDS = ((1, 5), (5, 47), (47, N_FFT // 2 + 1))  # rfft bins: ~43-172 Hz, ~215-2000 Hz, ~2-5.5 kHz


def _onset_envelope(x, np):
    """``(flux, t0, hop_s)``: the full-band onset envelope."""
    got = _onset_bands(x, np)
    return None if got is None else got[:3]


def _onset_bands(x, np):
    """``(flux, t0, hop_s, band_flux)``; ``band_flux`` is one flux row per entry of ``BANDS``
    (low/mid/high) so a kick that is quiet in the full-band sum still shows up in its own band."""
    pad = np.concatenate([np.zeros(N_FFT, dtype=x.dtype), x])  # lets an onset at t=0 register
    n = 1 + (len(pad) - N_FFT) // HOP
    if n < 4:
        return None
    win = np.hanning(N_FFT)
    cols = np.arange(N_FFT)[None, :]
    parts, bparts = [], []
    prev = None
    for c0 in range(0, n, CHUNK_FRAMES):  # bounded memory: never hold the whole STFT
        c1 = min(n, c0 + CHUNK_FRAMES)
        idx = cols + HOP * np.arange(c0, c1)[:, None]
        raw = np.abs(np.fft.rfft(pad[idx] * win, axis=1))
        if prev is not None:
            raw = np.vstack([prev, raw])
        if len(raw) > 1:
            mag = np.log1p(30.0 * raw)
            parts.append(np.maximum(mag[1:] - mag[:-1], 0).sum(axis=1))
            # bands use uncompressed magnitude: log compression lifts quiet hats/noise to the
            # level of a kick, which is exactly what the octave decision must not do
            d = np.maximum(raw[1:] - raw[:-1], 0)
            bparts.append(np.stack([d[:, a:b].sum(axis=1) for a, b in BANDS]))
        prev = raw[-1:]
    flux = np.concatenate(parts)
    bands = np.concatenate(bparts, axis=1)
    # flux j compares frames j and j+1; frame j+1 starts at (j+1)*HOP-N_FFT in source time
    t0 = (HOP - N_FFT / 2) / SR
    return flux, t0, HOP / SR, bands


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


def _pulse(env, bands, hop_s: float, np):
    """``(env_used, (bpm, strength))`` or ``(env, None)``. The full-band flux first; when it
    shows no steady pulse the low (kick) band's own flux is tried with a higher bar, because a
    kick-only pattern is buried under the other bins' noise in the full-band sum."""
    got = _comb_tempo(env, hop_s, np)
    if got is not None and got[1] >= MIN_STRENGTH:
        return env, got
    if bands is not None and len(bands):
        low = _comb_tempo(bands[0], hop_s, np)
        if low is not None and low[1] >= LOW_BAND_MIN_STRENGTH:
            return bands[0], low
    return env, got


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


def _comb_at(ac, lag: float) -> float:
    n = len(ac)
    vals = []
    for k in (1, 2, 3, 4):
        c = lag * k
        if c > 0.75 * n:
            break
        r = int(round(c))
        vals.append(max(ac[r - 1:r + 2]))
    return sum(vals) / len(vals) if len(vals) >= 2 else 0.0


def _octave_grid(env, hop_s: float, lo: float, np):
    """``(period_frames, phase_frames)`` of the ``lo`` beat grid fitted to the whole envelope."""
    e = np.maximum(env - np.median(env), 0)
    # refine first: the 0.5 BPM comb grid drifts by a large part of a beat over a minute,
    # which smears on- and off-slots together
    period, _ = _refine_period(e, 60.0 / lo / hop_s, np)
    return period, _best_phase(e, period, np)


def _octave_stats(env, bands, hop_s: float, lo: float, np, grid=None):
    """Evidence for ``lo`` vs ``2*lo``: full-band comb periodicity of both octaves, and per band
    the median on-slot (``lo`` beats) and off-slot (midpoints) onset strength over the band level.
    ``grid`` pins the (period, phase) fitted on the whole clip, so that sections of one clip are
    judged on the same beat slots instead of each locking on whichever phase suits it."""
    e = np.maximum(env - np.median(env), 0)
    period, ph = grid if grid is not None else _octave_grid(env, hop_s, lo, np)
    win = max(1, int(round(period * 0.08)))
    ac = _autocorr(env, np)
    out = {"s_lo": _comb_at(ac, period) if ac is not None else 0.0,
           "s_hi": _comb_at(ac, period / 2) if ac is not None else 0.0, "bands": [], "mean_r": []}
    for row in bands:
        eb = np.maximum(row - np.median(row), 0)
        on = _slot_peaks(eb, ph, period, win, np)
        off = _slot_peaks(eb, ph + period / 2, period, win, np)
        m = min(len(on), len(off))
        level = float(np.mean(eb)) or 1e-12
        out["bands"].append((float(np.median(on[:m])) / level, float(np.median(off[:m])) / level))
        # the mean, unlike the median, does not flip when a slot is occupied half the time
        out["mean_r"].append(float(np.mean(off[:m])) / (float(np.mean(on[:m])) or 1e-12))
    return out


def _octave_pair(bpm: float):
    """``(lo, 2*lo)`` when both octaves of ``bpm`` fall inside 60-200 BPM, else None."""
    if bpm * 2 <= BPM_MAX:
        return bpm, bpm * 2
    if bpm / 2 >= BPM_MIN:
        return bpm / 2, bpm
    return None


def _resolve_octave(env, bands, hop_s: float, bpm: float, np, grid=None, stats_out: dict | None = None):
    """``(primary_bpm, alt)``: ``alt`` is the other octave when the octave is ambiguous, else None.

    Faster is confident only when the low (kick) band keeps its pulse on the off-beats and no
    band leaves them nearly empty. The slower octave is never confident: inside 60-200 BPM a slow
    backbeat and a half-time groove (kick on 1, snare on 3) are the same evidence, a genuine 2:1
    ambiguity; the pulse band then only picks the primary. Over-flagging an ordinary groove is
    acceptable, a confident wrong octave is not. ``stats_out``, when given, receives the octave statistics."""
    pair = _octave_pair(bpm)
    if pair is None:
        return bpm, None
    lo, hi = pair
    st = _octave_stats(env, bands, hop_s, lo, np, grid)
    if stats_out is not None:
        stats_out.update(st)
    rs = [off / on if on > 0 else float("inf") for on, off in st["bands"] if max(on, off) >= BAND_MIN_LEVEL]
    if not rs:
        return bpm, (lo if bpm == hi else hi)
    on0, off0 = st["bands"][0]
    low_ok = max(on0, off0) >= BAND_MIN_LEVEL
    pulse_r = (off0 / on0 if on0 > 0 else float("inf")) if low_ok else float(np.median(rs))
    if pulse_r >= FAST_PULSE_R and min(rs) >= FAST_MIN_R:
        return hi, None
    return (hi, lo) if pulse_r >= PRIMARY_R else (lo, hi)


def _fold_score(e, per: float, np):
    """Best phase and strength of ``e`` folded at a (fractional) period, half-frame phase bins."""
    nb = max(3, int(math.ceil(per * 2)))  # the circle has exactly ceil(2*per) half-frame bins
    bins = np.minimum((np.mod(np.arange(len(e)), per) * 2).astype(int), nb - 1)
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


def _stable(env, bands, hop_s: float, np, bpm: float | None = None):
    """``(change_note, octave_flip, window_ambiguous)``. ``change_note`` is non-empty when the
    window tempi disagree even with octaves folded. ``octave_flip`` is True when windows read
    different octaves, at least one of them was settled and the two halves really differ in how
    much the off-beats are played (a 2:1 change between sections, not an evenly balanced groove
    whose per-window phase lock wobbles); ``window_ambiguous`` when any window could not settle
    its octave."""
    n = len(env)
    if n * hop_s < 2 * STABILITY_MIN_S:
        return "", False, False
    raw, res, mix = [], [], []
    pair = _octave_pair(bpm) if bpm else None
    grid = _octave_grid(env, hop_s, pair[0], np) if pair else None
    for a, b in ((0.0, 0.5), (0.5, 1.0), (0.25, 0.75)):
        i0, i1 = int(a * n), int(b * n)
        seg = env[i0:i1]
        got = _comb_tempo(seg, hop_s, np)
        if got is not None and got[1] >= MIN_STRENGTH:
            raw.append(got[0])
            # every window is judged on the clip's own beat grid (shifted to the window start)
            wg = (grid[0], (grid[1] - i0) % grid[0]) if grid else None
            st: dict = {}
            res.append(_resolve_octave(seg, bands[:, i0:i1], hop_s, got[0] if wg is None else bpm, np, wg, st))
            if (a, b) in ((0.0, 0.5), (0.5, 1.0)):
                mix.append(st.get("mean_r"))
    if len(raw) < 2:
        return "", False, False
    lo, hi = min(raw), max(raw)
    x = math.log2(hi / lo)
    if abs(x - round(x)) > math.log2(1 + STABILITY_TOL):
        return f"tempo changes within the clip ({lo:.0f} to {hi:.0f} BPM): no single tempo", False, False
    tol = math.log2(1 + STABILITY_TOL)
    # two windows that both admit ambiguity may pick different primaries without a real change
    flip = any(abs(math.log2(bi / bj)) > tol and (ai is None or aj is None)
               for i, (bi, ai) in enumerate(res) for bj, aj in res[i + 1:])
    if flip and len(mix) == 2 and mix[0] and mix[1]:
        flip = any(abs(r0 - r1) >= MIX_R_DELTA and max(r0, r1) >= MIX_R_CHANGE * max(min(r0, r1), 1e-6)
                   for r0, r1 in zip(mix[0], mix[1]))
    return "", flip, any(alt is not None for _, alt in res)


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
        got = _onset_bands(x, np)
        if got is None:
            return _nd(dur, ["audio too short for tempo"], loud)
        env, t0, hop_s, bands = got
        env, tempo = _pulse(env, bands, hop_s, np)
        if tempo is None or tempo[1] < MIN_STRENGTH:
            strength = 0.0 if tempo is None else tempo[1]
            return _nd(dur, [f"no steady pulse found (periodicity {strength:.2f} < {MIN_STRENGTH})"], loud)
        change, octave_flip, window_amb = _stable(env, bands, hop_s, np, tempo[0])
        if change:
            return _nd(dur, [change], loud)
        bpm0, alt = _resolve_octave(env, bands, hop_s, tempo[0], np)
        if alt is None and (window_amb or octave_flip):
            pair = _octave_pair(bpm0)  # a section could not settle the octave: neither can the clip
            if pair is not None:
                alt = pair[0] if bpm0 == pair[1] else pair[1]
        grid = _beat_grid(env, t0, hop_s, bpm0, np)
        if grid is None:
            return _nd(dur, ["tempo candidate not supported by repeated onsets"], loud)
        bpm, beats = grid
        if not (BPM_MIN - 2 <= bpm <= BPM_MAX + 2):
            return _nd(dur, ["refined tempo outside the 60-200 BPM range"], loud)
        conf = min(0.95, tempo[1])
        notes: list[str] = []
        extra: dict = {}
        if octave_flip:
            conf = min(conf, AMBIG_CONF)
            notes.append("octave changes between sections")
        if alt:
            conf = min(conf, AMBIG_CONF)
            notes.append("half/double-time ambiguous")
            extra["tempo_alt"] = round(alt * bpm / bpm0, 1)
            extra["tempo_alt_ratio"] = round(extra["tempo_alt"] / bpm, 3)
        return {"tempo_bpm": round(bpm, 1), "beats": beats, "loudness": loud, **extra,
                "evidence": _ev(beats[0], beats[-1], conf, periodicity=round(tempo[1], 3)),
                "loudness_evidence": _loud_ev(loud, dur),
                "notes": notes}
    except Exception as exc:  # noqa: BLE001 - a command must never raise out of an analysis
        return _nd(duration, [f"audio analysis failed: {type(exc).__name__}"])


def _alt_grids(beats: list[float], ratio: float) -> list[list[float]]:
    """Beat grids for the other octave: double time adds the midpoints, half time keeps
    every other beat (either parity, since the phase of the slower pulse is unknown)."""
    if ratio > 1.0:
        mids = [(a + b) / 2 for a, b in zip(beats, beats[1:])]
        return [sorted(beats + mids)]
    return [beats[0::2], beats[1::2]]


def cuts_on_beat(shots: list[dict], beats: list[float], fps: float, tempo_conf: float = 1.0,
                 alt_ratio: float | None = None):
    """Share of hard cuts within +-2 source frames of a beat, or "not determined".

    The confidence never exceeds ``tempo_conf`` (the evidence for the beat grid itself).
    When the tempo octave is ambiguous (``alt_ratio`` = alternative tempo / primary tempo)
    the share on the other octave's grid is reported too as ``share_alt`` (the lower of the
    two phases for half time): one confident share would be wrong for one of the readings."""
    if not beats or not fps or fps <= 0:
        return NOT_DETERMINED
    cuts = [s["boundary"]["t"] for s in shots or []
            if s.get("boundary") and s["boundary"].get("kind") == "cut"]
    if not cuts:
        return NOT_DETERMINED
    tol = TOLERANCE_FRAMES / fps + 1e-6

    def share(grid):
        return sum(1 for c in cuts if min(abs(c - b) for b in grid) <= tol) / len(cuts)

    hits = sum(1 for c in cuts if min(abs(c - b) for b in beats) <= tol)
    n = len(cuts)
    out = {"share": round(hits / n, 3), "tolerance_frames": TOLERANCE_FRAMES}
    # confidence of the share itself: 0.9 at most, scaled up linearly to 4 cuts (fewer cuts say little
    # about how the edit relates to the music), never above the evidence for the beat grid
    conf = min(0.9 * min(1.0, n / 4), float(tempo_conf))
    if alt_ratio:
        grids = [g for g in _alt_grids(beats, alt_ratio) if g]
        if grids:
            out["share_alt"] = round(min(share(g) for g in grids), 3)
            out["alt_ratio"] = alt_ratio
    out["evidence"] = _ev(min(cuts), max(cuts), conf, n_cuts=n, n_on_beat=hits)
    return out


MUSIC_WINDOW_S = 6.0
SILENT_PEAK = 1e-3


def music_windows(path: str, duration: float = 0.0, window: float = MUSIC_WINDOW_S):
    """Per-window music presence from onset periodicity, or None when unavailable.

    ``[{"t_start", "t_end", "state", "strength"}]`` with ``state`` ``"music"`` (a steady
    pulse, strength >= MIN_STRENGTH), ``"silent"`` (no signal) or ``"not determined"``
    (signal without a steady pulse: speech, ambience or non-rhythmic music). A trailing
    window shorter than MIN_SECONDS is folded into the previous one."""
    try:
        np = _np()
        x, _note = _decode(path)
        if x is None:
            return None
        dur = float(duration) if duration else len(x) / SR
        got = _onset_bands(x, np)
        if got is None:
            return None
        env, t0, hop_s, bands = got
        edges = [0.0]
        while edges[-1] + window < dur - MIN_SECONDS + 1e-9:
            edges.append(edges[-1] + window)
        edges.append(dur)
        out = []
        for a, b in zip(edges, edges[1:]):
            seg = x[int(a * SR):int(b * SR)]
            if len(seg) == 0:
                continue
            if float(np.max(np.abs(seg))) < SILENT_PEAK:
                out.append({"t_start": round(a, 3), "t_end": round(b, 3), "state": "silent", "strength": 0.0})
                continue
            i0, i1 = max(0, int((a - t0) / hop_s)), max(0, int((b - t0) / hop_s))
            e = env[i0:i1]
            tempo = _pulse(e, bands[:, i0:i1], hop_s, np)[1] if len(e) > 8 else None
            strength = 0.0 if tempo is None else float(tempo[1])
            out.append({"t_start": round(a, 3), "t_end": round(b, 3),
                        "state": "music" if strength >= MIN_STRENGTH else NOT_DETERMINED,
                        "strength": round(strength, 3)})
        return out or None
    except Exception:  # noqa: BLE001 - never raise out of an analysis
        return None
