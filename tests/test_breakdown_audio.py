import json
import statistics
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from breakdown_fixtures import make_reference  # noqa: E402
from breakdown import audio as A  # noqa: E402

SR = 22050


def _wav(path, x):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return str(path)


def _clicks(path, bpm, seconds=10.0, offset=0.0):
    x = np.zeros(int(seconds * SR))
    period = 60.0 / bpm
    t = offset
    burst = 0.8 * np.sin(2 * np.pi * 1000 * np.arange(int(0.02 * SR)) / SR)
    while t < seconds - 0.05:
        i = int(t * SR)
        x[i:i + len(burst)] += burst
        t += period
    return _wav(path, x), period


@pytest.fixture(scope="module")
def ref(tmp_path_factory):
    d = tmp_path_factory.mktemp("audio")
    p = d / "ref.mp4"
    truth = make_reference(p, with_audio=True)
    silent = d / "silent.mp4"
    make_reference(silent, with_audio=False)
    return truth, str(p), str(silent)


def test_tempo_120(ref):
    truth, p, _ = ref
    res = A.analyze(p, truth["duration"])
    assert abs(res["tempo_bpm"] - 120) <= 2
    json.dumps(res)
    ev = res["evidence"]
    assert ev["method"] == "measured" and ev["confidence"] > 0.5


def test_beats_regular(ref):
    truth, p, _ = ref
    beats = A.analyze(p, truth["duration"])["beats"]
    gaps = [b - a for a, b in zip(beats, beats[1:])]
    assert len(beats) >= 10
    assert abs(statistics.median(gaps) - 0.5) <= 0.02
    # beats land on the true click onsets (0.0, 0.5, ...)
    assert max(min(abs(b - t) for t in truth["beats"]) for b in beats) <= 0.05


def test_loudness_curve(ref):
    truth, p, _ = ref
    loud = A.analyze(p, truth["duration"])["loudness"]
    assert len(loud) == 14 and loud[0]["t"] == 0.0
    assert all(-70 <= w["lufs_approx"] < 0 for w in loud)


def test_cut_on_beat_share(ref):
    from breakdown import shots as S
    truth, p, silent = ref
    beats = A.analyze(p, truth["duration"])["beats"]
    shots = S.detect_shots(S.frame_series(silent, truth["fps"]), truth["duration"])
    res = A.cuts_on_beat(shots, beats, truth["fps"])
    assert res["tolerance_frames"] == 2
    assert res["share"] == 1.0
    assert res["evidence"]["method"] == "measured"


def test_cut_off_beat_not_counted():
    shots = [{"boundary": {"kind": "cut", "t": 2.25}}, {"boundary": {"kind": "gradual", "t": 4.0}},
             {"boundary": {"kind": "cut", "t": 3.0}}]
    res = A.cuts_on_beat(shots, [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], 25)
    assert res["share"] == 0.5  # gradual boundaries are not cuts; 2.25 is 0.25 s from a beat


def test_no_audio(ref):
    truth, _, silent = ref
    res = A.analyze(silent, truth["duration"])
    assert res["tempo_bpm"] == "not determined"
    assert res["beats"] == [] and res["evidence"]["confidence"] == 0
    assert any("audio" in n for n in res["notes"])
    assert A.cuts_on_beat([{"boundary": {"kind": "cut", "t": 2.0}}], res["beats"], 25) == "not determined"


def test_cuts_on_beat_guards():
    shots = [{"boundary": {"kind": "cut", "t": 1.0}}]
    assert A.cuts_on_beat(shots, [1.0], 0) == "not determined"
    assert A.cuts_on_beat([{"boundary": None}], [1.0], 25) == "not determined"
    assert A.cuts_on_beat([], [1.0], 25) == "not determined"


@pytest.mark.parametrize("bpm", [90, 140, 75, 160])
def test_other_tempi_no_octave_error(tmp_path, bpm):
    p, period = _clicks(tmp_path / f"c{bpm}.wav", bpm)
    res = A.analyze(p, 10.0)
    assert abs(res["tempo_bpm"] - bpm) <= 2, res["tempo_bpm"]
    gaps = [b - a for a, b in zip(res["beats"], res["beats"][1:])]
    assert abs(statistics.median(gaps) - period) <= 0.02


def test_offset_clicks_beats_follow_onsets(tmp_path):
    p, _ = _clicks(tmp_path / "off.wav", 100, offset=0.23)
    res = A.analyze(p, 10.0)
    assert abs(res["tempo_bpm"] - 100) <= 2
    assert abs(res["beats"][0] - 0.23) <= 0.05


def _noise_cases():
    rng = np.random.default_rng(7)
    n = 10 * SR
    white = rng.standard_normal(n)
    brown = np.cumsum(white)
    brown = brown / np.abs(brown).max()
    spec = np.fft.rfft(white)
    spec /= np.sqrt(np.maximum(np.arange(len(spec)), 1))
    pink = np.fft.irfft(spec, n)
    pink = pink / np.abs(pink).max()
    # speech-like: noise with a slow irregular amplitude envelope (syllable-ish, no pulse)
    env = np.convolve(np.abs(rng.standard_normal(n)), np.ones(SR // 8) / (SR // 8), "same")
    speech = white * env
    speech = speech / np.abs(speech).max()
    return {"white": 0.3 * white / np.abs(white).max(), "pink": 0.5 * pink, "brown": 0.5 * brown,
            "speechlike": 0.6 * speech}


@pytest.mark.parametrize("kind", ["white", "pink", "brown", "speechlike"])
def test_non_rhythmic_audio_is_not_a_confident_bpm(tmp_path, kind):
    p = _wav(tmp_path / f"{kind}.wav", _noise_cases()[kind])
    res = A.analyze(p, 10.0)
    assert res["tempo_bpm"] == "not determined", (kind, res["tempo_bpm"], res["evidence"])
    assert res["beats"] == [] and res["evidence"]["confidence"] == 0
    assert res["loudness"]


def test_without_numpy(monkeypatch, ref):
    truth, p, _ = ref
    def boom():
        raise ImportError("numpy")
    monkeypatch.setattr(A, "_np", boom)
    res = A.analyze(p, truth["duration"])
    assert res["tempo_bpm"] == "not determined"
    assert any("numpy" in n for n in res["notes"])


# ---- backbeat grooves, tempo change, memory, loudness evidence (fix round 1) ----
def _groove(bpm, secs, hats=True, snares=True, change=None, seed=1):
    rng = np.random.default_rng(seed)

    def place(x, s, t):
        i = int(t * SR)
        j = min(len(x), i + len(s))
        if j > i:
            x[i:j] += s[:j - i]

    tk = np.arange(int(0.15 * SR)) / SR
    kick = np.sin(2 * np.pi * (60 + 90 * np.exp(-tk * 30)) * tk) * np.exp(-tk * 20)
    nh, ns = int(0.04 * SR), int(0.12 * SR)
    x = rng.standard_normal(int(secs * SR)) * 0.05
    t, b = 0.0, 0
    while t < secs - 0.2:
        cur = bpm if change is None or t < secs / 2 else change
        p = 60 / cur
        place(x, kick, t)
        if snares and b % 2 == 1:
            place(x, rng.standard_normal(ns) * np.exp(-np.arange(ns) / SR * 25) * 0.5, t)
        if hats:
            place(x, rng.standard_normal(nh) * np.exp(-np.arange(nh) / SR * 80) * 0.3, t)
            place(x, rng.standard_normal(nh) * np.exp(-np.arange(nh) / SR * 80) * 0.2, t + p / 2)
        t += p
        b += 1
    return x / np.abs(x).max() * 0.8


@pytest.mark.parametrize("bpm,secs", [(120, 20), (120, 60), (120, 180), (90, 60), (90, 20), (100, 60)])
def test_backbeat_no_half_time_error(tmp_path, bpm, secs):
    p = _wav(tmp_path / "g.wav", _groove(bpm, secs))
    res = A.analyze(p, secs)
    assert abs(res["tempo_bpm"] - bpm) <= 2, res["tempo_bpm"]
    gaps = [b - a for a, b in zip(res["beats"], res["beats"][1:])]
    assert abs(statistics.median(gaps) - 60 / bpm) <= 0.02
    assert len(res["beats"]) >= 0.9 * secs * bpm / 60


def test_weak_beats_not_dropped(tmp_path):
    # kick every beat, snare on 2 and 4, no hats: beats stay consistent with the tempo
    p = _wav(tmp_path / "g90.wav", _groove(90, 20, hats=False))
    res = A.analyze(p, 20)
    assert abs(res["tempo_bpm"] - 90) <= 2
    gaps = [b - a for a, b in zip(res["beats"], res["beats"][1:])]
    assert abs(statistics.median(gaps) - 60 / 90) <= 0.02
    assert len(res["beats"]) >= 26


def test_tempo_change_is_not_a_confident_single_bpm(tmp_path):
    p = _wav(tmp_path / "chg.wav", _groove(100, 24, change=140))
    res = A.analyze(p, 24)
    assert res["tempo_bpm"] == "not determined"
    assert res["beats"] == [] and any("tempo changes" in n for n in res["notes"])
    assert res["loudness_evidence"]["confidence"] > 0  # loudness is independent of tempo


def test_loudness_has_its_own_evidence(ref):
    truth, p, silent = ref
    res = A.analyze(p, truth["duration"])
    ev = res["loudness_evidence"]
    assert ev["method"] == "measured" and ev["confidence"] > 0 and ev["t_end"] >= 6.5
    assert A.analyze(silent, truth["duration"])["loudness_evidence"]["confidence"] == 0


def test_onset_envelope_memory_bounded():
    import tracemalloc
    x = (np.random.default_rng(3).standard_normal(10 * 60 * A.SR) * 0.1).astype("float32")
    tracemalloc.start()
    env, _, _ = A._onset_envelope(x, np)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert len(env) > 100000
    assert peak < 150e6, peak  # full-STFT version needed ~800 MB for 10 min


# ---- fix round 2: octave-folded stability, ambiguity, long-track drift ----
def _kit(seed=1):
    rng = np.random.default_rng(seed)
    tk = np.arange(int(0.15 * SR)) / SR
    kick = np.sin(2 * np.pi * (60 + 90 * np.exp(-tk * 30)) * tk) * np.exp(-tk * 20)

    def noise(n, decay, amp):
        return rng.standard_normal(n) * np.exp(-np.arange(n) / SR * decay) * amp
    return rng, kick, (lambda: noise(int(0.04 * SR), 80, 0.3)), (lambda: noise(int(0.12 * SR), 25, 0.5))


def _put(x, s, t):
    i = int(t * SR)
    j = min(len(x), i + len(s))
    if j > i:
        x[i:j] += s[:j - i]


def _slow(bpm, secs):
    rng, kick, hat, snare = _kit()
    x = rng.standard_normal(int(secs * SR)) * 0.03
    p, t, b = 60 / bpm, 0.0, 0
    while t < secs - 0.3:
        _put(x, kick, t)
        if b % 2:
            _put(x, snare(), t)
        _put(x, hat(), t)
        _put(x, hat() * 0.7, t + p / 2)
        t += p
        b += 1
    return x / np.abs(x).max() * 0.8


def _dnb(bpm, secs):
    rng, kick, hat, snare = _kit()
    x = rng.standard_normal(int(secs * SR)) * 0.03
    p = 60 / bpm
    t = 0.0
    while t < secs - 4 * p:
        for off in (0, 2.5 * p):
            _put(x, kick, t + off)
        for off in (p, 3 * p):
            _put(x, snare() * 1.4, t + off)
        for k in range(8):
            _put(x, hat() * 0.6, t + k * p / 2)
        t += 4 * p
    return x / np.abs(x).max() * 0.8


@pytest.mark.parametrize("bpm", [70, 65])
def test_slow_backbeat_with_hats_is_stable(tmp_path, bpm):
    res = A.analyze(_wav(tmp_path / "s.wav", _slow(bpm, 60)), 60)
    assert res["tempo_bpm"] != "not determined", res["notes"]
    assert abs(res["tempo_bpm"] - bpm) <= 2 or abs(res["tempo_bpm"] - 2 * bpm) <= 4
    assert not any("tempo changes" in n for n in res["notes"])


def test_dnb_is_ambiguous_or_correct(tmp_path):
    res = A.analyze(_wav(tmp_path / "d.wav", _dnb(172, 60)), 60)
    t = res["tempo_bpm"]
    if abs(t - 172) <= 2:
        return
    assert abs(t - 86) <= 2
    assert abs(res["tempo_alt"] - 172) <= 4
    assert res["evidence"]["confidence"] <= 0.5
    assert "half/double-time ambiguous" in res["notes"]


def test_four_on_the_floor_170_stays_confident(tmp_path):
    res = A.analyze(_wav(tmp_path / "f.wav", _groove(170, 60, hats=False, snares=False)), 60)
    assert abs(res["tempo_bpm"] - 170) <= 2 and res["evidence"]["confidence"] > 0.3
    assert "tempo_alt" not in res and "half/double-time ambiguous" not in res["notes"]


def test_plain_120_backbeat_stays_confident(tmp_path):
    res = A.analyze(_wav(tmp_path / "b.wav", _groove(120, 60)), 60)
    assert abs(res["tempo_bpm"] - 120) <= 2 and res["evidence"]["confidence"] > 0.5
    assert "tempo_alt" not in res


def _never_confident_wrong(res, truth):
    """Spec section 3: a steady groove gets a tempo, and a wrong octave is never confident.
    Either the primary is right (any ``tempo_alt`` then names the other octave and the
    confidence is <= 0.5), or the primary is the other octave, ``tempo_alt`` is the truth,
    confidence <= 0.5 and the note says so."""
    t = res["tempo_bpm"]
    assert t != "not determined", res["notes"]
    conf = res["evidence"]["confidence"]
    if abs(t - truth) <= 2:
        if "tempo_alt" in res:
            assert min(abs(res["tempo_alt"] - 2 * t), abs(res["tempo_alt"] - t / 2)) <= 3, res
            assert conf <= 0.5 and "half/double-time ambiguous" in res["notes"], res
        return
    assert min(abs(t - 2 * truth), abs(t - truth / 2)) <= 3, res  # only octave errors are possible
    assert abs(res.get("tempo_alt", 0) - truth) <= 3, res
    assert conf <= 0.5, res
    assert "half/double-time ambiguous" in res["notes"], res


def _pattern(bpm, secs, kicks, snares, hat_amp, seed=1):
    rng, kick, hat, snare = _kit(seed)
    x = rng.standard_normal(int(secs * SR)) * 0.03
    p = 60 / bpm
    t = 0.0
    while t < secs - 4 * p:
        for o in kicks:
            _put(x, kick, t + o * p)
        for o in snares:
            _put(x, snare(), t + o * p)
        for k in range(8):
            _put(x, hat() * hat_amp / 0.3 * 0.3, t + k * p / 2)
        t += 4 * p
    return x / np.abs(x).max() * 0.8


# fix round 4: the probe set the octave rule is judged on (spec section 3)
OCTAVE_PROBES = [
    ("hiphop75_hats0.9", 75, [0, 2.5], [1, 3], 0.9),
    ("hiphop75_hats0.6", 75, [0, 2.5], [1, 3], 0.6),
    ("hiphop75_hats1.0", 75, [0, 2.5], [1, 3], 1.0),
    ("hiphop75_hats0.9_seed2", 75, [0, 2.5], [1, 3], 0.9, 2),
    ("hiphop75_hats0.6_seed3", 75, [0, 2.5], [1, 3], 0.6, 3),
    ("kick_every_beat140_no_hats", 140, [0, 1, 2, 3], [1, 3], 0.0),
    ("dubstep140", 140, [0], [2], 0.5),
    ("dubstep150", 150, [0], [2], 0.5),
    ("loud_hats_90", 90, [0, 1, 2, 3], [1, 3], 1.2),
]


@pytest.mark.parametrize("case", OCTAVE_PROBES, ids=[c[0] for c in OCTAVE_PROBES])
def test_octave_never_confidently_wrong(tmp_path, case):
    _, bpm, kicks, snares, hats, *seed = case
    res = A.analyze(_wav(tmp_path / "o.wav", _pattern(bpm, 60, kicks, snares, hats, *seed)), 60)
    _never_confident_wrong(res, bpm)


def test_kick_every_beat_140_is_not_confidently_70(tmp_path):
    # the reviewer's case: kick on every beat + snare on 2 and 4, no hats
    res = A.analyze(_wav(tmp_path / "k.wav", _groove(140, 60, hats=False)), 60)
    _never_confident_wrong(res, 140)


def test_dnb_172_never_confidently_86(tmp_path):
    _never_confident_wrong(A.analyze(_wav(tmp_path / "d.wav", _dnb(172, 60)), 60), 172)


def test_120_backbeat_60s_correct_primary(tmp_path):
    res = A.analyze(_wav(tmp_path / "b.wav", _groove(120, 60)), 60)
    _never_confident_wrong(res, 120)
    assert abs(res["tempo_bpm"] - 120) <= 2  # the right octave is the primary reading


def test_170_backbeat_with_hats_is_correct_or_ambiguous(tmp_path):
    res = A.analyze(_wav(tmp_path / "h.wav", _groove(170, 60)), 60)
    _never_confident_wrong(res, 170)


@pytest.mark.parametrize("hats", [False, True], ids=["no_hats", "hats"])
def test_octave_change_between_sections_is_flagged(tmp_path, hats):
    x = np.concatenate([_groove(90, 24, hats=hats, seed=1), _groove(180, 24, hats=hats, snares=hats, seed=2)])
    res = A.analyze(_wav(tmp_path / "oc.wav", x), 48)
    # unconditional: either no tempo at all, or a flagged low-confidence one; never a confident 90 or 180
    if res["tempo_bpm"] == "not determined":
        assert res["beats"] == [] and res["evidence"]["confidence"] == 0
        return
    assert res["evidence"]["confidence"] <= 0.5, res
    assert "octave changes between sections" in res["notes"], res["notes"]


@pytest.mark.parametrize("bpm,secs", [(150, 120), (128, 300)])
def test_beats_do_not_drift_on_long_tracks(tmp_path, bpm, secs):
    res = A.analyze(_wav(tmp_path / "l.wav", _groove(bpm, secs)), secs)
    assert abs(res["tempo_bpm"] - bpm) <= 0.3
    per = 60 / bpm
    err = np.abs(np.array(res["beats"]) - np.round(np.array(res["beats"]) / per) * per)
    assert err.max() < 0.03, err.max()
    assert len(res["beats"]) >= 0.97 * secs / per


# ---- 0.7.1 batch 2: decode failure wording, fractional fold, low-band pulse, octave settling ----
def test_ffmpeg_failure_is_not_reported_as_no_audio_stream(tmp_path):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"this is not a media file")
    res = A.analyze(str(junk), 5)
    assert res["tempo_bpm"] == "not determined"
    assert any("audio decode failed" in n and "ffmpeg exit" in n for n in res["notes"]), res["notes"]
    assert not any("no audio stream" in n for n in res["notes"])


def test_real_missing_audio_stream_still_says_so(ref):
    truth, _, silent = ref
    assert any("no audio stream" in n for n in A.analyze(silent, truth["duration"])["notes"])


def test_fold_score_wraps_at_the_fractional_period():
    # energy in the last phase bin and in bin 0 are neighbours on the circle of a 10.3-frame period
    per = 10.3
    idx = np.arange(400)
    ph = np.mod(idx, per)
    e = ((ph < 0.5) | (ph >= per - 0.5)).astype(float)
    score, _ = A._fold_score(e, per, np)
    assert score >= e.sum() - 1e-9, (score, e.sum())


def test_kick_only_140_is_determined(tmp_path):
    # low-frequency-only pattern: the kick is buried in the noise bed of the full-band flux
    res = A.analyze(_wav(tmp_path / "k140.wav", _groove(140, 60, hats=False, snares=False)), 60)
    assert res["tempo_bpm"] != "not determined", res["notes"]
    _never_confident_wrong(res, 140)
    assert len(res["beats"]) >= 100


def test_low_band_fallback_does_not_make_speech_or_noise_a_tempo(tmp_path):
    rng = np.random.default_rng(5)
    noise = _wav(tmp_path / "n.wav", rng.standard_normal(30 * SR) * 0.2)
    assert A.analyze(noise, 30)["tempo_bpm"] == "not determined"


def _kick8ths(bpm, secs, hats16=0.0):
    rng, kick, hat, snare = _kit()
    x = rng.standard_normal(int(secs * SR)) * 0.03
    p = 60 / bpm
    t = 0.0
    while t < secs - 4 * p:
        for o in (0, 0.5, 2, 2.5):
            _put(x, kick, t + o * p)
        for o in (1, 3):
            _put(x, snare(), t + o * p)
        if hats16:
            for k in range(16):
                _put(x, hat() * hats16, t + k * p / 4)
        t += 4 * p
    return x / np.abs(x).max() * 0.8


@pytest.mark.parametrize("hats16", [0.0, 0.5], ids=["no_hats", "hats16"])
def test_steady_70_kicks_on_8ths_has_no_octave_change_note(tmp_path, hats16):
    res = A.analyze(_wav(tmp_path / "k8.wav", _kick8ths(70, 60, hats16)), 60)
    assert not any("octave changes" in n for n in res["notes"]), res["notes"]
    _never_confident_wrong(res, 70)


@pytest.mark.parametrize("bpm", [80, 90, 100])
def test_kick_on_every_beat_with_8th_hats_stays_ambiguous(tmp_path, bpm):
    # a slow backbeat and a half-time groove are the same evidence: never a confident slower octave
    res = A.analyze(_wav(tmp_path / "g.wav", _groove(bpm, 60)), 60)
    _never_confident_wrong(res, bpm)
    assert res["evidence"]["confidence"] <= 0.5 and "tempo_alt" in res, res


def _halftime(bpm, secs, kicks=(0,), snares=(2,), hat_amp=0.0, accent=False, k808=False):
    rng, kick, hat, snare = _kit()
    if k808:
        tk = np.arange(int(0.3 * SR)) / SR
        kick = np.sin(2 * np.pi * (45 + 60 * np.exp(-tk * 10)) * tk) * np.exp(-tk * 6)
    x = rng.standard_normal(int(secs * SR)) * 0.03
    p = 60 / bpm
    t = 0.0
    while t < secs - 4 * p:
        for o in kicks:
            _put(x, kick, t + o * p)
        for o in snares:
            _put(x, snare(), t + o * p)
        if hat_amp:
            for o in range(4):
                _put(x, hat() * hat_amp * (2.0 if accent and o in (0, 2) else 1.0) / 0.3 * 0.3, t + o * p)
        t += 4 * p
    return x / np.abs(x).max() * 0.8


HALFTIME = [(b, h, acc, k8) for b in (130, 140, 150, 160, 170) for h in (0.0, 0.12, 0.3) for acc in (False, True)
            for k8 in (False, True) if not (h == 0.0 and acc)]


@pytest.mark.parametrize("bpm,hat,accent,k808", HALFTIME)
def test_half_time_groove_is_never_confidently_the_slower_octave(tmp_path, bpm, hat, accent, k808):
    res = A.analyze(_wav(tmp_path / "h.wav", _halftime(bpm, 60, hat_amp=hat, accent=accent, k808=k808)), 60)
    if res["tempo_bpm"] != "not determined":
        _never_confident_wrong(res, bpm)


# ---- fix round 2: a 3:2 misread must never be confident ----
def _ht_hats(bpm, secs, hat_s, kicks=(0,)):
    rng, kick, hat, snare = _kit()
    x = rng.standard_normal(int(secs * SR)) * 0.03
    p = 60 / bpm
    t = 0.0
    while t < secs - 4 * p:
        for o in kicks:
            _put(x, kick, t + o * p)
        _put(x, snare(), t + 2 * p)
        for i in range(8):
            o = i / 2
            _put(x, hat() * ((0.3 if o in (0.0, 2.0) else hat_s) / 0.3), t + o * p)
        t += 4 * p
    return x / np.abs(x).max() * 0.8


@pytest.mark.parametrize("bpm,hat_s,kicks", [(170, 0.22, (0,)), (170, 0.26, (0,)), (170, 0.22, (0, 2)),
                                              (160, 0.22, (0,)), (160, 0.26, (0,)), (160, 0.26, (0, 2))])
def test_three_to_two_misread_is_never_confident(tmp_path, bpm, hat_s, kicks):
    res = A.analyze(_wav(tmp_path / "t.wav", _ht_hats(bpm, 60, hat_s, kicks)), 60)
    if res["tempo_bpm"] == "not determined":
        return
    t = res["tempo_bpm"]
    right = min(abs(t - r) for r in (bpm, bpm / 2)) <= 3
    if not right:
        assert res["evidence"]["confidence"] <= 0.5, res
        assert abs(res.get("tempo_alt", 0) - bpm) <= 4 or abs(res.get("tempo_alt", 0) - bpm / 2) <= 3, res
        assert any("3:2" in n or "half/double" in n for n in res["notes"]), res
