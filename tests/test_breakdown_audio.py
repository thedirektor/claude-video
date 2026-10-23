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
