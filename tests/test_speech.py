"""Tests for speech-window computation and two-pass frame budgeting."""
import speech


def _segs(*pairs):
    return [{"start": a, "end": b, "text": "x"} for a, b in pairs]


def test_windows_merge_small_gaps():
    # 1.0s gap < 2.0 threshold -> single window
    wins = speech.compute_speech_windows(_segs((0, 5), (6, 10)))
    assert wins == [(0.0, 10.0)]


def test_windows_split_on_large_gaps():
    wins = speech.compute_speech_windows(_segs((0, 5), (10, 15)))
    assert len(wins) == 2


def test_windows_respect_range():
    wins = speech.compute_speech_windows(_segs((0, 5), (10, 15)), range_start=8.0, range_end=20.0)
    assert all(s >= 8.0 for s, _ in wins)


def test_two_pass_sample_returns_timestamps_within_range():
    # Real return shape (pinned from scripts/speech.py two_pass_sample docstring):
    # {"timestamps": [...], "speech_count": int, "non_speech_count": int,
    #  "speech_total_seconds": float, "non_speech_total_seconds": float,
    #  "gaps": [(s, e), ...]}
    wins = [(10.0, 20.0)]
    plan = speech.two_pass_sample(0.0, 30.0, wins, scenes=[], max_frames=10)
    assert set(plan.keys()) == {
        "timestamps",
        "speech_count",
        "non_speech_count",
        "speech_total_seconds",
        "non_speech_total_seconds",
        "gaps",
    }
    ts = plan["timestamps"]
    assert ts, "must produce timestamps"
    assert all(0.0 <= t <= 30.0 for t in ts)
    assert len(ts) <= 10
    assert plan["speech_count"] + plan["non_speech_count"] == len(ts)


def test_two_pass_majority_of_budget_in_speech():
    wins = [(10.0, 20.0)]
    plan = speech.two_pass_sample(0.0, 100.0, wins, scenes=[], max_frames=10, speech_share=0.7)
    ts = plan["timestamps"]
    in_speech = [t for t in ts if 10.0 <= t <= 20.0]
    assert len(in_speech) >= len(ts) * 0.5
    # speech_count reflects the same speech-window membership as ts filtering above.
    assert plan["speech_count"] == len(in_speech)
