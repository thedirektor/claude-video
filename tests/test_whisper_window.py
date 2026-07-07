"""Windowed transcription: extract only [start,end], then shift to source time."""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import whisper  # noqa: E402


def test_extract_audio_passes_ss_and_t(monkeypatch, tmp_path):
    captured = {}

    class _Result:
        returncode = 0
        stderr = ""

    def fake_run(cmd, *a, **k):
        captured["cmd"] = list(cmd)
        # Materialize a non-empty output so extract_audio's post-checks pass.
        out = Path(cmd[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"MP3")
        return _Result()

    monkeypatch.setattr(whisper.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(whisper.subprocess, "run", fake_run)

    whisper.extract_audio("in.mp4", tmp_path / "a.mp3", start_seconds=30.0, duration_seconds=15.0)
    cmd = captured["cmd"]
    assert "-ss" in cmd and cmd[cmd.index("-ss") + 1] == "30.000"
    assert "-t" in cmd and cmd[cmd.index("-t") + 1] == "15.000"
    # -ss must precede -i for fast input seek → 0-based output timestamps.
    assert cmd.index("-ss") < cmd.index("-i")


def test_shift_all_preserves_speaker(tmp_path):
    segs = [{"start": 1.0, "end": 2.0, "text": "hi", "speaker": "Speaker A"}]
    out = whisper._shift_all(segs, 30.0)
    assert out == [{"start": 31.0, "end": 32.0, "text": "hi", "speaker": "Speaker A"}]


def test_shift_all_noop_on_zero(tmp_path):
    segs = [{"start": 1.0, "end": 2.0, "text": "hi"}]
    assert whisper._shift_all(segs, 0.0) is segs
