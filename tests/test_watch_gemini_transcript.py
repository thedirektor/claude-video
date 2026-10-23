"""watch.main() falls back to Gemini (YouTube URL) when TranscriptAPI + captions miss."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import watch  # noqa: E402

YT = "https://www.youtube.com/watch?v=m0Jk8aIDiJ8"
SEGS = [{"start": 0.0, "end": 3.0, "text": "gemini hola"}, {"start": 3.0, "end": 8.0, "text": "mundo"}]


def _setup(monkeypatch, subtitle_path=None):
    monkeypatch.delenv("WATCH_GEMINI_TRANSCRIPT", raising=False)
    monkeypatch.setattr(watch, "fetch_captions",
                        lambda *a, **k: {"video_path": None, "subtitle_path": subtitle_path,
                                         "info": {"title": "T", "duration": 30}, "downloaded": False})
    monkeypatch.setattr(watch.transcriptapi, "load_api_key", lambda: None)
    calls = {"gemini": 0, "download": 0, "args": [], "approx": False}

    def fake_gemini(url, *a, **k):
        calls["gemini"] += 1
        calls["args"].append((url, k))
        return (list(SEGS), "gemini-test", calls["approx"])

    def fake_download(*a, **k):
        calls["download"] += 1
        return {"video_path": None, "subtitle_path": None,
                "info": {"title": "T", "duration": 30}, "downloaded": False}

    monkeypatch.setattr(watch.gemini, "transcribe_youtube", fake_gemini)
    monkeypatch.setattr(watch, "download", fake_download)
    return calls


def _argv(monkeypatch, tmp_path, *extra):
    monkeypatch.setattr(sys, "argv", ["watch.py", YT, "--detail", "transcript",
                                      "--no-whisper", "--out-dir", str(tmp_path), *extra])


def test_gemini_used_when_ta_and_captions_empty(monkeypatch, tmp_path, capsys):
    calls = _setup(monkeypatch)
    _argv(monkeypatch, tmp_path)
    assert watch.main() == 0
    out = capsys.readouterr().out
    assert calls["gemini"] == 1
    assert "gemini hola" in out
    assert "gemini (gemini-test)" in out


def test_detail_transcript_skips_download(monkeypatch, tmp_path):
    calls = _setup(monkeypatch)
    _argv(monkeypatch, tmp_path)
    assert watch.main() == 0
    assert calls["download"] == 0


def test_gemini_not_called_when_captions_exist(monkeypatch, tmp_path):
    calls = _setup(monkeypatch, subtitle_path=str(tmp_path / "x.vtt"))
    monkeypatch.setattr(watch, "parse_vtt", lambda *a, **k: [{"start": 0.0, "end": 1.0, "text": "cap"}])
    _argv(monkeypatch, tmp_path)
    assert watch.main() == 0
    assert calls["gemini"] == 0


def test_gemini_not_called_when_transcriptapi_hits(monkeypatch, tmp_path):
    calls = _setup(monkeypatch)
    monkeypatch.setattr(watch.transcriptapi, "load_api_key", lambda: "sk")
    monkeypatch.setattr(watch.transcriptapi, "fetch_transcript",
                        lambda *a, **k: ([{"start": 0.0, "end": 1.0, "text": "ta"}], "es"))
    _argv(monkeypatch, tmp_path)
    assert watch.main() == 0
    assert calls["gemini"] == 0


def test_flag_disables_gemini(monkeypatch, tmp_path):
    calls = _setup(monkeypatch)
    _argv(monkeypatch, tmp_path, "--no-gemini-transcript")
    assert watch.main() == 0
    assert calls["gemini"] == 0


@pytest.mark.parametrize("val", ["0", "false", "no", "FALSE"])
def test_env_disables_gemini(monkeypatch, tmp_path, val):
    calls = _setup(monkeypatch)
    monkeypatch.setenv("WATCH_GEMINI_TRANSCRIPT", val)
    _argv(monkeypatch, tmp_path)
    assert watch.main() == 0
    assert calls["gemini"] == 0


def test_gemini_gets_canonical_url_and_duration(monkeypatch, tmp_path):
    calls = _setup(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["watch.py", "https://youtu.be/m0Jk8aIDiJ8?si=track",
                                      "--detail", "transcript", "--no-whisper",
                                      "--out-dir", str(tmp_path)])
    assert watch.main() == 0
    url, kwargs = calls["args"][0]
    assert url == YT
    assert kwargs.get("duration") == 30


def test_approximate_label_and_header(monkeypatch, tmp_path, capsys):
    calls = _setup(monkeypatch)
    calls["approx"] = True
    _argv(monkeypatch, tmp_path)
    assert watch.main() == 0
    out = capsys.readouterr().out
    assert "gemini (gemini-test, approximate timestamps)" in out
    assert "Timestamps are model-generated and approximate" in out


def test_exact_label_has_no_approx_header(monkeypatch, tmp_path, capsys):
    _setup(monkeypatch)
    _argv(monkeypatch, tmp_path)
    assert watch.main() == 0
    out = capsys.readouterr().out
    assert "approximate" not in out
