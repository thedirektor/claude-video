"""watch.main() prefers TranscriptAPI for YouTube URLs; falls back on miss."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import watch  # noqa: E402


YT = "https://www.youtube.com/watch?v=m0Jk8aIDiJ8"


def _no_frames(monkeypatch):
    # Keep main() cheap + offline: transcript detail, no captions/whisper/frames.
    monkeypatch.setattr(watch, "fetch_captions",
                        lambda *a, **k: {"video_path": None, "subtitle_path": None,
                                         "info": {"title": "T", "duration": 30}, "downloaded": False})


def test_youtube_uses_transcriptapi(monkeypatch, tmp_path, capsys):
    _no_frames(monkeypatch)
    called = {"n": 0}
    def fake_fetch(url, key, language=None):
        called["n"] += 1
        return ([{"start": 0.0, "end": 2.0, "text": "hola"}], "es")
    monkeypatch.setattr(watch.transcriptapi, "fetch_transcript", fake_fetch)
    monkeypatch.setattr(watch.transcriptapi, "load_api_key", lambda: "sk_test")
    # If TranscriptAPI wins, parse_vtt must never run:
    monkeypatch.setattr(watch, "parse_vtt", lambda *a, **k: (_ for _ in ()).throw(AssertionError("captions used")))
    monkeypatch.setattr(sys, "argv", ["watch.py", YT, "--detail", "transcript", "--no-whisper", "--out-dir", str(tmp_path)])
    assert watch.main() == 0
    assert called["n"] == 1
    out = capsys.readouterr().out
    assert "transcriptapi" in out.lower()
    assert "hola" in out


def test_no_transcriptapi_flag_skips_it(monkeypatch, tmp_path):
    _no_frames(monkeypatch)
    def boom(*a, **k):
        raise AssertionError("TranscriptAPI called despite --no-transcriptapi")
    monkeypatch.setattr(watch.transcriptapi, "fetch_transcript", boom)
    monkeypatch.setattr(watch.transcriptapi, "load_api_key", lambda: "sk_test")
    # With TranscriptAPI off and no captions, transcript mode would otherwise hit
    # the real yt-dlp download — stub it so the test stays offline (no network).
    monkeypatch.setattr(watch, "download",
                        lambda *a, **k: {"video_path": None, "subtitle_path": None,
                                         "info": {"title": "T", "duration": 30}, "downloaded": False})
    monkeypatch.setattr(sys, "argv", ["watch.py", YT, "--detail", "transcript",
                                      "--no-whisper", "--no-transcriptapi", "--out-dir", str(tmp_path)])
    assert watch.main() == 0  # no transcript, but no crash


def test_transcriptapi_miss_falls_back(monkeypatch, tmp_path):
    _no_frames(monkeypatch)
    monkeypatch.setattr(watch.transcriptapi, "load_api_key", lambda: "sk_test")
    monkeypatch.setattr(watch.transcriptapi, "fetch_transcript", lambda *a, **k: None)  # miss
    hit = {"n": 0}
    def fake_vtt(*a, **k):
        hit["n"] += 1
        return [{"start": 0.0, "end": 1.0, "text": "caption fallback"}]
    # Give the caption path a subtitle to parse so the fallback is exercised:
    monkeypatch.setattr(watch, "fetch_captions",
                        lambda *a, **k: {"video_path": None, "subtitle_path": "x.vtt",
                                         "info": {"title": "T", "duration": 30}, "downloaded": False})
    monkeypatch.setattr(watch, "parse_vtt", fake_vtt)
    monkeypatch.setattr(sys, "argv", ["watch.py", YT, "--detail", "transcript", "--no-whisper", "--out-dir", str(tmp_path)])
    assert watch.main() == 0
    assert hit["n"] >= 1  # fell back to captions


def test_resume_reuses_transcriptapi_without_recall(monkeypatch, tmp_path):
    _no_frames(monkeypatch)
    calls = {"n": 0}
    def fake_fetch(url, key, language=None):
        calls["n"] += 1
        return ([{"start": 0.0, "end": 2.0, "text": "hola"}], "es")
    monkeypatch.setattr(watch.transcriptapi, "fetch_transcript", fake_fetch)
    monkeypatch.setattr(watch.transcriptapi, "load_api_key", lambda: "sk_test")
    argv = ["watch.py", YT, "--detail", "transcript", "--no-whisper", "--out-dir", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", argv)
    assert watch.main() == 0
    monkeypatch.setattr(sys, "argv", argv)
    assert watch.main() == 0
    assert calls["n"] == 1  # 2nd run served from stage_transcript.json — no re-bill
