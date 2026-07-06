"""Offline tests for the OpenRouter transcription fallback chain."""
import json
import types

import pytest

import openrouter


class _FakeResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode()
    def read(self):
        return self._body
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


def test_transcribe_audio_falls_back_through_chain(monkeypatch, tmp_path):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"\x00" * 128)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setattr(openrouter.time, "sleep", lambda s: None)

    calls = []

    def fake_urlopen(req, *a, **kw):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        calls.append(url)
        if "openrouter.ai" in url:
            import urllib.error
            raise urllib.error.HTTPError(url, 500, "boom", {}, None)
        return _FakeResponse({"segments": [{"start": 0.0, "end": 1.0, "text": "hola"}],
                              "text": "hola"})

    monkeypatch.setattr(openrouter, "urlopen", fake_urlopen, raising=False)
    segs = openrouter.transcribe_audio(audio)
    assert segs and segs[0]["text"] == "hola"
    assert any("openrouter.ai" in u for u in calls)   # tried primary + voxtral
    assert any("groq.com" in u for u in calls)        # landed on groq


def test_transcribe_audio_all_legs_fail_raises(monkeypatch, tmp_path):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"\x00" * 128)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))  # isolate ~/.config/watch/.env
    monkeypatch.setattr(openrouter.time, "sleep", lambda s: None)

    def fake_urlopen(req, *a, **kw):
        import urllib.error
        url = req.full_url if hasattr(req, "full_url") else str(req)
        raise urllib.error.HTTPError(url, 500, "boom", {}, None)

    monkeypatch.setattr(openrouter, "urlopen", fake_urlopen, raising=False)
    with pytest.raises(SystemExit):
        openrouter.transcribe_audio(audio)
