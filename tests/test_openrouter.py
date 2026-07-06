"""Offline tests for the OpenRouter transcription fallback chain."""
import json
import re

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
        model_used = None

        # Extract model from multipart request body for OpenRouter calls
        if "openrouter.ai" in url:
            body = req.data or b""
            m = re.search(rb'name="model"\r\n\r\n([^\r]+)', body)
            if m:
                model_used = m.group(1).decode()

        calls.append((url, model_used))

        if "openrouter.ai" in url:
            import urllib.error
            raise urllib.error.HTTPError(url, 500, "boom", {}, None)
        return _FakeResponse({"segments": [{"start": 0.0, "end": 1.0, "text": "hola"}],
                              "text": "hola"})

    monkeypatch.setattr(openrouter, "urlopen", fake_urlopen, raising=False)
    segs = openrouter.transcribe_audio(audio)
    assert segs and segs[0]["text"] == "hola"

    # Extract just URLs and models from calls for assertion
    urls = [url for url, _ in calls]
    models = [model for _, model in calls]

    # Verify OpenRouter was tried (both primary and fallback models)
    assert any("openrouter.ai" in u for u in urls)

    # Verify Groq was used as the final fallback
    assert any("groq.com" in u for u in urls)

    # Verify the exact fallback chain: qwen attempts, then voxtral attempts, then groq
    or_calls = [(url, model) for url, model in calls if "openrouter.ai" in url]
    groq_calls = [(url, model) for url, model in calls if "groq.com" in url]

    # Should have OpenRouter calls followed by Groq call
    assert len(or_calls) > 0, "Should have attempted OpenRouter models"
    assert len(groq_calls) > 0, "Should have fallen back to Groq"

    # Extract distinct consecutive models used on OpenRouter (filters repeated retries)
    or_models_distinct = []
    for _, model in or_calls:
        if not or_models_distinct or or_models_distinct[-1] != model:
            or_models_distinct.append(model)

    # Should have tried qwen then voxtral (in that order, not repeating within a model)
    expected_models = [
        openrouter.DEFAULT_AUDIO_MODEL,
        openrouter._VOXTRAL_FALLBACK_MODEL,
    ]
    assert or_models_distinct == expected_models, (
        f"Model fallback chain should be {expected_models}, got {or_models_distinct}"
    )


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
