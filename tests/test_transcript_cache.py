"""Content-addressed Whisper chunk cache: hit skips the upload, key tracks model."""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import whisper  # noqa: E402


def _fake_audio(tmp_path: Path, data: bytes = b"AUDIODATA") -> Path:
    p = tmp_path / "chunk_000.mp3"
    p.write_bytes(data)
    return p


def test_key_changes_with_model(tmp_path):
    audio = _fake_audio(tmp_path)
    k1 = whisper._chunk_cache_key(audio, "whisper-large-v3")
    k2 = whisper._chunk_cache_key(audio, "whisper-1")
    assert k1 != k2


def test_key_changes_with_bytes(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    ka = whisper._chunk_cache_key(_fake_audio(tmp_path / "a", b"AAA"), "m")
    kb = whisper._chunk_cache_key(_fake_audio(tmp_path / "b", b"BBB"), "m")
    assert ka != kb


def test_cache_miss_then_hit_skips_upload(tmp_path, monkeypatch):
    monkeypatch.setattr(whisper, "TRANSCRIPT_CACHE_DIR", tmp_path / "cache")
    audio = _fake_audio(tmp_path)

    calls = {"n": 0}
    segs = [{"start": 0.0, "end": 1.0, "text": "hello"}]

    def fake_post(endpoint, api_key, model, audio_path):
        calls["n"] += 1
        return {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}]}

    monkeypatch.setattr(whisper, "_post_whisper", fake_post)

    first = whisper._transcribe_file("groq", "key", audio, use_cache=True)
    assert first == segs
    assert calls["n"] == 1  # miss → uploaded

    second = whisper._transcribe_file("groq", "key", audio, use_cache=True)
    assert second == segs
    assert calls["n"] == 1  # hit → no second upload


def test_use_cache_false_always_uploads(tmp_path, monkeypatch):
    monkeypatch.setattr(whisper, "TRANSCRIPT_CACHE_DIR", tmp_path / "cache")
    audio = _fake_audio(tmp_path)
    calls = {"n": 0}

    def fake_post(endpoint, api_key, model, audio_path):
        calls["n"] += 1
        return {"segments": [{"start": 0.0, "end": 1.0, "text": "x"}]}

    monkeypatch.setattr(whisper, "_post_whisper", fake_post)
    whisper._transcribe_file("groq", "key", audio, use_cache=False)
    whisper._transcribe_file("groq", "key", audio, use_cache=False)
    assert calls["n"] == 2
