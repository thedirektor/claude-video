"""TranscriptAPI backend: URL detection, language mapping, HTTP mapping, fallback."""
from __future__ import annotations

import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import transcriptapi  # noqa: E402


class _Resp:
    def __init__(self, payload, status=200):
        self._b = json.dumps(payload).encode()
        self.status = status
    def read(self):
        return self._b
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


def test_is_youtube_url():
    assert transcriptapi.is_youtube_url("https://www.youtube.com/watch?v=abc12345678")
    assert transcriptapi.is_youtube_url("https://youtu.be/abc12345678")
    assert transcriptapi.is_youtube_url("http://m.youtube.com/watch?v=x")
    assert not transcriptapi.is_youtube_url("https://vimeo.com/12345")
    assert not transcriptapi.is_youtube_url("/local/file.mp4")
    assert not transcriptapi.is_youtube_url("notayoutubeimpostor.com")


def test_language_mapping_from_default_sub_langs():
    # region-stripped, deduped, drops the .*-orig regex, appends asr fallback
    out = transcriptapi.languages_from_sub_langs("es,es-419,es-ES,en,en-US,en-GB,.*-orig")
    assert out == "es,en,asr"


def test_language_mapping_custom():
    assert transcriptapi.languages_from_sub_langs("de,fr") == "de,fr,asr"
    assert transcriptapi.languages_from_sub_langs("all") == "asr"


def test_fetch_transcript_maps_segments(monkeypatch):
    payload = {"video_id": "x", "language": "es",
               "transcript": [{"text": "hola", "start": 0.0, "duration": 2.0},
                              {"text": "  ", "start": 2.0, "duration": 1.0},
                              {"text": "mundo", "start": 2.0, "duration": 1.5}]}
    monkeypatch.setattr(transcriptapi, "urlopen", lambda *a, **k: _Resp(payload))
    result = transcriptapi.fetch_transcript("vid", "sk_key", language="es,en,asr")
    assert result is not None
    segs, lang = result
    assert lang == "es"
    # empty-text segment dropped; end = start + duration
    assert segs == [{"start": 0.0, "end": 2.0, "text": "hola"},
                    {"start": 2.0, "end": 3.5, "text": "mundo"}]


def test_fetch_transcript_404_returns_none(monkeypatch, capsys):
    def raise404(*a, **k):
        raise urllib.error.HTTPError("u", 404, "no", None,
                                     io.BytesIO(b'{"detail":"none"}'))
    monkeypatch.setattr(transcriptapi, "urlopen", raise404)
    assert transcriptapi.fetch_transcript("vid", "sk_key") is None  # → caller falls back


def test_fetch_transcript_402_returns_none_with_warning(monkeypatch, capsys):
    def raise402(*a, **k):
        raise urllib.error.HTTPError("u", 402, "pay", None,
                                     io.BytesIO(b'{"detail":"no credits"}'))
    monkeypatch.setattr(transcriptapi, "urlopen", raise402)
    assert transcriptapi.fetch_transcript("vid", "sk_key") is None
    assert "credit" in capsys.readouterr().err.lower()


def test_fetch_transcript_network_error_returns_none(monkeypatch):
    def boom(*a, **k):
        raise urllib.error.URLError("dns")
    monkeypatch.setattr(transcriptapi, "urlopen", boom)
    monkeypatch.setattr(transcriptapi.time, "sleep", lambda *_a, **_k: None)
    assert transcriptapi.fetch_transcript("vid", "sk_key") is None  # never raises


def test_fetch_transcript_malformed_json_returns_none(monkeypatch):
    class _Bad:
        status = 200
        def read(self): return b"not valid json{{{"
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(transcriptapi, "urlopen", lambda *a, **k: _Bad())
    assert transcriptapi.fetch_transcript("vid", "sk") is None  # never raises


def test_fetch_transcript_wrong_shape_returns_none(monkeypatch):
    monkeypatch.setattr(transcriptapi, "urlopen",
                        lambda *a, **k: _Resp({"transcript": ["a", "b", "c"]}))
    assert transcriptapi.fetch_transcript("vid", "sk") is None  # non-dict items skipped → empty → None


def test_language_mapping_reserves_asr_slot():
    many = ",".join(f"l{i}" for i in range(12))
    out = transcriptapi.languages_from_sub_langs(many).split(",")
    assert out[-1] == "asr"
    assert len(out) <= 10


def test_load_api_key_env_and_dotenv(monkeypatch, tmp_path):
    monkeypatch.setenv("TRANSCRIPTAPI_API_KEY", "sk_env")
    assert transcriptapi.load_api_key() == "sk_env"
    monkeypatch.delenv("TRANSCRIPTAPI_API_KEY", raising=False)
    (tmp_path / ".env").write_text('TRANSCRIPTAPI_API_KEY="sk_dot"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(transcriptapi.Path, "home", lambda: tmp_path / "nohome")
    assert transcriptapi.load_api_key() == "sk_dot"
