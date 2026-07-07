"""Regression lock: a persistent 5xx stops after MAX_ATTEMPTS — never loops forever."""
from __future__ import annotations

import sys
import urllib.error
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import whisper  # noqa: E402


def test_persistent_500_is_bounded(monkeypatch, tmp_path):
    audio = tmp_path / "chunk.mp3"
    audio.write_bytes(b"MP3")

    attempts = {"n": 0}

    def always_500(request, timeout=None, context=None):
        attempts["n"] += 1
        raise urllib.error.HTTPError(
            url="http://x", code=500, msg="err", hdrs=None, fp=None
        )

    monkeypatch.setattr(whisper, "urlopen", always_500)
    monkeypatch.setattr(whisper.time, "sleep", lambda *_a, **_k: None)  # no real backoff

    with pytest.raises(SystemExit):
        whisper._post_whisper("http://x", "key", "model", audio)

    # Bounded: exactly MAX_ATTEMPTS uploads, then it gives up. No infinite loop.
    assert attempts["n"] == whisper.MAX_ATTEMPTS
