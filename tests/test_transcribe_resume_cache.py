"""Acceptance test: a second transcription over the same audio uploads zero
chunks, because the chunk cache is wired all the way through transcribe_video
(extract_audio -> size check -> _transcribe_file -> _post_whisper).

This closes the gap left by test_transcript_cache.py, which only exercises
_transcribe_file directly and never proves the cache survives a full
transcribe_video() round trip.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import whisper  # noqa: E402


def _make_clip_with_audio(tmp_path: Path) -> Path:
    clip = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-t", "2", "-i", "sine=frequency=440:sample_rate=16000",
            "-f", "lavfi", "-t", "2", "-i", "color=c=blue:s=160x120:r=10",
            "-map", "1:v", "-map", "0:a",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest",
            str(clip),
        ],
        check=True,
    )
    return clip


def test_transcribe_video_resume_skips_reupload_via_chunk_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(whisper, "TRANSCRIPT_CACHE_DIR", tmp_path / "cache")

    clip = _make_clip_with_audio(tmp_path)

    calls = {"n": 0}
    segs = [{"start": 0.0, "end": 1.0, "text": "x"}]

    def counting_stub(endpoint, api_key, model, audio_path):
        calls["n"] += 1
        return {"segments": [{"start": 0.0, "end": 1.0, "text": "x"}]}

    monkeypatch.setattr(whisper, "_post_whisper", counting_stub)

    first_segments, first_backend = whisper.transcribe_video(
        str(clip), tmp_path / "audio1.mp3", backend="groq", api_key="k", use_cache=True
    )
    assert first_backend == "groq"
    assert first_segments == segs
    first_calls = calls["n"]
    assert first_calls >= 1  # miss on first run -> at least one upload

    second_segments, second_backend = whisper.transcribe_video(
        str(clip), tmp_path / "audio2.mp3", backend="groq", api_key="k", use_cache=True
    )

    assert second_backend == "groq"
    assert second_segments == first_segments
    assert calls["n"] == first_calls  # second run: zero additional uploads
