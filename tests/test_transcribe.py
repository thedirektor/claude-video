"""Tests for transcribe.py formatting."""
import transcribe


def test_format_transcript_plain_segments():
    """Legacy format without speakers: [MM:SS] with zero-padded values."""
    segs = [{"start": 0.0, "end": 2.0, "text": "hello"},
            {"start": 65.0, "end": 67.0, "text": "world"}]
    out = transcribe.format_transcript(segs)
    assert "[00:00] hello" in out
    assert "[01:05] world" in out
    assert "Speaker" not in out


def test_format_transcript_with_speakers():
    """Speaker-aware format: [Speaker X] (M:SS-M:SS) with non-zero-padded minutes."""
    segs = [{"start": 0.0, "end": 2.5, "text": "hi", "speaker": "Speaker A"},
            {"start": 3.0, "end": 5.0, "text": "yo", "speaker": "Speaker B"}]
    out = transcribe.format_transcript(segs)
    assert "[Speaker A] (0:00-0:02) hi" in out
    assert "[Speaker B] (0:03-0:05) yo" in out
