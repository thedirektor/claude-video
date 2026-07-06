#!/usr/bin/env python3
"""AssemblyAI Whisper backend for /watch.

AssemblyAI's differentiating feature is automatic speaker diarization — when
`speaker_labels=True`, every transcribed utterance comes back tagged with a
speaker letter (A, B, C, …). For interview / podcast / multi-speaker UGC
content this is the only paid Whisper backend that lights up speaker turns
without an extra pyannote / WhisperX pipeline.

Returns segments in the same {start, end, text} shape as the other backends,
plus an optional `speaker` field (e.g. "Speaker A") when diarization is on.
Downstream `format_transcript` auto-detects the field and switches output
formatting accordingly.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


# Force UTF-8 on stdout/stderr (consistent with the rest of the scripts).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


INSTALL_HINT = (
    "AssemblyAI backend requires the SDK + an API key:\n"
    "    pip install assemblyai\n"
    "    Add ASSEMBLYAI_API_KEY=<key> to ~/.config/watch/.env\n"
    "Get a key at https://www.assemblyai.com/dashboard/signup ($50 free credits)."
)


def load_api_key() -> str | None:
    """Return ASSEMBLYAI_API_KEY from env or ~/.config/watch/.env (or cwd .env).

    Mirrors the dotenv pattern in scripts/whisper.py / scripts/gemini.py so
    every backend reads keys the same way.
    """
    value = os.environ.get("ASSEMBLYAI_API_KEY")
    if value and value.strip():
        return value.strip()

    candidates = [
        Path.home() / ".config" / "watch" / ".env",
        Path.cwd() / ".env",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            for line in candidate.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, raw = line.partition("=")
                if key.strip() != "ASSEMBLYAI_API_KEY":
                    continue
                raw = raw.strip()
                if len(raw) >= 2 and raw[0] in ('"', "'") and raw[-1] == raw[0]:
                    raw = raw[1:-1]
                if raw:
                    return raw
        except OSError:
            continue
    return None


def transcribe_assemblyai(
    audio_path: Path | str,
    enable_diarization: bool = True,
) -> list[dict]:
    """Transcribe audio via AssemblyAI; return {start, end, text, speaker?} segments.

    When `enable_diarization` is True, each segment also carries a `speaker`
    field like "Speaker A". When False, the speaker field is omitted and we
    return sentence-level segments (still timestamped).

    The SDK's synchronous `.transcribe()` blocks until the job leaves the
    queued/processing state. Raises SystemExit on missing key, SDK import
    error, or upstream transcription error.
    """
    api_key = load_api_key()
    if not api_key:
        raise SystemExit(
            "ASSEMBLYAI_API_KEY is not set. Add it to ~/.config/watch/.env or "
            "export ASSEMBLYAI_API_KEY in your environment.\n" + INSTALL_HINT
        )

    try:
        import assemblyai as aai
    except ImportError as exc:
        raise SystemExit(
            f"assemblyai SDK is not installed ({exc}).\n{INSTALL_HINT}"
        )

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise SystemExit(f"audio file not found: {audio_path}")

    aai.settings.api_key = api_key
    # `speech_models` (plural) is the current required field — the API
    # deprecated the singular `speech_model` and the SDK 0.64 enum's string
    # value `'universal'` no longer matches the accepted server-side names
    # (`universal-2` and `universal-3-pro`). Pass the string directly.
    # `universal-2` is the production tier that supports both diarization
    # and language_detection without the price premium of universal-3-pro.
    config = aai.TranscriptionConfig(
        speech_models=["universal-2"],
        speaker_labels=enable_diarization,
        language_detection=True,
        punctuate=True,
        format_text=True,
    )

    diarize_label = "with speaker diarization" if enable_diarization else "without diarization"
    print(
        f"[watch] uploading {audio_path.name} "
        f"({audio_path.stat().st_size / 1024:.0f} kB) to AssemblyAI ({diarize_label})…",
        file=sys.stderr,
    )

    transcriber = aai.Transcriber(config=config)
    try:
        # `.transcribe()` is synchronous: it uploads, polls until terminal,
        # and returns the finished Transcript. No manual polling needed.
        print("[watch] AssemblyAI processing — waiting for completion…", file=sys.stderr)
        transcript = transcriber.transcribe(str(audio_path))
    except Exception as exc:
        raise SystemExit(
            f"AssemblyAI request failed: {type(exc).__name__}: {exc}"
        )

    if transcript.status == aai.TranscriptStatus.error:
        raise SystemExit(
            f"AssemblyAI transcription returned an error: {transcript.error}"
        )

    out: list[dict] = []

    if enable_diarization and transcript.utterances:
        # Diarized: one segment per speaker turn.
        for utt in transcript.utterances:
            text = (utt.text or "").strip()
            if not text:
                continue
            out.append({
                "start": round((utt.start or 0) / 1000.0, 2),
                "end": round((utt.end or 0) / 1000.0, 2),
                "text": text,
                "speaker": f"Speaker {utt.speaker}",
            })
    else:
        # No diarization (or diarization returned nothing — single-channel
        # audio with detection disabled). Use sentence-level granularity so
        # speech-window logic still has timing.
        try:
            sentences = transcript.get_sentences()
        except Exception:
            sentences = []
        for sent in sentences:
            text = (sent.text or "").strip()
            if not text:
                continue
            out.append({
                "start": round((sent.start or 0) / 1000.0, 2),
                "end": round((sent.end or 0) / 1000.0, 2),
                "text": text,
            })
        # Final fallback: whole-transcript text without timing if sentences
        # came back empty for some reason (rare).
        if not out and transcript.text:
            out.append({
                "start": 0.0,
                "end": round((transcript.audio_duration or 0), 2),
                "text": transcript.text.strip(),
            })

    if not out:
        raise SystemExit("AssemblyAI returned no transcript segments")

    speakers = sorted({s["speaker"] for s in out if "speaker" in s})
    speaker_note = f", {len(speakers)} speakers ({', '.join(speakers)})" if speakers else ""
    detected_lang = getattr(transcript, "language_code", None) or "?"
    print(
        f"[watch] AssemblyAI transcribed {len(out)} segments — language: {detected_lang}{speaker_note}",
        file=sys.stderr,
    )

    return out


if __name__ == "__main__":
    import json

    if len(sys.argv) < 2:
        print(
            "usage: whisper_assemblyai.py <audio-or-video-path> [--no-diarize]",
            file=sys.stderr,
        )
        raise SystemExit(2)

    cli_audio = sys.argv[1]
    cli_diarize = "--no-diarize" not in sys.argv
    segments = transcribe_assemblyai(cli_audio, enable_diarization=cli_diarize)
    print(json.dumps(
        {"backend": "assemblyai", "diarize": cli_diarize, "segments": segments},
        indent=2,
        ensure_ascii=False,
    ))
