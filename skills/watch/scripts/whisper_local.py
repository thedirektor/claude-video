#!/usr/bin/env python3
"""Local Whisper transcription via faster-whisper on a CUDA GPU.

Runs Whisper directly on the user's GPU using faster-whisper + CTranslate2 — no
API key, no upload, no rate limits. Returns segments in the same {start, end,
text} shape as scripts/whisper.py so the rest of the pipeline doesn't care
which backend produced the transcript.

faster-whisper, ctranslate2, and the CUDA DLLs (nvidia-cublas-cu12,
nvidia-cudnn-cu12) are imported lazily. The module loads cleanly even when
the dependencies aren't installed; only is_available() / transcribe_local()
will fail.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Force UTF-8 on stdout/stderr so non-ASCII output (Spanish transcripts, etc.)
# doesn't crash on Windows where the default is cp1252.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


def _register_cuda_dlls_on_windows() -> None:
    """Register pip-installed nvidia-cublas / nvidia-cudnn DLL directories.

    Runs only on Windows. The cuBLAS / cuDNN wheels drop their DLLs under
    `<sys.prefix>/Lib/site-packages/nvidia/<lib>/bin`, but Windows doesn't
    search those directories by default — so `import ctranslate2` would fail
    with a DLL-load error. We hand them to the loader explicitly via
    os.add_dll_directory() so the GPU backend can initialize without the user
    having to edit their PATH.

    Silent in every failure case: non-Windows platforms, missing dirs (CPU-only
    install), or os.add_dll_directory raising. The downstream is_available()
    probe is the single place we surface "local backend isn't reachable".
    """
    if sys.platform != "win32":
        return

    site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    for sub in ("cublas", "cudnn"):
        bin_dir = site_packages / "nvidia" / sub / "bin"
        if not bin_dir.is_dir():
            continue
        try:
            os.add_dll_directory(str(bin_dir))
        except (OSError, AttributeError):
            # AttributeError on Python < 3.8 (we don't support that, but cheap
            # to guard); OSError if Windows refuses to register the path. Either
            # way, fall through — is_available() will catch the resulting CUDA
            # init failure and the user can apply the manual PATH fallback.
            pass


_register_cuda_dlls_on_windows()


VALID_MODELS = ("tiny", "base", "small", "medium", "large-v2", "large-v3")
DEFAULT_MODEL = "large-v3"

INSTALL_HINT = (
    "The local Whisper backend needs faster-whisper plus the CUDA runtime DLLs:\n"
    "    pip install faster-whisper nvidia-cublas-cu12 nvidia-cudnn-cu12\n"
    "It also needs an NVIDIA GPU with CUDA. Use --whisper groq or --whisper openai "
    "to transcribe via API instead."
)


def is_available() -> tuple[bool, str]:
    """Probe whether the local backend can actually run.

    Returns (ok, reason). When ok is False, reason explains why so the caller
    can either fall through to an API backend or surface the error.
    """
    try:
        import faster_whisper  # noqa: F401
    except ImportError as exc:
        return False, f"faster-whisper not installed ({exc})"

    try:
        import ctranslate2
    except ImportError as exc:
        return False, f"ctranslate2 not installed ({exc})"

    try:
        device_count = ctranslate2.get_cuda_device_count()
    except Exception as exc:
        # OSError is the typical failure when the cuBLAS / cuDNN DLLs aren't
        # on PATH. Catch broadly so a CUDA init failure becomes a fall-through,
        # not a crash.
        return False, f"CUDA init failed ({type(exc).__name__}: {exc})"

    if device_count <= 0:
        return False, "no CUDA-capable GPU detected"

    return True, ""


def transcribe_local(
    audio_path: Path | str,
    language: str | None = None,
    model_name: str = DEFAULT_MODEL,
    compute_type: str = "float16",
    device: str = "cuda",
) -> list[dict]:
    """Transcribe an audio or video file using faster-whisper on the GPU.

    `audio_path` can be any media ffmpeg can decode — faster-whisper invokes
    ffmpeg internally, so passing the original .mp4/.mp3/.m4a directly is fine
    and saves an extraction step.

    Returns a list of {start, end, text} segments, matching parse_vtt() shape.

    Raises SystemExit when the backend can't run or transcription fails.
    """
    ok, reason = is_available()
    if not ok:
        raise SystemExit(
            f"local Whisper backend unavailable: {reason}\n{INSTALL_HINT}"
        )

    if model_name not in VALID_MODELS:
        raise SystemExit(
            f"Unknown whisper model: {model_name!r}. "
            f"Choose from: {', '.join(VALID_MODELS)}"
        )

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise SystemExit(f"audio file not found: {audio_path}")

    from faster_whisper import WhisperModel

    print(
        f"[watch] loading faster-whisper {model_name!r} on {device} "
        f"({compute_type}) — first run downloads the model "
        f"(~3 GB for large-v3) into the HuggingFace cache; subsequent runs reuse it.",
        file=sys.stderr,
    )

    try:
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
    except Exception as exc:
        raise SystemExit(
            f"failed to load faster-whisper {model_name!r} on {device}: "
            f"{type(exc).__name__}: {exc}\n{INSTALL_HINT}"
        )

    print(
        f"[watch] transcribing {audio_path.name} via local GPU "
        f"(beam_size=5, vad_filter=True, language={'auto' if language is None else language})…",
        file=sys.stderr,
    )

    try:
        segments_iter, info = model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
            vad_filter=True,
        )
    except Exception as exc:
        raise SystemExit(
            f"local Whisper transcription failed: {type(exc).__name__}: {exc}"
        )

    out: list[dict] = []
    for seg in segments_iter:
        text = (seg.text or "").strip()
        if not text:
            continue
        out.append({
            "start": round(float(seg.start or 0.0), 2),
            "end": round(float(seg.end or 0.0), 2),
            "text": text,
        })

    detected_lang = getattr(info, "language", None) or "?"
    detected_prob = getattr(info, "language_probability", None)
    prob_str = f" (prob {detected_prob:.2f})" if detected_prob is not None else ""
    print(
        f"[watch] local transcribed {len(out)} segments — language: {detected_lang}{prob_str}",
        file=sys.stderr,
    )

    return out


if __name__ == "__main__":
    import json

    if len(sys.argv) < 2:
        print(
            "usage: whisper_local.py <audio-or-video-path> [model-name]",
            file=sys.stderr,
        )
        raise SystemExit(2)

    audio = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
    segments = transcribe_local(audio, model_name=model)
    print(json.dumps(
        {"backend": "local", "model": model, "segments": segments},
        indent=2,
        ensure_ascii=False,
    ))
