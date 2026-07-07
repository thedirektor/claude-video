#!/usr/bin/env python3
"""Transcribe a video via Groq or OpenAI Whisper API.

Strategy: extract audio (mono 16kHz mp3, tiny payload), upload to whichever
API has a key. Returns segments in the same shape as transcribe.parse_vtt so
the rest of the pipeline (filter_range, format_transcript) doesn't care where
the transcript came from.

Pure stdlib — no `pip install groq` or `pip install openai` needed.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import mimetypes
import os
import shutil
import ssl
import subprocess
import sys
import time
import urllib.error
import uuid
from pathlib import Path
from urllib.request import Request, urlopen

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_MODEL = "whisper-large-v3"

OPENAI_ENDPOINT = "https://api.openai.com/v1/audio/transcriptions"
OPENAI_MODEL = "whisper-1"

# Both Groq's free tier and OpenAI whisper-1 cap uploads at 25 MB. We target a
# margin under that so multipart framing overhead never pushes a chunk over.
MAX_UPLOAD_BYTES = 24 * 1024 * 1024

# Content-addressed chunk-transcript cache. Keyed by sha256(chunk bytes + model),
# so a re-run (crash resume, repeated question) reuses already-uploaded chunks
# instead of burning the hourly Whisper quota again. Best-effort: any cache I/O
# failure is swallowed and the chunk simply re-uploads. Source: PR #10.
TRANSCRIPT_CACHE_DIR = Path.home() / ".cache" / "watch" / "transcripts"


def _chunk_cache_key(audio_path: Path, model: str) -> str:
    h = hashlib.sha256()
    h.update(audio_path.read_bytes())
    h.update(b"\x00")
    h.update(model.encode("utf-8"))
    return h.hexdigest()


def _cache_load(key: str) -> list[dict] | None:
    path = TRANSCRIPT_CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        blob = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(blob, dict):
        return None
    segments = blob.get("segments")
    return segments if isinstance(segments, list) else None


def _cache_save(key: str, segments: list[dict], model: str) -> None:
    try:
        TRANSCRIPT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (TRANSCRIPT_CACHE_DIR / f"{key}.json").write_text(
            json.dumps({"model": model, "segments": segments}), encoding="utf-8"
        )
    except OSError:
        pass  # cache is best-effort; never fail transcription over it


def plan_chunks(
    total_seconds: float,
    total_bytes: int,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> list[tuple[float, float]]:
    """Split a duration into contiguous (offset, duration) chunks under max_bytes.

    Size scales linearly with duration (constant-bitrate mono mp3), so an even
    time split yields evenly-sized chunks. Returns a single full-length chunk
    when the audio already fits.
    """
    if total_bytes <= max_bytes or total_seconds <= 0:
        return [(0.0, total_seconds)]

    n = math.ceil(total_bytes / max_bytes)
    chunk = total_seconds / n
    plan: list[tuple[float, float]] = []
    for i in range(n):
        offset = i * chunk
        # The last chunk absorbs any rounding remainder so durations sum exactly.
        duration = (total_seconds - offset) if i == n - 1 else chunk
        plan.append((round(offset, 3), round(duration, 3)))
    return plan


def load_api_key(preferred: str | None = None) -> tuple[str, str] | tuple[None, None]:
    """Return (backend, api_key). Prefers Groq, falls back to OpenAI.

    If `preferred` is "groq" or "openai", only that backend's key is considered.
    """
    def _from_env(name: str) -> str | None:
        value = os.environ.get(name)
        return value.strip() if value else None

    def _from_dotenv(path: Path, name: str) -> str | None:
        if not path.exists():
            return None
        try:
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                if key.strip() != name:
                    continue
                value = value.strip()
                if len(value) >= 2 and value[0] in ('"', "'") and value[-1] == value[0]:
                    value = value[1:-1]
                return value or None
        except OSError:
            return None
        return None

    dotenv_paths = [
        Path.home() / ".config" / "watch" / ".env",
        Path.cwd() / ".env",
    ]

    candidates = (("GROQ_API_KEY", "groq"), ("OPENAI_API_KEY", "openai"))
    if preferred is not None:
        candidates = tuple(c for c in candidates if c[1] == preferred)

    for key_name, backend in candidates:
        value = _from_env(key_name)
        if not value:
            for candidate in dotenv_paths:
                value = _from_dotenv(candidate, key_name)
                if value:
                    break
        if value:
            return backend, value

    return None, None


def resolve_backend(
    preferred: str | None = None,
) -> tuple[str | None, str | None, str | None]:
    """Pick a Whisper backend across local / groq / openai.

    Returns (backend, api_key, error_hint).
      - For 'local', api_key is None.
      - When (None, None, hint) is returned, no backend is usable.

    If `preferred` is set, that backend is required:
      - 'local'      → must have faster-whisper + a CUDA GPU available
      - 'groq'       → must have GROQ_API_KEY
      - 'openai'     → must have OPENAI_API_KEY
      - 'assemblyai' → must have ASSEMBLYAI_API_KEY (paid; speaker diarization)
    No silent fallback to a different backend when the user asked for one.

    If `preferred` is None, auto-select with priority:
      1. 'local'  (if faster-whisper imports and a CUDA GPU is reachable)
      2. 'groq'   (if GROQ_API_KEY is set)
      3. 'openai' (if OPENAI_API_KEY is set)
    AssemblyAI is intentionally excluded from auto-pick because it's paid
    and only adds value when you specifically want speaker diarization.
    """
    # Make sure sibling modules (whisper_local, whisper_assemblyai) are
    # importable when whisper.py is imported standalone. watch.py already
    # inserts scripts/ on sys.path, but doing it here too keeps direct CLI
    # use of whisper.py honest.
    sys.path.insert(0, str(Path(__file__).parent))

    if preferred == "local":
        try:
            from whisper_local import is_available, INSTALL_HINT
        except ImportError as exc:
            return None, None, (
                f"--whisper local was set but whisper_local module is unavailable: {exc}"
            )
        ok, reason = is_available()
        if ok:
            return "local", None, None
        return None, None, (
            f"--whisper local was set but it can't run: {reason}\n{INSTALL_HINT}"
        )

    if preferred == "assemblyai":
        try:
            from whisper_assemblyai import (
                load_api_key as _aai_load_key,
                INSTALL_HINT as AAI_INSTALL_HINT,
            )
        except ImportError as exc:
            return None, None, (
                f"--whisper assemblyai was set but whisper_assemblyai module is unavailable: {exc}"
            )
        key = _aai_load_key()
        if key:
            return "assemblyai", key, None
        return None, None, (
            "--whisper assemblyai was set but ASSEMBLYAI_API_KEY is missing.\n"
            + AAI_INSTALL_HINT
        )

    if preferred in ("groq", "openai"):
        backend, key = load_api_key(preferred)
        if backend and key:
            return backend, key, None
        return None, None, (
            f"--whisper {preferred} was set but the matching API key is missing"
        )

    # Auto-select. Try local first (no API call, no upload).
    try:
        from whisper_local import is_available
        ok, _ = is_available()
        if ok:
            return "local", None, None
    except ImportError:
        pass  # faster-whisper not installed → fall through to API backends

    backend, key = load_api_key()
    if backend and key:
        return backend, key, None

    return None, None, (
        "no Whisper backend available — no local GPU + faster-whisper, "
        "and neither GROQ_API_KEY nor OPENAI_API_KEY is set"
    )


def extract_audio(
    video_path: str,
    out_path: Path,
    start_seconds: float | None = None,
    duration_seconds: float | None = None,
) -> Path:
    """Extract mono 16kHz 64kbps mp3 — ~480 kB/min, fits any Whisper limit.

    With start_seconds/duration_seconds, extract only that window (fast input
    seek before -i, so the output's timestamps start at 0).
    """
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not installed. Install with: brew install ffmpeg")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if start_seconds is not None and start_seconds > 0:
        cmd += ["-ss", f"{start_seconds:.3f}"]
    cmd += ["-i", str(Path(video_path).resolve())]
    if duration_seconds is not None and duration_seconds > 0:
        cmd += ["-t", f"{duration_seconds:.3f}"]
    cmd += [
        "-vn",
        "-acodec", "libmp3lame",
        "-ar", "16000",
        "-ac", "1",
        "-b:a", "64k",
        str(out_path.resolve()),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg audio extraction failed: {result.stderr.strip()}")
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise SystemExit("ffmpeg produced no audio — video may have no audio track")
    return out_path


def audio_duration(audio_path: Path) -> float:
    """Return the duration of an audio file in seconds via ffprobe."""
    if shutil.which("ffprobe") is None:
        raise SystemExit("ffprobe is not installed. Install with: brew install ffmpeg")

    result = subprocess.run(
        [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            str(audio_path.resolve()),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"ffprobe failed: {result.stderr.strip()}")
    fmt = json.loads(result.stdout or "{}").get("format", {})
    return float(fmt.get("duration") or 0.0)


def split_audio(
    full_audio: Path,
    work_dir: Path,
    plan: list[tuple[float, float]],
) -> list[tuple[Path, float]]:
    """Slice full_audio into per-plan chunk files, returning (path, offset) pairs.

    Uses stream copy (`-c copy`) so there is no re-encode and no quality loss;
    mp3 frame boundaries are close enough for transcription's purposes.
    """
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not installed. Install with: brew install ffmpeg")

    work_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[tuple[Path, float]] = []
    for index, (offset, duration) in enumerate(plan):
        out_path = work_dir / f"chunk_{index:03d}.mp3"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-ss", f"{offset:.3f}",
            "-i", str(full_audio.resolve()),
            "-t", f"{duration:.3f}",
            "-c", "copy",
            str(out_path.resolve()),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
            raise SystemExit(
                f"ffmpeg failed to split audio chunk {index + 1}: {result.stderr.strip()}"
            )
        chunks.append((out_path, offset))
    return chunks


def _build_multipart(fields: dict[str, str], file_path: Path) -> tuple[bytes, str]:
    """Assemble a multipart/form-data body the Whisper APIs accept.

    Whisper's multipart upload is small and predictable — doing it by hand
    keeps us on pure stdlib instead of pulling requests/groq/openai SDKs.
    """
    boundary = f"----WatchBoundary{uuid.uuid4().hex}"
    eol = b"\r\n"
    buf = io.BytesIO()

    for name, value in fields.items():
        buf.write(f"--{boundary}".encode()); buf.write(eol)
        buf.write(f'Content-Disposition: form-data; name="{name}"'.encode()); buf.write(eol)
        buf.write(eol)
        buf.write(str(value).encode()); buf.write(eol)

    mimetype = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    buf.write(f"--{boundary}".encode()); buf.write(eol)
    buf.write(
        f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"'.encode()
    )
    buf.write(eol)
    buf.write(f"Content-Type: {mimetype}".encode()); buf.write(eol)
    buf.write(eol)
    buf.write(file_path.read_bytes())
    buf.write(eol)
    buf.write(f"--{boundary}--".encode()); buf.write(eol)

    return buf.getvalue(), boundary


MAX_ATTEMPTS = 4       # initial + 3 retries
MAX_429_RETRIES = 2
RETRY_BASE_DELAY = 2.0


def _post_whisper(endpoint: str, api_key: str, model: str, audio_path: Path) -> dict:
    fields = {
        "model": model,
        "response_format": "verbose_json",
        "temperature": "0",
    }
    body, boundary = _build_multipart(fields, audio_path)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        # Groq sits behind Cloudflare — the default `Python-urllib/3.x` UA
        # trips WAF rule 1010 (403) before auth even runs. Any non-default
        # UA clears it; we identify honestly.
        "User-Agent": "watch-skill/1.0 (+claude-code; python-urllib)",
    }

    context = ssl.create_default_context()
    rate_limit_hits = 0
    last_exc: Exception | None = None
    last_detail = ""

    for attempt in range(MAX_ATTEMPTS):
        request = Request(endpoint, data=body, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=300, context=context) as response:
                payload = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = _read_error_body(exc)
            last_exc, last_detail = exc, detail

            # 4xx other than 429 are client errors — no retry will fix them.
            if 400 <= exc.code < 500 and exc.code != 429:
                raise SystemExit(f"Whisper request failed: {exc}{detail}")

            if exc.code == 429:
                rate_limit_hits += 1
                if rate_limit_hits >= MAX_429_RETRIES:
                    raise SystemExit(f"Whisper request failed: {exc}{detail}")
                delay = _retry_after(exc) or RETRY_BASE_DELAY * (2 ** attempt) + 1
            else:
                delay = RETRY_BASE_DELAY * (2 ** attempt)

            if attempt < MAX_ATTEMPTS - 1:
                print(
                    f"[watch] whisper HTTP {exc.code} — retrying in {delay:.1f}s "
                    f"(attempt {attempt + 2}/{MAX_ATTEMPTS})",
                    file=sys.stderr,
                )
                time.sleep(delay)
            continue
        except (urllib.error.URLError, TimeoutError, ConnectionResetError, OSError) as exc:
            last_exc, last_detail = exc, ""
            if attempt < MAX_ATTEMPTS - 1:
                delay = RETRY_BASE_DELAY * (attempt + 1)
                print(
                    f"[watch] whisper network error ({type(exc).__name__}: {exc}) — "
                    f"retrying in {delay:.1f}s (attempt {attempt + 2}/{MAX_ATTEMPTS})",
                    file=sys.stderr,
                )
                time.sleep(delay)
            continue

        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Whisper returned non-JSON response: {exc}: {payload[:200]}")

    raise SystemExit(
        f"Whisper request failed after {MAX_ATTEMPTS} attempts: {last_exc}{last_detail}"
    )


def _read_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read()
    except Exception:
        return ""
    if not body:
        return ""
    try:
        return f" — {body.decode('utf-8', errors='replace')[:400]}"
    except Exception:
        return ""


def _retry_after(exc: urllib.error.HTTPError) -> float | None:
    header = exc.headers.get("Retry-After") if getattr(exc, "headers", None) else None
    if not header:
        return None
    try:
        return float(header)
    except ValueError:
        return None


def shift_segments(segments: list[dict], offset_seconds: float) -> list[dict]:
    """Return a copy of segments with start/end shifted by offset_seconds.

    Each chunk is transcribed in isolation, so Whisper returns 0-based timestamps
    per chunk; shifting by the chunk's offset stitches them into source time.
    """
    if offset_seconds == 0:
        return segments
    return [
        {
            "start": round(seg["start"] + offset_seconds, 2),
            "end": round(seg["end"] + offset_seconds, 2),
            "text": seg["text"],
        }
        for seg in segments
    ]


def _shift_all(segments: list[dict], offset_seconds: float) -> list[dict]:
    """Shift start/end by offset, preserving every other key (e.g. `speaker`).

    Used to lift window-relative Whisper timestamps back to source-absolute time
    after windowed extraction. Unlike shift_segments, it keeps diarization tags.
    """
    if not offset_seconds:
        return segments
    out: list[dict] = []
    for seg in segments:
        moved = dict(seg)
        moved["start"] = round(seg["start"] + offset_seconds, 2)
        moved["end"] = round(seg["end"] + offset_seconds, 2)
        out.append(moved)
    return out


def _segments_from_response(data: dict) -> list[dict]:
    """Convert Whisper verbose_json into our {start, end, text} segment format."""
    out: list[dict] = []
    for seg in data.get("segments") or []:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        out.append({
            "start": round(float(seg.get("start") or 0.0), 2),
            "end": round(float(seg.get("end") or 0.0), 2),
            "text": text,
        })

    if not out:
        full = (data.get("text") or "").strip()
        if full:
            out.append({"start": 0.0, "end": 0.0, "text": full})

    return out


def transcribe_chunks(
    chunks: list[tuple[Path, float]],
    transcribe_one,
) -> list[dict]:
    """Transcribe each chunk, shift its segments by the chunk offset, concatenate.

    A chunk that fails after its own retries is logged and skipped so one bad
    slice doesn't discard the whole transcript. Raises only if every chunk fails.
    """
    segments: list[dict] = []
    failures = 0
    for index, (path, offset) in enumerate(chunks):
        try:
            chunk_segments = transcribe_one(path)
        except SystemExit as exc:
            failures += 1
            print(
                f"[watch] chunk {index + 1}/{len(chunks)} failed — skipping ({exc})",
                file=sys.stderr,
            )
            continue
        segments.extend(shift_segments(chunk_segments, offset))
        print(
            f"[watch] chunk {index + 1}/{len(chunks)} → {len(chunk_segments)} segments",
            file=sys.stderr,
        )

    if failures == len(chunks):
        raise SystemExit("Whisper failed on every audio chunk")
    return segments


def _transcribe_file(
    backend: str, api_key: str, audio_path: Path, use_cache: bool = True
) -> list[dict]:
    """Upload one audio file and return its 0-based segments, with an optional cache."""
    if backend == "groq":
        endpoint, model = GROQ_ENDPOINT, GROQ_MODEL
    elif backend == "openai":
        endpoint, model = OPENAI_ENDPOINT, OPENAI_MODEL
    else:
        raise SystemExit(f"Unknown whisper backend: {backend}")

    key = _chunk_cache_key(audio_path, model) if use_cache else None
    if key is not None:
        cached = _cache_load(key)
        if cached is not None:
            print(f"[watch] cache hit for {audio_path.name} ({model})", file=sys.stderr)
            return cached

    response = _post_whisper(endpoint, api_key, model, audio_path)
    segments = _segments_from_response(response)
    if key is not None:
        _cache_save(key, segments, model)
    return segments


def transcribe_video(
    video_path: str,
    audio_out: Path,
    backend: str | None = None,
    api_key: str | None = None,
    model_name: str | None = None,
    enable_diarization: bool = False,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    use_cache: bool = True,
) -> tuple[list[dict], str]:
    """Run the full flow: extract audio → transcribe (local/API) → parse segments.

    Returns (segments, backend_used). Raises SystemExit on any failure.

    Per-backend notes:
      - local:       runs faster-whisper on the extracted mono 16 kHz mp3;
                     `model_name` selects the faster-whisper model (defaults
                     to whisper_local.DEFAULT_MODEL).
      - assemblyai:  uploads the extracted mp3 to AssemblyAI; `enable_diarization`
                     toggles speaker labels on the returned segments.
      - groq/openai: upstream's existing size-check -> plan_chunks -> split_audio
                     -> transcribe_chunks flow, untouched.
    """
    if backend is None:
        backend, api_key, hint = resolve_backend(None)
        if backend is None:
            raise SystemExit(hint or "No Whisper backend available. Run setup.py")

    print(f"[watch] extracting audio for Whisper ({backend})…", file=sys.stderr)
    window_offset = start_seconds or 0.0
    duration_seconds = None
    if start_seconds is not None and end_seconds is not None and end_seconds > start_seconds:
        duration_seconds = end_seconds - start_seconds
    audio_path = extract_audio(
        video_path, audio_out, start_seconds=start_seconds, duration_seconds=duration_seconds
    )

    if backend == "local":
        sys.path.insert(0, str(Path(__file__).parent))
        from whisper_local import DEFAULT_MODEL, transcribe_local

        segments = transcribe_local(audio_path, model_name=model_name or DEFAULT_MODEL)
        if not segments:
            raise SystemExit("Whisper returned no transcript segments")
        segments = _shift_all(segments, window_offset)
        print(f"[watch] transcribed {len(segments)} segments via local", file=sys.stderr)
        return segments, "local"

    if backend == "assemblyai":
        sys.path.insert(0, str(Path(__file__).parent))
        from whisper_assemblyai import transcribe_assemblyai

        segments = transcribe_assemblyai(audio_path, enable_diarization=enable_diarization)
        if not segments:
            raise SystemExit("Whisper returned no transcript segments")
        segments = _shift_all(segments, window_offset)
        print(f"[watch] transcribed {len(segments)} segments via assemblyai", file=sys.stderr)
        return segments, "assemblyai"

    # groq / openai: keep upstream's existing size-check -> plan_chunks ->
    # split_audio -> transcribe_chunks flow EXACTLY as-is from here down.
    if backend is None or api_key is None:
        detected_backend, detected_key = load_api_key()
        backend = backend or detected_backend
        api_key = api_key or detected_key

    if not backend or not api_key:
        setup_py = Path(__file__).resolve().parent / "setup.py"
        raise SystemExit(
            "No Whisper API key available. Set GROQ_API_KEY (preferred) or OPENAI_API_KEY "
            "in the environment or in ~/.config/watch/.env. "
            f"Run `python3 {setup_py}` to configure."
        )

    audio_bytes = audio_path.stat().st_size

    def transcribe_one(path: Path) -> list[dict]:
        return _transcribe_file(backend, api_key, path, use_cache=use_cache)

    if audio_bytes <= MAX_UPLOAD_BYTES:
        print(
            f"[watch] audio: {audio_bytes / 1024:.0f} kB — uploading to {backend} Whisper…",
            file=sys.stderr,
        )
        segments = transcribe_one(audio_path)
    else:
        duration = audio_duration(audio_path)
        plan = plan_chunks(duration, audio_bytes, MAX_UPLOAD_BYTES)
        print(
            f"[watch] audio: {audio_bytes / (1024 * 1024):.0f} MB exceeds "
            f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB — splitting into {len(plan)} chunks…",
            file=sys.stderr,
        )
        chunks = split_audio(audio_path, audio_out.parent / "chunks", plan)
        segments = transcribe_chunks(chunks, transcribe_one)

    if not segments:
        raise SystemExit("Whisper returned no transcript segments")

    segments = _shift_all(segments, window_offset)
    print(f"[watch] transcribed {len(segments)} segments via {backend}", file=sys.stderr)
    return segments, backend


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: whisper.py <video-path> [<audio-out.mp3>] [--backend groq|openai]", file=sys.stderr)
        raise SystemExit(2)

    video = sys.argv[1]
    audio_out = Path(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else Path("audio.mp3")
    backend_override = None
    if "--backend" in sys.argv:
        backend_override = sys.argv[sys.argv.index("--backend") + 1]

    segments, backend = transcribe_video(video, audio_out, backend=backend_override)
    print(json.dumps({"backend": backend, "segments": segments}, indent=2))
