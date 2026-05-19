#!/usr/bin/env python3
"""OpenRouter backend for /watch — vision + audio transcription.

Vision path: encodes extracted frames as base64 image_url parts, prepends an
optional transcript, then POSTs to https://openrouter.ai/api/v1/chat/completions
using the OpenAI-compatible chat endpoint.

Audio path: sends an extracted mono mp3 to
https://openrouter.ai/api/v1/audio/transcriptions via multipart/form-data
(same shape as Groq / OpenAI Whisper). Returns segments in the same
{start, end, text} format consumed by the rest of the watch pipeline.

Pure stdlib — no new pip dependencies.
"""
from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
import ssl
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


OPENROUTER_BASE = "https://openrouter.ai/api/v1"
CHAT_ENDPOINT = f"{OPENROUTER_BASE}/chat/completions"
TRANSCRIPTION_ENDPOINT = f"{OPENROUTER_BASE}/audio/transcriptions"

DEFAULT_VISION_MODEL = "google/gemini-2.5-flash"
DEFAULT_AUDIO_MODEL = "openai/gpt-4o-mini-transcribe"

_HTTP_REFERER = "https://github.com/thedirektor/claude-video"
_X_TITLE = "claude-video /watch"

_MAX_ATTEMPTS = 3
_RETRY_BASE = 2.0

_GROQ_ENDPOINT = "https://api.groq.com/openai/v1/audio/transcriptions"
_GROQ_MODEL = "whisper-large-v3"


def load_api_key() -> str | None:
    """Return OPENROUTER_API_KEY from env or ~/.config/watch/.env."""
    value = os.environ.get("OPENROUTER_API_KEY")
    if value and value.strip():
        return value.strip()
    for candidate in [
        Path.home() / ".config" / "watch" / ".env",
        Path.cwd() / ".env",
    ]:
        if not candidate.exists():
            continue
        try:
            for line in candidate.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                if k.strip() != "OPENROUTER_API_KEY":
                    continue
                v = v.strip()
                if len(v) >= 2 and v[0] in ('"', "'") and v[-1] == v[0]:
                    v = v[1:-1]
                if v:
                    return v
        except OSError:
            continue
    return None


def _read_error(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read()
        return f" — {body.decode('utf-8', errors='replace')[:400]}" if body else ""
    except Exception:
        return ""


def _retry_after(exc: urllib.error.HTTPError) -> float | None:
    h = exc.headers.get("Retry-After") if getattr(exc, "headers", None) else None
    try:
        return float(h) if h else None
    except ValueError:
        return None


def _load_groq_key() -> str | None:
    """Return GROQ_API_KEY from env or ~/.config/watch/.env."""
    value = os.environ.get("GROQ_API_KEY")
    if value and value.strip():
        return value.strip()
    for candidate in [
        Path.home() / ".config" / "watch" / ".env",
        Path.cwd() / ".env",
    ]:
        if not candidate.exists():
            continue
        try:
            for line in candidate.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                if k.strip() != "GROQ_API_KEY":
                    continue
                v = v.strip()
                if len(v) >= 2 and v[0] in ('"', "'") and v[-1] == v[0]:
                    v = v[1:-1]
                if v:
                    return v
        except OSError:
            continue
    return None


def _groq_transcribe(audio_path: Path, groq_key: str) -> list[dict]:
    """Transcribe audio via Groq whisper-large-v3. Called as a fallback by transcribe_audio()."""
    boundary = f"----WatchGroq{uuid.uuid4().hex}"
    eol = b"\r\n"
    buf = io.BytesIO()
    for name, value in [
        ("model", _GROQ_MODEL),
        ("response_format", "verbose_json"),
        ("temperature", "0"),
    ]:
        buf.write(f"--{boundary}".encode()); buf.write(eol)
        buf.write(f'Content-Disposition: form-data; name="{name}"'.encode()); buf.write(eol)
        buf.write(eol)
        buf.write(str(value).encode()); buf.write(eol)
    mime = mimetypes.guess_type(audio_path.name)[0] or "application/octet-stream"
    buf.write(f"--{boundary}".encode()); buf.write(eol)
    buf.write(
        f'Content-Disposition: form-data; name="file"; filename="{audio_path.name}"'.encode()
    )
    buf.write(eol)
    buf.write(f"Content-Type: {mime}".encode()); buf.write(eol)
    buf.write(eol)
    buf.write(audio_path.read_bytes())
    buf.write(eol)
    buf.write(f"--{boundary}--".encode()); buf.write(eol)

    body = buf.getvalue()
    headers = {
        "Authorization": f"Bearer {groq_key}",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "User-Agent": "watch-skill/1.0 (+claude-code; python-urllib)",
    }

    ctx = ssl.create_default_context()
    req = Request(_GROQ_ENDPOINT, data=body, headers=headers, method="POST")
    try:
        with urlopen(req, timeout=300, context=ctx) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"Groq fallback transcription failed ({exc.code}){_read_error(exc)}")
    except Exception as exc:
        raise SystemExit(f"Groq fallback transcription failed: {type(exc).__name__}: {exc}")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Groq fallback returned non-JSON: {exc}: {raw[:200]}")

    out: list[dict] = []
    for seg in data.get("segments") or []:
        text = (seg.get("text") or "").strip()
        if text:
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


def transcribe_audio(
    audio_path: Path,
    model: str = DEFAULT_AUDIO_MODEL,
    api_key: str | None = None,
) -> list[dict]:
    """Transcribe audio via OpenRouter's transcription endpoint.

    Returns segments in {start, end, text} format. Falls back to a single
    segment from the plain-text response if verbose_json is not supported.

    If the OpenRouter endpoint fails after all retries, automatically falls
    back to Groq whisper-large-v3 using GROQ_API_KEY from the environment or
    ~/.config/watch/.env, logging a warning before the fallback triggers.
    Raises SystemExit only if both OpenRouter and the Groq fallback fail.
    """
    if api_key is None:
        api_key = load_api_key()
    if not api_key:
        raise SystemExit(
            "OPENROUTER_API_KEY not set. Add it to ~/.config/watch/.env or "
            "export OPENROUTER_API_KEY in your environment."
        )

    boundary = f"----WatchOR{uuid.uuid4().hex}"
    eol = b"\r\n"
    buf = io.BytesIO()
    for name, value in [
        ("model", model),
        ("response_format", "verbose_json"),
        ("temperature", "0"),
    ]:
        buf.write(f"--{boundary}".encode()); buf.write(eol)
        buf.write(f'Content-Disposition: form-data; name="{name}"'.encode()); buf.write(eol)
        buf.write(eol)
        buf.write(str(value).encode()); buf.write(eol)
    mime = mimetypes.guess_type(audio_path.name)[0] or "application/octet-stream"
    buf.write(f"--{boundary}".encode()); buf.write(eol)
    buf.write(
        f'Content-Disposition: form-data; name="file"; filename="{audio_path.name}"'.encode()
    )
    buf.write(eol)
    buf.write(f"Content-Type: {mime}".encode()); buf.write(eol)
    buf.write(eol)
    buf.write(audio_path.read_bytes())
    buf.write(eol)
    buf.write(f"--{boundary}--".encode()); buf.write(eol)

    body = buf.getvalue()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "HTTP-Referer": _HTTP_REFERER,
        "X-Title": _X_TITLE,
        "User-Agent": "watch-skill/1.0 (+claude-code; python-urllib)",
    }

    ctx = ssl.create_default_context()
    last_exc: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        req = Request(TRANSCRIPTION_ENDPOINT, data=body, headers=headers, method="POST")
        try:
            with urlopen(req, timeout=300, context=ctx) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            last_exc = exc
            detail = _read_error(exc)
            if 400 <= exc.code < 500 and exc.code != 429:
                raise SystemExit(f"OpenRouter transcription failed ({exc.code}){detail}")
            delay = _retry_after(exc) or _RETRY_BASE * (2 ** attempt)
            if attempt < _MAX_ATTEMPTS - 1:
                print(
                    f"[watch] openrouter audio HTTP {exc.code} — retrying in {delay:.1f}s",
                    file=sys.stderr,
                )
                time.sleep(delay)
            continue
        except (urllib.error.URLError, TimeoutError, ConnectionResetError, OSError) as exc:
            last_exc = exc
            if attempt < _MAX_ATTEMPTS - 1:
                delay = _RETRY_BASE * (attempt + 1)
                print(
                    f"[watch] openrouter audio network error — retrying in {delay:.1f}s",
                    file=sys.stderr,
                )
                time.sleep(delay)
            continue

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"OpenRouter transcription returned non-JSON: {exc}: {raw[:200]}"
            )

        out: list[dict] = []
        for seg in data.get("segments") or []:
            text = (seg.get("text") or "").strip()
            if text:
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

    # All OpenRouter retries exhausted — fall back to Groq whisper-large-v3
    groq_key = _load_groq_key()
    if not groq_key:
        raise SystemExit(
            f"OpenRouter transcription failed after {_MAX_ATTEMPTS} attempts: {last_exc}. "
            "No GROQ_API_KEY found for fallback — set it in ~/.config/watch/.env."
        )
    print(
        f"[watch] WARNING: OpenRouter audio transcription failed ({last_exc}) — "
        "falling back to Groq whisper-large-v3…",
        file=sys.stderr,
    )
    return _groq_transcribe(audio_path, groq_key)


def analyze_with_frames(
    frame_paths: list[str],
    transcript_text: str | None,
    question: str,
    vision_model: str = DEFAULT_VISION_MODEL,
    api_key: str | None = None,
) -> str:
    """POST frames (base64 image_url) + transcript + question to OpenRouter.

    Structures the message as: text preamble (with optional transcript), then
    one image_url part per frame, then the question as the final text part.
    Returns the model's text response. Raises SystemExit on failure.
    """
    if api_key is None:
        api_key = load_api_key()
    if not api_key:
        raise SystemExit(
            "OPENROUTER_API_KEY not set. Add it to ~/.config/watch/.env or "
            "export OPENROUTER_API_KEY in your environment."
        )

    content: list[dict] = []

    preamble_parts: list[str] = [
        f"The following {len(frame_paths)} frames were extracted from a video in chronological order."
    ]
    if transcript_text and transcript_text.strip():
        preamble_parts.append(f"Transcript:\n{transcript_text.strip()}")
    content.append({"type": "text", "text": "\n\n".join(preamble_parts)})

    loaded = 0
    for path_str in frame_paths:
        p = Path(path_str)
        if not p.exists():
            print(f"[watch] skipping missing frame: {p}", file=sys.stderr)
            continue
        mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        loaded += 1

    if loaded == 0:
        raise SystemExit("No valid frame files found to send to OpenRouter vision")

    content.append({"type": "text", "text": question.strip()})

    body = json.dumps({
        "model": vision_model,
        "messages": [{"role": "user", "content": content}],
    }).encode("utf-8")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": _HTTP_REFERER,
        "X-Title": _X_TITLE,
    }

    ctx = ssl.create_default_context()
    last_exc: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        req = Request(CHAT_ENDPOINT, data=body, headers=headers, method="POST")
        try:
            with urlopen(req, timeout=300, context=ctx) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            last_exc = exc
            detail = _read_error(exc)
            if 400 <= exc.code < 500 and exc.code != 429:
                raise SystemExit(f"OpenRouter vision failed ({exc.code}){detail}")
            delay = _retry_after(exc) or _RETRY_BASE * (2 ** attempt)
            if attempt < _MAX_ATTEMPTS - 1:
                print(
                    f"[watch] openrouter vision HTTP {exc.code} — retrying in {delay:.1f}s",
                    file=sys.stderr,
                )
                time.sleep(delay)
            continue
        except (urllib.error.URLError, TimeoutError, ConnectionResetError, OSError) as exc:
            last_exc = exc
            if attempt < _MAX_ATTEMPTS - 1:
                delay = _RETRY_BASE * (attempt + 1)
                print(
                    f"[watch] openrouter vision network error — retrying in {delay:.1f}s",
                    file=sys.stderr,
                )
                time.sleep(delay)
            continue

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"OpenRouter vision returned non-JSON: {exc}: {raw[:200]}")

        text = ""
        try:
            text = (data["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError, TypeError):
            pass
        if not text:
            raise SystemExit(f"OpenRouter vision returned empty response: {raw[:400]}")
        return text

    raise SystemExit(
        f"OpenRouter vision failed after {_MAX_ATTEMPTS} attempts: {last_exc}"
    )
