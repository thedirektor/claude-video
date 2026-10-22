#!/usr/bin/env python3
"""Gemini native video backend for /watch.

The Claude pipeline (frames + transcript) is the project's default. This module
is the alternative `--backend gemini` path: hand the *whole* video to a Gemini
multimodal model and let it answer the user's question natively, no frame
extraction or Whisper involved.

Two source modes:
- **Local file** — upload via the Gemini Files API, poll until state ACTIVE,
  then call `models.generate_content` with the file + the question.
- **YouTube URL** — pass the URL straight to the model as a FileData part.
  Gemini fetches the video server-side; no yt-dlp download required.

Returns the model's full text response so watch.py can print it as-is.
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse


# Force UTF-8 on stdout/stderr so non-ASCII output (Spanish responses, etc.)
# doesn't crash on Windows where the default is cp1252.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


DEFAULT_MODEL = "gemini-3.1-flash-lite"
# Three production tiers, picked at the time of writing for the
# generativelanguage.googleapis.com v1beta API. Override with --gemini-model.
#
#   gemini-3.1-flash-lite (default)
#       Stable May 7 2026. Newest and cheapest of the lineup; works on the
#       free tier where 2.0-flash often returns 429 limit:0. ~1M input /
#       ~65k output tokens, multimodal (text/image/video/audio/PDF).
#       Optimized for low-latency, high-frequency calls. Good general default.
#
#   gemini-2.5-flash
#       Balanced. Pick this when 3.1-flash-lite's quality isn't enough but
#       2.5-pro's cost / latency is overkill — e.g. mid-length videos that
#       need careful reasoning over the visual content.
#
#   gemini-2.5-pro
#       Highest quality, longest context (~2M tokens). Pick this for very
#       long videos or videos that need deep reasoning + extensive output.
#
# gemini-2.0-flash was dropped from this list: it's strictly older than the
# 2.5 / 3.1 tiers and the free-tier quota is gone on most accounts.
# gemini-1.5-pro was dropped earlier — the v1beta API now returns 404 for it.
VALID_MODELS = (
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
)

# File-API poll cadence + ceiling. Gemini's processing time scales with video
# length; 10 minutes covers anything we'd reasonably ship to the API.
UPLOAD_POLL_INTERVAL = 2.0
UPLOAD_TIMEOUT_SECONDS = 600.0


def load_api_key() -> str | None:
    """Return GEMINI_API_KEY from env or ~/.config/watch/.env (or cwd .env).

    Mirrors the dotenv pattern in scripts/whisper.py::load_api_key so the three
    backends (Groq, OpenAI, Gemini) all read keys the same way.
    """
    value = os.environ.get("GEMINI_API_KEY")
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
                if key.strip() != "GEMINI_API_KEY":
                    continue
                raw = raw.strip()
                if len(raw) >= 2 and raw[0] in ('"', "'") and raw[-1] == raw[0]:
                    raw = raw[1:-1]
                if raw:
                    return raw
        except OSError:
            continue
    return None


def is_youtube_url(source: str) -> bool:
    """True if source is a YouTube watch / youtu.be URL.

    Used by watch.py to decide whether to download with yt-dlp first or pass
    the URL directly to Gemini's native YouTube ingestion.
    """
    if not source:
        return False
    parsed = urlparse(source)
    if parsed.scheme not in ("http", "https"):
        return False
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if host.startswith("m."):
        host = host[2:]
    return host in ("youtube.com", "youtu.be")


def _state_name(file_obj) -> str:
    """Return the file's processing state as a plain string.

    The Files API returns an enum (`FileState.ACTIVE`) in current SDKs but
    older / mocked versions sometimes return the bare string. Normalize.
    """
    state = getattr(file_obj, "state", None)
    if state is None:
        return ""
    if hasattr(state, "name"):
        return state.name
    return str(state).rsplit(".", 1)[-1]


def _upload_and_wait(client, file_path: Path):
    """Upload a local video and block until Gemini finishes processing it.

    Raises SystemExit with a clear message if the upload fails, the poll
    fails, the file ends up FAILED, or it stays in PROCESSING past the
    timeout.
    """
    print(
        f"[watch] uploading {file_path.name} ({file_path.stat().st_size / 1024 / 1024:.1f} MB) "
        "to Gemini Files API…",
        file=sys.stderr,
    )
    try:
        uploaded = client.files.upload(file=str(file_path))
    except Exception as exc:
        raise SystemExit(
            f"Gemini Files API upload failed: {type(exc).__name__}: {exc}"
        )

    deadline = time.monotonic() + UPLOAD_TIMEOUT_SECONDS
    state = _state_name(uploaded)
    if state == "PROCESSING":
        print(
            f"[watch] uploaded as {uploaded.name} — waiting for Gemini to finish processing…",
            file=sys.stderr,
        )

    while state == "PROCESSING":
        if time.monotonic() > deadline:
            raise SystemExit(
                f"Gemini upload still PROCESSING after {UPLOAD_TIMEOUT_SECONDS:.0f}s — aborting"
            )
        time.sleep(UPLOAD_POLL_INTERVAL)
        try:
            uploaded = client.files.get(name=uploaded.name)
        except Exception as exc:
            raise SystemExit(
                f"Gemini Files API poll failed: {type(exc).__name__}: {exc}"
            )
        state = _state_name(uploaded)

    if state != "ACTIVE":
        raise SystemExit(
            f"Gemini file ended in state={state!r}, expected ACTIVE"
        )

    return uploaded


def generate_with_video(
    source: str,
    question: str,
    model_name: str = DEFAULT_MODEL,
    is_youtube: bool = False,
) -> str:
    """Send `source` (local path or YouTube URL) + question to Gemini.

    Returns the model's text response. Raises SystemExit on any failure
    (missing key, missing SDK, upload/poll/generation error, empty response).
    """
    if not question or not question.strip():
        raise SystemExit(
            "--backend gemini requires a question. Pass it as the trailing positional "
            'argument, e.g. `watch.py video.mp4 --backend gemini "Describe this video"`'
        )

    if model_name not in VALID_MODELS:
        raise SystemExit(
            f"Unknown --gemini-model: {model_name!r}. "
            f"Choose from: {', '.join(VALID_MODELS)}"
        )

    api_key = load_api_key()
    if not api_key:
        raise SystemExit(
            "GEMINI_API_KEY is not set. Add it to ~/.config/watch/.env or "
            "export GEMINI_API_KEY in your environment. "
            "Get a key at https://aistudio.google.com/apikey"
        )

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise SystemExit(
            "google-genai SDK is not installed. Run: pip install google-genai\n"
            f"(import error: {exc})"
        )

    try:
        client = genai.Client(api_key=api_key)
    except Exception as exc:
        raise SystemExit(
            f"Failed to construct Gemini client: {type(exc).__name__}: {exc}"
        )

    if is_youtube:
        print(
            f"[watch] sending YouTube URL to Gemini ({model_name}) — "
            "no download, processed server-side…",
            file=sys.stderr,
        )
        video_part = types.Part(
            file_data=types.FileData(file_uri=source, mime_type="video/*")
        )
        contents = [
            types.Content(
                role="user",
                parts=[video_part, types.Part(text=question)],
            )
        ]
    else:
        local_path = Path(source).expanduser().resolve()
        if not local_path.exists():
            raise SystemExit(f"video file not found: {local_path}")
        uploaded = _upload_and_wait(client, local_path)
        print(
            f"[watch] sending video + question to Gemini ({model_name})…",
            file=sys.stderr,
        )
        contents = [uploaded, question]

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
        )
    except Exception as exc:
        raise SystemExit(
            f"Gemini generate_content failed: {type(exc).__name__}: {exc}"
        )

    text = getattr(response, "text", None) or ""
    if not text:
        # Surface finish_reason / safety info when the model returned nothing
        # textual. Helps distinguish "blocked by safety" from "model went silent".
        finish_info = ""
        try:
            cand = response.candidates[0]
            finish_info = f" (finish_reason={getattr(cand, 'finish_reason', '?')})"
        except (AttributeError, IndexError, TypeError):
            pass
        raise SystemExit(f"Gemini returned an empty response{finish_info}")

    return text


TRANSCRIPT_PROMPT = (
    "Transcribe the spoken audio of this video verbatim in its original language. "
    "Output only lines of the form [MM:SS] text (use [H:MM:SS] past one hour), "
    "one line per sentence, covering the entire video to the end. No commentary."
)

_TS_LINE = re.compile(r"^\s*\[(\d{1,2}:)?\d{1,2}:\d{2}\]\s*(.+)$")
_TS_BRACKET = re.compile(r"\[([\d:]+)\]")
_LAST_SEGMENT_SECONDS = 5.0


def parse_timestamped_lines(text: str) -> list[dict]:
    """Parse `[MM:SS] text` / `[H:MM:SS] text` lines into transcript segments.

    Non-matching lines are ignored. A line whose start is earlier than the
    previous kept start is dropped; a line with the
    same start is merged into the previous segment's text.
    Each `end` is the next segment's `start`; the last is `start + 5.0`.
    """
    segs: list[dict] = []
    for line in (text or "").splitlines():
        m = _TS_LINE.match(line)
        if not m:
            continue
        stamp = _TS_BRACKET.search(line).group(1)
        start = 0.0
        for part in stamp.split(":"):
            start = start * 60 + int(part)
        body = m.group(2).strip()
        if not body:
            continue
        if segs and start < segs[-1]["start"]:
            continue
        if segs and start == segs[-1]["start"]:
            segs[-1]["text"] += " " + body
            continue
        segs.append({"start": float(start), "end": float(start), "text": body})
    for cur, nxt in zip(segs, segs[1:]):
        cur["end"] = nxt["start"]
    if segs:
        segs[-1]["end"] = segs[-1]["start"] + _LAST_SEGMENT_SECONDS
    return segs


def transcribe_youtube(
    url: str,
    model_name: str = DEFAULT_MODEL,
    api_key: str | None = None,
    client=None,
) -> tuple[list[dict], str] | None:
    """Transcribe a YouTube video by handing its URL to Gemini (server-side fetch).

    Never raises: returns (segments, model_name), or None on a missing SDK/key,
    client/API error, blocked/empty reply, or a reply with no timestamped lines.
    A MAX_TOKENS finish keeps the partial segments.
    """
    try:
        from google.genai import types

        if client is None:
            from google import genai

            key = api_key or load_api_key()
            if not key:
                return None
            client = genai.Client(api_key=key)

        print(
            f"[watch] requesting transcript from Gemini ({model_name}) via YouTube URL…",
            file=sys.stderr,
        )
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part(file_data=types.FileData(file_uri=url, mime_type="video/*")),
                    types.Part(text=TRANSCRIPT_PROMPT),
                ],
            )
        ]
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW
            ),
        )
        try:
            finish = str(getattr(response.candidates[0], "finish_reason", "") or "")
        except (AttributeError, IndexError, TypeError):
            finish = ""
        if "MAX_TOKENS" in finish:
            print(
                "[watch] Gemini hit MAX_TOKENS — transcript may be truncated",
                file=sys.stderr,
            )
        segments = parse_timestamped_lines(getattr(response, "text", None) or "")
        return (segments, model_name) if segments else None
    except Exception as exc:  # never raise: callers fall through the chain
        print(
            f"[watch] Gemini transcript failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return None


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "usage: gemini.py <video-or-url> <question> [model]",
            file=sys.stderr,
        )
        raise SystemExit(2)
    cli_source = sys.argv[1]
    cli_question = sys.argv[2]
    cli_model = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_MODEL
    out = generate_with_video(
        cli_source,
        cli_question,
        model_name=cli_model,
        is_youtube=is_youtube_url(cli_source),
    )
    print(out)
