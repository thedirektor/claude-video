#!/usr/bin/env python3
"""Fetch a YouTube transcript from TranscriptAPI (https://transcriptapi.com).

TranscriptAPI's servers handle YouTube's bot-gate / PO-token / nsig challenges,
so this works from datacenter IPs where yt-dlp caption fetch is blocked. Pure
stdlib HTTP; best-effort (any failure returns None so the caller falls back to
yt-dlp captions / Whisper).
"""
from __future__ import annotations

import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
from pathlib import Path
from urllib.request import Request, urlopen

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

API_BASE = "https://transcriptapi.com/api/v2"
MAX_ATTEMPTS = 3
RETRYABLE = {408, 429, 500, 503}


def is_youtube_url(source: str) -> bool:
    """True only for real YouTube hosts (not 'notyoutube.com' impostors)."""
    try:
        p = urllib.parse.urlparse(source)
    except ValueError:
        return False
    if p.scheme not in ("http", "https"):
        return False
    host = (p.netloc or "").lower().split(":")[0]
    return (
        host == "youtu.be"
        or host.endswith(".youtu.be")
        or host == "youtube.com"
        or host.endswith(".youtube.com")
    )


def load_api_key() -> str | None:
    """TRANSCRIPTAPI_API_KEY from env, then ~/.config/watch/.env, then ./.env."""
    def _from_env() -> str | None:
        v = os.environ.get("TRANSCRIPTAPI_API_KEY")
        return v.strip() if v else None

    def _from_dotenv(path: Path) -> str | None:
        if not path.exists():
            return None
        try:
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                if key.strip() != "TRANSCRIPTAPI_API_KEY":
                    continue
                value = value.strip()
                if len(value) >= 2 and value[0] in ('"', "'") and value[-1] == value[0]:
                    value = value[1:-1]
                return value or None
        except OSError:
            return None
        return None

    value = _from_env()
    if value:
        return value
    for candidate in (Path.home() / ".config" / "watch" / ".env", Path.cwd() / ".env"):
        value = _from_dotenv(candidate)
        if value:
            return value
    return None


def languages_from_sub_langs(sub_langs: str) -> str:
    """Map watch's regex --sub-langs to a TranscriptAPI priority list.

    TranscriptAPI ignores region and takes <=10 codes; `asr` = auto captions.
    'es,es-419,es-ES,en,en-US,en-GB,.*-orig' -> 'es,en,asr'. Drops regex tokens
    ('.*-orig'), 'all', and empties; region-strips ('es-419'->'es'); dedups;
    always appends 'asr' as the auto-generated fallback.
    """
    out: list[str] = []
    for tok in sub_langs.split(","):
        tok = tok.strip().lower()
        if not tok or tok == "all" or tok.endswith("-orig") or not tok[0].isalpha():
            continue
        base = tok.split("-")[0]
        if base and base not in out:
            out.append(base)
    if "asr" in out:
        return ",".join(out[:10])
    return ",".join(out[:9] + ["asr"])


def _retry_after(exc: urllib.error.HTTPError) -> float | None:
    header = exc.headers.get("Retry-After") if getattr(exc, "headers", None) else None
    if not header:
        return None
    try:
        return float(header)
    except ValueError:
        return None


def _get(path: str, params: dict) -> tuple[int, dict | None]:
    """GET with bounded retry on transient codes. Returns (status, json|None)."""
    api_key = params.pop("_api_key")
    url = f"{API_BASE}{path}?{urllib.parse.urlencode(params)}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "watch-skill/1.0 (+claude-code; python-urllib)",
        "Accept": "application/json",
    }
    context = ssl.create_default_context()
    for attempt in range(MAX_ATTEMPTS):
        request = Request(url, headers=headers, method="GET")
        try:
            with urlopen(request, timeout=60, context=context) as response:
                status = getattr(response, "status", 200)
                body = response.read().decode("utf-8", errors="replace")
            try:
                return status, json.loads(body or "{}")
            except (json.JSONDecodeError, ValueError):
                return 0, None
        except urllib.error.HTTPError as exc:
            if exc.code in RETRYABLE and attempt < MAX_ATTEMPTS - 1:
                time.sleep(_retry_after(exc) or 2.0 * (attempt + 1))
                continue
            try:
                detail = json.loads(exc.read().decode("utf-8", errors="replace") or "{}")
            except Exception:
                detail = None
            return exc.code, detail
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(2.0 * (attempt + 1))
                continue
            return 0, None  # network give-up → treated as failure by caller
    return 0, None


def fetch_transcript(
    video_url: str, api_key: str, language: str | None = None
) -> tuple[list[dict], str] | None:
    """Return (segments, resolved_language) or None on any failure (logged).

    Segments are {start, end, text}; end = start + duration. NEVER raises —
    TranscriptAPI is a best-effort source, so the caller can fall back.
    """
    params: dict = {
        "_api_key": api_key,
        "video_url": video_url,
        "format": "json",
        "include_timestamp": "true",
    }
    if language:
        params["language"] = language

    status, data = _get("/youtube/transcript", params)

    if status == 200 and isinstance(data, dict):
        segments: list[dict] = []
        for item in data.get("transcript") or []:
            if not isinstance(item, dict):
                continue
            text = (item.get("text") or "").strip()
            if not text:
                continue
            try:
                start = round(float(item.get("start") or 0.0), 2)
                duration = float(item.get("duration") or 0.0)
            except (TypeError, ValueError):
                continue
            segments.append({"start": start, "end": round(start + duration, 2), "text": text})
        if segments:
            return segments, str(data.get("language") or language or "")
        print("[watch] TranscriptAPI returned an empty transcript — falling back", file=sys.stderr)
        return None

    reason = ""
    if isinstance(data, dict):
        reason = str(data.get("detail") or "")
    if status == 401:
        msg = "invalid/missing TRANSCRIPTAPI_API_KEY"
    elif status == 402:
        msg = "no TranscriptAPI credits remaining (transcriptapi.com/billing)"
    elif status == 404:
        msg = f"no transcript available{': ' + reason if reason else ''}"
    elif status == 0:
        msg = "network error / no response"
    else:
        msg = f"HTTP {status}{': ' + reason if reason else ''}"
    print(f"[watch] TranscriptAPI unavailable ({msg}) — falling back to captions/Whisper",
          file=sys.stderr)
    return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: transcriptapi.py <youtube-url-or-id> [lang-csv]", file=sys.stderr)
        raise SystemExit(2)
    key = load_api_key()
    if not key:
        raise SystemExit("TRANSCRIPTAPI_API_KEY not set")
    lang = sys.argv[2] if len(sys.argv) > 2 else None
    res = fetch_transcript(sys.argv[1], key, language=lang)
    if res is None:
        raise SystemExit("no transcript")
    segs, resolved = res
    print(json.dumps({"language": resolved, "segments": segs}, ensure_ascii=False, indent=2))
