#!/usr/bin/env python3
"""Stable source identity for /watch jobs.

``canonical_source`` collapses the many spellings of one video URL (youtu.be
short links, shorts/live/embed paths, tracking params, fragments) into a single
string so the same video always maps to the same job. Pure stdlib, pure
function, never raises: on any parse problem it returns ``source.strip()``.
"""
from __future__ import annotations

import re
from urllib.parse import parse_qs, urlsplit, urlunsplit

YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}
YOUTU_BE = "youtu.be"
_VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}")
_ID_PATH_PREFIXES = ("shorts", "live", "embed")


def _youtube_id(host: str, path: str, query: str) -> str | None:
    """Return the 11-char video id for a YouTube URL, or None if there isn't one."""
    segments = [s for s in path.split("/") if s]
    candidate = None
    if host == YOUTU_BE:
        candidate = segments[0] if segments else None
    elif host in YOUTUBE_HOSTS:
        if path.rstrip("/") == "/watch":
            values = parse_qs(query).get("v")
            candidate = values[0] if values else None
        elif len(segments) >= 2 and segments[0] in _ID_PATH_PREFIXES:
            candidate = segments[1]
    if candidate and _VIDEO_ID.fullmatch(candidate):
        return candidate
    return None


def canonical_source(source: str) -> str:
    """Canonical form of a video source (URL or local path)."""
    stripped = source.strip()
    try:
        parts = urlsplit(stripped)
        if parts.scheme.lower() not in ("http", "https") or not parts.netloc:
            return stripped
        host = (parts.hostname or "").lower()
        video_id = _youtube_id(host, parts.path, parts.query)
        if video_id:
            return f"https://www.youtube.com/watch?v={video_id}"
        userinfo, at, hostport = parts.netloc.rpartition("@")
        port = f":{parts.port}" if parts.port else ""
        netloc = f"{userinfo}{at}{host}{port}"
        return urlunsplit((parts.scheme.lower(), netloc, parts.path, parts.query, ""))
    except ValueError:
        return stripped
