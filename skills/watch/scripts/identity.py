#!/usr/bin/env python3
"""Stable source identity for /watch jobs.

``canonical_source`` collapses the many spellings of one video URL (youtu.be
short links, shorts/live/embed paths, tracking params, fragments) into a single
string so the same video always maps to the same job. Pure stdlib, pure
function, never raises: on any parse problem it returns ``source.strip()``.
"""
from __future__ import annotations

import hashlib
import os
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
        scheme = parts.scheme.lower()
        port_num = parts.port  # ValueError if out of range -> caught below
        default = {"http": 80, "https": 443}[scheme]
        port = f":{port_num}" if port_num is not None and port_num != default else ""
        if ":" in host:  # IPv6 literal: hostname strips the brackets
            host = f"[{host}]"
        netloc = f"{userinfo}{at}{host}{port}"
        return urlunsplit((scheme, netloc, parts.path, parts.query, ""))
    except ValueError:
        return stripped


def _sha16(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _file_sha16(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def video_key(source: str, info: dict | None = None) -> str:
    """Stable identity of one video across spellings and runs. Never raises.

    file:<sha256(content)[:16]> for an existing local file, yt:<id> for
    YouTube, <extractor>:<id> when yt-dlp info supplies both, else
    url:<sha256(canonical_source)[:16]>.
    """
    try:
        stripped = (source or "").strip()
        try:
            if stripped and os.path.isfile(stripped):
                return "file:" + _file_sha16(stripped)
        except OSError:
            pass
        canon = canonical_source(stripped)
        m = re.fullmatch(r"https://www\.youtube\.com/watch\?v=([A-Za-z0-9_-]{11})", canon)
        if m:
            return f"yt:{m.group(1)}"
        if isinstance(info, dict):
            ek, vid = info.get("extractor_key"), info.get("id")
            if ek and vid:
                return f"{str(ek).lower()}:{vid}"
        return "url:" + _sha16(canon.encode("utf-8", "surrogateescape"))
    except Exception:
        return "url:" + _sha16(repr(source).encode("utf-8", "replace"))


def store_dirname(video_key: str) -> str:
    """Windows-safe directory name for a video_key."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", video_key)
