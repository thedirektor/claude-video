"""Media acquisition and pinning for breakdowns.

A breakdown needs the original media (cuts, audio, motion), but the memory
store may have evicted it. ``ensure_media`` returns stored media, refetches it,
or reports that only stored frames remain (``source == "frames"``,
``low_resolution``) so callers mark cut-dependent fields "not determined".
Pinning (``<vdir>/.pinned``) exempts a video's media from retention.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from config import memory_paths
from memory import store

PINNED_MARKER = store.PINNED_MARKER
FETCH_FORMAT = "bv*[height<=1080]+ba/b"
LOW_RES_HEIGHT = 240


def pin(store_dir: Path, video_key: str) -> None:
    vdir = store.video_dir(Path(store_dir), video_key)
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / PINNED_MARKER).touch()


def is_pinned(store_dir: Path, video_key: str) -> bool:
    return (store.video_dir(Path(store_dir), video_key) / PINNED_MARKER).exists()


def _rational_to_float(value: str) -> float:
    try:
        num, _, den = (value or "").partition("/")
        den_f = float(den) if den else 1.0
        return float(num) / den_f if den_f else 0.0
    except ValueError:
        return 0.0


def probe(path: Path) -> dict | None:
    """ffprobe summary of a media file, or None when it is unreadable."""
    try:
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
            capture_output=True, text=True, timeout=60)
        data = json.loads(res.stdout or "{}")
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        return None
    rate = video.get("avg_frame_rate")
    if not rate or rate.startswith("0/"):
        rate = video.get("r_frame_rate") or "0/1"
    try:
        duration = float(data.get("format", {}).get("duration") or video.get("duration") or 0.0)
    except ValueError:
        duration = 0.0
    return {
        "fps": _rational_to_float(rate),
        "fps_rational": rate,
        "duration": duration,
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
    }


def default_fetch(url: str, vdir: Path) -> Path | None:
    """Download ``url`` (<=1080p) via the plugin's yt-dlp wrapper; returns the file."""
    import download  # deferred: heavy and only needed on a refetch

    tmp = Path(vdir) / ".fetch"
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        res = download.download_url(url, tmp, fmt=FETCH_FORMAT)
        return Path(res["video_path"]) if res.get("video_path") else None
    except BaseException as exc:  # yt-dlp failures surface as SystemExit
        if isinstance(exc, KeyboardInterrupt):
            raise
        print(f"[watch] breakdown media refetch failed: {exc}", file=sys.stderr)
        return None


def _adopt(downloaded: Path, vdir: Path) -> Path:
    """Move a fetched file to ``<vdir>/media.<ext>`` and drop fetch scratch."""
    dest = vdir / ("media" + downloaded.suffix.lower())
    if downloaded.resolve() != dest.resolve():
        for old in vdir.glob("media.*"):
            old.unlink(missing_ok=True)
        shutil.move(str(downloaded), dest)
    shutil.rmtree(vdir / ".fetch", ignore_errors=True)
    return dest


def _frames_only(duration, vdir: Path) -> dict:
    width = height = 0
    frame = next(iter(sorted((vdir / "frames").glob("*.jpg"))), None) if (vdir / "frames").is_dir() else None
    if frame is not None:
        info = probe(frame)
        if info:
            width, height = info["width"], info["height"]
    return {
        "path": None, "fps": 0.0, "fps_rational": "0/1",
        "duration": float(duration or 0.0), "width": width, "height": height,
        "has_audio": False, "low_resolution": True, "source": "frames",
    }


def _result(path: Path, info: dict, source: str) -> dict:
    return {"path": str(path), **info, "low_resolution": info["height"] < LOW_RES_HEIGHT, "source": source}


def ensure_media(conn, video_key: str, store_dir: Path | None = None, fetch=None) -> dict:
    root = Path(store_dir) if store_dir is not None else memory_paths()["store_dir"]
    vdir = store.video_dir(root, video_key)
    row = conn.execute("SELECT url, source, duration FROM videos WHERE video_key = ?", (video_key,)).fetchone()

    for stored in sorted(vdir.glob("media.*")) if vdir.is_dir() else []:
        info = probe(stored)
        if info:
            return _result(stored, info, "stored")

    url = (row["url"] or row["source"]) if row else None
    if url and url.startswith(("http://", "https://")):
        fetcher = fetch or default_fetch
        try:
            got = fetcher(url, vdir)
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                raise
            got = None
        if got and Path(got).is_file():
            vdir.mkdir(parents=True, exist_ok=True)
            dest = _adopt(Path(got), vdir)
            info = probe(dest)
            if info:
                return _result(dest, info, "fetched")
    return _frames_only(row["duration"] if row else 0.0, vdir)
