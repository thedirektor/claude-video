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
# H.264 first: OpenCV's bundled ffmpeg has no software AV1 decoder, so an AV1 download leaves
# camera / look / typography / motion "not determined" (ensure_media transcodes as a fallback)
FETCH_FORMAT = "bv*[vcodec^=avc1][height<=1080]+ba/bv*[height<=1080]+ba/b"
LOW_RES_HEIGHT = 240


def pin(store_dir: Path, video_key: str) -> bool:
    """Exempt a video's media from retention. Returns False (no marker) when the media is above
    ``WATCH_BREAKDOWN_MEDIA_MAX_MB``: retention would evict it anyway, so "pinned" would be false."""
    vdir = store.video_dir(Path(store_dir), video_key)
    vdir.mkdir(parents=True, exist_ok=True)
    cap = store._breakdown_media_cap_bytes()
    if any(m.stat().st_size > cap for m in vdir.glob("media.*") if m.is_file()):
        return False
    (vdir / PINNED_MARKER).touch()
    return True


def is_pinned(store_dir: Path, video_key: str) -> bool:
    return (store.video_dir(Path(store_dir), video_key) / PINNED_MARKER).exists()


def _rational_to_float(value: str) -> float:
    try:
        num, _, den = (value or "").partition("/")
        den_f = float(den) if den else 1.0
        return float(num) / den_f if den_f else 0.0
    except ValueError:
        return 0.0


STANDARD_RATES = ("24000/1001", "24/1", "25/1", "30000/1001", "30/1", "48/1", "50/1", "60000/1001", "60/1",
                  "120/1")
SNAP_TOL = 0.005  # relative distance to a standard rate within which a measured rate is snapped
EXACT_TOL = 1e-4  # closer than this is the same rate written differently: no note


def snap_fps(rate: str) -> tuple[str, str | None, bool]:
    """``(fps_rational, note, off_standard)``.

    ffprobe's average rate of a variable-frame-rate file (``1784000/59623``) drifts a timeline
    over minutes. A rate within 0.5% of a standard one is snapped to it, with a note; further
    away it is kept as measured, flagged ``off_standard`` so timeline confidence is lowered."""
    val = _rational_to_float(rate)
    if val <= 0:
        return rate, None, False
    best = min(STANDARD_RATES, key=lambda r: abs(_rational_to_float(r) - val))
    std = _rational_to_float(best)
    rel = abs(std - val) / std
    if rel <= EXACT_TOL:
        return best, None, False
    if rel <= SNAP_TOL:
        return best, f"variable/non-standard frame rate {rate} ({val:.3f} fps) snapped to {best}", False
    return rate, f"non-standard frame rate {rate} ({val:.3f} fps) kept as measured: timeline timing may drift", True


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
    snapped, fps_note, off_standard = snap_fps(rate)
    # analysis (frame index -> time) uses the measured rate; only timelines use the snapped one
    return {
        "fps": _rational_to_float(rate),
        "fps_rational": rate,
        "timeline_fps_rational": snapped,
        "fps_note": fps_note,
        "fps_offstandard": off_standard,
        "duration": duration,
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
        "codec": video.get("codec_name") or "",
    }


def default_fetch(url: str, vdir: Path) -> dict | None:
    """Download ``url`` (<=1080p) via the plugin's yt-dlp wrapper.

    Returns ``{"video_path", "via_fallback_proxy"}`` (a fetch may also return a bare path)."""
    import download  # deferred: heavy and only needed on a refetch

    tmp = Path(vdir) / ".fetch"
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        res = download.download_url(url, tmp, fmt=FETCH_FORMAT)
        if res.get("video_path"):
            return res
        shutil.rmtree(tmp, ignore_errors=True)
        return None
    except BaseException as exc:  # yt-dlp failures surface as SystemExit
        shutil.rmtree(tmp, ignore_errors=True)  # partial downloads are scratch, not media
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
        "path": None, "fps": 0.0, "fps_rational": "0/1", "timeline_fps_rational": "0/1",
        "duration": float(duration or 0.0), "width": width, "height": height,
        "has_audio": False, "low_resolution": True, "source": "frames",
    }


def _cv_decodable(path: Path) -> bool:
    """Can OpenCV read a frame of ``path``? True when OpenCV is absent (nothing to protect)."""
    try:
        import cv2  # noqa: PLC0415
    except Exception:  # ImportError or a broken wheel: the cv2-based fields degrade on their own
        return True
    cap = cv2.VideoCapture(str(path))
    try:
        return bool(cap.isOpened() and cap.read()[0])
    finally:
        cap.release()


def _make_decodable(path: Path, info: dict) -> tuple[Path, dict, str | None]:
    """Transcode media OpenCV cannot decode (e.g. AV1) to H.264 in place, once; it stays the
    breakdown's pinned media. Returns ``(path, info, note)``; on failure the original is kept."""
    if _cv_decodable(path):
        return path, info, None
    codec = info.get("codec") or "unknown"
    tmp = path.with_name("media.transcode.mp4")
    try:
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(path),
                        "-map", "0:v:0", "-map", "0:a:0?", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(tmp)],
                       check=True, capture_output=True, timeout=7200)
        new_info = probe(tmp)
        if not new_info or not _cv_decodable(tmp):
            raise RuntimeError("transcoded copy is not readable")
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        tmp.unlink(missing_ok=True)
        return path, info, (f"media codec {codec} cannot be decoded by OpenCV and transcoding failed "
                            f"({type(exc).__name__}): camera, look, typography and motion not determined")
    dest = path.with_name("media.mp4")
    path.unlink(missing_ok=True)
    tmp.replace(dest)
    return dest, new_info, f"media codec {codec} transcoded to H.264 for frame analysis (timing unchanged)"


def _result(path: Path, info: dict, source: str) -> dict:
    path, info, note = _make_decodable(path, info)
    if note:
        info = {**info, "media_note": note}
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
            print(f"[watch] breakdown media refetch failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            got = None
        proxied = False
        if isinstance(got, dict):
            proxied = bool(got.get("via_fallback_proxy"))
            got = got.get("video_path")
        if got and Path(got).is_file():
            vdir.mkdir(parents=True, exist_ok=True)
            dest = _adopt(Path(got), vdir)
            info = probe(dest)
            if info:
                res = _result(dest, info, "fetched")
                if proxied:  # fallback proxy forces <=480p
                    res["low_resolution"] = True
                if dest.stat().st_size > store._breakdown_media_cap_bytes():
                    res["over_cap"] = True
                    res["media_note"] = ("fetched media exceeds WATCH_BREAKDOWN_MEDIA_MAX_MB: used for this "
                                         "run but not pinned, so retention may evict it and a later run refetch it")
                return res
    return _frames_only(row["duration"] if row else 0.0, vdir)
