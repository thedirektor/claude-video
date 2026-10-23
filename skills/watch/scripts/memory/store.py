"""Persistent per-video store: artifact copies, LRU retention, Qdrant inbox.

Layout under the store root, one directory per video
(``identity.store_dirname(video_key)``)::

    report.md  transcript.json  ocr.json  notes.json  meta.json
    frames/<t_ms>.jpg   media.<ext> (only when small enough / opted in)

The store is written by COPYING (never moving). JSON artifacts are rewritten
from the index, so they always show the merged evidence set. Retention only
deletes files (media first, then frames) and nulls ``frames.store_path``; index
rows are never evicted. stdlib only.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import identity
import library
from config import setting
from memory import index as _index

_MB = 1024 * 1024
_GB = 1024 * _MB


def video_dir(store_dir: Path, video_key: str) -> Path:
    return Path(store_dir) / identity.store_dirname(video_key)


def keep_media(size_bytes: int) -> bool:
    if (setting("WATCH_STORE_KEEP_MEDIA", "") or "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    try:
        cap_mb = float(setting("WATCH_STORE_MEDIA_MAX_MB", "300") or "300")
    except ValueError:
        cap_mb = 300.0
    return size_bytes <= cap_mb * _MB


def _plugin_version() -> str | None:
    try:
        manifest = Path(__file__).resolve().parents[4] / ".claude-plugin" / "plugin.json"
        return json.loads(manifest.read_text(encoding="utf-8")).get("version")
    except (OSError, ValueError, IndexError):
        return None


def _dump(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def _copy_frames(conn, root: Path, vdir: Path, new_frames) -> None:
    if not new_frames:
        return
    fdir = vdir / "frames"
    fdir.mkdir(parents=True, exist_ok=True)
    for frame_id, src in new_frames:
        src_path = Path(src)
        if not src_path.is_file():
            continue
        t = conn.execute("SELECT t FROM frames WHERE id = ?", (frame_id,)).fetchone()
        if t is None:
            continue
        t_ms = round(float(t[0]) * 1000)
        dest = fdir / f"{t_ms}.jpg"
        n = 1
        while dest.exists():
            dest = fdir / f"{t_ms}_{n}.jpg"
            n += 1
        shutil.copy2(src_path, dest)
        conn.execute(
            "UPDATE frames SET store_path = ? WHERE id = ?",
            (dest.relative_to(root).as_posix(), frame_id),
        )


def write_run(conn, store_dir: Path, run, result) -> Path:
    """Persist one merged run's artifacts; returns the video's store dir."""
    root = Path(store_dir)
    key = run.video_key
    vdir = video_dir(root, key)
    vdir.mkdir(parents=True, exist_ok=True)

    if run.report_path and Path(run.report_path).is_file():
        shutil.copy2(run.report_path, vdir / "report.md")

    if run.media_path and Path(run.media_path).is_file():
        mp = Path(run.media_path)
        if keep_media(mp.stat().st_size):
            for old in vdir.glob("media.*"):
                old.unlink(missing_ok=True)
            shutil.copy2(mp, vdir / ("media" + mp.suffix.lower()))

    with _index._tx(conn):
        _copy_frames(conn, root, vdir, result.new_frames)
        conn.execute(
            "UPDATE videos SET store_path = ? WHERE video_key = ?",
            (vdir.relative_to(root).as_posix(), key),
        )
        video = conn.execute("SELECT * FROM videos WHERE video_key = ?", (key,)).fetchone()
        segments = [
            {"start": r["start"], "end": r["end"], "speaker": r["speaker"], "text": r["text"], "source": r["source"]}
            for r in conn.execute(
                'SELECT start, "end", speaker, text, source FROM segments WHERE video_key = ? ORDER BY start, id',
                (key,),
            )
        ]
        ocr = [
            {"t": r["t"], "frame": r["frame_id"], "text": r["text"]}
            for r in conn.execute(
                "SELECT t, frame_id, text FROM ocr_blocks WHERE video_key = ? ORDER BY t, id", (key,)
            )
        ]
        notes = [
            {"kind": r["kind"], "t_start": r["t_start"], "t_end": r["t_end"], "text": r["text"]}
            for r in conn.execute(
                "SELECT kind, t_start, t_end, text FROM notes WHERE video_key = ? ORDER BY id", (key,)
            )
        ]
        last_run = conn.execute(
            "SELECT detail, backend FROM runs WHERE video_key = ? ORDER BY run_id DESC LIMIT 1", (key,)
        ).fetchone()

    _dump(vdir / "transcript.json", segments)
    _dump(vdir / "ocr.json", ocr)
    _dump(vdir / "notes.json", notes)
    _dump(vdir / "meta.json", {
        "video_key": key,
        "title": video["title"],
        "url": video["url"],
        "duration": video["duration"],
        "channel": video["channel"],
        "upload_date": video["upload_date"],
        "detail": last_run["detail"] if last_run else run.detail,
        "backend": last_run["backend"] if last_run else run.backend,
        "plugin_version": _plugin_version(),
        "library": {
            "immich_album_id": video["immich_album_id"],
            "nextcloud_path": video["nextcloud_path"],
            "paperless_doc_id": video["paperless_doc_id"],
            "paperless_task_id": video["paperless_task_id"],
        },
        "evidence_version": video["evidence_version"],
    })
    return vdir


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _tree_bytes(root: Path) -> int:
    return sum(_size(p) for p in root.rglob("*") if p.is_file()) if root.is_dir() else 0


PINNED_MARKER = ".pinned"


def _breakdown_media_cap_bytes() -> int:
    try:
        return int(float(setting("WATCH_BREAKDOWN_MEDIA_MAX_MB", "2048") or "2048") * _MB)
    except ValueError:
        return 2048 * _MB


def enforce_retention(conn, store_dir: Path, max_bytes: int | None = None,
                      protect: frozenset[str] = frozenset()) -> dict:
    """Evict media, then frames, in least-recently-accessed order until the
    store fits ``max_bytes`` (default WATCH_STORE_MAX_GB). Never touches
    ``protect`` videos or any DB row other than ``frames.store_path``."""
    root = Path(store_dir)
    if max_bytes is None:
        try:
            gb = float(setting("WATCH_STORE_MAX_GB", "20") or "20")
        except ValueError:
            gb = 20.0
        max_bytes = int(gb * _GB)
    out = {"freed": 0, "media_removed": 0, "frames_removed": 0}
    total = _tree_bytes(root)
    if total <= max_bytes:
        return out

    keys = [r[0] for r in conn.execute(
        "SELECT video_key FROM videos ORDER BY COALESCE(last_access, '') ASC, video_key ASC")]
    keys = [k for k in keys if k not in protect]

    cap = _breakdown_media_cap_bytes()
    for key in keys:  # pass 1: media (pinned breakdown media is exempt unless over its own cap)
        if total <= max_bytes:
            return out
        vdir = video_dir(root, key)
        pinned = (vdir / PINNED_MARKER).exists()
        for media in sorted(vdir.glob("media.*")):
            size = _size(media)
            if pinned and size <= cap:
                continue
            media.unlink(missing_ok=True)
            total -= size
            out["freed"] += size
            out["media_removed"] += 1
            if total <= max_bytes:
                return out

    for key in keys:  # pass 2: frames
        if total <= max_bytes:
            break
        fdir = video_dir(root, key) / "frames"
        removed_any = False
        for f in sorted(fdir.glob("*.jpg")) if fdir.is_dir() else []:
            size = _size(f)
            f.unlink(missing_ok=True)
            total -= size
            out["freed"] += size
            out["frames_removed"] += 1
            removed_any = True
            if total <= max_bytes:
                break
        if removed_any:
            with _index._tx(conn):
                for r in conn.execute(
                    "SELECT id, store_path FROM frames WHERE video_key = ? AND store_path IS NOT NULL", (key,)
                ).fetchall():
                    if not (root / r["store_path"]).exists():
                        conn.execute("UPDATE frames SET store_path = NULL WHERE id = ?", (r["id"],))
    return out


def write_qdrant_inbox(inbox: Path, video_key: str, title: str | None, url: str | None,
                       date: str, report_text: str) -> Path:
    """Atomically drop ``<slug>-<video_key>.md`` into the Qdrant inbox. The
    inbox is server-only, so ``:`` in the key is legal in the file name."""
    inbox = Path(inbox)
    inbox.mkdir(parents=True, exist_ok=True)
    slug = library.sanitize(title or video_key).lower().replace(" ", "-")[:60]
    path = inbox / f"{slug}-{video_key}.md"
    tmp = inbox / f".{path.name}.tmp"
    front = (
        "---\n"
        f"video_key: {video_key}\n"
        f"url: {json.dumps(url, ensure_ascii=False)}\n"
        f"title: {json.dumps(title, ensure_ascii=False)}\n"
        f"date: {date}\n"
        "---\n\n"
    )
    tmp.write_text(front + report_text, encoding="utf-8")
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        os.chown(tmp, 1000, 1000)
    os.replace(tmp, path)
    return path
