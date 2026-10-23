"""Run bundles: move one video's evidence from a local machine into the central memory.

``make_bundle`` zips what the local index + store hold for a video — ``run.json``
(metadata, segments, frames with their OCR, run notes), the stored frame images and
``report.md`` — with an ``origin_ref`` of ``<host>:<video_key>:v<evidence_version>``.
``import_bundle`` (on the central service) validates it (size cap, no path escape),
merges it as ``origin=import`` under the normal rules and is idempotent: the same
bundle twice is a no-op, notes already present are not duplicated.
"""
from __future__ import annotations

import json
import shutil
import socket
import tempfile
import zipfile
from pathlib import Path

from config import memory_paths, setting

from . import index, store, writer

MAX_BUNDLE_BYTES = 200 * 1024 * 1024
DERIVED_NOTE_KINDS = {"chapter", "visual"}   # rebuilt / re-fetched centrally, never shipped


class BundleError(ValueError):
    pass


def _max_bytes() -> int:
    try:
        return int(float(setting("WATCH_IMPORT_MAX_MB") or 0) * 1024 * 1024) or MAX_BUNDLE_BYTES
    except ValueError:
        return MAX_BUNDLE_BYTES


def make_bundle(conn, key: str, out_path: Path, *, store_dir=None, host: str | None = None,
                library_saved: bool = False) -> Path:
    v = conn.execute("SELECT title, url, source, duration, channel, upload_date, transcript_source, store_path, "
                     "evidence_version FROM videos WHERE video_key = ?", (key,)).fetchone()
    if v is None:
        raise BundleError(f"not indexed: {key}")
    root = Path(store_dir or memory_paths()["store_dir"])
    last = conn.execute("SELECT detail, backend FROM runs WHERE video_key = ? ORDER BY run_id DESC LIMIT 1",
                        (key,)).fetchone()
    ocr_by_frame = {fid: text for fid, text in conn.execute(
        "SELECT frame_id, text FROM ocr_blocks WHERE video_key = ? AND frame_id IS NOT NULL", (key,))}
    frames, files = [], []
    for fid, t, kind, sp in conn.execute("SELECT id, t, kind, store_path FROM frames WHERE video_key = ? ORDER BY t",
                                         (key,)):
        path = root / sp if sp else None
        entry = {"t": t, "kind": kind, "ocr": ocr_by_frame.get(fid), "file": None}
        if path and path.is_file():
            entry["file"] = f"frames/{path.name}"
            files.append((path, entry["file"]))
        frames.append(entry)
    run = {
        "video_key": key, "source": v[2] or v[1] or key, "title": v[0], "url": v[1], "duration": v[3],
        "channel": v[4], "upload_date": v[5], "transcript_source": v[6],
        "detail": last[0] if last else "balanced", "backend": last[1] if last else "claude",
        "segments": [{"start": s, "end": e, "speaker": sp, "text": t} for s, e, sp, t in conn.execute(
            'SELECT start, "end", speaker, text FROM segments WHERE video_key = ? ORDER BY start', (key,))],
        "frames": frames,
        "notes": [{"kind": k, "t_start": a, "t_end": b, "text": t} for k, a, b, t in conn.execute(
            "SELECT kind, t_start, t_end, text FROM notes WHERE video_key = ? ORDER BY id", (key,))
            if k not in DERIVED_NOTE_KINDS],
        "origin_ref": f"{host or socket.gethostname()}:{key}:v{v[8]}",
        "library_saved": bool(library_saved),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("run.json", json.dumps(run, ensure_ascii=False))
        for path, arc in files:
            z.write(path, arc)
        report = root / (v[7] or "") / "report.md" if v[7] else None
        if report and report.is_file():
            z.write(report, "report.md")
        bdir = root / v[7] / "breakdown" if v[7] else store.video_dir(root, key) / "breakdown"
        if bdir.is_dir():
            for f in sorted(bdir.iterdir()):
                if f.is_file():
                    z.write(f, f"breakdown/{f.name}")
    return out_path


def _safe_extract(zip_path: Path, dest: Path) -> None:
    if zip_path.stat().st_size > _max_bytes():
        raise BundleError(f"bundle larger than {_max_bytes() // (1024 * 1024)} MB")
    if not zipfile.is_zipfile(zip_path):
        raise BundleError("not a zip bundle")
    with zipfile.ZipFile(zip_path) as z:
        total = 0
        for m in z.infolist():
            name = m.filename
            parts = Path(name).parts
            if name.startswith(("/", "\\")) or ".." in parts or (parts and ":" in parts[0]):
                raise BundleError(f"unsafe path in bundle: {name}")
            total += m.file_size
            if total > _max_bytes() * 3:
                raise BundleError("bundle expands beyond the size cap")
        z.extractall(dest)


def import_bundle(zip_path, *, embedder: object = "auto") -> dict:
    """Merge a bundle into this machine's memory as origin=import."""
    zip_path = Path(zip_path)
    with tempfile.TemporaryDirectory(prefix="watch-import-") as tmp:
        root = Path(tmp)
        _safe_extract(zip_path, root)
        try:
            run = json.loads((root / "run.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise BundleError(f"bad run.json: {exc}") from exc
        if not run.get("video_key") or not run.get("origin_ref"):
            raise BundleError("run.json needs video_key and origin_ref")
        conn = index.connect(memory_paths()["index_db"])
        try:
            have = {(k, t) for k, t in conn.execute("SELECT kind, text FROM notes WHERE video_key = ?",
                                                    (run["video_key"],))}
        finally:
            conn.close()
        frames = []
        for f in run.get("frames") or []:
            p = root / f["file"] if f.get("file") else None
            frames.append({"t": f["t"], "kind": f.get("kind") or "scene", "ocr": f.get("ocr"),
                           "path": str(p) if p and p.is_file() else None})
        report = root / "report.md"
        record = index.RunRecord(
            video_key=run["video_key"], source=run.get("source") or run["video_key"], origin="import",
            backend=run.get("backend") or "claude", detail=run.get("detail") or "balanced",
            title=run.get("title"), url=run.get("url"), duration=run.get("duration"), channel=run.get("channel"),
            upload_date=run.get("upload_date"), transcript_source=run.get("transcript_source"),
            segments=[s for s in run.get("segments") or [] if (s.get("text") or "").strip()], frames=frames,
            notes=[n for n in run.get("notes") or [] if (n.get("kind"), n.get("text")) not in have],
            report_path=str(report) if report.is_file() else None, origin_ref=run["origin_ref"])
        result = writer.remember(record, archive=True, embedder=embedder)
        if result is None:
            raise BundleError("merge failed (see service log)")
        shipped = sorted((root / "breakdown").glob("*")) if (root / "breakdown").is_dir() else []
        if shipped:
            dest = store.video_dir(Path(memory_paths()["store_dir"]), run["video_key"]) / "breakdown"
            dest.mkdir(parents=True, exist_ok=True)
            for f in shipped:
                if f.is_file():
                    shutil.copyfile(f, dest / f.name)
        return {**result, "skipped": result.get("run_id") is None, "title": run.get("title"),
                "library_saved": bool(run.get("library_saved"))}
