"""Backfill and rebuild of the video-memory index.

* ``backfill_paperless`` re-reads every saved report (Paperless tag ``watch``)
  into the index, so videos watched before memory existed become searchable.
* ``rebuild`` re-creates one video's index rows from its store directory
  (``transcript.json`` / ``ocr.json`` / ``notes.json`` / ``meta.json`` + frames).

stdlib only; HTTP goes through ``library._paperless_get`` unless a ``get_json``
callable is injected (tests).
"""
from __future__ import annotations

import json
import sys
import urllib.parse
from datetime import date
from pathlib import Path

import identity
import library
from config import setting
from memory import index, store
from memory.reportparse import parse_report

PAGE_PATH = "/api/documents/?tags__name__iexact=watch&page_size=100"


def _as_path(next_url: str, base: str) -> str:
    """A Paperless ``next`` link (absolute URL) as an API path+query."""
    parts = urllib.parse.urlsplit(next_url)
    if parts.scheme and parts.netloc:
        return parts.path + (f"?{parts.query}" if parts.query else "")
    return next_url


def _default_get_json(cfg: dict):
    base = (cfg.get("PAPERLESS_URL") or library.DEFAULT_PAPERLESS_URL).rstrip("/")
    token = cfg.get("PAPERLESS_TOKEN")
    if not token or not base:
        raise RuntimeError("PAPERLESS_TOKEN / PAPERLESS_URL is not set")
    return lambda path: library._paperless_get(base, token, path)


def _album_index(cfg: dict) -> dict:
    """Immich albums by name, fetched once per backfill."""
    base, key = (cfg.get("IMMICH_BASE_URL") or "").rstrip("/"), cfg.get("IMMICH_API_KEY")
    if not base or not key:
        return {}
    try:
        req = library.Request(f"{base}/api/albums", headers=library._immich_headers(key), method="GET")
        with library.urlopen(req, timeout=30, context=library._ssl_context()) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace") or "[]")
        return {a["albumName"]: a["id"] for a in data if isinstance(a, dict) and a.get("albumName") and a.get("id")}
    except Exception as exc:  # noqa: BLE001
        print(f"[watch] backfill: Immich album list failed ({type(exc).__name__}); albums not linked", file=sys.stderr)
        return {}


def _record_for(doc: dict, parsed: dict) -> index.RunRecord:
    source = parsed["source"]
    key = identity.video_key(source)
    canon = identity.canonical_source(source)
    frames = [
        {"t": f["t"], "path": None, "kind": "scene", "ocr": f["ocr"], "immich_asset_id": f["immich_asset_id"]}
        for f in parsed["frames"]
    ]
    return index.RunRecord(
        video_key=key, source=source, origin="backfill", backend="claude", detail="backfill",
        title=parsed["title"] or doc.get("title"),
        url=canon if canon.lower().startswith(("http://", "https://")) else None,
        duration=parsed["duration"], range=parsed.get("range"), transcript_source=parsed["transcript_source"],
        segments=parsed["segments"], frames=frames, origin_ref=f"paperless:{doc['id']}",
    )


def backfill_paperless(conn, cfg: dict, get_json=None, find_album=None) -> dict:
    """Index every Paperless ``watch``-tagged report. Idempotent via UNIQUE(origin, origin_ref)."""
    cfg = cfg or {}
    get_json = get_json or _default_get_json(cfg)
    albums = None
    if find_album is None:
        albums = _album_index(cfg)
        find_album = lambda name: albums.get(name)  # noqa: E731
    base = (cfg.get("PAPERLESS_URL") or library.DEFAULT_PAPERLESS_URL).rstrip("/")
    inbox = setting("WATCH_QDRANT_INBOX")
    stats = {"seen": 0, "added": 0, "skipped": 0}
    path: str | None = PAGE_PATH
    while path:
        page = get_json(path) or {}
        for doc in page.get("results") or []:
            stats["seen"] += 1
            counted = False
            try:
                parsed = parse_report(doc.get("content") or "")
                if not parsed["source"] or doc.get("id") is None:
                    stats["skipped"] += 1
                    continue
                run = _record_for(doc, parsed)
                result = index.record_run(conn, run)
                if result.skipped:
                    stats["skipped"] += 1
                    continue
                stats["added"] += 1  # counted once, whatever fails below
                counted = True
                index.set_library_ref(conn, run.video_key, paperless_doc_id=doc["id"])
                album = find_album(doc.get("title") or run.title or "")
                if album:
                    index.set_library_ref(conn, run.video_key, immich_album_id=album)
                if inbox:
                    created = str(doc.get("created") or "")[:10] or date.today().isoformat()
                    store.write_qdrant_inbox(Path(inbox).expanduser(), run.video_key, run.title, run.url,
                                             created, doc.get("content") or "")
            except Exception as exc:  # noqa: BLE001 - one bad doc must not stop the backfill
                if not counted:
                    stats["skipped"] += 1
                print(f"[watch] backfill: doc {doc.get('id')} failed ({type(exc).__name__}: {exc})", file=sys.stderr)
        nxt = page.get("next")
        path = _as_path(nxt, base) if nxt else None
    return stats


def _load(vdir: Path, name: str, default):
    try:
        return json.loads((vdir / name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def rebuild(conn, store_dir: Path, video_key: str) -> index.RunResult:
    """Drop a video's index rows and re-create them from its store directory.

    Forget + re-insert run in one transaction, so a failure keeps the old rows.
    Not restored (not kept in the store files): per-frame ``immich_asset_id``,
    ``phash`` (recomputed from the frame file when present) and frame ``kind``
    (all rebuilt frames are "scene"); evicted frames without OCR are lost.
    """
    root = Path(store_dir)
    vdir = store.video_dir(root, video_key)
    if not vdir.is_dir():
        raise FileNotFoundError(f"no store directory for {video_key}: {vdir}")
    meta = _load(vdir, "meta.json", {}) or {}
    segments = _load(vdir, "transcript.json", []) or []
    ocr = _load(vdir, "ocr.json", []) or []
    notes = _load(vdir, "notes.json", []) or []

    ocr_by_ms: dict[int, str] = {}
    for o in ocr:
        ocr_by_ms.setdefault(round(float(o["t"]) * 1000), o.get("text") or "")
    frames: list[dict] = []
    seen_ms: set[int] = set()
    fdir = vdir / "frames"
    files = sorted(fdir.glob("*.jpg")) if fdir.is_dir() else []
    for f in files:
        try:
            ms = int(f.stem.split("_")[0])
        except ValueError:
            continue
        seen_ms.add(ms)
        frames.append({"t": ms / 1000, "path": str(f), "kind": "scene", "ocr": ocr_by_ms.get(ms)})
    for ms, text in ocr_by_ms.items():  # OCR whose frame file was evicted keeps its row
        if ms not in seen_ms:
            frames.append({"t": ms / 1000, "path": None, "kind": "scene", "ocr": text})

    version = meta.get("evidence_version", 0)
    tsource = next((s.get("source") for s in segments if s.get("source")), None)
    run = index.RunRecord(
        video_key=video_key, source=meta.get("url") or "", origin="backfill", backend=meta.get("backend") or "claude",
        detail=meta.get("detail") or "backfill", title=meta.get("title"), url=meta.get("url"),
        duration=meta.get("duration"), channel=meta.get("channel"), upload_date=meta.get("upload_date"),
        transcript_source=tsource,
        segments=[{k: s[k] for k in ("start", "end", "speaker", "text") if k in s and s[k] is not None}
                  for s in segments],
        frames=frames,
        notes=[{k: n.get(k) for k in ("kind", "t_start", "t_end", "text")} for n in notes],
        # origin_ref carries the key: UNIQUE(origin, origin_ref) is global across videos
        origin_ref=f"rebuild:{video_key}:{version}",
    )
    with index._tx(conn):
        index.forget(conn, video_key)
        result = index.record_run(conn, run)
        for frame_id, src in result.new_frames:
            conn.execute("UPDATE frames SET store_path = ? WHERE id = ?",
                         (Path(src).relative_to(root).as_posix(), frame_id))
        conn.execute("UPDATE videos SET store_path = ? WHERE video_key = ?",
                     (vdir.relative_to(root).as_posix(), video_key))
        lib = {k: v for k, v in (meta.get("library") or {}).items()
               if v is not None and k in index._LIBRARY_COLS}
        if lib:
            index.set_library_ref(conn, video_key, **lib)
    return result
