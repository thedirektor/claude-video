"""Write path of the video memory: remember one finished watch run.

``remember`` merges the run into the index, copies its artifacts into the
store, embeds anything still pending, enforces store retention, and (for
archived runs) hands the report to the Qdrant inbox and resolves the Paperless
document id. It NEVER raises: memory is a side effect and must never fail a
watch run, so any error becomes one ``[watch] memory: …`` stderr line.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import library
from config import memory_paths, setting
from memory import embeddings, index, store


def _paperless_task_id(run: index.RunRecord) -> str | None:
    paperless = (run.library or {}).get("paperless") or {}
    task_ids = paperless.get("task_ids") or []
    return str(task_ids[0]) if task_ids else None


def _qdrant_handoff(run: index.RunRecord, vdir: Path) -> None:
    inbox = setting("WATCH_QDRANT_INBOX")
    if not inbox:
        return
    report = vdir / "report.md"
    if not report.is_file():
        return
    store.write_qdrant_inbox(
        Path(inbox).expanduser(), run.video_key, run.title, run.url,
        date.today().isoformat(), report.read_text(encoding="utf-8"),
    )


def remember(run: index.RunRecord, *, archive: bool, embedder: object = "auto",
             paths: dict | None = None) -> dict | None:
    """Merge ``run`` into the video memory; returns
    ``{"video_key", "run_id", "evidence_version", "mode"}`` or None on error."""
    conn = None
    try:
        paths = paths or memory_paths()
        conn = index.connect(paths["index_db"])
        result = index.record_run(conn, run)
        vdir = store.write_run(conn, paths["store_dir"], run, result)

        emb = embeddings.get_embedder(paths.get("model_dir")) if embedder == "auto" else embedder
        embeddings.embed_pending(conn, emb)

        store.enforce_retention(conn, paths["store_dir"], protect=frozenset({run.video_key}))

        if archive:
            _qdrant_handoff(run, vdir)

        task_id = _paperless_task_id(run)
        if task_id:
            doc_id = library.resolve_paperless_doc(task_id, library.load_config(), attempts=3)
            if doc_id is not None:
                index.set_library_ref(conn, run.video_key, paperless_doc_id=doc_id)

        return {
            "video_key": run.video_key,
            "run_id": result.run_id,
            "evidence_version": result.evidence_version,
            "mode": "hybrid" if emb is not None else "fts",
        }
    except (Exception, SystemExit) as exc:  # noqa: BLE001 - memory must never fail a run
        try:
            print(f"[watch] memory: not indexed ({type(exc).__name__}: {exc})", file=sys.stderr)
        except Exception:  # noqa: BLE001
            pass
        return None
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass
