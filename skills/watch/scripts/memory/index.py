"""Video-memory index: one SQLite DB holding the merged evidence set per video.

Schema v1 (column names are a contract for store/retrieval/backfill), a tiny
migration runner, and ``record_run`` — the merge of one watch run into the
video's evidence: transcript replaced only by a better source, frames/OCR
unioned with timestamp + thumbnail dedup, notes appended, and the ``fts``
table kept in lock-step with every inserted/deleted evidence row.

stdlib only (sqlite3 + ffmpeg via ``frames.thumb_fingerprint``).
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import frames as _frames
import identity
from config import memory_paths
from memory.chunking import chunk_segments
from memory.textnorm import normalize_for_search

SCHEMA_VERSION = 2

# MIGRATIONS[i] upgrades the schema from version i to i + 1.
MIGRATIONS: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS meta (
        key   TEXT PRIMARY KEY,
        value TEXT
    );
    CREATE TABLE videos (
        video_key          TEXT PRIMARY KEY,
        evidence_version   INTEGER NOT NULL DEFAULT 0,
        title              TEXT,
        title_norm         TEXT,
        url                TEXT,
        source             TEXT,
        duration           REAL,
        channel            TEXT,
        upload_date        TEXT,
        created            TEXT,
        last_access        TEXT,
        transcript_source  TEXT,
        transcript_rank    INTEGER NOT NULL DEFAULT 0,
        immich_album_id    TEXT,
        nextcloud_path     TEXT,
        paperless_doc_id   INTEGER,
        paperless_task_id  TEXT,
        store_path         TEXT
    );
    CREATE TABLE runs (
        run_id             INTEGER PRIMARY KEY,
        video_key          TEXT NOT NULL,
        created            TEXT,
        detail             TEXT,
        backend            TEXT,
        transcript_source  TEXT,
        origin             TEXT,
        origin_ref         TEXT,
        UNIQUE (origin, origin_ref)
    );
    CREATE INDEX runs_video ON runs (video_key);
    CREATE TABLE segments (
        id         INTEGER PRIMARY KEY,
        video_key  TEXT NOT NULL,
        run_id     INTEGER,
        start      REAL,
        "end"      REAL,
        speaker    TEXT,
        text       TEXT,
        source     TEXT
    );
    CREATE INDEX segments_video_start ON segments (video_key, start);
    CREATE TABLE ocr_blocks (
        id         INTEGER PRIMARY KEY,
        video_key  TEXT NOT NULL,
        run_id     INTEGER,
        t          REAL,
        frame_id   INTEGER,
        text       TEXT
    );
    CREATE INDEX ocr_blocks_video_t ON ocr_blocks (video_key, t);
    CREATE TABLE frames (
        id               INTEGER PRIMARY KEY,
        video_key        TEXT NOT NULL,
        run_id           INTEGER,
        t                REAL,
        kind             TEXT,
        phash            BLOB,
        store_path       TEXT,
        immich_asset_id  TEXT
    );
    CREATE INDEX frames_video_t ON frames (video_key, t);
    CREATE TABLE notes (
        id         INTEGER PRIMARY KEY,
        video_key  TEXT NOT NULL,
        run_id     INTEGER,
        kind       TEXT,
        t_start    REAL,
        t_end      REAL,
        text       TEXT
    );
    CREATE INDEX notes_video ON notes (video_key);
    CREATE TABLE chunks (
        id         INTEGER PRIMARY KEY,
        video_key  TEXT NOT NULL,
        t_start    REAL,
        t_end      REAL,
        speaker    TEXT,
        text       TEXT
    );
    CREATE INDEX chunks_video ON chunks (video_key, t_start);
    CREATE VIRTUAL TABLE fts USING fts5(
        text, text_norm,
        kind UNINDEXED, video_key UNINDEXED, src_table UNINDEXED, src_id UNINDEXED, t UNINDEXED,
        tokenize="unicode61 remove_diacritics 2"
    );
    CREATE TABLE embeddings (
        src_table  TEXT NOT NULL,
        src_id     INTEGER NOT NULL,
        video_key  TEXT,
        model      TEXT,
        dim        INTEGER,
        vector     BLOB,
        PRIMARY KEY (src_table, src_id)
    );
    CREATE INDEX embeddings_video ON embeddings (video_key);
    """,
    # v2 (0.5.1): grounded-answer cache + escalation attempts
    """
    CREATE TABLE answers (
        id                INTEGER PRIMARY KEY,
        video_key         TEXT NOT NULL,
        evidence_version  INTEGER,
        question_norm     TEXT,
        q_model           TEXT,
        q_dim             INTEGER,
        q_vec             BLOB,
        result            TEXT,
        provider          TEXT,
        tokens_in         INTEGER NOT NULL DEFAULT 0,
        tokens_out        INTEGER NOT NULL DEFAULT 0,
        created           TEXT,
        hits              INTEGER NOT NULL DEFAULT 0,
        tokens_saved      INTEGER NOT NULL DEFAULT 0
    );
    CREATE INDEX answers_video_ev ON answers (video_key, evidence_version);
    CREATE TABLE escalations (
        id         INTEGER PRIMARY KEY,
        video_key  TEXT NOT NULL,
        t          REAL,
        added      INTEGER NOT NULL DEFAULT 0,
        created    TEXT
    );
    CREATE INDEX escalations_video_t ON escalations (video_key, t);
    """,
]

# Tables holding per-video rows (forget() clears all of them).
_VIDEO_TABLES = ("segments", "ocr_blocks", "frames", "notes", "chunks", "embeddings", "fts", "runs",
                 "answers", "escalations", "videos")
_LIBRARY_COLS = frozenset({"immich_album_id", "nextcloud_path", "paperless_doc_id", "paperless_task_id"})
# A frame within this many seconds of an existing one is a dedup candidate.
FRAME_DEDUP_WINDOW_S = 1.0


@dataclass
class RunRecord:
    video_key: str
    source: str
    origin: str
    backend: str
    detail: str
    title: str | None = None
    url: str | None = None
    duration: float | None = None
    channel: str | None = None
    upload_date: str | None = None
    transcript_source: str | None = None
    range: tuple[float, float] | None = None
    segments: list[dict] = field(default_factory=list)   # {start, end, text, speaker?}
    frames: list[dict] = field(default_factory=list)     # {t, path|None, kind, ocr, phash?, immich_asset_id?}
    notes: list[dict] = field(default_factory=list)      # {kind, t_start, t_end, text}
    report_path: str | None = None
    media_path: str | None = None
    library: dict | None = None
    origin_ref: str | None = None


@dataclass
class RunResult:
    video_key: str
    run_id: int | None
    evidence_version: int
    added: dict[str, int]
    transcript_replaced: bool
    skipped: bool = False
    new_frames: list[tuple[int, str]] = field(default_factory=list)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def _tx(conn: sqlite3.Connection):
    """One write transaction (BEGIN IMMEDIATE so concurrent writers queue on
    busy_timeout instead of failing on lock upgrade). Nested use joins the
    outer transaction."""
    if conn.in_transaction:
        yield
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        conn.rollback()
        raise
    conn.commit()


def _statements(script: str):
    buf = ""
    for line in script.splitlines(keepends=True):
        buf += line
        if sqlite3.complete_statement(buf):
            if buf.strip():
                yield buf
            buf = ""
    if buf.strip():
        raise ValueError(f"incomplete SQL statement in migration: {buf.strip()[:80]}")


def _schema_version(conn: sqlite3.Connection) -> int:
    has_meta = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'meta'"
    ).fetchone()
    if not has_meta:
        return 0
    row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    return int(row[0]) if row else 0


def _migrate(conn: sqlite3.Connection) -> None:
    if _schema_version(conn) >= len(MIGRATIONS):
        return
    with _tx(conn):
        current = _schema_version(conn)  # re-read under the write lock
        for version in range(current, len(MIGRATIONS)):
            for stmt in _statements(MIGRATIONS[version]):
                conn.execute(stmt)
            conn.execute(
                "INSERT INTO meta (key, value) VALUES ('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (str(version + 1),),
            )


def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open (creating + migrating as needed) the index DB."""
    path = Path(db_path) if db_path is not None else memory_paths()["index_db"]
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 10000")
    conn.execute("PRAGMA journal_mode = WAL")
    _migrate(conn)
    return conn


def transcript_rank(source: str | None) -> int:
    """Quality rank of a transcript label (higher wins; ties → newest run)."""
    if not source:
        return 0
    s = source.strip().lower()
    if s.startswith("whisper (assemblyai, diarized"):
        return 5
    if s.startswith("whisper ("):
        return 4
    if s.startswith("transcriptapi"):
        return 3
    if s.startswith("captions"):
        return 2
    if s.startswith("gemini (") or s.startswith("openrouter ("):
        return 1
    return 0


def frame_kind(frame: dict) -> str:
    """Map a watch.py frame dict onto the index's frame kinds."""
    reason = frame.get("reason")
    if reason == "transcript-cue":
        return "cue"
    if str(frame.get("path") or "").endswith("_hires.jpg"):
        return "ocr"
    if reason in {"uniform", "two-pass"}:
        return "even"
    return "scene"


def _fts_add(conn, kind: str, video_key: str, src_table: str, src_id: int, t, text: str) -> None:
    conn.execute(
        "INSERT INTO fts (text, text_norm, kind, video_key, src_table, src_id, t) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (text, normalize_for_search(text), kind, video_key, src_table, src_id, t),
    )


def _upsert_video(conn, run: RunRecord, now: str) -> None:
    lib = run.library or {}
    immich = lib.get("immich") or {}
    nextcloud = lib.get("nextcloud") or {}
    paperless = lib.get("paperless") or {}
    task_ids = paperless.get("task_ids") or []
    incoming = {
        "title": run.title,
        "title_norm": normalize_for_search(run.title) if run.title else None,
        "url": run.url,
        "source": identity.canonical_source(run.source) if run.source else None,
        "duration": run.duration,
        "channel": run.channel,
        "upload_date": run.upload_date,
        "immich_album_id": immich.get("album_id"),
        "nextcloud_path": nextcloud.get("folder"),
        "paperless_task_id": task_ids[0] if task_ids else None,
    }
    conn.execute(
        "INSERT INTO videos (video_key, created, last_access) VALUES (?, ?, ?) "
        "ON CONFLICT(video_key) DO UPDATE SET last_access = excluded.last_access",
        (run.video_key, now, now),
    )
    updates = {k: v for k, v in incoming.items() if v is not None}
    if updates:
        sets = ", ".join(f"{k} = ?" for k in updates)
        conn.execute(f"UPDATE videos SET {sets} WHERE video_key = ?", (*updates.values(), run.video_key))


def _rebuild_chunks(conn, video_key: str) -> None:
    conn.execute(
        "DELETE FROM embeddings WHERE src_table = 'chunks' AND src_id IN "
        "(SELECT id FROM chunks WHERE video_key = ?)",
        (video_key,),
    )
    conn.execute("DELETE FROM chunks WHERE video_key = ?", (video_key,))
    segs = [
        {"start": r["start"], "end": r["end"], "speaker": r["speaker"], "text": r["text"]}
        for r in conn.execute(
            'SELECT start, "end", speaker, text FROM segments WHERE video_key = ? ORDER BY start, id',
            (video_key,),
        )
    ]
    for c in chunk_segments(segs):
        conn.execute(
            "INSERT INTO chunks (video_key, t_start, t_end, speaker, text) VALUES (?, ?, ?, ?, ?)",
            (video_key, c["t_start"], c["t_end"], c.get("speaker"), c["text"]),
        )


def _merge_transcript(conn, run: RunRecord, run_id: int, current_rank: int) -> tuple[bool, int, int]:
    """Returns (replaced, inserted, deleted)."""
    incoming = [s for s in run.segments if (s.get("text") or "").strip()]
    if run.range is not None:
        # Clip with the SAME predicate as the delete below: callers pass
        # transcribe.filter_range output, which keeps segments that merely
        # overlap the range; inserting those would duplicate them every run.
        a, b = run.range
        incoming = [s for s in incoming if a <= s["start"] < b]
    if not incoming:
        return False, 0, 0  # nothing usable: never wipe or re-rank
    new_rank = transcript_rank(run.transcript_source)
    if new_rank < current_rank:
        return False, 0, 0
    key = run.video_key
    had_transcript = conn.execute(
        "SELECT 1 FROM segments WHERE video_key = ? LIMIT 1", (key,)
    ).fetchone() is not None
    if run.range is None:
        doomed = [r[0] for r in conn.execute("SELECT id FROM segments WHERE video_key = ?", (key,))]
    else:
        doomed = [r[0] for r in conn.execute(
            "SELECT id FROM segments WHERE video_key = ? AND start >= ? AND start < ?", (key, a, b))]
    if doomed:
        conn.executemany("DELETE FROM fts WHERE src_table = 'segments' AND src_id = ?", [(i,) for i in doomed])
        conn.executemany("DELETE FROM segments WHERE id = ?", [(i,) for i in doomed])
    inserted = 0
    for s in incoming:
        text = s["text"].strip()
        cur = conn.execute(
            'INSERT INTO segments (video_key, run_id, start, "end", speaker, text, source) VALUES (?, ?, ?, ?, ?, ?, ?)',
            (key, run_id, s["start"], s.get("end"), s.get("speaker"), text, run.transcript_source),
        )
        _fts_add(conn, "transcript", key, "segments", cur.lastrowid, s["start"], text)
        inserted += 1
    # Rank follows full runs only; a range run on a video with no transcript
    # yet sets it too, so a later lower-rank full run cannot wipe it.
    if run.range is None or not had_transcript:
        conn.execute(
            "UPDATE videos SET transcript_rank = ?, transcript_source = ? WHERE video_key = ?",
            (new_rank, run.transcript_source, key),
        )
    _rebuild_chunks(conn, key)
    return True, inserted, len(doomed)


def _is_duplicate_frame(conn, video_key: str, t: float, phash: bytes | None) -> bool:
    w = FRAME_DEDUP_WINDOW_S + 1e-9
    for row in conn.execute(
        "SELECT phash FROM frames WHERE video_key = ? AND t BETWEEN ? AND ?", (video_key, t - w, t + w)
    ):
        other = row[0]
        if phash is None or other is None:
            return True
        if _frames._frame_delta(bytes(other), phash) <= _frames.DEDUP_THRESHOLD:
            return True
    return False


def _merge_frames(
    conn, run: RunRecord, run_id: int, phashes: list[bytes | None]
) -> tuple[int, int, list[tuple[int, str]]]:
    immich_frames = ((run.library or {}).get("immich") or {}).get("frames") or {}
    n_frames = n_ocr = 0
    new_frames: list[tuple[int, str]] = []
    for f, phash in zip(run.frames, phashes):
        t = float(f["t"])
        path = f.get("path")
        if _is_duplicate_frame(conn, run.video_key, t, phash):
            continue  # skipped together with its OCR
        asset = f.get("immich_asset_id") or immich_frames.get(str(round(t * 1000)))
        cur = conn.execute(
            "INSERT INTO frames (video_key, run_id, t, kind, phash, store_path, immich_asset_id) "
            "VALUES (?, ?, ?, ?, ?, NULL, ?)",
            (run.video_key, run_id, t, f.get("kind") or frame_kind(f), phash, asset),
        )
        frame_id = cur.lastrowid
        n_frames += 1
        if path:
            new_frames.append((frame_id, str(path)))
        ocr = (f.get("ocr") or "").strip()
        if ocr:
            cur = conn.execute(
                "INSERT INTO ocr_blocks (video_key, run_id, t, frame_id, text) VALUES (?, ?, ?, ?, ?)",
                (run.video_key, run_id, t, frame_id, ocr),
            )
            _fts_add(conn, "ocr", run.video_key, "ocr_blocks", cur.lastrowid, t, ocr)
            n_ocr += 1
    return n_frames, n_ocr, new_frames


def _merge_notes(conn, run: RunRecord, run_id: int) -> int:
    n = 0
    for note in run.notes:
        text = (note.get("text") or "").strip()
        if not text:
            continue
        cur = conn.execute(
            "INSERT INTO notes (video_key, run_id, kind, t_start, t_end, text) VALUES (?, ?, ?, ?, ?, ?)",
            (run.video_key, run_id, note.get("kind"), note.get("t_start"), note.get("t_end"), text),
        )
        _fts_add(conn, "note", run.video_key, "notes", cur.lastrowid, note.get("t_start"), text)
        n += 1
    return n


def record_run(conn: sqlite3.Connection, run: RunRecord) -> RunResult:
    """Merge one run into the video's evidence set (single transaction)."""
    key = run.video_key
    # ffmpeg fingerprints are computed before taking the write lock.
    phashes = [f.get("phash") or _frames.thumb_fingerprint(f.get("path")) for f in run.frames]
    with _tx(conn):
        if run.origin_ref is not None:
            dup = conn.execute(
                "SELECT 1 FROM runs WHERE origin = ? AND origin_ref = ?", (run.origin, run.origin_ref)
            ).fetchone()
            if dup:
                row = conn.execute("SELECT evidence_version FROM videos WHERE video_key = ?", (key,)).fetchone()
                return RunResult(key, None, row[0] if row else 0, {}, False, skipped=True)
        now = _now()
        _upsert_video(conn, run, now)
        run_id = conn.execute(
            "INSERT INTO runs (video_key, created, detail, backend, transcript_source, origin, origin_ref) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (key, now, run.detail, run.backend, run.transcript_source, run.origin, run.origin_ref),
        ).lastrowid
        current_rank = conn.execute(
            "SELECT transcript_rank FROM videos WHERE video_key = ?", (key,)
        ).fetchone()[0]

        replaced, n_seg, n_seg_deleted = _merge_transcript(conn, run, run_id, current_rank)
        n_frames, n_ocr, new_frames = _merge_frames(conn, run, run_id, phashes)
        n_notes = _merge_notes(conn, run, run_id)

        added = {"segments": n_seg, "frames": n_frames, "ocr": n_ocr, "notes": n_notes}
        if any(added.values()) or n_seg_deleted:
            conn.execute("UPDATE videos SET evidence_version = evidence_version + 1 WHERE video_key = ?", (key,))
        version = conn.execute("SELECT evidence_version FROM videos WHERE video_key = ?", (key,)).fetchone()[0]
    return RunResult(key, run_id, version, added, replaced, new_frames=new_frames)


def touch(conn: sqlite3.Connection, video_key: str) -> None:
    with _tx(conn):
        conn.execute("UPDATE videos SET last_access = ? WHERE video_key = ?", (_now(), video_key))


def forget(conn: sqlite3.Connection, video_key: str) -> bool:
    """Delete every index row of a video. Store files are the caller's job."""
    with _tx(conn):
        existed = conn.execute("SELECT 1 FROM videos WHERE video_key = ?", (video_key,)).fetchone() is not None
        for table in _VIDEO_TABLES:
            conn.execute(f"DELETE FROM {table} WHERE video_key = ?", (video_key,))
    return existed


def bump_evidence(conn: sqlite3.Connection, video_key: str) -> int:
    """Mark the video's evidence set as changed (invalidates cached answers)."""
    with _tx(conn):
        conn.execute("UPDATE videos SET evidence_version = evidence_version + 1 WHERE video_key = ?", (video_key,))
        row = conn.execute("SELECT evidence_version FROM videos WHERE video_key = ?", (video_key,)).fetchone()
    return int(row[0]) if row else 0


def set_library_ref(conn: sqlite3.Connection, video_key: str, **cols) -> None:
    bad = set(cols) - _LIBRARY_COLS
    if bad:
        raise ValueError(f"not a library ref column: {', '.join(sorted(bad))}")
    if not cols:
        return
    sets = ", ".join(f"{k} = ?" for k in cols)
    with _tx(conn):
        conn.execute(f"UPDATE videos SET {sets} WHERE video_key = ?", (*cols.values(), video_key))
