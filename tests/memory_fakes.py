"""Shared helpers for the video-memory tests (Tasks 3-7).

Builders (``fake_run``, ``seg``, ``fr``, ``jpg``) plus small SQL readers over the
index so tests assert on stored rows instead of re-implementing queries.
"""
from __future__ import annotations

import itertools
import subprocess
from pathlib import Path

from memory.index import RunRecord

KEY = "yt:AAAAAAAAAAA"
_jpg_counter = itertools.count()


def fake_run(**kw) -> RunRecord:
    base = dict(
        video_key=KEY,
        source="https://www.youtube.com/watch?v=AAAAAAAAAAA",
        origin="skill",
        backend="claude",
        detail="balanced",
    )
    base.update(kw)
    return RunRecord(**base)


def seg(start: float, end: float, text: str, speaker: str | None = None) -> dict:
    d = {"start": float(start), "end": float(end), "text": text}
    if speaker is not None:
        d["speaker"] = speaker
    return d


def fr(t: float, path, ocr: str | None, kind: str = "scene", **extra) -> dict:
    d = {"t": float(t), "path": str(path) if path else None, "kind": kind, "ocr": ocr}
    d.update(extra)
    return d


def jpg(tmp_path: Path, color: str, size: str = "64x64") -> Path:
    """A solid-colour JPEG made by ffmpeg (unique name per call)."""
    out = Path(tmp_path) / f"{color}_{next(_jpg_counter)}.jpg"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", f"color=c={color}:s={size}", "-frames:v", "1", str(out)],
        check=True, capture_output=True,
    )
    return out


_ORDER = {
    "segments": "start, id",
    "ocr_blocks": "t, id",
    "frames": "t, id",
    "notes": "t_start, id",
    "chunks": "t_start, id",
}


def texts(conn, table: str, video_key: str = KEY) -> list[str]:
    order = _ORDER.get(table, "id")
    rows = conn.execute(
        f"SELECT text FROM {table} WHERE video_key = ? ORDER BY {order}", (video_key,)
    ).fetchall()
    return [r[0] for r in rows]


def fts_match(conn, query: str) -> list[str]:
    rows = conn.execute("SELECT text FROM fts WHERE fts MATCH ? ORDER BY rowid", (query,)).fetchall()
    return [r[0] for r in rows]


def chunk_texts(conn, video_key: str = KEY) -> list[str]:
    return texts(conn, "chunks", video_key)


def frame_times(conn, video_key: str = KEY) -> list[float]:
    rows = conn.execute("SELECT t FROM frames WHERE video_key = ? ORDER BY t, id", (video_key,)).fetchall()
    return [r[0] for r in rows]


def ocr_count(conn, video_key: str = KEY) -> int:
    return conn.execute("SELECT count(*) FROM ocr_blocks WHERE video_key = ?", (video_key,)).fetchone()[0]


def rank(conn, video_key: str = KEY) -> int:
    return conn.execute("SELECT transcript_rank FROM videos WHERE video_key = ?", (video_key,)).fetchone()[0]


def stored_times(conn, video_key: str = KEY) -> set[float]:
    """Every evidence time stored for a video (segment starts, OCR/frame t, note starts)."""
    out: set[float] = set()
    for sql in (
        "SELECT start FROM segments WHERE video_key = ?",
        "SELECT t FROM ocr_blocks WHERE video_key = ?",
        "SELECT t FROM frames WHERE video_key = ?",
        "SELECT t_start FROM notes WHERE video_key = ? AND t_start IS NOT NULL",
    ):
        out.update(r[0] for r in conn.execute(sql, (video_key,)))
    return out


def source_of(conn, video_key: str = KEY) -> str | None:
    row = conn.execute("SELECT source FROM videos WHERE video_key = ?", (video_key,)).fetchone()
    return row[0] if row else None


def immich_ids(conn, video_key: str = KEY) -> list[str | None]:
    rows = conn.execute(
        "SELECT immich_asset_id FROM frames WHERE video_key = ? ORDER BY t, id", (video_key,)
    ).fetchall()
    return [r[0] for r in rows]


def run_count(conn, video_key: str = KEY) -> int:
    return conn.execute("SELECT count(*) FROM runs WHERE video_key = ?", (video_key,)).fetchone()[0]


def meta(conn, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


class FakeEmbedder:
    """Deterministic bag-of-words embedder hashed into 16 dims (no model, no network)."""

    model_name = "fake-16"
    dim = 16
    SYNONYMS = {"coche": "automovil", "carro": "automovil", "car": "automovil"}

    def embed(self, texts: list[str]) -> list[list[float]]:
        import zlib

        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in t.lower().split():
                w = self.SYNONYMS.get(w, w)
                v[zlib.crc32(w.encode()) % self.dim] += 1.0
            out.append(v)
        return out


def embedding_models(conn) -> set[str]:
    return {r[0] for r in conn.execute("SELECT DISTINCT model FROM embeddings")}
