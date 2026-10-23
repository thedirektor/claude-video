"""Optional multilingual embeddings (fastembed) for the video-memory index.

Everything heavy (fastembed, numpy) is imported lazily; a missing or broken
dependency degrades to FTS-only with a single stderr notice and never fails a run.
"""
from __future__ import annotations

import sys
from array import array
from pathlib import Path
from typing import Protocol

from config import embeddings_enabled, memory_paths

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DIM = 384

_notified = False


class Embedder(Protocol):
    model_name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _notice(msg: str) -> None:
    global _notified
    if not _notified:
        _notified = True
        print(f"[watch] {msg}", file=sys.stderr)


class _FastEmbedder:
    model_name = MODEL_NAME
    dim = DIM

    def __init__(self, model) -> None:
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in v] for v in self._model.embed(texts)]


def get_embedder(model_dir: Path | None = None, local_only: bool = False) -> Embedder | None:
    if not embeddings_enabled():
        _notice("embeddings disabled (WATCH_EMBEDDINGS=0); memory search is full-text only")
        return None
    try:
        from fastembed import TextEmbedding

        model = TextEmbedding(
            model_name=MODEL_NAME,
            cache_dir=str(model_dir or memory_paths()["model_dir"]),
            local_files_only=local_only,
        )
        return _FastEmbedder(model)
    except ImportError:
        _notice("fastembed not installed; memory search is full-text only")
    except Exception as exc:  # noqa: BLE001 - any load error degrades to FTS
        _notice(f"embedding model unavailable ({type(exc).__name__}); memory search is full-text only")
    return None


def warm_up(model_dir: Path | None = None) -> bool:
    """Import the heavy stack on the calling thread, then load the model offline."""
    try:
        import numpy  # noqa: F401
        import fastembed  # noqa: F401
    except Exception:  # noqa: BLE001
        pass
    return get_embedder(model_dir, local_only=True) is not None


def pack(vec) -> bytes:
    return array("f", [float(x) for x in vec]).tobytes()


def unpack(blob: bytes, dim: int) -> list[float]:
    a = array("f")
    a.frombytes(blob)
    return list(a[:dim])


def cosine_many(query: list[float], blobs: list[bytes], dim: int) -> list[float]:
    if not blobs:
        return []
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        q = np.asarray(query, dtype=np.float32)
        m = np.frombuffer(b"".join(blobs), dtype=np.float32).reshape(len(blobs), dim)
        denom = np.linalg.norm(m, axis=1) * (np.linalg.norm(q) or 1.0)
        denom[denom == 0] = 1.0
        return [float(x) for x in (m @ q) / denom]
    qn = sum(x * x for x in query) ** 0.5 or 1.0
    out = []
    for b in blobs:
        v = unpack(b, dim)
        vn = sum(x * x for x in v) ** 0.5 or 1.0
        out.append(sum(a * c for a, c in zip(query, v)) / (qn * vn))
    return out


_SOURCES = (("chunks", "text"), ("ocr_blocks", "text"), ("notes", "text"))
_BATCH = 64


def embed_pending(conn, embedder: Embedder | None, limit: int = 2000) -> int:
    if embedder is None:
        return 0
    row = conn.execute("SELECT value FROM meta WHERE key = 'embedding_model'").fetchone()
    if (row[0] if row else None) != embedder.model_name:
        conn.execute("DELETE FROM embeddings")
        conn.execute(
            "INSERT INTO meta (key, value) VALUES ('embedding_model', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (embedder.model_name,),
        )
        conn.commit()
    pending: list[tuple[str, int, str, str]] = []
    for table, col in _SOURCES:
        rows = conn.execute(
            f"SELECT t.id, t.video_key, t.{col} FROM {table} t "
            f"LEFT JOIN embeddings e ON e.src_table = ? AND e.src_id = t.id "
            f"WHERE e.src_id IS NULL AND t.{col} IS NOT NULL AND trim(t.{col}) != '' "
            f"ORDER BY t.id LIMIT ?",
            (table, max(limit - len(pending), 0)),
        ).fetchall()
        pending.extend((table, r[0], r[1], r[2]) for r in rows)
    done = 0
    for i in range(0, len(pending), _BATCH):
        batch = pending[i : i + _BATCH]
        vecs = embedder.embed([b[3] for b in batch])
        conn.executemany(
            "INSERT OR REPLACE INTO embeddings (src_table, src_id, video_key, model, dim, vector) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [(b[0], b[1], b[2], embedder.model_name, embedder.dim, pack(v)) for b, v in zip(batch, vecs)],
        )
        done += len(batch)
    conn.commit()
    return done
