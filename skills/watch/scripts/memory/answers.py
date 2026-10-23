"""Answer cache for grounded answers, keyed by (video_key, evidence_version).

A question hits the cache when its normalised text matches exactly, or when
its embedding is within ``CACHE_SIM`` cosine of a cached question embedded
with the same model. Any change to a video's evidence set bumps its
evidence_version, so stale answers are never served; ``store`` prunes them.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from .embeddings import cosine_many, pack
from .index import _tx
from .textnorm import normalize_for_search

CACHE_SIM = 0.92


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _norm(question: str) -> str:
    return " ".join(re.findall(r"\w+", normalize_for_search(question or "")))


def _qvec(embedder, question: str):
    if embedder is None:
        return None
    try:
        return embedder.embed([question])[0]
    except Exception:  # a broken embedder only disables the semantic lookup
        return None


def _hit(conn, row_id: int, tokens: int, result: str) -> dict:
    with _tx(conn):
        conn.execute("UPDATE answers SET hits = hits + 1, tokens_saved = tokens_saved + ? WHERE id = ?",
                     (tokens, row_id))
    return {**json.loads(result), "cached": True}


def lookup(conn, video_key: str, evidence_version: int, question: str, embedder=None) -> dict | None:
    rows = conn.execute(
        "SELECT id, question_norm, q_model, q_dim, q_vec, result, tokens_in + tokens_out AS tok "
        "FROM answers WHERE video_key = ? AND evidence_version = ? ORDER BY id DESC",
        (video_key, evidence_version)).fetchall()
    if not rows:
        return None
    norm = _norm(question)
    for r in rows:
        if r["question_norm"] == norm:
            return _hit(conn, r["id"], r["tok"], r["result"])
    qv = _qvec(embedder, question)
    if qv is None:
        return None
    cands = [r for r in rows if r["q_vec"] is not None and r["q_model"] == embedder.model_name and r["q_dim"] == len(qv)]
    if not cands:
        return None
    sims = cosine_many(qv, [r["q_vec"] for r in cands], len(qv))
    best, sim = max(zip(cands, sims), key=lambda p: p[1])
    return _hit(conn, best["id"], best["tok"], best["result"]) if sim >= CACHE_SIM else None


def store(conn, video_key: str, evidence_version: int, question: str, result: dict, *,
          provider: str | None = None, tokens_in: int = 0, tokens_out: int = 0, embedder=None) -> None:
    qv = _qvec(embedder, question)
    with _tx(conn):
        conn.execute("DELETE FROM answers WHERE video_key = ? AND evidence_version != ?",
                     (video_key, evidence_version))
        conn.execute(
            "INSERT INTO answers (video_key, evidence_version, question_norm, q_model, q_dim, q_vec, result, "
            "provider, tokens_in, tokens_out, created) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (video_key, evidence_version, _norm(question),
             embedder.model_name if qv is not None else None, len(qv) if qv is not None else None,
             pack(qv) if qv is not None else None, json.dumps(result, ensure_ascii=False),
             provider, int(tokens_in), int(tokens_out), _now()))


def stats(conn) -> dict:
    r = conn.execute(
        "SELECT count(*), coalesce(sum(hits), 0), coalesce(sum(tokens_in + tokens_out), 0), "
        "coalesce(sum(tokens_saved), 0) FROM answers").fetchone()
    by = {(p or "none"): n for p, n in conn.execute(
        "SELECT provider, count(*) FROM answers GROUP BY provider ORDER BY provider")}
    return {"answers": r[0], "cache_hits": r[1], "tokens_used": r[2], "tokens_saved": r[3], "by_provider": by}
