"""Hybrid retrieval over the video-memory index (Task 6).

0.45 * BM25 + 0.55 * cosine, each min-max normalised per query. Transcript hits
fold into their chunk; consecutive near-identical OCR hits collapse into one.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass, replace

from .textnorm import normalize_for_search

W_FTS = 0.45
W_VEC = 0.55
CANDIDATES = 50

_KIND = {"chunks": "transcript", "ocr_blocks": "ocr", "notes": "note"}


@dataclass
class Hit:
    video_key: str
    kind: str  # transcript | ocr | note
    src_table: str
    src_id: int
    t: float | None
    t_end: float | None
    text: str
    score: float
    dup_count: int = 1
    t_first: float | None = None
    t_last: float | None = None


class Hits(list):
    """list[Hit] that also records how it was produced: ``mode`` is "hybrid" or "fts-only"."""

    mode: str = "fts-only"


def _minmax(d: dict) -> dict:
    if not d:
        return {}
    lo, hi = min(d.values()), max(d.values())
    if hi == lo:
        return {k: 1.0 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


def hybrid_scores(fts: dict, vec: dict) -> dict:
    f, v = _minmax(fts), _minmax(vec)
    if not v:
        return f
    return {k: W_FTS * f.get(k, 0.0) + W_VEC * v.get(k, 0.0) for k in set(f) | set(v)}


def _fts_query(query: str) -> str:
    terms = []
    for tok in normalize_for_search(query).split():
        tok = tok.replace('"', '""')
        if tok:
            terms.append(f'"{tok}"')
    return " OR ".join(terms)


def _fts_candidates(conn, query: str, video_key: str | None) -> tuple[dict, dict]:
    """-> (scores by (table, id), t of best-scoring segment by chunk key)."""
    q = _fts_query(query)
    if not q:
        return {}, {}
    sql = (
        "SELECT src_table, src_id, video_key, t, -bm25(fts) AS s FROM fts "
        "WHERE fts MATCH ? "
    )
    args: list = [f"text_norm : ({q})"]
    if video_key:
        sql += "AND video_key = ? "
        args.append(video_key)
    sql += "ORDER BY s DESC LIMIT ?"
    args.append(CANDIDATES)
    scores: dict = {}
    seg_t: dict = {}
    for table, sid, vkey, t, s in (tuple(r) for r in conn.execute(sql, args)):
        key = (table, sid)
        if table == "segments":
            row = conn.execute(
                "SELECT start FROM segments WHERE id = ?", (sid,)
            ).fetchone()
            if row is None:
                continue
            start = row[0]
            c = conn.execute(
                "SELECT id FROM chunks WHERE video_key = ? AND t_start <= ? AND t_end >= ? "
                "ORDER BY t_start, id LIMIT 1",
                (vkey, start, start),
            ).fetchone()
            if c is None:
                continue
            key = ("chunks", c[0])
            if key in scores and scores[key] >= s:
                continue
            seg_t[key] = start
        elif table not in _KIND:
            continue
        scores[key] = max(scores.get(key, s), s)
    return scores, seg_t


def _vec_candidates(conn, query: str, video_key: str | None, embedder) -> dict:
    from .embeddings import cosine_many

    qv = embedder.embed([query])[0]
    sql = "SELECT src_table, src_id, dim, vector FROM embeddings WHERE model = ?"
    args: list = [embedder.model_name]
    if video_key:
        sql += " AND video_key = ?"
        args.append(video_key)
    rows = [tuple(r) for r in conn.execute(sql, args)]
    out: dict = {}
    by_dim: dict = {}
    for r in rows:
        by_dim.setdefault(r[2], []).append(r)
    for dim, grp in by_dim.items():
        if dim != len(qv):
            continue
        sims = cosine_many(qv, [g[3] for g in grp], dim)
        for g, s in zip(grp, sims):
            out[(g[0], g[1])] = s
    top = sorted(out.items(), key=lambda kv: -kv[1])[:CANDIDATES]
    # a cosine <= 0 carries no semantic signal; keeping it would pad results with noise
    return {k: s for k, s in top if s > 0}


def _load(conn, key: tuple) -> Hit | None:
    table, sid = key
    if table == "chunks":
        r = conn.execute("SELECT video_key, t_start, t_end, text FROM chunks WHERE id = ?", (sid,)).fetchone()
        return r and Hit(r[0], "transcript", table, sid, r[1], r[2], r[3], 0.0)
    if table == "ocr_blocks":
        r = conn.execute("SELECT video_key, t, text FROM ocr_blocks WHERE id = ?", (sid,)).fetchone()
        return r and Hit(r[0], "ocr", table, sid, r[1], None, r[2], 0.0)
    if table == "notes":
        r = conn.execute("SELECT video_key, t_start, t_end, text FROM notes WHERE id = ?", (sid,)).fetchone()
        return r and Hit(r[0], "note", table, sid, r[1], r[2], r[3], 0.0)
    return None


def collapse_ocr_duplicates(hits: list[Hit], window: float = 15.0, similarity: float = 0.9) -> list[Hit]:
    """Collapse chains of near-identical OCR hits (per video, within ``window`` s of the previous one)."""
    by_video: dict[str, list[int]] = {}
    for i, h in enumerate(hits):
        if h.kind == "ocr" and h.t is not None:
            by_video.setdefault(h.video_key, []).append(i)
    drop: set[int] = set()
    rep_for: dict[int, Hit] = {}
    for idxs in by_video.values():
        idxs.sort(key=lambda i: (hits[i].t, i))
        groups: list[list[int]] = []
        for i in idxs:
            if groups:
                prev = hits[groups[-1][-1]]
                cur = hits[i]
                if cur.t - prev.t <= window and difflib.SequenceMatcher(
                    None, normalize_for_search(prev.text), normalize_for_search(cur.text)
                ).ratio() >= similarity:
                    groups[-1].append(i)
                    continue
            groups.append([i])
        for g in groups:
            rep = min(g, key=lambda i: (-hits[i].score, hits[i].t, i))
            ts = [hits[i].t for i in g]
            rep_for[rep] = replace(hits[rep], dup_count=len(g), t_first=min(ts), t_last=max(ts))
            drop.update(i for i in g if i != rep)
    return [rep_for.get(i, h) for i, h in enumerate(hits) if i not in drop]


def search(conn, query: str, video_key: str | None = None, k: int = 8, embedder=None) -> Hits:
    fts, seg_t = _fts_candidates(conn, query, video_key)
    vec = _vec_candidates(conn, query, video_key, embedder) if embedder is not None else {}
    scores = hybrid_scores(fts, vec)
    out = Hits()
    out.mode = "hybrid" if embedder is not None else "fts-only"
    hits: list[Hit] = []
    for key, sc in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0])):
        h = _load(conn, key)
        if h is None:
            continue
        h.score = sc
        if key in seg_t:
            h.t = seg_t[key]
        hits.append(h)
    out.extend(collapse_ocr_duplicates(hits)[:k])
    return out
