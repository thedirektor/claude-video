"""Hybrid retrieval over the video-memory index (Task 6).

0.45 * BM25 + 0.55 * cosine, each min-max normalised per query. Transcript hits
fold into their chunk; consecutive near-identical OCR hits collapse into one.
"""
from __future__ import annotations

import difflib
import os
import re
import sqlite3
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import identity

from . import index as _index
from .textnorm import normalize_for_search

W_FTS = 0.45
W_VEC = 0.55
CANDIDATES = 50

_KIND = {"chunks": "transcript", "ocr_blocks": "ocr", "notes": "note"}


@dataclass
class Hit:
    """One retrieval result.

    ``t`` is the evidence time (best-matching segment start for transcript hits). For OCR hits
    ``dup_count`` is how many near-identical consecutive hits were collapsed into this one
    (1 = none), and ``t_first``/``t_last`` the earliest/latest time of that run (None when not
    an OCR hit). ``t`` is the representative's own time (highest score, then earliest).
    """

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


_CTRL = re.compile(r"[\x00-\x1f]")


def _fts_query(query: str) -> str:
    terms = []
    for tok in normalize_for_search(query).split():
        tok = _CTRL.sub("", tok)
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
    try:
        rows = [tuple(r) for r in conn.execute(sql, args)]
    except sqlite3.OperationalError:
        return {}, {}
    for table, sid, vkey, t, s in rows:
        key = (table, sid)
        if table == "segments":
            row = conn.execute(
                "SELECT start FROM segments WHERE id = ?", (sid,)
            ).fetchone()
            if row is None:
                continue
            start = row[0]
            # half-open [t_start, t_end) so a boundary segment folds into the chunk it starts;
            # inclusive end only as a fallback (final chunk, zero-length chunk)
            c = conn.execute(
                "SELECT id FROM chunks WHERE video_key = ? AND t_start <= ? AND t_end > ? "
                "ORDER BY t_start DESC, id LIMIT 1",
                (vkey, start, start),
            ).fetchone() or conn.execute(
                "SELECT id FROM chunks WHERE video_key = ? AND t_start <= ? AND t_end >= ? "
                "ORDER BY t_start DESC, id LIMIT 1",
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
    vec: dict = {}
    if embedder is not None:
        try:
            vec = _vec_candidates(conn, query, video_key, embedder)
        except Exception as e:  # embedder/numpy failure must never fail a search
            print(f"watch: vector search unavailable ({type(e).__name__}: {e}); using text search only", file=sys.stderr)
            vec = {}
    scores = hybrid_scores(fts, vec)
    out = Hits()
    out.mode = "hybrid" if vec else "fts-only"
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


# --------------------------------------------------------------------------- read API (Task 7)

RULE = (
    "Cite only timestamps listed in evidence[].timestamp; "
    "if the evidence does not answer the question, say it was not found in the video."
)
SNIPPET_MAX = 240
FRAME_NEAR = 5.0
CONTEXT_PAD = 10.0


def stamp(seconds: float) -> str:
    """Plain transcript stamp, identical to ``transcribe.format_transcript`` ([MM:SS], int-truncated)."""
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


def stamp_diarized(seconds: float) -> str:
    """Start stamp of a diarized line's ``(M:SS-M:SS)``."""
    s = int(seconds)
    return f"{s // 60}:{s % 60:02d}"


class VideoNotFound(LookupError):
    pass


class AmbiguousVideo(LookupError):
    def __init__(self, ref: str, candidates: list[dict]):
        self.candidates = candidates[:10]
        super().__init__(f"{ref!r} matches {len(candidates)} videos")


def _like(s: str) -> str:
    return "%" + s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


def _video_row(conn, key: str) -> dict | None:
    r = conn.execute("SELECT * FROM videos WHERE video_key = ?", (key,)).fetchone()
    return dict(r) if r else None


def resolve_video(conn, ref: str) -> dict:
    """video_key | URL/path (any spelling) | title fragment -> a ``videos`` row. Never guesses."""
    ref = (ref or "").strip()
    if not ref:
        raise VideoNotFound("empty video reference")
    row = _video_row(conn, ref) or _video_row(conn, identity.video_key(ref))
    if row:
        return row
    r = conn.execute(
        "SELECT * FROM videos WHERE source = ? ORDER BY created DESC LIMIT 1", (identity.canonical_source(ref),)
    ).fetchone()
    if r:
        return dict(r)
    norm = normalize_for_search(ref)
    rows = []
    if norm:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM videos WHERE title_norm LIKE ? ESCAPE '\\' ORDER BY created DESC, video_key",
            (_like(norm),),
        )]
    if not rows:
        raise VideoNotFound(f"no video matches {ref!r}")
    if len(rows) > 1:
        raise AmbiguousVideo(ref, [
            {"video_key": v["video_key"], "title": v["title"], "url": v["url"]} for v in rows])
    return rows[0]


def _video_info(v: dict) -> dict:
    return {k: v.get(k) for k in ("video_key", "title", "url", "duration", "evidence_version")}


def _immich_base() -> str:
    base = os.environ.get("IMMICH_BASE_URL")
    if not base:
        try:
            import library

            base = library.load_config().get("IMMICH_BASE_URL")
        except Exception:
            base = None
    return (base or "").rstrip("/")


def _store_root() -> Path:
    from config import memory_paths

    return memory_paths()["store_dir"]


def _frame_dict(f, root: Path, base: str) -> dict:
    path = None
    if f["store_path"]:
        p = root / f["store_path"]
        if p.exists():
            path = str(p)
    url = f"{base}/photos/{f['immich_asset_id']}" if f["immich_asset_id"] and base else None
    return {"t": f["t"], "path": path, "immich_url": url}


def _seg_stamp(seg_row) -> str:
    return stamp_diarized(seg_row["start"]) if seg_row["speaker"] else stamp(seg_row["start"])


def _timestamp_at(conn, video_key: str, t: float | None, kind: str) -> str | None:
    if t is None:
        return None
    if kind == "transcript":
        r = conn.execute(
            "SELECT speaker FROM segments WHERE video_key = ? AND start = ? LIMIT 1", (video_key, t)
        ).fetchone()
        if r is not None and r["speaker"]:
            return stamp_diarized(t)
    return stamp(t)


def _snippet(text: str | None) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= SNIPPET_MAX else text[: SNIPPET_MAX - 1] + "\u2026"


def hit_dicts(conn, hits, mode: str) -> dict:
    """The single JSON-safe shaping of retrieval hits (CLI --json, recall, service)."""
    titles: dict = {}
    out = []
    for h in hits:
        if h.video_key not in titles:
            r = conn.execute("SELECT title FROM videos WHERE video_key = ?", (h.video_key,)).fetchone()
            titles[h.video_key] = r["title"] if r else None
        out.append({
            "video_key": h.video_key,
            "title": titles[h.video_key],
            "t": h.t,
            "timestamp": _timestamp_at(conn, h.video_key, h.t, h.kind),
            "t_end": h.t_end,
            "kind": h.kind,
            "snippet": _snippet(h.text),
            "score": round(float(h.score), 6),
            "dup_count": h.dup_count,
        })
    return {"mode": "hybrid" if mode == "hybrid" else "fts", "hits": out}


def _segments_between(conn, video_key: str, lo: float, hi: float) -> list:
    return conn.execute(
        "SELECT * FROM segments WHERE video_key = ? AND start >= ? AND start <= ? ORDER BY start, id",
        (video_key, lo, hi),
    ).fetchall()


def _context(conn, video_key: str, t: float | None, t_end: float | None) -> str:
    if t is None:
        return ""
    rows = conn.execute(
        "SELECT * FROM segments WHERE video_key = ? AND \"end\" >= ? AND start <= ? ORDER BY start, id",
        (video_key, t - CONTEXT_PAD, (t_end if t_end is not None else t) + CONTEXT_PAD),
    ).fetchall()
    return "\n".join(f"[{_seg_stamp(r)}] {r['text']}" for r in rows)


def _nearest_frame(conn, video_key: str, t: float | None, within: float):
    if t is None:
        return None
    return conn.execute(
        "SELECT * FROM frames WHERE video_key = ? AND ABS(t - ?) <= ? ORDER BY ABS(t - ?), t, id LIMIT 1",
        (video_key, t, within, t),
    ).fetchone()


def ask(conn, video_ref: str, question: str, k: int = 8, embedder=None) -> dict:
    """Evidence pack for one video (no answer synthesis). Every ``t`` is a stored row time."""
    v = resolve_video(conn, video_ref)
    key = v["video_key"]
    hits = search(conn, question, video_key=key, k=k, embedder=embedder)
    root, base = _store_root(), _immich_base()
    evidence = []
    for i, h in enumerate(hits, 1):
        f = _nearest_frame(conn, key, h.t, FRAME_NEAR)
        evidence.append({
            "id": f"E{i}",
            "t": h.t,
            "t_end": h.t_end,
            "timestamp": _timestamp_at(conn, key, h.t, h.kind),
            "kind": h.kind,
            "text": h.text,
            "context": _context(conn, key, h.t, h.t_end),
            "frame": _frame_dict(f, root, base) if f is not None else None,
            "score": round(float(h.score), 6),
        })
    _index.touch(conn, key)
    return {"video": _video_info(v), "question": question, "evidence": evidence, "rule": RULE,
            "mode": "hybrid" if hits.mode == "hybrid" else "fts"}


def moment(conn, video_ref: str, t, window: float = 10.0, frames: int = 4) -> dict:
    """Everything stored within ``window`` seconds of ``t``."""
    from frames import parse_time

    v = resolve_video(conn, video_ref)
    key = v["video_key"]
    tt = parse_time(t)
    if tt is None:
        raise ValueError("t is required")
    lo, hi = tt - window, tt + window
    segs = [{"start": r["start"], "end": r["end"], "timestamp": _seg_stamp(r), "speaker": r["speaker"],
             "text": r["text"]} for r in _segments_between(conn, key, lo, hi)]
    ocr = [{"t": r["t"], "timestamp": stamp(r["t"]), "text": r["text"]} for r in conn.execute(
        "SELECT t, text FROM ocr_blocks WHERE video_key = ? AND t >= ? AND t <= ? ORDER BY t, id", (key, lo, hi))]
    n = max(int(frames), 0)
    inside = conn.execute(
        "SELECT * FROM frames WHERE video_key = ? AND t >= ? AND t <= ? ORDER BY ABS(t - ?), t, id LIMIT ?",
        (key, lo, hi, tt, n)).fetchall()
    if not inside and n:
        inside = conn.execute(
            "SELECT * FROM frames WHERE video_key = ? ORDER BY ABS(t - ?), t, id LIMIT ?", (key, tt, n)).fetchall()
    root, base = _store_root(), _immich_base()
    fr = [_frame_dict(f, root, base) for f in sorted(inside, key=lambda f: (f["t"], f["id"]))]
    _index.touch(conn, key)
    return {"video": _video_info(v), "t": tt, "segments": segs, "ocr": ocr, "frames": fr}


def list_videos(conn, query: str | None = None, limit: int = 20) -> list[dict]:
    sql, args = "SELECT * FROM videos", []
    if query and query.strip():
        norm = normalize_for_search(query)
        sql += " WHERE title_norm LIKE ? ESCAPE '\\' OR url LIKE ? ESCAPE '\\'"
        args += [_like(norm), _like(query.strip())]
    sql += " ORDER BY created DESC, video_key LIMIT ?"
    args.append(max(int(limit), 0))
    out = []
    for r in conn.execute(sql, args):
        d = dict(r)
        out.append({
            **{k: d.get(k) for k in ("video_key", "title", "url", "duration", "created", "last_access",
                                     "evidence_version")},
            "library": {k: d.get(k) for k in ("immich_album_id", "nextcloud_path", "paperless_doc_id")},
        })
    return out
