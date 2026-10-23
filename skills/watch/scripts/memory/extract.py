"""Deterministic chapters and cited notes from one video's stored evidence.

Boundaries come from transcript pauses (≥ 2.5 s), speaker changes and scene-cut
frame times; about one chapter per 150 s, clamped to 2–12 (1 under 90 s). Titles are
the chapter's first sentence (optionally rewritten by one LLM call). Each chapter
gets 2–5 bullets picked from its transcript/OCR rows, each ending ``(MM:SS)`` taken
from the row it came from — never an invented time.
"""
from __future__ import annotations

import re

from config import setting

from . import llm
from .index import _fts_add, _tx, bump_evidence
from .retrieval import stamp
from .textnorm import normalize_for_search

PAUSE_S = 2.5
SECONDS_PER_CHAPTER = 150.0
MAX_CHAPTERS = 12
SINGLE_CHAPTER_UNDER_S = 90.0
SCENE_SNAP_S = 3.0
TITLE_MAX = 70
BULLET_MAX = 160
_SENTENCE = re.compile(r"(?<=[.!?…])\s+")
# Caption/subtitle credit lines ("Transcriptor: … Revisor: …") are never a chapter title.
_CREDITS = re.compile(r"^\W*(transcri(ptor|ber|ption)|translat(or|ion|ed by)|revisor|reviewer|traductor|traducci[oó]n|"
                      r"subt[ií]tulos|subtitles|captions?)\b", re.IGNORECASE)


def _video(conn, key: str):
    return conn.execute("SELECT title, duration FROM videos WHERE video_key = ?", (key,)).fetchone()


def _segments(conn, key: str) -> list[dict]:
    return [{"start": float(r[0]), "end": float(r[1] if r[1] is not None else r[0]), "speaker": r[2], "text": r[3] or ""}
            for r in conn.execute('SELECT start, "end", speaker, text FROM segments WHERE video_key = ? ORDER BY start',
                                  (key,))]


def _frame_times(conn, key: str, kinds: tuple[str, ...] | None) -> list[float]:
    rows = conn.execute("SELECT DISTINCT t, kind FROM frames WHERE video_key = ? ORDER BY t", (key,))
    return [float(t) for t, kind in rows if kinds is None or kind in kinds]


def _ocr(conn, key: str) -> list[tuple[float, str]]:
    return [(float(t), text or "") for t, text in
            conn.execute("SELECT t, text FROM ocr_blocks WHERE video_key = ? ORDER BY t", (key,))]


def _first_sentence(text: str) -> str:
    s = _SENTENCE.split(" ".join((text or "").split()), maxsplit=1)[0].strip()
    s = s.rstrip(".")
    if len(s) > TITLE_MAX:
        cut = s[:TITLE_MAX - 1].rsplit(" ", 1)[0]
        s = cut + "…"
    return s


def _target(duration: float) -> int:
    if duration < SINGLE_CHAPTER_UNDER_S:
        return 1
    return max(2, min(MAX_CHAPTERS, round(duration / SECONDS_PER_CHAPTER)))


def _boundaries(segs: list[dict], scene: list[float], all_frames: list[float], duration: float, n: int) -> list[float]:
    if n <= 1:
        return []
    score: dict[float, float] = {}
    starts = [s["start"] for s in segs]
    for prev, cur in zip(segs, segs[1:]):
        gap = cur["start"] - prev["end"]
        bonus = 0.0
        if gap >= PAUSE_S:
            bonus += 2 + min(gap, 10.0) / 10
        if prev["speaker"] and cur["speaker"] and prev["speaker"] != cur["speaker"]:
            bonus += 1.5
        if bonus:
            score[cur["start"]] = score.get(cur["start"], 0.0) + bonus
    if starts:
        for t in scene:
            near = min(starts, key=lambda s: abs(s - t))
            if abs(near - t) <= SCENE_SNAP_S:
                score[near] = score.get(near, 0.0) + 1.0
        anchors = starts
    else:
        for t in all_frames:
            score[t] = score.get(t, 0.0) + 1.0
        anchors = all_frames

    # Even split as the skeleton; each ideal cut snaps to the strongest candidate
    # within half a chapter of it (ties → the closest), else to the nearest anchor.
    half = duration / n / 2
    chosen: list[float] = []
    for ideal in (duration * k / n for k in range(1, n)):
        lo = max(chosen[-1] + half if chosen else half, ideal - half)
        hi = min(duration - half, ideal + half)
        window = [t for t in score if lo <= t <= hi]
        if window:
            chosen.append(max(window, key=lambda t: (round(score[t], 6), -abs(t - ideal))))
            continue
        near = [t for t in anchors if lo <= t <= hi]
        if near:
            chosen.append(min(near, key=lambda t: abs(t - ideal)))
    return chosen


def _bullet_score(text: str, ocr: bool) -> float:
    words = [w for w in re.findall(r"\w+", text) if len(w) >= 4]
    return len(words) + (2 if re.search(r"\d", text) else 0) + (1 if ocr else 0)


def _bullets(segs: list[dict], ocr: list[tuple[float, str]], a: float, b: float) -> list[dict]:
    cands: list[tuple[float, str, bool]] = [(s["start"], " ".join(s["text"].split()), False)
                                            for s in segs if a <= s["start"] < b and s["text"].strip()]
    for t, text in ocr:
        flat = " ".join(text.split())
        if a <= t < b and len(flat) >= 12 and len(flat.split()) >= 3:
            cands.append((t, flat, True))
    seen: list[set[str]] = []
    unique = []
    for t, text, is_ocr in cands:
        toks = set(re.findall(r"\w+", normalize_for_search(text)))
        if not toks or any(len(toks & o) / len(toks | o) >= 0.8 for o in seen):
            continue  # same line again (OCR re-read, repeated caption)
        seen.append(toks)
        unique.append((t, text, is_ocr))
    if not unique:
        return []
    want = max(2, min(5, round((b - a) / 60) + 1))
    picked = sorted(unique, key=lambda c: (-_bullet_score(c[1], c[2]), c[0]))[:want]
    out = []
    for t, text, _ in sorted(picked, key=lambda c: c[0]):
        if len(text) > BULLET_MAX:
            text = text[:BULLET_MAX - 1].rsplit(" ", 1)[0] + "…"
        out.append({"t": float(int(t)), "text": f"{text} ({stamp(t)})"})
    return out


def _llm_titles(chapters: list[dict], segs: list[dict], complete) -> list[str] | None:
    lines = []
    for i, c in enumerate(chapters, 1):
        excerpt = " ".join(s["text"] for s in segs if c["t_start"] <= s["start"] < c["t_end"])[:400]
        lines.append(f"{i}. ({stamp(c['t_start'])}) {excerpt or c['title']}")
    system = ("You write short chapter titles for a video, in the language of the excerpts. "
              "Reply with exactly one line per chapter: `N. Title` (max 8 words). No other text.")
    try:
        reply = complete(system, "\n".join(lines), max_tokens=60 * len(chapters) + 40)
    except Exception:  # noqa: BLE001 - titles are cosmetic
        return None
    if reply is None:
        return None
    got: dict[int, str] = {}
    for line in (reply.text or "").splitlines():
        m = re.match(r"^\s*(\d+)[.)]\s*(.+?)\s*$", line)
        if m and m.group(2):
            got[int(m.group(1))] = m.group(2).strip("*# ")[:TITLE_MAX]
    if sorted(got) != list(range(1, len(chapters) + 1)):
        return None
    return [got[i] for i in range(1, len(chapters) + 1)]


def chapters(conn, key: str, *, llm_titles: bool | None = None, complete=None) -> list[dict]:
    """[{t_start, t_end, title, bullets: [{t, text}]}] covering the whole video."""
    row = _video(conn, key)
    segs = _segments(conn, key)
    all_frames = _frame_times(conn, key, None)
    duration = float((row[1] if row else None) or 0)
    if duration <= 0:
        duration = max([s["end"] for s in segs] + all_frames + [0.0])
    if duration <= 0:
        return []
    cuts = _boundaries(segs, _frame_times(conn, key, ("scene",)), all_frames, duration, _target(duration))
    ocr = _ocr(conn, key)
    edges = [0.0, *cuts, duration]
    out = []
    for i, (a, b) in enumerate(zip(edges, edges[1:]), 1):
        inside = [s for s in segs if a <= s["start"] < b]
        title = next((t for t in (_first_sentence(s["text"]) for s in inside[:4]) if t and not _CREDITS.match(t)), "")
        out.append({"t_start": a, "t_end": b, "title": title or f"Chapter {i} ({stamp(a)})",
                    "bullets": _bullets(segs, ocr, a, b)})
    if llm_titles is None:
        llm_titles = (setting("WATCH_LLM_TITLES") or "0").strip().lower() in ("1", "true", "yes")
    if llm_titles and len(out) > 1:
        titles = _llm_titles(out, segs, complete or llm.complete)
        if titles:
            for c, t in zip(out, titles):
                c["title"] = t
    return out


def store_chapters(conn, key: str, **kw) -> list[dict]:
    """Compute chapters and store them as notes(kind=chapter), replacing older ones."""
    chs = chapters(conn, key, **kw)
    new = [(c["t_start"], c["t_end"], c["title"]) for c in chs]
    old_rows = conn.execute("SELECT id, t_start, t_end, text FROM notes WHERE video_key = ? AND kind = 'chapter' "
                            "ORDER BY t_start", (key,)).fetchall()
    if [(r[1], r[2], r[3]) for r in old_rows] == new:
        return chs
    with _tx(conn):
        ids = [(r[0],) for r in old_rows]
        if ids:
            conn.executemany("DELETE FROM fts WHERE src_table = 'notes' AND src_id = ?", ids)
            conn.executemany("DELETE FROM embeddings WHERE src_table = 'notes' AND src_id = ?", ids)
            conn.executemany("DELETE FROM notes WHERE id = ?", ids)
        for a, b, title in new:
            cur = conn.execute("INSERT INTO notes (video_key, run_id, kind, t_start, t_end, text) "
                               "VALUES (?, NULL, 'chapter', ?, ?, ?)", (key, a, b, title))
            _fts_add(conn, "note", key, "notes", cur.lastrowid, a, title)
    bump_evidence(conn, key)
    return chs


def notes_md(conn, key: str, chs: list[dict] | None = None) -> str:
    """Cited notes: one `##` section per chapter, bullets ending in (MM:SS)."""
    row = _video(conn, key)
    chs = chapters(conn, key) if chs is None else chs
    lines = [f"# {(row[0] if row else None) or key}", ""]
    for c in chs:
        lines += [f"## {c['title']} ({stamp(c['t_start'])}–{stamp(c['t_end'])})", ""]
        lines += [f"- {b['text']}" for b in c["bullets"]] or ["- (no transcript or on-screen text)"]
        lines.append("")
    return "\n".join(lines)
