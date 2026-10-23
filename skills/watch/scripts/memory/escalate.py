"""Model-free escalation for low-confidence questions.

When the video's media is in the store, re-sample 1 frame/s around the best
evidence times at 2x the hi-res width and OCR them again. Text that is new
(not already an OCR row within ±2 s) is written back permanently as an
``escalated`` frame + OCR row, and the video's evidence_version is bumped so
cached answers are invalidated. Each time is attempted once (``escalations``),
the whole call is bounded by ``WATCH_ESCALATE_BUDGET_S``, and nothing raises.
"""
from __future__ import annotations

import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from config import memory_paths, setting

from . import store
from .index import _fts_add, _tx, bump_evidence
from .textnorm import normalize_for_search

ZOOM_WIDTH = 2048  # 2x the plugin's 1024 px hi-res OCR frames
MAX_TIMES = 3
DUP_WINDOW_S = 2.0
END_GUARD_S = 0.5  # seeking to the very last instant yields no frame


def _budget() -> float:
    try:
        return float(setting("WATCH_ESCALATE_BUDGET_S", "45"))
    except ValueError:
        return 45.0


def _norm(text: str) -> str:
    return " ".join(re.findall(r"\w+", normalize_for_search(text or "")))


def _default_extract(media: str, out: Path, t: float) -> bool:
    import frames

    return frames.reextract_frame(media, out, t, ZOOM_WIDTH)


def _default_ocr(paths: list[str]) -> dict:
    import ocr

    return ocr.run_ocr(paths)


def _default_probe(media: str) -> float | None:
    import frames

    return frames.get_metadata(media).get("duration_seconds")


def _significant(text: str) -> bool:
    import ocr

    return ocr.is_significant(text)


def _media(store_dir, video_key: str) -> Path | None:
    vdir = store.video_dir(Path(store_dir), video_key)
    found = sorted(vdir.glob("media.*")) if vdir.is_dir() else []
    return found[0] if found else None


def _is_dup(conn, video_key: str, t: float, norm: str) -> bool:
    rows = conn.execute("SELECT text FROM ocr_blocks WHERE video_key = ? AND t BETWEEN ? AND ?",
                        (video_key, t - DUP_WINDOW_S, t + DUP_WINDOW_S))
    return any(_norm(r[0]) == norm for r in rows)


def _insert(conn, video_key: str, t: float, text: str) -> None:
    with _tx(conn):
        cur = conn.execute("INSERT INTO frames (video_key, run_id, t, kind) VALUES (?, NULL, ?, 'escalated')",
                           (video_key, t))
        cur = conn.execute("INSERT INTO ocr_blocks (video_key, run_id, t, frame_id, text) VALUES (?, NULL, ?, ?, ?)",
                           (video_key, t, cur.lastrowid, text))
        _fts_add(conn, "ocr", video_key, "ocr_blocks", cur.lastrowid, t, text)


def escalate(conn, video_key: str, times: list[float], *, window: float = 5.0, budget_s: float | None = None,
             store_dir=None, extract=None, ocr=None, probe=None, clock=time.monotonic) -> int:
    added = 0
    try:
        media = _media(store_dir or memory_paths()["store_dir"], video_key)
        if media is None:
            return 0
        extract = extract or _default_extract
        ocr = ocr or _default_ocr
        budget = _budget() if budget_s is None else budget_s
        row = conn.execute("SELECT duration FROM videos WHERE video_key = ?", (video_key,)).fetchone()
        duration = row[0] if row and row[0] else None
        if duration is None:  # local files / transcript-only runs may not record it
            try:
                duration = (probe or _default_probe)(str(media)) or None
            except Exception:
                duration = None
        start = clock()
        out_of_time = False
        with tempfile.TemporaryDirectory(prefix="watch-escalate-") as tmp:
            for t in [x for x in times if x is not None][:MAX_TIMES]:
                if conn.execute("SELECT 1 FROM escalations WHERE video_key = ? AND ABS(t - ?) <= ?",
                                (video_key, t, window)).fetchone():
                    continue
                n = 0
                lo, hi = max(0.0, t - window), t + window
                if duration is not None:
                    hi = min(hi, float(duration) - END_GUARD_S)
                step = lo
                while step <= hi + 1e-9:
                    if clock() - start > budget:
                        out_of_time = True
                        break
                    out = Path(tmp) / f"esc_{int(step * 1000)}.jpg"
                    if extract(str(media), out, step) and out.exists():
                        text = (ocr([str(out)]).get(str(out)) or "").strip()
                        norm = _norm(text)
                        if norm and _significant(text) and not _is_dup(conn, video_key, step, norm):
                            _insert(conn, video_key, step, text)
                            n += 1
                    step = round(step + 1.0, 3)
                with _tx(conn):
                    conn.execute("INSERT INTO escalations (video_key, t, added, created) VALUES (?, ?, ?, ?)",
                                 (video_key, t, n, datetime.now(timezone.utc).isoformat(timespec="seconds")))
                added += n
                if out_of_time:
                    break
    except Exception as exc:  # noqa: BLE001 - escalation is best-effort
        print(f"[watch] escalation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    if added:
        bump_evidence(conn, video_key)
    return added
