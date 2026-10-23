"""Visual search: Immich's CLIP over the watch albums, mapped back to (video_key, t).

Archived runs upload their frames to an Immich album and the index keeps the album
id (``videos.immich_album_id``) and each frame's asset id (``frames.immich_asset_id``).
A visual query runs Immich smart search restricted to those albums and returns hits
shaped like ``retrieval.hit_dicts`` (``kind="visual"``, plus ``immich_url``). Never
raises: Immich errors and unlinked memories come back as a ``note``.
"""
from __future__ import annotations

import json
import urllib.request

from config import setting

from .retrieval import stamp


def _immich_smart(query: str, album_ids: list[str], size: int) -> list[str]:
    import library

    cfg = library.load_config()
    base = (cfg.get("IMMICH_BASE_URL") or "").rstrip("/")
    key = cfg.get("IMMICH_API_KEY")
    if not base or not key:
        raise RuntimeError("Immich is not configured (IMMICH_BASE_URL / IMMICH_API_KEY)")
    req = urllib.request.Request(base + "/api/search/smart", method="POST",
                                 data=json.dumps({"query": query, "size": size, "albumIds": album_ids}).encode(),
                                 headers=library._immich_headers(key, "application/json"))
    with urllib.request.urlopen(req, timeout=30, context=library._ssl_context()) as r:  # noqa: S310
        return [a["id"] for a in json.load(r)["assets"]["items"]]


def visual_search(conn, query: str, video_key: str | None = None, k: int = 8, smart=None) -> dict:
    sql = "SELECT video_key, immich_album_id, title FROM videos WHERE immich_album_id IS NOT NULL"
    rows = conn.execute(sql + (" AND video_key = ?" if video_key else ""), (video_key,) if video_key else ()).fetchall()
    if not rows:
        return {"mode": "visual", "hits": [],
                "note": "no video in the memory is linked to Immich yet (archived runs upload their frames there)"}
    titles = {r[0]: r[2] for r in rows}
    try:
        ids = (smart or _immich_smart)(query, [r[1] for r in rows], max(15, int(k) * 3))
    except Exception as exc:  # noqa: BLE001
        return {"mode": "visual", "hits": [], "note": f"Immich search failed: {exc}"}
    base = (setting("IMMICH_BASE_URL") or "").rstrip("/")
    hits = []
    for rank, asset in enumerate(ids):
        row = conn.execute("SELECT video_key, t FROM frames WHERE immich_asset_id = ? LIMIT 1", (asset,)).fetchone()
        if row is None or row[0] not in titles:
            continue
        hits.append({"video_key": row[0], "title": titles[row[0]], "t": row[1], "timestamp": stamp(row[1]),
                     "t_end": None, "kind": "visual", "snippet": None, "score": round(1.0 / (rank + 1), 6),
                     "dup_count": 0, "immich_url": f"{base}/photos/{asset}" if base else None})
        if len(hits) >= k:
            break
    return {"mode": "visual", "hits": hits}
