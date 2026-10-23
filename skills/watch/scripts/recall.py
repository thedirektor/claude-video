#!/usr/bin/env python3
"""/watch-recall — retrieve a saved /watch report so a later session can read,
see, or learn from a video it never watched.

Retrieval paths, in order (step 0 is the local video-memory index; when it has
hits they win, otherwise recall falls through unchanged):
  0. **Video-memory index** — hybrid search over transcript / OCR / notes.
  1. **Paperless-ngx full-text search** (primary) — `GET /api/documents/?query=Q`
     finds reports by topic even when you don't know the title, then fetches each
     match's full `content` (the whole report.md: transcript + OCR + frame index).
  2. **Nextcloud folder-title match** (fallback) — when Paperless is
     unconfigured, empty, or its search index is stale, list `Watch/` and match
     the query against folder names, then GET each `report.md`.

Every match also points at its **Immich album** (same `<title> (<id>)` name) so a
vision-capable step can pull the frames when visual detail is actually needed.

Pure stdlib. Reuses `library.py` for config + Nextcloud auth/SSL helpers.
Best-effort: `recall()` NEVER raises — any failure yields a readable note.
"""
from __future__ import annotations

import argparse
import re
import sys
import urllib.error
import urllib.parse
from urllib.request import Request, urlopen  # noqa: F401  (urlopen monkeypatched in tests)

import library

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

SNIPPET_CHARS = 600


def _extract_source(content: str) -> str | None:
    """Pull the `- **Source:** <url>` line the report writes at the top."""
    m = re.search(r"\*\*Source:\*\*\s*(\S+)", content or "")
    return m.group(1) if m else None


# --------------------------------------------------------------------------
# Paperless full-text search (primary)
# --------------------------------------------------------------------------

def search_paperless(query: str, cfg: dict, limit: int = 5) -> list[dict]:
    """Full-text search Paperless for `query`; return up to `limit` matches,
    each {title, doc_id, content, source_url}. Best-effort: [] on any failure
    or missing token."""
    cfg = cfg or {}
    base = (cfg.get("PAPERLESS_URL") or library.DEFAULT_PAPERLESS_URL).rstrip("/")
    token = cfg.get("PAPERLESS_TOKEN")
    if not token or not query or not base:
        return []
    headers = {"Authorization": f"Token {token}"}
    try:
        url = f"{base}/api/documents/?query={urllib.parse.quote(query)}"
        with urlopen(Request(url, headers=headers), timeout=30, context=library._ssl_context()) as resp:
            data = library.json.loads(resp.read().decode("utf-8", errors="replace") or "{}")
    except Exception as exc:
        print(f"[watch-recall] Paperless search failed: {exc}", file=sys.stderr)
        return []

    matches: list[dict] = []
    for item in (data.get("results") or [])[:limit]:
        doc_id = item.get("id")
        title = item.get("title") or "(untitled)"
        content = item.get("content") or ""
        if doc_id is not None and not content:
            try:
                with urlopen(
                    Request(f"{base}/api/documents/{doc_id}/", headers=headers),
                    timeout=30, context=library._ssl_context(),
                ) as resp:
                    full = library.json.loads(resp.read().decode("utf-8", errors="replace") or "{}")
                content = full.get("content") or ""
                title = full.get("title") or title
            except Exception as exc:
                print(f"[watch-recall] Paperless doc {doc_id} fetch failed: {exc}", file=sys.stderr)
        matches.append({
            "title": title,
            "doc_id": doc_id,
            "content": content,
            "source_url": _extract_source(content),
        })
    return matches


# --------------------------------------------------------------------------
# Nextcloud folder-title match (fallback)
# --------------------------------------------------------------------------

def list_nextcloud_watch(cfg: dict) -> list[str]:
    """Return the subfolder names under Watch/ (one per saved video). [] on
    any failure or missing password."""
    cfg = cfg or {}
    base = (cfg.get("NEXTCLOUD_URL") or library.DEFAULT_NEXTCLOUD_URL).rstrip("/")
    user = cfg.get("NEXTCLOUD_USER") or library.DEFAULT_NEXTCLOUD_USER
    password = cfg.get("NEXTCLOUD_PASS")
    if not password or not base or not user:
        return []
    auth = library._nextcloud_auth_header(user, password)
    url = f"{base}/remote.php/dav/files/{urllib.parse.quote(user)}/Watch/"
    try:
        with urlopen(
            Request(url, headers={"Authorization": auth, "Depth": "1"}, method="PROPFIND"),
            timeout=30, context=library._ssl_context(),
        ) as resp:
            xml = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        print(f"[watch-recall] Nextcloud list failed: {exc}", file=sys.stderr)
        return []
    names: list[str] = []
    for href in re.findall(r"<d:href>([^<]*)</d:href>", xml):
        href = urllib.parse.unquote(href)
        if "/Watch/" not in href:
            continue
        tail = href.split("/Watch/", 1)[1].rstrip("/")
        if tail:  # skip the Watch/ collection itself
            names.append(tail)
    return names


def fetch_nextcloud_report(folder: str, cfg: dict) -> str | None:
    """GET Watch/<folder>/report.md. None on any failure."""
    cfg = cfg or {}
    base = (cfg.get("NEXTCLOUD_URL") or library.DEFAULT_NEXTCLOUD_URL).rstrip("/")
    user = cfg.get("NEXTCLOUD_USER") or library.DEFAULT_NEXTCLOUD_USER
    password = cfg.get("NEXTCLOUD_PASS")
    if not password or not base or not user:
        return None
    auth = library._nextcloud_auth_header(user, password)
    url = (
        f"{base}/remote.php/dav/files/{urllib.parse.quote(user)}/Watch/"
        f"{urllib.parse.quote(folder)}/report.md"
    )
    try:
        with urlopen(Request(url, headers={"Authorization": auth}), timeout=30, context=library._ssl_context()) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        print(f"[watch-recall] Nextcloud report fetch failed for {folder}: {exc}", file=sys.stderr)
        return None


# --------------------------------------------------------------------------
# Orchestration + formatting
# --------------------------------------------------------------------------

def _format_match(idx: int, title: str, source_url: str | None, content: str,
                  doc_id, via: str, full: bool) -> str:
    lines = [f"## {idx}. {title}"]
    lines.append(f"- Video: {source_url or '(source not recorded)'}")
    where = []
    if doc_id is not None:
        where.append(f"Paperless doc {doc_id}")
    where.append(f"Nextcloud Watch/{title}/report.md")
    lines.append(f"- Report: {' | '.join(where)}")
    lines.append(f"- Frames: Immich album \"{title}\" (read these only if you need visual detail)")
    lines.append("")
    body = content.strip()
    if not full and len(body) > SNIPPET_CHARS:
        body = body[:SNIPPET_CHARS].rstrip() + "\n\n… (truncated — re-run with --full for the whole report)"
    lines.append(body)
    return "\n".join(lines)


def _recall_from_index(query: str, limit: int) -> str | None:
    """Step 0: the video-memory index. None when it is absent, empty for this query, or broken."""
    try:
        from config import memory_paths
        from memory import embeddings, index, retrieval

        db = memory_paths()["index_db"]
        if not db.exists():
            return None
        conn = index.connect(db)
        try:
            embedder = embeddings.get_embedder(memory_paths().get("model_dir"), local_only=True)
            hits = retrieval.search(conn, query, k=limit * 3, embedder=embedder)
            shaped = retrieval.hit_dicts(conn, hits, hits.mode)
            if not shaped["hits"]:
                return None
            groups: dict[str, list[dict]] = {}
            for h in shaped["hits"]:
                groups.setdefault(h["video_key"], []).append(h)
            parts = [f"# watch-recall: \"{query}\"\n",
                     f"Found {min(len(groups), limit)} video(s) in the video index ({shaped['mode']} search).\n"]
            for i, (key, hs) in enumerate(list(groups.items())[:limit], 1):
                v = conn.execute("SELECT * FROM videos WHERE video_key = ?", (key,)).fetchone()
                lines = [f"## {i}. {(v['title'] if v else None) or key}",
                         f"- Video: {(v['url'] if v else None) or '(source not recorded)'}"]
                where = []
                if v and v["paperless_doc_id"] is not None:
                    where.append(f"Paperless doc {v['paperless_doc_id']}")
                if v and v["nextcloud_path"]:
                    where.append(f"Nextcloud {v['nextcloud_path']}")
                if v and v["immich_album_id"]:
                    where.append(f"Immich album {v['immich_album_id']}")
                if where:
                    lines.append(f"- Library: {' | '.join(where)}")
                lines.append(f"- Key: `{key}` (ask: `watch.py ask {key} \"...\"`)")
                lines.append("")
                lines.extend(f"[{h['timestamp'] or '--:--'}] {h['kind']}: {h['snippet']}" for h in hs)
                parts.append("\n".join(lines))
                parts.append("\n---\n")
            return "\n".join(parts).rstrip() + "\n"
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001 - recall must fall through to Paperless
        print(f"[watch-recall] index lookup failed: {exc}", file=sys.stderr)
        return None


def recall(query: str, cfg: dict | None = None, limit: int = 5, full: bool = False) -> str:
    """Find saved /watch reports matching `query` and return a readable,
    agent-consumable summary. NEVER raises."""
    try:
        if cfg is None:
            cfg = library.load_config()
        cfg = cfg or {}
        header = f"# watch-recall: \"{query}\"\n"

        # 0) The video-memory index (transcript, OCR and notes with timestamps).
        indexed = _recall_from_index(query, limit)
        if indexed:
            return indexed

        # 1) Paperless full-text search.
        matches = search_paperless(query, cfg, limit=limit)
        if matches:
            parts = [header, f"Found {len(matches)} matching report(s) via paperless (full-text search).\n"]
            for i, m in enumerate(matches, 1):
                parts.append(_format_match(i, m["title"], m["source_url"], m["content"], m["doc_id"], "paperless", full))
                parts.append("\n---\n")
            return "\n".join(parts).rstrip() + "\n"

        # 2) Nextcloud folder-title fallback.
        folders = list_nextcloud_watch(cfg)
        q = query.lower()
        hits = [f for f in folders if q in f.lower()][:limit]
        if hits:
            parts = [header, f"Found {len(hits)} matching report(s) via nextcloud (folder-title match).\n"]
            for i, folder in enumerate(hits, 1):
                content = fetch_nextcloud_report(folder, cfg) or "(report.md could not be read)"
                parts.append(_format_match(i, folder, _extract_source(content), content, None, "nextcloud", full))
                parts.append("\n---\n")
            return "\n".join(parts).rstrip() + "\n"

        # 3) Nothing.
        hint = []
        if not cfg.get("PAPERLESS_TOKEN"):
            hint.append("PAPERLESS_TOKEN not set (Paperless search unavailable)")
        if not cfg.get("NEXTCLOUD_PASS"):
            hint.append("NEXTCLOUD_PASS not set (Nextcloud fallback unavailable)")
        suffix = f" — {'; '.join(hint)}" if hint else ""
        return (
            f"{header}\nNo matching reports found for \"{query}\"{suffix}.\n"
            "Either nothing has been watched on that topic yet, or the Paperless "
            "search index is stale (run `docker exec paperless document_index reindex`).\n"
        )
    except Exception as exc:
        return f"# watch-recall: \"{query}\"\n\nRecall failed: {exc}\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieve saved /watch reports by topic or title.")
    parser.add_argument("query", help="Topic or title to search for across saved watch reports.")
    parser.add_argument("--limit", type=int, default=5, help="Max reports to return (default 5).")
    parser.add_argument("--full", action="store_true", help="Print each report's full text, not a snippet.")
    args = parser.parse_args()
    print(recall(args.query, limit=args.limit, full=args.full))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
