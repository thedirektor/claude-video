#!/usr/bin/env python3
"""Playlist / channel / file-list enumeration and `watch.py batch`.

``entries(source)`` lists videos without downloading anything: a YouTube (or any
yt-dlp) playlist or channel via ``--flat-playlist`` (retried through the fallback
proxy when blocked), or a text file with one URL/path per line (``#`` comments).
``watch.py batch <source> [--limit N] [--force] [-- watch flags]`` runs each entry
serially as its own /watch run, skipping videos already in the video memory.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from config import memory_paths, setting
from download import run_ytdlp


def entries(source: str, limit: int | None = None) -> tuple[str, list[dict]]:
    """(title, [{id, url, title, duration}]) — newest first for channels."""
    p = Path(source).expanduser()
    if p.is_file():
        lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]
        urls = [ln for ln in lines if ln and not ln.startswith("#")]
        return p.name, [{"id": None, "url": u, "title": None, "duration": None} for u in urls[:limit or None]]
    args = ["--flat-playlist", "--dump-single-json", "--no-warnings"]
    if limit:
        args += ["--playlist-end", str(int(limit))]
    r = run_ytdlp([*args, "--", source], capture=True)
    proxy = setting("WATCH_YTDLP_FALLBACK_PROXY")
    if r.returncode != 0 and proxy:
        r = run_ytdlp([*args, "--proxy", proxy, "--socket-timeout", "20", "--", source], capture=True)
    if r.returncode != 0 or not (r.stdout or "").strip():
        raise RuntimeError(f"could not list {source}: {(r.stderr or '').strip()[-300:]}")
    data = json.loads(r.stdout)
    items = []
    for e in data.get("entries") or ([data] if data.get("id") else []):
        vid = e.get("id")
        url = e.get("url") or e.get("webpage_url")
        if not url and vid and (e.get("ie_key") or "").lower() == "youtube":
            url = f"https://www.youtube.com/watch?v={vid}"
        if url:
            items.append({"id": vid, "url": url, "title": e.get("title"), "duration": e.get("duration")})
    return data.get("title") or source, items


def is_indexed(url: str) -> bool:
    """True when the video memory already holds this video. Never raises."""
    try:
        import identity
        from memory import index

        key = identity.video_key(url, {})
        conn = index.connect(memory_paths()["index_db"])
        try:
            return conn.execute("SELECT 1 FROM videos WHERE video_key = ?", (key,)).fetchone() is not None
        finally:
            conn.close()
    except Exception:  # noqa: BLE001
        return False


def _run_one(url: str, flags: list[str]) -> int:
    script = Path(__file__).resolve().parent / "watch.py"
    return subprocess.run([sys.executable, str(script), *flags, url]).returncode


def main(argv: list[str]) -> int:
    flags: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, flags = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(prog="watch.py batch", description="Watch every video of a playlist, "
                                 "channel or URL list (serial; already-indexed videos are skipped).")
    ap.add_argument("source", help="playlist / channel URL, or a text file with one URL or path per line")
    ap.add_argument("--limit", type=int, default=None, help="only the first N entries (newest first for channels)")
    ap.add_argument("--force", action="store_true", help="also re-watch videos that are already indexed")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    title, items = entries(a.source, a.limit)
    ok = failed = skipped = 0
    results = []
    for item in items:
        if not a.force and is_indexed(item["url"]):
            skipped += 1
            results.append({**item, "status": "skipped"})
            continue
        print(f"[watch] batch: {item.get('title') or item['url']}", file=sys.stderr)
        rc = _run_one(item["url"], flags)
        ok += rc == 0
        failed += rc != 0
        results.append({**item, "status": "ok" if rc == 0 else f"failed ({rc})"})
    if a.json:
        print(json.dumps({"source": a.source, "title": title, "ok": ok, "failed": failed, "skipped": skipped,
                          "items": results}, ensure_ascii=False, indent=2))
    else:
        print(f"# watch batch: {title}\n- entries: {len(items)}\n- ok: {ok}\n- failed: {failed}\n"
              f"- skipped (already indexed): {skipped}")
        for r in results:
            print(f"  - [{r['status']}] {r.get('title') or ''} {r['url']}")
    return 0 if failed == 0 else 1
