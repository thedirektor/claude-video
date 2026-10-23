#!/usr/bin/env python3
"""Push local runs to a central watch service (``WATCH_REMOTE_URL`` + ``WATCH_REMOTE_TOKEN``).

After a successful local run the video's bundle (see ``memory/bundle.py``) is POSTed to
``<remote>/runs/import``. A failed push stays in the outbox
(``WATCH_OUTBOX_DIR``, default ``~/.local/share/watch/outbox``) and is retried on the
next run or with ``watch.py push --retry``; a bundle the server rejects (4xx) moves to
``outbox/rejected``. ``watch.py push --all`` sends the whole local history. Never raises.
"""
from __future__ import annotations

import argparse
import json
import shutil
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from config import memory_paths, setting


def enabled() -> bool:
    return bool(setting("WATCH_REMOTE_URL") and setting("WATCH_REMOTE_TOKEN"))


def outbox() -> Path:
    d = Path(setting("WATCH_OUTBOX_DIR") or Path.home() / ".local" / "share" / "watch" / "outbox").expanduser()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _post(url: str, path: str, token: str) -> tuple[int | None, dict | None]:
    data = Path(path).read_bytes()
    req = urllib.request.Request(url, data=data, method="POST", headers={
        "Authorization": f"Bearer {token}", "Content-Type": "application/zip", "User-Agent": "watch-push"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:  # noqa: S310 - configured endpoint
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read() or b"{}")
        except ValueError:
            return exc.code, None
    except Exception as exc:  # noqa: BLE001 - offline / DNS / power cut at the far end
        print(f"[watch] push: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None, None


def _send(path: Path, post) -> bool | None:
    """True sent, False rejected (moved aside), None retry later."""
    url = setting("WATCH_REMOTE_URL").rstrip("/") + "/runs/import"
    code, body = post(url, str(path), setting("WATCH_REMOTE_TOKEN"))
    if code is not None and 200 <= code < 300:
        path.unlink(missing_ok=True)
        return True
    if code is not None and 400 <= code < 500 and code not in (401, 403, 408, 429):
        rej = path.parent / "rejected"
        rej.mkdir(exist_ok=True)
        shutil.move(str(path), rej / path.name)
        print(f"[watch] push: rejected ({code}): {body}", file=sys.stderr)
        return False
    return None


def push_video(key: str, *, library_saved: bool = False, post=None) -> dict:
    if not enabled():
        return {"ok": False, "reason": "WATCH_REMOTE_URL / WATCH_REMOTE_TOKEN not set"}
    post = post or _post
    try:
        from memory import bundle, index

        conn = index.connect(memory_paths()["index_db"])
        try:
            name = f"{int(time.time())}-{key.replace(':', '_')}.zip"
            path = bundle.make_bundle(conn, key, outbox() / name, host=socket.gethostname(),
                                      library_saved=library_saved)
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"bundle failed: {exc}"}
    status = _send(path, post)
    if status is None:
        print(f"[watch] push: {key} kept in the outbox for retry", file=sys.stderr)
    return {"ok": status is True, "bundle": path.name, "queued": status is None}


def flush_outbox(post=None) -> dict:
    if not enabled():
        return {"sent": 0, "failed": 0}
    post = post or _post
    sent = failed = 0
    for path in sorted(outbox().glob("*.zip")):
        status = _send(path, post)
        sent += status is True
        failed += status is not True
        if status is None:
            break  # the remote is unreachable; keep order for next time
    return {"sent": sent, "failed": failed}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="watch.py push", description="Send local runs to the central watch service.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true", help="bundle and send every indexed video")
    g.add_argument("--retry", action="store_true", help="only retry the outbox")
    g.add_argument("--video", help="one video_key")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if not enabled():
        print("watch push: set WATCH_REMOTE_URL and WATCH_REMOTE_TOKEN in ~/.config/watch/.env", file=sys.stderr)
        return 2
    res: dict = {"outbox": flush_outbox(post=_post)}
    keys: list[str] = []
    if a.all or a.video:
        from memory import index

        conn = index.connect(memory_paths()["index_db"])
        try:
            keys = [a.video] if a.video else [r[0] for r in conn.execute(
                "SELECT video_key FROM videos WHERE video_key IN (SELECT video_key FROM runs WHERE origin != 'import')")]
        finally:
            conn.close()
    results = [push_video(k, post=_post) for k in keys]
    res.update(pushed=sum(r["ok"] for r in results), queued=sum(bool(r.get("queued")) for r in results),
               errors=[r["reason"] for r in results if r.get("reason")])
    print(json.dumps(res, indent=2) if a.json else
          f"pushed {res['pushed']}, queued {res['queued']}, outbox sent {res['outbox']['sent']}")
    return 0 if not res["errors"] else 1
