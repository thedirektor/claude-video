#!/usr/bin/env python3
"""`watch.py doctor`: check what /watch depends on, and repair what is safe to repair.

Checks ffmpeg/ffprobe, yt-dlp (version + an online canary extraction), tesseract
languages, the embedding model, index/store disk usage, the configured library and
LLM endpoints (only those set), and which API keys are present (never their values).
`--fix` runs the safe repairs: a forced yt-dlp upgrade and the embedding-model
download; missing system packages only get a hint.
"""
from __future__ import annotations

import argparse
import json
import os
import importlib.util
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import download
from config import memory_paths, setting

CANARY_URL = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
PROXY_PROBE_URL = "https://www.youtube.com/generate_204"
LOW_DISK_BYTES = 2 * 1024**3
ENDPOINTS = (("immich", "IMMICH_BASE_URL"), ("nextcloud", "NEXTCLOUD_URL"),
             ("paperless", "PAPERLESS_URL"), ("llm proxy", "WATCH_LLM_BASE_URL"))
_ICON = {"ok": "✓", "warn": "!", "fail": "✗"}


def _check(name: str, status: str, detail: str, **extra) -> dict:
    return {"name": name, "status": status, "detail": detail, **extra}


def _first_line(text: str) -> str:
    return next((ln.strip() for ln in (text or "").splitlines() if ln.strip()), "")


def _last_error(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    errs = [ln for ln in lines if ln.startswith("ERROR")]
    return (errs or lines or [""])[-1][:200]


def _http_status(url: str) -> int | None:
    req = urllib.request.Request(url, method="GET", headers={"User-Agent": "watch-doctor"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310 - configured endpoint
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except Exception:  # noqa: BLE001 - unreachable
        return None


def _embedder_status() -> bool | None:
    """True loaded, False unavailable, None disabled by WATCH_EMBEDDINGS=0."""
    from memory import embeddings

    if not embeddings.embeddings_enabled():
        return None
    return embeddings.warm_up()


def _fetch_model() -> bool:
    from memory import embeddings

    return embeddings.get_embedder(local_only=False) is not None


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except OSError:
                pass
    return total


def _gb(n: int) -> str:
    return f"{n / 1024**3:.2f} GB" if n >= 1024**3 else f"{n / 1024**2:.1f} MB"


def _keys() -> list[dict]:
    import gemini
    import openrouter
    import transcriptapi

    out = []
    for name, loader, without in (
        ("gemini key", gemini.load_api_key, "Gemini backend, YouTube transcript fallback and visual log off"),
        ("transcriptapi key", transcriptapi.load_api_key, "YouTube transcripts use captions/Gemini instead"),
        ("openrouter key", openrouter.load_api_key, "OpenRouter backend off"),
    ):
        try:
            present = bool(loader())
        except Exception:  # noqa: BLE001
            present = False
        out.append(_check(name, "ok" if present else "warn", "set" if present else f"not set — {without}"))
    return out


def _proxy_status(proxy: str) -> int | None:
    handler = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
    try:
        with urllib.request.build_opener(handler).open(PROXY_PROBE_URL, timeout=15) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except Exception:  # noqa: BLE001 - unreachable proxy (e.g. the home machine is off)
        return None


def run_checks(*, online: bool = True, which=shutil.which, run=subprocess.run, http=_http_status,
               embedder_ok=_embedder_status, proxy_status=_proxy_status) -> dict:
    checks: list[dict] = []

    def version(cmd: list[str]) -> tuple[bool, str]:
        try:
            r = run(cmd, capture_output=True, text=True, timeout=30, env=download.ytdlp_env())
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"
        return r.returncode == 0, _first_line(r.stdout or r.stderr)

    for binary in ("ffmpeg", "ffprobe"):
        if which(binary) is None:
            checks.append(_check(binary, "fail", "not found — install ffmpeg"))
        else:
            ok, line = version([binary, "-version"])
            checks.append(_check(binary, "ok" if ok else "fail", line or "unreadable version"))

    ytdlp = download.ytdlp_cmd()
    if ytdlp == ["yt-dlp"] and which("yt-dlp") is None:
        checks.append(_check("yt-dlp", "fail", "not found — pip install -U yt-dlp", fixable=True))
    else:
        ok, line = version([*ytdlp, "--version"])
        where = f" (self-healed copy in {setting('WATCH_YTDLP_PYLIB')})" if ytdlp[0] != "yt-dlp" else ""
        checks.append(_check("yt-dlp", "ok" if ok else "fail", f"{line}{where}", fixable=not ok))
        if online:
            canary = setting("WATCH_DOCTOR_CANARY") or CANARY_URL
            try:
                r = run([*ytdlp, "--skip-download", "--dump-single-json", "--no-warnings", "--no-playlist",
                         "--", canary], capture_output=True, text=True, timeout=120, env=download.ytdlp_env())
                info = json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else {}
            except Exception as exc:  # noqa: BLE001
                r, info = SimpleResult(1, "", f"{type(exc).__name__}: {exc}"), {}
            if info.get("id"):
                checks.append(_check("yt-dlp canary", "ok", f"extracted {info.get('title') or info['id']!r}"))
            else:
                err = (r.stderr or r.stdout or "")
                broken = download.is_breakage(err)
                checks.append(_check("yt-dlp canary", "fail" if broken else "warn",
                                     _last_error(err) or "no metadata returned", fixable=broken))

    if which("tesseract") is None:
        checks.append(_check("tesseract", "warn", "not found — on-screen text (OCR) disabled; install tesseract-ocr "
                             "(breakdown fields affected: typography, region lookup)"))
    else:
        ok, _ = version(["tesseract", "--version"])
        try:
            r = run(["tesseract", "--list-langs"], capture_output=True, text=True, timeout=30)
            langs = {ln.strip() for ln in (r.stdout or "").splitlines()[1:] if ln.strip()}
        except Exception:  # noqa: BLE001
            langs = set()
        missing = [lang for lang in ("spa", "eng") if lang not in langs]
        checks.append(_check("tesseract", "warn" if missing else "ok",
                             f"missing language data: {', '.join(missing)} — install tesseract-ocr-{missing[0]}"
                             if missing else f"languages: {', '.join(sorted(langs))}"))

    for mod, label, pip, affected in (("cv2", "opencv", "opencv-python-headless", "camera, look, typography, motion"),
                                      ("numpy", "numpy", "numpy", "look, typography, audio, motion, compare")):
        if importlib.util.find_spec(mod) is None:
            checks.append(_check(label, "warn",
                                 f"not installed — breakdown fields affected: {affected} (reported as "
                                 f"'not determined'); pip install {pip}"))
        else:
            checks.append(_check(label, "ok", "installed"))

    emb = embedder_ok()
    if emb is None:
        checks.append(_check("embedding model", "ok", "disabled (WATCH_EMBEDDINGS=0): full-text search only"))
    else:
        checks.append(_check("embedding model", "ok" if emb else "warn",
                             "loaded" if emb else "not downloaded — semantic search off", fixable=not emb))

    paths = memory_paths()
    db, store = paths["index_db"], paths["store_dir"]
    used = (_size(db) if db.exists() else 0) + (_size(store) if store.exists() else 0)
    probe = next((p for p in (store, store.parent, db.parent) if p.exists()), Path.home())
    free = shutil.disk_usage(probe).free
    checks.append(_check("storage", "warn" if free < LOW_DISK_BYTES else "ok",
                         f"index+store {_gb(used)} at {store.parent}; {_gb(free)} free"))

    if online:
        for name, var in ENDPOINTS:
            url = setting(var)
            if not url:
                continue
            code = http(url)
            reachable = code is not None and code < 500
            checks.append(_check(name, "ok" if reachable else "warn",
                                 f"{url} reachable (HTTP {code})" if reachable
                                 else f"{url} unreachable" + (f" (HTTP {code})" if code else "")))

        proxy = setting("WATCH_YTDLP_FALLBACK_PROXY")
        if proxy:
            code = proxy_status(proxy)
            ok = code is not None and code < 400
            checks.append(_check("fallback proxy", "ok" if ok else "warn",
                                 f"{proxy} → YouTube HTTP {code}" if code is not None
                                 else f"{proxy} unreachable — blocked downloads fall back to storyboards"))

    checks.extend(_keys())
    return {"ok": not any(c["status"] == "fail" for c in checks), "checks": checks}


class SimpleResult:
    def __init__(self, returncode: int, stdout: str, stderr: str) -> None:
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def apply_fixes(report: dict, *, fetch_model=_fetch_model) -> list[dict]:
    """Run the safe repairs for every fixable check; returns what was attempted."""
    done: list[dict] = []
    names = {c["name"] for c in report["checks"] if c.get("fixable") and c["status"] != "ok"}
    if names & {"yt-dlp", "yt-dlp canary"}:
        ok = download.self_heal(force=True)
        done += [{"name": n, "fixed": ok, "action": "upgraded yt-dlp"} for n in sorted(names & {"yt-dlp", "yt-dlp canary"})]
    if "embedding model" in names:
        try:
            ok = fetch_model()
        except Exception:  # noqa: BLE001
            ok = False
        done.append({"name": "embedding model", "fixed": ok, "action": "downloaded the embedding model"})
    return done


def render(report: dict, fixes: list[dict] | None = None) -> str:
    lines = [f"{_ICON[c['status']]} {c['name']}: {c['detail']}" for c in report["checks"]]
    for f in fixes or []:
        lines.append(f"→ fix {f['name']}: {f['action']} — {'done' if f['fixed'] else 'failed'}")
    lines.append("overall: " + ("healthy" if report["ok"] else "problems found"))
    hint = [c for c in report["checks"] if c.get("fixable") and c["status"] != "ok"]
    if hint and fixes is None:
        lines.append("run `watch.py doctor --fix` to repair: " + ", ".join(c["name"] for c in hint))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="watch.py doctor", description="Check /watch dependencies.")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--fix", action="store_true", help="run safe repairs (yt-dlp upgrade, model download)")
    ap.add_argument("--offline", action="store_true", help="skip network checks (canary, endpoints)")
    args = ap.parse_args(argv)
    report = run_checks(online=not args.offline)
    fixes = None
    if args.fix:
        fixes = apply_fixes(report)
        if fixes:
            report = run_checks(online=not args.offline)
    if args.json:
        print(json.dumps({**report, "fixes": fixes or []}, ensure_ascii=False, indent=2))
    else:
        print(render(report, fixes))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
