#!/usr/bin/env python3
"""Download a video via yt-dlp, or resolve a local file path.

Also fetches subtitles (manual first, then auto-generated) in VTT format so
transcribe.py can parse them without needing Whisper.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from config import setting

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi", ".flv", ".wmv"}

# Default subtitle language preference. Spanish first (owner's primary content),
# then English, then any original-language track. yt-dlp treats each token as a
# REGEX (matched against available track ids), tries them in order and writes
# every match; _pick_subtitle then selects by this same preference.
# `.*-orig` catches the uploader's original auto-caption track (e.g. `en-orig`)
# when localized tracks are absent — it MUST be a valid regex, so `.*-orig`, not
# `*-orig` (a bare `*` makes yt-dlp abort with "Wrong regex for subtitlelangs").
# Never "all" — that pulls YouTube's hundreds of auto-translated tracks and
# stalls the run. Sources: PRs #12, #26, #30.
DEFAULT_SUB_LANGS = "es,es-419,es-ES,en,en-US,en-GB,.*-orig"

# yt-dlp hardening. YouTube's SABR/403 rollout breaks the default `web` client;
# a player_client fallback chain (tried in order) keeps most public videos
# reachable without cookies. Anti-bot retry + request jitter absorb transient
# 403/429s. Sources: upstream PRs #46, #42, #27, #21.
HARDENING_ARGS = [
    "--extractor-args", "youtube:player_client=default,android,mweb",
    "--retries", "10",
    "--fragment-retries", "10",
    "--extractor-retries", "3",
    "--sleep-requests", "1",
    "--sleep-interval", "1",
    "--max-sleep-interval", "5",
]


# Extractor breakage (yt-dlp is out of date with the site), as opposed to an
# access block (403 / bot wall / unavailable) that an upgrade cannot fix.
_BREAKAGE = re.compile(
    r"unable to extract|failed to extract any player response|signature extraction failed"
    r"|nsig extraction failed|please report this issue on https://github\.com/yt-dlp",
    re.IGNORECASE,
)
SELF_HEAL_WINDOW_S = 6 * 3600
# Residential fallback (e.g. tinyproxy on a home machine over Tailscale): one retry,
# capped, when the normal egress is blocked. Off unless WATCH_YTDLP_FALLBACK_PROXY is set.
FALLBACK_FORMAT = "bv*[height<=480]+ba/b[height<=480]/b"
FALLBACK_SOCKET_TIMEOUT_S = "20"


def is_breakage(stderr: str) -> bool:
    """True when yt-dlp's stderr says its extractor is broken (an upgrade may fix it)."""
    return bool(_BREAKAGE.search(stderr or ""))


def _pylib() -> Path | None:
    value = setting("WATCH_YTDLP_PYLIB")
    return Path(value).expanduser() if value else None


def ytdlp_cmd() -> list[str]:
    """argv prefix for yt-dlp: the self-healed copy in WATCH_YTDLP_PYLIB when present."""
    lib = _pylib()
    if lib is not None and (lib / "yt_dlp" / "__init__.py").exists():
        return [sys.executable, "-m", "yt_dlp"]
    return ["yt-dlp"]


def ytdlp_env() -> dict[str, str]:
    env = dict(os.environ)
    lib = _pylib()
    if lib is not None:
        env["PYTHONPATH"] = os.pathsep.join(p for p in (str(lib), env.get("PYTHONPATH", "")) if p)
    return env


def _ytdlp_available() -> bool:
    return ytdlp_cmd()[0] != "yt-dlp" or shutil.which("yt-dlp") is not None


def run_ytdlp(args: list[str], capture: bool = False):
    """Run yt-dlp (progress live on stderr), returning the result with stderr captured + echoed.
    ``capture=True`` captures stdout too (e.g. ``--dump-single-json``)."""
    result = subprocess.run([*ytdlp_cmd(), *args], stdout=subprocess.PIPE if capture else sys.stderr,
                            stderr=subprocess.PIPE, text=True, env=ytdlp_env())
    if result.stderr:
        sys.stderr.write(result.stderr)
    return result


def _stamp_path() -> Path:
    lib = _pylib()
    if lib is not None:
        return lib / ".self-heal-stamp"
    base = setting("WATCH_STATE_DIR") or str(Path.home() / ".cache" / "watch")
    return Path(base).expanduser() / "ytdlp-self-heal-stamp"


def self_heal(clock=time.time, force: bool = False) -> bool:
    """Upgrade yt-dlp at most once per SELF_HEAL_WINDOW_S. Never raises.

    With WATCH_YTDLP_PYLIB set (the service: the image is read-only), installs into
    that directory, which ytdlp_cmd() then prefers; otherwise `pip install --user`.
    WATCH_YTDLP_SELF_HEAL=0 disables it; ``force`` (doctor --fix) skips the rate limit.
    Returns True when an upgrade ran successfully.
    """
    if (setting("WATCH_YTDLP_SELF_HEAL") or "1").strip().lower() in ("0", "false", "no"):
        return False
    try:
        stamp = _stamp_path()
        now = clock()
        try:
            if not force and now - float(stamp.read_text().strip()) < SELF_HEAL_WINDOW_S:
                return False
        except (OSError, ValueError):
            pass
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(str(now))
        lib = _pylib()
        target = ["--target", str(lib)] if lib is not None else ["--user"]
        print("[watch] yt-dlp looks broken — upgrading it once and retrying…", file=sys.stderr)
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "--upgrade-strategy",
                            "only-if-needed", *target, "yt-dlp"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)
        if r.returncode != 0:
            print(f"[watch] yt-dlp upgrade failed (exit {r.returncode})", file=sys.stderr)
            return False
        return True
    except Exception as exc:  # noqa: BLE001 - self-heal is best-effort
        print(f"[watch] yt-dlp self-heal skipped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return False


def _fallback_args(args: list[str], proxy: str, audio_only: bool) -> list[str]:
    """The download argv re-pointed at the fallback proxy: ≤480p, size-capped, quick timeout."""
    out = list(args)
    if not audio_only:
        out[out.index("-f") + 1] = FALLBACK_FORMAT
    try:
        max_mb = max(1, int(setting("WATCH_FALLBACK_MAX_MB") or "400"))
    except ValueError:
        max_mb = 400
    pos = out.index("--")
    out[pos:pos] = ["--proxy", proxy, "--max-filesize", f"{max_mb}M",
                    "--socket-timeout", FALLBACK_SOCKET_TIMEOUT_S]
    return out


def _cookie_args(cookies_from_browser: str | None, cookies_file: str | None) -> list[str]:
    """Build the yt-dlp cookie flags. Off by default — both None yields []."""
    args: list[str] = []
    if cookies_from_browser:
        args += ["--cookies-from-browser", cookies_from_browser]
    if cookies_file:
        args += ["--cookies", cookies_file]
    return args


def is_url(source: str) -> bool:
    if source.startswith("-"):
        return False
    parsed = urlparse(source)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def resolve_local(path: str) -> dict:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise SystemExit(f"File not found: {p}")
    if p.suffix.lower() not in VIDEO_EXTS:
        print(
            f"[watch] warning: {p.suffix} is not a known video extension, proceeding anyway",
            file=sys.stderr,
        )
    return {
        "video_path": str(p),
        "subtitle_path": None,
        "info": {"title": p.name, "url": str(p)},
        "downloaded": False,
    }


def _pick_subtitle(out_dir: Path, sub_langs: str = DEFAULT_SUB_LANGS) -> Path | None:
    candidates = sorted(out_dir.glob("video*.vtt"))
    if not candidates:
        return None
    # Preference follows the requested language order. For "es,en,*-orig" we look
    # for video.es.vtt, then video.en.vtt, then any *-orig track, then anything.
    markers: list[str] = []
    for lang in sub_langs.split(","):
        lang = lang.strip()
        if not lang:
            continue
        if lang.endswith("-orig"):
            markers.append("-orig.")
        else:
            markers.append(f".{lang}.")
    for marker in markers:
        for c in candidates:
            if marker in c.name:
                return c
    return candidates[0]


def _pick_video(out_dir: Path) -> Path | None:
    for ext in (".mp4", ".mkv", ".webm", ".mov", ".m4a", ".mp3", ".opus"):
        for candidate in out_dir.glob(f"video*{ext}"):
            return candidate
    for candidate in out_dir.glob("video.*"):
        if candidate.suffix.lower() in VIDEO_EXTS:
            return candidate
    return None


def fetch_captions(
    url: str,
    out_dir: Path,
    sub_langs: str = DEFAULT_SUB_LANGS,
    cookies_from_browser: str | None = None,
    cookies_file: str | None = None,
) -> dict:
    """Fetch metadata and best available VTT captions without downloading video."""
    if not _ytdlp_available():
        raise SystemExit("yt-dlp is not installed. Install with: brew install yt-dlp")

    out_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(out_dir / "video.%(ext)s")
    args = [
        "--skip-download",
        "--write-info-json",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs", sub_langs,
        "--sub-format", "vtt",
        "--convert-subs", "vtt",
        "--no-playlist",
        "--ignore-errors",
        *HARDENING_ARGS,
        *_cookie_args(cookies_from_browser, cookies_file),
        "-o", output_template,
        "--",
        url,
    ]
    result = run_ytdlp(args)
    if not (out_dir / "video.info.json").exists() and is_breakage(result.stderr) and self_heal():
        result = run_ytdlp(args)
    proxy = setting("WATCH_YTDLP_FALLBACK_PROXY")
    if not (out_dir / "video.info.json").exists() and proxy and not is_breakage(result.stderr):
        print("[watch] metadata/captions blocked — retrying once via the fallback proxy…", file=sys.stderr)
        pos = args.index("--")
        run_ytdlp([*args[:pos], "--proxy", proxy, "--socket-timeout", FALLBACK_SOCKET_TIMEOUT_S, *args[pos:]])
    subtitle = _pick_subtitle(out_dir, sub_langs)
    info = _read_info(out_dir / "video.info.json", url)
    return {
        "video_path": None,
        "subtitle_path": str(subtitle) if subtitle else None,
        "info": info or {"url": url},
        "downloaded": False,
    }


def _read_info(info_path: Path, url: str) -> dict:
    info: dict = {}
    if info_path.exists():
        try:
            raw = json.loads(info_path.read_text(encoding="utf-8"))
            info = {
                "title": raw.get("title"),
                "uploader": raw.get("uploader") or raw.get("channel"),
                "duration": raw.get("duration"),
                "url": raw.get("webpage_url") or url,
            }
        except Exception as exc:
            print(f"[watch] info.json parse failed: {exc}", file=sys.stderr)
            info = {"url": url}
    return info


def download_url(
    url: str,
    out_dir: Path,
    audio_only: bool = False,
    sub_langs: str = DEFAULT_SUB_LANGS,
    cookies_from_browser: str | None = None,
    cookies_file: str | None = None,
) -> dict:
    if not _ytdlp_available():
        raise SystemExit("yt-dlp is not installed. Install with: brew install yt-dlp")

    out_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(out_dir / "video.%(ext)s")

    fmt = "ba/bestaudio" if audio_only else "bv*[height<=720]+ba/b[height<=720]/bv+ba/b"
    args = [
        "-N", "8",
        "-f", fmt,
        "--merge-output-format", "mp4",
        "--write-info-json",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs", sub_langs,
        "--sub-format", "vtt",
        "--convert-subs", "vtt",
        "--no-playlist",
        "--ignore-errors",
        *HARDENING_ARGS,
        *_cookie_args(cookies_from_browser, cookies_file),
        "-o", output_template,
        "--",
        url,
    ]

    # yt-dlp may exit non-zero if a subtitle variant fails (e.g. 429) even when
    # the video itself downloaded fine. Treat "video file present" as success.
    result = run_ytdlp(args)
    video = _pick_video(out_dir)
    if video is None and is_breakage(result.stderr) and self_heal():
        result = run_ytdlp(args)
        video = _pick_video(out_dir)
    via_proxy = False
    proxy = setting("WATCH_YTDLP_FALLBACK_PROXY")
    if video is None and proxy and not is_breakage(result.stderr):
        print("[watch] download blocked — retrying once via the fallback proxy (≤480p)…", file=sys.stderr)
        result = run_ytdlp(_fallback_args(args, proxy, audio_only))
        video = _pick_video(out_dir)
        via_proxy = video is not None
    if video is None:
        raise SystemExit(
            f"yt-dlp did not produce a video file in {out_dir} (exit {result.returncode})"
        )

    subtitle = _pick_subtitle(out_dir, sub_langs)
    info = _read_info(out_dir / "video.info.json", url)

    return {
        "video_path": str(video),
        "subtitle_path": str(subtitle) if subtitle else None,
        "info": info or {"url": url},
        "downloaded": True,
        "via_fallback_proxy": via_proxy,
    }


def download(
    source: str,
    out_dir: Path,
    audio_only: bool = False,
    sub_langs: str = DEFAULT_SUB_LANGS,
    cookies_from_browser: str | None = None,
    cookies_file: str | None = None,
) -> dict:
    if is_url(source):
        return download_url(
            source,
            out_dir,
            audio_only=audio_only,
            sub_langs=sub_langs,
            cookies_from_browser=cookies_from_browser,
            cookies_file=cookies_file,
        )
    return resolve_local(source)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: download.py <url-or-path> <out-dir>", file=sys.stderr)
        raise SystemExit(2)
    result = download(sys.argv[1], Path(sys.argv[2]))
    print(json.dumps(result, indent=2))
