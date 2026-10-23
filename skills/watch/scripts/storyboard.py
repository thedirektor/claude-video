#!/usr/bin/env python3
"""YouTube storyboard frames: a no-account fallback when the media download is blocked.

YouTube publishes low-res seek-preview "storyboards" (sheets of 320x180 tiles, one
tile every few seconds). They come from the public player response, so yt-dlp can
fetch them from datacenter IPs where the video itself is bot-walled. The sheets are
split with ffmpeg's ``untile`` filter and each tile becomes a frame with its time.
"""
from __future__ import annotations

import email
import json
import subprocess
import sys
from pathlib import Path

from download import HARDENING_ARGS, _read_info, run_ytdlp

FORMAT_CHAIN = "sb0/sb1/sb2/sb3"


def _is_storyboard(fmt: dict) -> bool:
    return str(fmt.get("format_id", "")).startswith("sb") and bool(fmt.get("fragments"))


def pick_format(info: dict) -> dict | None:
    """The highest-resolution storyboard format in a yt-dlp info dict, or None."""
    boards = [f for f in info.get("formats") or [] if _is_storyboard(f)]
    if not boards:
        return None
    chosen = next((f for f in boards if f.get("format_id") == info.get("format_id")), None)
    return chosen or max(boards, key=lambda f: (f.get("width") or 0) * (f.get("height") or 0))


def tile_times(fmt: dict, duration: float | None) -> list[list[float]]:
    """Per sheet, the start time of each tile (tiles at/after ``duration`` dropped)."""
    per_sheet = max(1, int(fmt.get("rows") or 1) * int(fmt.get("columns") or 1))
    fps = fmt.get("fps")
    out: list[list[float]] = []
    start = 0.0
    for frag in fmt.get("fragments") or []:
        dur = float(frag.get("duration") or 0)
        step = 1.0 / fps if fps else (dur / per_sheet if dur else 0)
        sheet: list[float] = []
        for k in range(per_sheet):
            t = round(start + k * step, 2)
            if step <= 0 or k * step >= dur - 1e-6 or (duration and t >= duration):
                break
            sheet.append(t)
        out.append(sheet)
        start += dur
    return out


def parse_mhtml(path: Path) -> list[bytes]:
    """The image payloads of a yt-dlp storyboard .mhtml, in sheet order."""
    msg = email.message_from_bytes(Path(path).read_bytes())
    return [p.get_payload(decode=True) for p in msg.walk() if p.get_content_maintype() == "image"]


def _probe_size(path: Path) -> tuple[int, int]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, timeout=30).stdout.strip()
    w, h = (int(x) for x in out.split(",")[:2])
    return w, h


def split_sheet(sheet: Path, tile_w: int, tile_h: int, out_prefix: Path) -> list[Path]:
    """Split one sheet into its tiles (row-major), sized from the sheet itself."""
    w, h = _probe_size(sheet)
    cols, rows = max(1, round(w / tile_w)), max(1, round(h / tile_h))
    pattern = f"{out_prefix}_%03d.jpg"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(sheet), "-vf", f"untile={cols}x{rows}",
                    "-q:v", "3", pattern], check=True, timeout=60)
    return sorted(out_prefix.parent.glob(f"{out_prefix.name}_*.jpg"))


def _even(items: list, n: int) -> list:
    if n <= 0 or len(items) <= n:
        return items
    if n == 1:
        return [items[len(items) // 2]]
    return [items[round(i * (len(items) - 1) / (n - 1))] for i in range(n)]


def fetch(url: str, out_dir: Path, *, start: float | None = None, end: float | None = None,
          max_frames: int | None = None) -> tuple[list[dict], dict, dict]:
    """Download the best storyboard and return (frames, frame_meta, info).

    Raises RuntimeError when the video has no storyboard or the fetch fails.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[watch] media download blocked — fetching the YouTube storyboard for frames…", file=sys.stderr)
    result = run_ytdlp(["-f", FORMAT_CHAIN, "--write-info-json", "--no-playlist", "--no-progress", *HARDENING_ARGS,
                        "-o", str(out_dir / "sb.%(ext)s"), "--", url])
    info_path, mhtml = out_dir / "sb.info.json", out_dir / "sb.mhtml"
    if not info_path.exists() or not mhtml.exists():
        raise RuntimeError(f"no storyboard available (yt-dlp exit {result.returncode})")
    raw = json.loads(info_path.read_text(encoding="utf-8"))
    fmt = pick_format(raw)
    if fmt is None:
        raise RuntimeError("no storyboard format in the video metadata")
    duration = raw.get("duration")
    times = tile_times(fmt, duration)
    sheets = parse_mhtml(mhtml)
    if not sheets:
        raise RuntimeError("storyboard file holds no images")

    tiles: list[tuple[float, Path]] = []
    for i, (data, sheet_times) in enumerate(zip(sheets, times)):
        if not sheet_times:
            continue
        sheet = out_dir / f"sheet_{i:03d}.img"
        sheet.write_bytes(data)
        paths = split_sheet(sheet, int(fmt["width"]), int(fmt["height"]), out_dir / f"tile_{i:03d}")
        sheet.unlink(missing_ok=True)
        tiles.extend(zip(sheet_times, paths))

    in_range = [(t, p) for t, p in tiles
                if (start is None or t >= start) and (end is None or t <= end)]
    chosen = _even(in_range, max_frames) if max_frames else in_range
    keep = {p for _, p in chosen}
    for _, p in tiles:
        if p not in keep:
            p.unlink(missing_ok=True)
    frames = [{"timestamp_seconds": t, "path": str(p), "reason": "storyboard"} for t, p in chosen]
    step = round(times[0][1] - times[0][0], 2) if times and len(times[0]) > 1 else None
    meta = {"engine": "storyboard", "candidate_count": len(in_range), "selected_count": len(frames),
            "fallback": False, "tile": f"{fmt['width']}x{fmt['height']}", "interval_s": step}
    return frames, meta, _read_info(info_path, url)
