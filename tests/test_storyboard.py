"""YouTube storyboard frames: mhtml parsing, tile timing, untile, range + cap."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import storyboard  # noqa: E402

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")

URL = "https://www.youtube.com/watch?v=XUqrvbsTfck"

# 3x3 tiles of 32x18, one tile every 10 s → 90 s per sheet; 2 sheets, the
# second only 2 rows tall (a partially filled last sheet), duration 140 s.
FMT = {"format_id": "sb0", "width": 32, "height": 18, "rows": 3, "columns": 3, "fps": 0.1,
       "fragments": [{"duration": 90.0}, {"duration": 50.0}]}
INFO = {"title": "t", "uploader": "u", "duration": 140, "webpage_url": URL, "format_id": "sb0",
        "formats": [{"format_id": "sb2", "width": 8, "height": 4, "rows": 1, "columns": 1,
                     "fragments": [{"duration": 140.0}]}, FMT,
                    {"format_id": "18", "width": 640, "height": 360}]}


def test_pick_format_prefers_largest_storyboard():
    assert storyboard.pick_format(INFO)["format_id"] == "sb0"


def test_pick_format_none_without_storyboards():
    assert storyboard.pick_format({"formats": [{"format_id": "18", "width": 640}]}) is None


def test_tile_times_uses_fps_interval_and_stops_at_duration():
    times = storyboard.tile_times(FMT, 140)
    # sheet 1: 0..80 every 10 s (9 tiles); sheet 2 starts at 90: 90..130 (5 tiles < 140)
    assert times == [[0, 10, 20, 30, 40, 50, 60, 70, 80], [90, 100, 110, 120, 130]]


def test_tile_times_without_fps_divides_fragment():
    fmt = {**FMT, "fps": None, "fragments": [{"duration": 90.0}]}
    assert storyboard.tile_times(fmt, 90) == [[0, 10, 20, 30, 40, 50, 60, 70, 80]]


def _sheet(path: Path, rows: int):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    f"testsrc=size=96x{18 * rows}:rate=1", "-frames:v", "1", str(path)], check=True)


def _mhtml(path: Path, sheets: list[Path]):
    msg = MIMEMultipart("related")
    msg.attach(MIMEText("<html></html>", "html"))
    for s in sheets:
        part = MIMEImage(s.read_bytes(), "jpeg")
        part["Content-Location"] = f"https://i.ytimg.com/sb/x/{s.name}"
        msg.attach(part)
    path.write_bytes(msg.as_bytes())


@needs_ffmpeg
def test_parse_mhtml_returns_images_in_order(tmp_path):
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    _sheet(a, 3)
    _sheet(b, 2)
    _mhtml(tmp_path / "sb.mhtml", [a, b])
    parts = storyboard.parse_mhtml(tmp_path / "sb.mhtml")
    assert [p for p in parts] == [a.read_bytes(), b.read_bytes()]


def _fake_ytdlp(monkeypatch, tmp_path, info=INFO, rows=(3, 2)):
    def fake(args):
        out = Path(args[args.index("-o") + 1]).parent
        out.mkdir(parents=True, exist_ok=True)
        if info is not None:
            (out / "sb.info.json").write_text(json.dumps(info))
            sheets = []
            for i, r in enumerate(rows):
                p = tmp_path / f"src{i}.jpg"
                _sheet(p, r)
                sheets.append(p)
            _mhtml(out / "sb.mhtml", sheets)

        class R:
            returncode = 0 if info is not None else 1
            stderr = "" if info is not None else "ERROR: Requested format is not available"
        return R()
    monkeypatch.setattr(storyboard, "run_ytdlp", fake)


@needs_ffmpeg
def test_fetch_splits_tiles_with_times(monkeypatch, tmp_path):
    _fake_ytdlp(monkeypatch, tmp_path)
    frames, meta, info = storyboard.fetch(URL, tmp_path / "sb")
    assert [f["timestamp_seconds"] for f in frames] == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]
    assert all(f["reason"] == "storyboard" and Path(f["path"]).exists() for f in frames)
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=width,height", "-of", "csv=p=0",
                          frames[-1]["path"]], capture_output=True, text=True).stdout.strip()
    assert out == "32,18"
    assert meta["engine"] == "storyboard" and meta["selected_count"] == 14 and meta["tile"] == "32x18"
    assert info["duration"] == 140 and info["title"] == "t"


@needs_ffmpeg
def test_fetch_honours_range_and_cap(monkeypatch, tmp_path):
    _fake_ytdlp(monkeypatch, tmp_path)
    frames, meta, _ = storyboard.fetch(URL, tmp_path / "sb", start=25, end=115, max_frames=3)
    ts = [f["timestamp_seconds"] for f in frames]
    assert ts == [30, 70, 110]
    assert meta["candidate_count"] == 9 and meta["selected_count"] == 3


def test_fetch_raises_when_no_storyboard(monkeypatch, tmp_path):
    _fake_ytdlp(monkeypatch, tmp_path, info=None)
    with pytest.raises(RuntimeError):
        storyboard.fetch(URL, tmp_path / "sb")
