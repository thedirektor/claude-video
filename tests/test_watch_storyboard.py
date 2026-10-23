"""watch.main(): a blocked YouTube media download falls back to storyboard frames."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import watch  # noqa: E402
from memory import index  # noqa: E402

YT = "https://www.youtube.com/watch?v=XUqrvbsTfck"
SEGS = [{"start": 0.0, "end": 5.0, "text": "hola mundo"}, {"start": 5.0, "end": 9.0, "text": "adios"}]


def _common(monkeypatch):
    monkeypatch.setattr(watch, "fetch_captions",
                        lambda *a, **k: {"video_path": None, "subtitle_path": None,
                                         "info": {"title": "T", "duration": 20}, "downloaded": False})
    monkeypatch.setattr(watch.transcriptapi, "load_api_key", lambda: "sk_test")
    monkeypatch.setattr(watch.transcriptapi, "fetch_transcript", lambda *a, **k: (SEGS, "es"))

    def blocked(*a, **k):
        raise SystemExit("yt-dlp did not produce a video file (exit 1)")
    monkeypatch.setattr(watch, "download", blocked)
    monkeypatch.setattr(watch, "reextract_frame",
                        lambda *a, **k: pytest.fail("no media to re-extract from"))
    monkeypatch.setattr(watch, "run_ocr", lambda paths: {p: "BIG SLIDE TITLE here" for p in paths})


def _tiles(tmp_path) -> list[dict]:
    out = []
    for i, t in enumerate((0.0, 10.0)):
        p = tmp_path / f"tile_{i}.jpg"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=0x{i}0{i}0{i}0:size=32x18",
                        "-frames:v", "1", str(p)], check=True)
        out.append({"timestamp_seconds": t, "path": str(p), "reason": "storyboard"})
    return out


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_blocked_youtube_uses_storyboard(monkeypatch, tmp_path, capsys):
    _common(monkeypatch)
    tiles = _tiles(tmp_path)
    seen = {}

    def fake_fetch(url, out_dir, *, start=None, end=None, max_frames=None):
        seen.update(url=url, max_frames=max_frames)
        return tiles, {"engine": "storyboard", "candidate_count": 2, "selected_count": 2,
                       "fallback": False, "tile": "32x18", "interval_s": 10.0}, {"title": "Real", "duration": 20, "url": url}
    monkeypatch.setattr(watch.storyboard, "fetch", fake_fetch)
    monkeypatch.setattr(sys, "argv", ["watch.py", YT, "--no-dedup", "--out-dir", str(tmp_path / "w")])
    assert watch.main() == 0
    out = capsys.readouterr().out
    assert "storyboard" in out and "32x18" in out
    assert tiles[1]["path"] in out
    assert seen["url"] == YT and seen["max_frames"]
    conn = index.connect()
    kinds = {r[0] for r in conn.execute("SELECT kind FROM frames")}
    assert kinds == {"storyboard"}


def test_blocked_youtube_without_storyboard_completes_frameless(monkeypatch, tmp_path, capsys):
    _common(monkeypatch)

    def no_board(*a, **k):
        raise RuntimeError("no storyboard available")
    monkeypatch.setattr(watch.storyboard, "fetch", no_board)
    monkeypatch.setattr(sys, "argv", ["watch.py", YT, "--out-dir", str(tmp_path / "w")])
    assert watch.main() == 0
    out = capsys.readouterr().out
    assert "hola mundo" in out
    assert "media download blocked" in out.lower()


def test_blocked_youtube_without_anything_still_exits(monkeypatch, tmp_path):
    _common(monkeypatch)
    monkeypatch.setattr(watch.transcriptapi, "fetch_transcript", lambda *a, **k: None)
    monkeypatch.setattr(watch.gemini, "transcribe_youtube", lambda *a, **k: None)
    monkeypatch.setattr(watch.storyboard, "fetch", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    monkeypatch.setattr(sys, "argv", ["watch.py", YT, "--no-whisper", "--out-dir", str(tmp_path / "w")])
    with pytest.raises(SystemExit):
        watch.main()


def test_non_youtube_download_failure_is_unchanged(monkeypatch, tmp_path):
    _common(monkeypatch)
    monkeypatch.setattr(watch.storyboard, "fetch", lambda *a, **k: pytest.fail("storyboard is YouTube-only"))
    monkeypatch.setattr(sys, "argv", ["watch.py", "https://vimeo.com/123", "--no-whisper",
                                      "--out-dir", str(tmp_path / "w")])
    with pytest.raises(SystemExit):
        watch.main()


def test_storyboard_frame_kind():
    assert index.frame_kind({"reason": "storyboard", "path": "/x/tile_000_001.jpg"}) == "storyboard"
