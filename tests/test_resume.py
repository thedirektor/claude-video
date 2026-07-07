"""Resume: a second run with the same --out-dir reuses the downloaded video."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import watch  # noqa: E402


def _silent_clip(path: Path) -> None:
    # A short silent clip: no audio track (whisper skipped), a few keyframes.
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-t", "1.5", "-i", "color=c=blue:s=160x120:r=10",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )


def test_second_run_skips_download(tmp_path, monkeypatch, capsys):
    clip = tmp_path / "clip.mp4"
    _silent_clip(clip)
    work = tmp_path / "work"

    calls = {"n": 0}
    real_download = watch.download

    def counting_download(*args, **kwargs):
        calls["n"] += 1
        return real_download(*args, **kwargs)

    monkeypatch.setattr(watch, "download", counting_download)

    argv = [str(clip), "--out-dir", str(work), "--detail", "efficient", "--no-ocr"]
    monkeypatch.setattr(sys, "argv", ["watch.py", *argv])
    assert watch.main() == 0
    monkeypatch.setattr(sys, "argv", ["watch.py", *argv])
    assert watch.main() == 0

    # Local resolve counts as a "download" call; the second run must reuse the
    # saved stage (video file still present) rather than resolving again.
    assert calls["n"] == 1


def test_fresh_forces_redownload(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    _silent_clip(clip)
    work = tmp_path / "work"

    calls = {"n": 0}
    real_download = watch.download

    def counting_download(*args, **kwargs):
        calls["n"] += 1
        return real_download(*args, **kwargs)

    monkeypatch.setattr(watch, "download", counting_download)

    base = [str(clip), "--out-dir", str(work), "--detail", "efficient", "--no-ocr"]
    monkeypatch.setattr(sys, "argv", ["watch.py", *base])
    assert watch.main() == 0
    monkeypatch.setattr(sys, "argv", ["watch.py", *base, "--fresh"])
    assert watch.main() == 0
    assert calls["n"] == 2
