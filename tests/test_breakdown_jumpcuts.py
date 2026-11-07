"""Jump cuts inside one framing (breakdown/shots.py: _jump_cut, weak candidates)."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skills" / "watch" / "scripts"))

from breakdown import shots  # noqa: E402

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
FPS = 25
SWAY = "10*sin(2*PI*0.3*t)+2*sin(2*PI*3*t)"  # a talking head never holds perfectly still
BOB = "60+3*sin(2*PI*2.1*t)"


def _ff(*args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


@pytest.fixture(scope="module")
def stills(tmp_path_factory):
    d = tmp_path_factory.mktemp("jump")
    _ff("-f", "lavfi", "-i", "mandelbrot=s=640x360", "-frames:v", "1", str(d / "bg.png"))
    _ff("-f", "lavfi", "-i", "testsrc2=s=180x240", "-frames:v", "1", str(d / "subj.png"))
    return d


def _clip(d: Path, name: str, x: str) -> Path:
    out = d / f"{name}.mp4"
    _ff("-loop", "1", "-i", str(d / "bg.png"), "-loop", "1", "-i", str(d / "subj.png"), "-filter_complex",
        f"[0][1]overlay=x='{x}':y='{BOB}',noise=alls=10:allf=t,format=yuv420p", "-t", "4", "-r", str(FPS), str(out))
    return out


def _bounds(path: Path) -> list[dict]:
    ser = shots.frame_series(str(path), float(FPS))
    return [s["boundary"] for s in shots.detect_shots(ser, 4.0)[1:]]


@needs_ffmpeg
def test_subject_jump_on_a_held_set_is_a_jump_cut(stills):
    b = _bounds(_clip(stills, "jump", f"if(lt(t,2),220,245)+{SWAY}"))
    assert len(b) == 1 and b[0]["kind"] == "cut" and b[0].get("jump") is True
    assert abs(b[0]["t"] - 2.0) <= 1.0 / FPS
    assert b[0]["evidence"]["confidence"] <= shots.JUMP_CONF


@needs_ffmpeg
def test_one_frame_glitch_is_not_a_cut(stills):
    assert _bounds(_clip(stills, "glitch", f"if(between(t,2,2.04),245,220)+{SWAY}")) == []


@needs_ffmpeg
def test_fast_continuous_move_is_not_a_jump_cut(stills):
    b = _bounds(_clip(stills, "move", f"220+25*min(1,max(0,(t-1.9)/0.24))+{SWAY}"))
    assert not any(x.get("jump") for x in b)


# ----------------------------------------------------------------------------- _jump_cut on hand-made series

def _series(steps: list[float], jump_at: int, returns_at: int | None = None) -> tuple[dict, list[float]]:
    """Thumbnails whose consecutive distance follows ``steps`` (one channel value shifted per frame)."""
    n = len(steps)
    level, th = 100.0, []
    for i in range(n):
        if i:
            level += steps[i] * 255.0 * (1 if i != returns_at else -1)
        th.append([level] * (16 * 9 * 3))
    if returns_at is not None:  # put the picture back where it was before the jump
        for k in range(returns_at, n):
            th[k] = list(th[jump_at - 1])
    series = {"t": [i / FPS for i in range(n)], "thumb": th}
    return series, shots._steps(series)


def test_jump_cut_unit_cases():
    calm = [0.0] + [0.004 * (1 + (i % 3) / 10) for i in range(1, 60)]
    steps = list(calm)
    steps[30] = 0.05
    ser, d = _series(steps, 30)
    med = 0.004
    assert shots._jump_cut(ser, d, 30, med)
    # frozen picture: nothing to stand out of
    assert not shots._jump_cut(ser, d, 30, 0.0005)
    # weak stand-out
    assert not shots._jump_cut(ser, d, 30, 0.01)
    # a second jump 0.2 s later: the side is not calm (would be a sliver shot)
    steps2 = list(steps)
    steps2[35] = 0.05
    ser2, d2 = _series(steps2, 30)
    assert not shots._jump_cut(ser2, d2, 30, med)
    # too close to the clip edge for a 0.3 s side
    ser3, d3 = _series(steps[:34], 30)
    assert not shots._jump_cut(ser3, d3, 30, med)


def test_jump_cut_rejects_a_picture_that_comes_back():
    steps = [0.0] + [0.004] * 59
    steps[30] = 0.05
    ser, d = _series(steps, 30, returns_at=33)
    assert not shots._jump_cut(ser, d, 30, 0.004)
