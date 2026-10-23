"""Ground-truth media for the breakdown tests, built with ffmpeg lavfi sources."""
from __future__ import annotations

import subprocess
from pathlib import Path

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
)

SIZE = "640x360"
DURATION = 7.0


def _font() -> str | None:
    for cand in _FONT_CANDIDATES:
        if Path(cand).is_file():
            return cand
    return None


def _ffmpeg(args: list[str]) -> None:
    res = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
                         capture_output=True, text=True)
    if res.returncode:
        raise RuntimeError(f"ffmpeg failed: {res.stderr[-800:]}")


def make_reference(path, *, fps: int = 25, with_audio: bool = True) -> dict:
    """7.0 s clip with known cuts, title, beat grid. Returns the ground truth.

    Timeline: shot A 0-2 (red), hard cut at 2.0 to shot B (green) 2-4, a 1.0 s
    cross-dissolve 4-5 (centre 4.5) into shot C (blue) 5-7, fade to black 6-7,
    title "SALE 50%" at top-left 1.0-3.0 s, click every 0.5 s (120 BPM).
    """
    path = Path(path)
    font = _font()
    text_filter = ""
    if font:
        text_filter = (
            f",drawtext=fontfile={font}:text='SALE 50%':x=20:y=20:fontsize=48:"
            "fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=8:enable='between(t,1,3)'"
        )
    graph = (
        f"color=c=red:s={SIZE}:r={fps}:d=2[a];"
        f"color=c=0x00b000:s={SIZE}:r={fps}:d=3[b];"
        f"color=c=blue:s={SIZE}:r={fps}:d=3[c];"
        f"[a][b]concat=n=2:v=1:a=0,fps={fps},settb=1/{fps}[ab];"
        f"[c]settb=1/{fps}[c2];"
        "[ab][c2]xfade=transition=fade:duration=1:offset=4,"
        f"fade=t=out:st=6:d=1{text_filter},format=yuv420p[v]"
    )
    args = ["-filter_complex", graph]
    if with_audio:
        # 1 kHz 20 ms burst at the start of every 0.5 s => 120 BPM
        args = ["-f", "lavfi", "-i",
                f"aevalsrc=0.8*sin(2*PI*1000*t)*lt(mod(t\\,0.5)\\,0.02):s=44100:d={DURATION}",
                *args, "-map", "[v]", "-map", "0:a", "-c:a", "aac", "-b:a", "128k"]
    else:
        args = [*args, "-map", "[v]", "-an"]
    _ffmpeg([*args, "-r", str(fps), "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
             "-t", str(DURATION), "-movflags", "+faststart", str(path)])
    return {
        "duration": DURATION,
        "fps": fps,
        "shots": [0, 2.0, 4.0, 5.0],
        "cuts": [{"t": 2.0, "type": "hard"}],
        "dissolve": {"start": 4.0, "end": 5.0, "center": 4.5},
        "fade_out": {"start": 6.0, "end": 7.0},
        "title": {"text": "SALE 50%", "start": 1.0, "end": 3.0, "pos": "top-left", "rendered": font is not None},
        "bpm": 120,
        "beats": [round(0.5 * i, 3) for i in range(int(DURATION / 0.5))],
        "has_audio": with_audio,
    }


def make_motion_clip(path, *, fps: int = 60) -> dict:
    """White square 0 -> 600 px over 0.5 s (from t=0.2) with Godot CUBIC/OUT easing."""
    path = Path(path)
    expr = "600*(1-pow(1-clip((t-0.2)/0.5,0,1),3))"
    graph = (
        f"color=c=black:s=1280x720:r={fps}:d=1.2[bg];"
        f"color=c=white:s=100x100:r={fps}:d=1.2[sq];"
        f"[bg][sq]overlay=x='{expr}':y=310:eval=frame,format=yuv420p[v]"
    )
    _ffmpeg(["-filter_complex", graph, "-map", "[v]", "-r", str(fps), "-c:v", "libx264",
             "-preset", "veryfast", "-crf", "16", "-t", "1.2", str(path)])
    return {"start": 0.2, "end": 0.7, "trans": "CUBIC", "ease": "OUT"}
