"""Shared pytest fixtures: ffmpeg-synthesized clips and scripts/ on sys.path."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# Make the bundled scripts importable (mirrors watch.py's sys.path insert).
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

# Test isolation — NEVER touch the real library (Immich/Nextcloud/Paperless).
# watch.py auto-saves after every run unless --no-save, so the e2e subprocess
# tests would otherwise upload their ffmpeg fixtures (cuts.mp4/static.mp4/…) to
# the user's real services. WATCH_NO_SAVE force-disables saving for the whole
# session (honored by watch.py and inherited by every subprocess it spawns);
# stripping the creds from the environment is defense-in-depth for any
# in-process load_config() fallback.
os.environ["WATCH_NO_SAVE"] = "1"
for _k in (
    "IMMICH_API_KEY", "IMMICH_BASE_URL",
    "NEXTCLOUD_URL", "NEXTCLOUD_USER", "NEXTCLOUD_PASS",
    "PAPERLESS_URL", "PAPERLESS_TOKEN",
):
    os.environ.pop(_k, None)

# Video-memory isolation (Review Focus 1). ASSIGN, never setdefault: a host
# env pointing at the central index/store must not leak into tests. The session
# dir is a safety net; the autouse fixture below gives each test its own.
_SESSION_MEM = tempfile.mkdtemp(prefix="watch-test-mem-")
os.environ["WATCH_INDEX_DB"] = str(Path(_SESSION_MEM) / "index.db")
os.environ["WATCH_STORE_DIR"] = str(Path(_SESSION_MEM) / "store")
os.environ["WATCH_EMBEDDINGS"] = "0"
os.environ["WATCH_CONFIG_FILE"] = str(Path(_SESSION_MEM) / "no-such-config" / ".env")
os.environ.pop("WATCH_QDRANT_INBOX", None)
os.environ["WATCH_LLM_CHAIN"] = "none"  # grounded answers never reach a real LLM; tests inject fakes
os.environ.pop("WATCH_MODEL_DIR", None)

import config as _config  # noqa: E402

# Otherwise popped vars fall through to the host's ~/.config/watch/.env.
_config.CONFIG_FILE = Path(os.environ["WATCH_CONFIG_FILE"])


@pytest.fixture(autouse=True)
def _isolated_memory(tmp_path, monkeypatch):
    monkeypatch.setenv("WATCH_INDEX_DB", str(tmp_path / "index.db"))
    monkeypatch.setenv("WATCH_STORE_DIR", str(tmp_path / "store"))
    yield


@pytest.fixture
def conn(tmp_path):
    """A fresh, migrated video-memory index for one test."""
    from memory import index

    c = index.connect(tmp_path / "index.db")
    yield c
    c.close()


# 14 visually distinct fills → 14 abrupt cuts → x264 emits a keyframe per cut.
COLORS = [
    "red", "green", "blue", "white", "black", "yellow", "cyan",
    "magenta", "gray", "orange", "purple", "brown", "navy", "olive",
]


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}\n{result.stderr}")


def build_cut_clip(
    path: Path,
    n: int = 14,
    seg: float = 0.4,
    size: str = "320x240",
    fps: int = 10,
) -> None:
    """Concatenate ``n`` solid-color segments into one clip with ``n`` cuts.

    Each color change is a hard scene cut, so the scene selector finds ~n-1
    changes. x264's own scenecut detection is unreliable on flat fills, so we
    force a keyframe at every ``seg`` boundary — giving ~n real keyframes for
    the keyframe engine to find.
    """
    inputs: list[str] = []
    for i in range(n):
        color = COLORS[i % len(COLORS)]
        inputs += ["-f", "lavfi", "-t", str(seg), "-i", f"color=c={color}:s={size}:r={fps}"]
    streams = "".join(f"[{i}:v]" for i in range(n))
    filt = f"{streams}concat=n={n}:v=1:a=0[out]"
    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        *inputs,
        "-filter_complex", filt, "-map", "[out]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-force_key_frames", f"expr:gte(t,n_forced*{seg})",
        str(path),
    ])


def build_static_clip(
    path: Path,
    duration: float = 3.0,
    size: str = "320x240",
    fps: int = 10,
) -> None:
    """One solid color: 1 keyframe, no scene changes → triggers both fallbacks."""
    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-t", str(duration), "-i", f"color=c=blue:s={size}:r={fps}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "600",
        str(path),
    ])


@pytest.fixture(scope="session")
def cut_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("clips") / "cuts.mp4"
    build_cut_clip(path)
    return path


@pytest.fixture(scope="session")
def static_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("clips") / "static.mp4"
    build_static_clip(path)
    return path
