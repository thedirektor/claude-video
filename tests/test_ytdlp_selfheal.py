"""yt-dlp self-heal: breakage classification, rate-limited upgrade, one retry."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import download  # noqa: E402

URL = "https://www.youtube.com/watch?v=rlOpbu3Enkw"


@pytest.mark.parametrize("stderr", [
    "ERROR: [youtube] abc: Unable to extract initial player response; please report this issue on https://github.com/yt-dlp/yt-dlp/issues",
    "WARNING: [youtube] nsig extraction failed: You may experience throttling",
    "ERROR: Signature extraction failed: Some formats may be missing",
    "ERROR: [youtube] abc: Failed to extract any player response",
])
def test_is_breakage_true(stderr):
    assert download.is_breakage(stderr)


@pytest.mark.parametrize("stderr", [
    "",
    "ERROR: unable to download video data: HTTP Error 403: Forbidden",
    "ERROR: [youtube] abc: Sign in to confirm you’re not a bot",
    "ERROR: [youtube] abc: Video unavailable",
    "ERROR: [youtube] abc: Requested format is not available",
])
def test_is_breakage_false(stderr):
    assert not download.is_breakage(stderr)


def test_ytdlp_cmd_defaults_to_binary(monkeypatch):
    monkeypatch.delenv("WATCH_YTDLP_PYLIB", raising=False)
    assert download.ytdlp_cmd() == ["yt-dlp"]


def test_ytdlp_cmd_prefers_pylib_copy(monkeypatch, tmp_path):
    (tmp_path / "yt_dlp").mkdir()
    (tmp_path / "yt_dlp" / "__init__.py").write_text("")
    monkeypatch.setenv("WATCH_YTDLP_PYLIB", str(tmp_path))
    assert download.ytdlp_cmd() == [sys.executable, "-m", "yt_dlp"]
    env = download.ytdlp_env()
    assert env["PYTHONPATH"].split(":")[0] == str(tmp_path)


def test_ytdlp_cmd_ignores_empty_pylib(monkeypatch, tmp_path):
    monkeypatch.setenv("WATCH_YTDLP_PYLIB", str(tmp_path))
    assert download.ytdlp_cmd() == ["yt-dlp"]


class _Pip:
    def __init__(self, rc=0):
        self.calls = []
        self.rc = rc

    def __call__(self, cmd, *a, **k):
        self.calls.append(list(cmd))

        class R:
            returncode = self.rc
            stdout = ""
            stderr = ""
        return R()


def test_self_heal_upgrades_into_pylib_and_rate_limits(monkeypatch, tmp_path):
    pylib = tmp_path / "pylib"
    monkeypatch.setenv("WATCH_YTDLP_PYLIB", str(pylib))
    pip = _Pip()
    monkeypatch.setattr(download.subprocess, "run", pip)
    now = [1_000_000.0]
    assert download.self_heal(clock=lambda: now[0]) is True
    assert pip.calls[0][:4] == [sys.executable, "-m", "pip", "install"]
    assert "--target" in pip.calls[0] and str(pylib) in pip.calls[0] and "yt-dlp" in pip.calls[0]
    now[0] += 3600  # 1 h later: still inside the 6 h window
    assert download.self_heal(clock=lambda: now[0]) is False
    assert len(pip.calls) == 1
    now[0] += 6 * 3600
    assert download.self_heal(clock=lambda: now[0]) is True
    assert len(pip.calls) == 2


def test_self_heal_user_install_without_pylib(monkeypatch, tmp_path):
    monkeypatch.delenv("WATCH_YTDLP_PYLIB", raising=False)
    monkeypatch.setenv("WATCH_STATE_DIR", str(tmp_path))
    pip = _Pip()
    monkeypatch.setattr(download.subprocess, "run", pip)
    assert download.self_heal(clock=lambda: 5.0) is True
    assert "--user" in pip.calls[0]


def test_self_heal_never_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("WATCH_YTDLP_PYLIB", str(tmp_path / "p"))

    def boom(*a, **k):
        raise OSError("no pip")
    monkeypatch.setattr(download.subprocess, "run", boom)
    assert download.self_heal(clock=lambda: 5.0) is False


def test_self_heal_disabled_by_env(monkeypatch, tmp_path):
    monkeypatch.setenv("WATCH_YTDLP_PYLIB", str(tmp_path / "p"))
    monkeypatch.setenv("WATCH_YTDLP_SELF_HEAL", "0")
    pip = _Pip()
    monkeypatch.setattr(download.subprocess, "run", pip)
    assert download.self_heal(clock=lambda: 5.0) is False
    assert pip.calls == []


def test_download_url_retries_once_after_breakage(monkeypatch, tmp_path):
    out = tmp_path / "dl"
    calls = []

    def fake_run(cmd, *a, **k):
        calls.append(list(cmd))

        class R:
            returncode = 1
            stdout = ""
            stderr = "ERROR: Unable to extract initial player response"
        if len(calls) == 2:  # the retry succeeds
            out.mkdir(parents=True, exist_ok=True)
            (out / "video.mp4").write_bytes(b"x")
            R.returncode, R.stderr = 0, ""
        return R()

    healed = []
    monkeypatch.setattr(download.subprocess, "run", fake_run)
    monkeypatch.setattr(download.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(download, "self_heal", lambda **k: healed.append(1) or True)
    r = download.download_url(URL, out)
    assert r["video_path"].endswith("video.mp4")
    assert len(calls) == 2 and healed == [1]


def test_download_url_no_retry_on_bot_wall(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, *a, **k):
        calls.append(list(cmd))

        class R:
            returncode = 1
            stdout = ""
            stderr = "ERROR: unable to download video data: HTTP Error 403: Forbidden"
        return R()

    monkeypatch.setattr(download.subprocess, "run", fake_run)
    monkeypatch.setattr(download.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(download, "self_heal", lambda **k: pytest.fail("must not heal a bot wall"))
    with pytest.raises(SystemExit):
        download.download_url(URL, tmp_path / "dl")
    assert len(calls) == 1
