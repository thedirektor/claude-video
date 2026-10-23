"""Residential fallback proxy: one capped retry when the normal download is blocked."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import download  # noqa: E402

URL = "https://www.youtube.com/watch?v=XUqrvbsTfck"
PROXY = "http://100.64.0.9:8888"


def _runner(out: Path, succeed_on_proxy: bool, stderr="ERROR: unable to download video data: HTTP Error 403: Forbidden"):
    calls = []

    def fake_run(cmd, *a, **k):
        calls.append(list(cmd))

        class R:
            returncode, stdout = 1, ""
        R.stderr = stderr
        if "--proxy" in cmd and succeed_on_proxy:
            out.mkdir(parents=True, exist_ok=True)
            (out / "video.mp4").write_bytes(b"x")
            R.returncode, R.stderr = 0, ""
        return R()
    return fake_run, calls


@pytest.fixture(autouse=True)
def _ytdlp(monkeypatch):
    monkeypatch.setattr(download.shutil, "which", lambda n: "/usr/bin/" + n)
    monkeypatch.setattr(download, "self_heal", lambda **k: False)


def test_blocked_download_retries_via_proxy_capped(monkeypatch, tmp_path):
    monkeypatch.setenv("WATCH_YTDLP_FALLBACK_PROXY", PROXY)
    out = tmp_path / "dl"
    fake, calls = _runner(out, succeed_on_proxy=True)
    monkeypatch.setattr(download.subprocess, "run", fake)
    r = download.download_url(URL, out)
    assert r["video_path"].endswith("video.mp4") and r["via_fallback_proxy"] is True
    assert len(calls) == 2
    retry = calls[1]
    assert retry[retry.index("--proxy") + 1] == PROXY
    assert "480" in retry[retry.index("-f") + 1]
    assert retry[retry.index("--max-filesize") + 1] == "400M"
    assert "--socket-timeout" in retry
    assert "--proxy" not in calls[0]


def test_max_mb_setting(monkeypatch, tmp_path):
    monkeypatch.setenv("WATCH_YTDLP_FALLBACK_PROXY", PROXY)
    monkeypatch.setenv("WATCH_FALLBACK_MAX_MB", "120")
    fake, calls = _runner(tmp_path / "dl", succeed_on_proxy=True)
    monkeypatch.setattr(download.subprocess, "run", fake)
    download.download_url(URL, tmp_path / "dl")
    assert calls[1][calls[1].index("--max-filesize") + 1] == "120M"


def test_no_proxy_configured_no_retry(monkeypatch, tmp_path):
    monkeypatch.delenv("WATCH_YTDLP_FALLBACK_PROXY", raising=False)
    fake, calls = _runner(tmp_path / "dl", succeed_on_proxy=True)
    monkeypatch.setattr(download.subprocess, "run", fake)
    with pytest.raises(SystemExit):
        download.download_url(URL, tmp_path / "dl")
    assert len(calls) == 1


def test_proxy_also_fails_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("WATCH_YTDLP_FALLBACK_PROXY", PROXY)
    fake, calls = _runner(tmp_path / "dl", succeed_on_proxy=False)
    monkeypatch.setattr(download.subprocess, "run", fake)
    with pytest.raises(SystemExit):
        download.download_url(URL, tmp_path / "dl")
    assert len(calls) == 2


def test_audio_only_fallback_keeps_audio_format(monkeypatch, tmp_path):
    monkeypatch.setenv("WATCH_YTDLP_FALLBACK_PROXY", PROXY)
    fake, calls = _runner(tmp_path / "dl", succeed_on_proxy=True)
    monkeypatch.setattr(download.subprocess, "run", fake)
    download.download_url(URL, tmp_path / "dl", audio_only=True)
    assert calls[1][calls[1].index("-f") + 1].startswith("ba")
