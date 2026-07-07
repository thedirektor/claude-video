"""yt-dlp argv construction for download.py.

Regression guard: ``--sub-langs all`` makes yt-dlp fetch YouTube's hundreds of
auto-translated caption tracks, which can take minutes and stalls before the
video download even starts. Phase 2 makes the default Spanish-first (the
owner's primary content language), then English, then any original-language
track — but the request must always stay bounded and never request "all".
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import download  # noqa: E402

URL = "https://www.youtube.com/watch?v=rlOpbu3Enkw"


def _capture_argv(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Stub subprocess.run inside download.py and record every argv."""
    calls: list[list[str]] = []

    class _Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        return _Result()

    monkeypatch.setattr(download.subprocess, "run", fake_run)
    return calls


def _sub_langs(argv: list[str]) -> str:
    idx = argv.index("--sub-langs")
    return argv[idx + 1]


DEFAULT_LANGS = "es,es-419,es-ES,en,en-US,en-GB,.*-orig"


def _assert_bounded(langs: str) -> None:
    tokens = [t.strip() for t in langs.split(",")]
    assert "all" not in tokens, f"sub-langs must never request all languages, got {langs!r}"
    assert tokens, f"sub-langs must be non-empty, got {langs!r}"


def test_fetch_captions_default_langs_are_spanish_first_and_bounded(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch)
    download.fetch_captions(URL, tmp_path / "download")
    langs = _sub_langs(calls[0])
    _assert_bounded(langs)
    assert langs == DEFAULT_LANGS


def test_download_url_default_langs_are_spanish_first_and_bounded(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch)
    # _pick_video returns None with no real file, which raises SystemExit after
    # the yt-dlp argv is already built — that's all we need to inspect.
    with pytest.raises(SystemExit):
        download.download_url(URL, tmp_path / "download")
    langs = _sub_langs(calls[0])
    _assert_bounded(langs)
    assert langs == DEFAULT_LANGS


def test_sub_langs_override_is_honored(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch)
    download.fetch_captions(URL, tmp_path / "download", sub_langs="de,fr")
    langs = _sub_langs(calls[0])
    _assert_bounded(langs)
    assert langs == "de,fr"


def test_pick_subtitle_prefers_language_order(tmp_path):
    d = tmp_path / "download"
    d.mkdir()
    for name in ("video.en.vtt", "video.es.vtt", "video.fr.vtt"):
        (d / name).write_text("WEBVTT\n", encoding="utf-8")
    picked = download._pick_subtitle(d)  # default es-first
    assert picked is not None and picked.name == "video.es.vtt"


def test_pick_subtitle_falls_back_to_orig(tmp_path):
    d = tmp_path / "download"
    d.mkdir()
    (d / "video.en-orig.vtt").write_text("WEBVTT\n", encoding="utf-8")
    picked = download._pick_subtitle(d)
    assert picked is not None and picked.name == "video.en-orig.vtt"


def _has_flag_value(argv: list[str], flag: str, value: str) -> bool:
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv) and argv[i + 1] == value:
            return True
    return False


def test_fetch_captions_uses_player_client_chain(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch)
    download.fetch_captions(URL, tmp_path / "download")
    argv = calls[0]
    assert _has_flag_value(argv, "--extractor-args", "youtube:player_client=default,android,mweb")
    assert "--retries" in argv and "--extractor-retries" in argv


def test_download_url_uses_player_client_chain(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch)
    with pytest.raises(SystemExit):
        download.download_url(URL, tmp_path / "download")
    argv = calls[0]
    assert _has_flag_value(argv, "--extractor-args", "youtube:player_client=default,android,mweb")
    assert "--sleep-interval" in argv and "--max-sleep-interval" in argv


def test_no_cookie_flags_by_default(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch)
    download.fetch_captions(URL, tmp_path / "download")
    argv = calls[0]
    assert "--cookies" not in argv
    assert "--cookies-from-browser" not in argv


def test_cookies_from_browser_passed_through(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch)
    download.fetch_captions(URL, tmp_path / "download", cookies_from_browser="firefox")
    assert _has_flag_value(calls[0], "--cookies-from-browser", "firefox")


def test_cookies_file_passed_through(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch)
    with pytest.raises(SystemExit):
        download.download_url(URL, tmp_path / "download", cookies_file="/tmp/c.txt")
    assert _has_flag_value(calls[0], "--cookies", "/tmp/c.txt")


def test_every_default_sub_lang_token_is_a_valid_regex():
    """yt-dlp compiles each --sub-langs token as a regex; a malformed one (e.g.
    a bare `*-orig`) makes it abort with "Wrong regex for subtitlelangs", which
    silently kills caption fetching. Guard the whole default set at import time
    so no invalid token can ship again — the mocked argv tests never run yt-dlp,
    so only this catches it."""
    import re

    for token in download.DEFAULT_SUB_LANGS.split(","):
        token = token.strip()
        assert token, "empty sub-lang token"
        try:
            re.compile(token)
        except re.error as exc:  # pragma: no cover - assertion carries the detail
            raise AssertionError(f"invalid sub-lang regex {token!r}: {exc}")
