"""Playlist / channel enumeration + watch.py batch (0.6.0 phase C)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import playlist  # noqa: E402

CHANNEL = "https://www.youtube.com/@TEDx/videos"
FLAT = {"_type": "playlist", "id": "UCx", "title": "TEDx Talks - Videos", "entries": [
    {"id": "aaaaaaaaaaa", "url": "https://www.youtube.com/watch?v=aaaaaaaaaaa", "title": "A", "duration": 60},
    {"id": "bbbbbbbbbbb", "url": "https://www.youtube.com/watch?v=bbbbbbbbbbb", "title": "B", "duration": 90},
    {"id": "ccccccccccc", "url": None, "title": "no url", "ie_key": "Youtube"}]}


class _R:
    def __init__(self, out, rc=0, err=""):
        self.stdout, self.returncode, self.stderr = out, rc, err


def test_entries_from_flat_playlist(monkeypatch):
    calls = []
    monkeypatch.setattr(playlist, "run_ytdlp", lambda args, capture=False: calls.append(args) or _R(json.dumps(FLAT)))
    title, items = playlist.entries(CHANNEL, limit=5)
    assert title == "TEDx Talks - Videos"
    assert [i["id"] for i in items] == ["aaaaaaaaaaa", "bbbbbbbbbbb", "ccccccccccc"]
    assert items[2]["url"] == "https://www.youtube.com/watch?v=ccccccccccc"
    assert "--flat-playlist" in calls[0] and calls[0][calls[0].index("--playlist-end") + 1] == "5"


def test_entries_retry_via_proxy(monkeypatch):
    monkeypatch.setenv("WATCH_YTDLP_FALLBACK_PROXY", "http://100.64.0.9:3128")
    calls = []

    def fake(args, capture=False):
        calls.append(args)
        return _R(json.dumps(FLAT)) if "--proxy" in args else _R("", 1, "ERROR: Sign in to confirm")
    monkeypatch.setattr(playlist, "run_ytdlp", fake)
    assert len(playlist.entries(CHANNEL)[1]) == 3 and len(calls) == 2


def test_entries_from_file_list(tmp_path):
    f = tmp_path / "list.txt"
    f.write_text("# my list\nhttps://youtu.be/aaaaaaaaaaa\n\n/videos/clip.mp4\n")
    title, items = playlist.entries(str(f))
    assert title == "list.txt" and [i["url"] for i in items] == ["https://youtu.be/aaaaaaaaaaa", "/videos/clip.mp4"]


def test_entries_failure_raises(monkeypatch):
    monkeypatch.delenv("WATCH_YTDLP_FALLBACK_PROXY", raising=False)
    monkeypatch.setattr(playlist, "run_ytdlp", lambda args, capture=False: _R("", 1, "ERROR: boom"))
    with pytest.raises(RuntimeError):
        playlist.entries(CHANNEL)


def test_batch_skips_indexed_and_runs_the_rest(monkeypatch, tmp_path, capsys):
    import watch
    monkeypatch.setattr(playlist, "entries", lambda src, limit=None: ("T", [
        {"id": "aaaaaaaaaaa", "url": "https://www.youtube.com/watch?v=aaaaaaaaaaa", "title": "A"},
        {"id": "bbbbbbbbbbb", "url": "https://www.youtube.com/watch?v=bbbbbbbbbbb", "title": "B"}]))
    monkeypatch.setattr(playlist, "is_indexed", lambda url: url.endswith("aaaaaaaaaaa"))
    ran = []
    monkeypatch.setattr(playlist, "_run_one", lambda url, flags: ran.append((url, flags)) or 0)
    monkeypatch.setattr(sys, "argv", ["watch.py", "batch", CHANNEL, "--limit", "5", "--", "--detail", "efficient"])
    assert watch.main() == 0
    assert ran == [("https://www.youtube.com/watch?v=bbbbbbbbbbb", ["--detail", "efficient"])]
    out = capsys.readouterr().out
    assert "skipped (already indexed): 1" in out and "ok: 1" in out
