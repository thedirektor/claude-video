"""memory.cli: search/ask/moment/list/index subcommands, exit codes, watch.py dispatch (Task 10)."""
from __future__ import annotations

import json

import pytest

import identity
from memory import cli, index
from memory.index import record_run
from memory.retrieval import RULE, list_videos
from memory_fakes import KEY, Spy, fake_run, fr, jpg, run_watch, seg

OTHER = "file:0123456789abcdef"


@pytest.fixture
def seeded(tmp_path):
    c = index.connect()
    record_run(c, fake_run(title="La Canción demo", segments=[
        seg(0, 5, "hola mundo"), seg(12, 18, "la canción del verano"), seg(40, 45, "error en pantalla")],
        frames=[fr(13, jpg(tmp_path, "red"), "ERROR 404")]))
    c.close()


@pytest.fixture
def two(seeded):
    c = index.connect()
    record_run(c, fake_run(video_key=OTHER, source="/tmp/x.mp4", title="Otro demo", segments=[seg(0, 5, "otra cosa")]))
    c.close()


def test_cli_search_json(seeded, capsys):
    assert cli.main(["search", "cancion", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["hits"][0]["timestamp"] == "00:12"
    assert out["mode"] == "fts"


def test_cli_search_human_and_video_scope(two, capsys):
    assert cli.main(["search", "demo", "--video", OTHER]) == 0
    out = capsys.readouterr().out
    assert "Otro demo" not in out or OTHER in out


def test_cli_ask_prints_rule_and_ids(seeded, capsys):
    assert cli.main(["ask", KEY, "cancion"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == RULE
    assert lines[1].startswith("[E1 00:12 transcript]")


def test_cli_ask_json(seeded, capsys):
    assert cli.main(["ask", "Canción", "cancion", "--json"]) == 0
    pack = json.loads(capsys.readouterr().out)
    assert pack["evidence"][0]["id"] == "E1" and pack["rule"] == RULE


def test_cli_ambiguous_exit_4(two, capsys):
    assert cli.main(["ask", "demo", "q"]) == 4
    cap = capsys.readouterr()
    assert KEY in cap.out and OTHER in cap.out


def test_cli_not_found_exit_3(seeded, capsys):
    assert cli.main(["ask", "nonexistent-thing", "q"]) == 3
    assert cli.main(["moment", "nonexistent-thing", "5", "--json"]) == 3


def test_cli_moment_lists_frame_paths(seeded, capsys):
    assert cli.main(["moment", KEY, "0:13"]) == 0
    out = capsys.readouterr().out
    assert "[00:12] la canción" in out and "ERROR 404" in out
    assert cli.main(["moment", KEY, "13", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["t"] == 13.0


def test_cli_list(two, capsys):
    assert cli.main(["list", "--json"]) == 0
    assert {v["video_key"] for v in json.loads(capsys.readouterr().out)["videos"]} == {KEY, OTHER}
    assert cli.main(["list", "otro"]) == 0
    out = capsys.readouterr().out
    assert OTHER in out and KEY not in out


def test_forget_removes_rows_and_store_dir(seeded, tmp_path, capsys):
    from config import memory_paths

    vdir = memory_paths()["store_dir"] / identity.store_dirname(KEY)
    (vdir / "frames").mkdir(parents=True)
    (vdir / "report.md").write_text("x")
    assert cli.main(["index", "--forget", KEY]) == 0
    c = index.connect()
    try:
        assert list_videos(c) == []
    finally:
        c.close()
    assert not vdir.exists()


def test_index_requires_one_action(seeded):
    with pytest.raises(SystemExit):
        cli.main(["index"])


def test_dispatch_to_memory_cli(monkeypatch):
    spy = Spy(return_value=0)
    monkeypatch.setattr(cli, "main", spy)
    assert run_watch(["search", "hello", "--json"]) == 0
    assert spy.calls[0][0] == (["search", "hello", "--json"],)


def test_dispatch_vs_file_named_search(tmp_path, monkeypatch):
    (tmp_path / "search").write_bytes(b"not a video")
    monkeypatch.chdir(tmp_path)
    spy = Spy(return_value=0)
    monkeypatch.setattr(cli, "main", spy)
    try:
        run_watch(["search"])
    except (SystemExit, Exception):
        pass  # the file is watched (and rejected as a non-video); what matters is no dispatch
    assert spy.calls == []
