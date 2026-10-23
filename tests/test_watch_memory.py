"""Task 9: every watch backend writes the video memory (index + store).

All in-process, per-test tmp index/store from conftest. Network-free: yt-dlp,
Gemini, OpenRouter and the library are faked; paid transcript backends are
spied on to prove the gemini backend never calls them.
"""
from __future__ import annotations


import pytest

import gemini
import library
import openrouter
import transcriptapi
import watch
from identity import store_dirname
from memory import index
from memory.index import connect
from memory.retrieval import ask, list_videos
from memory_fakes import Spy, frame_count, note_texts, run_watch, segment_count, store_dir
from watch import should_save

YT = "https://www.youtube.com/watch?v=AAAAAAAAAAA"


@pytest.fixture
def save_spy(monkeypatch):
    spy = Spy(return_value=[])
    monkeypatch.setattr(library, "save_artifacts", spy)
    return spy


@pytest.fixture
def ta_spy(monkeypatch):
    spy = Spy(return_value=None)
    monkeypatch.setattr(transcriptapi, "fetch_transcript", spy)
    monkeypatch.setattr(transcriptapi, "load_api_key", lambda: "ta-test-key")
    return spy


@pytest.fixture
def gemini_tx_spy(monkeypatch):
    spy = Spy(return_value=None)
    monkeypatch.setattr(gemini, "transcribe_youtube", spy)
    return spy


def _fake_captions(monkeypatch, tmp_path, vtt: str | None):
    sub = None
    if vtt is not None:
        sub = tmp_path / "captions.es.vtt"
        sub.write_text(vtt, encoding="utf-8")
    calls = []

    def fake(url, out_dir, **kw):
        calls.append(url)
        return {"video_path": None, "subtitle_path": str(sub) if sub else None, "info": {}, "downloaded": False}

    monkeypatch.setattr(watch, "fetch_captions", fake)
    return calls


HOLA_VTT = "WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nhola\n"


@pytest.mark.parametrize("backend,origin,detail,save,no_save,env,exp", [
    ("claude", "skill", "balanced", False, False, False, True),
    ("claude", "ingest", "transcript", False, False, False, False),
    ("claude", "ingest", "balanced", False, False, False, True),
    ("gemini", "ingest", "balanced", False, False, False, False),
    ("gemini", "ingest", "balanced", True, False, False, True),
    ("openrouter", "service", "balanced", True, True, False, False),
    ("claude", "skill", "balanced", False, False, True, False)])
def test_should_save_table(backend, origin, detail, save, no_save, env, exp):   # Review Focus 4
    assert should_save(backend=backend, origin=origin, detail=detail, save=save,
                       no_save=no_save, env_no_save=env) is exp


def test_claude_run_indexes_with_no_save(cut_clip):      # WATCH_NO_SAVE=1 still indexes
    assert run_watch([str(cut_clip), "--detail", "efficient", "--no-ocr"]) == 0
    v = list_videos(connect())
    assert len(v) == 1 and v[0]["video_key"].startswith("file:") and frame_count(v[0]) > 0
    assert (store_dir() / store_dirname(v[0]["video_key"]) / "report.md").exists()
    assert v[0]["title"] == "cuts.mp4"


def test_claude_run_writes_memory_json(cut_clip, tmp_path):
    out = tmp_path / "work"
    assert run_watch([str(cut_clip), "--detail", "efficient", "--no-ocr", "--out-dir", str(out)]) == 0
    import json

    mem = json.loads((out / "memory.json").read_text(encoding="utf-8"))
    assert mem["video_key"].startswith("file:") and mem["mode"] == "fts"
    assert mem["run_id"] and mem["evidence_version"] >= 1


def test_no_index_skips(cut_clip, tmp_path):
    out = tmp_path / "work"
    assert run_watch([str(cut_clip), "--detail", "efficient", "--no-ocr", "--no-index",
                      "--out-dir", str(out)]) == 0
    assert list_videos(connect()) == []
    assert not (out / "memory.json").exists()


def test_origin_recorded(cut_clip):
    assert run_watch([str(cut_clip), "--detail", "efficient", "--no-ocr", "--origin", "service"]) == 0
    c = connect()
    assert [r[0] for r in c.execute("SELECT origin FROM runs")] == ["service"]


def test_ingest_transcript_run_indexes_but_never_archives(monkeypatch, tmp_path, save_spy):
    monkeypatch.delenv("WATCH_NO_SAVE")
    _fake_captions(monkeypatch, tmp_path, HOLA_VTT)
    monkeypatch.setattr(transcriptapi, "load_api_key", lambda: None)
    assert run_watch([YT, "--detail", "transcript", "--origin", "ingest", "--no-whisper"]) == 0
    assert save_spy.calls == []
    assert segment_count() == 1


def test_gemini_stdout_unchanged(monkeypatch, capsys, tmp_path, save_spy, ta_spy, gemini_tx_spy):  # RF 4
    monkeypatch.setattr(gemini, "generate_with_video", lambda **kw: "ANSWER")
    _fake_captions(monkeypatch, tmp_path, HOLA_VTT)
    out_dir = tmp_path / "work"
    assert run_watch([YT, "--backend", "gemini", "--out-dir", str(out_dir), "qué pasa"]) == 0
    out = capsys.readouterr().out
    assert "## Gemini response" in out and "## Transcript" not in out
    report = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "# watch: video report (Gemini backend)" in report
    assert "## Gemini response" in report and "## Transcript" in report and "hola" in report
    pack = ask(connect(), "yt:AAAAAAAAAAA", "hola")
    assert {e["kind"] for e in pack["evidence"]} >= {"transcript"} and note_texts() == ["Q: qué pasa\n\nANSWER"]
    assert save_spy.calls == []                           # gemini without --save never archives
    assert ta_spy.calls == [] and gemini_tx_spy.calls == []   # no paid transcript calls on this backend
    assert (out_dir / "memory.json").exists()


def test_gemini_without_captions_indexes_note_only(monkeypatch, tmp_path):
    monkeypatch.setattr(gemini, "generate_with_video", lambda **kw: "ANSWER")
    _fake_captions(monkeypatch, tmp_path, None)
    assert run_watch([YT, "--backend", "gemini", "q"]) == 0
    assert segment_count() == 0 and note_texts() == ["Q: q\n\nANSWER"]


def test_gemini_save_flag_archives_without_media(monkeypatch, tmp_path, save_spy):
    monkeypatch.delenv("WATCH_NO_SAVE")
    monkeypatch.setattr(gemini, "generate_with_video", lambda **kw: "ANSWER")
    _fake_captions(monkeypatch, tmp_path, HOLA_VTT)
    assert run_watch([YT, "--backend", "gemini", "--save", "q"]) == 0
    assert len(save_spy.calls) == 1
    args, _kw = save_spy.calls[0]
    assert args[0] is None and args[1] == []          # YouTube via URL: nothing downloaded → no Immich media


def test_gemini_no_index_skips_caption_fetch(monkeypatch, tmp_path):
    monkeypatch.setattr(gemini, "generate_with_video", lambda **kw: "ANSWER")
    calls = _fake_captions(monkeypatch, tmp_path, HOLA_VTT)
    assert run_watch([YT, "--backend", "gemini", "--no-index", "q"]) == 0
    assert calls == [] and list_videos(connect()) == []


def test_openrouter_stdout_unchanged_and_indexed(monkeypatch, capsys, cut_clip, tmp_path, save_spy):
    monkeypatch.setattr(openrouter, "analyze_with_frames", lambda **kw: "OR-ANSWER")
    out_dir = tmp_path / "work"
    assert run_watch([str(cut_clip), "--backend", "openrouter", "--no-whisper",
                      "--out-dir", str(out_dir), "what"]) == 0
    out = capsys.readouterr().out
    assert "## OpenRouter response" in out and "## Transcript" not in out
    assert "## OpenRouter response" in (out_dir / "report.md").read_text(encoding="utf-8")
    v = list_videos(connect())
    assert len(v) == 1 and frame_count(v[0]) > 0 and v[0]["title"] == "cuts.mp4"
    assert note_texts() == ["Q: what\n\nOR-ANSWER"]
    assert save_spy.calls == []


def test_qdrant_handoff_only_when_archived(monkeypatch, tmp_path, cut_clip):
    monkeypatch.delenv("WATCH_NO_SAVE"); monkeypatch.setenv("WATCH_QDRANT_INBOX", str(tmp_path / "inbox"))
    monkeypatch.setattr(library, "save_artifacts", Spy(return_value=[]))
    run_watch([str(cut_clip), "--detail", "efficient", "--no-ocr"])
    assert len(list((tmp_path / "inbox").glob("*.md"))) == 1
    for f in (tmp_path / "inbox").glob("*.md"):
        f.unlink()
    run_watch([str(cut_clip), "--detail", "efficient", "--no-ocr", "--no-save"])
    assert len(list((tmp_path / "inbox").glob("*.md"))) == 0          # no second file


def test_paperless_task_resolved_to_doc_id(monkeypatch, cut_clip):
    monkeypatch.delenv("WATCH_NO_SAVE")

    def fake_save(*a, refs=None, **kw):
        refs["paperless"] = {"task_ids": ["task-1"], "url": "x"}
        return []

    monkeypatch.setattr(library, "save_artifacts", fake_save)
    seen = []
    monkeypatch.setattr(library, "resolve_paperless_doc",
                        lambda tid, cfg, attempts=3, **kw: seen.append((tid, attempts)) or 42)
    run_watch([str(cut_clip), "--detail", "efficient", "--no-ocr"])
    row = connect().execute("SELECT paperless_task_id, paperless_doc_id FROM videos").fetchone()
    assert tuple(row) == ("task-1", 42) and seen == [("task-1", 3)]


def test_memory_failure_never_fails_run(monkeypatch, cut_clip, capsys):
    monkeypatch.setattr(index, "record_run", Spy(raises=RuntimeError("boom")))
    assert run_watch([str(cut_clip), "--detail", "efficient", "--no-ocr"]) == 0
    assert capsys.readouterr().err.count("[watch] memory:") == 1


def test_remember_returns_none_on_error(tmp_path, capsys):
    from memory.writer import remember

    run = index.RunRecord(video_key="file:x", source=str(tmp_path / "x.mp4"), origin="skill",
                          backend="claude", detail="balanced")
    bad = {"index_db": tmp_path / "d" / "index.db", "store_dir": tmp_path / "f", "model_dir": tmp_path}
    (tmp_path / "f").write_text("not a dir", encoding="utf-8")   # store dir is a file → write_run fails
    assert remember(run, archive=False, embedder=None, paths=bad) is None
    assert capsys.readouterr().err.count("[watch] memory:") == 1


def test_remember_embeds_with_given_embedder(tmp_path):
    from memory.writer import remember
    from memory_fakes import FakeEmbedder, embedding_models

    run = index.RunRecord(video_key="file:y", source=str(tmp_path / "y.mp4"), origin="skill",
                          backend="claude", detail="balanced",
                          notes=[{"kind": "backend_answer", "t_start": None, "t_end": None, "text": "hola mundo"}])
    paths = {"index_db": tmp_path / "i.db", "store_dir": tmp_path / "s", "model_dir": tmp_path}
    out = remember(run, archive=False, embedder=FakeEmbedder(), paths=paths)
    assert out["mode"] == "hybrid" and out["video_key"] == "file:y"
    assert embedding_models(connect(tmp_path / "i.db")) == {"fake-16"}
