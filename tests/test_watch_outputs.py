"""watch runs: chapters + notes.md always, exports per --exports, library routing (0.5.3 Task 3)."""
from __future__ import annotations

import json

import exports
import library
from memory.index import connect
from memory_fakes import Spy, run_watch


def _chapter_count():
    return connect().execute("SELECT count(*) FROM notes WHERE kind = 'chapter'").fetchone()[0]


def test_run_writes_chapters_notes_and_requested_exports(cut_clip, tmp_path, monkeypatch):
    monkeypatch.delenv("WATCH_EXPORTS", raising=False)
    out = tmp_path / "work"
    assert run_watch([str(cut_clip), "--detail", "efficient", "--no-ocr", "--exports", "viewer,json",
                      "--out-dir", str(out)]) == 0
    res = json.loads((out / "exports.json").read_text(encoding="utf-8"))
    assert set(res) == {"viewer", "json", "notes"}
    assert all(json.loads(json.dumps(v)).get("path") for v in res.values())
    assert (out / "notes.md").read_text(encoding="utf-8").startswith("# cuts.mp4")
    assert _chapter_count() >= 1


def test_default_without_clis_is_notes_only(cut_clip, tmp_path, monkeypatch):
    monkeypatch.delenv("WATCH_EXPORTS", raising=False)
    monkeypatch.setattr(exports, "find_cli", lambda app: None)
    out = tmp_path / "work"
    assert run_watch([str(cut_clip), "--detail", "efficient", "--no-ocr", "--out-dir", str(out)]) == 0
    assert set(json.loads((out / "exports.json").read_text(encoding="utf-8"))) == {"notes"}


def test_no_index_means_no_outputs(cut_clip, tmp_path):
    out = tmp_path / "work"
    assert run_watch([str(cut_clip), "--detail", "efficient", "--no-ocr", "--no-index", "--exports", "json",
                      "--out-dir", str(out)]) == 0
    assert not (out / "exports.json").exists() and not (out / "notes.md").exists()


def _saving(monkeypatch, tmp_path, pdf: bool):
    monkeypatch.delenv("WATCH_NO_SAVE")
    arts = Spy(return_value=[])
    exp = Spy(return_value=[])
    monkeypatch.setattr(library, "save_artifacts", arts)
    monkeypatch.setattr(library, "save_exports", exp)
    monkeypatch.setattr(exports, "find_cli", lambda app: f"/x/{app}-cli")
    merged = tmp_path / "watch-x.pdf"
    merged.write_bytes(b"%PDF")
    built = {"notes": {"path": str(tmp_path / "notes.md")}}
    if pdf:
        built["pdf"] = {"path": str(merged)}
    monkeypatch.setattr(exports, "build", lambda conn, key, formats, **k: dict(built))
    return arts, exp, merged


def test_pdf_export_replaces_report_in_paperless(cut_clip, tmp_path, monkeypatch):
    arts, exp, merged = _saving(monkeypatch, tmp_path, pdf=True)
    assert run_watch([str(cut_clip), "--detail", "efficient", "--no-ocr", "--save", "--exports", "pdf"]) == 0
    assert arts.calls[0][1]["paperless"] is False
    kw = exp.calls[0][1]
    assert kw["paperless_doc"] == str(merged)


def test_failed_pdf_falls_back_to_report_md(cut_clip, tmp_path, monkeypatch):
    arts, exp, _ = _saving(monkeypatch, tmp_path, pdf=False)
    out = tmp_path / "work"
    assert run_watch([str(cut_clip), "--detail", "efficient", "--no-ocr", "--save", "--exports", "pdf",
                      "--out-dir", str(out)]) == 0
    assert arts.calls[0][1]["paperless"] is False
    assert exp.calls[0][1]["paperless_doc"] == str(out / "report.md")


def test_no_pdf_planned_keeps_paperless_in_save_artifacts(cut_clip, tmp_path, monkeypatch):
    arts, exp, _ = _saving(monkeypatch, tmp_path, pdf=False)
    assert run_watch([str(cut_clip), "--detail", "efficient", "--no-ocr", "--save", "--exports", "viewer"]) == 0
    assert arts.calls[0][1]["paperless"] is True
    assert exp.calls[0][1]["paperless_doc"] is None


def test_export_subcommand_rebuilds_without_rewatch(cut_clip, tmp_path, capsys):
    assert run_watch([str(cut_clip), "--detail", "efficient", "--no-ocr", "--exports", "none"]) == 0
    out = tmp_path / "exp"
    capsys.readouterr()
    assert run_watch(["export", "cuts.mp4", "--formats", "json,viewer", "--out", str(out), "--json"]) == 0
    res = json.loads(capsys.readouterr().out)
    assert res["video_key"].startswith("file:")
    assert {k for k, v in res["exports"].items() if v.get("path")} == {"json", "viewer", "notes"}
    assert (out / "viewer.html").is_file() and (out / "notes.md").is_file()
