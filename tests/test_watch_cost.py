"""watch.py --plan / ceiling / spend.json (0.6.0 phase B)."""
from __future__ import annotations

import json

import pytest

import cost
import spend
import watch
from memory_fakes import run_watch


def test_plan_prints_and_does_nothing_else(cut_clip, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(watch, "download", lambda *a, **k: pytest.fail("--plan must not download"))
    out = tmp_path / "w"
    assert run_watch([str(cut_clip), "--plan", "--out-dir", str(out)]) == 0
    text = capsys.readouterr().out
    assert text.startswith("# watch: cost plan") and "total: $" in text
    assert json.loads((out / "plan.json").read_text())["source"] == str(cut_clip)
    assert not (out / "report.md").exists()


def test_ceiling_blocks_before_any_paid_call(cut_clip, tmp_path, monkeypatch):
    monkeypatch.setenv("WATCH_COST_CEILING_USD", "0.01")
    monkeypatch.setattr(watch, "_probe", lambda args: {"duration": 7200, "has_audio": True})
    monkeypatch.setattr(cost, "_whisper_route", lambda args: "openai")
    monkeypatch.setattr(watch, "transcribe_video", lambda *a, **k: pytest.fail("paid call after the gate"))
    with pytest.raises(SystemExit) as exc:
        run_watch([str(cut_clip), "--whisper", "openai", "--out-dir", str(tmp_path / "w")])
    assert "exceeds WATCH_COST_CEILING_USD" in str(exc.value)


def test_allow_cost_skips_the_gate(cut_clip, tmp_path, monkeypatch):
    monkeypatch.setenv("WATCH_COST_CEILING_USD", "0.01")
    monkeypatch.setattr(watch, "_probe", lambda args: pytest.fail("no probe with --allow-cost"))
    assert run_watch([str(cut_clip), "--detail", "efficient", "--no-ocr", "--allow-cost",
                      "--out-dir", str(tmp_path / "w")]) == 0


def test_run_writes_spend_json(cut_clip, tmp_path, monkeypatch):
    out = tmp_path / "w"
    assert run_watch([str(cut_clip), "--detail", "efficient", "--no-ocr", "--out-dir", str(out)]) == 0
    data = json.loads((out / "spend.json").read_text())
    assert data["total_usd"] == 0.0 and data["items"] == []


def test_whisper_spend_recorded(cut_clip, tmp_path, monkeypatch):
    out = tmp_path / "w"
    monkeypatch.setattr(watch, "resolve_backend", lambda pref: ("groq", "k", None))
    monkeypatch.setattr(watch, "get_metadata", lambda p: {"duration_seconds": 600.0, "width": 64, "height": 36,
                                                          "codec": "h264", "has_audio": True})
    monkeypatch.setattr(watch, "transcribe_video", lambda *a, **k: ([{"start": 0.0, "end": 2.0, "text": "hola"}], "groq"))
    assert run_watch([str(cut_clip), "--detail", "transcript", "--out-dir", str(out)]) == 0
    data = json.loads((out / "spend.json").read_text())
    assert data["items"][0]["provider"] == "groq" and data["total_usd"] == pytest.approx(0.111 / 6, rel=0.01)
