"""Grounded answers: answer() orchestration + CLI (0.5.1 Task 6)."""
from __future__ import annotations

import json

import pytest

from memory import answer as answer_mod
from memory import cli, index
from memory.answer import REFUSAL, answer
from memory.index import _fts_add, bump_evidence, record_run
from memory.llm import LLMReply
from memory_fakes import KEY, fake_run, seg


@pytest.fixture
def song(conn):
    record_run(conn, fake_run(title="La Canción", segments=[
        seg(0, 5, "hola a todos"), seg(12, 18, "esta canción habla del mar azul"), seg(40, 45, "hola de nuevo")]))
    return conn


class FakeLLM:
    def __init__(self, text="Habla del mar azul [E1]. Ver 9:99"):
        self.calls, self.text = [], text

    def __call__(self, system, user, max_tokens=700):
        self.calls.append(user)
        return LLMReply(self.text, "fake", "fake-1", 30, 7) if self.text is not None else None


def no_escalate(*a, **k):
    raise AssertionError("escalation must not run")


def test_answer_happy_path_cites_evidence(song):
    llm = FakeLLM()
    r = answer(song, "La Canción", "¿de qué habla la canción del mar?", complete=llm, escalate_fn=no_escalate)
    assert r["found"] is True and r["cached"] is False and len(llm.calls) == 1
    assert "[E1 00:12]" in r["answer"] and "9:99" not in r["answer"]
    assert r["citations"][0]["timestamp"] == "00:12"
    assert {c["timestamp"] for c in r["citations"]} <= {e["timestamp"] for e in r["evidence"]}
    assert r["confidence"]["level"] in ("high", "medium") and r["provider"] == "fake"


def test_answer_second_call_cached_no_llm(song):
    llm = FakeLLM()
    answer(song, "La Canción", "¿de qué habla la canción del mar?", complete=llm, escalate_fn=no_escalate)
    r = answer(song, "La Canción", "De qué habla la canción del mar", complete=llm, escalate_fn=no_escalate)
    assert r["cached"] is True and len(llm.calls) == 1
    bump_evidence(song, KEY)
    answer(song, "La Canción", "¿de qué habla la canción del mar?", complete=llm, escalate_fn=no_escalate)
    assert len(llm.calls) == 2


def test_answer_low_confidence_refuses_without_llm(song):
    llm = FakeLLM()
    r = answer(song, "La Canción", "receta de paella valenciana", complete=llm, escalate_fn=lambda *a, **k: 0)
    assert r["found"] is False and r["answer"] == REFUSAL and llm.calls == []
    assert r["confidence"]["level"] == "low" and r["citations"] == []


def test_answer_low_confidence_escalates_then_answers(song):
    llm = FakeLLM("El precio aparece en pantalla [E1].")
    seen = []

    def fake_escalate(conn, key, times, **kw):
        seen.append(times)
        cur = conn.execute("INSERT INTO ocr_blocks (video_key, t, text) VALUES (?, 1.0, 'PRECIO OFERTA DESCUENTO')", (key,))
        _fts_add(conn, "ocr", key, "ocr_blocks", cur.lastrowid, 1.0, "PRECIO OFERTA DESCUENTO")
        conn.commit()
        bump_evidence(conn, key)
        return 1

    r = answer(song, "La Canción", "hola precio oferta descuento", complete=llm, escalate_fn=fake_escalate)
    assert seen and r["escalated"] == 1 and r["found"] is True and len(llm.calls) == 1
    assert "PRECIO OFERTA" in llm.calls[0]


def test_answer_all_providers_fail_evidence_only(song):
    llm = FakeLLM(text=None)
    r = answer(song, "La Canción", "¿de qué habla la canción del mar?", complete=llm, escalate_fn=no_escalate)
    assert r["answer"] is None and r["found"] is None and r["evidence"] and "note" in r
    assert song.execute("SELECT count(*) FROM answers").fetchone()[0] == 0


def test_answer_model_not_found_reply_refuses(song):
    r = answer(song, "La Canción", "¿de qué habla la canción del mar?", complete=FakeLLM("NOT_FOUND"),
               escalate_fn=no_escalate)
    assert r["found"] is False and r["answer"] == REFUSAL and len(r["nearest"]) >= 1


def test_cli_ask_answer_and_stats_json(tmp_path, capsys, monkeypatch):
    c = index.connect()
    record_run(c, fake_run(title="La Canción", segments=[seg(12, 18, "esta canción habla del mar azul")]))
    c.close()
    monkeypatch.setattr(answer_mod.llm, "complete", FakeLLM())
    assert cli.main(["ask", "La Canción", "¿de qué habla la canción del mar?", "--answer", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["found"] is True and "[E1 00:12]" in out["answer"]
    assert cli.main(["ask", "La Canción", "¿de qué habla la canción del mar?", "--answer"]) == 0
    text = capsys.readouterr().out
    assert "[E1 00:12]" in text and "Confidence:" in text
    assert cli.main(["stats", "--json"]) == 0
    s = json.loads(capsys.readouterr().out)
    assert s["answers"] == 1 and s["cache_hits"] == 1


def test_watch_py_dispatches_stats(capsys):
    from memory_fakes import run_watch
    assert run_watch(["stats", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["answers"] == 0


def test_answer_no_evidence_escalates_around_stored_frames(conn):
    from memory_fakes import fr
    record_run(conn, fake_run(title="Mudo", frames=[fr(t, None, None) for t in (1.0, 4.0, 7.0, 10.0, 13.0)]))
    seen = []

    def fake_escalate(c, key, times, **kw):
        seen.append(times)
        cur = c.execute("INSERT INTO ocr_blocks (video_key, t, text) VALUES (?, 7.0, 'PRECIO ESPECIAL 42 EUROS')", (key,))
        _fts_add(c, "ocr", key, "ocr_blocks", cur.lastrowid, 7.0, "PRECIO ESPECIAL 42 EUROS")
        c.commit()
        bump_evidence(c, key)
        return 1

    r = answer(conn, "Mudo", "¿cuál es el precio especial en euros?", complete=FakeLLM("Cuesta 42 euros [E1]."),
               escalate_fn=fake_escalate)
    assert seen == [[1.0, 7.0, 13.0]] and r["found"] is True and "[E1 00:07]" in r["answer"]
