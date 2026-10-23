"""Confidence score for evidence packs (0.5.1 Task 2)."""
from __future__ import annotations

from memory.confidence import HIGH, MEDIUM, assess
from memory.embeddings import embed_pending
from memory.index import record_run
from memory.retrieval import ask
from memory_fakes import KEY, FakeEmbedder, fake_run, fr, jpg, seg


def ev(i, kind, t, text, score, cos=None, frame=None, context=""):
    return {"id": f"E{i}", "kind": kind, "t": t, "t_end": None, "timestamp": "00:00",
            "text": text, "context": context, "frame": frame, "score": score, "cos": cos}


def test_empty_is_low():
    r = assess("lo que sea", [])
    assert r["level"] == "low" and r["value"] == 0.0


def test_exact_phrase_transcript_is_high():
    e = [ev(1, "transcript", 12, "los elefantes tienen trompas muy largas", 1.0, frame={"t": 12}),
         ev(2, "transcript", 40, "otra cosa", 0.2)]
    r = assess("¿qué tienen los elefantes? trompas largas", e)
    assert r["level"] == "high" and r["value"] >= HIGH


def test_unrelated_question_is_low():
    e = [ev(1, "transcript", 12, "los elefantes tienen trompas", 1.0), ev(2, "transcript", 30, "y eso es todo", 0.9)]
    r = assess("receta de la paella valenciana", e)
    assert r["level"] == "low" and r["value"] < MEDIUM


def test_cross_modal_agreement_raises_value():
    q = "pantalla error"
    base = [ev(1, "transcript", 10, "hay un error", 1.0), ev(2, "transcript", 60, "nada", 0.5)]
    multi = [ev(1, "transcript", 10, "hay un error", 1.0), ev(2, "ocr", 12, "ERROR 404", 0.5, frame={"t": 12})]
    a, b = assess(q, base), assess(q, multi)
    assert b["signals"]["agreement"] > a["signals"]["agreement"] and b["value"] > a["value"]


def test_cosine_strength_used_when_present():
    e = [ev(1, "transcript", 1, "zzz", 1.0, cos=0.7)]
    assert assess("palabras ajenas", e)["signals"]["strength"] == 1.0


def test_ask_evidence_has_cos(conn, tmp_path):
    record_run(conn, fake_run(segments=[seg(0, 5, "el coche rojo"), seg(30, 35, "la casa azul")],
                              frames=[fr(2, jpg(tmp_path, "red"), "COCHE")]))
    fts = ask(conn, KEY, "coche")
    assert fts["evidence"] and all(e["cos"] is None for e in fts["evidence"])
    emb = FakeEmbedder()
    embed_pending(conn, emb)
    hyb = ask(conn, KEY, "coche", embedder=emb)
    assert any(isinstance(e["cos"], float) for e in hyb["evidence"])
