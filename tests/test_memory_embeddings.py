"""Chunking + optional embeddings (Task 5)."""
from __future__ import annotations

import sys

import pytest

from memory.chunking import chunk_segments
from memory.embeddings import cosine_many, embed_pending, get_embedder, pack, unpack
from memory.index import record_run
from memory_fakes import FakeEmbedder, embedding_models, fake_run, fr, seg

SEGS = [(0, 6, "A"), (6, 13, "A"), (13, 21, "A"), (21, 25, "A"), (25, 30, "B"), (30, 70, "B"), (70, 72, "A")]


def test_chunk_segments_windows_and_turns():
    out = chunk_segments([{"start": s, "end": e, "speaker": sp, "text": f"{s}"} for s, e, sp in SEGS])
    assert [(c["t_start"], c["t_end"], c["speaker"]) for c in out] == [
        (0, 21, "A"), (21, 30, None), (30, 70, "B"), (70, 72, "A"),
    ]


def test_chunk_segments_empty_and_joined_text():
    assert chunk_segments([]) == []
    out = chunk_segments([seg(0, 2, "hola"), seg(2, 4, "mundo")])
    assert out == [{"t_start": 0.0, "t_end": 4.0, "speaker": None, "text": "hola mundo"}]


def test_embed_pending_and_model_change(conn):
    record_run(conn, fake_run(
        transcript_source="whisper (groq)",
        segments=[seg(0, 25, "el coche rojo"), seg(25, 60, "otra cosa")],
        frames=[fr(5, None, "texto en pantalla")],
        notes=[{"kind": "answer", "text": "una nota"}],
    ))
    n_rows = sum(conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in ("chunks", "ocr_blocks", "notes"))
    assert n_rows == 4
    assert embed_pending(conn, FakeEmbedder()) == 4 and embed_pending(conn, FakeEmbedder()) == 0
    e2 = FakeEmbedder()
    e2.model_name = "fake-16b"
    assert embed_pending(conn, e2) == 4 and embedding_models(conn) == {"fake-16b"}
    assert embed_pending(conn, None) == 0


def test_fake_embedder_synonyms():
    e = FakeEmbedder()
    a, b = e.embed(["coche"]), e.embed(["automovil"])
    assert cosine_many(a[0], [pack(b[0])], 16) == pytest.approx([1.0])


def test_get_embedder_disabled_returns_none_once(capsys):  # WATCH_EMBEDDINGS=0 from conftest
    sys.modules.pop("fastembed", None)
    assert get_embedder() is None and get_embedder() is None
    assert "fastembed" not in sys.modules and capsys.readouterr().err.count("[watch]") <= 1


def test_cosine_many_without_numpy(monkeypatch):
    monkeypatch.setitem(sys.modules, "numpy", None)
    assert cosine_many([1, 0], [pack([1, 0]), pack([0, 1])], 2) == pytest.approx([1.0, 0.0])


def test_pack_roundtrip():
    assert unpack(pack([0.5, -2.0]), 2) == [0.5, -2.0]
