"""Hybrid retrieval + OCR duplicate collapse (Task 6)."""
from __future__ import annotations

import pytest

from memory.embeddings import embed_pending
from memory.index import record_run
from memory.retrieval import Hit, collapse_ocr_duplicates, hybrid_scores, search
from memory_fakes import FakeEmbedder, fake_run, fr, jpg, seg

OTHER = "file:0123456789abcdef"


def test_hybrid_scores_exact():
    s = hybrid_scores({"a": 3.0, "b": 1.0}, {"a": 0.2, "c": 0.8})
    assert s == pytest.approx({"a": 0.45, "b": 0.0, "c": 0.55})
    assert hybrid_scores({"a": 3.0, "b": 1.0}, {}) == pytest.approx({"a": 1.0, "b": 0.0})
    assert hybrid_scores({"a": 2.0}, {}) == {"a": 1.0}


def test_search_diacritic_insensitive(conn):
    record_run(conn, fake_run(segments=[seg(12, 18, "la canción del verano")]))
    hits = search(conn, "CANCION")
    assert hits and hits[0].kind == "transcript" and hits[0].t == 12.0


def test_search_scoped_to_video(conn):
    record_run(conn, fake_run(segments=[seg(0, 5, "hola mundo")]))
    record_run(conn, fake_run(video_key=OTHER, source="/tmp/x.mp4", segments=[seg(0, 5, "hola mundo")]))
    assert {h.video_key for h in search(conn, "hola")} == {"yt:AAAAAAAAAAA", OTHER}
    assert {h.video_key for h in search(conn, "hola", video_key=OTHER)} == {OTHER}


def test_ocr_collapse_before_cut(conn, tmp_path):
    colors = {1: "red", 2: "blue", 3: "white", 60: "green"}  # distinct pHashes so frame dedup keeps all four
    frames = [fr(t, jpg(tmp_path, c), "ERROR 404") for t, c in colors.items()]
    record_run(conn, fake_run(segments=[seg(30, 35, "el error 404 aparece")], frames=frames))
    hits = search(conn, "error 404", k=3)
    ocr = [h for h in hits if h.kind == "ocr"]
    assert [(h.dup_count, h.t_first, h.t_last) for h in ocr] == [(3, 1.0, 3.0), (1, 60.0, 60.0)]
    assert any(h.kind == "transcript" for h in hits)


def test_search_hybrid_with_fake_embedder(conn):
    record_run(conn, fake_run(segments=[seg(0, 5, "automovil rojo")]))
    embed_pending(conn, FakeEmbedder())
    hits = search(conn, "coche", embedder=FakeEmbedder())
    assert hits[0].text == "automovil rojo" and hits.mode == "hybrid"
    none = search(conn, "coche")
    assert none == [] and none.mode == "fts-only"


def test_collapse_chain_and_order():
    def h(t, text, score, vk="v"):
        return Hit(vk, "ocr", "ocr_blocks", int(t), t, None, text, score)

    hits = [h(10, "ERROR 404", 0.5), h(1, "ERROR 404", 0.9), h(20, "ERROR 404", 0.7), h(100, "ERROR 404", 0.1),
            h(5, "ERROR 404", 0.4, vk="w")]
    out = collapse_ocr_duplicates(hits)
    assert [(x.video_key, x.t, x.dup_count, x.t_first, x.t_last) for x in out] == [
        ("v", 1, 3, 1, 20), ("v", 100, 1, 100, 100), ("w", 5, 1, 5, 5)]
