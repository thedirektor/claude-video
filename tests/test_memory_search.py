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


def test_boundary_segment_folds_into_chunk_it_starts(conn):
    segs = [seg(i * 10, i * 10 + 10, f"alpha{i} beta{i}") for i in range(12)]
    record_run(conn, fake_run(segments=segs))
    bounds = conn.execute("SELECT t_start, t_end, text FROM chunks ORDER BY t_start").fetchall()
    hits = search(conn, "alpha4 alpha5 alpha6 alpha7")
    assert hits
    for h in hits:
        assert "alpha" in h.text and h.t_end is not None
    h4 = search(conn, "alpha4")[0]
    assert "alpha4" in h4.text and h4.t == 40.0
    assert any(b[0] <= h4.t <= b[1] and b[2] == h4.text for b in bounds)


@pytest.mark.parametrize("q", ['"', 'a"b', "foo:bar", "AND", "OR", "NEAR(a b)", "a*", "(", "((a", "^a", "{a}", "\\",
                               "😀", "你好世界", '""', "  ", "\x00", "x\x00y"])
def test_hostile_queries_do_not_raise(conn, q):
    record_run(conn, fake_run(segments=[seg(0, 5, "hola mundo")]))
    assert isinstance(search(conn, q), list)


class BoomEmbedder:
    model_name = "boom"
    dim = 16

    def embed(self, texts):
        raise RuntimeError("model gone")


def test_embedder_failure_falls_back_to_fts(conn, capsys):
    record_run(conn, fake_run(segments=[seg(0, 5, "hola mundo")]))
    hits = search(conn, "hola", embedder=BoomEmbedder())
    assert hits and hits.mode == "fts-only"
    assert "vector search unavailable" in capsys.readouterr().err


def test_vector_scope_model_and_nonpositive(conn):
    record_run(conn, fake_run(segments=[seg(0, 5, "automovil rojo")]))
    record_run(conn, fake_run(video_key=OTHER, source="/tmp/x.mp4", segments=[seg(0, 5, "automovil azul")]))
    embed_pending(conn, FakeEmbedder())
    assert {h.video_key for h in search(conn, "coche", embedder=FakeEmbedder())} == {"yt:AAAAAAAAAAA", OTHER}
    assert {h.video_key for h in search(conn, "coche", video_key=OTHER, embedder=FakeEmbedder())} == {OTHER}

    class Other(FakeEmbedder):
        model_name = "other-model"

    miss = search(conn, "coche", embedder=Other())
    assert miss == [] and miss.mode == "fts-only"  # other-model embeddings ignored

    class Orth(FakeEmbedder):
        def embed(self, texts):
            return [[-1.0] + [0.0] * 15 for _ in texts]

    conn.execute("UPDATE embeddings SET vector = ?", (__import__("memory.embeddings", fromlist=["pack"]).pack([1.0] + [0.0] * 15),))
    assert search(conn, "zzz", embedder=Orth()) == []  # cosine <= 0 dropped
