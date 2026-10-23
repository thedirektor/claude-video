"""Schema v2: answer cache, escalation log, stats (0.5.1 Task 4)."""
from __future__ import annotations

import sqlite3

from memory import answers, index
from memory.index import bump_evidence, forget, record_run
from memory_fakes import KEY, FakeEmbedder, fake_run, seg

RES = {"answer": "Habla del mar [E1 00:12].", "found": True}


def _ev(conn):
    return conn.execute("SELECT evidence_version FROM videos WHERE video_key=?", (KEY,)).fetchone()[0]


def test_migration_v2_on_v1_db(tmp_path):
    db = tmp_path / "v1.db"
    raw = sqlite3.connect(db)
    for stmt in index._statements(index.MIGRATIONS[0]):
        raw.execute(stmt)
    raw.execute("INSERT INTO meta VALUES ('schema_version', '1')")
    raw.execute("INSERT INTO videos (video_key, title) VALUES ('yt:x', 'keep me')")
    raw.commit()
    raw.close()
    conn = index.connect(db)
    assert index._schema_version(conn) == 2 == index.SCHEMA_VERSION
    assert conn.execute("SELECT title FROM videos").fetchone()[0] == "keep me"
    assert conn.execute("SELECT count(*) FROM answers").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM escalations").fetchone()[0] == 0


def test_lookup_exact_then_semantic(conn):
    record_run(conn, fake_run(segments=[seg(0, 5, "hola")]))
    emb = FakeEmbedder()
    ev = _ev(conn)
    answers.store(conn, KEY, ev, "De qué habla el coche", RES, provider="gemini", tokens_in=10, tokens_out=5, embedder=emb)
    hit = answers.lookup(conn, KEY, ev, "  de que HABLA el coche ", embedder=None)  # exact after normalising
    assert hit == {**RES, "cached": True}
    sem = answers.lookup(conn, KEY, ev, "De qué habla el carro", embedder=emb)  # synonym → same vector
    assert sem == {**RES, "cached": True}
    assert answers.lookup(conn, KEY, ev, "algo totalmente distinto aquí", embedder=emb) is None


def test_lookup_misses_after_evidence_bump(conn):
    record_run(conn, fake_run(segments=[seg(0, 5, "hola")]))
    ev = _ev(conn)
    answers.store(conn, KEY, ev, "q", RES)
    assert bump_evidence(conn, KEY) == ev + 1
    assert answers.lookup(conn, KEY, _ev(conn), "q") is None


def test_store_prunes_old_versions(conn):
    record_run(conn, fake_run(segments=[seg(0, 5, "hola")]))
    answers.store(conn, KEY, 1, "q1", RES)
    answers.store(conn, KEY, 2, "q2", RES)
    rows = conn.execute("SELECT evidence_version FROM answers").fetchall()
    assert [r[0] for r in rows] == [2]


def test_stats_counts_tokens_saved(conn):
    record_run(conn, fake_run(segments=[seg(0, 5, "hola")]))
    ev = _ev(conn)
    answers.store(conn, KEY, ev, "q", RES, provider="gemini", tokens_in=100, tokens_out=20)
    answers.store(conn, KEY, ev, "r", {"answer": None}, provider=None)
    answers.lookup(conn, KEY, ev, "q")
    answers.lookup(conn, KEY, ev, "q")
    s = answers.stats(conn)
    assert s == {"answers": 2, "cache_hits": 2, "tokens_used": 120, "tokens_saved": 240,
                 "by_provider": {"gemini": 1, "none": 1}}


def test_forget_clears_answers(conn):
    record_run(conn, fake_run(segments=[seg(0, 5, "hola")]))
    answers.store(conn, KEY, _ev(conn), "q", RES)
    conn.execute("INSERT INTO escalations (video_key, t, added, created) VALUES (?, 1, 0, 'x')", (KEY,))
    forget(conn, KEY)
    assert conn.execute("SELECT count(*) FROM answers").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM escalations").fetchone()[0] == 0
