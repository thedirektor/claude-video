"""SQLite index: schema v1, migrations, merge rules, FTS kept in sync (Task 3)."""
from __future__ import annotations

import pytest

import frames
from memory.index import (
    connect, forget, frame_kind, record_run, set_library_ref, touch, transcript_rank,
)
from memory_fakes import (
    KEY, chunk_texts, fake_run, fr, frame_times, fts_match, immich_ids, jpg, meta,
    ocr_count, rank, run_count, seg, source_of, stored_times, texts,
)


def test_merge_better_source_replaces_and_clears_fts(conn):          # Review Focus 2
    record_run(conn, fake_run(transcript_source="captions", segments=[seg(0, 5, "viejo texto")]))
    r = record_run(conn, fake_run(transcript_source="whisper (groq)", segments=[seg(0, 5, "nuevo")]))
    assert r.transcript_replaced and texts(conn, "segments") == ["nuevo"]
    assert fts_match(conn, "viejo") == [] and chunk_texts(conn) == ["nuevo"]


def test_merge_lower_rank_keeps(conn):
    record_run(conn, fake_run(transcript_source="whisper (groq)", segments=[seg(0, 5, "a")]))
    r = record_run(conn, fake_run(transcript_source="transcriptapi (es)", segments=[seg(0, 5, "b")]))
    assert not r.transcript_replaced and texts(conn, "segments") == ["a"] and r.evidence_version == 1


def test_merge_tie_newest_wins(conn):
    record_run(conn, fake_run(transcript_source="whisper (groq)", segments=[seg(0, 5, "uno")]))
    record_run(conn, fake_run(transcript_source="whisper (openai)", segments=[seg(0, 5, "dos")]))
    assert texts(conn, "segments") == ["dos"]


def test_merge_range_run_replaces_inside_only(conn):
    record_run(conn, fake_run(transcript_source="captions", segments=[seg(0, 5, "alfa"), seg(10, 15, "beta"), seg(30, 35, "gamma")]))
    record_run(conn, fake_run(transcript_source="whisper (groq)", range=(10.0, 20.0), segments=[seg(10, 14, "delta")]))
    assert texts(conn, "segments") == ["alfa", "delta", "gamma"]
    assert rank(conn) == transcript_rank("captions")
    assert fts_match(conn, "alfa") and fts_match(conn, "gamma") and fts_match(conn, "delta")
    assert fts_match(conn, "beta") == []                  # only the replaced segment's FTS row is gone


def test_record_run_canonical_source_and_immich_ids(conn, tmp_path):
    record_run(conn, fake_run(source="https://youtu.be/AAAAAAAAAAA?si=x", frames=[fr(5.0, jpg(tmp_path, "red"), None)],
                              library={"immich": {"album_id": None, "frames": {"5000": "a1"}}}))
    assert source_of(conn) == "https://www.youtube.com/watch?v=AAAAAAAAAAA" and immich_ids(conn) == ["a1"]


def test_merge_frames_union_dedup(conn, tmp_path):
    same, other = jpg(tmp_path, "red"), jpg(tmp_path, "blue")
    record_run(conn, fake_run(frames=[fr(5.0, same, "TITLE")]))
    record_run(conn, fake_run(frames=[fr(5.4, same, "TITLE"), fr(5.4, other, None), fr(7.0, same, None)]))
    assert frame_times(conn) == [5.0, 5.4, 7.0] and ocr_count(conn) == 1


def test_backfill_origin_ref_idempotent(conn):
    r1 = record_run(conn, fake_run(origin="backfill", origin_ref="paperless:39"))
    r2 = record_run(conn, fake_run(origin="backfill", origin_ref="paperless:39"))
    assert not r1.skipped and r2.skipped and run_count(conn) == 1


def test_transcript_rank_order():
    assert [transcript_rank(s) for s in ("whisper (assemblyai, diarized)", "whisper (groq)",
            "transcriptapi (es)", "captions", "gemini (m, approximate timestamps)", None)] == [5, 4, 3, 2, 1, 0]


def test_migrations_idempotent(tmp_path):
    connect(tmp_path / "i.db").close(); c = connect(tmp_path / "i.db")
    assert meta(c, "schema_version") == "1" and c.execute("pragma journal_mode").fetchone()[0] == "wal"


# --- extra coverage beyond the brief -----------------------------------------

def test_replacement_drops_chunk_embeddings(conn):                    # Review Focus 2
    record_run(conn, fake_run(transcript_source="captions", segments=[seg(0, 5, "viejo")]))
    cid = conn.execute("SELECT id FROM chunks").fetchone()[0]
    conn.execute("INSERT INTO embeddings VALUES ('chunks', ?, ?, 'm', 1, x'00')", (cid, KEY))
    conn.execute("INSERT INTO embeddings VALUES ('notes', 99, ?, 'm', 1, x'00')", (KEY,))
    conn.commit()
    record_run(conn, fake_run(transcript_source="whisper (groq)", segments=[seg(0, 5, "nuevo")]))
    left = [r[0] for r in conn.execute("SELECT src_table FROM embeddings")]
    assert left == ["notes"]


def test_lower_rank_range_run_keeps(conn):
    record_run(conn, fake_run(transcript_source="whisper (groq)", segments=[seg(10, 15, "bueno")]))
    r = record_run(conn, fake_run(transcript_source="captions", range=(0.0, 60.0), segments=[seg(10, 15, "malo")]))
    assert not r.transcript_replaced and texts(conn, "segments") == ["bueno"]


def test_full_run_updates_rank_and_metadata_nulls_never_overwrite(conn):
    record_run(conn, fake_run(title="Título", duration=60.0, transcript_source="captions", segments=[seg(0, 1, "x")]))
    record_run(conn, fake_run(title=None, channel="chan", transcript_source="whisper (groq)", segments=[seg(0, 1, "y")]))
    row = conn.execute("SELECT * FROM videos").fetchone()
    assert (row["title"], row["title_norm"], row["duration"], row["channel"]) == ("Título", "titulo", 60.0, "chan")
    assert rank(conn) == 4 and row["transcript_source"] == "whisper (groq)" and row["evidence_version"] == 2


def test_notes_append_and_fts_kinds(conn):
    record_run(conn, fake_run(notes=[{"kind": "backend_answer", "t_start": None, "t_end": None, "text": "respuesta uno"}]))
    r = record_run(conn, fake_run(notes=[{"kind": "backend_answer", "t_start": 3.0, "t_end": 4.0, "text": "respuesta dos"}],
                                  segments=[seg(1, 2, "hola")], transcript_source="captions"))
    assert texts(conn, "notes") == ["respuesta uno", "respuesta dos"] or texts(conn, "notes") == ["respuesta dos", "respuesta uno"]
    kinds = sorted(r[0] for r in conn.execute("SELECT kind FROM fts"))
    assert kinds == ["note", "note", "transcript"] and r.evidence_version == 2
    assert stored_times(conn) == {1.0, 3.0}


def test_new_frames_and_ocr_fts(conn, tmp_path):
    p = jpg(tmp_path, "green")
    r = record_run(conn, fake_run(frames=[fr(2.0, p, "Pantalla de Código"), fr(9.0, None, None, kind="cue")]))
    fid = conn.execute("SELECT id FROM frames WHERE t = 2.0").fetchone()[0]
    assert r.new_frames == [(fid, str(p))] and r.added["frames"] == 2 and r.added["ocr"] == 1
    assert fts_match(conn, "codigo") == ["Pantalla de Código"]
    assert conn.execute("SELECT store_path FROM frames WHERE id = ?", (fid,)).fetchone()[0] is None


def test_touch_forget_and_library_refs(conn):
    record_run(conn, fake_run(segments=[seg(0, 1, "uno")], transcript_source="captions",
                              library={"nextcloud": {"folder": "/Watch/x"}, "paperless": {"task_ids": ["t-1", "t-2"]}}))
    row = conn.execute("SELECT nextcloud_path, paperless_task_id FROM videos").fetchone()
    assert tuple(row) == ("/Watch/x", "t-1")
    conn.execute("UPDATE videos SET last_access = NULL"); conn.commit()
    touch(conn, KEY)
    assert conn.execute("SELECT last_access FROM videos").fetchone()[0]
    set_library_ref(conn, KEY, paperless_doc_id=39, immich_album_id="alb")
    assert tuple(conn.execute("SELECT paperless_doc_id, immich_album_id FROM videos").fetchone()) == (39, "alb")
    with pytest.raises(ValueError):
        set_library_ref(conn, KEY, title="nope")
    assert forget(conn, KEY) is True and forget(conn, KEY) is False
    for table in ("videos", "runs", "segments", "chunks", "fts"):
        assert conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0


def test_frame_kind():
    assert frame_kind({"reason": "transcript-cue", "path": "a_hires.jpg"}) == "cue"
    assert frame_kind({"path": "/x/frame_0001_hires.jpg"}) == "ocr"
    assert frame_kind({"reason": "uniform"}) == "even" and frame_kind({"reason": "two-pass"}) == "even"
    assert frame_kind({"reason": "scene", "path": "f.jpg"}) == "scene" and frame_kind({}) == "scene"


def test_thumb_fingerprint(tmp_path):
    assert frames.thumb_fingerprint(None) is None
    assert frames.thumb_fingerprint(tmp_path / "missing.jpg") is None
    fp = frames.thumb_fingerprint(jpg(tmp_path, "white"))
    assert isinstance(fp, bytes) and len(fp) == frames.DEDUP_THUMB ** 2
