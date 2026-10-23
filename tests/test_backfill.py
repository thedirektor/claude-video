"""reportparse + Paperless backfill + rebuild (Task 10)."""
from __future__ import annotations

import json

from memory import backfill, index
from memory.index import record_run
from memory.reportparse import parse_report
from memory_fakes import KEY, fake_run, fr, frame_times, jpg, seg, texts
from memory import store


def test_parse_report_immich_and_ocr_tail():
    md = (
        "# watch: video report\n\n- **Source:** https://youtu.be/AAAAAAAAAAA\n- **Title:** La demo\n"
        "- **Duration:** 01:05 (65.0s)\n\n## Frames\n\n"
        "- `/w/f1.jpg` (t=00:05, reason=keyframe) [speech] — Immich: https://i/photos/a1 — OCR: f(x) — y\n"
        "- `/w/f2.jpg` (t=1:02:03, reason=scene)\n\n"
        "## Transcript\n\n_Source: whisper (assemblyai, diarized)._\n\n```\n"
        "[Speaker B] (62:03-62:09) hola\n[Speaker A] (0:05-0:07) que tal\n```\n"
    )
    p = parse_report(md)
    assert p["frames"][0] == {"t": 5.0, "ocr": "f(x) — y", "immich_asset_id": "a1"}
    assert p["frames"][1] == {"t": 3723.0, "ocr": "", "immich_asset_id": None}
    assert p["segments"][0]["start"] == 3723.0 and p["segments"][0]["end"] == 3729.0
    assert p["segments"][0]["speaker"] == "Speaker B"
    assert (p["source"], p["title"], p["duration"]) == ("https://youtu.be/AAAAAAAAAAA", "La demo", 65.0)
    assert p["transcript_source"] == "whisper (assemblyai, diarized)"


def test_parse_report_plain_lines():
    md = "## Transcript\n\n_Source: captions. Filtered to 00:00 → 02:00:_\n\n```\n[75:03] x\n[75:10] y\n[1:16:00] z\n```\n"
    p = parse_report(md)
    assert [s["start"] for s in p["segments"]] == [4503.0, 4510.0, 4560.0]
    assert [s["end"] for s in p["segments"]] == [4510.0, 4560.0, 4565.0]
    assert p["transcript_source"] == "captions"
    assert parse_report("") == {"source": None, "title": None, "transcript_source": None,
                                "duration": None, "segments": [], "frames": []}


def _doc(i, vid):
    return {"id": i, "title": f"Video {i} (abc)", "content": (
        f"- **Source:** https://www.youtube.com/watch?v={vid}\n- **Title:** Video {i}\n\n## Frames\n\n"
        f"- `/w/f.jpg` (t=00:0{i}, reason=scene) — Immich: https://i/photos/as{i} — OCR: texto {i}\n\n"
        f"## Transcript\n\n_Source: captions._\n\n```\n[00:0{i}] hola {i}\n```\n")}


def test_backfill_idempotent_two_docs(conn):
    pages = {
        backfill.PAGE_PATH: {"results": [_doc(1, "AAAAAAAAAA1")], "next": "https://pl/api/documents/?page=2&tags__name__iexact=watch"},
        "/api/documents/?page=2&tags__name__iexact=watch": {"results": [_doc(2, "AAAAAAAAAA2")], "next": None},
    }
    got = []

    def get_json(path):
        got.append(path)
        return pages[path]

    albums = {"Video 1 (abc)": "alb1"}
    r = backfill.backfill_paperless(conn, {"PAPERLESS_URL": "https://pl"}, get_json=get_json,
                                    find_album=albums.get)
    assert r == {"seen": 2, "added": 2, "skipped": 0} and len(got) == 2
    row = conn.execute("SELECT * FROM videos WHERE video_key = 'yt:AAAAAAAAAA1'").fetchone()
    assert row["paperless_doc_id"] == 1 and row["immich_album_id"] == "alb1" and row["title"] == "Video 1"
    assert conn.execute("SELECT immich_asset_id FROM frames WHERE video_key='yt:AAAAAAAAAA1'").fetchone()[0] == "as1"
    assert texts(conn, "segments", "yt:AAAAAAAAAA1") == ["hola 1"]
    assert backfill.backfill_paperless(conn, {}, get_json=get_json, find_album=albums.get) == \
        {"seen": 2, "added": 0, "skipped": 2}
    assert conn.execute("SELECT count(*) FROM segments").fetchone()[0] == 2


def test_backfill_writes_qdrant_inbox(conn, tmp_path, monkeypatch):
    inbox = tmp_path / "inbox"
    monkeypatch.setenv("WATCH_QDRANT_INBOX", str(inbox))
    backfill.backfill_paperless(conn, {}, get_json=lambda p: {"results": [_doc(1, "AAAAAAAAAA1")]},
                                find_album=lambda n: None)
    assert len(list(inbox.glob("*.md"))) == 1


def test_rebuild_restores_rows(conn, tmp_path):
    root = tmp_path / "store"
    run = fake_run(title="Demo", segments=[seg(0, 5, "hola"), seg(12, 18, "adios mundo")],
                   frames=[fr(3, jpg(tmp_path, "red"), "ERROR 404"), fr(20, jpg(tmp_path, "blue"), None)],
                   notes=[{"kind": "answer", "t_start": None, "t_end": None, "text": "a note"}],
                   transcript_source="captions")
    res = record_run(conn, run)
    store.write_run(conn, root, run, res)
    before = {t: texts(conn, t) for t in ("segments", "ocr_blocks", "notes")}
    times = frame_times(conn)
    index.forget(conn, KEY)
    assert texts(conn, "segments") == []
    out = backfill.rebuild(conn, root, KEY)
    assert not out.skipped
    assert {t: texts(conn, t) for t in before} == before
    assert frame_times(conn) == times
    sp = conn.execute("SELECT store_path FROM frames WHERE video_key=? ORDER BY t", (KEY,)).fetchall()
    assert all(r[0] and (root / r[0]).exists() for r in sp)
    assert conn.execute("SELECT transcript_source FROM videos WHERE video_key=?", (KEY,)).fetchone()[0] == "captions"
    # a second video rebuilt at the same evidence_version must not collide on origin_ref
    run2 = fake_run(video_key="file:0123456789abcdef", source="/x.mp4", segments=[seg(0, 5, "otro")])
    store.write_run(conn, root, run2, record_run(conn, run2))
    index.forget(conn, run2.video_key)
    assert not backfill.rebuild(conn, root, run2.video_key).skipped
