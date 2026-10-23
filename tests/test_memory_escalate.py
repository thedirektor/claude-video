"""Model-free escalation: dense re-sample + zoom re-OCR (0.5.1 Task 5)."""
from __future__ import annotations

from pathlib import Path

from memory import store
from memory.escalate import escalate
from memory.index import record_run
from memory_fakes import KEY, fake_run, fr, seg


def _video(conn, tmp_path, media=True, ocr_at=None):
    frames = [fr(ocr_at, None, "TEXTO YA CONOCIDO EN PANTALLA")] if ocr_at is not None else []
    record_run(conn, fake_run(duration=60.0, segments=[seg(0, 5, "hola")], frames=frames))
    sd = tmp_path / "store"
    if media:
        vdir = store.video_dir(sd, KEY)
        vdir.mkdir(parents=True)
        (vdir / "media.mp4").write_bytes(b"fake")
    return sd


def _fakes(texts_by_t):
    calls = []

    def extract(media, out, t):
        calls.append(t)
        Path(out).write_bytes(b"jpg")
        Path(str(out) + ".t").write_text(str(t))
        return True

    def ocr(paths):
        return {p: texts_by_t.get(float(Path(p + ".t").read_text()), "") for p in paths}

    return extract, ocr, calls


def _ev(conn):
    return conn.execute("SELECT evidence_version FROM videos WHERE video_key=?", (KEY,)).fetchone()[0]


def test_no_media_returns_zero_without_extract(conn, tmp_path):
    sd = _video(conn, tmp_path, media=False)
    extract, ocr, calls = _fakes({})
    assert escalate(conn, KEY, [10.0], store_dir=sd, extract=extract, ocr=ocr) == 0 and calls == []


def test_adds_new_ocr_and_bumps_evidence(conn, tmp_path):
    sd = _video(conn, tmp_path)
    ev = _ev(conn)
    extract, ocr, calls = _fakes({12.0: "PRECIO 42 EUROS OFERTA"})
    assert escalate(conn, KEY, [10.0], window=3, store_dir=sd, extract=extract, ocr=ocr) == 1
    assert calls == [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]
    row = conn.execute("SELECT o.t, o.text, f.kind FROM ocr_blocks o JOIN frames f ON f.id = o.frame_id "
                       "WHERE o.text LIKE 'PRECIO%'").fetchone()
    assert tuple(row) == (12.0, "PRECIO 42 EUROS OFERTA", "escalated")
    assert conn.execute("SELECT count(*) FROM fts WHERE fts MATCH 'precio'").fetchone()[0] == 1
    assert _ev(conn) == ev + 1


def test_duplicate_ocr_not_added(conn, tmp_path):
    sd = _video(conn, tmp_path, ocr_at=11.0)
    ev = _ev(conn)
    extract, ocr, _ = _fakes({12.0: "texto ya conocido en pantalla", 30.0: "corto"})
    assert escalate(conn, KEY, [10.0], window=3, store_dir=sd, extract=extract, ocr=ocr) == 0
    assert _ev(conn) == ev


def test_time_attempted_once(conn, tmp_path):
    sd = _video(conn, tmp_path)
    extract, ocr, calls = _fakes({})
    escalate(conn, KEY, [10.0], window=2, store_dir=sd, extract=extract, ocr=ocr)
    n = len(calls)
    assert escalate(conn, KEY, [11.0], window=2, store_dir=sd, extract=extract, ocr=ocr) == 0
    assert len(calls) == n
    assert conn.execute("SELECT count(*) FROM escalations").fetchone()[0] == 1


def test_budget_stops_early(conn, tmp_path):
    sd = _video(conn, tmp_path)
    extract, ocr, calls = _fakes({})
    ticks = iter([0.0, 0.0, 1.0, 100.0] + [100.0] * 50)
    escalate(conn, KEY, [10.0, 40.0], window=5, budget_s=45, store_dir=sd, extract=extract, ocr=ocr,
             clock=lambda: next(ticks))
    assert 0 < len(calls) < 11


def test_clamps_to_duration_and_zero(conn, tmp_path):
    sd = _video(conn, tmp_path)
    extract, ocr, calls = _fakes({})
    escalate(conn, KEY, [1.0, 59.0], window=2, store_dir=sd, extract=extract, ocr=ocr)
    assert min(calls) == 0.0 and max(calls) == 60.0


def test_unknown_duration_probes_media(conn, tmp_path):
    record_run(conn, fake_run(segments=[seg(0, 5, "hola")]))  # duration None (local files today)
    sd = tmp_path / "store"
    vdir = store.video_dir(sd, KEY)
    vdir.mkdir(parents=True)
    (vdir / "media.mp4").write_bytes(b"fake")
    extract, ocr, calls = _fakes({})
    escalate(conn, KEY, [9.0], window=3, store_dir=sd, extract=extract, ocr=ocr, probe=lambda media: 10.0)
    assert max(calls) == 10.0
