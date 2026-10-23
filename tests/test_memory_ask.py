"""Retrieval read API: resolve / ask / moment / list / hit_dicts (Task 7)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from memory import store
from memory.index import record_run, set_library_ref
from memory.retrieval import (
    AmbiguousVideo, VideoNotFound, ask, hit_dicts, list_videos, moment, resolve_video, search,
    stamp, stamp_diarized,
)
from memory_fakes import KEY, fake_run, fr, jpg, seg, stored_times
from transcribe import format_transcript

OTHER = "file:0123456789abcdef"


@pytest.fixture
def two(conn, tmp_path):
    record_run(conn, fake_run(title="La Canción demo", segments=[
        seg(0, 5, "hola mundo"), seg(12, 18, "la canción del verano"), seg(40, 45, "error en pantalla")],
        frames=[fr(13, jpg(tmp_path, "red"), "ERROR 404")]))
    record_run(conn, fake_run(video_key=OTHER, source="/tmp/x.mp4", title="Otro demo",
                              segments=[seg(0, 5, "otra cosa")]))


def test_stamp_matches_format_transcript():
    for s in (0, 59.9, 75.9, 3725.0):
        assert format_transcript([{"start": s, "end": s + 1, "text": "x"}]).startswith(f"[{stamp(s)}]")
    assert stamp(3725.0) == "62:05"
    d = format_transcript([{"start": 3723, "end": 3729, "text": "x", "speaker": "A"}])
    assert d.startswith(f"[A] ({stamp_diarized(3723)}-") and stamp_diarized(3723) == "62:03"


def test_ask_timestamps_are_stored_rows(conn, two):
    pack = ask(conn, KEY, "canción error")
    stored = stored_times(conn, KEY) | {r[0] for r in conn.execute("SELECT t_start FROM chunks WHERE video_key=?", (KEY,))}
    assert pack["evidence"] and all(e["t"] is None or e["t"] in stored for e in pack["evidence"])
    assert [e["id"] for e in pack["evidence"]] == [f"E{i}" for i in range(1, len(pack["evidence"]) + 1)]
    assert pack["rule"].startswith("Cite only timestamps")
    json.dumps(pack)
    assert pack["video"]["video_key"] == KEY and pack["video"]["evidence_version"] >= 1
    assert any(e["frame"] for e in pack["evidence"])
    assert any("[00:12] la canción" in e["context"] for e in pack["evidence"])


def test_ask_diarized_timestamp(conn):
    record_run(conn, fake_run(segments=[seg(3723, 3729, "texto raro aqui", speaker="A")]))
    e = ask(conn, KEY, "raro")["evidence"][0]
    assert e["t"] == 3723.0 and e["timestamp"] == "62:03"


def test_resolve_by_url_spelling_and_title(conn, two):
    assert resolve_video(conn, "https://youtu.be/AAAAAAAAAAA")["video_key"] == KEY
    assert resolve_video(conn, KEY)["video_key"] == KEY
    assert resolve_video(conn, "cancion")["video_key"] == KEY
    with pytest.raises(VideoNotFound):
        resolve_video(conn, "nothing-like-this")


def test_resolve_ambiguous(conn, two):
    with pytest.raises(AmbiguousVideo) as e:
        resolve_video(conn, "demo")
    assert {c["video_key"] for c in e.value.candidates} == {KEY, OTHER}


def test_moment_window_and_frame_fallback(conn, tmp_path, monkeypatch):
    monkeypatch.setenv("IMMICH_BASE_URL", "https://i")
    f1, f2 = jpg(tmp_path, "red"), jpg(tmp_path, "blue")
    run = fake_run(segments=[seg(0, 5, "a"), seg(20, 25, "b"), seg(40, 45, "c")],
                   frames=[fr(18, f1, "uno"), fr(22, f2, "dos")])
    res = record_run(conn, run)
    root = tmp_path / "store"
    store.write_run(conn, root, run, res)
    conn.execute("UPDATE frames SET immich_asset_id='a2' WHERE t=22")
    conn.commit()
    (root / conn.execute("SELECT store_path FROM frames WHERE t=22").fetchone()[0]).unlink()
    m = moment(conn, KEY, "0:20", window=5, frames=4)
    assert [s["start"] for s in m["segments"]] == [20.0]
    assert [(f["t"], f["path"] is None, f["immich_url"]) for f in m["frames"]] == [
        (18.0, False, None), (22.0, True, "https://i/photos/a2")]
    assert {o["t"] for o in m["ocr"]} == {18.0, 22.0}
    far = moment(conn, KEY, 100, window=2, frames=1)
    assert len(far["frames"]) == 1 and far["frames"][0]["t"] == 22.0 and far["segments"] == []


def test_list_videos_query(conn, two):
    assert [v["video_key"] for v in list_videos(conn, "cancion")] == [KEY]
    assert len(list_videos(conn, limit=1)) == 1
    v = list_videos(conn)[0]
    assert set(v["library"]) == {"immich_album_id", "nextcloud_path", "paperless_doc_id"}
    set_library_ref(conn, KEY, immich_album_id="alb")
    assert [x for x in list_videos(conn, "cancion")][0]["library"]["immich_album_id"] == "alb"


def test_hit_dicts_shape(conn, two):
    d = hit_dicts(conn, search(conn, "cancion"), "fts")
    assert d["mode"] == "fts" and d["hits"][0]["title"] == "La Canción demo" and d["hits"][0]["timestamp"] == "00:12"
    json.dumps(d)
    assert hit_dicts(conn, [], "fts-only") == {"mode": "fts", "hits": []}
    assert len(hit_dicts(conn, search(conn, "hola"), "hybrid")["hits"][0]["snippet"]) <= 240
