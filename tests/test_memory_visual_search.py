"""Visual search: Immich CLIP over watch albums, mapped back to (video_key, t) (0.6.0 phase E)."""
from __future__ import annotations

from memory import visual
from memory.index import record_run
from memory_fakes import KEY, fake_run, fr, seg


def _linked(conn):
    record_run(conn, fake_run(title="Talk", duration=60.0, segments=[seg(0, 4, "hola")],
                              frames=[fr(5, None, None), fr(30, None, None)],
                              library={"immich": {"album_id": "alb-1", "frames": {"5000": "as-5", "30000": "as-30"}}}))
    conn.execute("UPDATE videos SET immich_album_id = 'alb-1' WHERE video_key = ?", (KEY,))
    conn.commit()


def test_maps_assets_to_video_and_time(conn, monkeypatch):
    monkeypatch.setenv("IMMICH_BASE_URL", "https://photos.example.com")
    _linked(conn)
    calls = []

    def smart(query, album_ids, size):
        calls.append((query, album_ids, size))
        return ["as-30", "unrelated", "as-5"]
    r = visual.visual_search(conn, "man on stage", k=5, smart=smart)
    assert calls == [("man on stage", ["alb-1"], 15)]
    assert [(h["video_key"], h["timestamp"], h["kind"]) for h in r["hits"]] == [(KEY, "00:30", "visual"), (KEY, "00:05", "visual")]
    assert r["hits"][0]["score"] > r["hits"][1]["score"]
    assert r["hits"][0]["immich_url"] == "https://photos.example.com/photos/as-30" and r["mode"] == "visual"


def test_scoped_to_one_video(conn):
    _linked(conn)
    seen = {}
    visual.visual_search(conn, "x", video_key=KEY, smart=lambda q, a, s: seen.setdefault("albums", a) and [])
    assert seen["albums"] == ["alb-1"]


def test_no_linked_videos_explains(conn):
    record_run(conn, fake_run(segments=[seg(0, 1, "a")]))
    r = visual.visual_search(conn, "x", smart=lambda *a: pytest_fail())
    assert r["hits"] == [] and "Immich" in r["note"]


def test_immich_error_is_a_note(conn):
    _linked(conn)

    def boom(*a):
        raise OSError("immich down")
    r = visual.visual_search(conn, "x", smart=boom)
    assert r["hits"] == [] and "immich down" in r["note"]


def pytest_fail():
    raise AssertionError("must not call Immich")
