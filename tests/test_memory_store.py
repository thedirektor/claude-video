from __future__ import annotations

import json
import os

import pytest

from memory import index, store
from memory_fakes import KEY, fake_run, fr, jpg, seg

MB = 1024 * 1024


def _record(conn, tmp_path, key=KEY, frames=None, **kw):
    rep = tmp_path / f"report-{abs(hash(key))}.md"
    rep.write_text("# report\n")
    run = fake_run(video_key=key, title="T", report_path=str(rep),
                   transcript_source="captions (en)", segments=[seg(0, 2, "hello world")],
                   frames=frames or [], **kw)
    return run, index.record_run(conn, run)


def test_write_run_layout(conn, tmp_path):
    f = jpg(tmp_path, "red")
    run, res = _record(conn, tmp_path, frames=[fr(5.0, f, "slide text")])
    root = tmp_path / "st"
    vdir = store.write_run(conn, root, run, res)
    assert vdir.name.startswith("yt_AAAAAAAAAAA")
    assert (vdir / "report.md").read_text() == "# report\n"
    assert (vdir / "frames" / "5000.jpg").exists()
    assert f.exists()  # copy, not move
    rel = conn.execute("SELECT store_path FROM frames").fetchone()[0]
    assert rel == f"{vdir.name}/frames/5000.jpg" and (root / rel).exists()
    tr = json.loads((vdir / "transcript.json").read_text())
    assert tr == [{"start": 0.0, "end": 2.0, "speaker": None, "text": "hello world", "source": "captions (en)"}]
    assert json.loads((vdir / "ocr.json").read_text())[0]["text"] == "slide text"
    meta = json.loads((vdir / "meta.json").read_text())
    assert meta["evidence_version"] == 1 and meta["video_key"] == KEY
    assert conn.execute("SELECT store_path FROM videos").fetchone()[0] == vdir.name


def test_frame_name_clash(conn, tmp_path):
    a, b = jpg(tmp_path, "red"), jpg(tmp_path, "blue")
    run, res = _record(conn, tmp_path)
    ids = [conn.execute("INSERT INTO frames (video_key, t, kind) VALUES (?, 5.0, 'scene')", (KEY,)).lastrowid
           for _ in range(2)]
    res.new_frames = [(ids[0], str(a)), (ids[1], str(b))]
    vdir = store.write_run(conn, tmp_path / "st", run, res)
    assert sorted(p.name for p in (vdir / "frames").iterdir()) == ["5000.jpg", "5000_1.jpg"]


def test_keep_media_rules(monkeypatch):
    monkeypatch.delenv("WATCH_STORE_KEEP_MEDIA", raising=False)
    monkeypatch.delenv("WATCH_STORE_MEDIA_MAX_MB", raising=False)
    assert store.keep_media(299 * MB) is True
    assert store.keep_media(301 * MB) is False
    monkeypatch.setenv("WATCH_STORE_KEEP_MEDIA", "1")
    assert store.keep_media(301 * MB) is True


def test_media_copied_when_small(conn, tmp_path):
    m = tmp_path / "v.MP4"
    m.write_bytes(b"x" * 10)
    run, res = _record(conn, tmp_path, media_path=str(m))
    vdir = store.write_run(conn, tmp_path / "st", run, res)
    assert (vdir / "media.mp4").read_bytes() == b"x" * 10 and m.exists()


def _three(conn, tmp_path):
    root = tmp_path / "st"
    keys = ["yt:AAAAAAAAAA1", "yt:AAAAAAAAAA2", "yt:AAAAAAAAAA3"]
    for i, k in enumerate(keys):
        _record(conn, tmp_path, key=k)
        vdir = store.video_dir(root, k)
        (vdir / "frames").mkdir(parents=True)
        (vdir / "media.mp4").write_bytes(b"m" * MB)
        (vdir / "frames" / "1000.jpg").write_bytes(b"f" * MB)
        conn.execute("INSERT INTO frames (video_key, t, kind, store_path) VALUES (?, 1.0, 'scene', ?)",
                     (k, f"{vdir.name}/frames/1000.jpg"))
        conn.execute("UPDATE videos SET last_access = ? WHERE video_key = ?", (f"2026-01-0{i + 1}", k))
    conn.commit()
    return root, keys


def test_retention_media_then_frames_lru(conn, tmp_path):
    root, keys = _three(conn, tmp_path)
    r = store.enforce_retention(conn, root, max_bytes=int(4.5 * MB))
    assert r["media_removed"] == 2 and r["frames_removed"] == 0 and r["freed"] == 2 * MB
    assert not list(store.video_dir(root, keys[0]).glob("media.*"))
    assert not list(store.video_dir(root, keys[1]).glob("media.*"))
    assert list(store.video_dir(root, keys[2]).glob("media.*"))

    root, keys = tmp_path / "st", keys
    r = store.enforce_retention(conn, root, max_bytes=int(1.5 * MB), protect=frozenset({keys[2]}))
    assert r["frames_removed"] == 2
    assert (store.video_dir(root, keys[2]) / "media.mp4").exists()
    assert (store.video_dir(root, keys[2]) / "frames" / "1000.jpg").exists()
    nulls = dict(conn.execute("SELECT video_key, store_path IS NULL FROM frames"))
    assert nulls == {keys[0]: 1, keys[1]: 1, keys[2]: 0}
    assert conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 3  # rows never evicted


def test_retention_noop_under_cap(conn, tmp_path):
    root, _ = _three(conn, tmp_path)
    assert store.enforce_retention(conn, root, max_bytes=100 * MB) == {
        "freed": 0, "media_removed": 0, "frames_removed": 0}


def test_qdrant_inbox_atomic_and_front_matter(tmp_path):
    inbox = tmp_path / "inbox"
    p = store.write_qdrant_inbox(inbox, "yt:AAAAAAAAAAA", "My: Great Video", "https://y/x", "2026-10-23", "body")
    assert p.name.endswith("-yt:AAAAAAAAAAA.md") and p.name.startswith("my-great-video-")
    assert [f.name for f in inbox.iterdir()] == [p.name]
    text = p.read_text()
    assert text.startswith("---\nvideo_key: yt:AAAAAAAAAAA\n") and text.endswith("body")
    p2 = store.write_qdrant_inbox(inbox, "yt:AAAAAAAAAAA", None, None, "2026-10-23", "b2")
    assert p2.name.startswith("yt-aaaaaaaaaaa-") and "title: null" in p2.read_text()


def test_rewatch_keeps_pinned_media(conn, tmp_path):
    """A breakdown pins the media its timestamps were measured on; a re-watch that downloaded a
    smaller copy (<=720p) must not replace it."""
    run, res = _record(conn, tmp_path)
    root = tmp_path / "st"
    vdir = store.video_dir(root, KEY)
    vdir.mkdir(parents=True)
    (vdir / "media.mp4").write_bytes(b"1080p-original")
    (vdir / store.PINNED_MARKER).touch()
    small = tmp_path / "small.mp4"
    small.write_bytes(b"720p")
    run.media_path = str(small)
    store.write_run(conn, root, run, res)
    assert (vdir / "media.mp4").read_bytes() == b"1080p-original"
    # unpinned media is still refreshed as before
    (vdir / store.PINNED_MARKER).unlink()
    store.write_run(conn, root, run, res)
    assert (vdir / "media.mp4").read_bytes() == b"720p"
