from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import identity
from breakdown import media
from breakdown_fixtures import make_motion_clip, make_reference
from memory import store

KEY_A = "yt:AAAAAAAAAAA"
KEY_B = "yt:BBBBBBBBBBB"


def _video_row(conn, key, url="https://example.com/v", duration=7.0, access="2026-01-01"):
    conn.execute(
        "INSERT INTO videos (video_key, url, source, duration, last_access) VALUES (?, ?, ?, ?, ?)",
        (key, url, url, duration, access),
    )
    conn.commit()


def test_pin_survives_retention(conn, tmp_path):
    root = tmp_path / "st"
    _video_row(conn, KEY_A, access="2026-01-01")  # oldest: would be evicted first
    _video_row(conn, KEY_B, access="2026-01-02")
    for key in (KEY_A, KEY_B):
        vdir = store.video_dir(root, key)
        vdir.mkdir(parents=True)
        (vdir / "media.mp4").write_bytes(b"x" * 1000)
    media.pin(root, KEY_A)
    assert media.is_pinned(root, KEY_A) and not media.is_pinned(root, KEY_B)
    out = store.enforce_retention(conn, root, max_bytes=1)
    assert (store.video_dir(root, KEY_A) / "media.mp4").exists()
    assert not (store.video_dir(root, KEY_B) / "media.mp4").exists()
    assert out["media_removed"] == 1


def test_pinned_media_over_cap_is_evictable(conn, tmp_path, monkeypatch):
    root = tmp_path / "st"
    _video_row(conn, KEY_A)
    vdir = store.video_dir(root, KEY_A)
    vdir.mkdir(parents=True)
    (vdir / "media.mp4").write_bytes(b"x" * (2 * 1024 * 1024))
    media.pin(root, KEY_A)
    monkeypatch.setenv("WATCH_BREAKDOWN_MEDIA_MAX_MB", "1")
    store.enforce_retention(conn, root, max_bytes=1)
    assert not (vdir / "media.mp4").exists()


def test_ensure_media_stored(conn, tmp_path):
    root = tmp_path / "st"
    _video_row(conn, KEY_A)
    vdir = store.video_dir(root, KEY_A)
    vdir.mkdir(parents=True)
    make_reference(vdir / "media.mp4", fps=25)
    info = media.ensure_media(conn, KEY_A, store_dir=root, fetch=lambda *a, **k: pytest.fail("no fetch"))
    assert info["source"] == "stored"
    assert info["path"] == str(vdir / "media.mp4")
    assert info["fps"] == 25.0 and info["fps_rational"] == "25/1"
    assert abs(info["duration"] - 7.0) < 0.1
    assert (info["width"], info["height"]) == (640, 360)
    assert info["has_audio"] is True
    assert info["low_resolution"] is False


def test_ensure_media_fetched(conn, tmp_path):
    root = tmp_path / "st"
    _video_row(conn, KEY_A, url="https://example.com/clip")
    src = tmp_path / "dl" / "video.mp4"
    src.parent.mkdir()
    make_reference(src, fps=25, with_audio=False)
    seen = {}

    def fetch(url, vdir):
        seen["url"] = url
        return src

    info = media.ensure_media(conn, KEY_A, store_dir=root, fetch=fetch)
    assert seen["url"] == "https://example.com/clip"
    assert info["source"] == "fetched" and info["has_audio"] is False
    assert Path(info["path"]).parent == store.video_dir(root, KEY_A)
    assert Path(info["path"]).name.startswith("media.")


def test_refetch_or_flag(conn, tmp_path):
    root = tmp_path / "st"
    _video_row(conn, KEY_A, duration=42.0)

    def fetch(url, vdir):
        raise SystemExit("yt-dlp failed")

    info = media.ensure_media(conn, KEY_A, store_dir=root, fetch=fetch)
    assert info["source"] == "frames"
    assert info["low_resolution"] is True
    assert info["path"] is None
    assert info["duration"] == 42.0


def test_ensure_media_unknown_video(conn, tmp_path):
    info = media.ensure_media(conn, "yt:ZZZZZZZZZZZ", store_dir=tmp_path / "st", fetch=lambda *a: None)
    assert info["source"] == "frames" and info["path"] is None


def _probe(path, *entries):
    out = subprocess.run(
        ["ffprobe", "-v", "error", *entries, "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True).stdout.split()
    return out


def test_fixture_ground_truth(tmp_path):
    p = tmp_path / "ref.mp4"
    gt = make_reference(p)
    assert p.exists()
    assert abs(float(_probe(p, "-show_entries", "format=duration")[0]) - 7.0) <= 0.05
    assert gt["duration"] == 7.0 and gt["shots"] == [0, 2.0, 4.0, 5.0]
    assert gt["title"]["text"] == "SALE 50%" and (gt["title"]["start"], gt["title"]["end"]) == (1.0, 3.0)
    assert gt["bpm"] == 120 and gt["beats"][:3] == [0.0, 0.5, 1.0]
    assert gt["dissolve"] == {"start": 4.0, "end": 5.0, "center": 4.5}
    assert gt["fade_out"] == {"start": 6.0, "end": 7.0}
    assert _probe(p, "-select_streams", "a", "-show_entries", "stream=codec_type") == ["audio"]


def test_fixture_no_audio(tmp_path):
    p = tmp_path / "silent.mp4"
    make_reference(p, with_audio=False)
    assert _probe(p, "-select_streams", "a", "-show_entries", "stream=codec_type") == []


def test_motion_clip(tmp_path):
    p = tmp_path / "m.mp4"
    gt = make_motion_clip(p, fps=60)
    assert gt == {"start": 0.2, "end": 0.7, "trans": "CUBIC", "ease": "OUT"}
    assert _probe(p, "-select_streams", "v", "-show_entries", "stream=r_frame_rate") == ["60/1"]
