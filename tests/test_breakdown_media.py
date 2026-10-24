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


def test_fetch_via_fallback_proxy_is_low_resolution(conn, tmp_path):
    root = tmp_path / "st"
    _video_row(conn, KEY_A, url="https://example.com/clip")
    src = tmp_path / "dl" / "video.mp4"
    src.parent.mkdir()
    make_reference(src, with_audio=False)
    info = media.ensure_media(conn, KEY_A, store_dir=root,
                              fetch=lambda u, v: {"video_path": str(src), "via_fallback_proxy": True})
    assert info["source"] == "fetched" and info["low_resolution"] is True


def test_pinned_media_does_not_starve_other_files(conn, tmp_path):
    root = tmp_path / "st"
    _video_row(conn, KEY_A, access="2026-01-01")
    _video_row(conn, KEY_B, access="2026-01-02")
    va, vb = store.video_dir(root, KEY_A), store.video_dir(root, KEY_B)
    for v in (va, vb):
        (v / "frames").mkdir(parents=True)
        (v / "frames" / "1000.jpg").write_bytes(b"f" * 10)
    (va / "media.mp4").write_bytes(b"x" * 100_000)
    media.pin(root, KEY_A)
    (vb / "media.mp4").write_bytes(b"y" * 100)
    store.enforce_retention(conn, root, max_bytes=50_000)
    assert (va / "media.mp4").exists() and (va / "frames" / "1000.jpg").exists()
    assert (vb / "media.mp4").exists() and (vb / "frames" / "1000.jpg").exists()


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


def _title_pixels(path, t):
    import cv2
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    assert ok
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # white text on a dark box vs flat colour background
    return int((gray[10:80, 10:260] > 235).sum())


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
    assert gt["title"]["rendered"] is True
    assert _title_pixels(p, 2.0) > 300
    assert _title_pixels(p, 0.5) < 50 and _title_pixels(p, 3.5) < 50


def test_fixture_no_audio(tmp_path):
    p = tmp_path / "silent.mp4"
    make_reference(p, with_audio=False)
    assert _probe(p, "-select_streams", "a", "-show_entries", "stream=codec_type") == []


def test_motion_clip(tmp_path):
    p = tmp_path / "m.mp4"
    gt = make_motion_clip(p, fps=60)
    assert gt == {"start": 0.2, "end": 0.7, "trans": "CUBIC", "ease": "OUT"}
    assert _probe(p, "-select_streams", "v", "-show_entries", "stream=r_frame_rate") == ["60/1"]


def test_av1_media_transcoded_for_opencv(conn, tmp_path):
    """Real-media e2e regression: YouTube served AV1, which OpenCV's bundled ffmpeg cannot decode,
    so every cv2 field came out not determined. ensure_media now transcodes it to H.264 once."""
    cv2 = pytest.importorskip("cv2")
    root = tmp_path / "st"
    _video_row(conn, KEY_A)
    vdir = store.video_dir(root, KEY_A)
    vdir.mkdir(parents=True)
    src = tmp_path / "src.mp4"
    make_reference(src, fps=25)
    av1 = vdir / "media.mkv"
    res = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
                          "-c:v", "libsvtav1", "-preset", "12", "-c:a", "copy", str(av1)], capture_output=True)
    if res.returncode:
        pytest.skip("no AV1 encoder in this ffmpeg")
    cap = cv2.VideoCapture(str(av1))
    if cap.isOpened() and cap.read()[0]:
        pytest.skip("this OpenCV decodes AV1; nothing to transcode")
    info = media.ensure_media(conn, KEY_A, store_dir=root, fetch=lambda *a, **k: pytest.fail("no fetch"))
    assert info["path"] == str(vdir / "media.mp4") and not av1.exists()
    assert info["codec"] == "h264" and "transcoded to H.264" in info["media_note"]
    assert abs(info["duration"] - 7.0) < 0.1 and info["has_audio"] is True
    cap = cv2.VideoCapture(info["path"])
    assert cap.read()[0]
    assert sorted(p.name for p in vdir.glob("media.*")) == ["media.mp4"]
    # second call: already decodable, nothing redone
    again = media.ensure_media(conn, KEY_A, store_dir=root, fetch=lambda *a, **k: pytest.fail("no fetch"))
    assert again["path"] == info["path"] and "media_note" not in again


def test_fetch_prefers_h264():
    assert media.FETCH_FORMAT.startswith("bv*[vcodec^=avc1]")
