"""library.py: Immich+Nextcloud storage router — router, frame selection,
both uploaders, best-effort behavior. All HTTP is mocked via library.urlopen;
never touches the real network."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import library  # noqa: E402


class _Resp:
    """Minimal urlopen()-context-manager stand-in."""

    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8") if not isinstance(payload, bytes) else payload

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# --------------------------------------------------------------------------
# is_immich_media
# --------------------------------------------------------------------------

def test_is_immich_media():
    for ext in (".mp4", ".jpg", ".png", ".webp"):
        assert library.is_immich_media(f"clip{ext}") is True
    for ext in (".md", ".txt", ".json", ".vtt"):
        assert library.is_immich_media(f"report{ext}") is False


# --------------------------------------------------------------------------
# select_representative_frames
# --------------------------------------------------------------------------

def test_select_frames_keeps_ocr_and_caps():
    frames = []
    ocr_text = {}
    for i in range(40):
        path = f"frame_{i:03d}.jpg"
        frames.append({"path": path, "timestamp_seconds": float(i), "reason": "keyframe"})
    # 5 OCR-significant frames, spread across the timeline.
    ocr_paths = ["frame_003.jpg", "frame_011.jpg", "frame_019.jpg", "frame_027.jpg", "frame_035.jpg"]
    for p in ocr_paths:
        ocr_text[p] = "some on-screen text"

    result = library.select_representative_frames(frames, ocr_text, cap=10)

    assert len(result) == 10
    result_paths = {f["path"] for f in result}
    assert set(ocr_paths).issubset(result_paths)
    # timestamp-ordered
    timestamps = [f["timestamp_seconds"] for f in result]
    assert timestamps == sorted(timestamps)


def test_select_frames_cap_zero_returns_all():
    frames = [
        {"path": "b.jpg", "timestamp_seconds": 2.0, "reason": "keyframe"},
        {"path": "a.jpg", "timestamp_seconds": 1.0, "reason": "keyframe"},
        {"path": "c.jpg", "timestamp_seconds": 3.0, "reason": "keyframe"},
    ]
    result = library.select_representative_frames(frames, {}, cap=0)
    assert len(result) == 3
    assert [f["path"] for f in result] == ["a.jpg", "b.jpg", "c.jpg"]


# --------------------------------------------------------------------------
# save_artifacts router
# --------------------------------------------------------------------------

def test_router_splits_no_dup(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video")
    frame_paths = []
    for i in range(3):
        f = tmp_path / f"frame_{i}.jpg"
        f.write_bytes(b"fake-frame")
        frame_paths.append(f)
    report = tmp_path / "report.md"
    report.write_text("# report", encoding="utf-8")

    frames = [
        {"path": str(frame_paths[i]), "timestamp_seconds": float(i), "reason": "scene-change"}
        for i in range(3)
    ]

    immich_calls = []
    nextcloud_calls = []

    def fake_immich(files, album_name, cfg):
        immich_calls.append(list(files))
        return {"album": album_name, "uploaded": len(files), "url": None, "error": None}

    def fake_nextcloud(files, folder, cfg):
        nextcloud_calls.append(list(files))
        return {"folder": folder, "uploaded": len(files), "url": None, "error": None}

    monkeypatch.setattr(library, "upload_to_immich", fake_immich)
    monkeypatch.setattr(library, "upload_to_nextcloud", fake_nextcloud)

    cfg = {
        "IMMICH_BASE_URL": "https://immich.example",
        "IMMICH_API_KEY": "key",
        "NEXTCLOUD_URL": "https://nc.example",
        "NEXTCLOUD_USER": "user",
        "NEXTCLOUD_PASS": "pass",
        "WATCH_SAVE_FRAME_CAP": 20,
    }

    summary = library.save_artifacts(
        str(video), frames, str(report), {}, "My Video", "abc123", cfg=cfg
    )

    assert len(immich_calls) == 1
    assert len(nextcloud_calls) == 1

    immich_set = set(immich_calls[0])
    nextcloud_set = set(nextcloud_calls[0])

    assert str(video) in immich_set
    for fp in frame_paths:
        assert str(fp) in immich_set
    assert str(report) in nextcloud_set

    # each file exactly once, no overlap
    assert immich_set.isdisjoint(nextcloud_set)
    assert len(immich_calls[0]) == len(immich_set) == 4
    assert len(nextcloud_calls[0]) == len(nextcloud_set) == 1
    assert any("Immich" in line for line in summary)
    assert any("Nextcloud" in line for line in summary)


def test_router_transcript_only(tmp_path, monkeypatch):
    report = tmp_path / "report.md"
    report.write_text("# report", encoding="utf-8")

    immich_calls = []
    nextcloud_calls = []
    monkeypatch.setattr(
        library, "upload_to_immich",
        lambda files, album_name, cfg: immich_calls.append(files) or {"uploaded": 0, "error": None},
    )
    monkeypatch.setattr(
        library, "upload_to_nextcloud",
        lambda files, folder, cfg: nextcloud_calls.append(files) or {"uploaded": len(files), "error": None},
    )

    cfg = {"IMMICH_API_KEY": "key", "NEXTCLOUD_PASS": "pass", "WATCH_SAVE_FRAME_CAP": 20}
    library.save_artifacts(None, [], str(report), {}, "Transcript Only", "vid2", cfg=cfg)

    assert len(immich_calls) == 0
    assert len(nextcloud_calls) == 1
    assert nextcloud_calls[0] == [str(report)]


# --------------------------------------------------------------------------
# Immich upload flow
# --------------------------------------------------------------------------

def test_immich_upload_builds_asset_and_album(tmp_path, monkeypatch):
    frame = tmp_path / "frame_0.jpg"
    frame.write_bytes(b"fake-frame-bytes")

    calls = []

    def fake_urlopen(request, timeout=None, context=None):
        calls.append(request)
        method = request.get_method()
        url = request.full_url
        if url.endswith("/api/assets"):
            return _Resp({"id": "asset-1", "status": "created"})
        if url.endswith("/api/albums") and method == "GET":
            return _Resp([])
        if url.endswith("/api/albums") and method == "POST":
            return _Resp({"id": "album-1"})
        if url.endswith("/api/albums/album-1/assets") and method == "PUT":
            return _Resp({})
        raise AssertionError(f"unexpected request {method} {url}")

    monkeypatch.setattr(library, "urlopen", fake_urlopen)

    cfg = {"IMMICH_BASE_URL": "https://immich.example", "IMMICH_API_KEY": "sk_key"}
    result = library.upload_to_immich([str(frame)], "My Video (abc123)", cfg)

    assert result["error"] is None
    assert result["uploaded"] == 1

    asset_calls = [c for c in calls if c.full_url.endswith("/api/assets")]
    assert len(asset_calls) == 1
    assert asset_calls[0].get_header("X-api-key") == "sk_key"
    body = asset_calls[0].data
    assert b'name="assetData"' in body
    assert b'name="deviceAssetId"' in body
    assert b"deviceId" in body and b"watch-skill" in body

    album_post = [c for c in calls if c.full_url.endswith("/api/albums") and c.get_method() == "POST"]
    assert len(album_post) == 1
    assert json.loads(album_post[0].data)["albumName"] == "My Video (abc123)"

    put_calls = [c for c in calls if c.get_method() == "PUT"]
    assert len(put_calls) == 1
    assert json.loads(put_calls[0].data) == {"ids": ["asset-1"]}

    # deviceAssetId is stable across a repeat call (dedup on re-run). The
    # multipart boundary is randomized per call, so compare the field value,
    # not the raw body.
    def _device_asset_id(raw: bytes) -> bytes:
        marker = b'name="deviceAssetId"'
        idx = raw.index(marker)
        # value is on the line after two CRLFs following the header line
        rest = raw[idx:].split(b"\r\n\r\n", 1)[1]
        return rest.split(b"\r\n", 1)[0]

    calls.clear()
    library.upload_to_immich([str(frame)], "My Video (abc123)", cfg)
    body2 = [c for c in calls if c.full_url.endswith("/api/assets")][0].data
    assert _device_asset_id(body) == _device_asset_id(body2)


# --------------------------------------------------------------------------
# Nextcloud upload flow
# --------------------------------------------------------------------------

def test_nextcloud_mkcol_put(tmp_path, monkeypatch):
    report = tmp_path / "report.md"
    report.write_text("# hello", encoding="utf-8")

    calls = []

    def fake_urlopen(request, timeout=None, context=None):
        calls.append(request)
        method = request.get_method()
        url = request.full_url
        if method == "MKCOL":
            if url.endswith("/Watch/"):
                return _Resp(b"")  # 200 default via getattr(response, "status", 200)
            # existing folder: simulate 405 via HTTPError
            import urllib.error
            raise urllib.error.HTTPError(url, 405, "exists", None, None)
        if method == "PUT":
            return _Resp(b"")
        raise AssertionError(f"unexpected request {method} {url}")

    monkeypatch.setattr(library, "urlopen", fake_urlopen)

    cfg = {
        "NEXTCLOUD_URL": "https://nc.example",
        "NEXTCLOUD_USER": "thedirektor",
        "NEXTCLOUD_PASS": "app-pass",
    }
    result = library.upload_to_nextcloud([str(report)], "My Video (abc123)", cfg)

    assert result["error"] is None
    assert result["uploaded"] == 1

    mkcol_calls = [c for c in calls if c.get_method() == "MKCOL"]
    put_calls = [c for c in calls if c.get_method() == "PUT"]
    assert len(mkcol_calls) == 2  # Watch/ then Watch/<folder>/
    assert any(c.full_url.endswith("/Watch/My%20Video%20%28abc123%29/") for c in mkcol_calls)
    assert len(put_calls) == 1
    assert put_calls[0].full_url.endswith("/report.md")
    assert put_calls[0].get_header("Authorization", "").startswith("Basic ")


# --------------------------------------------------------------------------
# Paperless upload flow
# --------------------------------------------------------------------------

def test_paperless_missing_token_never_raises():
    result = library.upload_to_paperless(["report.md"], "My Video (abc123)", {})
    assert result["uploaded"] == 0
    assert result["error"] and "PAPERLESS_TOKEN" in result["error"]


def test_paperless_upload_resolves_tag_and_posts(tmp_path, monkeypatch):
    report = tmp_path / "report.md"
    report.write_text("# report body", encoding="utf-8")

    calls = []

    def fake_urlopen(request, timeout=None, context=None):
        calls.append(request)
        method = request.get_method()
        url = request.full_url
        if "/api/tags/" in url and method == "GET":
            return _Resp({"count": 1, "results": [{"id": 7, "name": "watch"}]})
        if url.endswith("/api/documents/post_document/") and method == "POST":
            return _Resp(b'"11111111-2222-3333-4444-555555555555"')
        raise AssertionError(f"unexpected request {method} {url}")

    monkeypatch.setattr(library, "urlopen", fake_urlopen)

    cfg = {"PAPERLESS_URL": "https://paperless.example", "PAPERLESS_TOKEN": "tok_abc"}
    result = library.upload_to_paperless([str(report)], "My Video (abc123)", cfg)

    assert result["error"] is None
    assert result["uploaded"] == 1

    # Auth header on every call
    for c in calls:
        assert c.get_header("Authorization", "") == "Token tok_abc"

    # NEVER deletes: Paperless keeps deleted docs (and their content hash) in a
    # 30-day trash, so a delete-then-reupload of identical bytes is rejected as
    # a duplicate — leaving zero active docs. We rely on Paperless's own
    # content-hash dedup instead (identical re-run is auto-rejected, original
    # stays). So no DELETE is ever issued.
    assert not [c for c in calls if c.get_method() == "DELETE"]

    # Document posted with the tag id and title, file field named "document"
    post_calls = [c for c in calls if c.full_url.endswith("/api/documents/post_document/")]
    assert len(post_calls) == 1
    body = post_calls[0].data
    assert b'name="document"; filename="report.md"' in body
    assert b'name="title"' in body and b"My Video (abc123)" in body
    assert b'name="tags"' in body and b"7" in body


def test_paperless_creates_tag_when_missing(tmp_path, monkeypatch):
    report = tmp_path / "report.md"
    report.write_text("# body", encoding="utf-8")

    calls = []

    def fake_urlopen(request, timeout=None, context=None):
        calls.append(request)
        method = request.get_method()
        url = request.full_url
        if "/api/tags/" in url and method == "GET":
            return _Resp({"count": 0, "results": []})
        if url.endswith("/api/tags/") and method == "POST":
            return _Resp({"id": 99, "name": "watch"})
        if "/api/documents/" in url and method == "GET":
            return _Resp({"count": 0, "results": []})
        if url.endswith("/api/documents/post_document/") and method == "POST":
            return _Resp(b'"task-uuid"')
        raise AssertionError(f"unexpected request {method} {url}")

    monkeypatch.setattr(library, "urlopen", fake_urlopen)

    cfg = {"PAPERLESS_URL": "https://paperless.example", "PAPERLESS_TOKEN": "tok"}
    result = library.upload_to_paperless([str(report)], "Vid (x1)", cfg)

    assert result["uploaded"] == 1
    tag_post = [c for c in calls if c.full_url.endswith("/api/tags/") and c.get_method() == "POST"]
    assert len(tag_post) == 1
    assert json.loads(tag_post[0].data)["name"] == "watch"
    # new tag id 99 flows into the document post
    post = [c for c in calls if c.full_url.endswith("/api/documents/post_document/")][0]
    assert b"99" in post.data


def test_paperless_upload_errors_never_raise(monkeypatch):
    def boom(request, timeout=None, context=None):
        raise RuntimeError("paperless down")

    monkeypatch.setattr(library, "urlopen", boom)
    cfg = {"PAPERLESS_TOKEN": "tok"}
    # even a hard failure on tag lookup / post must not raise
    result = library.upload_to_paperless(["/nonexistent/report.md"], "T (v)", cfg)
    assert result["uploaded"] == 0
    assert isinstance(result.get("error"), str)


# --------------------------------------------------------------------------
# Paperless routing in save_artifacts (additive with Nextcloud)
# --------------------------------------------------------------------------

def test_router_sends_report_to_paperless_additive(tmp_path, monkeypatch):
    report = tmp_path / "report.md"
    report.write_text("# report", encoding="utf-8")

    nextcloud_calls = []
    paperless_calls = []
    monkeypatch.setattr(
        library, "upload_to_nextcloud",
        lambda files, folder, cfg: nextcloud_calls.append(list(files)) or {"uploaded": len(files), "error": None},
    )
    monkeypatch.setattr(
        library, "upload_to_paperless",
        lambda files, title, cfg: paperless_calls.append(list(files)) or {"uploaded": len(files), "error": None},
    )

    cfg = {
        "NEXTCLOUD_PASS": "pass",
        "PAPERLESS_URL": "https://paperless.example",
        "PAPERLESS_TOKEN": "tok",
        "WATCH_SAVE_FRAME_CAP": 20,
    }
    summary = library.save_artifacts(None, [], str(report), {}, "My Video", "abc123", cfg=cfg)

    # report.md goes to BOTH targets (additive), not one or the other
    assert nextcloud_calls == [[str(report)]]
    assert paperless_calls == [[str(report)]]
    assert any("Paperless" in line for line in summary)
    assert any("Nextcloud" in line for line in summary)


def test_router_paperless_skipped_without_token(tmp_path, monkeypatch):
    report = tmp_path / "report.md"
    report.write_text("# report", encoding="utf-8")

    paperless_calls = []
    monkeypatch.setattr(
        library, "upload_to_nextcloud",
        lambda files, folder, cfg: {"uploaded": len(files), "error": None},
    )
    monkeypatch.setattr(
        library, "upload_to_paperless",
        lambda files, title, cfg: paperless_calls.append(files) or {"uploaded": len(files), "error": None},
    )

    cfg = {"NEXTCLOUD_PASS": "pass", "WATCH_SAVE_FRAME_CAP": 20}
    summary = library.save_artifacts(None, [], str(report), {}, "T", "v1", cfg=cfg)

    assert len(paperless_calls) == 0
    assert any("Paperless" in line and "skipped" in line for line in summary)


def test_load_config_paperless(monkeypatch, tmp_path):
    for k in ("PAPERLESS_URL", "PAPERLESS_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(library.Path, "home", lambda: tmp_path / "nohome")
    monkeypatch.chdir(tmp_path)

    cfg = library.load_config()
    assert cfg["PAPERLESS_URL"] == "https://paperless.cort3x.me"
    assert cfg["PAPERLESS_TOKEN"] is None

    monkeypatch.setenv("PAPERLESS_TOKEN", "tok_env")
    cfg2 = library.load_config()
    assert cfg2["PAPERLESS_TOKEN"] == "tok_env"


# --------------------------------------------------------------------------
# Missing config skips only the affected target
# --------------------------------------------------------------------------

def test_missing_config_skips_target(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"v")
    report = tmp_path / "report.md"
    report.write_text("# r", encoding="utf-8")

    immich_calls = []
    nextcloud_calls = []
    monkeypatch.setattr(
        library, "upload_to_immich",
        lambda files, album_name, cfg: immich_calls.append(files) or {"uploaded": len(files), "error": None},
    )
    monkeypatch.setattr(
        library, "upload_to_nextcloud",
        lambda files, folder, cfg: nextcloud_calls.append(files) or {"uploaded": len(files), "error": None},
    )

    # No NEXTCLOUD_PASS -> nextcloud skipped, immich still attempted.
    cfg_no_nc = {"IMMICH_API_KEY": "key", "WATCH_SAVE_FRAME_CAP": 20}
    summary = library.save_artifacts(str(video), [], str(report), {}, "T", "v1", cfg=cfg_no_nc)
    assert len(immich_calls) == 1
    assert len(nextcloud_calls) == 0
    assert any("skipped" in line and "NEXTCLOUD_PASS" in line for line in summary)

    immich_calls.clear()
    nextcloud_calls.clear()

    # No IMMICH_API_KEY -> immich skipped, nextcloud still attempted.
    cfg_no_immich = {"NEXTCLOUD_PASS": "pass", "WATCH_SAVE_FRAME_CAP": 20}
    summary2 = library.save_artifacts(str(video), [], str(report), {}, "T", "v1", cfg=cfg_no_immich)
    assert len(immich_calls) == 0
    assert len(nextcloud_calls) == 1
    assert any("skipped" in line and "IMMICH_API_KEY" in line for line in summary2)


# --------------------------------------------------------------------------
# save_artifacts never raises
# --------------------------------------------------------------------------

def test_save_artifacts_never_raises(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"v")

    def boom(files, album_name, cfg):
        raise RuntimeError("network exploded")

    monkeypatch.setattr(library, "upload_to_immich", boom)

    cfg = {"IMMICH_API_KEY": "key", "WATCH_SAVE_FRAME_CAP": 20}
    summary = library.save_artifacts(str(video), [], None, {}, "T", "v1", cfg=cfg)

    assert isinstance(summary, list)
    assert any("network exploded" in line or "error" in line.lower() for line in summary)


# --------------------------------------------------------------------------
# load_config (bonus: cheap, protects the interface Task 3 depends on)
# --------------------------------------------------------------------------

def test_load_config_defaults_and_env(monkeypatch, tmp_path):
    monkeypatch.delenv("IMMICH_BASE_URL", raising=False)
    monkeypatch.delenv("IMMICH_API_KEY", raising=False)
    monkeypatch.delenv("NEXTCLOUD_URL", raising=False)
    monkeypatch.delenv("NEXTCLOUD_USER", raising=False)
    monkeypatch.delenv("NEXTCLOUD_PASS", raising=False)
    monkeypatch.delenv("WATCH_SAVE_FRAME_CAP", raising=False)
    monkeypatch.setattr(library.Path, "home", lambda: tmp_path / "nohome")
    monkeypatch.chdir(tmp_path)

    cfg = library.load_config()
    assert cfg["IMMICH_BASE_URL"] == "https://immich.cort3x.me"
    assert cfg["NEXTCLOUD_URL"] == "https://nextcloud.cort3x.me"
    assert cfg["NEXTCLOUD_USER"] == "thedirektor"
    assert cfg["WATCH_SAVE_FRAME_CAP"] == 20
    assert cfg["IMMICH_API_KEY"] is None
    assert cfg["NEXTCLOUD_PASS"] is None

    monkeypatch.setenv("IMMICH_API_KEY", "sk_env")
    monkeypatch.setenv("WATCH_SAVE_FRAME_CAP", "5")
    cfg2 = library.load_config()
    assert cfg2["IMMICH_API_KEY"] == "sk_env"
    assert cfg2["WATCH_SAVE_FRAME_CAP"] == 5
