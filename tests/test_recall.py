"""recall.py: /watch-recall retrieval — Paperless full-text search primary,
Nextcloud folder-title match fallback. All HTTP mocked via recall.urlopen;
never touches the network."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import recall  # noqa: E402


class _Resp:
    def __init__(self, payload):
        self._body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


DICE_REPORT = (
    "# watch: video report\n\n"
    "- **Source:** https://www.youtube.com/watch?v=7kSKQLd6WLs\n"
    "- **Title:** How to Make a DICE in Blender\n\n"
    "## Transcript\n\nNow round these edges with the Bevel Tool.\n"
)


# --------------------------------------------------------------------------
# Paperless search
# --------------------------------------------------------------------------

def test_search_paperless_returns_matches(monkeypatch):
    def fake_urlopen(request, timeout=None, context=None):
        url = request.full_url
        if "/api/documents/?query=" in url:
            return _Resp({"count": 1, "results": [{"id": 2, "title": "How to Make a DICE in Blender (Step-by-Step Tutorial) (e0d1e37a)"}]})
        if url.endswith("/api/documents/2/"):
            return _Resp({"id": 2, "title": "How to Make a DICE in Blender (Step-by-Step Tutorial) (e0d1e37a)", "content": DICE_REPORT})
        raise AssertionError(f"unexpected {url}")

    monkeypatch.setattr(recall, "urlopen", fake_urlopen)
    cfg = {"PAPERLESS_URL": "https://paperless.example", "PAPERLESS_TOKEN": "tok"}
    matches = recall.search_paperless("bevel", cfg, limit=5)

    assert len(matches) == 1
    m = matches[0]
    assert m["doc_id"] == 2
    assert "DICE in Blender" in m["title"]
    assert "Bevel Tool" in m["content"]
    assert m["source_url"] == "https://www.youtube.com/watch?v=7kSKQLd6WLs"


def test_search_paperless_no_token_returns_empty():
    assert recall.search_paperless("bevel", {}, limit=5) == []


# --------------------------------------------------------------------------
# recall() orchestration + formatting
# --------------------------------------------------------------------------

def test_recall_paperless_hit_formats_output(monkeypatch):
    def fake_urlopen(request, timeout=None, context=None):
        url = request.full_url
        if "/api/documents/?query=" in url:
            return _Resp({"count": 1, "results": [{"id": 2, "title": "How to Make a DICE in Blender (Step-by-Step Tutorial) (e0d1e37a)"}]})
        if url.endswith("/api/documents/2/"):
            return _Resp({"id": 2, "title": "How to Make a DICE in Blender (Step-by-Step Tutorial) (e0d1e37a)", "content": DICE_REPORT})
        raise AssertionError(f"unexpected {url}")

    monkeypatch.setattr(recall, "urlopen", fake_urlopen)
    cfg = {"PAPERLESS_URL": "https://paperless.example", "PAPERLESS_TOKEN": "tok"}

    out = recall.recall("bevel", cfg=cfg, limit=5, full=False)
    assert "watch-recall" in out
    assert "How to Make a DICE in Blender" in out
    assert "https://www.youtube.com/watch?v=7kSKQLd6WLs" in out
    # points at the Immich album by the same title for frames
    assert "Immich" in out
    # snippet mode: does NOT dump the whole transcript header line-for-line
    assert "via paperless" in out.lower()

    out_full = recall.recall("bevel", cfg=cfg, limit=5, full=True)
    assert "Now round these edges with the Bevel Tool." in out_full


def test_recall_falls_back_to_nextcloud_by_title(monkeypatch):
    # Paperless configured but returns nothing -> fall back to Nextcloud folder
    # listing, matching the query against folder titles.
    propfind_xml = (
        "<d:multistatus xmlns:d='DAV:'>"
        "<d:response><d:href>/remote.php/dav/files/thedirektor/Watch/</d:href></d:response>"
        "<d:response><d:href>/remote.php/dav/files/thedirektor/Watch/How%20to%20Make%20a%20DICE%20in%20Blender%20(Step-by-Step%20Tutorial)%20(e0d1e37a)/</d:href></d:response>"
        "</d:multistatus>"
    ).encode("utf-8")

    def fake_urlopen(request, timeout=None, context=None):
        url = request.full_url
        method = request.get_method()
        if "/api/documents/?query=" in url:
            return _Resp({"count": 0, "results": []})
        if method == "PROPFIND":
            return _Resp(propfind_xml)
        if method == "GET" and url.endswith("/report.md"):
            return _Resp(DICE_REPORT.encode("utf-8"))
        raise AssertionError(f"unexpected {method} {url}")

    monkeypatch.setattr(recall, "urlopen", fake_urlopen)
    cfg = {
        "PAPERLESS_URL": "https://paperless.example", "PAPERLESS_TOKEN": "tok",
        "NEXTCLOUD_URL": "https://nc.example", "NEXTCLOUD_USER": "thedirektor", "NEXTCLOUD_PASS": "pass",
    }
    out = recall.recall("dice", cfg=cfg, limit=5, full=True)
    assert "How to Make a DICE in Blender" in out
    assert "via nextcloud" in out.lower()
    assert "Bevel Tool" in out


def test_recall_no_matches(monkeypatch):
    def fake_urlopen(request, timeout=None, context=None):
        url = request.full_url
        if "/api/documents/?query=" in url:
            return _Resp({"count": 0, "results": []})
        if request.get_method() == "PROPFIND":
            return _Resp(b"<d:multistatus xmlns:d='DAV:'><d:response><d:href>/remote.php/dav/files/thedirektor/Watch/</d:href></d:response></d:multistatus>")
        raise AssertionError(f"unexpected {url}")

    monkeypatch.setattr(recall, "urlopen", fake_urlopen)
    cfg = {
        "PAPERLESS_URL": "https://p.example", "PAPERLESS_TOKEN": "tok",
        "NEXTCLOUD_URL": "https://nc.example", "NEXTCLOUD_USER": "thedirektor", "NEXTCLOUD_PASS": "pass",
    }
    out = recall.recall("nonexistent-topic", cfg=cfg, limit=5)
    assert "no matching" in out.lower()


def test_recall_never_raises(monkeypatch):
    def boom(request, timeout=None, context=None):
        raise RuntimeError("network down")

    monkeypatch.setattr(recall, "urlopen", boom)
    cfg = {"PAPERLESS_URL": "https://p.example", "PAPERLESS_TOKEN": "tok"}
    out = recall.recall("anything", cfg=cfg)
    assert isinstance(out, str)
    assert out  # non-empty, no exception
