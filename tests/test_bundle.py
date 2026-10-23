"""PC → central import: run bundles (0.6.0 phase D)."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from memory import bundle
from memory.index import connect, record_run
from memory_fakes import KEY, fake_run, fr, jpg, seg


def _local_video(conn, tmp_path):
    frames = [fr(1, jpg(tmp_path, "red"), "TEXTO EN PANTALLA"), fr(5, jpg(tmp_path, "blue"), None)]
    record_run(conn, fake_run(title="Clip", duration=10.0, transcript_source="captions",
                              segments=[seg(0, 4, "hola"), seg(4, 8, "adiós")], frames=frames,
                              notes=[{"kind": "backend_answer", "t_start": None, "t_end": None, "text": "Q: a\n\nB"}]))
    store = tmp_path / "store"
    vdir = store / "vid"
    (vdir / "frames").mkdir(parents=True)
    conn.execute("UPDATE videos SET store_path = 'vid' WHERE video_key = ?", (KEY,))
    for fid, t in conn.execute("SELECT id, t FROM frames ORDER BY t").fetchall():
        name = f"{int(t * 1000)}.jpg"
        (vdir / "frames" / name).write_bytes((tmp_path / "x.jpg").read_bytes() if (tmp_path / "x.jpg").exists() else b"\xff\xd8jpg")
        conn.execute("UPDATE frames SET store_path = ? WHERE id = ?", (f"vid/frames/{name}", fid))
    (vdir / "report.md").write_text("# watch: video report\n")
    conn.commit()
    return store


def test_make_bundle_contains_manifest_frames_report(conn, tmp_path):
    store = _local_video(conn, tmp_path)
    out = bundle.make_bundle(conn, KEY, tmp_path / "b.zip", store_dir=store, host="pc1", library_saved=True)
    with zipfile.ZipFile(out) as z:
        names = set(z.namelist())
        run = json.loads(z.read("run.json"))
    assert {"run.json", "report.md", "frames/1000.jpg", "frames/5000.jpg"} <= names
    assert run["video_key"] == KEY and run["title"] == "Clip" and run["transcript_source"] == "captions"
    assert [s["text"] for s in run["segments"]] == ["hola", "adiós"]
    assert run["frames"][0] == {"t": 1.0, "kind": "scene", "ocr": "TEXTO EN PANTALLA", "file": "frames/1000.jpg"}
    assert run["origin_ref"].startswith("pc1:" + KEY + ":v") and run["library_saved"] is True
    assert [n["kind"] for n in run["notes"]] == ["backend_answer"]


def test_import_bundle_merges_as_origin_import(conn, tmp_path, monkeypatch):
    store = _local_video(conn, tmp_path)
    zpath = bundle.make_bundle(conn, KEY, tmp_path / "b.zip", store_dir=store, host="pc1")
    central_db = tmp_path / "central.db"
    monkeypatch.setenv("WATCH_INDEX_DB", str(central_db))
    monkeypatch.setenv("WATCH_STORE_DIR", str(tmp_path / "central-store"))
    res = bundle.import_bundle(zpath, embedder=None)
    assert res["video_key"] == KEY and res["skipped"] is False
    c = connect(central_db)
    assert c.execute("SELECT origin FROM runs").fetchone()[0] == "import"
    assert c.execute("SELECT count(*) FROM segments").fetchone()[0] == 2
    assert c.execute("SELECT count(*) FROM ocr_blocks").fetchone()[0] == 1
    assert c.execute("SELECT count(*) FROM frames WHERE store_path IS NOT NULL").fetchone()[0] == 2
    again = bundle.import_bundle(zpath, embedder=None)            # same bundle twice → no-op
    assert again["skipped"] is True
    assert c.execute("SELECT count(*) FROM notes WHERE kind='backend_answer'").fetchone()[0] == 1


def test_import_rejects_path_traversal(tmp_path):
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("run.json", json.dumps({"video_key": "yt:x", "source": "s", "segments": [], "frames": [],
                                           "origin_ref": "h:yt:x:v1"}))
        f.writestr("../escape.txt", "x")
    with pytest.raises(bundle.BundleError):
        bundle.import_bundle(z, embedder=None)


def test_import_rejects_oversize(tmp_path, monkeypatch):
    z = tmp_path / "big.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("run.json", "{}")
    monkeypatch.setattr(bundle, "MAX_BUNDLE_BYTES", 10)
    with pytest.raises(bundle.BundleError):
        bundle.import_bundle(z, embedder=None)


def test_import_rejects_non_zip(tmp_path):
    p = tmp_path / "x.zip"
    p.write_bytes(b"not-a-zip")
    with pytest.raises(bundle.BundleError):
        bundle.import_bundle(p, embedder=None)
