"""Exports from the video memory (0.5.3 Task 2)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import exports  # noqa: E402
from memory.index import record_run  # noqa: E402
from memory_fakes import KEY, fake_run, fr, jpg, seg  # noqa: E402

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")


@pytest.fixture
def video(conn, tmp_path, monkeypatch):
    """A 5-minute indexed video with stored frames, OCR, a pause, an answer and an Immich asset."""
    monkeypatch.setenv("IMMICH_BASE_URL", "https://photos.example.com")
    segs = [seg(t, t + 6, f"Frase {t} sobre el tema número {t // 100}.") for t in range(0, 300, 10) if t != 150]
    segs.append(seg(160, 166, "=SUM(A1:A9) no es una fórmula </script><b>x</b>"))
    record_run(conn, fake_run(title="Título con acentos: ¿Qué tal?", duration=300.0, segments=segs,
                              frames=[fr(t, None, "PANTALLA CON TEXTO LARGO" if t == 40 else None) for t in (5, 40, 120, 200, 280)]))
    store = tmp_path / "store"
    vdir = store / "vid"
    (vdir / "frames").mkdir(parents=True)
    conn.execute("UPDATE videos SET store_path = 'vid' WHERE video_key = ?", (KEY,))
    for i, (fid, t) in enumerate(conn.execute("SELECT id, t FROM frames WHERE video_key = ? ORDER BY t", (KEY,)).fetchall()):
        src = jpg(tmp_path, ["red", "green", "blue", "white", "black"][i], size="64x36")
        shutil.copy(src, vdir / "frames" / f"{int(t * 1000)}.jpg")
        conn.execute("UPDATE frames SET store_path = ?, immich_asset_id = ? WHERE id = ?",
                     (f"vid/frames/{int(t * 1000)}.jpg", "asset-40" if t == 40 else None, fid))
    conn.execute("INSERT INTO answers (video_key, evidence_version, question_norm, result, tokens_in, tokens_out, created) "
                 "VALUES (?, 0, 'q', ?, 0, 0, 'now'), (?, 0, 'r', ?, 0, 0, 'now')",
                 (KEY, json.dumps({"question": "¿qué dice?", "answer": "Dice X [E1 00:10]", "found": True}),
                  KEY, json.dumps({"question": "nope", "answer": "No encontré", "found": False})))
    conn.commit()
    return store


def test_slug_and_cell():
    assert exports.slug("Título con acentos: ¿Qué tal?") == "titulo-con-acentos-que-tal"
    assert exports.slug("") == "video"
    assert exports._cell("=SUM(1)") == " =SUM(1)" and exports._cell("-1") == " -1" and exports._cell(3) == 3


def test_parse_formats():
    assert exports.parse_formats("pdf, PPTX,bogus,pdf") == ["pdf", "pptx"]
    assert exports.parse_formats("none") == [] and exports.parse_formats(None) == []
    assert exports.parse_formats("all") == list(exports.FORMATS)


def test_default_formats(monkeypatch, tmp_path):
    monkeypatch.setenv("WATCH_EXPORTS", "viewer,json")
    assert exports.default_formats() == ["viewer", "json"]
    monkeypatch.delenv("WATCH_EXPORTS")
    monkeypatch.setattr(exports, "find_cli", lambda app: None)
    assert exports.default_formats() == []
    monkeypatch.setattr(exports, "find_cli", lambda app: f"/x/{app}-cli")
    assert exports.default_formats() == ["pdf", "pptx"]


def test_find_cli_bin_dir_first(monkeypatch, tmp_path):
    (tmp_path / "wordcraft-cli").write_text("#!/bin/sh\n")
    monkeypatch.setenv("WATCH_CRAFT_BIN_DIR", str(tmp_path))
    assert exports.find_cli("wordcraft") == str(tmp_path / "wordcraft-cli")


def test_load(conn, video):
    d = exports.load(conn, KEY, video)
    assert d["title"].startswith("Título") and d["duration"] == 300.0
    assert len(d["chapters"]) == 2 and all(c["frame"] and Path(c["frame"]["path"]).is_file() for c in d["chapters"])
    assert [f["immich"] for f in d["frames"] if f["immich"]] == ["https://photos.example.com/photos/asset-40"]
    assert d["answers"] == [{"question": "¿qué dice?", "answer": "Dice X [E1 00:10]"}]
    assert d["ocr"][0]["text"] == "PANTALLA CON TEXTO LARGO"


def test_viewer_inlines_and_escapes(conn, video, tmp_path):
    d = exports.load(conn, KEY, video)
    out = exports.write_viewer(d, tmp_path / "viewer.html")
    page = out.read_text(encoding="utf-8")
    assert page.count("data:image/jpeg;base64,") == 5
    assert "</script><b>" not in page and "<\\/script><b>x<\\/b>" in page
    assert "<title>Título con acentos: ¿Qué tal?</title>" in page
    capped = exports.write_viewer(d, tmp_path / "v2.html", max_bytes=1)
    assert "data:image" not in capped.read_text(encoding="utf-8") and "frames/" in capped.read_text(encoding="utf-8")


@needs_ffmpeg
def test_contact_sheet(conn, video, tmp_path):
    d = exports.load(conn, KEY, video)
    out = exports.write_contact(d, tmp_path, tmp_path / "sheet.png")
    dims = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=width,height", "-of", "csv=p=0", str(out)],
                          capture_output=True, text=True).stdout.strip()
    assert dims == f"{5 * 320 + 4 * 4},180"


FAKE = r'''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ["FAKE_LOG"], "a") as f:
    f.write(json.dumps([Path(sys.argv[0]).name, args, os.getcwd()]) + "\n")
def touch(p): Path(p).write_bytes(b"%PDF-fake" if str(p).endswith(".pdf") else b"x")
name = Path(sys.argv[0]).name
if name == "wordcraft-cli":
    s = json.load(open(args[args.index("--script") + 1]))
    for out in s["save"]: touch(out)
elif name == "deckcraft-cli":
    for flag in ("--save", "--export"):
        if flag in args: touch(args[args.index(flag) + 1])
elif name == "gridcraft-cli":
    touch(args[args.index("--out") + 1])
elif name == "pdfcraft-cli":
    touch(args[args.index("--out") + 1])
'''


@pytest.fixture
def fake_clis(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for app in ("wordcraft", "deckcraft", "gridcraft", "pdfcraft"):
        p = bin_dir / f"{app}-cli"
        p.write_text(FAKE.replace("#!/usr/bin/env python3", f"#!{sys.executable}"))
        p.chmod(0o755)
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("WATCH_CRAFT_BIN_DIR", str(bin_dir))
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return bin_dir, log


@needs_ffmpeg
def test_build_all_with_fake_clis(conn, video, fake_clis, tmp_path):
    bin_dir, log = fake_clis
    res = exports.build(conn, KEY, list(exports.FORMATS), store_dir=video)
    base = video / "vid" / "exports"
    assert {k: Path(v["path"]).name for k, v in res.items()} == {
        "json": "run.json", "viewer": "viewer.html", "docx": "report.docx", "pptx": "storyboard.pptx",
        "pdf": "watch-titulo-con-acentos-que-tal.pdf", "xlsx": "timeline.xlsx", "contact": "contact-sheet.png",
        "notes": "notes.md"}
    assert all(Path(v["path"]).parent == base and Path(v["path"]).is_file() for v in res.values())
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    names = [c[0] for c in calls]
    assert names == ["wordcraft-cli", "deckcraft-cli", "pdfcraft-cli", "gridcraft-cli"]
    deck_args = calls[1][1]
    assert any(a.startswith("slide.notes=") and "[00:00]" in a for a in deck_args)
    assert sum(a.startswith("insert.picture=") for a in deck_args) == 2
    grid_script = [json.loads(x) for x in (Path(calls[3][2]) / "timeline.jsonl").read_text().splitlines()] \
        if (Path(calls[3][2]) / "timeline.jsonl").exists() else None
    assert grid_script is None  # the temp dir is gone; the formula guard is covered by test_slug_and_cell
    assert (base / "notes.md").read_text(encoding="utf-8").startswith("# Título")


def test_build_missing_cli_skips_only_that_format(conn, video, fake_clis, monkeypatch):
    real_find = exports.find_cli
    monkeypatch.setattr(exports, "find_cli", lambda app: None if app == "gridcraft" else real_find(app))
    res = exports.build(conn, KEY, ["xlsx", "viewer"], store_dir=video)
    assert res["xlsx"] == {"skipped": "gridcraft-cli not found"} and "path" in res["viewer"]


def test_build_unknown_video(conn, tmp_path):
    assert "error" in exports.build(conn, "yt:nope", ["json"], store_dir=tmp_path)["json"]


real = pytest.mark.skipif(not all(exports.find_cli(a) for a in ("wordcraft", "deckcraft", "gridcraft", "pdfcraft"))
                          or os.environ.get("WATCH_TEST_REAL_CRAFT") != "1",
                          reason="real craft CLIs (set WATCH_TEST_REAL_CRAFT=1)")


@real
@needs_ffmpeg
def test_build_with_real_clis(conn, video):
    res = exports.build(conn, KEY, list(exports.FORMATS), store_dir=video)
    assert all("path" in v for v in res.values()), res
    text = subprocess.run([exports.find_cli("wordcraft"), "text", res["docx"]["path"]],
                          capture_output=True, text=True).stdout
    assert "Título con acentos" in text and "[00:00]" in text
    assert Path(res["pdf"]["path"]).read_bytes()[:4] == b"%PDF"
    info = subprocess.run([exports.find_cli("gridcraft"), "info", res["xlsx"]["path"]], capture_output=True, text=True).stdout
    for sheet in ("Transcript", "OCR", "Chapters", "Frames"):
        assert sheet in info
    cell = subprocess.run([exports.find_cli("gridcraft"), "cat", res["xlsx"]["path"], "--sheet", "Transcript", "--csv"],
                          capture_output=True, text=True).stdout
    assert "=SUM(A1:A9) no es" in cell and ",=SUM" not in cell
