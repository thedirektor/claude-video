"""Final-review fixes for the breakdown feature (see the findings list)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from breakdown import assemble  # noqa: E402
from breakdown import audio as A  # noqa: E402

SHOTS = [{"boundary": {"kind": "cut", "t": t}} for t in (1.0, 2.0, 3.0, 4.0)]
BEATS = [0.5 * i for i in range(10)]


# ---- 1: cuts on beat is never more confident than the tempo, and shows the other octave ----
def test_cuts_on_beat_confidence_capped_by_tempo():
    res = A.cuts_on_beat(SHOTS, BEATS, 25, tempo_conf=0.4)
    assert res["evidence"]["confidence"] <= 0.4


def test_cuts_on_beat_reports_alt_octave_share():
    shots = [{"boundary": {"kind": "cut", "t": t}} for t in (0.75, 1.25, 1.75, 2.25)]  # midpoints only
    res = A.cuts_on_beat(shots, BEATS, 25, tempo_conf=0.5, alt_ratio=2.0)
    assert res["share"] == 0.0 and res["share_alt"] == 1.0


def test_ambiguous_octave_brief_marks_low_confidence(monkeypatch):
    amb = {"tempo_bpm": 86.0, "tempo_alt": 172.0, "tempo_alt_ratio": 2.0, "beats": BEATS, "loudness": None,
           "evidence": assemble._ev(0, 5, "measured", 0.5), "loudness_evidence": assemble._ev(0, 5, "measured", 0.5),
           "notes": ["half/double-time ambiguous"]}
    monkeypatch.setattr(A, "analyze", lambda *a, **k: dict(amb))
    monkeypatch.setattr(A, "music_windows", lambda *a, **k: None)
    sec = assemble._audio_section({"path": "x", "has_audio": True, "fps": 25.0}, SHOTS, 5.0, [], [])
    cob = sec["cuts_on_beat"]
    assert cob["evidence"]["confidence"] <= 0.5 and "share_alt" in cob


# ---- 2: typography never reports a wrong colour or a fragment at high confidence ----
from breakdown import typography as TY  # noqa: E402


def test_color_not_determined_over_busy_background(tmp_path):
    import subprocess
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    if not Path(font).is_file():
        pytest.skip("no font for drawtext")
    p = tmp_path / "busy.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=s=640x360:r=25:d=4", "-vf",
                    f"drawtext=fontfile={font}:text='HELLO WORLD':fontsize=56:x=40:y=40:fontcolor=white",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(p)], check=True)
    tracks, notes = TY.text_tracks(str(p), 4.0)
    assert notes == []
    ours = [t for t in tracks if any(w in t["text"].upper() for w in ("HELLO", "WORLD", "ORLD"))]
    assert ours, tracks
    for t in ours:
        if t["color_hex"] != "not determined":  # only acceptable if it is actually right (near white)
            assert all(int(t["color_hex"][i:i + 2], 16) >= 0xc0 for i in (1, 3, 5)), t["color_hex"]


def test_color_hex_guard_low_contrast_and_busy():
    import cv2
    import numpy as np
    rng = np.random.default_rng(1)
    busy = rng.integers(0, 255, (200, 400, 3), dtype=np.uint8)
    busy[80:120, 100:300] = 255
    assert TY._color_hex(cv2, np, busy, [0.25, 0.4, 0.5, 0.2]) == "not determined"
    flat = np.full((200, 400, 3), 120, np.uint8)
    flat[80:120, 100:300] = 130  # text barely differs from its background
    assert TY._color_hex(cv2, np, flat, [0.25, 0.4, 0.5, 0.2]) == "not determined"
    navy = np.full((200, 400, 3), (60, 20, 10), np.uint8)
    navy[80:120, 100:300] = 255
    assert TY._color_hex(cv2, np, navy, [0.25, 0.4, 0.5, 0.2]) == "#ffffff"


def test_track_confidence_folds_ocr_confidence():
    assert TY._track_conf(2, 0.62) < 0.5          # a short fragment read with shaky confidence
    assert TY._track_conf(3, 0.65) < 0.6
    assert TY._track_conf(6, 0.96) > 0.85         # a long, well-read title stays confident
    assert TY._track_conf(6, 0.96) <= 0.95


# ---- fixtures shared with the assemble tests ----
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402

from breakdown import cli, media, motion  # noqa: E402
from memory import bundle, index, store as mstore  # noqa: E402
from memory_fakes import KEY, fake_run, seg  # noqa: E402
from test_breakdown_assemble import ref_clip, stored  # noqa: E402,F401


def _static_clip(path, seconds=3):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    f"color=c=gray:s=320x180:r=25:d={seconds}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(path)], check=True)


@pytest.fixture
def one_shot(conn, tmp_path):
    index.record_run(conn, fake_run(title="Still", duration=3.0, segments=[seg(0, 1, "hi")]))
    root = tmp_path / "st1"
    vdir = mstore.video_dir(root, KEY)
    vdir.mkdir(parents=True)
    _static_clip(vdir / "media.mp4")
    return root


# ---- 5: the build contract travels in the JSON ----
def test_contract_in_breakdown_json(conn, stored):
    bd = assemble.build(conn, KEY, store_dir=stored)
    assert bd["contract"] == assemble.CONTRACT
    out = assemble.store(conn, KEY, bd, store_dir=stored)
    assert json.loads((out / "breakdown.json").read_text())["contract"] == assemble.CONTRACT


def test_contract_in_motion_json(conn, tmp_path):
    index.record_run(conn, fake_run(title="Gone", duration=3.0))
    m = motion.analyze(conn, KEY, 0.0, 1.0, store_dir=tmp_path / "none")
    assert m["contract"] == assemble.CONTRACT


# ---- T-a / T-b: honest sampling and blind-spot disclosure ----
def test_single_shot_confidence_capped_with_blind_spot_note(conn, one_shot):
    bd = assemble.build(conn, KEY, store_dir=one_shot)
    assert len(bd["shots"]) == 1
    assert bd["shots"][0]["evidence"]["confidence"] <= 0.6
    assert any("dissolves between two moving shots" in n for n in bd["notes"])


def test_no_text_brief_discloses_sampling(conn, one_shot):
    bd = assemble.build(conn, KEY, store_dir=one_shot)
    md = assemble.brief_md(bd)
    assert "No on-screen text found." not in md
    assert "sampled every 0.5 s; shorter or stylised text may be missed" in md
    assert "sampled every 0.5 s" in bd["typography"]["evidence"]["sampling"]


# ---- 8: variable / non-standard frame rates ----
def test_snap_fps_nearest_standard():
    r, note, off = media.snap_fps("1784000/59623")
    assert r == "30000/1001" and "1784000/59623" in note and "30000/1001" in note and not off
    assert media.snap_fps("25/1") == ("25/1", None, False)
    assert media.snap_fps("30000/1001") == ("30000/1001", None, False)
    r, note, off = media.snap_fps("1000/33")  # 30.30 fps: 1.1% from 30000/1001 -> keep measured
    assert r == "1000/33" and off and "non-standard" in note


def test_probe_snaps_vfr(monkeypatch, tmp_path):
    out = json.dumps({"streams": [{"codec_type": "video", "avg_frame_rate": "1784000/59623", "width": 640,
                                   "height": 360}], "format": {"duration": "10"}})
    monkeypatch.setattr(media.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=out, stderr=""))
    info = media.probe(tmp_path / "x.mp4")
    # analysis keeps the measured rate; only timelines use the snapped one
    assert info["fps_rational"] == "1784000/59623" and abs(info["fps"] - 29.921) < 0.01
    assert info["timeline_fps_rational"] == "30000/1001"
    assert info["fps_note"] and info["fps_offstandard"] is False


def test_offstandard_fps_lowers_confidence_and_notes(conn, stored, monkeypatch):
    real = media.ensure_media

    def fake(*a, **k):
        m = real(*a, **k)
        return {**m, "fps_rational": "1000/33", "timeline_fps_rational": "1000/33", "fps": 1000 / 33,
                "fps_offstandard": True,
                "fps_note": "non-standard frame rate 1000/33 kept as measured"}

    monkeypatch.setattr(media, "ensure_media", fake)
    bd = assemble.build(conn, KEY, store_dir=stored)
    assert any("non-standard frame rate" in n for n in bd["notes"])
    assert bd["video"]["evidence"]["confidence"] <= 0.5


# ---- 9: media is pinned as soon as it is secured ----
def test_pinned_before_analysis_runs(conn, stored, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("analysis died")

    monkeypatch.setattr(assemble.shots_mod, "frame_series", boom)
    with pytest.raises(RuntimeError):
        assemble.build(conn, KEY, store_dir=stored)
    assert media.is_pinned(stored, KEY)


def test_motion_pins_as_soon_as_media_is_secured(conn, stored, monkeypatch):
    monkeypatch.setattr(motion, "_read_frames", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    assert not media.is_pinned(stored, KEY)
    motion.analyze(conn, KEY, 0.0, 1.0, store_dir=stored)
    assert media.is_pinned(stored, KEY)


# ---- 3b: --stored read path and the interpretation that must survive a re-run ----
def test_cli_stored_returns_without_rerun(conn, stored, monkeypatch, capsys):
    monkeypatch.setenv("WATCH_STORE_DIR", str(stored))
    assert cli.main(["breakdown", KEY, "--stored", "--json"]) == 2  # nothing stored yet
    assert "no stored breakdown" in capsys.readouterr().err
    assert cli.main(["breakdown", KEY, "--json"]) == 0
    first = json.loads(capsys.readouterr().out)
    monkeypatch.setattr(assemble, "build", lambda *a, **k: (_ for _ in ()).throw(AssertionError("re-ran")))
    assert cli.main(["breakdown", KEY, "--stored", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["schema"] == first["schema"]
    assert cli.main(["breakdown", KEY, "--stored"]) == 0
    assert capsys.readouterr().out.startswith("> **Build contract:**")


def test_rerun_keeps_stored_interpretation(conn, stored, monkeypatch, capsys):
    monkeypatch.setenv("WATCH_STORE_DIR", str(stored))
    assert cli.main(["breakdown", KEY]) == 0
    capsys.readouterr()
    out = mstore.video_dir(stored, KEY) / "breakdown"
    bdj = json.loads((out / "breakdown.json").read_text())
    bdj["interpretation"] = {"text": "Punchy.", "made_from": "measured fields (no frames)", "provider": "p",
                             "model": "m", "evidence": {"t_start": 0, "t_end": 1, "method": "interpretation",
                                                         "confidence": 0.3}, "evidence_version": 3}
    (out / "breakdown.json").write_text(json.dumps(bdj))
    assert cli.main(["breakdown", KEY, "--json"]) == 0
    new = json.loads(capsys.readouterr().out)
    assert new["interpretation"]["text"] == "Punchy." and new["interpretation"]["evidence_version"] == 3
    assert new["interpretation"].get("stale") is not None
    assert "Punchy." in (out / "breakdown.md").read_text()
    assert json.loads((out / "breakdown.json").read_text())["interpretation"]["text"] == "Punchy."


# ---- 4: the PC pushes breakdowns; bundles carry the subfolders (not compare-*) ----
def test_cli_pushes_after_breakdown_and_motion(conn, stored, monkeypatch):
    import push

    monkeypatch.setenv("WATCH_STORE_DIR", str(stored))
    pushed = []
    monkeypatch.setattr(push, "enabled", lambda: True)
    monkeypatch.setattr(push, "flush_outbox", lambda *a, **k: {"sent": 0, "failed": 0})
    monkeypatch.setattr(push, "push_video", lambda key, **k: pushed.append(key) or {"ok": True})
    assert cli.main(["breakdown", KEY]) == 0
    assert cli.main(["motion", KEY, "--start", "0", "--end", "1", "--region", "0.1,0.1,0.2,0.2"]) == 0
    assert pushed == [KEY, KEY]
    monkeypatch.setattr(push, "enabled", lambda: False)
    assert cli.main(["breakdown", KEY]) == 0
    assert pushed == [KEY, KEY]


def test_bundle_recurses_into_motion_not_compare(conn, stored, tmp_path, monkeypatch):
    conn.execute("UPDATE videos SET store_path = ? WHERE video_key = ?", (mstore.video_dir(stored, KEY).name, KEY))
    conn.commit()
    bdir = mstore.video_dir(stored, KEY) / "breakdown"
    (bdir / "motion-0-1").mkdir(parents=True)
    (bdir / "motion-0-1" / "motion.json").write_text('{"x": 1}')
    (bdir / "motion-0-1" / "godot.gd").write_text("tween")
    (bdir / "compare-2026").mkdir()
    (bdir / "compare-2026" / "compare.mp4").write_bytes(b"0" * 100)
    (bdir / "breakdown.json").write_text('{"schema": "watch.breakdown/1"}')
    z = bundle.make_bundle(conn, KEY, tmp_path / "b.zip", store_dir=stored, host="pc1")
    import zipfile
    names = zipfile.ZipFile(z).namelist()
    assert "breakdown/motion-0-1/motion.json" in names and "breakdown/motion-0-1/godot.gd" in names
    assert "breakdown/breakdown.json" in names
    assert not any("compare" in n for n in names)
    monkeypatch.setenv("WATCH_INDEX_DB", str(tmp_path / "central.db"))
    monkeypatch.setenv("WATCH_STORE_DIR", str(tmp_path / "central"))
    bundle.import_bundle(z, embedder=None)
    restored = mstore.video_dir(tmp_path / "central", KEY) / "breakdown"
    assert (restored / "motion-0-1" / "motion.json").read_text() == '{"x": 1}'


def test_import_refuses_unsafe_breakdown_paths(conn, stored, tmp_path, monkeypatch):
    import zipfile
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("run.json", json.dumps({"video_key": "yt:evil", "origin_ref": "h:yt:evil:v1"}))
        f.writestr("breakdown/../../escape.txt", "x")
    monkeypatch.setenv("WATCH_INDEX_DB", str(tmp_path / "central.db"))
    monkeypatch.setenv("WATCH_STORE_DIR", str(tmp_path / "central"))
    with pytest.raises(bundle.BundleError):
        bundle.import_bundle(z, embedder=None)


# ---- 6: compare output does not pile up in the media budget ----
def test_retention_prunes_compare_dirs_and_excludes_them_from_budget(conn, tmp_path):
    import os
    import time
    index.record_run(conn, fake_run(title="A", duration=3.0))
    root = tmp_path / "store"
    vdir = mstore.video_dir(root, KEY)
    now = time.time()
    for i in range(25):
        d = vdir / "breakdown" / f"compare-2026{i:02d}"
        d.mkdir(parents=True)
        (d / "compare.mp4").write_bytes(b"0" * 1000)
        os.utime(d, (now - i * 60, now - i * 60))
    old = vdir / "breakdown" / "compare-old"
    old.mkdir()
    (old / "compare.mp4").write_bytes(b"0" * 1000)
    os.utime(old, (now - 15 * 86400, now - 15 * 86400))
    loose = root / "compare" / "20260101"
    loose.mkdir(parents=True)
    (loose / "compare.mp4").write_bytes(b"0" * 1000)
    (vdir / "media.mp4").write_bytes(b"0" * 500)
    res = mstore.enforce_retention(conn, root, max_bytes=10_000)
    left = [p for p in list((vdir / "breakdown").glob("compare-*")) + list((root / "compare").glob("*"))]
    assert len(left) <= 20 and not old.exists()
    assert res["media_removed"] == 0  # compare output no longer pushes the store over its media budget
    assert (vdir / "media.mp4").exists()


# ---- stdio MCP: args that the CLI already supports ----
def test_stdio_extra_args(monkeypatch):
    import mcp_stdio
    calls = []
    monkeypatch.setattr(mcp_stdio, "_run_watch", lambda argv, timeout=3600: calls.append(argv)
                        or subprocess.CompletedProcess(argv, 0, stdout="{}", stderr=""))
    mcp_stdio.call_tool("breakdown", {"video": "v", "allow_cost": True, "refresh": False})
    assert calls[-1] == ["breakdown", "v", "--json", "--allow-cost", "--stored"]
    mcp_stdio.call_tool("breakdown", {"video": "v", "refresh": True})
    assert "--stored" not in calls[-1]
    mcp_stdio.call_tool("compare", {"reference": "r", "candidate": "c", "slow": 2})
    assert calls[-1] == ["compare", "r", "c", "--json", "--slow", "2"]
    desc = {t["name"]: t["description"] for t in mcp_stdio.TOOLS}
    assert "motion" not in desc["breakdown"]


# ---- doctor: which fields each dependency affects ----
def test_doctor_field_lists():
    src = (Path(__file__).parent.parent / "skills/watch/scripts/doctor.py").read_text()
    assert '"camera, look, typography, motion"' in src
    assert '"look, typography, audio, motion, compare"' in src


# ---- compare summary names the video by absolute path ----
def test_compare_summary_has_absolute_video_path(tmp_path):
    from breakdown import compare
    from test_breakdown_compare import _delay
    from breakdown_fixtures import make_reference
    clip = tmp_path / "ref.mp4"
    make_reference(clip)
    cand = tmp_path / "late.mp4"
    _delay(clip, cand, 0.4)
    out = tmp_path / "rel" / "out"
    res = compare.run(str(clip), str(cand), slow=2, out_dir=out)
    assert f"{out.resolve() / 'compare.mp4'}" in compare.summary_md(res)


def test_cli_compare_reports_output_folder(monkeypatch, capsys, tmp_path):
    from breakdown import compare
    monkeypatch.setattr(compare, "run", lambda *a, **k: {"video": "compare.mp4", "out_dir": str(tmp_path)})
    monkeypatch.setattr(compare, "summary_md", lambda res: "summary")
    assert cli.main(["compare", "ref", "cand"]) == 0
    assert f"[watch] compare stored in {tmp_path}" in capsys.readouterr().err


def test_timeline_uses_snapped_rate_analysis_uses_measured(conn, stored, monkeypatch):
    real = media.ensure_media

    def fake(*a, **k):
        m = real(*a, **k)
        return {**m, "fps_rational": "1784000/59623", "fps": 1784000 / 59623,
                "timeline_fps_rational": "30000/1001", "fps_note": "snapped"}

    monkeypatch.setattr(media, "ensure_media", fake)
    seen = {}
    real_fs = assemble.shots_mod.frame_series
    monkeypatch.setattr(assemble.shots_mod, "frame_series",
                        lambda path, fps: seen.setdefault("fps", fps) and real_fs(path, 25.0))
    bd = assemble.build(conn, KEY, store_dir=stored)
    assert seen["fps"] == pytest.approx(29.921, abs=0.01)
    assert bd["video"]["fps_rational"] == "30000/1001" and bd["video"]["fps"] == pytest.approx(29.921, abs=0.01)


def test_probe_nonstandard_clip_keeps_true_rate(tmp_path):
    p = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "color=c=gray:s=160x90:r=59.8:d=2", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(p)],
                   check=True)
    info = media.probe(p)
    assert info["fps"] == pytest.approx(59.8, abs=0.1)  # not snapped to 60 for frame-index -> time maths
    assert info["timeline_fps_rational"] in ("60/1", "60000/1001")


# ---- re-review residuals ----
def test_stored_path_prints_output_folder(conn, stored, monkeypatch, capsys):
    monkeypatch.setenv("WATCH_STORE_DIR", str(stored))
    assert cli.main(["breakdown", KEY]) == 0
    capsys.readouterr()
    assert cli.main(["breakdown", KEY, "--stored", "--json"]) == 0
    err = capsys.readouterr().err
    assert "[watch] breakdown stored in " in err and err.strip().endswith("breakdown")


def test_fragment_track_dropped_when_inside_longer_overlapping_track():
    box = [0.1, 0.1, 0.3, 0.1]
    full = {"text": "HELLO WORLD", "t_in": 1.0, "t_out": 3.0, "box": box, "evidence": {"confidence": 0.9}}
    frag = {"text": "ORLD", "t_in": 1.0, "t_out": 3.0, "box": [0.25, 0.1, 0.15, 0.1], "evidence": {"confidence": 0.87}}
    other = {"text": "ORLD", "t_in": 4.0, "t_out": 5.0, "box": [0.25, 0.1, 0.15, 0.1], "evidence": {"confidence": 0.87}}
    out = TY._drop_fragments([full, frag, other])
    assert [t["text"] for t in out] == ["HELLO WORLD", "ORLD"] and out[1]["t_in"] == 4.0


def test_notes_mention_merged_jump_cuts_and_dropped_same_picture_cuts(conn, one_shot):
    bd = assemble.build(conn, KEY, store_dir=one_shot)
    assert any("jump cuts within one framing may be merged" in n for n in bd["notes"])


def test_notes_count_cuts_dropped_inside_a_pan(conn, one_shot, monkeypatch):
    real = assemble.shots_mod.detect_shots

    def counted(series, duration):
        out = real(series, duration)
        series["pan_dropped"] = 2
        return out

    monkeypatch.setattr(assemble.shots_mod, "detect_shots", counted)
    bd = assemble.build(conn, KEY, store_dir=one_shot)
    assert any("2 cut candidate(s) dropped inside one continuous pan" in n for n in bd["notes"])
