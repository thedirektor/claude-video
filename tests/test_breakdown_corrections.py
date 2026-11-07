"""User corrections (breakdown/corrections.py + `watch.py correct`)."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from breakdown import assemble, cli, corrections  # noqa: E402
from breakdown import shots as S  # noqa: E402
from breakdown_fixtures import make_reference  # noqa: E402
from memory import index, store  # noqa: E402
from memory_fakes import KEY, fake_run, seg  # noqa: E402

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg needed")


# ----------------------------------------------------------------------------- pure

def test_make_validates():
    with pytest.raises(corrections.CorrectionError):
        corrections.make("production", 1.0, "nonsense")
    with pytest.raises(corrections.CorrectionError):
        corrections.make("add_cut", None)
    with pytest.raises(corrections.CorrectionError):
        corrections.make("bogus", 1.0)
    e = corrections.make("production", 1.23456, "camera", "phone footage")
    assert e["t"] == 1.235 and e["label"] == "camera" and e["note"] == "phone footage"


def _raw(bounds, duration=6.0):
    return S.build_shots([{"kind": "cut", "t": t, "frames": 1, "t_start": t - 0.04, "t_end": t, "confidence": 0.9}
                          for t in bounds], duration)


def test_apply_cuts_adds_removes_and_keeps_existing():
    raw = _raw([2.0, 4.0])
    out = corrections.apply_cuts(raw, [corrections.make("add_cut", 3.0), corrections.make("remove_cut", 4.1),
                                       corrections.make("add_cut", 2.01)], 6.0, 25.0)
    assert [s["start"] for s in out] == [0.0, 2.0, 3.0]
    added = out[2]["boundary"]
    assert added["corrected"] is True and added["evidence"]["method"] == "correction"
    assert corrections.apply_cuts(raw, [], 6.0, 25.0) is raw


def test_apply_production_keeps_the_measured_value():
    shots = _raw([2.0])
    shots[0]["production"] = {"label": "not determined", "candidates": [{"label": "camera", "confidence": 0.4}],
                              "evidence": {"method": "measured", "confidence": 0.0}}
    n = corrections.apply_production(shots, [corrections.make("production", 1.0, "ai_video", "garbled logo")])
    p = shots[0]["production"]
    assert n == 1 and p["label"] == "ai_video" and p["evidence"]["method"] == "correction"
    assert p["measured"]["label"] == "not determined"
    assert any("garbled logo" in x for x in p["notes"])
    # a second correction keeps the original measurement, not the first correction
    corrections.apply_production(shots, [corrections.make("production", 1.5, "camera")])
    assert shots[0]["production"]["measured"]["label"] == "not determined"


def test_load_tolerates_garbage(tmp_path):
    p = tmp_path / "c.json"
    p.write_text("{not json")
    assert corrections.load(p) == []
    p.write_text(json.dumps([{"kind": "add_cut", "t": 1}, {"kind": "evil"}, 3]))
    assert corrections.load(p) == [{"kind": "add_cut", "t": 1}]


# ----------------------------------------------------------------------------- CLI end to end

@pytest.fixture(scope="module")
def ref_clip(tmp_path_factory):
    p = tmp_path_factory.mktemp("corr") / "ref.mp4"
    return p, make_reference(p)


@pytest.fixture
def stored(conn, tmp_path, ref_clip, monkeypatch):
    clip, truth = ref_clip
    root = tmp_path / "st"
    index.record_run(conn, fake_run(title="Sale clip", duration=truth["duration"], segments=[seg(0, 3, "hello sale")]))
    vdir = store.video_dir(root, KEY)
    vdir.mkdir(parents=True)
    shutil.copy(clip, vdir / "media.mp4")
    monkeypatch.setenv("WATCH_STORE_DIR", str(root))
    return root


def _stored_bd(root):
    return json.loads((store.video_dir(root, KEY) / "breakdown" / "breakdown.json").read_text())


@needs_ffmpeg
def test_correct_production_is_applied_to_the_stored_breakdown(stored, capsys):
    assert cli.main(["breakdown", KEY, "--json"]) == 0
    n_shots = len(_stored_bd(stored)["shots"])
    capsys.readouterr()
    assert cli.main(["correct", KEY, "--at", "1.0", "--production", "camera", "--note", "shot on a phone",
                     "--json"]) == 0
    res = json.loads(capsys.readouterr().out)
    assert res["status"] == "applied" and len(res["corrections"]) == 1
    bd = _stored_bd(stored)
    assert len(bd["shots"]) == n_shots
    p = bd["shots"][0]["production"]
    assert p["label"] == "camera" and p["evidence"]["method"] == "correction" and "measured" in p
    assert bd["corrections"]["production"] == 1
    assert bd["production"]["share"].get("camera", 0) > 0
    md = (store.video_dir(stored, KEY) / "breakdown" / "breakdown.md").read_text()
    assert "corrected by the user" in md
    # a fresh measurement re-applies it
    assert cli.main(["breakdown", KEY, "--json"]) == 0
    assert _stored_bd(stored)["shots"][0]["production"]["label"] == "camera"


@needs_ffmpeg
def test_cut_corrections_need_a_rebuild_then_stick(stored, capsys):
    assert cli.main(["breakdown", KEY, "--json"]) == 0
    before = [s["start"] for s in _stored_bd(stored)["shots"]]
    assert 2.0 in [round(x, 2) for x in before]
    capsys.readouterr()
    assert cli.main(["correct", KEY, "--add-cut", "3.0", "--json"]) == 0
    assert "next breakdown" in json.loads(capsys.readouterr().out)["status"]
    assert [s["start"] for s in _stored_bd(stored)["shots"]] == before  # not re-measured yet
    assert cli.main(["correct", KEY, "--remove-cut", "2.0", "--rebuild", "--json"]) == 0
    starts = [round(s["start"], 2) for s in _stored_bd(stored)["shots"]]
    assert 3.0 in starts and 2.0 not in starts
    # undo the removal: 2.0 comes back on the next rebuild
    capsys.readouterr()
    assert cli.main(["correct", KEY, "--undo", "2", "--rebuild", "--json"]) == 0
    starts = [round(s["start"], 2) for s in _stored_bd(stored)["shots"]]
    assert 2.0 in starts and 3.0 in starts


@needs_ffmpeg
def test_export_labels_only_lists_verified_cuts(stored, tmp_path, capsys):
    assert cli.main(["breakdown", KEY, "--json"]) == 0
    assert cli.main(["correct", KEY, "--at", "0.5", "--production", "synthetic"]) == 0
    out = tmp_path / "labels.json"
    assert cli.main(["correct", "--export-labels", str(out)]) == 0
    data = json.loads(out.read_text())
    assert len(data["videos"]) == 1 and "cuts" not in data["videos"][0]
    assert data["videos"][0]["production"][0][2] == "synthetic"
    assert Path(data["videos"][0]["path"]).is_file()
    assert cli.main(["correct", KEY, "--cuts-verified"]) == 0
    assert cli.main(["correct", "--export-labels", str(out)]) == 0
    assert json.loads(out.read_text())["videos"][0]["cuts"]


def test_correct_usage_errors(stored, capsys):
    assert cli.main(["correct"]) == 2
    assert cli.main(["correct", "no-such-video", "--at", "1", "--production", "camera"]) == cli.EXIT_NOT_FOUND
