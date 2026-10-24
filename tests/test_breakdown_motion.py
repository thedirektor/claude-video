from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from breakdown import easing  # noqa: E402


def _samples(trans, ease, n=30, noise=0.0, seed=1):
    rnd = random.Random(seed)
    f = easing.CURVES[trans][ease]
    t = [i / (n - 1) for i in range(n)]
    return t, [f(x) + rnd.gauss(0, noise) for x in t]


def test_curves_endpoints():
    for trans in easing.TRANS:
        for ease in easing.EASES:
            f = easing.CURVES[trans][ease]
            assert abs(f(0.0)) < 1e-9, (trans, ease)
            assert abs(f(1.0) - 1.0) < 1e-9, (trans, ease)


def test_curves_known_values():
    c = easing.CURVES
    assert c["CUBIC"]["OUT"](0.5) == pytest.approx(0.875)
    assert c["QUAD"]["IN"](0.5) == pytest.approx(0.25)
    assert c["QUAD"]["IN_OUT"](0.25) == pytest.approx(0.125)
    assert c["BACK"]["OUT"](0.5) > 1.0 or c["BACK"]["OUT"](0.8) > 1.0  # overshoots
    assert c["QUAD"]["OUT_IN"](0.25) == pytest.approx(c["QUAD"]["OUT"](0.5) / 2)
    assert c["QUAD"]["OUT_IN"](0.75) == pytest.approx(c["QUAD"]["IN"](0.5) / 2 + 0.5)
    assert c["BACK"]["IN"](0.5) < 0.0  # pulls back first


@pytest.mark.parametrize("trans,ease", [("CUBIC", "OUT"), ("QUAD", "IN"), ("SINE", "IN_OUT"),
                                        ("QUINT", "OUT"), ("EXPO", "IN"), ("CIRC", "OUT"),
                                        ("BACK", "OUT"), ("BOUNCE", "OUT"), ("QUART", "IN_OUT")])
def test_fit_recovers_known_pairs(trans, ease):
    t, v = _samples(trans, ease)
    r = easing.fit(t, v)
    assert r["godot"] == {"trans": trans, "ease": ease}
    assert r["error_rms"] < 0.02
    assert len(r["bezier"]) == 4 or trans == "BOUNCE"  # a bounce is not a cubic-bezier
    assert {"time", "value", "inTangent", "outTangent"} <= set(r["unity_keys"][0])


def test_fit_recovers_cubic_out():
    t, v = _samples("CUBIC", "OUT")
    r = easing.fit([x * 0.5 for x in t], [x * 600 for x in v])  # arbitrary units
    assert r["godot"] == {"trans": "CUBIC", "ease": "OUT"}
    assert r["error_rms"] < 0.02


def test_fit_indistinguishable_pair_is_flagged():
    t, v = _samples("CUBIC", "IN_OUT", noise=0.03, seed=3)
    r = easing.fit(t, v)
    assert r["godot"] != easing.NOT_DETERMINED
    assert r["runner_up"] is not None
    assert r["ambiguous"] and r["confidence"] < 0.6


def test_fit_garbage_not_forced():
    rnd = random.Random(7)
    t = [i / 29 for i in range(30)]
    v = [rnd.random() for _ in t]
    v[0], v[-1] = 0.0, 1.0
    r = easing.fit(t, v)
    assert r["godot"] == easing.NOT_DETERMINED and r["confidence"] == 0.0


def test_fit_too_few_samples():
    assert easing.fit([0, 1, 2], [0, 1, 2])["godot"] == easing.NOT_DETERMINED


# ---- tracking / phases / region / snippets (need ffmpeg + cv2) -------------------------

pytestmark_media = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg needed")

from breakdown import motion  # noqa: E402
from breakdown_fixtures import make_motion_clip, make_reference  # noqa: E402
from memory import index, store  # noqa: E402
from memory_fakes import KEY, fake_run, seg  # noqa: E402


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    d = tmp_path_factory.mktemp("mo")
    mp = d / "motion.mp4"
    truth = make_motion_clip(mp)
    rp = d / "ref.mp4"
    rtruth = make_reference(rp, with_audio=False)
    return {"motion": (mp, truth), "ref": (rp, rtruth)}


def _store(conn, tmp_path, clip, duration):
    root = tmp_path / "st"
    index.record_run(conn, fake_run(title="Motion clip", duration=duration, segments=[seg(0, 1, "hi")]))
    vdir = store.video_dir(root, KEY)
    vdir.mkdir(parents=True)
    shutil.copy(clip, vdir / "media.mp4")
    return root


@pytest.fixture
def mstored(conn, tmp_path, clips):
    return _store(conn, tmp_path, clips["motion"][0], 1.2)


@pytest.fixture
def rstored(conn, tmp_path, clips):
    return _store(conn, tmp_path, clips["ref"][0], 7.0)


@pytestmark_media
def test_track_and_phase(conn, mstored, clips):
    _, truth = clips["motion"]
    m = motion.analyze(conn, KEY, 0.0, 1.2, store_dir=mstored)
    assert m["schema"] == "watch.motion/1"
    assert m["region"]["method"] == "auto_motion"
    assert m["segment"]["fps"] == pytest.approx(60, abs=0.5)
    actions = [p for p in m["phases"] if p["name"] == "action"]
    assert len(actions) == 1, m
    a = actions[0]
    two = 2 / 60
    assert a["t_start"] == pytest.approx(truth["start"], abs=two)
    assert a["t_end"] == pytest.approx(truth["end"], abs=two)
    assert a["displacement"]["px"] == pytest.approx(600, abs=10)
    assert a["curve"]["godot"] == {"trans": truth["trans"], "ease": truth["ease"]}
    assert a["evidence"]["method"] == "measured"
    assert [p["name"] for p in m["phases"]][-1] == "hold"
    assert len(m["track"]) >= 70 and {"t", "x", "y", "w", "h"} <= set(m["track"][0])


@pytestmark_media
def test_given_region_and_units(conn, mstored):
    m = motion.analyze(conn, KEY, 0.0, 1.2, region=[0.0, 0.40, 0.2, 0.2], px_per_unit=100, store_dir=mstored)
    assert m["region"]["method"] == "given"
    a = next(p for p in m["phases"] if p["name"] == "action")
    assert a["displacement"]["px"] == pytest.approx(600, abs=10)
    assert a["displacement"]["units"] == pytest.approx(6.0, abs=0.1)
    assert a["displacement"]["x"] == pytest.approx(600 / 1280, abs=0.01)


@pytestmark_media
def test_region_text_resolution(conn, rstored, clips):
    if not clips["ref"][1]["title"]["rendered"]:
        pytest.skip("no font for drawtext")
    m = motion.analyze(conn, KEY, 1.2, 2.8, region="SALE", store_dir=rstored)
    assert m["region"]["method"] == "ocr"
    x, y, w, h = m["region"]["box"]
    assert x < 0.15 and y < 0.2 and 0.1 < w < 0.6 and 0.05 < h < 0.3


@pytestmark_media
def test_region_unresolved(conn, rstored):
    m = motion.analyze(conn, KEY, 1.2, 2.8, region="dragon", store_dir=rstored)
    assert m["region"]["method"] == "whole_frame"
    assert m["region"]["box"] == [0.0, 0.0, 1.0, 1.0]
    assert "dragon" in m["region"]["note"]
    assert m["phases"] == []  # a static title card has no motion to report


@pytestmark_media
def test_no_motion_is_not_determined(conn, rstored):
    m = motion.analyze(conn, KEY, 0.1, 0.9, region=[0.1, 0.1, 0.5, 0.5], store_dir=rstored)
    assert m["phases"] == [] and m["notes"]  # flat template: nothing to follow, nothing invented


@pytestmark_media
def test_tracking_lost_not_forced(conn, tmp_path):
    import subprocess
    p = tmp_path / "vanish.mp4"
    graph = ("color=c=black:s=640x360:r=30:d=2[bg];color=c=white:s=60x60:r=30:d=0.6[sq];"
             "[bg][sq]overlay=x='20+500*t':y=150:enable='lt(t,0.6)',format=yuv420p[v]")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-filter_complex", graph, "-map", "[v]",
                    "-t", "2", str(p)], check=True)
    root = _store(conn, tmp_path, p, 2.0)
    m = motion.analyze(conn, KEY, 0.0, 2.0, region=[0.02, 0.38, 0.12, 0.22], store_dir=root)
    assert m["phases"] == []
    assert any("lost" in n for n in m["notes"])


def test_frames_only_not_determined(conn, tmp_path):
    index.record_run(conn, fake_run(title="Gone", duration=5.0, segments=[seg(0, 1, "hi")]))
    m = motion.analyze(conn, KEY, 0.0, 2.0, store_dir=tmp_path / "empty")
    assert m["phases"] == [] and m["notes"]


@pytestmark_media
def test_without_cv2(conn, mstored, monkeypatch):
    monkeypatch.setattr(motion, "_import_cv", lambda: (_ for _ in ()).throw(ImportError("no cv2")))
    m = motion.analyze(conn, KEY, 0.0, 1.2, store_dir=mstored)
    assert m["phases"] == [] and any("cv2" in n for n in m["notes"])


@pytestmark_media
def test_snippets_generic(conn, mstored):
    m = motion.analyze(conn, KEY, 0.0, 1.2, px_per_unit=100, store_dir=mstored)
    sn = motion.snippets(m)
    assert set(sn) == {"godot.gd", "unity.json", "table.md"}
    gd = sn["godot.gd"]
    assert "set_trans(Tween.TRANS_CUBIC)" in gd and "set_ease(Tween.EASE_OUT)" in gd
    assert "tween_property(node, \"position:x\", <to>" in gd
    u = json.loads(sn["unity.json"])
    assert u["keys"][0]["time"] == 0 and {"value", "inTangent", "outTangent"} <= set(u["keys"][0])
    assert "| action |" in sn["table.md"] and "cubic-bezier" in sn["table.md"]
    blob = " ".join(sn.values()).lower()
    for banned in ("swu", "feel_spec", "godotease", "/opt/", "cort3x"):
        assert banned not in blob


def test_snippets_undetermined_curve():
    m = {"segment": {"start": 0, "end": 1, "fps": 30}, "notes": ["x"], "phases": [
        {"name": "action", "t_start": 0.1, "t_end": 0.5, "duration": 0.4,
         "displacement": {"x": 0.1, "y": 0.0, "px": 128.0}, "curve": "not determined", "evidence": {}}]}
    sn = motion.snippets(m)
    assert "set_trans" not in sn["godot.gd"] and "not determined" in sn["godot.gd"]
    assert json.loads(sn["unity.json"])["keys"] == "not determined"


@pytestmark_media
def test_in_out_move_and_vertical(conn, tmp_path):
    import subprocess
    p = tmp_path / "sine.mp4"
    expr = "250*(0.5-0.5*cos(PI*clip((t-0.3)/0.6,0,1)))"
    graph = ("color=c=0x202020:s=640x360:r=50:d=1.5[bg];color=c=0xffd000:s=50x50:r=50:d=1.5[sq];"
             f"[bg][sq]overlay=x=300:y='20+{expr}':eval=frame,format=yuv420p[v]")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-filter_complex", graph, "-map", "[v]",
                    "-c:v", "libx264", "-crf", "14", "-t", "1.5", str(p)], check=True)
    root = _store(conn, tmp_path, p, 1.5)
    m = motion.analyze(conn, KEY, 0.0, 1.5, store_dir=root)
    a = next(ph for ph in m["phases"] if ph["name"] == "action")
    assert a["curve"]["godot"] == {"trans": "SINE", "ease": "IN_OUT"}
    assert a["displacement"]["px"] == pytest.approx(250, abs=8)
    assert abs(a["displacement"]["y"]) == pytest.approx(250 / 360, abs=0.03) and abs(a["displacement"]["x"]) < 0.02
    assert a["t_start"] == pytest.approx(0.3, abs=0.06) and a["t_end"] == pytest.approx(0.9, abs=0.06)
    assert "position:y" in motion.snippets(m)["godot.gd"]


@pytestmark_media
def test_cli_motion_writes_outputs(conn, mstored, monkeypatch, capsys):
    from breakdown import cli
    monkeypatch.setenv("WATCH_STORE_DIR", str(mstored))
    rc = cli.main(["motion", KEY, "--start", "0", "--end", "1.2", "--px-per-unit", "50", "--json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["schema"] == "watch.motion/1" and data["px_per_unit"] == 50
    out = store.video_dir(mstored, KEY) / "breakdown" / "motion-0-1.2"
    for name in ("motion.json", "godot.gd", "unity.json", "table.md"):
        assert (out / name).is_file(), name
    assert cli.main(["motion", KEY, "--start", "2", "--end", "1"]) == 2


@pytestmark_media
def test_anticipation_action_hold(conn, tmp_path):
    import subprocess
    p = tmp_path / "antic.mp4"
    expr = ("if(lt(t\\,0.4)\\,300-40*(0.5-0.5*cos(PI*clip((t-0.2)/0.2\\,0\\,1)))\\,"
            "260+300*(0.5-0.5*cos(PI*clip((t-0.4)/0.6\\,0\\,1))))")
    graph = ("color=c=0x202020:s=1000x300:r=50:d=1.6[bg];color=c=0xffd000:s=60x60:r=50:d=1.6[sq];"
             f"[bg][sq]overlay=x='{expr}':y=120:eval=frame,format=yuv420p[v]")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-filter_complex", graph, "-map", "[v]",
                    "-c:v", "libx264", "-crf", "14", "-t", "1.6", str(p)], check=True)
    root = _store(conn, tmp_path, p, 1.6)
    m = motion.analyze(conn, KEY, 0.0, 1.6, store_dir=root)
    names = [ph["name"] for ph in m["phases"]]
    assert names == ["anticipation", "action", "hold"], (names, m["notes"])
    ant, act = m["phases"][0], m["phases"][1]
    assert ant["displacement"]["x"] < 0 < act["displacement"]["x"]
    assert ant["t_start"] == pytest.approx(0.2, abs=0.05) and act["t_end"] == pytest.approx(1.0, abs=0.06)
    assert act["curve"]["godot"] == {"trans": "SINE", "ease": "IN_OUT"}


def _ov(conn, tmp_path, x_expr, d=1.6, size="1280x400", sq=60):
    import subprocess
    p = tmp_path / "ov.mp4"
    graph = (f"color=c=0x202020:s={size}:r=60:d={d}[bg];"
             f"color=c=0xffd000:s={sq}x{sq}:r=60:d={d},drawbox=x=10:y=10:w=20:h=20:c=red:t=fill[sq];"
             f"[bg][sq]overlay=x='{x_expr}':y=120:eval=frame,format=yuv420p[v]")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-filter_complex", graph, "-map", "[v]",
                    "-c:v", "libx264", "-crf", "14", "-t", str(d), str(p)], check=True)
    return _store(conn, tmp_path, p, d)


def _u(t0, t1):
    return f"clip((t-{t0})/{t1 - t0}\\,0\\,1)"


@pytestmark_media
def test_anticipation_curve_not_cut_short(conn, tmp_path):
    cub = f"(1-pow(1-{_u(0.2, 0.4)}\\,3))"
    root = _ov(conn, tmp_path, f"300-40*{cub}+340*(1-pow(1-{_u(0.4, 0.9)}\\,4))")
    m = motion.analyze(conn, KEY, 0.0, 1.6, store_dir=root)
    names = [p["name"] for p in m["phases"]]
    assert names == ["anticipation", "action", "hold"], (names, m["notes"])  # no hold inside the motion
    ant = m["phases"][0]
    assert ant["t_end"] == pytest.approx(0.4, abs=0.04)
    cands = [ant["curve"]["godot"]] + ([ant["curve"]["runner_up"]] if ant["curve"].get("runner_up") else [])
    assert any((c["trans"], c.get("ease")) == ("CUBIC", "OUT") for c in cands)
    assert ant["curve"]["godot"] == {"trans": "CUBIC", "ease": "OUT"}


@pytestmark_media
def test_overshoot_ease_is_one_action(conn, tmp_path):
    u = _u(0.2, 0.7)
    back = f"(1+2.70158*pow({u}-1\\,3)+1.70158*pow({u}-1\\,2))"
    root = _ov(conn, tmp_path, f"100+600*{back}", d=1.2, size="1280x400", sq=80)
    m = motion.analyze(conn, KEY, 0.0, 1.2, store_dir=root)
    acts = [p for p in m["phases"] if p["name"] == "action"]
    assert [p["name"] for p in m["phases"]] == ["action", "hold"], m["phases"]
    assert acts[0]["curve"]["godot"] == {"trans": "BACK", "ease": "OUT"}
    assert acts[0]["displacement"]["px"] == pytest.approx(600, abs=10)
    gd = motion.snippets(m)["godot.gd"]
    assert gd.count("tween_property") == 1 and "TRANS_BACK" in gd and "EASE_OUT" in gd


def _cub(t0, t1):
    return f"(1-pow(1-{_u(t0, t1)}\\,3))"


@pytestmark_media
@pytest.mark.parametrize("ret", [
    f"-60*(0.5-0.5*cos(PI*{_u(0.6, 0.8)}))",  # immediate sine return of 10%
    f"-40*pow({_u(0.633, 0.8)}\\,2)",  # 2-frame settle, then a quad return
])
def test_move_plus_separate_return_is_not_a_fake_back(conn, tmp_path, ret):
    root = _ov(conn, tmp_path, f"100+600*{_cub(0.2, 0.6)}{ret}", d=1.4, sq=80)
    m = motion.analyze(conn, KEY, 0.0, 1.4, store_dir=root)
    names = [p["name"] for p in m["phases"]]
    assert names[:2] == ["action", "recovery"], (names, m["notes"])
    act = m["phases"][0]["curve"]
    assert act["godot"] == {"trans": "CUBIC", "ease": "OUT"}
    for ph in m["phases"]:
        c = ph["curve"]
        assert not (isinstance(c, dict) and isinstance(c.get("godot"), dict) and c["godot"]["trans"] == "BACK")
    holds = [p for p in m["phases"] if p["name"] == "hold"]
    for a, b in zip(m["phases"], m["phases"][1:]):
        assert a["t_end"] <= b["t_start"] + 1e-6 or a["name"] != "hold"
    assert holds
