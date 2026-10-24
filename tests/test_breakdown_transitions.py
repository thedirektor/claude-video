import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from breakdown_fixtures import make_reference  # noqa: E402
from breakdown import camera as C  # noqa: E402
from breakdown import shots as S  # noqa: E402
from breakdown import transitions as T  # noqa: E402


@pytest.fixture(scope="module")
def ref(tmp_path_factory):
    p = tmp_path_factory.mktemp("trans") / "ref.mp4"
    truth = make_reference(p, with_audio=False)
    series = S.frame_series(str(p), truth["fps"])
    shots = S.detect_shots(series, truth["duration"])
    return p, truth, series, shots


def test_cut_and_dissolve_and_fade(ref):
    _, truth, series, shots = ref
    out = T.classify(series, shots)
    kinds = {e["kind"] for e in out}
    assert "not determined" not in kinds
    cut = [e for e in out if e["kind"] == "cut"]
    assert len(cut) == 1 and abs(cut[0]["t"] - 2.0) <= 1 / truth["fps"] + 1e-6
    dis = [e for e in out if e["kind"] == "dissolve"]
    assert len(dis) == 1 and truth["dissolve"]["start"] <= dis[0]["t"] <= truth["dissolve"]["end"]
    assert 0.5 <= dis[0]["duration"] <= 1.5
    fade = [e for e in out if e["kind"] == "fade_to_black"]
    assert len(fade) == 1
    assert truth["fade_out"]["start"] - 0.2 <= fade[0]["t"] <= truth["fade_out"]["end"]
    assert 0.6 <= fade[0]["duration"] <= 1.2
    assert not [e for e in out if e["kind"].startswith("fade_from")]
    for e in out:
        assert e["evidence"]["method"] == "measured" and 0 <= e["evidence"]["confidence"] <= 1


def test_fade_from_black_at_start():
    t = [i / 25 for i in range(50)]
    y = [16 + min(i, 20) * 4.0 for i in range(50)]
    out = T.classify({"t": t, "scene": [0.0] * 50, "yavg": y, "thumb": []}, [])
    assert [e["kind"] for e in out] == ["fade_from_black"]


def test_static_camera(ref):
    p, _, _, shots = ref
    r = C.camera_motion(str(p), shots[0])
    assert r["type"] == "static" and 0 < r["confidence"] <= 1
    assert r["evidence"]["method"] == "measured"


def test_pan_detected(tmp_path):
    p = tmp_path / "pan.mp4"
    res = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                          "testsrc=s=1280x360:r=25:d=3", "-vf", "crop=640:360:x='t*200':y=0,format=yuv420p",
                          "-c:v", "libx264", "-crf", "16", str(p)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    r = C.camera_motion(str(p), {"start": 0.0, "end": 3.0})
    assert r["type"] == "pan_right", r


def test_without_cv2(ref, monkeypatch):
    p, _, _, shots = ref
    monkeypatch.setitem(sys.modules, "cv2", None)
    r = C.camera_motion(str(p), shots[0])
    assert r["type"] == "not determined" and r["confidence"] == 0
    assert any("opencv" in n for n in r["notes"])


def _ff(*a):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *a], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


ENC = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"]


@pytest.fixture(scope="module")
def still(tmp_path_factory):
    d = tmp_path_factory.mktemp("cam")
    _ff("-f", "lavfi", "-i", "mandelbrot=s=1600x900:start_scale=0.5,trim=end_frame=1", "-frames:v", "1", str(d / "s.png"))
    _ff("-i", str(d / "s.png"), "-vf", "drawbox=x=800:y=0:w=800:h=900:c=gray:t=fill", "-frames:v", "1", str(d / "half.png"))
    return d


def _cam(d, name, vf, img="s.png"):
    p = d / f"{name}.mp4"
    _ff("-loop", "1", "-framerate", "25", "-t", "3", "-i", str(d / img), "-vf", vf, *ENC, str(p))
    return C.camera_motion(str(p), {"start": 0.0, "end": 3.0})


def test_tilt_up(still):
    assert _cam(still, "tu", "crop=640:360:0:'360-t*100',scale=640:360")["type"] == "tilt_up"


def test_zoom_in(still):
    r = _cam(still, "zi", "zoompan=z='1+0.004*on':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=640x360:fps=25")
    assert r["type"] == "zoom_in"


def test_handheld(still):
    r = _cam(still, "hh", "crop=640:360:'320+30*sin(t*7.3)+20*sin(t*13.1+1)':'180+25*sin(t*9.7+2)+15*sin(t*17.3)',scale=640:360")
    assert r["type"] == "handheld" and r["confidence"] < 0.8


def test_slow_one_sided_pan_is_not_zoom(still):
    r = _cam(still, "sp", "crop=640:360:'400+t*60':200", img="half.png")
    assert r["type"] == "pan_right"


def _series(p, dur=5.0):
    s = S.frame_series(str(p), 25)
    return s, T.classify(s, S.detect_shots(s, dur))


def test_fade_in_and_out(tmp_path):
    p = tmp_path / "io.mp4"
    _ff("-f", "lavfi", "-i", "testsrc2=s=320x180:r=25:d=5", "-vf", "fade=t=in:st=0:d=1,fade=t=out:st=4:d=1", *ENC, str(p))
    _, out = _series(p)
    assert [e["kind"] for e in out] == ["fade_from_black", "fade_to_black"]


def test_fade_to_white(tmp_path):
    p = tmp_path / "w.mp4"
    _ff("-f", "lavfi", "-i", "testsrc2=s=320x180:r=25:d=5", "-vf", "fade=t=out:st=4:d=1:c=white", *ENC, str(p))
    s, out = _series(p)
    assert [e["kind"] for e in out] == ["fade_to_white"]
    assert len(S.detect_shots(s, 5.0)) == 1


def test_dip_through_black(tmp_path):
    a, b, p = tmp_path / "a.mp4", tmp_path / "b.mp4", tmp_path / "x.mp4"
    _ff("-f", "lavfi", "-i", "testsrc2=s=320x180:r=25:d=3", *ENC, str(a))
    _ff("-f", "lavfi", "-i", "testsrc=s=320x180:r=25:d=3", *ENC, str(b))
    _ff("-i", str(a), "-i", str(b), "-filter_complex", "[0][1]xfade=transition=fadeblack:duration=1:offset=2", *ENC, str(p))
    _, out = _series(p)
    assert [e["kind"] for e in out] == ["fade_to_black", "fade_from_black"]
    assert out[0]["t"] < out[1]["t"]


def test_hold_then_fade_in_single_entry(tmp_path):
    p = tmp_path / "h.mp4"
    _ff("-f", "lavfi", "-i", "testsrc2=s=320x180:r=25:d=5", "-vf", "fade=t=in:st=1:d=1", *ENC, str(p))
    _, out = _series(p)
    assert [e["kind"] for e in out] == ["fade_from_black"]


def test_fade_out_then_hold_single_entry(tmp_path):
    p = tmp_path / "o.mp4"
    _ff("-f", "lavfi", "-i", "testsrc2=s=320x180:r=25:d=5", "-vf", "fade=t=out:st=3:d=1", *ENC, str(p))
    _, out = _series(p)
    assert [e["kind"] for e in out] == ["fade_to_black"]


def test_interior_hold_fade_single_entry(tmp_path):
    a, b, p = tmp_path / "a.mp4", tmp_path / "b.mp4", tmp_path / "c.mp4"
    _ff("-f", "lavfi", "-i", "testsrc2=s=320x180:r=25:d=2", "-vf", "fade=t=out:st=1:d=1", *ENC, str(a))
    _ff("-f", "lavfi", "-i", "testsrc=s=320x180:r=25:d=3", "-vf", "fade=t=in:st=1:d=1", *ENC, str(b))
    _ff("-i", str(a), "-i", str(b), "-filter_complex", "[0][1]concat=n=2:v=1", *ENC, str(p))
    _, out = _series(p)
    assert [e["kind"] for e in out] == ["fade_to_black", "fade_from_black"]


def _concat(tmp_path, name, a_src, a_vf, b_src, b_vf):
    a, b, p = tmp_path / "a.mp4", tmp_path / "b.mp4", tmp_path / f"{name}.mp4"
    _ff("-f", "lavfi", "-i", a_src, "-vf", a_vf, *ENC, str(a))
    _ff("-f", "lavfi", "-i", b_src, "-vf", b_vf, *ENC, str(b))
    _ff("-i", str(a), "-i", str(b), "-filter_complex", "[0][1]concat=n=2:v=1", *ENC, str(p))
    return p


def test_fade_in_then_cut(tmp_path):
    p = _concat(tmp_path, "fc", "testsrc2=s=320x180:r=25:d=2", "fade=t=in:st=1:d=1",
                "mandelbrot=s=320x180:r=25,trim=duration=3", "null")
    _, out = _series(p)
    assert [e["kind"] for e in out] == ["fade_from_black", "cut"]
    assert abs(out[1]["t"] - 2.0) <= 0.04 + 1e-6
    assert out[0]["evidence"]["t_end"] <= 2.0 + 1e-6 and out[0]["evidence"]["t_start"] >= 0.9


def test_cut_then_fade_out(tmp_path):
    p = _concat(tmp_path, "cf", "mandelbrot=s=320x180:r=25,trim=duration=2", "null",
                "testsrc2=s=320x180:r=25:d=2", "fade=t=out:st=0:d=1")
    _, out = _series(p, 4.0)
    assert [e["kind"] for e in out] == ["cut", "fade_to_black"]
    assert abs(out[0]["t"] - 2.0) <= 0.04 + 1e-6
    assert out[1]["evidence"]["t_start"] >= 2.0 - 1e-6


# ------------------------------------------------------------------ 0.7.1 minors

@pytest.mark.parametrize("name,expect_to,expect_from", [("fadeblack", "black", "black"), ("fadewhite", "white", "white")])
def test_xfade_dip_halves_match_the_luma_ramp(tmp_path, name, expect_to, expect_from):
    """xfade's dip fades out over the first 20% (0.2 s) and in over the rest (0.8 s); the halves were
    measured 0.16 s and 0.64 s because the ramp end was taken where luma entered the black / white band."""
    a, b, p = tmp_path / "a.mp4", tmp_path / "b.mp4", tmp_path / "x.mp4"
    _ff("-f", "lavfi", "-i", "testsrc2=s=320x180:r=25:d=3", *ENC, str(a))
    _ff("-f", "lavfi", "-i", "testsrc=s=320x180:r=25:d=3", *ENC, str(b))
    _ff("-i", str(a), "-i", str(b), "-filter_complex", f"[0][1]xfade=transition={name}:duration=1:offset=2", *ENC, str(p))
    _, out = _series(p)
    kinds = [e["kind"] for e in out]
    assert kinds == [f"fade_to_{expect_to}", f"fade_from_{expect_from}"]
    to, frm = out
    assert to["duration"] == pytest.approx(0.2, abs=0.05)
    assert frm["duration"] == pytest.approx(0.8, abs=0.05)
    assert to["evidence"]["t_start"] == pytest.approx(2.0, abs=0.05) and frm["evidence"]["t_end"] == pytest.approx(3.0, abs=0.05)


@pytest.fixture(scope="module")
def soft(tmp_path_factory):
    d = tmp_path_factory.mktemp("soft")
    _ff("-f", "lavfi", "-i", "mandelbrot=s=1920x180:r=25:start_scale=0.05:start_x=-0.74:start_y=0.1", "-frames:v", "1", str(d / "s1.png"))
    _ff("-f", "lavfi", "-i", "mandelbrot=s=1920x180:r=25:start_scale=0.4:start_x=-0.6:start_y=0.0", "-frames:v", "1", str(d / "s2.png"))
    return d


@pytest.mark.parametrize("travel,secs", [(300, 0.5), (500, 0.5), (300, 1.0), (500, 1.0)])
def test_whip_pan_between_held_framings_is_never_a_dissolve(soft, travel, secs):
    """A smooth pan over a soft scene fits a straight blend, so it used to come out as a dissolve at
    0.9+ confidence. The in-between frames are one picture sliding, not a cross-fade."""
    ss = f"(3*pow((t-1.5)/{secs},2)-2*pow((t-1.5)/{secs},3))"
    x = f"if(lt(t,1.5),0,if(lt(t,{1.5 + secs}),{travel}*{ss},{travel}))"
    p = soft / f"pan{travel}_{secs}.mp4"
    _ff("-loop", "1", "-framerate", "25", "-i", str(soft / "s1.png"), "-t", "4", "-vf", f"crop=320:180:x='{x}':y=0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "25", str(p))
    s, out = _series(p, 4.0)
    assert out, "the pan should at least register as a boundary"
    for e in out:
        assert e["kind"] != "dissolve", e
        if e["kind"] == T.NOT_DETERMINED:
            assert e["evidence"]["confidence"] == 0


@pytest.mark.parametrize("xa,xb", [(0, 0), (0, 900), (300, 310)])
def test_real_dissolve_of_soft_stills_is_still_a_dissolve(soft, xa, xb):
    p = soft / f"dis{xa}_{xb}.mp4"
    _ff("-loop", "1", "-framerate", "25", "-i", str(soft / "s1.png"), "-loop", "1", "-framerate", "25", "-i", str(soft / "s2.png"),
        "-t", "4", "-filter_complex", f"[0]crop=320:180:{xa}:0[a];[1]crop=320:180:{xb}:0[b];[a][b]xfade=transition=fade:duration=1:offset=1.5",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "25", str(p))
    _, out = _series(p, 4.0)
    assert [e["kind"] for e in out] == ["dissolve"]


def _synthetic_fade_with_cut(real_cut: bool):
    """40 frames at 25 fps: picture A at full level, a 20-frame ramp to black that (a) keeps scaling
    the same picture or (b) hands over to a different picture at frame 30 on the same luma ramp, then a
    black hold."""
    import random
    rnd = random.Random(7)
    pa = [rnd.randrange(60, 200) for _ in range(16 * 9 * 3)]
    pb = [rnd.randrange(60, 200) for _ in range(16 * 9 * 3)]
    n, thumbs = 50, []
    for i in range(n):
        scale = 1.0 if i < 20 else max(0.0, 1.0 - (i - 20) * 0.05) if i < 40 else 0.0
        pic = pb if (real_cut and 30 <= i < 40) else pa
        thumbs.append(bytes(int(v * scale) for v in pic))
    y = [sum(th) / len(th) for th in thumbs]
    return {"t": [i / 25 for i in range(n)], "scene": [0.0] * n, "yavg": y, "thumb": thumbs}


def test_on_ramp_does_not_absorb_a_real_cut_between_matched_ramps():
    series = _synthetic_fade_with_cut(real_cut=True)
    shots = [{"index": 1, "boundary": {"kind": "cut", "t": series["t"][30],
                                       "evidence": {"t_start": series["t"][29], "t_end": series["t"][30], "confidence": 0.9}}}]
    kinds = [e["kind"] for e in T.classify(series, shots)]
    assert "cut" in kinds, kinds


def test_on_ramp_still_absorbs_a_fade_step():
    series = _synthetic_fade_with_cut(real_cut=False)
    shots = [{"index": 1, "boundary": {"kind": "cut", "t": series["t"][30],
                                       "evidence": {"t_start": series["t"][29], "t_end": series["t"][30], "confidence": 0.9}}}]
    kinds = [e["kind"] for e in T.classify(series, shots)]
    assert "cut" not in kinds and "fade_to_black" in kinds, kinds
