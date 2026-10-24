import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from breakdown_fixtures import _ffmpeg, make_reference  # noqa: E402
from breakdown import shots as S  # noqa: E402


@pytest.fixture(scope="module")
def ref(tmp_path_factory):
    p = tmp_path_factory.mktemp("shots") / "ref.mp4"
    truth = make_reference(p, with_audio=False)
    series = S.frame_series(str(p), truth["fps"])
    return truth, series, S.detect_shots(series, truth["duration"])


def test_hard_cut_frame_accurate(ref):
    truth, _, shots = ref
    cuts = [s["boundary"] for s in shots if s["boundary"] and s["boundary"]["kind"] == "cut"]
    assert len(cuts) == 1
    assert abs(cuts[0]["t"] - truth["cuts"][0]["t"]) <= 1 / truth["fps"] + 1e-6


def test_gradual_boundary_found(ref):
    truth, _, shots = ref
    grads = [s["boundary"] for s in shots if s["boundary"] and s["boundary"]["kind"] == "gradual"]
    assert len(grads) == 1
    assert truth["dissolve"]["start"] <= grads[0]["t"] <= truth["dissolve"]["end"]
    assert grads[0]["frames"] >= 3


def test_fade_end(ref):
    truth, _, shots = ref
    assert abs(shots[-1]["end"] - truth["duration"]) <= 1 / truth["fps"] + 1e-6


def test_rhythm_stats(ref):
    truth, _, shots = ref
    assert len(shots) in (3, 4)
    r = S.rhythm(shots)
    assert r["count"] == len(shots)
    assert r["mean"] == pytest.approx(truth["duration"] / r["count"])
    assert 0 <= r["evidence"]["confidence"] <= 1 and r["evidence"]["method"] == "measured"
    for s in shots:
        assert s["evidence"]["method"] == "measured"


def test_single_shot_clip(tmp_path):
    p = tmp_path / "one.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "color=c=0x336699:s=320x180:r=25:d=3", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(p)],
                   check=True)
    series = S.frame_series(str(p), 25)
    shots = S.detect_shots(series, 3.0)
    assert len(shots) == 1 and shots[0]["boundary"] is None
    r = S.rhythm(shots)
    assert r["count"] == 1
    assert r["fastest_window"]["t_start"] == 0.0 and r["fastest_window"]["t_end"] == 3.0
    assert r["fastest_window"]["mean"] == pytest.approx(3.0)


def test_unknown_fps_not_determined():
    series = S.frame_series("/nonexistent", 0.0)
    assert series == {"t": [], "scene": [], "yavg": [], "thumb": []}
    assert S.detect_shots(series, 3.0) == []
    assert S.rhythm([])["mean"] == "not determined"


def _ff(*a):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *a], check=True)


_ENC = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "25"]


def _shots(path, dur):
    return S.detect_shots(S.frame_series(str(path), 25), dur)


def test_moving_shot_no_boundary(tmp_path):
    a, b = tmp_path / "anim.mp4", tmp_path / "pan.mp4"
    _ff("-f", "lavfi", "-i", "testsrc2=s=320x180:r=25:d=4", *_ENC, str(a))
    _ff("-f", "lavfi", "-i", "testsrc2=s=1280x180:r=25:d=4", "-vf", "crop=320:180:'t*300':0", *_ENC, str(b))
    assert len(_shots(a, 4.0)) == 1
    assert len(_shots(b, 4.0)) == 1


def test_flash_frame_not_a_cut(tmp_path):
    p = tmp_path / "flash.mp4"
    _ff("-f", "lavfi", "-i", "color=c=gray:s=320x180:r=25:d=4",
        "-vf", "drawbox=c=white:t=fill:enable='eq(n,50)'", *_ENC, str(p))
    assert len(_shots(p, 4.0)) == 1


def test_equal_luma_dissolve(tmp_path):
    p = tmp_path / "diss.mp4"
    _ff("-f", "lavfi", "-i", "color=c=0x808080:s=320x180:r=25:d=4",
        "-f", "lavfi", "-i", "testsrc2=s=320x180:r=25:d=4",
        "-filter_complex", "[0][1]xfade=transition=fade:duration=1:offset=1.5", *_ENC, str(p))
    shots = _shots(p, 4.5)
    grads = [s["boundary"] for s in shots if s["boundary"]]
    assert len(shots) == 2 and grads[0]["kind"] == "gradual"
    assert 1.5 <= grads[0]["t"] <= 2.5


def test_rhythm_slowest_window():
    mk = lambda i, a, b: {"index": i, "start": a, "end": b, "duration": b - a,
                          "evidence": {"t_start": a, "t_end": b, "method": "measured", "confidence": 0.9}}
    shots = [mk(0, 0, 1), mk(1, 1, 2), mk(2, 2, 3), mk(3, 3, 8), mk(4, 8, 13)]
    r = S.rhythm(shots)
    assert r["fastest_window"] == {"t_start": 0, "t_end": 3, "mean": 1.0}
    assert r["slowest_window"]["t_start"] == 2 and r["slowest_window"]["t_end"] == 13
    assert r["slowest_window"]["mean"] == pytest.approx(11 / 3)


def test_cuts_found_in_small_picture_with_big_bars(tmp_path):
    """Real-media e2e regression: a 16:9 edit inside a 9:16 phone frame (≈30% of the pixels) lost
    almost every cut because the black bars diluted the scene score."""
    src = tmp_path / "ref.mp4"
    gt = make_reference(src)
    tall = tmp_path / "tall.mp4"
    _ffmpeg(["-i", str(src), "-vf", "scale=720:-2,pad=720:1280:0:(oh-ih)/2", "-c:a", "copy",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", str(tall)])
    assert S.active_crop(str(tall)) is not None
    series = S.frame_series(str(tall), 25.0)
    found = S.detect_shots(series, gt["duration"])
    kinds = [(round(s["boundary"]["t"], 2), s["boundary"]["kind"]) for s in found[1:]]
    assert any(k == "cut" and abs(t - 2.0) <= 0.04 for t, k in kinds), kinds
    assert any(k == "gradual" and 4.0 <= t <= 5.0 for t, k in kinds), kinds


def test_no_crop_on_full_frame(tmp_path):
    src = tmp_path / "ref.mp4"
    make_reference(src)
    assert S.active_crop(str(src)) is None


# ------------------------------------------------------------------ 0.7.1 minors

def _fake_run(monkeypatch, *, fail_signalstats=False, log=None):
    """Route shots' ffmpeg calls through a recorder; optionally fail the signalstats pass."""
    real = subprocess.run

    def run(cmd, *a, **k):
        if log is not None:
            log.append(list(cmd))
        if fail_signalstats and "signalstats" in " ".join(map(str, cmd)):
            out = "".join(f"frame:{i} pts:{i} pts_time:{i / 25}\nlavfi.scene_score=0.001\n" for i in range(5))
            return subprocess.CompletedProcess(cmd, 1, stdout=out, stderr="decode error")
        return real(cmd, *a, **k)

    monkeypatch.setattr(S.subprocess, "run", run)


def test_frame_series_ffmpeg_failure_is_empty(tmp_path, monkeypatch):
    """A decode failure used to leave a truncated series, so the last shot stretched to the duration."""
    p = tmp_path / "one.mp4"
    _ff("-f", "lavfi", "-i", "color=c=0x336699:s=320x180:r=25:d=3", *_ENC, str(p))
    _fake_run(monkeypatch, fail_signalstats=True)
    series = S.frame_series(str(p), 25)
    assert series["t"] == [] and S.detect_shots(series, 3.0) == []


def test_frame_series_uses_fps_mode_not_deprecated_vsync(tmp_path, monkeypatch):
    p = tmp_path / "one.mp4"
    _ff("-f", "lavfi", "-i", "color=c=0x336699:s=320x180:r=25:d=2", *_ENC, str(p))
    log: list = []
    _fake_run(monkeypatch, log=log)
    S._PASSTHROUGH = None
    assert S.frame_series(str(p), 25)["t"]
    flat = [" ".join(map(str, c)) for c in log if "signalstats" in " ".join(map(str, c))]
    assert flat and all("-vsync" not in f and "-fps_mode passthrough" in f for f in flat)


def test_thumbnails_are_bytes_slices(ref):
    _, series, _ = ref
    assert series["thumb"] and all(isinstance(th, bytes) and len(th) == 16 * 9 * 3 for th in series["thumb"])


def test_frame_series_docstring_describes_both_passes():
    doc = S.frame_series.__doc__
    assert "thumb" in doc and "second pass" in doc


def test_empty_rhythm_evidence_is_not_a_claim():
    ev = S.rhythm([])["evidence"]
    assert ev["confidence"] == 0 and ev["method"] in ("measured", "ocr", "transcript", "interpretation")


def test_single_shot_confidence_is_capped(tmp_path):
    p = tmp_path / "one.mp4"
    _ff("-f", "lavfi", "-i", "testsrc2=s=320x180:r=25:d=3", *_ENC, str(p))
    shots = _shots(p, 3.0)
    assert len(shots) == 1 and shots[0]["evidence"]["confidence"] <= 0.6


@pytest.mark.parametrize("color", ["red", "white", "0x2040c0"])
@pytest.mark.parametrize("where", ["out", "in"])
def test_fade_at_clip_edge_is_not_a_gradual_boundary(tmp_path, color, where):
    """The edge-fade rule only knew black: a fade to/from white or any single colour became a
    gradual shot boundary (and a confident "dissolve")."""
    p = tmp_path / "edge.mp4"
    vf = f"fade=t=out:st=4:d=1:c={color}" if where == "out" else f"fade=t=in:st=0:d=1:c={color}"
    _ff("-f", "lavfi", "-i", "testsrc2=s=320x180:r=25:d=5", "-vf", vf, *_ENC, str(p))
    assert len(_shots(p, 5.0)) == 1


def test_dissolve_window_is_refined_on_the_blend_ramp(ref):
    """Windows come from a few quantised lengths: 3.88-4.96 for a dissolve that truly spans 4.0-5.0."""
    truth, _, shots = ref
    g = [s["boundary"] for s in shots if s["boundary"] and s["boundary"]["kind"] == "gradual"][0]
    assert g["evidence"]["t_start"] == pytest.approx(truth["dissolve"]["start"], abs=0.05)
    assert g["evidence"]["t_end"] == pytest.approx(truth["dissolve"]["end"], abs=0.05)
    assert g["t"] == pytest.approx(truth["dissolve"]["center"], abs=0.04)


# ------------------------------------------------------------------ fix round 1: motion bursts

def _whip_clip(tmp_path, img, travel, secs, blur):
    name = tmp_path / f"{img}_{travel}_{secs}_{blur}.mp4"
    src = tmp_path / f"{img}.png"
    if not src.exists():
        spec = ("mandelbrot=s=3200x180:r=25:start_scale=0.05:start_x=-0.74:start_y=0.1" if img == "w"
                else "testsrc2=s=3200x180:r=25:d=1")
        _ff("-f", "lavfi", "-i", spec, "-frames:v", "1", str(src))
    ss = f"(3*pow((t-1.5)/{secs},2)-2*pow((t-1.5)/{secs},3))"
    x = f"if(lt(t,1.5),0,if(lt(t,{1.5 + secs}),{travel}*{ss},{travel}))"
    vf = f"crop=320:180:x='{x}':y=0" + (",tmix=frames=3" if blur else "")
    _ff("-loop", "1", "-framerate", "25", "-i", str(src), "-t", "4", "-vf", vf, *_ENC, str(name))
    return name


@pytest.mark.parametrize("img", ["w", "w2"])
@pytest.mark.parametrize("travel,secs", [(1200, 0.2), (2000, 0.24), (2400, 0.32)])
@pytest.mark.parametrize("blur", [False, True])
def test_fast_whip_pan_is_one_boundary_never_a_confident_cut(tmp_path, img, travel, secs, blur):
    """Fast moves change every frame: they used to leave 1-3 cuts at confidence 1.0 and 0.1 s phantom shots."""
    p = _whip_clip(tmp_path, img, travel, secs, blur)
    series = S.frame_series(str(p), 25)
    found = S.detect_shots(series, 4.0)
    assert len(found) == 2, [(s["start"], s["boundary"] and s["boundary"]["kind"]) for s in found]
    b = found[1]["boundary"]
    assert b["kind"] == "motion" and b["evidence"]["confidence"] <= 0.5
    from breakdown import transitions as T
    out = T.classify(series, found)
    assert [e["kind"] for e in out] == [T.NOT_DETERMINED] and out[0]["evidence"]["confidence"] == 0


def test_montage_of_unrelated_stills_keeps_every_cut(tmp_path):
    """Three cuts 0.12 s apart between unrelated stills are edits: the picture holds between them."""
    parts = []
    specs = [("testsrc=s=320x180:r=25", 25), ("smptebars=s=320x180:r=25", 3), ("color=c=red:s=320x180:r=25", 3),
             ("mandelbrot=s=320x180:r=25:start_scale=0.3", 3), ("testsrc2=s=320x180:r=25", 25)]
    for i, (spec, n) in enumerate(specs):
        q = tmp_path / f"p{i}.mp4"
        _ff("-f", "lavfi", "-i", spec, "-frames:v", str(n), *_ENC, str(q))
        parts.append(q)
    p = tmp_path / "montage.mp4"
    args = [a for q in parts for a in ("-i", str(q))]
    _ff(*args, "-filter_complex", f"concat=n={len(parts)}:v=1", *_ENC, str(p))
    found = _shots(p, 59 / 25)
    kinds = [s["boundary"]["kind"] for s in found if s["boundary"]]
    assert kinds == ["cut"] * 4, kinds


# ------------------------------------------------------------------ fix round 2

def _flat_series(levels, fps=25):
    """Thumbnails of one flat level each: the step between frames is exactly |dv| / 255."""
    return {"t": [i / fps for i in range(len(levels))], "scene": [0.0] * len(levels),
            "yavg": [float(v) for v in levels], "thumb": [bytes([v]) * 432 for v in levels]}


def _cut(series, i, conf=1.0):
    t = series["t"]
    return {"kind": "cut", "t": t[i], "frames": 1, "t_start": t[i - 1], "t_end": t[i], "confidence": conf}


def test_handheld_move_between_two_real_cuts_keeps_both_cuts():
    """Held A -> cut -> ~1 s fast handheld motion with cut-like jumps -> cut -> held B: the bracketing
    edits are real; only the move between them is not a set of cuts."""
    levels = [100] * 10
    levels.append(176)  # cut 1: step 76 (0.30), outer side held
    for k in range(25):  # interior: steps 23-43 levels, a jump every third frame
        levels.append(levels[-1] + (43 if k % 3 == 0 else 23) * (1 if k % 2 == 0 else -1))
    levels.append(levels[-1] - 60)  # cut 2: step 60 (0.235)
    levels += [levels[-1]] * 10
    assert min(levels) >= 0 and max(levels) <= 255
    series = _flat_series(levels)
    first, last = 10, 10 + 26
    jumps = [i for i in range(first + 1, last) if (i - first - 1) % 3 == 0]
    bounds = [_cut(series, first)] + [_cut(series, i) for i in jumps] + [_cut(series, last)]
    out = S._collapse_bursts(series, bounds)
    assert [round(b["t"], 2) for b in out] == [round(series["t"][first], 2), round(series["t"][last], 2)]
    assert all(b["kind"] == "cut" for b in out)
    assert len(S.build_shots(out, len(levels) / 25)) == 3


def test_same_picture_nudge_is_not_a_cut_but_a_real_cut_stays():
    import random
    rnd = random.Random(5)
    base = [rnd.randrange(40, 200) for _ in range(432)]
    other = [rnd.randrange(40, 200) for _ in range(432)]

    def frame(pic, shift=0):
        return bytes(max(0, min(255, v + shift + rnd.randint(-9, 9))) for v in pic)

    thumbs = [frame(base) for _ in range(10)] + [frame(base, 14) for _ in range(10)]   # brightness nudge at 10
    thumbs += [frame(other) for _ in range(10)]                                         # real cut at 20
    series = {"t": [i / 25 for i in range(30)], "scene": [0.0] * 30, "yavg": [0.0] * 30, "thumb": thumbs}
    out = S._vet_cuts(series, [_cut(series, 10), _cut(series, 20)])
    assert [round(b["t"] * 25) for b in out] == [20]


# ------------------------------------------------------------------ fix round 3

@pytest.fixture(scope="module")
def strips(tmp_path_factory):
    d = tmp_path_factory.mktemp("strips")
    _ff("-f", "lavfi", "-i", "mandelbrot=s=4000x180:start_scale=0.05:start_x=-0.74:start_y=0.1", "-frames:v", "1", str(d / "a.png"))
    _ff("-f", "lavfi", "-i", "testsrc2=s=4000x180:d=1", "-frames:v", "1", str(d / "b.png"))
    return d


def _pan(d, img, frames, v, start, sign, name):
    q = d / name
    x = f"{start}{'+' if sign > 0 else '-'}n*{v}"
    _ff("-loop", "1", "-framerate", "25", "-i", str(d / f"{img}.png"), "-frames:v", str(frames),
        "-vf", f"crop=320:180:{x}:0", *_ENC, str(q))
    return q


@pytest.mark.parametrize("v", [10, 20, 40])
def test_hard_cut_between_two_pans_is_a_cut(strips, v):
    """A hard cut between two moving pictures used to be read as a gradual boundary at 1.0."""
    a, b = _pan(strips, "a", 50, v, 0, 1, f"ca{v}.mp4"), _pan(strips, "b", 50, v, 3000, -1, f"cb{v}.mp4")
    p = strips / f"cut{v}.mp4"
    _ff("-i", str(a), "-i", str(b), "-filter_complex", "concat=n=2:v=1", *_ENC, str(p))
    found = S.detect_shots(S.frame_series(str(p), 25), 4.0)
    kinds = [(s["boundary"]["kind"], round(s["boundary"]["t"], 2)) for s in found if s["boundary"]]
    assert [k for k, _ in kinds] == ["cut"] and abs(kinds[0][1] - 2.0) <= 0.04, kinds


@pytest.mark.parametrize("v", [10, 20, 40])
def test_one_continuous_pan_is_one_shot(strips, v):
    p = _pan(strips, "b", 50, v, 3000, -1, f"one{v}.mp4")
    assert len(S.detect_shots(S.frame_series(str(p), 25), 2.0)) == 1


def test_same_picture_frames_are_dropped_not_demoted():
    import random
    rnd = random.Random(8)
    base = [rnd.randrange(40, 200) for _ in range(432)]
    jump = [min(255, max(0, v + (60 if i % 7 == 0 else 0))) for i, v in enumerate(base)]
    thumbs = [bytes(base)] * 8 + [bytes(jump)] * 8
    series = {"t": [i / 25 for i in range(16)], "scene": [0.0] * 16, "yavg": [0.0] * 16, "thumb": thumbs}
    assert S._vet_cuts(series, [_cut(series, 8)]) == []


# ------------------------------------------------------------------ fix round 4

@pytest.fixture(scope="module")
def mont(tmp_path_factory):
    d = tmp_path_factory.mktemp("mont")
    specs = {"a": "mandelbrot=s=4000x180:start_scale=0.05:start_x=-0.74:start_y=0.1", "b": "testsrc2=s=4000x180:d=1",
             "c": "mandelbrot=s=4000x180:start_scale=0.3:start_x=-0.5:start_y=0.0",
             "e": "mandelbrot=s=4000x180:start_scale=0.01:start_x=-1.25:start_y=0.02"}
    for k, v in specs.items():
        _ff("-f", "lavfi", "-i", v, "-frames:v", "1", str(d / f"{k}.png"))
    return d


def _moving_montage(d, v):
    parts = [_pan(d, "a", 25, v, 0, 1, f"m0{v}.mp4")]
    for j, (img, sg) in enumerate((("b", -1), ("c", 1), ("e", 1), ("b", 1), ("c", -1))):
        parts.append(_pan(d, img, 4, v, 1500 if sg < 0 else 200 * j, sg, f"m{j + 1}{v}.mp4"))
    parts.append(_pan(d, "e", 25, v, 2000, -1, f"m9{v}.mp4"))
    p = d / f"montage{v}.mp4"
    _ff(*[a for q in parts for a in ("-i", str(q))], "-filter_complex", f"concat=n={len(parts)}:v=1", *_ENC, str(p))
    return p


@pytest.mark.parametrize("v", [8, 20])
def test_moving_montage_keeps_every_cut(mont, v):
    """Two different strips cut at 1.32 s: thumbnail step 0.475 but scene score 0.058, correlation 0.805."""
    found = S.detect_shots(S.frame_series(str(_moving_montage(mont, v)), 25), 70 / 25)
    assert len(found) == 7, [(s["start"], s["boundary"] and s["boundary"]["kind"]) for s in found]
    assert any(abs(s["start"] - 1.32) < 0.05 and s["boundary"]["kind"] == "cut" for s in found)


def test_large_step_with_correlated_frames_is_still_a_cut():
    import random
    rnd = random.Random(2)
    base = [rnd.randrange(40, 200) for _ in range(432)]
    other = [min(255, max(0, int(0.5 * v + 90 + rnd.randint(-30, 30)))) for v in base]  # correlated, far apart
    series = {"t": [i / 25 for i in range(16)], "scene": [0.0] * 16, "yavg": [0.0] * 16,
              "thumb": [bytes(base)] * 8 + [bytes(other)] * 8}
    assert S.correlation(bytes(base), bytes(other)) >= S.SAME_PICTURE and S._steps(series)[8] > S.SAME_PICTURE_MAX_STEP
    assert len(S._vet_cuts(series, [_cut(series, 8)])) == 1


def test_without_opencv_a_pan_boundary_is_capped_and_says_so(strips, monkeypatch):
    p = _pan(strips, "b", 100, 20, 0, 1, "nocv.mp4")  # a pan the cv2 check would remove
    series = S.frame_series(str(p), 25)
    monkeypatch.setitem(sys.modules, "cv2", None)
    found = S.detect_shots(series, 4.0)
    bounds = [s["boundary"] for s in found if s["boundary"] and s["boundary"]["kind"] != "cut"]
    assert bounds, "without the frame-level check the pan stays a boundary"
    assert all(b["evidence"]["confidence"] <= 0.4 and "opencv missing" in b["note"] for b in bounds)
    monkeypatch.undo()
    assert [s for s in S.detect_shots(series, 4.0) if s["boundary"] and s["boundary"]["kind"] != "cut"] == []
