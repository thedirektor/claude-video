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
