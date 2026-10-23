import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from breakdown_fixtures import make_reference  # noqa: E402
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
    assert series == {"t": [], "scene": [], "yavg": []}
    assert S.detect_shots(series, 3.0) == []
    assert S.rhythm([])["mean"] == "not determined"
