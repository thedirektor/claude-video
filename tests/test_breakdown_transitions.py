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
