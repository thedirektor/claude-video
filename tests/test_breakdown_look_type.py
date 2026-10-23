import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from breakdown_fixtures import make_reference  # noqa: E402
from breakdown import look as L  # noqa: E402
from breakdown import typography as TY  # noqa: E402


@pytest.fixture(scope="module")
def ref(tmp_path_factory):
    p = tmp_path_factory.mktemp("look") / "ref.mp4"
    truth = make_reference(p, with_audio=False)
    return p, truth


def _lab(h):
    import numpy as np
    rgb = np.array([[int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]])
    return L._srgb_to_lab(np, rgb)[0]


def test_palette_dominant(ref):
    import numpy as np
    p, _ = ref
    look = L.shot_look(str(p), {"start": 0.0, "end": 2.0})
    assert look["palette"] and len(look["palette"]) <= 5
    de = float(np.linalg.norm(_lab(look["palette"][0]["hex"]) - _lab("#ff0000")))
    assert de <= 10
    assert look["temperature"] == "warm"
    assert look["letterbox"] is None
    assert look["evidence"]["method"] == "measured"
    for k in ("brightness", "contrast", "saturation"):
        assert 0.0 <= look[k] <= 1.0


def test_letterbox_measured(tmp_path):
    import subprocess
    p = tmp_path / "lb.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc=s=640x360:r=25:d=2", "-vf", "drawbox=x=0:y=0:w=640:h=45:color=black:t=fill,"
                    "drawbox=x=0:y=315:w=640:h=45:color=black:t=fill,format=yuv420p", str(p)], check=True)
    lb = L.shot_look(str(p), {"start": 0.0, "end": 2.0})["letterbox"]
    assert lb and abs(lb["top"] - 0.125) < 0.03 and abs(lb["bottom"] - 0.125) < 0.03


def test_look_without_cv2(ref, monkeypatch):
    p, _ = ref
    monkeypatch.setattr(L, "_import_cv", lambda: (_ for _ in ()).throw(ImportError("no cv2")))
    look = L.shot_look(str(p), {"start": 0.0, "end": 2.0})
    assert look["palette"] == "not determined" and "cv2" in look["notes"][0]


def test_text_track(ref):
    p, truth = ref
    if not truth["title"]["rendered"]:
        pytest.skip("no font for drawtext")
    tracks, notes = TY.text_tracks(str(p), truth["duration"])
    assert notes == []
    hit = [t for t in tracks if "SALE" in t["text"].upper() and "50" in t["text"]]
    assert hit, tracks
    t = hit[0]
    assert abs(t["t_in"] - truth["title"]["start"]) <= 0.5
    assert abs(t["t_out"] - truth["title"]["end"]) <= 0.5
    x, y, w, h = t["box"]
    assert x + w / 2 < 0.5 and y + h / 2 < 0.5
    assert t["font_family"] == "not determined"
    assert t["evidence"]["method"] == "ocr"
    assert t["color_hex"].lower() >= "#c0c0c0"  # near white text
    assert 0 < t["height_rel"] < 0.3


def test_without_tesseract(ref, monkeypatch):
    p, truth = ref
    def boom():
        raise RuntimeError("tesseract binary not found on PATH")
    monkeypatch.setattr(TY, "_load_ocr", boom)
    tracks, notes = TY.text_tracks(str(p), truth["duration"])
    assert tracks == [] and len(notes) == 1 and "tesseract" in notes[0].lower()
