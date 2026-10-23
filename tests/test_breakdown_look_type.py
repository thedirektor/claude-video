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
    assert all(int(t["color_hex"][i:i + 2], 16) >= 0xc0 for i in (1, 3, 5))  # near white text
    assert t["entrance"] == "cut" and t["exit"] == "cut"
    assert t["evidence"]["box_track"]
    assert 0 < t["height_rel"] < 0.3


def test_without_tesseract(ref, monkeypatch):
    p, truth = ref
    def boom():
        raise RuntimeError("tesseract binary not found on PATH")
    monkeypatch.setattr(TY, "_load_ocr", boom)
    tracks, notes = TY.text_tracks(str(p), truth["duration"])
    assert tracks == [] and len(notes) == 1 and "tesseract" in notes[0].lower()


def test_pillarbox_measured(tmp_path):
    import subprocess
    p = tmp_path / "pb.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc=s=640x360:r=25:d=2", "-vf", "drawbox=x=0:y=0:w=80:h=360:color=black:t=fill,"
                    "drawbox=x=560:y=0:w=80:h=360:color=black:t=fill,format=yuv420p", str(p)], check=True)
    lb = L.shot_look(str(p), {"start": 0.0, "end": 2.0})["letterbox"]
    assert lb and abs(lb["left"] - 0.125) < 0.03 and abs(lb["right"] - 0.125) < 0.03
    assert lb["top"] == 0.0 and lb["bottom"] == 0.0


_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _title_clip(path, extra, size="48"):
    import subprocess
    if not Path(_FONT).is_file():
        pytest.skip("no font for drawtext")
    dt = (f"drawtext=fontfile={_FONT}:text='HELLO WORLD':fontsize={size}:y=40:fontcolor=white:{extra}")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "color=c=navy:s=640x360:r=25:d=5", "-vf", dt, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "18", str(path)], check=True)
    return str(path)


def _only(path):
    tracks, notes = TY.text_tracks(path, 5.0)
    assert notes == []
    hit = [t for t in tracks if "HELLO" in t["text"].upper()]
    assert len(hit) == 1, tracks
    return hit[0]


def test_entrance_cut(tmp_path):
    t = _only(_title_clip(tmp_path / "a.mp4", "x=40:enable='between(t,1,3)'"))
    assert t["entrance"] == "cut" and abs(t["t_in"] - 1.0) <= 0.05


def test_entrance_fade(tmp_path):
    t = _only(_title_clip(tmp_path / "b.mp4", "x=40:alpha='if(lt(t,1),0,if(lt(t,1.6),(t-1)/0.6,1))'"))
    assert t["entrance"] == "fade" and abs(t["t_in"] - 1.0) <= 0.1


def test_entrance_slide_left(tmp_path):
    t = _only(_title_clip(tmp_path / "c.mp4", "x='if(lt(t,1.6),40+600*(1.6-t)/0.6,40)':enable='gte(t,1)'"))
    assert t["entrance"] == "slide_left" and abs(t["t_in"] - 1.0) <= 0.15


def test_entrance_scale(tmp_path):
    t = _only(_title_clip(tmp_path / "d.mp4", "x=40:enable='gte(t,1)'",
                          size="'if(lt(t,1.5),10+76*(t-1),48)'"))
    assert t["entrance"] == "scale" and abs(t["t_in"] - 1.0) <= 0.15


def test_exit_fade(tmp_path):
    t = _only(_title_clip(tmp_path / "e.mp4", "x=40:alpha='if(lt(t,1),0,if(lt(t,2),1,if(lt(t,2.6),(2.6-t)/0.6,0)))'"))
    assert t["exit"] == "fade" and abs(t["t_out"] - 2.6) <= 0.15


def test_unclear_never_claims_cut(tmp_path):
    t = _only(_title_clip(tmp_path / "f.mp4", "x=40:alpha='if(lt(t,1),0,if(lt(t,1.6),(t-1)/0.6,1))'"))
    assert t["entrance"] != "cut"
    assert t["font_family"] == "not determined"
