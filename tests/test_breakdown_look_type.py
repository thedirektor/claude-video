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
    # 45 px bars on 360 rows read 0.125 exactly (the 64 px work width used to quantise them to 0.111)
    assert lb and abs(lb["top"] - 0.125) < 0.004 and abs(lb["bottom"] - 0.125) < 0.004, lb


def test_letterbox_odd_bar_sizes_are_measured_at_full_resolution(tmp_path):
    import subprocess
    p = tmp_path / "lb2.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc=s=1280x720:r=25:d=2", "-vf", "drawbox=x=0:y=0:w=1280:h=70:color=black:t=fill,"
                    "drawbox=x=0:y=670:w=1280:h=50:color=black:t=fill,format=yuv420p", str(p)], check=True)
    lb = L.shot_look(str(p), {"start": 0.0, "end": 2.0})["letterbox"]
    assert lb and abs(lb["top"] - 70 / 720) < 0.004 and abs(lb["bottom"] - 50 / 720) < 0.004, lb


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
    assert lb and abs(lb["left"] - 0.125) < 0.004 and abs(lb["right"] - 0.125) < 0.004, lb
    assert lb["top"] == 0.0 and lb["bottom"] == 0.0


_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _title_clip(path, extra, size="48", src="color=c=navy:s=640x360:r=25:d=5", pre=""):
    import subprocess
    if not Path(_FONT).is_file():
        pytest.skip("no font for drawtext")
    dt = (pre + f"drawtext=fontfile={_FONT}:text='HELLO WORLD':fontsize={size}:y=40:fontcolor=white:{extra}")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    src, "-vf", dt, "-c:v", "libx264", "-pix_fmt", "yuv420p",
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


_MOVING = "testsrc2=s=640x360:r=25:d=5"  # raw: running timecode sits in the title's own rows


@pytest.mark.parametrize("name,extra", [
    ("cut", "x=40:enable='between(t,1,3)'"),
    ("fade", "x=40:alpha='if(lt(t,1),0,if(lt(t,1.6),(t-1)/0.6,1))'"),
    ("slide", "x='if(lt(t,1.4),40+600*(1.4-t)/0.4,40)':enable='gte(t,1)'"),
])
def test_moving_background_never_invents_motion(tmp_path, name, extra):
    path = _title_clip(tmp_path / f"m_{name}.mp4", extra, src=_MOVING)
    tracks, notes = TY.text_tracks(path, 5.0)
    assert notes == []
    for t in tracks:
        assert t["font_family"] == "not determined"
        for k in ("entrance", "exit"):
            assert not t[k].startswith("slide") and t[k] != "scale", (name, t["text"], k, t[k])
        if "HELLO" in t["text"].upper():
            assert t["t_in"] >= 0.9, (name, t["text"], t["t_in"])


# ---- 0.7.1 batch 2: punch words shorter than the OCR step, one-sample tracks, OCR cost ----
@pytest.mark.parametrize("a,b", [(1.1, 1.4), (2.12, 2.34)], ids=["0.3s", "0.22s"])
def test_title_shorter_than_ocr_step_is_found(tmp_path, a, b):
    t = _only(_title_clip(tmp_path / "s.mp4", f"x=40:enable='between(t,{a},{b})'"))
    assert abs(t["t_in"] - a) <= 0.12 and abs(t["t_out"] - b) <= 0.12, t
    assert t["evidence"]["method"] == "ocr" and t["evidence"]["confidence"] > 0


def test_no_title_means_no_extra_tracks_and_few_extra_ocr_calls(tmp_path, monkeypatch):
    import subprocess
    p = tmp_path / "plain.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=s=640x360:r=25:d=6", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(p)], check=True)
    calls, in_sampling = [], []
    orig, orig_sample = TY._detect, TY._sample
    monkeypatch.setattr(TY, "_detect", lambda ocr, img, cv2: (calls.append(1), orig(ocr, img, cv2))[1])

    def counted_sample(*a, **kw):
        res = orig_sample(*a, **kw)
        in_sampling.append(len(calls))
        return res
    monkeypatch.setattr(TY, "_sample", counted_sample)
    tracks, notes = TY.text_tracks(str(p), 6.0)
    assert notes == []
    # 12 base samples plus a bounded budget of extras (0.6 per base sample + 4) even on a cluttered picture
    assert in_sampling[0] <= 12 + int(TY.EXTRA_SHARE * 12) + TY.EXTRA_MIN, in_sampling


def test_one_sample_track_gets_entrance_and_exit(tmp_path):
    # visible 1.35-1.7: exactly one OCR sample (t=1.5) sees it
    t = _only(_title_clip(tmp_path / "one.mp4", "x=40:enable='between(t,1.35,1.7)'"))
    assert t["evidence"]["samples"] == 1, t["evidence"]
    assert t["entrance"] == "cut" and t["exit"] == "cut", (t["entrance"], t["exit"])


def test_detect_downscales_tall_frames_for_ocr():
    import numpy as np
    seen = {}

    class FakePT:
        class Output:
            DICT = "dict"

        @staticmethod
        def image_to_data(pil, output_type=None, config=None):
            seen["size"] = pil.size
            return {"text": ["HELLO"], "conf": ["90"], "block_num": [1], "par_num": [1], "line_num": [1],
                    "left": [100], "top": [50], "width": [200], "height": [40]}

    from PIL import Image
    import cv2
    img = np.zeros((2160, 3840, 3), np.uint8)
    out = TY._detect((FakePT, Image), img, cv2)
    assert seen["size"][1] <= 720 and seen["size"][0] == round(3840 * seen["size"][1] / 2160)
    w, h = seen["size"]
    assert abs(out[0]["box"][0] - 100 / w) < 1e-9 and abs(out[0]["box"][3] - 40 / h) < 1e-9


def test_unchanged_frames_are_not_re_ocrd(tmp_path, monkeypatch):
    path = _title_clip(tmp_path / "static.mp4", "x=40:enable='between(t,1,4)'")
    calls, in_sampling = [], []
    orig, orig_sample = TY._detect, TY._sample
    monkeypatch.setattr(TY, "_detect", lambda ocr, img, cv2: (calls.append(1), orig(ocr, img, cv2))[1])

    def counted_sample(*a, **kw):
        res = orig_sample(*a, **kw)
        in_sampling.append(len(calls))
        return res
    monkeypatch.setattr(TY, "_sample", counted_sample)
    tracks, _ = TY.text_tracks(path, 5.0)
    assert [t for t in tracks if "HELLO" in t["text"].upper()]
    # 10 base samples used to be 10 OCR runs; a still picture needs one per change
    assert in_sampling[0] <= 4, in_sampling


def test_brief_piece_of_a_longer_line_is_dropped_wherever_it_sits():
    full = {"text": "HELLO WORLD", "t_in": 1.0, "t_out": 5.0, "box": [0.07, 0.11, 0.6, 0.1],
            "evidence": {"samples": 8}}
    piece = {"text": "HELLO", "t_in": 1.18, "t_out": 1.22, "box": [0.69, 0.1, 0.3, 0.16], "evidence": {"samples": 1}}
    long_other = {"text": "HELLO", "t_in": 1.5, "t_out": 4.0, "box": [0.69, 0.1, 0.3, 0.16], "evidence": {"samples": 6}}
    assert [t["text"] for t in TY._drop_fragments([full, piece])] == ["HELLO WORLD"]
    assert len(TY._drop_fragments([full, long_other])) == 2  # a sustained, separately placed line stays


def test_title_survives_the_ocr_downscale_on_a_1080p_source(tmp_path):
    t_ = _title_clip(tmp_path / "hd.mp4", "x=120:enable='between(t,1,3)'", size="150",
                     src="color=c=navy:s=1920x1080:r=25:d=4")
    tracks, notes = TY.text_tracks(t_, 4.0)
    hit = [t for t in tracks if "HELLO" in t["text"].upper() and "WORLD" in t["text"].upper()]
    assert notes == [] and hit, tracks
    assert abs(hit[0]["t_in"] - 1.0) <= 0.1 and 0.1 < hit[0]["height_rel"] < 0.2


def test_bisect_probes_read_only_a_crop_around_the_text(tmp_path, monkeypatch):
    path = _title_clip(tmp_path / "crop.mp4", "x=40:enable='between(t,1.2,3.3)'")
    shapes = []
    orig = TY._detect
    monkeypatch.setattr(TY, "_detect", lambda ocr, img, cv2: (shapes.append(img.shape[:2]), orig(ocr, img, cv2))[1])
    t = _only(path)
    full = [s for s in shapes if s == (360, 640)]
    crops = [s for s in shapes if s != (360, 640)]
    assert crops and len(crops) >= 6, shapes  # the bisect probes
    assert all(c[0] * c[1] < 0.5 * 360 * 640 for c in crops)
    assert abs(t["t_in"] - 1.2) <= 0.1 and abs(t["t_out"] - 3.3) <= 0.1


# ---- fix round 1: frame stamps under low / variable frame rates ----
def _fps_clip(tmp_path, r, d=8):
    return _title_clip(tmp_path / f"f{r}.mp4", "x=40:enable='between(t,5,6)'",
                       src=f"color=c=navy:s=640x360:r={r}:d={d}")


@pytest.mark.parametrize("r", [5, 8, 12])
def test_low_frame_rate_sources_are_stamped_with_real_times(tmp_path, r):
    tracks, _ = TY.text_tracks(_fps_clip(tmp_path, r), 8.0)
    hit = [t for t in tracks if "HELLO" in t["text"].upper()]
    assert hit and abs(hit[0]["t_in"] - 5.0) <= 0.3 and abs(hit[0]["t_out"] - 6.0) <= 0.3, (r, tracks)


def test_variable_frame_rate_source_is_found_or_flagged(tmp_path):
    import subprocess
    src = _title_clip(tmp_path / "v0.mp4", "x=40:enable='between(t,5,6)'", src="color=c=navy:s=640x360:r=25:d=8")
    p = tmp_path / "vfr.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src, "-vf", "mpdecimate",
                    "-fps_mode", "vfr", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(p)], check=True)
    tracks, notes = TY.text_tracks(str(p), 8.0)
    hit = [t for t in tracks if "HELLO" in t["text"].upper()]
    assert (hit and abs(hit[0]["t_in"] - 5.0) <= 0.3) or notes, (tracks, notes)


def test_sampling_note_is_worded_per_path(tmp_path):
    from breakdown import assemble
    assert TY.scan_available(_fps_clip(tmp_path, 25), 8.0) is True
    assert TY.scan_available(_fps_clip(tmp_path, 5), 8.0) is False
    assert "seeking" in assemble.SAMPLING_NOTE_SEEK and "word-shape scan" not in assemble.SAMPLING_NOTE_SEEK


@pytest.mark.parametrize("a", [2.13, 2.27, 2.41])
def test_short_title_over_moving_footage_is_timed_or_flagged_approximate(tmp_path, a):
    path = _title_clip(tmp_path / "bm.mp4", f"x=40:enable='between(t,{a},{a + 0.4:.2f})'", src=_MOVING)
    tracks, _ = TY.text_tracks(path, 5.0)
    hit = [t for t in tracks if "HELLO" in t["text"].upper()]
    for t in hit:
        late = t["t_in"] - a
        assert late <= 0.15 or (t["evidence"]["confidence"] <= 0.4
                                and t["evidence"]["timing_note"] == "start time approximate (busy background)"), (a, t)


def test_dark_gradient_edge_is_not_a_pillarbox(tmp_path):
    import subprocess
    p = tmp_path / "grad.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "rgbtestsrc=s=1280x720:r=25:d=2", "-pix_fmt", "yuv420p", str(p)], check=True)
    assert L.shot_look(str(p), {"start": 0.0, "end": 2.0})["letterbox"] is None
