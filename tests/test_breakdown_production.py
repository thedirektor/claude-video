"""Per-shot production method (breakdown/production.py) and file provenance (breakdown/provenance.py)."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skills" / "watch" / "scripts"))

from breakdown import production, provenance  # noqa: E402

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

SHOT = {"start": 0.0, "end": 4.0, "duration": 4.0}


def _ff(*args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def _shot(path, fps=25.0, prov=None):
    return production.shot_production(str(path), SHOT, fps, prov)


# ----------------------------------------------------------------------------- classify (pure)

def _m(**kw):
    base = {"frames": 29, "held_share": 0.0, "source_fps": None, "motion": 5.0, "still_diff": 20.0,
            "flat": 0.02, "rigid_residual": None, "rigid_pairs": 0}
    return {**base, **kw}


def test_textured_footage_is_not_determined_with_ranked_candidates():
    r = production.classify(SHOT, _m())
    assert r["label"] == production.NOT_DETERMINED
    assert r["evidence"]["confidence"] == 0.0
    assert [c["label"] for c in r["candidates"]] == ["camera", "ai_video"]


def test_held_cadence_leans_ai_video_but_does_not_claim_it():
    r = production.classify(SHOT, _m(held_share=0.2, source_fps=24.0))
    assert r["label"] == production.NOT_DETERMINED
    assert r["candidates"][0]["label"] == "ai_video"
    assert any("24 fps" in n for n in r["notes"])


def test_declared_ai_and_camera_override_textured_footage():
    assert production.classify(SHOT, _m(), {"declared": "ai_generated"})["label"] == "ai_generated"
    assert production.classify(SHOT, _m(), {"declared": "camera"})["label"] == "camera"


def test_smooth_surfaces_are_synthetic_even_when_declared_ai():
    r = production.classify(SHOT, _m(flat=0.5), {"declared": "ai_generated"})
    assert r["label"] == "synthetic"
    assert {c["label"] for c in r["candidates"]} == {"synthetic", "ai_generated"}
    assert r["evidence"]["confidence"] <= production.CAP_UNDECLARED


def test_still_and_rigid_plane():
    assert production.classify(SHOT, _m(still_diff=0.1))["label"] == "still"
    assert production.classify(SHOT, _m(rigid_residual=1.0, rigid_pairs=4))["label"] == "still_moved"
    # one rigid pair is not enough
    assert production.classify(SHOT, _m(rigid_residual=1.0, rigid_pairs=1))["label"] == production.NOT_DETERMINED


def test_no_frames_is_not_determined():
    r = production.classify(SHOT, {"frames": 0})
    assert r["label"] == production.NOT_DETERMINED and r["notes"]


# ----------------------------------------------------------------------------- measured on synthesised clips

@needs_ffmpeg
def test_still_slide(tmp_path):
    p = tmp_path / "slide.mp4"
    _ff("-f", "lavfi", "-i", "color=c=0x3060c0:s=640x360:r=25:d=4,drawbox=x=80:y=80:w=300:h=120:c=white:t=fill",
        "-pix_fmt", "yuv420p", str(p))
    assert _shot(p)["label"] == "still"


@needs_ffmpeg
def test_flat_graphics_animation_is_synthetic(tmp_path):
    p = tmp_path / "ui.mp4"
    # drawbox does not animate with t: move solid cards with overlay instead
    _ff("-f", "lavfi", "-i", "color=c=0x606a78:s=640x360:r=25:d=4", "-f", "lavfi", "-i", "color=c=0xf0a020:s=180x90:r=25:d=4",
        "-f", "lavfi", "-i", "color=c=white:s=200x60:r=25:d=4", "-filter_complex",
        "[0][1]overlay=x='40+t*60':y=60[a];[a][2]overlay=x=380:y='200-t*30'", "-pix_fmt", "yuv420p", str(p))
    r = _shot(p)
    assert r["label"] == "synthetic", r
    assert r["measurements"]["flat"] >= production.FLAT_SYNTHETIC


@needs_ffmpeg
def test_white_ui_background_is_synthetic(tmp_path):
    p = tmp_path / "white_ui.mp4"
    _ff("-f", "lavfi", "-i", "color=c=white:s=640x360:r=25:d=4", "-f", "lavfi", "-i",
        "testsrc2=s=200x120:r=25:d=4", "-filter_complex", "[0][1]overlay=x='60+t*40':y=120", "-pix_fmt", "yuv420p", str(p))
    r = _shot(p)
    assert r["label"] == "synthetic", r
    assert r["measurements"]["white_fill"] >= production.WHITE_FILL


def test_denoised_footage_between_bounds_only_leans_synthetic():
    r = production.classify(SHOT, _m(flat=0.35))
    assert r["label"] == production.NOT_DETERMINED and r["candidates"][0]["label"] == "synthetic"


@needs_ffmpeg
def test_photo_push_in_is_still_moved(tmp_path):
    img = tmp_path / "photo.png"
    _ff("-f", "lavfi", "-i", "mandelbrot=s=1280x720", "-frames:v", "1", str(img))
    p = tmp_path / "push.mp4"
    _ff("-loop", "1", "-i", str(img), "-vf",
        "zoompan=z='1+0.0025*on':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=100:s=640x360:fps=25",
        "-t", "4", "-pix_fmt", "yuv420p", str(p))
    r = _shot(p)
    assert r["label"] == "still_moved", r


@needs_ffmpeg
def test_textured_motion_at_native_rate_vs_doubled_cadence(tmp_path):
    native = tmp_path / "native.mp4"
    _ff("-f", "lavfi", "-i", "life=s=640x360:r=25:mold=10:ratio=0.5:death_color=#203040:life_color=#c0a080,"
        "noise=alls=25:allf=t,gblur=sigma=0.6", "-t", "4", "-pix_fmt", "yuv420p", str(native))
    r = _shot(native)
    assert r["label"] == production.NOT_DETERMINED, r
    assert r["measurements"]["source_fps"] is None
    doubled = tmp_path / "doubled.mp4"
    _ff("-f", "lavfi", "-i", "life=s=640x360:r=12:mold=10:ratio=0.5:death_color=#203040:life_color=#c0a080,"
        "noise=alls=25:allf=t,gblur=sigma=0.6", "-vf", "fps=24", "-t", "4", "-pix_fmt", "yuv420p", str(doubled))
    r = _shot(doubled, fps=24.0)
    assert r["label"] == production.NOT_DETERMINED, r
    assert r["measurements"]["source_fps"] and r["measurements"]["source_fps"] < 16, r["measurements"]
    assert r["candidates"][0]["label"] == "ai_video"


# ----------------------------------------------------------------------------- provenance

@needs_ffmpeg
def test_provenance_aigc_label_and_camera_tags(tmp_path):
    aigc = tmp_path / "aigc.mp4"
    label = json.dumps({"AIGC": {"Label": "1", "ContentProducer": "001191110108MA01KP2T5U00000"}})
    _ff("-f", "lavfi", "-i", "testsrc2=s=320x180:r=25:d=1", "-metadata", f"comment={label}",
        "-pix_fmt", "yuv420p", str(aigc))
    pv = provenance.probe(str(aigc))
    assert pv["declared"] == "ai_generated"
    assert pv["aigc"]["label"] == "1" and pv["aigc"]["content_producer"].startswith("001191")

    cam = tmp_path / "cam.mov"
    _ff("-f", "lavfi", "-i", "testsrc2=s=320x180:r=25:d=1", "-movflags", "use_metadata_tags",
        "-metadata", "com.apple.quicktime.make=Apple", "-metadata", "com.apple.quicktime.model=iPhone 15 Pro",
        "-pix_fmt", "yuv420p", str(cam))
    pv = provenance.probe(str(cam))
    assert pv["declared"] == "camera" and pv["camera"]

    plain = tmp_path / "plain.mp4"
    _ff("-f", "lavfi", "-i", "testsrc2=s=320x180:r=25:d=1", "-pix_fmt", "yuv420p", str(plain))
    pv = provenance.probe(str(plain))
    assert pv["declared"] is None and pv["evidence"] == []


def test_provenance_signature_and_aigc_parsing():
    assert provenance.parse_aigc('{"Label":"2","ContentProducer":"abc"}')["meaning"] == "possibly AI-generated"
    assert provenance.parse_aigc("no label here") is None


def test_provenance_unreadable_file(tmp_path):
    pv = provenance.probe(str(tmp_path / "missing.mp4"))
    assert pv["declared"] is None
