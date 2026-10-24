from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from breakdown_fixtures import _ffmpeg, make_motion_clip, make_reference  # noqa: E402

from breakdown import compare  # noqa: E402
from breakdown.media import probe  # noqa: E402


def _delay(src: Path, dst: Path, seconds: float, audio: bool = True) -> None:
    args = ["-i", str(src), "-vf", f"tpad=start_duration={seconds}:start_mode=clone"]
    if audio:
        args += ["-af", f"adelay={int(seconds * 1000)}:all=1", "-c:a", "aac"]
    else:
        args += ["-an"]
    _ffmpeg([*args, "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", str(dst)])


@pytest.fixture(scope="module")
def ref(tmp_path_factory):
    d = tmp_path_factory.mktemp("cmp")
    p = d / "ref.mp4"
    gt = make_reference(p)
    return p, gt


def test_audio_alignment(ref, tmp_path):
    p, gt = ref
    cand = tmp_path / "late.mp4"
    _delay(p, cand, 0.4)
    res = compare.run(str(p), str(cand), out_dir=tmp_path / "out")
    al = res["alignment"]
    assert al["method"] == "audio"
    assert al["offset_s"] == pytest.approx(0.4, abs=1 / gt["fps"])
    # the 2.0 s cut lines up once aligned
    cut = [s for s in res["shots"] if abs(s["ref"] - 2.0) < 0.1]
    assert cut and abs(cut[0]["delta_ms"]) <= 1000 / gt["fps"] + 1


def test_motion_onset_alignment(tmp_path):
    a, b = tmp_path / "a.mp4", tmp_path / "b0.mp4"
    make_motion_clip(a)
    make_motion_clip(b)
    late = tmp_path / "b.mp4"
    _delay(b, late, 0.1, audio=False)
    res = compare.run(str(a), str(late), out_dir=tmp_path / "out")
    al = res["alignment"]
    assert al["method"] == "motion_onset"
    assert al["offset_s"] * 1000 == pytest.approx(100, abs=1000 / 60 + 1)
    assert res["phases"], "the move is compared as a motion phase"
    ph = res["phases"][0]
    assert abs(ph["start_delta_ms"]) <= 1000 / 60 * 2 + 1
    assert abs(ph["duration_delta_ms"]) <= 1000 / 60 * 3 + 1


def test_unmatched_reported(ref, tmp_path):
    p, gt = ref
    # same edit without the red -> green cut at 2.0 (shot B is red too)
    cand = tmp_path / "fewer.mp4"
    graph = (
        "color=c=red:s=640x360:r=25:d=2[a];color=c=red:s=640x360:r=25:d=3[b];"
        "color=c=blue:s=640x360:r=25:d=3[c];"
        "[a][b]concat=n=2:v=1:a=0,fps=25,settb=1/25[ab];[c]settb=1/25[c2];"
        "[ab][c2]xfade=transition=fade:duration=1:offset=4,fade=t=out:st=6:d=1,format=yuv420p[v]"
    )
    _ffmpeg(["-f", "lavfi", "-i",
             "aevalsrc=0.8*sin(2*PI*1000*t)*lt(mod(t\\,0.5)\\,0.02):s=44100:d=7",
             "-filter_complex", graph, "-map", "[v]", "-map", "0:a", "-c:a", "aac",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-t", "7", str(cand)])
    res = compare.run(str(p), str(cand), out_dir=tmp_path / "out")
    assert res["alignment"]["offset_s"] == pytest.approx(0.0, abs=0.05)
    missing = [u for u in res["unmatched"] if u["side"] == "ref" and abs(u["t"] - 2.0) < 0.1]
    assert missing, res["unmatched"]
    # no forced pairing: nothing in the candidate is mapped onto the 2.0 s cut
    assert not [s for s in res["shots"] if abs(s["ref"] - 2.0) < 0.1]


def test_side_by_side_video(ref, tmp_path):
    p, gt = ref
    cand = tmp_path / "late.mp4"
    _delay(p, cand, 0.4)
    out = tmp_path / "out"
    res = compare.run(str(p), str(cand), slow=2, out_dir=out)
    video = out / res["video"]
    assert video.is_file()
    info = probe(video)
    assert info["width"] == 2 * res["normalised"]["width"]
    assert info["height"] == res["normalised"]["height"]
    overlap = res["overlap"]["ref_end"] - res["overlap"]["ref_start"]
    assert overlap == pytest.approx(7.0, abs=0.1)
    assert info["duration"] == pytest.approx(2 * overlap, abs=0.25)
    assert json.loads((out / "compare.json").read_text())["schema"] == "watch.compare/1"


def test_periodic_audio_not_confident(tmp_path):
    """Click track + flat picture: lags one beat apart score the same -> never a confident offset."""
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    for path, delay in ((a, 0), (b, 500)):
        _ffmpeg(["-f", "lavfi", "-i", "color=c=gray:s=320x180:r=25:d=6", "-f", "lavfi", "-i",
                 "aevalsrc=0.8*sin(2*PI*1000*t)*lt(mod(t\\,0.5)\\,0.02):s=44100:d=6",
                 "-af", f"adelay={delay}:all=1", "-c:v", "libx264", "-preset", "veryfast",
                 "-c:a", "aac", "-t", "6", str(path)])
    res = compare.run(str(a), str(b), out_dir=tmp_path / "out")
    al = res["alignment"]
    assert al["confidence"] <= 0.5
    assert any("ambiguous" in n for n in res["notes"])


def test_explicit_offset_and_no_alignment(tmp_path):
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    for path in (a, b):
        _ffmpeg(["-f", "lavfi", "-i", "color=c=gray:s=320x180:r=25:d=3", "-c:v", "libx264",
                 "-preset", "veryfast", "-t", "3", str(path)])
    res = compare.run(str(a), str(b), out_dir=tmp_path / "o1")
    assert res["alignment"]["method"] == "not determined"
    assert res["alignment"]["confidence"] == 0
    assert res["shots"] == [] and res["phases"] == []
    res = compare.run(str(a), str(b), offset=0.25, out_dir=tmp_path / "o2")
    assert res["alignment"] == {"method": "offset", "offset_s": 0.25, "confidence": 1.0,
                                "evidence": res["alignment"]["evidence"]}


def test_cli_compare(ref, tmp_path, capsys):
    from breakdown import cli

    p, gt = ref
    cand = tmp_path / "late.mp4"
    _delay(p, cand, 0.4)
    rc = cli.main(["compare", str(p), str(cand), "--json", "--out", str(tmp_path / "o")])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["alignment"]["method"] == "audio"
    assert (tmp_path / "o" / "compare.mp4").is_file()


def test_missing_candidate(ref, tmp_path):
    p, _ = ref
    with pytest.raises(compare.CompareError):
        compare.run(str(p), str(tmp_path / "nope.mp4"), out_dir=tmp_path / "o")
