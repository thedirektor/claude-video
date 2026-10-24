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


# ---- realistic cases (reviewer probes): different music, black lead, uniform retime ----

SRCS = ["testsrc2", "mandelbrot", "smptebars", "rgbtestsrc", "testsrc"]
REF_CUTS = [1.5, 3.2, 4.0, 6.1]
DUR = 8.0


def _wav(path, times, freq, dur, seed):
    np = pytest.importorskip("numpy")
    import wave

    sr = 44100
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * sr)) / sr
    x = 0.02 * rng.standard_normal(len(t))
    for i, c in enumerate(times):
        m = (t >= c) & (t < c + 0.06)
        x[m] += 0.7 * np.sin(2 * np.pi * (freq + 37 * (i % 5)) * (t[m] - c)) * np.exp(-(t[m] - c) * 40)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32000).astype("<i2").tobytes())


def _edit(path, cuts, fps, size, audio=None, lead=0.0):
    """Concatenated lavfi shots cut at ``cuts``; ``lead`` seconds of black first (shifts every cut)."""
    srcs = (["color=c=black"] if lead else []) + SRCS
    cuts = ([lead] if lead else []) + [c + lead for c in cuts]
    bounds = [0.0, *cuts, DUR + lead]
    chains = []
    for k, (t0, t1) in enumerate(zip(bounds, bounds[1:])):
        src = srcs[k % len(srcs)]
        sep = ":" if "=" in src else "="
        chains.append(f"{src}{sep}s={size}:r={fps},trim=duration={t1 - t0:.4f},setpts=PTS-STARTPTS,"
                      f"format=yuv420p,setsar=1[s{k}]")
    g = ";".join(chains) + ";" + "".join(f"[s{k}]" for k in range(len(chains))) + f"concat=n={len(chains)}:v=1:a=0"
    args = ["-i", str(audio)] if audio else []
    args += ["-filter_complex", g + "[v]", "-map", "[v]"]
    args += ["-map", "0:a", "-c:a", "aac"] if audio else ["-an"]
    _ffmpeg([*args, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-r", str(fps),
             "-t", str(DUR + lead), str(path)])


A_T = [0.3, 0.9, 1.25, 2.1, 2.4, 3.3, 3.95, 4.6, 5.2, 5.35, 6.0, 6.8, 7.4]
B_T = [0.15, 0.7, 1.6, 1.9, 2.75, 3.1, 4.2, 4.45, 5.6, 6.3, 6.55, 7.1, 7.7]


@pytest.fixture(scope="module")
def edit_ref(tmp_path_factory):
    d = tmp_path_factory.mktemp("edit")
    _wav(d / "a.wav", A_T, 800, DUR, 1)
    _wav(d / "b.wav", B_T, 1300, DUR, 2)
    _edit(d / "ref.mp4", REF_CUTS, 25, "640x360", d / "a.wav")
    return d


def _never_confidently_wrong(res, true_offset, tol=0.1):
    al = res["alignment"]
    if al["method"] == "not determined":
        return
    assert al["confidence"] <= 0.5 or abs(al["offset_s"] - true_offset) <= tol, al


def test_different_music_is_not_audio_aligned(edit_ref, tmp_path):
    cand = tmp_path / "c.mp4"
    _edit(cand, REF_CUTS, 60, "360x640", edit_ref / "b.wav")  # own music, same cuts, 9:16 60 fps
    res = compare.run(str(edit_ref / "ref.mp4"), str(cand), out_dir=tmp_path / "o")
    assert res["alignment"]["method"] != "audio"
    _never_confidently_wrong(res, 0.0)
    for s in res["shots"]:
        assert abs(s["delta_ms"]) <= 60, s


def test_black_lead_not_confidently_wrong(edit_ref, tmp_path):
    cand = tmp_path / "c.mp4"
    _edit(cand, REF_CUTS, 60, "1280x720", lead=0.5)  # silent, 0.5 s of black first
    res = compare.run(str(edit_ref / "ref.mp4"), str(cand), out_dir=tmp_path / "o")
    _never_confidently_wrong(res, 0.48)  # the reference's cuts land on 25 fps frames (1.52 s)
    if res["alignment"]["method"] != "not determined" and res["alignment"]["confidence"] > 0.5:
        for s in res["shots"]:
            assert abs(s["delta_ms"]) <= 60, s


def test_uniform_shift_half_cut_spacing_not_forced(edit_ref, tmp_path):
    """Video sits ~0.4 s from the aligned audio: cuts between two partners are left unpaired."""
    _wav(tmp_path / "a37.wav", [t + 0.37 for t in A_T], 800, DUR + 0.37, 1)
    cand = tmp_path / "c.mp4"
    _edit(cand, REF_CUTS, 60, "360x640", tmp_path / "a37.wav")
    res = compare.run(str(edit_ref / "ref.mp4"), str(cand), out_dir=tmp_path / "o")
    assert res["alignment"]["method"] == "audio"
    for s in res["shots"]:  # every reported pairing is the true one (video ~0.39 s early)
        assert -0.47 <= s["delta_ms"] / 1000 <= -0.33, s
    assert any("ambiguous" in u["reason"] for u in res["unmatched"])


def test_not_determined_offset_is_a_string(tmp_path):
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    for path in (a, b):
        _ffmpeg(["-f", "lavfi", "-i", "color=c=gray:s=320x180:r=25:d=3", "-c:v", "libx264",
                 "-preset", "veryfast", "-t", "3", str(path)])
    res = compare.run(str(a), str(b), out_dir=tmp_path / "o")
    assert res["alignment"]["offset_s"] == "not determined"
    assert "_not determined_" in compare.summary_md(res)


# ---- 0.7.1 backlog -----------------------------------------------------------


def test_exact_pairs_next_to_a_flash_are_paired():
    """Flash-on / flash-off cuts 133 ms apart, both exact: not ambiguous."""
    ref = [1.0, 1.133, 3.0]
    pairs, amb_r, amb_c = compare._match(ref, list(ref), 0.0)
    assert pairs == [(0, 0), (1, 1), (2, 2)] and not amb_r and not amb_c


def test_equally_close_partners_stay_unpaired():
    pairs, amb_r, _ = compare._match([3.24], [2.83, 3.64], 0.0)  # 0.41 vs 0.40 away
    assert pairs == [] and amb_r == {0}


def test_onset_check_probes_the_cut_period():
    np = pytest.importorskip("numpy")
    # change impulses every 0.8 s: an onset alignment could slip by one period
    series = np.zeros(1000)
    series[::80] = 1.0
    rv = (np.zeros(1000), series)
    shifts = compare._confusable_shifts(rv, np)
    assert any(abs(abs(s) - 0.8) < 0.02 for s in shifts)
