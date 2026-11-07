"""Subtitle files from a breakdown (breakdown/subtitles.py, `breakdown --formats srt,ass`)."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from breakdown import cli, subtitles  # noqa: E402
from breakdown_fixtures import make_reference  # noqa: E402
from memory import index, store  # noqa: E402
from memory_fakes import KEY, fake_run, seg  # noqa: E402


def _track(text, t0, t1, box, conf=0.9, colour="#ff8000", entrance="fade"):
    return {"text": text, "t_in": t0, "t_out": t1, "box": box, "height_rel": box[3], "color_hex": colour,
            "entrance": entrance, "exit": "not determined", "evidence": {"confidence": conf}}


BD = {"video": {"key": "k", "width": 1000, "height": 500},
      "typography": {"tracks": [_track("SALE 50%", 1.0, 3.0, [0.1, 0.2, 0.3, 0.1]),
                                _track("noise", 1.0, 2.0, [0, 0, 0.1, 0.1], conf=0.2)]}}


def test_srt_times_and_numbering():
    out = subtitles.srt([{"t_in": 3661.5, "t_out": 3662.004, "text": "b  two"}, {"t_in": 0, "t_out": 1.2, "text": "a"}])
    assert out.startswith("1\n00:00:00,000 --> 00:00:01,200\na\n")
    assert "2\n01:01:01,500 --> 01:01:02,004\nb two\n" in out


def test_low_confidence_text_is_left_out():
    assert [t["text"] for t in subtitles.text_cues(BD)] == ["SALE 50%"]
    assert subtitles.text_cues({"typography": {"tracks": "not determined"}}) == []


def test_ass_uses_only_measured_values():
    out = subtitles.ass(BD, subtitles.speech_cues([{"start": 0.5, "end": 2.0, "text": "hello {there}"}]))
    assert "PlayResX: 1000" in out and "PlayResY: 500" in out
    line = next(ln for ln in out.splitlines() if "SALE 50%" in ln)
    assert "\\pos(100,100)" in line and "\\fs62" in line  # 0.1 * 500 px * 1.25
    assert "\\1c&H0080FF&" in line  # #ff8000 -> BGR
    assert ",OnScreen,in:fade," in line and "\\fad" not in line  # animation noted, never rendered
    assert "Dialogue: 0,0:00:00.50,0:00:02.00,Speech,,0,0,0,,hello (there)" in out
    assert "noise" not in out


def test_write_skips_empty_files(tmp_path):
    assert subtitles.write({"video": {}, "typography": {"tracks": []}}, [], tmp_path, ["srt", "ass"]) == []
    files = subtitles.write(BD, [], tmp_path, ["srt"])
    assert [f.name for f in files] == ["onscreen.srt"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg needed")
def test_breakdown_cli_writes_subtitles(conn, tmp_path, monkeypatch, capsys):
    clip = tmp_path / "ref.mp4"
    truth = make_reference(clip)
    root = tmp_path / "st"
    index.record_run(conn, fake_run(title="Sale clip", duration=truth["duration"],
                                    segments=[seg(0, 1.5, "hello sale"), seg(1.5, 3, "fifty percent")]))
    vdir = store.video_dir(root, KEY)
    vdir.mkdir(parents=True)
    shutil.copy(clip, vdir / "media.mp4")
    monkeypatch.setenv("WATCH_STORE_DIR", str(root))
    assert cli.main(["breakdown", KEY, "--json", "--formats", "srt,ass,edl"]) == 0
    json.loads(capsys.readouterr().out)
    out = vdir / "breakdown"
    cap = (out / "captions.srt").read_text()
    assert "hello sale" in cap and "00:00:01,500 --> 00:00:03,000" in cap
    assert (out / "timeline.edl").is_file() and (out / "onscreen.ass").is_file()
    if (out / "onscreen.srt").is_file():  # OCR present (tesseract): the title is there
        assert "SALE" in (out / "onscreen.srt").read_text().upper()
    assert cli.main(["breakdown", KEY, "--formats", "vtt"]) == 2
