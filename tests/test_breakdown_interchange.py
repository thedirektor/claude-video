from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from breakdown import assemble, cli, interchange  # noqa: E402

ND = "not determined"


def _ev(t0, t1, conf=0.9):
    return {"t_start": t0, "t_end": t1, "method": "measured", "confidence": conf}


def make_bd(fps_rational="25/1", cuts=(2.0, 4.5, 6.0), duration=7.0, kinds=None, beats=(0.5, 1.0),
            width=1280, height=720):
    """Synthetic breakdown: shot boundaries at ``cuts``; ``kinds`` maps boundary index -> (kind, dur)."""
    kinds = kinds or {}
    num, den = (int(x) for x in fps_rational.split("/"))
    edges = [0.0, *cuts, duration]
    shots = []
    for i in range(len(edges) - 1):
        tr = None
        if i:
            kind, dur = kinds.get(i, ("cut", 0.0))
            tr = {"t": edges[i], "kind": kind, "duration": dur, "evidence": _ev(edges[i] - dur / 2, edges[i] + dur / 2)}
        shots.append({"index": i, "start": edges[i], "end": edges[i + 1], "duration": edges[i + 1] - edges[i],
                      "transition": tr, "evidence": _ev(edges[i], edges[i + 1])})
    return {
        "schema": "watch.breakdown/1",
        "video": {"key": "yt-abc", "title": "T", "url": "https://example.com/v", "duration": duration,
                  "fps": num / den, "fps_rational": fps_rational, "width": width, "height": height},
        "low_resolution": False,
        "shots": shots,
        "fades": [{"t": 0.0, "kind": "fade_from_black", "duration": 0.5, "evidence": _ev(0.0, 0.5)}],
        "typography": {"tracks": [{"text": "SALE 50%", "t_in": 1.0, "t_out": 3.0, "evidence": _ev(1.0, 3.0)}]},
        "audio": {"tempo_bpm": 120.0, "beats": list(beats)},
        "notes": ["sample note"],
    }


def _bd_mixed():
    return make_bd(kinds={1: ("dissolve", 1.0), 2: ("dip_black", 0.6), 3: (ND, 0.0)}, cuts=(2.0, 4.5, 5.5))


# ---------------------------------------------------------------- OTIO

def _frames(rt):
    return rt["value"]


def test_otio_roundtrip_durations():
    bd = make_bd()
    d = interchange.to_otio(bd, "/m/media.mp4")
    d2 = json.loads(json.dumps(d))
    assert d2["OTIO_SCHEMA"] == "Timeline.1"
    track = d2["tracks"]["children"][0]
    clips = [c for c in track["children"] if c["OTIO_SCHEMA"] == "Clip.2"]
    assert len(clips) == len(bd["shots"])
    total = sum(_frames(c["source_range"]["duration"]) for c in clips)
    assert abs(total - 7.0 * 25) <= 1
    ref = clips[0]["media_references"]["DEFAULT_MEDIA"]
    assert ref["OTIO_SCHEMA"] == "ExternalReference.1" and ref["target_url"].endswith("media.mp4")
    assert clips[1]["source_range"]["start_time"]["value"] == 50
    assert d2["global_start_time"]["value"] == 3600 * 25
    assert d2["global_start_time"]["rate"] == 25.0
    otio = pytest.importorskip("opentimelineio")
    tl = otio.adapters.read_from_string(json.dumps(d), "otio_json")
    assert len(list(tl.find_clips())) == len(bd["shots"])


def test_otio_transitions_and_markers():
    d = interchange.to_otio(_bd_mixed(), "/m/media.mp4")
    children = d["tracks"]["children"][0]["children"]
    trans = [c for c in children if c["OTIO_SCHEMA"] == "Transition.1"]
    assert len(trans) == 1  # only the dissolve; dip and not-determined are cuts + markers
    t = trans[0]
    assert t["transition_type"] == "SMPTE_Dissolve"
    assert _frames(t["in_offset"]) + _frames(t["out_offset"]) == 25
    # transition sits between clips 1 and 2
    i = children.index(t)
    assert children[i - 1]["OTIO_SCHEMA"] == children[i + 1]["OTIO_SCHEMA"] == "Clip.2"
    clips = [c for c in children if c["OTIO_SCHEMA"] == "Clip.2"]
    markers = [m for c in clips for m in c["markers"]]
    assert all(m["OTIO_SCHEMA"] == "Marker.2" for m in markers)
    names = " | ".join(m["name"] + " " + m.get("comment", "") for m in markers)
    assert "transition not determined" in names
    assert "dip_black" in names
    assert "SALE 50%" in names and "sample note" in names and "beat" in names and "fade_from_black" in names


def test_markers_land_in_the_clip_that_contains_them_in_source_time():
    d = interchange.to_otio(make_bd(beats=(3.0,)), "/m/media.mp4")
    clips = d["tracks"]["children"][0]["children"]
    holder = [c for c in clips if any(m["name"].startswith("beat") for m in c["markers"])]
    assert len(holder) == 1 and holder[0]["name"] == "Shot 002"
    m = [m for m in holder[0]["markers"] if m["name"].startswith("beat")][0]
    assert m["marked_range"]["start_time"]["value"] == 75


# ---------------------------------------------------------------- FCPXML

def test_fcpxml_well_formed():
    bd = _bd_mixed()
    xml = interchange.to_fcpxml(bd, "/m/media.mp4")
    root = ET.fromstring(xml)
    assert root.tag == "fcpxml" and root.get("version") == "1.10"
    assert len(root.findall(".//asset")) == 1
    assert root.find(".//asset/media-rep").get("src").startswith("file:///m/media.mp4")
    clips = root.findall(".//spine/asset-clip")
    assert len(clips) == len(bd["shots"])
    rational = re.compile(r"^(\d+/\d+|0)s$")
    for c in clips:
        for a in ("offset", "duration", "start"):
            assert rational.match(c.get(a)), (a, c.get(a))
    trans = root.findall(".//spine/transition")
    assert len(trans) == 1 and trans[0].get("duration") == "25/25s"
    seq = root.find(".//sequence")
    assert seq.get("tcStart") == "3600/1s" or seq.get("tcStart") == "90000/25s"
    marks = [m.get("value") for m in root.findall(".//marker")]
    assert any("transition not determined" in v for v in marks)
    assert any("SALE 50%" in v for v in marks)
    total = sum(int(c.get("duration").split("/")[0]) for c in clips)
    assert abs(total - 7 * 25) <= 1


def test_fcpxml_ntsc_frame_duration_and_df():
    bd = make_bd("30000/1001", cuts=(2.0,), duration=4.0)
    root = ET.fromstring(interchange.to_fcpxml(bd, "/m/media.mp4"))
    assert root.find(".//format").get("frameDuration") == "1001/30000s"
    seq = root.find(".//sequence")
    assert seq.get("tcFormat") == "DF"
    # 01:00:00;00 in drop frame = frame 107892
    assert seq.get("tcStart") == f"{107892 * 1001}/30000s"
    for c in root.findall(".//spine/asset-clip"):
        assert int(c.get("duration").split("/")[0]) % 1001 == 0


# ---------------------------------------------------------------- EDL

def _events(edl):
    return [ln for ln in edl.splitlines() if re.match(r"^\d{3}\s", ln)]


def test_edl_events():
    bd = _bd_mixed()
    edl = interchange.to_edl(bd, "/m/media.mp4")
    assert edl.startswith("TITLE:")
    assert "FCM: NON-DROP FRAME" in edl
    ev = _events(edl)
    first = ev[0].split()
    assert first[3] == "C" and first[-2] == "01:00:00:00"
    d_events = [e for e in ev if re.search(r"\sD\s+025\s", e)]
    assert len(d_events) == 1
    # one numbered event per shot (a dissolve adds the zero-length C line under the same number)
    assert len({e.split()[0] for e in ev}) == len(bd["shots"])
    assert "* FROM CLIP NAME:" in edl
    assert "transition not determined" in edl and "dip_black" in edl
    last = ev[-1].split()
    assert last[-1] == "01:00:07:00"  # 7 s at 25 fps


def test_edl_dissolve_geometry():
    bd = make_bd(cuts=(2.0,), duration=4.0, kinds={1: ("dissolve", 1.0)})
    ev = _events(interchange.to_edl(bd, "/m/media.mp4"))
    # event 1: 2s minus half dissolve; event 2: zero-length C then D 025
    assert ev[0].split()[-2:] == ["01:00:00:00", "01:00:01:13"]
    c0, d0 = ev[1].split(), ev[2].split()
    assert c0[3] == "C" and c0[-2] == c0[-1] == "01:00:01:13"
    assert d0[3] == "D" and d0[4] == "025"
    assert d0[-2:] == ["01:00:01:13", "01:00:04:00"]
    assert d0[-4] == "00:00:01:13"  # source in = shot start - half dissolve (same media, handles)


def test_ntsc_rate_no_drift():
    bd = make_bd("30000/1001", cuts=(300.0,), duration=600.0, beats=())
    edl = interchange.to_edl(bd, "/m/media.mp4")
    assert "FCM: DROP FRAME" in edl
    ev = _events(edl)
    assert ev[0].split()[-2] == "01:00:00;00"
    assert ev[-1].split()[-1] == "01:10:00;00"
    # the second shot starts where the first ends, drop-frame aware
    assert ev[1].split()[-2] == ev[0].split()[-1]
    d = interchange.to_otio(bd, "/m/media.mp4")
    clips = d["tracks"]["children"][0]["children"]
    assert sum(c["source_range"]["duration"]["value"] for c in clips) == 17982


def test_timecode_roundtrip_df():
    for rate_num, rate_den in ((30000, 1001), (60000, 1001), (25, 1), (24000, 1001)):
        tc = interchange.Timecode(rate_num, rate_den)
        for f in (0, 1, 1799, 1800, 1801, 17981, 17982, 107892, 215784 + 5):
            assert tc.parse(tc.format(f)) == f
    assert interchange.Timecode(30000, 1001).format(1800) == "00:01:00;02"
    assert interchange.Timecode(30000, 1001).start_frames() == 107892


def test_skip_when_shots_not_determined(tmp_path):
    bd = make_bd()
    bd["shots"] = []
    bd["low_resolution"] = True
    assert interchange.skip_reason(bd)
    assert assemble.write_timelines(bd, tmp_path, ["otio", "fcpxml", "edl"]) == []
    assert not list(tmp_path.glob("timeline.*"))
    bd2 = make_bd()
    bd2["video"]["fps_rational"] = "0/1"
    assert interchange.skip_reason(bd2)
    assert interchange.skip_reason(make_bd()) is None


def test_write_timelines_files_and_media_discovery(tmp_path):
    out = tmp_path / "breakdown"
    out.mkdir()
    (tmp_path / "media.mp4").write_bytes(b"x")
    paths = assemble.write_timelines(make_bd(), out, ["otio", "fcpxml", "edl"])
    assert [p.name for p in paths] == ["timeline.otio", "timeline.fcpxml", "timeline.edl"]
    assert str(tmp_path / "media.mp4") in (out / "timeline.edl").read_text()
    assert json.loads((out / "timeline.otio").read_text())["OTIO_SCHEMA"] == "Timeline.1"


def test_single_shot_clip():
    bd = make_bd(cuts=(), duration=3.0)
    d = interchange.to_otio(bd, "/m/media.mp4")
    assert len(d["tracks"]["children"][0]["children"]) == 1
    assert len(_events(interchange.to_edl(bd, "/m/media.mp4"))) == 1
    ET.fromstring(interchange.to_fcpxml(bd, "/m/media.mp4"))


def test_skip_reason_text():
    assert cli  # cli prints this reason when nothing was written
    bd = make_bd()
    bd["shots"] = []
    assert interchange.skip_reason(bd).startswith("no shots")


def test_edl_parses_with_otio_cmx3600_adapter_when_available():
    otio = pytest.importorskip("opentimelineio")
    try:
        otio.adapters.from_name("cmx_3600")
    except Exception:  # noqa: BLE001 - adapter ships as a separate package
        pytest.skip("cmx_3600 adapter not installed")
    bd = _bd_mixed()
    tl = otio.adapters.read_from_string(interchange.to_edl(bd, "/m/media.mp4"), "cmx_3600", rate=25)
    assert len(list(tl.find_clips())) == len(bd["shots"])
    assert tl.duration().value == 7 * 25
