from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from breakdown import assemble, cli  # noqa: E402
from breakdown import shots as S  # noqa: E402
from breakdown import transitions as T  # noqa: E402
from breakdown_fixtures import make_reference  # noqa: E402
from memory import bundle, index, llm, retrieval, store  # noqa: E402
from memory_fakes import KEY, fake_run, seg  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg needed")


@pytest.fixture(scope="module")
def ref_clip(tmp_path_factory):
    p = tmp_path_factory.mktemp("asm") / "ref.mp4"
    truth = make_reference(p)
    return p, truth


@pytest.fixture
def stored(conn, tmp_path, ref_clip):
    clip, truth = ref_clip
    root = tmp_path / "st"
    index.record_run(conn, fake_run(title="Sale clip", duration=truth["duration"],
                                    segments=[seg(0, 3, "hello sale")]))
    vdir = store.video_dir(root, KEY)
    vdir.mkdir(parents=True)
    shutil.copy(clip, vdir / "media.mp4")
    return root


@pytest.fixture
def bd(conn, stored):
    return assemble.build(conn, KEY, store_dir=stored)


def _walk(node, path=""):
    if isinstance(node, dict):
        if "evidence" in node:
            yield path, node
        for k, v in node.items():
            yield from _walk(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, f"{path}[{i}]")


def test_every_element_has_evidence(bd):
    must = ["video", "cuts", "rhythm", "typography", "audio", "audio.segments", "video.active_aspect"]
    for dotted in must:
        node = bd
        for part in dotted.split("."):
            node = node[part]
        assert "evidence" in node, dotted
    for s in bd["shots"]:
        for part in (s, s["camera"], s["look"]):
            assert "evidence" in part
        if s["transition"]:
            assert "evidence" in s["transition"]
    for tr in bd["typography"]["tracks"]:
        assert "evidence" in tr
    nodes = list(_walk(bd))
    assert len(nodes) > 10
    for path, n in nodes:
        ev = n["evidence"]
        assert ev["method"] in assemble.METHODS, path
        assert 0 <= ev["confidence"] <= 1 and ev["t_start"] <= ev["t_end"], path
        if n.get("value") == "not determined" or n.get("tempo_bpm") == "not determined":
            assert ev["confidence"] == 0, path
    assert bd["schema"] == "watch.breakdown/1"


def test_reference_content(bd):
    assert len(bd["shots"]) in (3, 4)
    assert bd["video"]["fps_rational"] == "25/1"
    assert bd["audio"]["tempo_bpm"] != "not determined"
    assert bd["audio"]["segments"]["speech"][0]["evidence"]["method"] == "transcript"
    assert bd["audio"]["segments"]["music"]["evidence"]["method"] == "measured"
    # every boundary has exactly one transition
    assert all(s["transition"] for s in bd["shots"][1:]) and bd["shots"][0]["transition"] is None


def test_brief_starts_with_contract(bd):
    first = assemble.brief_md(bd).splitlines()[0]
    assert first == ("> **Build contract:** follow only what this breakdown states. Anything marked "
                     "_not determined_ stays an explicit placeholder in what you build. Never fill gaps "
                     "with taste or assumptions.")


def test_not_determined_rendered(bd):
    md = assemble.brief_md(bd)
    assert "_not determined_" in md  # font family is never determined
    assert "not determined" not in md.replace("_not determined_", "")


def test_interpretation_isolated(conn, stored):
    calls = []

    def fake(system, user, max_tokens=700):
        calls.append(user)
        return llm.LLMReply("Punchy red-green-blue sequence.", "fake", "m1", 100, 20)

    import spend
    spend.reset()
    out = assemble.build(conn, KEY, store_dir=stored, interpret=True, complete=fake)
    assert len(calls) == 1
    assert out["interpretation"]["text"] == "Punchy red-green-blue sequence."
    assert out["interpretation"]["evidence"]["method"] == "interpretation"
    assert out["interpretation"]["made_from"] == "measured fields (no frames)"
    assert "frame_times" not in out["interpretation"]
    assert "made from: measured fields (no frames)" in assemble.brief_md(out)
    assert "Punchy" not in json.dumps({k: v for k, v in out.items() if k != "interpretation"})
    assert [i["item"] for i in spend.items()] == ["breakdown_interpretation"]
    free = assemble.build(conn, KEY, store_dir=stored)
    assert "interpretation" not in free


def test_store_and_search(conn, stored, bd):
    out = assemble.store(conn, KEY, bd, store_dir=stored)
    assert (out / "breakdown.json").is_file() and (out / "breakdown.md").read_text().startswith("> **Build")
    assert (out / "contact.png").stat().st_size > 0
    assert (store.video_dir(stored, KEY) / ".pinned").exists()
    hits = retrieval.search(conn, "SALE")
    assert any(h.kind == "note" and "Build contract" in h.text for h in hits)
    assemble.store(conn, KEY, bd, store_dir=stored)  # replace, not duplicate
    assert conn.execute("SELECT count(*) FROM notes WHERE kind = 'breakdown'").fetchone()[0] == 1


def test_bundle_carries_breakdown(conn, stored, bd, tmp_path, monkeypatch):
    conn.execute("UPDATE videos SET store_path = ? WHERE video_key = ?",
                 (store.video_dir(stored, KEY).name, KEY))
    conn.commit()
    assemble.store(conn, KEY, bd, store_dir=stored)
    z = bundle.make_bundle(conn, KEY, tmp_path / "b.zip", store_dir=stored, host="pc1")
    monkeypatch.setenv("WATCH_INDEX_DB", str(tmp_path / "central.db"))
    monkeypatch.setenv("WATCH_STORE_DIR", str(tmp_path / "central"))
    bundle.import_bundle(z, embedder=None)
    restored = store.video_dir(tmp_path / "central", KEY) / "breakdown"
    assert json.loads((restored / "breakdown.json").read_text())["schema"] == "watch.breakdown/1"
    assert (restored / "breakdown.md").is_file()


def test_reconcile_fadewhite_dip(tmp_path):
    p = tmp_path / "fw.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "color=c=red:s=320x180:r=25:d=3", "-f", "lavfi", "-i", "color=c=blue:s=320x180:r=25:d=3",
                    "-filter_complex", "[0][1]xfade=transition=fadewhite:duration=0.5:offset=2.5,format=yuv420p",
                    "-c:v", "libx264", str(p)], check=True)
    series = S.frame_series(str(p), 25.0)
    raw = S.detect_shots(series, 5.5)
    entries = T.classify(series, raw)
    shots, loose = assemble.reconcile(raw, entries)
    assert all(s["duration"] > 0.3 for s in shots), [s["duration"] for s in shots]
    assert all(s["transition"] for s in shots[1:])
    assert len(shots) == 2 and shots[1]["transition"]["kind"] == "dip_white"
    assert loose == []
    assert shots[0]["end"] == shots[1]["start"] and abs(shots[-1]["end"] - 5.5) < 1e-6
    assert 2.6 <= shots[1]["start"] <= 3.0


def _sh(i, a, b, kind=None):
    ev = {"t_start": a, "t_end": a, "method": "measured", "confidence": 0.9}
    return {"index": i, "start": a, "end": b, "duration": b - a, "evidence": ev,
            "boundary": None if kind is None else {"kind": kind, "t": a, "frames": 1, "evidence": ev}}


def _en(kind, t, d):
    return {"t": t, "kind": kind, "duration": d,
            "evidence": {"t_start": t - d / 2, "t_end": t + d / 2, "method": "measured", "confidence": 0.85}}


def test_reconcile_dip_keeps_one_boundary():
    raw = [_sh(0, 0, 2.12), _sh(1, 2.12, 2.2, "cut"), _sh(2, 2.2, 5.0, "cut")]
    entries = [_en("fade_to_white", 2.0, 0.2), _en("fade_from_white", 2.3, 0.2)]
    shots, loose = assemble.reconcile(raw, entries)
    assert len(shots) == 2 and shots[1]["transition"]["kind"] == "dip_white"
    assert shots[1]["transition"]["evidence"]["t_start"] == pytest.approx(1.9)
    assert shots[0]["end"] == 2.12 and shots[1]["end"] == 5.0 and loose == []


def test_frames_only(conn, tmp_path, monkeypatch):
    index.record_run(conn, fake_run(title="Gone", duration=10.0))
    root = tmp_path / "st2"
    out = assemble.build(conn, KEY, store_dir=root, fetch=lambda url, vdir: None)
    assert out["low_resolution"] is True and out["shots"] == []
    assert out["cuts"]["value"] == "not determined" and out["video"]["fps"] == 0.0
    assert out["rhythm"]["count"] == 0
    assert any("not available" in n for n in out["notes"])
    assert "_not determined_" in assemble.brief_md(out)


def test_cli_formats_validated(conn, stored, monkeypatch, capsys):
    assert cli.main(["breakdown", KEY, "--formats", "nope"]) == 2
    assert cli.main(["compare", "x"]) == 2  # not available yet


def test_music_windows_half_music_half_silence(conn, tmp_path):
    p = tmp_path / "half.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "color=c=gray:s=320x180:r=25:d=12", "-f", "lavfi", "-i",
                    "aevalsrc=0.8*sin(2*PI*1000*t)*lt(mod(t\\,0.5)\\,0.02)*lt(t\\,6):s=44100:d=12",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(p)], check=True)
    index.record_run(conn, fake_run(title="Half", duration=12.0, segments=[seg(0, 3, "talking")]))
    root = tmp_path / "st"
    vdir = store.video_dir(root, KEY)
    vdir.mkdir(parents=True)
    shutil.copy(p, vdir / "media.mp4")
    out = assemble.build(conn, KEY, store_dir=root)
    mu = out["audio"]["segments"]["music"]
    states = [(x["state"], x["t_start"], x["t_end"]) for x in mu["segments"]]
    assert [x[0] for x in states] == ["music", "silent"], states
    assert mu["present"] is True and mu["segments"][0]["overlaps_speech_s"] == pytest.approx(3.0)
    assert mu["segments"][1]["evidence"]["method"] == "measured"


def test_brief_low_confidence_and_units(bd):
    bd = json.loads(json.dumps(bd))
    bd["audio"]["evidence"]["confidence"] = 0.23
    bd["audio"]["tempo_alt"] = 60.0
    bd["shots"][0]["camera"]["evidence"]["confidence"] = 0.1
    bd["shots"][0]["look"]["letterbox"] = {"top": 0.1, "bottom": 0.1, "left": 0.0, "right": 0.0}
    md = assemble.brief_md(bd)
    assert "(low confidence 0.23)" in md and "(low confidence 0.10)" in md
    assert "alternative reading 60.0 BPM" in md
    assert "letterbox top 10.0% bottom 10.0% left 0.0% right 0.0%" in md
    assert "0-1 fractions of the frame" in md and "are 0-1" in md
    assert "{'top'" not in md


def test_interpret_cost_uses_payload_and_provider_price(conn, stored, monkeypatch):
    monkeypatch.setenv("WATCH_COST_CEILING_USD", "0.0000001")
    import cost
    with pytest.raises(cost.CostCeilingExceeded):
        assemble.build(conn, KEY, store_dir=stored, interpret=True,
                       complete=lambda *a, **k: llm.LLMReply("x", "fake", "m", 1, 1))
    ok = assemble.build(conn, KEY, store_dir=stored, interpret=True, allow_cost=True,
                        complete=lambda *a, **k: llm.LLMReply("x", "fake", "m", 1, 1))
    assert ok["interpretation"]["text"] == "x"


def test_cli_success_path(conn, stored, monkeypatch, capsys):
    monkeypatch.setenv("WATCH_STORE_DIR", str(stored))
    rc = cli.main(["breakdown", KEY, "--json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["schema"] == "watch.breakdown/1"
    out = store.video_dir(stored, KEY) / "breakdown"
    assert (out / "breakdown.json").is_file() and (out / "breakdown.md").is_file()
