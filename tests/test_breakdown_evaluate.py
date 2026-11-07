"""Eval harness (breakdown/evaluate.py): cut matching, production scoring, labels loading."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skills" / "watch" / "scripts"))

from breakdown import evaluate  # noqa: E402


def test_score_cuts_tolerance_one_to_one_and_uncertain():
    c = evaluate.score_cuts([1.0, 1.02, 2.5, 4.0, 6.0], [1.0, 2.52, 5.0], [6.0], tol=0.05)
    assert c["hits"] == 2 and c["missed"] == [5.0]
    assert c["false_positives"] == [1.02, 4.0]  # the second detection near 1.0 does not reuse the match
    assert c["precision"] == pytest.approx(2 / 4) and c["recall"] == pytest.approx(2 / 3)


def test_score_cuts_empty_truth_is_all_false_positives():
    c = evaluate.score_cuts([3.0], [], [], tol=0.1)
    assert c["recall"] is None and c["precision"] == 0.0 and c["false_positives"] == [3.0]


def _shot(t0, t1, label, cands=()):
    return {"start": t0, "end": t1, "production": {"label": label,
                                                   "candidates": [{"label": c, "confidence": 0.4} for c in cands]}}


def test_score_production_weights_by_overlap_and_never_credits_undetermined():
    shots = [_shot(0, 2, "synthetic"), _shot(2, 4, "not determined", ["ai_video"]),
             _shot(4, 6, "camera"), _shot(6, 8, "ai_generated")]
    p = evaluate.score_production(shots, [[0, 4, "synthetic"], [3, 6, "ai_video"], [6, 8, ["ai_video", "camera"]]])
    # labelled: 0-2 synthetic right, 2-4 undetermined (once vs synthetic, 3-4 also vs ai_video), 4-6 camera wrong,
    # 6-8 ai_generated satisfies ai_video
    assert p["labelled_s"] == pytest.approx(9.0)
    assert p["determined_share"] == pytest.approx(6 / 9, abs=1e-3)
    assert p["accuracy_determined"] == pytest.approx(4 / 6, abs=1e-3)
    assert p["undetermined_lean_right"] == pytest.approx(1 / 3, abs=1e-3)


def test_run_with_fake_analyser_and_relative_paths(tmp_path):
    labels = {"tolerance_frames": 2, "videos": [
        {"id": "a", "path": "a.mp4", "cuts": [1.0, 2.0], "production": [[0, 3, "camera"]]},
        {"id": "broken", "path": "/nope.mp4", "cuts": [1.0]}]}
    lp = tmp_path / "labels.json"
    lp.write_text(json.dumps(labels))
    seen = []

    def fake(path):
        seen.append(path)
        if "nope" in path:
            raise OSError("missing")
        return {"fps": 25.0, "duration": 3.0, "shots": [
            _shot(0, 1.0, "not determined", ["camera"]),
            {**_shot(1.0, 2.0, "camera"), "boundary": {"t": 1.0}},
            {**_shot(2.0, 3.0, "camera"), "boundary": {"t": 2.04}}]}

    res = evaluate.run(lp, analyse_fn=fake)
    assert seen[0] == str(tmp_path / "a.mp4")
    assert res["videos"][1]["error"].startswith("OSError")
    assert res["summary"]["cuts"]["recall"] == 1.0 and res["summary"]["cuts"]["precision"] == 1.0
    assert res["summary"]["production"]["accuracy_determined"] == 1.0
    md = evaluate.summary_md(res)
    assert "| a | 3 | 100% | 100% |" in md and "broken" in md


def test_load_labels_rejects_bad_files(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('{"videos": [{"id": "x"}]}')
    with pytest.raises(evaluate.LabelsError):
        evaluate.load_labels(p)
    p.write_text("[]")
    with pytest.raises(evaluate.LabelsError):
        evaluate.load_labels(p)


def test_evaluate_cli_usage_error(tmp_path, capsys):
    assert evaluate.main([str(tmp_path / "missing.json")]) == 2
