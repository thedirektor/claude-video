"""Resume state: params signature stability + stage roundtrip + sig gating."""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import state  # noqa: E402


def test_signature_is_stable_and_order_independent():
    a = state.params_signature("url", {"detail": "balanced", "start": "0:30"})
    b = state.params_signature("url", {"start": "0:30", "detail": "balanced"})
    assert a == b
    assert len(a) == 16


def test_signature_changes_with_params():
    a = state.params_signature("url", {"detail": "balanced"})
    b = state.params_signature("url", {"detail": "efficient"})
    c = state.params_signature("url2", {"detail": "balanced"})
    assert a != b and a != c


def test_stage_roundtrip(tmp_path):
    sig = state.params_signature("url", {"detail": "balanced"})
    payload = {"video_path": "/x/video.mp4", "downloaded": True}
    state.save_stage(tmp_path, "download", payload, sig)
    loaded = state.load_stage(tmp_path, "download", sig)
    assert loaded == payload


def test_load_returns_none_on_sig_mismatch(tmp_path):
    state.save_stage(tmp_path, "download", {"a": 1}, "sigA")
    assert state.load_stage(tmp_path, "download", "sigB") is None


def test_load_returns_none_when_absent(tmp_path):
    assert state.load_stage(tmp_path, "download", "sig") is None


def test_load_tolerates_corrupt_file(tmp_path):
    (tmp_path / "stage_download.json").write_text("{not json", encoding="utf-8")
    assert state.load_stage(tmp_path, "download", "sig") is None


def test_load_stage_swallows_permission_error(tmp_path, monkeypatch):
    def boom(self):
        raise PermissionError("denied")
    monkeypatch.setattr(state.Path, "exists", boom)
    assert state.load_stage(tmp_path, "download", "sig") is None


def test_load_returns_none_on_non_dict_json(tmp_path):
    (tmp_path / "stage_download.json").write_text("null", encoding="utf-8")
    assert state.load_stage(tmp_path, "download", "sig") is None


def test_clear_stages_removes_only_stage_files(tmp_path):
    (tmp_path / "stage_download.json").write_text("{}", encoding="utf-8")
    (tmp_path / "stage_transcript.json").write_text("{}", encoding="utf-8")
    (tmp_path / "video.info.json").write_text("{}", encoding="utf-8")
    state.clear_stages(tmp_path)
    assert not (tmp_path / "stage_download.json").exists()
    assert not (tmp_path / "stage_transcript.json").exists()
    assert (tmp_path / "video.info.json").exists()  # non-stage file untouched
