"""watch.main() writes work/report.md and pushes artifacts to the library
unless --no-save is passed. Offline: local silent clip, --no-whisper, and
library.save_artifacts is monkeypatched to record its call instead of
touching the network.
"""
from __future__ import annotations

import sys
from pathlib import Path

WATCH_SCRIPTS = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(WATCH_SCRIPTS))

import watch  # noqa: E402


def _record_save(monkeypatch):
    calls = []

    def fake_save_artifacts(video_path, frames, report_path, ocr_text, title, video_id, cfg=None):
        calls.append(
            {
                "video_path": video_path,
                "frames": frames,
                "report_path": report_path,
                "ocr_text": ocr_text,
                "title": title,
                "video_id": video_id,
            }
        )
        return ["fake save summary line"]

    monkeypatch.setattr(watch.library, "save_artifacts", fake_save_artifacts)
    return calls


def test_writes_report_and_saves_by_default(cut_clip: Path, monkeypatch, tmp_path):
    # conftest sets WATCH_NO_SAVE=1 for session-wide isolation; this test
    # exercises the default-save path explicitly, and is safe because
    # save_artifacts is mocked below (never touches the real library).
    monkeypatch.delenv("WATCH_NO_SAVE", raising=False)
    calls = _record_save(monkeypatch)
    argv = [
        "watch.py", str(cut_clip), "--no-whisper", "--no-scene-detect",
        "--out-dir", str(tmp_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert watch.main() == 0

    report_path = tmp_path / "report.md"
    assert report_path.exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "watch: video report" in report_text

    assert len(calls) == 1
    call = calls[0]
    assert call["report_path"] == str(report_path)
    assert call["title"]
    assert call["video_id"]


def test_no_save_skips_library_push(cut_clip: Path, monkeypatch, tmp_path):
    calls = _record_save(monkeypatch)
    argv = [
        "watch.py", str(cut_clip), "--no-whisper", "--no-scene-detect",
        "--out-dir", str(tmp_path), "--no-save",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert watch.main() == 0

    assert (tmp_path / "report.md").exists()
    assert calls == []
