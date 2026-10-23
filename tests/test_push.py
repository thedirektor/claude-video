"""Local → central push with an outbox (0.6.0 phase D)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import push  # noqa: E402
from memory.index import connect, record_run  # noqa: E402
from memory_fakes import KEY, fake_run, seg  # noqa: E402


@pytest.fixture
def remote(monkeypatch, tmp_path):
    monkeypatch.setenv("WATCH_REMOTE_URL", "https://watch.example.com")
    monkeypatch.setenv("WATCH_REMOTE_TOKEN", "tok")
    monkeypatch.setenv("WATCH_OUTBOX_DIR", str(tmp_path / "outbox"))
    record_run(connect(), fake_run(title="Clip", segments=[seg(0, 2, "hola")]))
    sent = []

    def post(url, path, token):
        sent.append((url, Path(path).name, token))
        return 200, {"video_key": KEY, "skipped": False}
    return sent, post


def test_push_video_sends_bundle_and_clears_outbox(remote, tmp_path):
    sent, post = remote
    r = push.push_video(KEY, post=post)
    assert r["ok"] is True and sent[0][0] == "https://watch.example.com/runs/import" and sent[0][2] == "tok"
    assert list((tmp_path / "outbox").glob("*.zip")) == []


def test_failed_push_stays_in_outbox_then_retry(remote, tmp_path):
    sent, post = remote
    assert push.push_video(KEY, post=lambda *a: (None, None))["ok"] is False
    assert len(list((tmp_path / "outbox").glob("*.zip"))) == 1
    res = push.flush_outbox(post=post)
    assert res == {"sent": 1, "failed": 0} and list((tmp_path / "outbox").glob("*.zip")) == []


def test_rejected_bundle_is_not_retried_forever(remote, tmp_path):
    sent, post = remote
    push.push_video(KEY, post=lambda *a: (400, {"detail": "bad bundle"}))
    assert list((tmp_path / "outbox").glob("*.zip")) == []
    assert len(list((tmp_path / "outbox" / "rejected").glob("*.zip"))) == 1


def test_not_configured_is_noop(monkeypatch):
    monkeypatch.delenv("WATCH_REMOTE_URL", raising=False)
    assert push.enabled() is False
    assert push.push_video(KEY, post=lambda *a: pytest.fail("not configured"))["ok"] is False


def test_push_cli_all(remote, capsys, monkeypatch):
    sent, post = remote
    monkeypatch.setattr(push, "_post", post)
    assert push.main(["--all", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["pushed"] == 1 and sent
