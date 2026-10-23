"""Gemini visual log: once-per-video on-screen notes for media-less YouTube videos."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import gemini  # noqa: E402
from memory import store  # noqa: E402
from memory.answer import answer  # noqa: E402
from memory.escalate import VISUAL_T, visual_log  # noqa: E402
from memory.index import record_run  # noqa: E402
from memory.llm import LLMReply  # noqa: E402
from memory_fakes import KEY, fake_run, seg  # noqa: E402

LOG = [{"start": 30.0, "end": 42.0, "text": "A whiteboard with the words PLAN B in red"},
       {"start": 42.0, "end": 47.0, "text": "The speaker holds a blue mug"}]


@pytest.fixture(autouse=True)
def _visual_on(monkeypatch):
    monkeypatch.setenv("WATCH_VISUAL_LOG", "1")


def _video(conn, source="https://www.youtube.com/watch?v=AAAAAAAAAAA"):
    record_run(conn, fake_run(source=source, duration=60.0, segments=[seg(0, 5, "hola a todos")]))


def _ev(conn):
    return conn.execute("SELECT evidence_version FROM videos WHERE video_key=?", (KEY,)).fetchone()[0]


def test_visual_log_writes_notes_once(conn):
    _video(conn)
    calls = []
    ev0 = _ev(conn)

    def describe(url, duration):
        calls.append((url, duration))
        return LOG
    assert visual_log(conn, KEY, describe=describe) == 2
    rows = conn.execute("SELECT kind, t_start, t_end, text FROM notes WHERE video_key=? ORDER BY t_start", (KEY,)).fetchall()
    assert [tuple(r) for r in rows] == [("visual", 30.0, 42.0, "[on screen] A whiteboard with the words PLAN B in red"),
                                         ("visual", 42.0, 47.0, "[on screen] The speaker holds a blue mug")]
    assert _ev(conn) == ev0 + 1
    assert calls == [("https://www.youtube.com/watch?v=AAAAAAAAAAA", 60.0)]
    assert conn.execute("SELECT added FROM escalations WHERE video_key=? AND t=?", (KEY, VISUAL_T)).fetchone()[0] == 2
    assert visual_log(conn, KEY, describe=describe) == 0
    assert len(calls) == 1


def test_visual_log_skips_non_youtube(conn):
    _video(conn, source="/home/u/clip.mp4")
    assert visual_log(conn, KEY, describe=lambda *a: pytest.fail("local files never go to Gemini")) == 0


def test_visual_log_skips_when_media_stored(conn, tmp_path):
    _video(conn)
    vdir = store.video_dir(tmp_path / "store", KEY)
    vdir.mkdir(parents=True)
    (vdir / "media.mp4").write_bytes(b"x")
    assert visual_log(conn, KEY, describe=lambda *a: pytest.fail("media present"),
                      store_dir=tmp_path / "store") == 0


def test_visual_log_unavailable_is_not_recorded(conn):
    _video(conn)
    assert visual_log(conn, KEY, describe=lambda *a: None) == 0
    assert conn.execute("SELECT count(*) FROM escalations").fetchone()[0] == 0


def test_visual_log_disabled(conn, monkeypatch):
    _video(conn)
    monkeypatch.setenv("WATCH_VISUAL_LOG", "0")
    assert visual_log(conn, KEY, describe=lambda *a: pytest.fail("disabled")) == 0


def test_answer_uses_visual_log_when_low(conn):
    _video(conn)
    seen = []

    def llm(system, user, max_tokens=700):
        seen.append(user)
        return LLMReply("Dice PLAN B en una pizarra [E1].", "fake", "fake-1", 10, 5)

    def vis(c, key, **k):
        return visual_log(c, key, describe=lambda *a: LOG)
    r = answer(conn, KEY, "what is written on the whiteboard", complete=llm,
               escalate_fn=lambda *a, **k: 0, visual_fn=vis)
    assert r["found"] is True and r["visual"] == 2
    assert r["citations"][0]["timestamp"] == "00:30"
    assert "PLAN B" in seen[0]


def test_answer_skips_visual_when_confident(conn):
    record_run(conn, fake_run(segments=[seg(12, 18, "esta canción habla del mar azul")]))
    r = answer(conn, KEY, "¿de qué habla la canción del mar?",
               complete=lambda *a, **k: LLMReply("Del mar [E1].", "f", "f", 1, 1),
               escalate_fn=lambda *a, **k: 0, visual_fn=lambda *a, **k: pytest.fail("not needed"))
    assert r["found"] is True and r["visual"] == 0


class _Models:
    def __init__(self, text):
        self.text, self.calls = text, []

    def generate_content(self, model, contents, config):
        self.calls.append(contents)
        return SimpleNamespace(text=self.text, candidates=[])


def test_gemini_visual_log_parses_lines():
    client = SimpleNamespace(models=_Models("[00:30] A whiteboard: PLAN B\n[0:42] A blue mug\nnoise"))
    segs = gemini.visual_log("https://www.youtube.com/watch?v=abc", duration=60, client=client)
    assert [(s["start"], s["text"]) for s in segs] == [(30.0, "A whiteboard: PLAN B"), (42.0, "A blue mug")]
    prompt = client.models.calls[0][0].parts[1].text
    assert "on screen" in prompt.lower() and "not transcribe" in prompt.lower()


def test_gemini_visual_log_never_raises():
    class Boom:
        def generate_content(self, *a, **k):
            raise RuntimeError("quota")
    assert gemini.visual_log("https://www.youtube.com/watch?v=abc", client=SimpleNamespace(models=Boom())) is None
