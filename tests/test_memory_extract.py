"""Deterministic chapters + cited notes from the index (0.5.3 Task 1)."""
from __future__ import annotations

import re

from memory import extract
from memory.index import record_run
from memory.llm import LLMReply
from memory_fakes import KEY, fake_run, fr, seg


def _talk(conn, duration=600.0, frames=()):
    """A 10-minute talk: one sentence every 10 s, a long pause at 300 s, a speaker change at 450 s."""
    segs = []
    for i in range(60):
        t = i * 10.0
        if 300 <= t < 310:
            continue  # the pause
        speaker = "B" if t >= 450 else "A"
        segs.append(seg(t, t + 6, f"Frase número {i} sobre el tema {i // 15} con datos 20{i:02d}.", speaker))
    record_run(conn, fake_run(duration=duration, segments=segs, frames=list(frames)))


def test_chapters_count_bounds_and_cover_video(conn):
    _talk(conn)
    chs = extract.chapters(conn, KEY)
    assert 2 <= len(chs) <= 12
    assert chs[0]["t_start"] == 0.0
    assert chs[-1]["t_end"] == 600.0
    for a, b in zip(chs, chs[1:]):
        assert a["t_end"] == b["t_start"]


def test_chapters_prefer_pause_and_speaker_change(conn):
    _talk(conn)
    starts = [c["t_start"] for c in extract.chapters(conn, KEY)]
    assert 310.0 in starts  # first sentence after the long pause
    assert 450.0 in starts  # speaker change


def test_chapter_titles_and_cited_bullets(conn):
    _talk(conn)
    chs = extract.chapters(conn, KEY)
    assert chs[0]["title"].startswith("Frase número 0")
    assert len(chs[0]["title"]) <= 70
    for c in chs:
        assert 2 <= len(c["bullets"]) <= 5
        for b in c["bullets"]:
            m = re.search(r"\((\d\d):(\d\d)\)$", b["text"])
            assert m, b
            t = int(m.group(1)) * 60 + int(m.group(2))
            assert c["t_start"] <= t < c["t_end"] and t == int(b["t"])


def test_chapters_stored_as_notes_and_replaced(conn):
    _talk(conn)
    ev = lambda: conn.execute("SELECT evidence_version FROM videos").fetchone()[0]  # noqa: E731
    v0 = ev()
    first = extract.store_chapters(conn, KEY)
    v1 = ev()
    assert v1 == v0 + 1
    rows = conn.execute("SELECT t_start, text FROM notes WHERE kind='chapter' ORDER BY t_start").fetchall()
    assert len(rows) == len(first) and rows[0][1] == first[0]["title"]
    extract.store_chapters(conn, KEY)  # unchanged → no duplicate rows, no bump
    assert conn.execute("SELECT count(*) FROM notes WHERE kind='chapter'").fetchone()[0] == len(first)
    assert ev() == v1


def test_frames_only_video_uses_scene_times(conn):
    frames = [fr(t, None, None, kind="scene") for t in (0, 40, 95, 160, 230)]
    record_run(conn, fake_run(duration=240.0, segments=[], frames=frames))
    chs = extract.chapters(conn, KEY)
    assert len(chs) >= 1 and chs[0]["title"].startswith("Chapter 1")
    assert all(c["t_start"] in (0, 40, 95, 160, 230) for c in chs)


def test_short_video_single_chapter(conn):
    record_run(conn, fake_run(duration=40.0, segments=[seg(0, 4, "Hola."), seg(5, 9, "Adiós amigos.")]))
    chs = extract.chapters(conn, KEY)
    assert len(chs) == 1 and chs[0]["t_end"] == 40.0
    assert 1 <= len(chs[0]["bullets"]) <= 5


def test_ocr_lines_can_be_bullets(conn):
    record_run(conn, fake_run(duration=60.0, segments=[seg(0, 4, "eh")],
                              frames=[fr(20, None, "PRECIO TOTAL 49 EUROS POR MES")]))
    bullets = extract.chapters(conn, KEY)[0]["bullets"]
    assert any("PRECIO TOTAL 49 EUROS" in b["text"] and b["text"].endswith("(00:20)") for b in bullets)


def test_llm_titles_used_when_valid(conn):
    _talk(conn)
    n = len(extract.chapters(conn, KEY))

    def complete(system, user, max_tokens=700):
        return LLMReply("\n".join(f"{i + 1}. Título {i + 1}" for i in range(n)), "fake", "f", 1, 1)
    chs = extract.chapters(conn, KEY, llm_titles=True, complete=complete)
    assert [c["title"] for c in chs] == [f"Título {i + 1}" for i in range(n)]


def test_llm_titles_fall_back_on_bad_reply(conn):
    _talk(conn)
    plain = [c["title"] for c in extract.chapters(conn, KEY)]
    chs = extract.chapters(conn, KEY, llm_titles=True,
                           complete=lambda *a, **k: LLMReply("nope", "f", "f", 1, 1))
    assert [c["title"] for c in chs] == plain
    chs = extract.chapters(conn, KEY, llm_titles=True, complete=lambda *a, **k: None)
    assert [c["title"] for c in chs] == plain


def test_notes_md(conn):
    _talk(conn)
    md = extract.notes_md(conn, KEY)
    assert md.startswith("# ")
    assert md.count("\n## ") == len(extract.chapters(conn, KEY))
    assert re.search(r"^- .+\(\d\d:\d\d\)$", md, re.M)
