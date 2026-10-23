"""gemini.transcribe_youtube: timestamp parsing + never-raising Gemini call (no network)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import gemini  # noqa: E402

URL = "https://www.youtube.com/watch?v=abc123"


class _FakeModels:
    def __init__(self, response=None, exc=None):
        self.response, self.exc, self.calls = response, exc, []

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self.exc:
            raise self.exc
        return self.response


class _FakeClient:
    def __init__(self, response=None, exc=None):
        self.models = _FakeModels(response, exc)


def _resp(text, finish=None):
    cands = [SimpleNamespace(finish_reason=finish)] if finish is not None else []
    return SimpleNamespace(text=text, candidates=cands)


def test_parse_timestamped_lines_mixed_reply():
    text = (
        "Sure, here is the transcript:\n"
        "[00:00] Hello there.\n"
        "\n"
        "[0:04] Second line.\n"
        "[0:04] Duplicate timestamp line.\n"
        "not a transcript line\n"
        "[1:02:03] Over an hour in.\n"
        "That's all."
    )
    assert gemini.parse_timestamped_lines(text) == [
        {"start": 0.0, "end": 4.0, "text": "Hello there."},
        {"start": 4.0, "end": 3723.0, "text": "Second line. Duplicate timestamp line."},
        {"start": 3723.0, "end": 3728.0, "text": "Over an hour in."},
    ]


def test_parse_drops_non_monotonic():
    text = "[00:10] a\n[00:05] b\n[00:20] c\n"
    assert gemini.parse_timestamped_lines(text) == [
        {"start": 10.0, "end": 20.0, "text": "a"},
        {"start": 20.0, "end": 25.0, "text": "c"},
    ]


def test_transcribe_youtube_success_with_fake_client():
    from google.genai import types

    client = _FakeClient(_resp("[00:00] Hi.\n[00:03] Bye.\n", finish="STOP"))
    out = gemini.transcribe_youtube(URL, client=client)
    assert out is not None
    segs, model, approx = out
    assert model == gemini.DEFAULT_MODEL
    assert approx is False
    assert [s["text"] for s in segs] == ["Hi.", "Bye."]
    call = client.models.calls[0]
    assert call["model"] == gemini.DEFAULT_MODEL
    assert call["config"].temperature == 0
    assert call["config"].max_output_tokens == 65536
    parts = call["contents"][0].parts
    assert parts[0].file_data.file_uri == URL
    assert parts[1].text == gemini.TRANSCRIPT_PROMPT


def test_transcribe_youtube_none_on_exception():
    client = _FakeClient(exc=RuntimeError("boom"))
    assert gemini.transcribe_youtube(URL, client=client) is None


def test_transcribe_youtube_none_on_empty():
    assert gemini.transcribe_youtube(URL, client=_FakeClient(_resp("", finish="SAFETY"))) is None
    assert gemini.transcribe_youtube(URL, client=_FakeClient(_resp("no timestamps here"))) is None


def test_transcribe_youtube_partial_on_max_tokens(capsys):
    client = _FakeClient(_resp("[00:00] Partial.\n", finish="FinishReason.MAX_TOKENS"))
    out = gemini.transcribe_youtube(URL, client=client)
    assert out is not None and out[0][0]["text"] == "Partial."
    assert "truncated" in capsys.readouterr().err


def test_transcribe_youtube_none_without_key(monkeypatch):
    monkeypatch.setattr(gemini, "load_api_key", lambda: None)
    assert gemini.transcribe_youtube(URL) is None


def test_media_resolution_override():
    from google.genai import types

    client = _FakeClient(_resp("[00:00] Hi.\n", finish="STOP"))
    gemini.transcribe_youtube(URL, client=client, media_resolution="low")
    assert client.models.calls[0]["config"].media_resolution == types.MediaResolution.MEDIA_RESOLUTION_LOW
    client = _FakeClient(_resp("[00:00] Hi.\n", finish="STOP"))
    gemini.transcribe_youtube(URL, client=client, media_resolution=None)
    assert client.models.calls[0]["config"].media_resolution is None


def test_parse_bullets_bold_and_ranges():
    text = (
        "- [00:01] dash.\n"
        "* [00:02] star.\n"
        "**[00:03]** bold.\n"
        "[00:04 - 00:06] range.\n"
        "- **[1:00:00 - 1:00:05]** both.\n"
    )
    segs = gemini.parse_timestamped_lines(text)
    assert [(s["start"], s["text"]) for s in segs] == [
        (1.0, "dash."), (2.0, "star."), (3.0, "bold."), (4.0, "range."), (3600.0, "both."),
    ]


def test_unparseable_reply_logs_first_80_chars(capsys):
    reply = "I cannot transcribe this video because " + "x" * 200
    assert gemini.transcribe_youtube(URL, client=_FakeClient(_resp(reply, finish="STOP"))) is None
    err = capsys.readouterr().err
    assert reply[:80] in err
    assert reply[:81] not in err


def _lines(n, step, words_per_line):
    body = " ".join(["w"] * words_per_line)
    return "".join(f"[{(i * step) // 60:02d}:{(i * step) % 60:02d}] {body}\n" for i in range(n))


def test_normal_wpm_not_approximate():
    # 20 lines, 6s apart (114s span), 15 words each -> ~158 wpm
    out = gemini.transcribe_youtube(URL, client=_FakeClient(_resp(_lines(20, 6, 15), finish="STOP")),
                                    duration=200)
    assert out is not None and out[2] is False


def test_high_wpm_flagged_approximate(capsys):
    # 20 lines, 2s apart (38s span), 15 words each -> ~473 wpm
    out = gemini.transcribe_youtube(URL, client=_FakeClient(_resp(_lines(20, 2, 15), finish="STOP")))
    assert out is not None and out[2] is True
    assert "approximate" in capsys.readouterr().err


def test_past_duration_flagged_approximate():
    # last start 114s, duration 100s -> > 105% -> approximate
    out = gemini.transcribe_youtube(URL, client=_FakeClient(_resp(_lines(20, 6, 15), finish="STOP")),
                                    duration=100)
    assert out is not None and out[2] is True
