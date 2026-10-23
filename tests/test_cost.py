"""Cost plan + ceiling (0.6.0 phase B)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import cost  # noqa: E402

YT = "https://www.youtube.com/watch?v=XUqrvbsTfck"


def _args(**kw):
    base = dict(source=YT, backend="claude", detail="balanced", max_frames=None, start=None, end=None,
                whisper=None, diarize=False, no_whisper=False, no_transcriptapi=False,
                no_gemini_transcript=False, gemini_model=None, openrouter_vision_model=None, audio=None,
                resolution=512, llm_titles=False, exports=None)
    base.update(kw)
    return SimpleNamespace(**base)


def _keys(monkeypatch, transcriptapi=False, gemini=True, groq=False, openai=False, assembly=False):
    monkeypatch.setattr(cost, "_has_key", lambda name: {
        "TRANSCRIPTAPI_API_KEY": transcriptapi, "GEMINI_API_KEY": gemini, "GROQ_API_KEY": groq,
        "OPENAI_API_KEY": openai, "ASSEMBLYAI_API_KEY": assembly, "OPENROUTER_API_KEY": True}.get(name, False))


def _item(plan, provider):
    return next((c for c in plan["costs"] if c["provider"] == provider), None)


def test_youtube_with_captions_is_free(monkeypatch):
    _keys(monkeypatch)
    p = cost.plan(_args(), {"duration": 1200, "has_captions": True, "youtube": True})
    assert p["transcript_route"] == "captions" and p["total_usd"] == 0
    assert p["frames_est"] == 100 and p["informational"]["claude_image_tokens"] > 0
    assert any("youtube" in d.lower() for d in p["leaves_box"])


def test_youtube_without_captions_uses_gemini_transcript(monkeypatch):
    _keys(monkeypatch)
    p = cost.plan(_args(), {"duration": 3600, "has_captions": False, "youtube": True})
    assert p["transcript_route"] == "gemini"
    g = _item(p, "gemini")
    # 3600 s x (66 video + 32 audio) tokens/s at low resolution, priced per million
    assert g["usd"] == pytest.approx(3600 * 66 * 0.125e-6 + 3600 * 32 * 0.25e-6 + 8000 * 0.75e-6, rel=0.01)
    assert p["within_ceiling"] is True


def test_transcriptapi_reports_credit_not_dollars(monkeypatch):
    _keys(monkeypatch, transcriptapi=True)
    p = cost.plan(_args(), {"duration": 600, "has_captions": True, "youtube": True})
    assert p["transcript_route"] == "transcriptapi" and p["credits"] == {"transcriptapi": 1}


def test_local_file_whisper_openai_priced_by_minute(monkeypatch):
    _keys(monkeypatch, gemini=False, openai=True)
    p = cost.plan(_args(source="/v/clip.mp4", whisper="openai"), {"duration": 600, "has_captions": False,
                                                                   "youtube": False, "has_audio": True})
    assert p["transcript_route"] == "whisper:openai"
    assert _item(p, "openai")["usd"] == pytest.approx(0.06)


def test_gemini_backend_default_resolution(monkeypatch):
    _keys(monkeypatch)
    p = cost.plan(_args(backend="gemini"), {"duration": 600, "has_captions": True, "youtube": True})
    g = _item(p, "gemini")
    assert g["usd"] == pytest.approx(600 * 258 * 0.125e-6 + 600 * 32 * 0.25e-6 + 2000 * 0.75e-6, rel=0.01)


def test_ceiling_and_override(monkeypatch):
    _keys(monkeypatch, gemini=False, openai=True)
    monkeypatch.setenv("WATCH_COST_CEILING_USD", "0.05")
    p = cost.plan(_args(source="/v/clip.mp4", whisper="openai"), {"duration": 3600, "has_audio": True})
    assert p["within_ceiling"] is False and p["ceiling_usd"] == 0.05
    with pytest.raises(cost.CostCeilingExceeded) as exc:
        cost.enforce(p)
    assert "0.05" in str(exc.value) and "--allow-cost" in str(exc.value)
    cost.enforce(p, allow=True)  # no raise


def test_price_override(monkeypatch):
    _keys(monkeypatch, gemini=False, openai=True)
    monkeypatch.setenv("WATCH_PRICES", '{"openai:whisper-1": {"per_hour": 1.0}}')
    p = cost.plan(_args(source="/v/clip.mp4", whisper="openai"), {"duration": 1800, "has_audio": True})
    assert _item(p, "openai")["usd"] == pytest.approx(0.5)


def test_focus_range_limits_duration(monkeypatch):
    _keys(monkeypatch, gemini=False, openai=True)
    p = cost.plan(_args(source="/v/clip.mp4", whisper="openai", start="1:00", end="2:00"), {"duration": 3600, "has_audio": True})
    assert p["duration_s"] == 60 and _item(p, "openai")["usd"] == pytest.approx(0.006)


def test_render_mentions_total_and_route(monkeypatch):
    _keys(monkeypatch)
    text = cost.render(cost.plan(_args(), {"duration": 1200, "has_captions": True, "youtube": True}))
    assert "captions" in text and "$0.00" in text
