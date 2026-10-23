"""Actual spend ledger (0.6.0 phase B)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import gemini  # noqa: E402
import spend  # noqa: E402


@pytest.fixture(autouse=True)
def _reset():
    spend.reset()
    yield
    spend.reset()


def test_tokens_and_hours_priced_from_table():
    spend.record("gemini", "transcript", price_key="gemini:gemini-3.1-flash-lite", tokens_in=1_000_000, tokens_out=100_000)
    spend.record("openai", "whisper", price_key="openai:whisper-1", seconds=600)
    spend.record("transcriptapi", "transcript", credits=1)
    assert spend.total() == pytest.approx(0.125 + 0.075 + 0.06)
    assert spend.credits() == {"transcriptapi": 1}


def test_flush_writes_spend_json(tmp_path):
    spend.record("groq", "whisper", price_key="groq:whisper-large-v3", seconds=3600)
    out = spend.flush(tmp_path)
    data = json.loads(out.read_text())
    assert data["total_usd"] == pytest.approx(0.111) and data["items"][0]["provider"] == "groq"


def test_flush_empty_ledger_is_zero(tmp_path):
    assert json.loads(spend.flush(tmp_path).read_text()) == {"total_usd": 0.0, "credits": {}, "items": []}


def test_gemini_response_usage_is_recorded():
    resp = SimpleNamespace(usage_metadata=SimpleNamespace(prompt_token_count=200_000, candidates_token_count=4_000))
    spend.record_gemini(resp, "gemini-3.1-flash-lite", "visual log")
    item = spend.items()[0]
    assert item["tokens_in"] == 200_000 and item["usd"] == pytest.approx(200_000 * 0.125e-6 + 4_000 * 0.75e-6)


def test_gemini_visual_log_records_spend():
    class Models:
        def generate_content(self, **k):
            return SimpleNamespace(text="[00:01] a cat", candidates=[],
                                   usage_metadata=SimpleNamespace(prompt_token_count=1000, candidates_token_count=10))
    gemini.visual_log("https://www.youtube.com/watch?v=abc", client=SimpleNamespace(models=Models()))
    assert spend.items()[0]["item"] == "visual log" and spend.items()[0]["tokens_in"] == 1000
