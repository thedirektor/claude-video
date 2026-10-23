"""LLM provider chain (0.5.1 Task 1)."""
from __future__ import annotations

import io
import json

import pytest

from memory import llm
from memory.llm import LLMReply


def test_chain_default_skips_headroom_without_key(monkeypatch):
    monkeypatch.delenv("WATCH_LLM_CHAIN", raising=False)
    monkeypatch.delenv("WATCH_LLM_API_KEY", raising=False)
    assert llm.chain() == ["gemini", "openrouter"]
    monkeypatch.setenv("WATCH_LLM_API_KEY", "k")
    assert llm.chain() == ["headroom", "gemini", "openrouter"]


def test_chain_env_order_and_unknown_dropped(monkeypatch):
    monkeypatch.setenv("WATCH_LLM_CHAIN", " OpenRouter, bogus ,gemini")
    assert llm.chain() == ["openrouter", "gemini"]


def test_complete_falls_through_errors_and_empty(monkeypatch, capsys):
    monkeypatch.setenv("WATCH_LLM_CHAIN", "a,b,c")
    monkeypatch.setattr(llm, "KNOWN", ("a", "b", "c"))

    def boom(*_):
        raise RuntimeError("down")

    providers = {
        "a": boom,
        "b": lambda *_: LLMReply("", "b", "m"),
        "c": lambda s, u, n: LLMReply(f"ok {n}", "c", "m", 3, 4),
    }
    r = llm.complete("sys", "user", max_tokens=50, providers=providers)
    assert r == LLMReply("ok 50", "c", "m", 3, 4)
    assert capsys.readouterr().err.count("[watch] llm") == 2


def test_complete_all_fail_returns_none(monkeypatch):
    monkeypatch.setenv("WATCH_LLM_CHAIN", "gemini")
    assert llm.complete("s", "u", providers={"gemini": lambda *_: None}) is None
    assert llm.complete("s", "u", providers={}) is None


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_openai_compatible_parses_usage(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout):
        seen["url"], seen["body"], seen["auth"] = req.full_url, json.loads(req.data), req.get_header("Authorization")
        return _Resp(json.dumps({
            "choices": [{"message": {"content": " hola [E1] "}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 5},
        }).encode())

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    r = llm._openai_compatible("http://h:1/", "key", "mod", "sys", "usr", 99, "headroom")
    assert r == LLMReply("hola [E1]", "headroom", "mod", 11, 5)
    assert seen["url"] == "http://h:1/v1/chat/completions" and seen["auth"] == "Bearer key"
    assert seen["body"]["temperature"] == 0 and seen["body"]["max_tokens"] == 99
    assert [m["role"] for m in seen["body"]["messages"]] == ["system", "user"]
