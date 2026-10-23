"""Small text-LLM client for grounded answers (and later cheap text tasks).

A configurable chain of providers is tried in order (``WATCH_LLM_CHAIN``,
default ``headroom,gemini,openrouter``); the first non-empty reply wins. Any
failure moves on to the next provider with one stderr line; when every
provider fails the caller gets ``None`` and falls back to evidence only.
Nothing here ever raises.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from dataclasses import dataclass

from config import setting

KNOWN = ("headroom", "gemini", "openrouter")
DEFAULT_CHAIN = "headroom,gemini,openrouter"


@dataclass
class LLMReply:
    text: str
    provider: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0


def _timeout() -> float:
    try:
        return float(setting("WATCH_LLM_TIMEOUT", "30"))
    except ValueError:
        return 30.0


def chain() -> list[str]:
    names = [n.strip().lower() for n in (setting("WATCH_LLM_CHAIN", DEFAULT_CHAIN) or "").split(",")]
    out = [n for n in names if n in KNOWN]
    if not setting("WATCH_LLM_API_KEY"):
        out = [n for n in out if n != "headroom"]
    return out


def _openai_compatible(base_url: str, api_key: str, model: str, system: str, user: str,
                       max_tokens: int, provider: str) -> LLMReply | None:
    body = json.dumps({
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions", data=body, method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=_timeout()) as resp:
        data = json.loads(resp.read())
    text = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
    usage = data.get("usage") or {}
    return LLMReply(text, provider, model, int(usage.get("prompt_tokens") or 0),
                    int(usage.get("completion_tokens") or 0))


def _headroom(system: str, user: str, max_tokens: int) -> LLMReply | None:
    return _openai_compatible(
        setting("WATCH_LLM_BASE_URL", "http://127.0.0.1:8789"), setting("WATCH_LLM_API_KEY"),
        setting("WATCH_LLM_MODEL", "gemini-3.1-flash-lite"), system, user, max_tokens, "headroom")


def _openrouter(system: str, user: str, max_tokens: int) -> LLMReply | None:
    import openrouter

    key = openrouter.load_api_key()
    if not key:
        return None
    return _openai_compatible(
        "https://openrouter.ai/api", key,
        setting("WATCH_LLM_OPENROUTER_MODEL", "google/gemini-2.5-flash-lite"),
        system, user, max_tokens, "openrouter")


def _gemini(system: str, user: str, max_tokens: int) -> LLMReply | None:
    import gemini

    key = gemini.load_api_key()
    if not key:
        return None
    from google import genai
    from google.genai import types

    model = setting("WATCH_LLM_GEMINI_MODEL", "gemini-3.1-flash-lite")
    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=int(_timeout() * 1000)))
    resp = client.models.generate_content(
        model=model, contents=user,
        config=types.GenerateContentConfig(system_instruction=system, temperature=0,
                                           max_output_tokens=max_tokens))
    usage = getattr(resp, "usage_metadata", None)
    return LLMReply((resp.text or "").strip(), "gemini", model,
                    int(getattr(usage, "prompt_token_count", 0) or 0),
                    int(getattr(usage, "candidates_token_count", 0) or 0))


_PROVIDERS = {"headroom": _headroom, "gemini": _gemini, "openrouter": _openrouter}


def complete(system: str, user: str, *, max_tokens: int = 700,
             providers: dict | None = None) -> LLMReply | None:
    providers = _PROVIDERS if providers is None else providers
    for name in chain():
        fn = providers.get(name)
        if fn is None:
            continue
        try:
            reply = fn(system, user, max_tokens)
        except Exception as exc:  # noqa: BLE001 - any provider failure → next one
            print(f"[watch] llm {name} failed: {type(exc).__name__}: {str(exc)[:200]}", file=sys.stderr)
            continue
        if reply is None or not reply.text:
            print(f"[watch] llm {name} returned nothing", file=sys.stderr)
            continue
        return reply
    return None
