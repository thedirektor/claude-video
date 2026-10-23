#!/usr/bin/env python3
"""Actual spend of one /watch run: a process-wide ledger written to work/spend.json.

Every paid call records what it used (tokens, audio seconds, prepaid credits);
dollars come from the same price table as the cost plan (``cost.prices``). Never
raises — accounting must not fail a run.
"""
from __future__ import annotations

import json
from pathlib import Path

_LEDGER: list[dict] = []


def reset() -> None:
    _LEDGER.clear()


def record(provider: str, item: str, usd: float | None = None, *, price_key: str | None = None,
           tokens_in: int = 0, tokens_out: int = 0, seconds: float = 0, credits: int = 0) -> None:
    try:
        if usd is None and price_key:
            import cost

            p = cost.prices().get(price_key) or cost.FALLBACK_TOKEN_PRICE
            usd = (tokens_in * p.get("in", 0) + tokens_out * p.get("out", 0)) / 1e6 \
                + p.get("per_hour", 0) * seconds / 3600
        _LEDGER.append({"provider": provider, "item": item, "usd": round(float(usd or 0), 6),
                        "price_key": price_key, "tokens_in": int(tokens_in), "tokens_out": int(tokens_out),
                        "seconds": round(float(seconds), 1), "credits": int(credits)})
    except Exception:  # noqa: BLE001
        pass


def record_gemini(response, model: str, item: str) -> None:
    usage = getattr(response, "usage_metadata", None)
    record("gemini", item, price_key=f"gemini:{model}",
           tokens_in=int(getattr(usage, "prompt_token_count", 0) or 0),
           tokens_out=int(getattr(usage, "candidates_token_count", 0) or 0)
           + int(getattr(usage, "thoughts_token_count", 0) or 0))


def items() -> list[dict]:
    return list(_LEDGER)


def total() -> float:
    return round(sum(i["usd"] for i in _LEDGER), 6)


def credits() -> dict[str, int]:
    out: dict[str, int] = {}
    for i in _LEDGER:
        if i["credits"]:
            out[i["provider"]] = out.get(i["provider"], 0) + i["credits"]
    return out


def flush(work: Path) -> Path:
    path = Path(work) / "spend.json"
    path.write_text(json.dumps({"total_usd": total(), "credits": credits(), "items": items()}, indent=2),
                    encoding="utf-8")
    return path
