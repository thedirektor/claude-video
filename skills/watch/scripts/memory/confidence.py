"""How well an evidence pack answers a question (high / medium / low).

Signals, each 0..1:
- strength: the top item's raw cosine (when embeddings ran) or the share of
  the question's content words found in its text + context, whichever is higher;
- margin: how far the top item's (per-query normalised) score leads the runner-up;
- agreement: other modalities (transcript / OCR / note) and a frame near the top item.
"""
from __future__ import annotations

import re

from .textnorm import normalize_for_search

W_STRENGTH, W_MARGIN, W_AGREEMENT = 0.6, 0.15, 0.25
HIGH, MEDIUM = 0.6, 0.35
COS_FLOOR, COS_SPAN = 0.25, 0.45
NEAR_S = 5.0
_STOP = frozenset(
    "que qué cual cuál como cómo cuando donde quien los las del una uno unos unas por para con sin "
    "sobre este esta esto ese esa eso hay son está esta fue the and what which when where who how "
    "does did are was were this that with from for about video".split())


def _clip(x: float) -> float:
    return max(0.0, min(1.0, x))


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"\w+", normalize_for_search(text or "")) if len(w) >= 3 and w not in _STOP}


def _coverage(question: str, item: dict) -> float:
    q = _words(question)
    if not q:
        return 0.0
    have = _words(f"{item.get('text') or ''} {item.get('context') or ''}")
    return len(q & have) / len(q)


def assess(question: str, evidence: list[dict]) -> dict:
    if not evidence:
        return {"level": "low", "value": 0.0, "signals": {"strength": 0.0, "margin": 0.0, "agreement": 0.0}}
    top = evidence[0]
    cos = top.get("cos")
    strength = max(_clip((cos - COS_FLOOR) / COS_SPAN) if cos is not None else 0.0, _coverage(question, top))
    margin = 1.0 if len(evidence) == 1 else _clip(float(top.get("score") or 0) - float(evidence[1].get("score") or 0))
    agreement = 0.0
    if top.get("t") is not None:
        kinds = {e["kind"] for e in evidence if e.get("t") is not None and abs(e["t"] - top["t"]) <= NEAR_S}
        agreement = {1: 0.0, 2: 0.7}.get(len(kinds), 1.0)
    if top.get("frame"):
        agreement = min(1.0, agreement + 0.3)
    value = round(W_STRENGTH * strength + W_MARGIN * margin + W_AGREEMENT * agreement, 4)
    level = "high" if value >= HIGH else "medium" if value >= MEDIUM else "low"
    return {"level": level, "value": value,
            "signals": {"strength": round(strength, 4), "margin": round(margin, 4), "agreement": round(agreement, 4)}}
