"""Grounded answers over one video's evidence.

cache → evidence pack → confidence → (low: model-free escalation, re-pack) →
(still low: refusal + nearest moments, no LLM) → LLM over numbered evidence →
citation guard → cache. Every timestamp in an answer is an evidence row's.
"""
from __future__ import annotations

from . import answers, confidence, embeddings, escalate, guard, llm, retrieval

REFUSAL = "No encontré esto en el video."
NEAREST = 3


def _evidence_version(conn, key: str) -> int:
    row = conn.execute("SELECT evidence_version FROM videos WHERE video_key = ?", (key,)).fetchone()
    return int(row[0]) if row else 0


def _nearest(evidence: list[dict]) -> list[dict]:
    return [{"id": e["id"], "timestamp": e["timestamp"], "kind": e["kind"], "snippet": retrieval._snippet(e["text"])}
            for e in evidence[:NEAREST]]


def _frame_times(conn, key: str, n: int) -> list[float]:
    """n stored frame times spread across the video (first, middle, last …)."""
    ts = [r[0] for r in conn.execute("SELECT DISTINCT t FROM frames WHERE video_key = ? ORDER BY t", (key,))]
    if len(ts) <= n:
        return ts
    return [ts[round(i * (len(ts) - 1) / (n - 1))] for i in range(n)]


def _refusal(base: dict) -> dict:
    return {**base, "answer": REFUSAL, "found": False, "citations": [], "nearest": _nearest(base["evidence"])}


def answer(conn, video_ref: str, question: str, k: int = 8, *, embedder=None, complete=None,
           escalate_fn=None, budget_s: float | None = None) -> dict:
    complete = complete or llm.complete
    escalate_fn = escalate_fn or escalate.escalate
    key = retrieval.resolve_video(conn, video_ref)["video_key"]
    cached = answers.lookup(conn, key, _evidence_version(conn, key), question, embedder=embedder)
    if cached is not None:
        return cached

    pack = retrieval.ask(conn, key, question, k=k, embedder=embedder)
    conf = confidence.assess(question, pack["evidence"])
    escalated = 0
    if conf["level"] == "low":
        times = [e["t"] for e in pack["evidence"] if e.get("t") is not None][:escalate.MAX_TIMES]
        if not pack["evidence"]:  # nothing matched at all: look around stored frames instead
            times = _frame_times(conn, key, escalate.MAX_TIMES)
        escalated = escalate_fn(conn, key, times, budget_s=budget_s) if times else 0
        if escalated:
            if embedder is not None:
                embeddings.embed_pending(conn, embedder)
            pack = retrieval.ask(conn, key, question, k=k, embedder=embedder)
            conf = confidence.assess(question, pack["evidence"])

    base = {"video": pack["video"], "question": question, "answer": None, "found": None, "citations": [],
            "confidence": conf, "provider": None, "model": None, "cached": False, "escalated": escalated,
            "evidence": pack["evidence"], "rule": pack["rule"], "mode": pack["mode"]}
    ev = _evidence_version(conn, key)

    if conf["level"] == "low":
        result = _refusal(base)
        answers.store(conn, key, ev, question, result, embedder=embedder)
        return result

    reply = complete(*guard.build_prompt(question, pack["evidence"]))
    if reply is None:
        return {**base, "note": "no LLM provider available — evidence only"}
    g = guard.apply(reply.text, pack["evidence"])
    base = {**base, "provider": reply.provider, "model": reply.model}
    result = {**base, **g} if g["found"] else _refusal(base)
    answers.store(conn, key, ev, question, result, provider=reply.provider,
                  tokens_in=reply.tokens_in, tokens_out=reply.tokens_out, embedder=embedder)
    return result
