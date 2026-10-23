"""Grounded-answer prompt and citation guard.

The model only ever sees numbered evidence (``[E3 12:31 transcript] …``) and
must cite ``[E#]``. ``apply`` keeps known ids (rendered with the evidence row's
own timestamp), drops unknown ones, and strips every other time-looking string,
so an answer's timestamps can only come from stored evidence rows.
"""
from __future__ import annotations

import re

NOT_FOUND = "NOT_FOUND"
CONTEXT_MAX = 600

SYSTEM = (
    "You answer questions about one video using ONLY the numbered evidence provided. "
    "Answer in the language of the question, concisely (at most a few sentences). "
    "After every claim put the id of the evidence that supports it, like [E2] or [E1, E3]. "
    "Never write times or timestamps yourself; the [E#] ids carry them. "
    f"If the evidence does not answer the question, reply exactly {NOT_FOUND}."
)

_GROUP = re.compile(r"(?:\[\s*E\d+(?:\s*[,;]\s*E\d+)*\s*\])+")
_ID = re.compile(r"E(\d+)")
_TIME = re.compile(r"\s*[\[(]?\b\d{1,2}:\d{2}(?::\d{2})?\b[\])]?")
_SLOT = re.compile("\x00(\\d+)\x00")


def build_prompt(question: str, evidence: list[dict]) -> tuple[str, str]:
    lines = ["Evidence:"]
    for e in evidence:
        lines.append(f"[{e['id']} {e.get('timestamp') or '--:--'} {e['kind']}] {e.get('text') or ''}")
        ctx = (e.get("context") or "").strip()
        if ctx:
            ctx = ctx[:CONTEXT_MAX]
            lines.extend("    " + ln for ln in ctx.splitlines())
    lines += ["", f"Question: {question}"]
    return SYSTEM, "\n".join(lines)


def _tidy(text: str) -> str:
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" +([.,;:!?])", r"\1", text)
    return text.strip()


def apply(text: str, evidence: list[dict]) -> dict:
    not_found = {"answer": "", "citations": [], "found": False}
    if not text or NOT_FOUND in text:
        return not_found
    by_id = {e["id"]: e for e in evidence}
    groups: list[list[str]] = []
    order: list[str] = []

    def keep(m: re.Match) -> str:
        ids = []
        for n in _ID.findall(m.group(0)):
            eid = f"E{n}"
            if eid in by_id and eid not in ids:
                ids.append(eid)
        if not ids:
            return ""
        order.extend(i for i in ids if i not in order)
        groups.append(ids)
        return f"\x00{len(groups) - 1}\x00"

    text = _GROUP.sub(keep, text)
    text = _TIME.sub("", text)
    text = _SLOT.sub(lambda m: "[" + "; ".join(
        f"{i} {by_id[i].get('timestamp') or '--:--'}" for i in groups[int(m.group(1))]) + "]", text)
    if not order:
        return not_found
    return {"answer": _tidy(text),
            "citations": [{"id": i, "t": by_id[i].get("t"), "timestamp": by_id[i].get("timestamp")} for i in order],
            "found": True}
