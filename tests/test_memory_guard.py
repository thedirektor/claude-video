"""Grounded-answer prompt + citation guard (0.5.1 Task 3)."""
from __future__ import annotations

from memory.guard import NOT_FOUND, apply, build_prompt

EV = [
    {"id": "E1", "t": 12.0, "timestamp": "00:12", "kind": "transcript", "text": "habla del mar", "context": "[00:12] habla del mar"},
    {"id": "E2", "t": 100.0, "timestamp": "01:40", "kind": "ocr", "text": "MAR AZUL", "context": ""},
]


def test_apply_maps_ids_to_evidence_timestamps():
    r = apply("Habla del mar [E1]. Aparece escrito [E1, E2] y luego [E2][E1].", EV)
    assert r["found"] is True
    assert r["answer"] == "Habla del mar [E1 00:12]. Aparece escrito [E1 00:12; E2 01:40] y luego [E2 01:40; E1 00:12]."
    assert [c["id"] for c in r["citations"]] == ["E1", "E2"]
    assert r["citations"][0] == {"id": "E1", "t": 12.0, "timestamp": "00:12"}


def test_apply_strips_model_timestamps():
    r = apply("Pasa at 3:15 y también (1:02:03) o [03:15], según [E2].", EV)
    assert r["answer"] == "Pasa at y también o, según [E2 01:40]."
    for bad in ("3:15", "1:02:03", "03:15"):
        assert bad not in r["answer"]


def test_apply_unknown_ids_dropped_and_none_left_is_not_found():
    r = apply("Inventado [E99].", EV)
    assert r == {"answer": "", "citations": [], "found": False}
    r = apply("Mezcla [E99, E2].", EV)
    assert r["answer"] == "Mezcla [E2 01:40]." and r["found"]


def test_apply_not_found_literal():
    assert apply(f"{NOT_FOUND}", EV)["found"] is False
    assert apply(f"No sé [E1]. {NOT_FOUND}", EV)["found"] is False


def test_build_prompt_lists_items_with_ids():
    system, user = build_prompt("¿de qué habla?", EV)
    assert NOT_FOUND in system and "[E#]" in system
    assert "[E1 00:12 transcript] habla del mar" in user and "[E2 01:40 ocr] MAR AZUL" in user
    assert user.rstrip().endswith("¿de qué habla?")
    long = [{**EV[0], "context": "x" * 5000}]
    assert len(build_prompt("q", long)[1]) < 1200
