#!/usr/bin/env python3
"""Per-stage resume state for /watch.

A run with a stable work dir (--out-dir) persists each expensive stage's result
as `stage_<name>.json` alongside a signature of the source + output-affecting
params. A later run with the same signature reloads finished stages and skips
them; a changed signature (or --fresh) ignores the cache. Late-stage crashes
therefore never destroy earlier network work. Source: danielfrey63 fork.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


def params_signature(source: str, params: dict) -> str:
    """Stable 16-hex signature over source + output-affecting params.

    Key order does not matter (sort_keys). Values are coerced to str so None,
    ints, and strings all serialize deterministically.
    """
    normalized = {k: ("" if v is None else str(v)) for k, v in params.items()}
    payload = json.dumps({"source": source, "params": normalized}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _stage_path(work: Path, name: str) -> Path:
    return work / f"stage_{name}.json"


def save_stage(work: Path, name: str, data, sig: str) -> None:
    """Persist a stage result. Best-effort — never raises on I/O failure."""
    try:
        work.mkdir(parents=True, exist_ok=True)
        _stage_path(work, name).write_text(
            json.dumps({"sig": sig, "data": data}), encoding="utf-8"
        )
    except (OSError, TypeError):
        pass  # unserializable / unwritable → just don't cache this stage


def load_stage(work: Path, name: str, sig: str):
    """Return the saved `data` iff the stage file exists and its sig matches."""
    path = _stage_path(work, name)
    if not path.exists():
        return None
    try:
        blob = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(blob, dict) or blob.get("sig") != sig:
        return None
    return blob.get("data")


def clear_stages(work: Path) -> None:
    """Delete every stage_*.json (used by --fresh). Leaves other files alone."""
    try:
        for f in work.glob("stage_*.json"):
            f.unlink()
    except OSError:
        pass
