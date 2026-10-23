"""``watch.py breakdown|motion|compare`` entry point.

``watch.py`` dispatches those three words here. Only ``breakdown`` exists so
far; later tasks add ``motion`` and ``compare`` by registering a handler in
``COMMANDS`` (each is ``handler(argv) -> int``).
Exit codes: 0 ok, 2 usage/other error, 3 video not found, 4 ambiguous video.
"""
from __future__ import annotations

import argparse
import json
import sys

import cost
from config import memory_paths
from memory import index, retrieval

from . import assemble

EXIT_NOT_FOUND = 3
EXIT_AMBIGUOUS = 4
INTERPRET_EST_TOKENS = (3000, 400)


def _parse_formats(value: str | None) -> list[str]:
    out = []
    for f in (value or "").replace(";", ",").split(","):
        f = f.strip().lower()
        if not f:
            continue
        if f not in assemble.FORMATS:
            raise ValueError(f"unknown timeline format {f!r} (choose from {', '.join(assemble.FORMATS)})")
        if f not in out:
            out.append(f)
    return out


def _interpret_gate(allow: bool) -> None:
    """The one optional paid call is checked against the cost ceiling first."""
    ceiling = cost.ceiling()
    if allow or ceiling <= 0:
        return
    p = cost.FALLBACK_TOKEN_PRICE
    est = (INTERPRET_EST_TOKENS[0] * p["in"] + INTERPRET_EST_TOKENS[1] * p["out"]) / 1e6
    if est > ceiling:
        raise cost.CostCeilingExceeded(
            f"estimated interpretation cost ${est:.4f} exceeds WATCH_COST_CEILING_USD ${ceiling:.2f} - "
            "re-run with --allow-cost (or raise the ceiling).")


def _cmd_breakdown(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="watch breakdown",
                                 description="Measure a stored video into breakdown.json / breakdown.md.")
    ap.add_argument("video", help="video_key, URL/path or title fragment of an already watched video")
    ap.add_argument("--interpret", action="store_true", help="add one LLM-written interpretation (paid)")
    ap.add_argument("--json", action="store_true", help="print breakdown.json instead of the brief")
    ap.add_argument("--formats", default="", help="timeline files to write: otio,fcpxml,edl")
    ap.add_argument("--allow-cost", action="store_true", help="ignore WATCH_COST_CEILING_USD for --interpret")
    a = ap.parse_args(argv)
    try:
        formats = _parse_formats(a.formats)
    except ValueError as exc:
        print(f"[watch] {exc}", file=sys.stderr)
        return 2
    conn = index.connect(memory_paths()["index_db"])
    try:
        try:
            key = retrieval.resolve_video(conn, a.video)["video_key"]
        except retrieval.VideoNotFound as exc:
            print(f"[watch] {exc}. Run watch on it first.", file=sys.stderr)
            return EXIT_NOT_FOUND
        except retrieval.AmbiguousVideo as exc:
            print(f"[watch] {exc}", file=sys.stderr)
            return EXIT_AMBIGUOUS
        if a.interpret:
            try:
                _interpret_gate(a.allow_cost)
            except cost.CostCeilingExceeded as exc:
                print(f"[watch] {exc}", file=sys.stderr)
                return 2
        bd = assemble.build(conn, key, interpret=a.interpret)
        out_dir = assemble.store(conn, key, bd)
        if formats:
            written = assemble.write_timelines(bd, out_dir, formats)
            if not written:
                print("[watch] timeline writers are not available yet; no timeline files written",
                      file=sys.stderr)
    finally:
        conn.close()
    if a.json:
        print(json.dumps(bd, ensure_ascii=False, indent=1))
    else:
        print(assemble.brief_md(bd))
    print(f"[watch] breakdown stored in {out_dir}", file=sys.stderr)
    return 0


COMMANDS = {"breakdown": _cmd_breakdown}


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in ("breakdown", "motion", "compare"):
        print("usage: watch.py breakdown|motion|compare ...", file=sys.stderr)
        return 2
    handler = COMMANDS.get(argv[0])
    if handler is None:
        print(f"[watch] {argv[0]} is not available yet", file=sys.stderr)
        return 2
    return handler(argv[1:])
