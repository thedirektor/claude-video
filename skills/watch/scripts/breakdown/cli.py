"""``watch.py breakdown|motion|compare`` entry point.

``watch.py`` dispatches those three words here. Each command is a
handler in ``COMMANDS`` (``handler(argv) -> int``).
Exit codes: 0 ok, 2 usage/other error, 3 video not found, 4 ambiguous video.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cost
from config import memory_paths
from memory import index, retrieval

from . import assemble, compare, interchange, media, motion

EXIT_NOT_FOUND = 3
EXIT_AMBIGUOUS = 4


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


def _push(key: str) -> None:
    """Send the video's bundle (now carrying breakdown/ and motion-*/) to the central service, like a run does."""
    try:
        import push  # noqa: PLC0415

        if push.enabled():
            push.flush_outbox()
            r = push.push_video(key)
            print(f"[watch] central memory: {'pushed' if r.get('ok') else 'queued in outbox'}", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001 - a push never fails a measurement
        print(f"[watch] push skipped ({type(exc).__name__}: {exc})", file=sys.stderr)


def _cmd_breakdown(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="watch breakdown",
                                 description="Measure a stored video into breakdown.json / breakdown.md.")
    ap.add_argument("video", help="video_key, URL/path or title fragment of an already watched video")
    ap.add_argument("--interpret", action="store_true", help="add one LLM-written interpretation (paid)")
    ap.add_argument("--json", action="store_true", help="print breakdown.json instead of the brief")
    ap.add_argument("--formats", default="", help="timeline files to write: otio,fcpxml,edl")
    ap.add_argument("--allow-cost", action="store_true", help="ignore WATCH_COST_CEILING_USD for --interpret")
    ap.add_argument("--stored", action="store_true",
                    help="return the stored breakdown.json / .md as they are, without re-measuring")
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
        if a.stored:
            bd = assemble.read_stored(conn, key)
            out_dir = assemble.stored_path(conn, key, None).parent
            if bd is None:
                print("[watch] no stored breakdown for this video; run `breakdown` without --stored first",
                      file=sys.stderr)
                return 2
            if not a.json:
                md = out_dir / "breakdown.md"
                print(md.read_text(encoding="utf-8") if md.is_file() else assemble.brief_md(bd))
            else:
                print(json.dumps(bd, ensure_ascii=False, indent=1))
            return 0
        try:
            bd = assemble.build(conn, key, interpret=a.interpret, allow_cost=a.allow_cost)
        except cost.CostCeilingExceeded as exc:
            print(f"[watch] {exc}", file=sys.stderr)
            return 2
        out_dir = assemble.store(conn, key, bd)
        if formats:
            written = assemble.write_timelines(bd, out_dir, formats)
            if not written:
                print(f"[watch] no timeline files written: {interchange.skip_reason(bd)}", file=sys.stderr)
        _push(key)
    finally:
        conn.close()
    if a.json:
        print(json.dumps(bd, ensure_ascii=False, indent=1))
    else:
        print(assemble.brief_md(bd))
    print(f"[watch] breakdown stored in {out_dir}", file=sys.stderr)
    return 0


def _cmd_motion(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="watch motion",
                                 description="Track a region over a segment: phases, easing fit, engine snippets.")
    ap.add_argument("video", help="video_key, URL/path or title fragment of an already watched video")
    ap.add_argument("--start", type=float, required=True, help="segment start, seconds")
    ap.add_argument("--end", type=float, required=True, help="segment end, seconds")
    ap.add_argument("--region", default=None, help='"x,y,w,h" (normalised 0-1) or a short text resolved by OCR')
    ap.add_argument("--px-per-unit", type=float, default=None, help="pixels per game unit, to convert displacements")
    ap.add_argument("--json", action="store_true", help="print motion.json instead of the table")
    a = ap.parse_args(argv)
    if a.end <= a.start or a.start < 0:
        print("[watch] --end must be after --start (both >= 0)", file=sys.stderr)
        return 2
    if a.px_per_unit is not None and a.px_per_unit <= 0:
        print("[watch] --px-per-unit must be positive", file=sys.stderr)
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
        root = Path(memory_paths()["store_dir"])
        # a follow-up call on a stored video may refetch evicted media (same as breakdown)
        m = motion.analyze(conn, key, a.start, a.end, region=a.region, px_per_unit=a.px_per_unit,
                           store_dir=root, fetch=media.default_fetch)
        out_dir = assemble.breakdown_dir(conn, root, key) / f"motion-{a.start:g}-{a.end:g}"
        out_dir.mkdir(parents=True, exist_ok=True)
        sn = motion.snippets(m)
        (out_dir / "motion.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
        for name, text in sn.items():
            (out_dir / name).write_text(text, encoding="utf-8")
        media.pin(root, key)
        _push(key)
    finally:
        conn.close()
    print(json.dumps(m, ensure_ascii=False, indent=1) if a.json else sn["table.md"])
    print(f"[watch] motion stored in {out_dir}", file=sys.stderr)
    return 0


def _cmd_compare(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="watch compare",
                                 description="Reference vs your version: alignment, timing deltas, side-by-side video.")
    ap.add_argument("reference", help="video_key / title of a watched video, URL or local file")
    ap.add_argument("candidate", help="your version: local file or URL (analysed, never stored)")
    ap.add_argument("--offset", type=float, default=None,
                    help="candidate time = reference time + OFFSET seconds (skips automatic alignment)")
    ap.add_argument("--slow", type=int, default=4, help="slow-motion factor of compare.mp4 (default 4)")
    ap.add_argument("--out", default=None, help="output folder (default: next to the reference's breakdown)")
    ap.add_argument("--json", action="store_true", help="print compare.json instead of the summary")
    a = ap.parse_args(argv)
    if a.slow < 1:
        print("[watch] --slow must be >= 1", file=sys.stderr)
        return 2
    try:
        res = compare.run(a.reference, a.candidate, offset=a.offset, slow=a.slow,
                          out_dir=Path(a.out) if a.out else None)
    except compare.CompareError as exc:
        print(f"[watch] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(res, ensure_ascii=False, indent=1) if a.json else compare.summary_md(res))
    return 0


COMMANDS = {"breakdown": _cmd_breakdown, "motion": _cmd_motion, "compare": _cmd_compare}


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in ("breakdown", "motion", "compare"):
        print("usage: watch.py breakdown|motion|compare ...", file=sys.stderr)
        return 2
    handler = COMMANDS.get(argv[0])
    if handler is None:
        print(f"[watch] {argv[0]} is not available yet", file=sys.stderr)
        return 2
    return handler(argv[1:])
