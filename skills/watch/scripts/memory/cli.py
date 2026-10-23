"""Read-path CLI of the video memory: search / ask / moment / list / index.

``watch.py`` dispatches here when argv[1] is one of the subcommands (and is not
an existing path). Every subcommand takes ``--json``; human output is markdown.
Exit codes: 0 ok, 2 usage/other error, 3 video not found, 4 ambiguous video.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys

import identity
from config import memory_paths
from memory import answer as answer_mod
from memory import answers, embeddings, index, retrieval

COMMANDS = ("search", "ask", "moment", "list", "index")
EXIT_NOT_FOUND = 3
EXIT_AMBIGUOUS = 4


def _out(text: str = "") -> None:
    print(text)


def _json(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=1))


def _embedder():
    # Read path never downloads the model: a missing one degrades to text search.
    return embeddings.get_embedder(memory_paths().get("model_dir"), local_only=True)


def _fmt_t(seconds) -> str:
    return "" if seconds is None else retrieval.stamp(seconds)


def _cmd_search(conn, a) -> int:
    key = retrieval.resolve_video(conn, a.video)["video_key"] if a.video else None
    hits = retrieval.search(conn, a.query, video_key=key, k=a.limit, embedder=_embedder())
    shaped = retrieval.hit_dicts(conn, hits, hits.mode)
    if a.json:
        _json(shaped)
        return 0
    if not shaped["hits"]:
        _out(f"No matches for {a.query!r}.")
        return 0
    _out(f"# Search: {a.query}  ({shaped['mode']})")
    for h in shaped["hits"]:
        _out(f"- **{h['title'] or h['video_key']}** `{h['video_key']}` [{h['timestamp'] or '--:--'}] "
             f"{h['kind']}: {h['snippet']}")
    return 0


def _cmd_answer(conn, a) -> int:
    r = answer_mod.answer(conn, a.video, a.question, k=a.k, embedder=_embedder())
    if a.json:
        _json(r)
        return 0
    v = r["video"]
    _out(r["answer"] if r["answer"] is not None else f"({r.get('note', 'no answer')})")
    _out("")
    c = r["confidence"]
    _out(f"Confidence: {c['level']} ({c['value']})" + ("  [cached]" if r["cached"] else "")
         + (f"  [escalated +{r['escalated']}]" if r["escalated"] else ""))
    cited = {x["id"] for x in r["citations"]}
    for e in r["evidence"]:
        if e["id"] in cited:
            _out(f"[{e['id']} {e['timestamp'] or '--:--'} {e['kind']}] {e['text']}")
    for n in r.get("nearest") or []:
        _out(f"nearest: [{n['timestamp'] or '--:--'} {n['kind']}] {n['snippet']}")
    _out(f"Video: {v['title'] or v['video_key']} `{v['video_key']}`"
         + (f" — {r['provider']}/{r['model']}" if r["provider"] else ""))
    return 0


def _cmd_stats(conn, a) -> int:
    s = answers.stats(conn)
    if a.json:
        _json(s)
        return 0
    _out(f"answers cached: {s['answers']}  cache hits: {s['cache_hits']}  "
         f"tokens used: {s['tokens_used']}  tokens saved: {s['tokens_saved']}")
    for p, n in s["by_provider"].items():
        _out(f"- {p}: {n}")
    return 0


def _cmd_ask(conn, a) -> int:
    if a.answer:
        return _cmd_answer(conn, a)
    pack = retrieval.ask(conn, a.video, a.question, k=a.k, embedder=_embedder())
    if a.json:
        _json(pack)
        return 0
    _out(pack["rule"])
    for e in pack["evidence"]:
        _out(f"[{e['id']} {e['timestamp'] or '--:--'} {e['kind']}] {e['text']}")
        if e["context"]:
            for line in e["context"].splitlines():
                _out(f"    {line}")
        fr = e["frame"]
        if fr and (fr["path"] or fr["immich_url"]):
            _out(f"    frame: {fr['path'] or fr['immich_url']}")
    if not pack["evidence"]:
        _out("(no evidence found in this video)")
    v = pack["video"]
    _out("")
    _out(f"Video: {v['title'] or v['video_key']} `{v['video_key']}` ({pack['mode']})")
    return 0


def _cmd_moment(conn, a) -> int:
    m = retrieval.moment(conn, a.video, a.t, window=a.window, frames=a.frames)
    if a.json:
        _json(m)
        return 0
    v = m["video"]
    _out(f"# {v['title'] or v['video_key']} @ {_fmt_t(m['t'])} (+-{a.window:g}s)")
    for s in m["segments"]:
        who = f"{s['speaker']}: " if s["speaker"] else ""
        _out(f"[{s['timestamp']}] {who}{s['text']}")
    for o in m["ocr"]:
        _out(f"[{o['timestamp']} ocr] {o['text']}")
    for f in m["frames"]:
        where = f["path"] or f["immich_url"] or "(frame file evicted)"
        _out(f"frame t={_fmt_t(f['t'])}: {where}")
    if m["frames"] and any(f["path"] for f in m["frames"]):
        _out("Read each frame path above with the Read tool to see the image.")
    return 0


def _cmd_list(conn, a) -> int:
    vids = retrieval.list_videos(conn, a.query, limit=a.limit)
    if a.json:
        _json({"videos": vids})
        return 0
    if not vids:
        _out("No videos in the index.")
        return 0
    for v in vids:
        dur = f" {_fmt_t(v['duration'])}" if v["duration"] else ""
        _out(f"- **{v['title'] or '(untitled)'}** `{v['video_key']}`{dur} {v['url'] or ''}".rstrip())
    return 0


def _cmd_index(conn, a) -> int:
    try:
        return _index_actions(conn, a)
    except (retrieval.VideoNotFound, retrieval.AmbiguousVideo):
        raise
    except Exception as exc:  # noqa: BLE001 - HTTP/config errors: one line, no traceback
        print(f"watch: index failed ({type(exc).__name__}: {exc})", file=sys.stderr)
        return 2


def _index_actions(conn, a) -> int:
    paths = memory_paths()
    res: dict = {}
    if a.backfill_paperless:
        import library
        from memory import backfill

        res["backfill"] = backfill.backfill_paperless(conn, library.load_config())
    if a.rebuild:
        from memory import backfill

        r = backfill.rebuild(conn, paths["store_dir"], a.rebuild)
        res["rebuild"] = {"video_key": r.video_key, "evidence_version": r.evidence_version, "added": r.added}
    if a.forget:
        try:
            key = retrieval.resolve_video(conn, a.forget)["video_key"]
        except retrieval.VideoNotFound:
            # nothing indexed: still honour an exact key whose store dir is orphaned
            key = a.forget if (paths["store_dir"] / identity.store_dirname(a.forget)).is_dir() else None
            if key is None:
                raise
        existed = index.forget(conn, key)
        vdir = paths["store_dir"] / identity.store_dirname(key)
        had_dir = vdir.is_dir()
        if had_dir:
            shutil.rmtree(vdir, ignore_errors=True)
        res["forget"] = {"video_key": key, "rows": existed, "store_dir": had_dir}
    if a.embed_missing or a.backfill_paperless or a.rebuild:
        emb = embeddings.get_embedder(paths.get("model_dir"))
        n = 0
        while emb is not None:
            step = embeddings.embed_pending(conn, emb)
            n += step
            if not step:
                break
        res["embedded"] = n if emb is not None else "skipped (no embedder; text search only)"
    if a.json:
        _json(res)
    else:
        for k, v in res.items():
            _out(f"{k}: {json.dumps(v, ensure_ascii=False)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="watch", description="Query the /watch video memory.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, help_):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--json", action="store_true", help="machine-readable output")
        return sp

    s = add("search", "hybrid search across watched videos")
    s.add_argument("query")
    s.add_argument("--video", help="video_key, URL, or title fragment")
    s.add_argument("--limit", type=int, default=8)
    s = add("ask", "evidence pack for a question about one video")
    s.add_argument("video")
    s.add_argument("question")
    s.add_argument("-k", type=int, default=8)
    s.add_argument("--answer", action="store_true",
                   help="write a short grounded answer (LLM over the evidence; cites only evidence timestamps)")
    s = add("moment", "everything stored around a timestamp")
    s.add_argument("video")
    s.add_argument("t", help="seconds, MM:SS or H:MM:SS")
    s.add_argument("--window", type=float, default=10.0)
    s.add_argument("--frames", type=int, default=4)
    s = add("list", "list indexed videos")
    s.add_argument("query", nargs="?")
    s.add_argument("--limit", type=int, default=20)
    s = add("index", "maintain the index")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--backfill-paperless", action="store_true")
    g.add_argument("--rebuild", metavar="KEY")
    g.add_argument("--forget", metavar="KEY")
    g.add_argument("--embed-missing", action="store_true")
    add("stats", "answer-cache usage (answers, cache hits, tokens used / saved)")
    return p


_HANDLERS = {"search": _cmd_search, "ask": _cmd_ask, "moment": _cmd_moment, "list": _cmd_list, "index": _cmd_index,
             "stats": _cmd_stats}


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    conn = None
    try:
        conn = index.connect(memory_paths()["index_db"])
        return _HANDLERS[args.cmd](conn, args)
    except retrieval.VideoNotFound as exc:
        if getattr(args, "json", False):
            _json({"error": "video_not_found", "message": str(exc)})
        else:
            print(f"watch: {exc}", file=sys.stderr)
        return EXIT_NOT_FOUND
    except retrieval.AmbiguousVideo as exc:
        if getattr(args, "json", False):
            _json({"error": "ambiguous_video", "message": str(exc), "candidates": exc.candidates})
        else:
            print(f"watch: {exc}; pass one of these video_keys:", file=sys.stderr)
            for c in exc.candidates:
                _out(f"- `{c['video_key']}` {c['title'] or ''} {c['url'] or ''}".rstrip())
        return EXIT_AMBIGUOUS
    except (ValueError, FileNotFoundError) as exc:
        print(f"watch: {exc}", file=sys.stderr)
        return 2
    finally:
        if conn is not None:
            conn.close()
