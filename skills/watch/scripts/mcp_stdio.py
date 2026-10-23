#!/usr/bin/env python3
"""`watch.py mcp`: a local MCP server over stdio (no dependencies).

For Claude Desktop (or any MCP client) on the same machine — Windows, macOS or
Linux — so local files (``C:\\Videos\\clip.mp4``) can be watched and the local video
memory searched without the central service. Claude Desktop config::

    {"mcpServers": {"watch": {"command": "python",
                              "args": ["<path>/skills/watch/scripts/watch.py", "mcp"]}}}

JSON-RPC 2.0, one message per line (MCP stdio transport). Logs go to stderr only.
"""
from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path

from config import memory_paths

PROTOCOL_VERSION = "2025-06-18"
SCRIPT = Path(__file__).resolve().parent / "watch.py"
MAX_IMAGES = 12


def _schema(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required}


S, I, N, B = {"type": "string"}, {"type": "integer"}, {"type": "number"}, {"type": "boolean"}
TOOLS = [
    {"name": "watch", "description": "Watch a video (local file path or URL) on this machine: frames, transcript, "
     "OCR; returns the markdown report and indexes it into the local video memory (pushed to the central memory "
     "when WATCH_REMOTE_URL is set). detail: efficient | balanced | token-burner | transcript.",
     "inputSchema": _schema({"source": S, "question": S, "detail": S, "start": S, "end": S, "backend": S,
                             "max_frames": I, "allow_cost": B}, ["source"])},
    {"name": "search", "description": "Search everything watched on this machine (transcript, on-screen text, "
     "notes) with timestamps.", "inputSchema": _schema({"query": S, "video": S, "limit": I}, ["query"])},
    {"name": "ask", "description": "Answer a question about one watched video from its stored evidence, citing "
     "only stored timestamps (answer=false returns the evidence pack only).",
     "inputSchema": _schema({"video": S, "question": S, "k": I, "answer": B}, ["video", "question"])},
    {"name": "moment", "description": "What happens around time t (\"1:23\") in a watched video: transcript + OCR "
     "within ±window s, plus the stored frames as images.",
     "inputSchema": _schema({"video": S, "t": S, "window": N, "frames": I}, ["video", "t"])},
    {"name": "videos", "description": "List videos in the local memory (newest first), optionally filtered.",
     "inputSchema": _schema({"query": S, "limit": I}, [])},
    {"name": "export", "description": "Build chapters, notes.md and documents (pdf, docx, pptx, xlsx, viewer, "
     "json, contact) for an indexed video without re-watching; returns the file paths.",
     "inputSchema": _schema({"video": S, "formats": S}, ["video"])},
    {"name": "plan", "description": "Cost plan for watching a video (no side effects).",
     "inputSchema": _schema({"source": S, "detail": S, "backend": S, "start": S, "end": S}, ["source"])},
    {"name": "doctor", "description": "Check this machine's /watch dependencies (ffmpeg, yt-dlp, tesseract, "
     "model, keys present).", "inputSchema": _schema({"offline": B}, [])},
]


def _text(obj, is_error: bool = False) -> dict:
    txt = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, indent=1)
    return {"content": [{"type": "text", "text": txt}], "isError": is_error}


def _run_watch(argv: list[str], timeout: int = 3600) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *argv], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)


def _flags(args: dict, keys: dict[str, str]) -> list[str]:
    out: list[str] = []
    for k, flag in keys.items():
        v = args.get(k)
        if v is None or v is False:
            continue
        out += [flag] if v is True else [flag, str(v)]
    return out


def _conn():
    from memory import index

    return index.connect(memory_paths()["index_db"])


def _embedder():
    try:
        from memory import embeddings

        return embeddings.get_embedder(local_only=True)
    except Exception:  # noqa: BLE001
        return None


def call_tool(name: str, args: dict) -> dict:
    from memory import retrieval

    try:
        if name in ("watch", "plan"):
            flags = _flags(args, {"detail": "--detail", "start": "--start", "end": "--end", "backend": "--backend",
                                  "max_frames": "--max-frames", "allow_cost": "--allow-cost"})
            argv = [*flags, *(["--plan"] if name == "plan" else []), args["source"]]
            if name == "watch" and args.get("question"):
                argv.append(args["question"])
            r = _run_watch(argv)
            out = (r.stdout or "").strip()
            if r.returncode != 0:
                return _text(f"watch failed (exit {r.returncode}):\n{(r.stderr or '')[-1500:]}", True)
            return _text(out)
        if name == "doctor":
            r = _run_watch(["doctor", *(["--offline"] if args.get("offline") else [])], timeout=300)
            return _text(r.stdout or r.stderr, r.returncode != 0)
        conn = _conn()
        try:
            if name == "search":
                key = retrieval.resolve_video(conn, args["video"])["video_key"] if args.get("video") else None
                hits = retrieval.search(conn, args["query"], video_key=key, k=int(args.get("limit") or 8),
                                        embedder=_embedder())
                return _text({"query": args["query"], **retrieval.hit_dicts(conn, hits, hits.mode)})
            if name == "ask":
                if args.get("answer", True):
                    from memory import answer

                    return _text(answer.answer(conn, args["video"], args["question"], k=int(args.get("k") or 8),
                                               embedder=_embedder()))
                return _text(retrieval.ask(conn, args["video"], args["question"], k=int(args.get("k") or 8),
                                           embedder=_embedder()))
            if name == "videos":
                return _text({"videos": retrieval.list_videos(conn, args.get("query"), int(args.get("limit") or 20))})
            if name == "moment":
                data = retrieval.moment(conn, args["video"], args["t"], window=float(args.get("window") or 10),
                                        frames=min(int(args.get("frames") or 4), MAX_IMAGES))
                shots = data.pop("frames")
                content = [{"type": "text", "text": json.dumps(data, ensure_ascii=False)}]
                for f in shots:
                    content.append({"type": "text", "text": f"t={retrieval.stamp(f['t'])}"})
                    if f.get("path") and Path(f["path"]).is_file():
                        content.append({"type": "image", "mimeType": "image/jpeg",
                                        "data": base64.b64encode(Path(f["path"]).read_bytes()).decode()})
                return {"content": content, "isError": False}
            if name == "export":
                import exports
                from memory import extract

                key = retrieval.resolve_video(conn, args["video"])["video_key"]
                extract.store_chapters(conn, key)
                fmts = exports.parse_formats(args["formats"]) if args.get("formats") else exports.default_formats()
                return _text({"video_key": key, "exports": exports.build(conn, key, fmts)})
        finally:
            conn.close()
        return _text(f"unknown tool: {name}", True)
    except retrieval.VideoNotFound as exc:
        return _text(f"video not found: {exc}", True)
    except retrieval.AmbiguousVideo as exc:
        return _text({"error": "ambiguous", "candidates": exc.candidates}, True)
    except (KeyError, ValueError) as exc:
        return _text(f"invalid arguments: {exc}", True)
    except Exception as exc:  # noqa: BLE001 - a tool error never kills the server
        return _text(f"{type(exc).__name__}: {exc}", True)


def handle(msg: dict) -> dict | None:
    method, mid = msg.get("method"), msg.get("id")
    if mid is None:
        return None  # notification (initialized, cancelled, …)
    if method == "initialize":
        version = (msg.get("params") or {}).get("protocolVersion") or PROTOCOL_VERSION
        result = {"protocolVersion": version, "capabilities": {"tools": {}},
                  "serverInfo": {"name": "watch", "version": _version()}}
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        p = msg.get("params") or {}
        result = call_tool(p.get("name", ""), p.get("arguments") or {})
    else:
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}}
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def _version() -> str:
    try:
        root = Path(__file__).resolve().parents[3]
        return json.loads((root / ".claude-plugin" / "plugin.json").read_text())["version"]
    except Exception:  # noqa: BLE001
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdin, sys.stdout):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass
    print("[watch] MCP server on stdio", file=sys.stderr)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            reply = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
        else:
            reply = handle(msg)
        if reply is not None:
            sys.stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
            sys.stdout.flush()
    return 0
