"""Local stdio MCP server (`watch.py mcp`) for Claude Desktop (0.6.0 phase D)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import mcp_stdio  # noqa: E402
from memory.index import connect, record_run  # noqa: E402
from memory_fakes import KEY, fake_run, seg  # noqa: E402


def _rpc(proc, msg):
    proc.stdin.write(json.dumps(msg) + "\n")
    proc.stdin.flush()
    if "id" not in msg:
        return None
    return json.loads(proc.stdout.readline())


@pytest.fixture
def server(tmp_path):
    record_run(connect(), fake_run(title="La Canción", segments=[seg(12, 18, "esta canción habla del mar azul")]))
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    p = subprocess.Popen([sys.executable, str(SCRIPTS_DIR / "watch.py"), "mcp"], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env)
    yield p
    p.kill()


def test_initialize_list_and_call(server):
    init = _rpc(server, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                         "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t"}}})
    assert init["result"]["serverInfo"]["name"] == "watch" and init["result"]["capabilities"]["tools"] == {}
    _rpc(server, {"jsonrpc": "2.0", "method": "notifications/initialized"})
    tools = _rpc(server, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]
    names = {t["name"] for t in tools}
    assert {"watch", "search", "ask", "moment", "videos", "export", "plan", "doctor"} <= names
    assert all(t["inputSchema"]["type"] == "object" for t in tools)
    r = _rpc(server, {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                      "params": {"name": "search", "arguments": {"query": "mar azul"}}})
    body = json.loads(r["result"]["content"][0]["text"])
    assert body["hits"][0]["video_key"] == KEY and body["hits"][0]["timestamp"] == "00:12"
    r = _rpc(server, {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                      "params": {"name": "videos", "arguments": {}}})
    assert "La Canción" in r["result"]["content"][0]["text"]


def test_unknown_tool_and_method(server):
    r = _rpc(server, {"jsonrpc": "2.0", "id": 9, "method": "tools/call", "params": {"name": "nope", "arguments": {}}})
    assert r["result"]["isError"] is True
    r = _rpc(server, {"jsonrpc": "2.0", "id": 10, "method": "bogus/method"})
    assert r["error"]["code"] == -32601
    assert _rpc(server, {"jsonrpc": "2.0", "id": 11, "method": "ping"})["result"] == {}


def test_ask_not_found_is_tool_error():
    out = mcp_stdio.call_tool("ask", {"video": "yt:nope", "question": "x"})
    assert out["isError"] is True and "not found" in out["content"][0]["text"].lower()


def test_watch_tool_runs_watch_py(monkeypatch):
    seen = {}

    def fake_run(argv, **k):
        seen["argv"] = argv
        return subprocess.CompletedProcess(argv, 0, "# watch: video report\n- **Frames:** 0", "")
    monkeypatch.setattr(mcp_stdio.subprocess, "run", fake_run)
    out = mcp_stdio.call_tool("watch", {"source": "C:/Videos/clip.mp4", "question": "qué pasa", "detail": "efficient"})
    assert out["isError"] is False and out["content"][0]["text"].startswith("# watch: video report")
    assert seen["argv"][-2:] == ["C:/Videos/clip.mp4", "qué pasa"] and "--detail" in seen["argv"]


def test_breakdown_tool_listed_and_runs(monkeypatch):
    names = {t["name"] for t in mcp_stdio.TOOLS}
    assert {"breakdown", "motion", "compare"} <= names
    calls = []

    def fake(argv, timeout=3600):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout='{"ok": true}\n', stderr="")

    monkeypatch.setattr(mcp_stdio, "_run_watch", fake)
    r = mcp_stdio.call_tool("breakdown", {"video": "yt:abc"})
    assert r["isError"] is False and calls[-1] == ["breakdown", "yt:abc", "--json"]
    mcp_stdio.call_tool("breakdown", {"video": "yt:abc", "interpret": True, "formats": "otio,edl"})
    assert calls[-1] == ["breakdown", "yt:abc", "--json", "--interpret", "--formats", "otio,edl"]
    mcp_stdio.call_tool("motion", {"video": "yt:abc", "start": 1.0, "end": 2.5, "region": "0.1,0.1,0.2,0.2",
                                   "px_per_unit": 32})
    assert calls[-1] == ["motion", "yt:abc", "--start", "1.0", "--end", "2.5", "--json",
                         "--region", "0.1,0.1,0.2,0.2", "--px-per-unit", "32"]
    mcp_stdio.call_tool("compare", {"reference": "yt:abc", "candidate": "C:\\v\\mine.mp4", "offset": 0.5})
    assert calls[-1] == ["compare", "yt:abc", "C:\\v\\mine.mp4", "--json", "--offset", "0.5"]


def test_breakdown_tool_failure_is_error(monkeypatch):
    monkeypatch.setattr(mcp_stdio, "_run_watch", lambda argv, timeout=3600: subprocess.CompletedProcess(
        argv, 3, stdout="", stderr="[watch] not found"))
    r = mcp_stdio.call_tool("motion", {"video": "x", "start": 0, "end": 1})
    assert r["isError"] is True and "not found" in r["content"][0]["text"]
