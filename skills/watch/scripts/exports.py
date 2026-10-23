#!/usr/bin/env python3
"""Exports of one indexed video, built from the video memory (index + store).

Formats: ``docx`` / ``pdf`` (report, wordcraft; ``pdf`` also merges in the storyboard
deck via pdfcraft when available), ``pptx`` (storyboard deck, deckcraft), ``xlsx``
(timeline workbook, gridcraft), ``viewer`` (single-file offline HTML), ``json`` and
``contact`` (contact sheet PNG, ffmpeg). The craft CLIs are optional: a missing one
skips its format with a notice. Never raises — ``build`` returns a result per format.
"""
from __future__ import annotations

import base64
import glob
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

from config import memory_paths, setting
from memory import extract
from memory.retrieval import stamp

FORMATS = ("docx", "pdf", "pptx", "xlsx", "viewer", "json", "contact")
VIEWER_MAX_BYTES = 25 * 1024 * 1024
CLI_TIMEOUT_S = 300
NOTES_MAX = 1500
_FORMULA_START = ("=", "+", "-", "@")


# --------------------------------------------------------------------------- helpers

def find_cli(app: str) -> str | None:
    """`<app>-cli` from WATCH_CRAFT_BIN_DIR, PATH, or ~/.local/opt/artcraft/<app>-<ver>/bin."""
    names = [f"{app}-cli.exe", f"{app}-cli"] if os.name == "nt" else [f"{app}-cli"]
    bin_dir = setting("WATCH_CRAFT_BIN_DIR")
    if bin_dir:
        for n in names:
            p = Path(bin_dir).expanduser() / n
            if p.is_file():
                return str(p)
    for n in names:
        found = shutil.which(n)
        if found:
            return found
    pattern = str(Path.home() / ".local" / "opt" / "artcraft" / f"{app}-*" / "bin" / names[-1])
    hits = sorted(glob.glob(pattern))
    return hits[-1] if hits else None


def slug(title: str, limit: int = 60) -> str:
    ascii_ = unicodedata.normalize("NFKD", title or "").encode("ascii", "ignore").decode()
    s = re.sub(r"[^A-Za-z0-9]+", "-", ascii_).strip("-").lower()
    return (s[:limit].rstrip("-") or "video")


def _cell(v):
    if isinstance(v, str) and v.startswith(_FORMULA_START):
        return " " + v  # never let transcript text become a spreadsheet formula
    return v


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=CLI_TIMEOUT_S)
    if r.returncode != 0:
        raise RuntimeError(f"{Path(cmd[0]).name} exit {r.returncode}: {(r.stderr or r.stdout).strip()[:300]}")


def _font() -> str | None:
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/TTF/DejaVuSans.ttf",
              "/System/Library/Fonts/Supplemental/Arial.ttf", "C:/Windows/Fonts/arial.ttf"):
        if Path(p).is_file():
            return p
    return None


# --------------------------------------------------------------------------- data

def load(conn, key: str, store_dir=None) -> dict:
    """Everything an export needs, from the index and the store (no re-watch)."""
    row = conn.execute("SELECT title, url, source, duration, channel, upload_date, store_path, transcript_source "
                       "FROM videos WHERE video_key = ?", (key,)).fetchone()
    if row is None:
        raise LookupError(key)
    root = Path(store_dir or memory_paths()["store_dir"])
    segs = [{"start": r[0], "end": r[1], "speaker": r[2], "text": r[3] or ""} for r in conn.execute(
        'SELECT start, "end", speaker, text FROM segments WHERE video_key = ? ORDER BY start', (key,))]
    ocr = [{"t": r[0], "text": r[1] or ""} for r in conn.execute(
        "SELECT t, text FROM ocr_blocks WHERE video_key = ? ORDER BY t", (key,))]
    immich = (setting("IMMICH_BASE_URL") or "").rstrip("/")
    frames = []
    for t, kind, sp, asset in conn.execute(
            "SELECT t, kind, store_path, immich_asset_id FROM frames WHERE video_key = ? ORDER BY t", (key,)):
        path = root / sp if sp else None
        frames.append({"t": t, "kind": kind, "path": str(path) if path and path.is_file() else None,
                       "immich": f"{immich}/photos/{asset}" if asset and immich else None})
    chapters = extract.chapters(conn, key, llm_titles=False)
    stored = conn.execute("SELECT t_start, text FROM notes WHERE video_key = ? AND kind = 'chapter' ORDER BY t_start",
                          (key,)).fetchall()
    if len(stored) == len(chapters) and all(abs(s[0] - c["t_start"]) < 0.01 for s, c in zip(stored, chapters)):
        for s, c in zip(stored, chapters):
            c["title"] = s[1]
    for c in chapters:
        inside = [f for f in frames if f["path"] and c["t_start"] <= f["t"] < c["t_end"]]
        c["frame"] = inside[len(inside) // 2] if inside else None
    summary = conn.execute("SELECT text FROM notes WHERE video_key = ? AND kind = 'backend_answer' "
                           "ORDER BY id DESC LIMIT 1", (key,)).fetchone()
    answers = []
    for (res,) in conn.execute("SELECT result FROM answers WHERE video_key = ? ORDER BY id", (key,)):
        try:
            r = json.loads(res)
        except ValueError:
            continue
        if r.get("found"):
            answers.append({"question": r.get("question"), "answer": r.get("answer")})
    return {"video_key": key, "title": row[0] or key, "url": row[1] or row[2], "duration": row[3] or 0,
            "channel": row[4], "upload_date": row[5], "transcript_source": row[7],
            "store_dir": str(root / row[6]) if row[6] else None,
            "segments": segs, "ocr": ocr, "frames": frames, "chapters": chapters,
            "summary": summary[0] if summary else None, "answers": answers}


def _meta_line(d: dict) -> str:
    parts = [p for p in (d.get("channel"), stamp(d["duration"]) if d.get("duration") else None, d.get("url")) if p]
    return " · ".join(parts)


def _transcript_lines(d: dict, a: float = 0.0, b: float = float("inf")) -> list[str]:
    out = []
    for s in d["segments"]:
        if a <= s["start"] < b:
            who = f"{s['speaker']}: " if s.get("speaker") else ""
            out.append(f"[{stamp(s['start'])}] {who}{' '.join(s['text'].split())}")
    return out


# --------------------------------------------------------------------------- writers

def write_json(d: dict, out: Path) -> Path:
    out.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def write_report(d: dict, tmp: Path, want_docx: bool, want_pdf: bool) -> dict[str, Path]:
    cli = find_cli("wordcraft")
    if cli is None:
        raise FileNotFoundError("wordcraft-cli not found")
    cmds: list = []

    def para(text: str, style: str = "para.normal"):
        cmds.extend([{"command": style}, {"command": "text.insert", "params": {"text": text}},
                     {"command": "text.newParagraph"}])

    para(d["title"], "para.heading1")
    if _meta_line(d):
        para(_meta_line(d))
    if d.get("summary"):
        para("Summary", "para.heading2")
        for line in d["summary"].splitlines():
            if line.strip():
                para(line.strip())
    for i, c in enumerate(d["chapters"], 1):
        para(f"{i}. {c['title']} ({stamp(c['t_start'])}–{stamp(c['t_end'])})", "para.heading2")
        if c.get("frame"):
            pic = tmp / f"ch{i}{Path(c['frame']['path']).suffix}"
            shutil.copy2(c["frame"]["path"], pic)
            cmds.extend([{"command": "insert.picture", "params": {"path": pic.name, "width": 400,
                                                                   "alt": f"{stamp(c['frame']['t'])}"}},
                         {"command": "text.newParagraph"}])
        for b in c["bullets"]:
            para(f"• {b['text']}")
    lines = _transcript_lines(d)
    if lines:
        para("Transcript", "para.heading2")
        for line in lines:
            para(line)
    saves = (["report.docx"] if want_docx else []) + (["report.pdf"] if want_pdf else [])
    (tmp / "report.json").write_text(json.dumps({"commands": cmds, "save": saves}, ensure_ascii=False),
                                     encoding="utf-8")
    _run([cli, "run", "--script", "report.json"], cwd=tmp)
    return {name.split(".")[1]: tmp / name for name in saves if (tmp / name).is_file()}


def write_deck(d: dict, tmp: Path, want_pptx: bool, want_pdf: bool) -> dict[str, Path]:
    cli = find_cli("deckcraft")
    if cli is None:
        raise FileNotFoundError("deckcraft-cli not found")
    cmd = [cli, "run",
           "--cmd", "slide.new=" + json.dumps({"layout": "title", "title": d["title"], "body": _meta_line(d)},
                                              ensure_ascii=False),
           "--cmd", 'slide.delete={"index":0}']
    for i, c in enumerate(d["chapters"], 1):
        cmd += ["--cmd", "slide.new=" + json.dumps(
            {"layout": "titleOnly", "title": f"{i}. {c['title']} ({stamp(c['t_start'])})"}, ensure_ascii=False)]
        if c.get("frame"):
            cmd += ["--cmd", "insert.picture=" + json.dumps({"path": c["frame"]["path"], "rect": [120, 110, 720, 405]})]
        notes = "\n".join(_transcript_lines(d, c["t_start"], c["t_end"]))[:NOTES_MAX]
        if notes:
            cmd += ["--cmd", "slide.notes=" + json.dumps({"text": notes}, ensure_ascii=False)]
    out: dict[str, Path] = {}
    pptx, pdf = tmp / "storyboard.pptx", tmp / "storyboard.pdf"
    cmd += ["--save", str(pptx)]
    if want_pdf:  # rasterised slides: 96 dpi + text layer keeps it small and searchable
        cmd += ["--export", str(pdf), "--export-params", json.dumps({"dpi": 96, "textLayer": True})]
    _run(cmd, cwd=tmp)
    if want_pptx and pptx.is_file():
        out["pptx"] = pptx
    if want_pdf and pdf.is_file():
        out["pdf"] = pdf
    return out


def write_xlsx(d: dict, tmp: Path) -> Path:
    cli = find_cli("gridcraft")
    if cli is None:
        raise FileNotFoundError("gridcraft-cli not found")
    sheets = [
        ("Transcript", [["start", "end", "speaker", "text"]] +
         [[s["start"], s["end"], s.get("speaker") or "", _cell(s["text"])] for s in d["segments"]]),
        ("OCR", [["t", "text"]] + [[o["t"], _cell(o["text"])] for o in d["ocr"]]),
        ("Chapters", [["start", "end", "title", "notes"]] +
         [[c["t_start"], c["t_end"], _cell(c["title"]), _cell(" | ".join(b["text"] for b in c["bullets"]))]
          for c in d["chapters"]]),
        ("Frames", [["t", "kind", "immich"]] + [[f["t"], f["kind"], f.get("immich") or ""] for f in d["frames"]]),
    ]
    lines = []
    for i, (name, values) in enumerate(sheets):
        if i == 0:
            lines.append({"command": "sheet.rename", "params": {"name": name}})
        else:
            lines.append({"command": "home.insertSheet", "params": {"name": name, "before": i}})
        lines.append({"command": "range.setValues", "params": {"range": "A1", "values": values}})
    script = tmp / "timeline.jsonl"
    script.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in lines) + "\n", encoding="utf-8")
    out = tmp / "timeline.xlsx"
    _run([cli, "run", "--script", str(script), "--out", str(out), "--quiet"], cwd=tmp)
    return out


def merge_pdfs(parts: list[Path], out: Path) -> Path:
    cli = find_cli("pdfcraft")
    if cli is None:
        raise FileNotFoundError("pdfcraft-cli not found")
    _run([cli, "combine", *map(str, parts), "--out", str(out)])
    return out


def write_contact(d: dict, tmp: Path, out: Path, max_tiles: int = 24) -> Path:
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("ffmpeg not found")
    picks = [c["frame"] for c in d["chapters"] if c.get("frame")]
    rest = [f for f in d["frames"] if f["path"] and f not in picks]
    room = max(0, max_tiles - len(picks))
    if rest and room:
        step = max(1, len(rest) // room)
        picks += rest[::step][:room]
    picks = sorted(picks, key=lambda f: f["t"])[:max_tiles]
    if not picks:
        raise ValueError("no stored frames")
    font = _font()
    for i, f in enumerate(picks):
        vf = "scale=320:180:force_original_aspect_ratio=decrease,pad=320:180:(ow-iw)/2:(oh-ih)/2:black"
        if font:
            label = stamp(f["t"]).replace(":", r"\:")
            vf += (f",drawtext=fontfile='{font}':text='{label}':x=6:y=h-th-6:fontsize=18:fontcolor=white:"
                   "box=1:boxcolor=black@0.6:boxborderw=4")
        _run(["ffmpeg", "-v", "error", "-y", "-i", f["path"], "-vf", vf, "-frames:v", "1", str(tmp / f"c_{i:03d}.png")])
    cols = min(6, len(picks))
    rows = -(-len(picks) // cols)
    _run(["ffmpeg", "-v", "error", "-y", "-framerate", "1", "-start_number", "0", "-i", str(tmp / "c_%03d.png"),
          "-vf", f"tile={cols}x{rows}:padding=4:color=white", "-frames:v", "1", str(out)])
    return out


_VIEWER = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>__TITLE__</title>
<style>
:root{--bg:#fff;--fg:#1d1d1f;--muted:#6e6e73;--line:#e5e5ea;--accent:#0a64d6;--hi:#fff4c2}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--fg:#f2f2f7;--muted:#a1a1a6;--line:#2c2c2e;--accent:#4c9bff;--hi:#3a3420}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
header{padding:16px;border-bottom:1px solid var(--line)}h1{margin:0 0 4px;font-size:20px}.muted{color:var(--muted)}
main{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(0,1fr);gap:16px;padding:16px}
@media (max-width:800px){main{grid-template-columns:1fr}}
#bar{display:flex;height:28px;border-radius:6px;overflow:hidden;margin:12px 0}#bar div{border-right:2px solid var(--bg);
background:var(--accent);opacity:.55;cursor:pointer;font-size:11px;color:#fff;padding:5px;white-space:nowrap;overflow:hidden}
#bar div:hover,#bar div.on{opacity:1}#big{width:100%;border-radius:8px;background:#000;aspect-ratio:16/9;object-fit:contain}
#strip{display:flex;gap:6px;overflow-x:auto;padding:8px 0}#strip img{height:64px;border-radius:4px;cursor:pointer;opacity:.7}
#strip img.on{opacity:1;outline:2px solid var(--accent)}input{width:100%;padding:8px;border:1px solid var(--line);
border-radius:6px;background:var(--bg);color:var(--fg)}#rows{max-height:70vh;overflow:auto;margin-top:8px}
.r{padding:4px 6px;border-radius:4px;cursor:pointer}.r:hover{background:var(--line)}.r.on{background:var(--hi)}
.t{color:var(--accent);font-variant-numeric:tabular-nums;margin-right:6px}.ocr{font-style:italic;color:var(--muted)}
section{margin-top:16px}h2{font-size:16px;margin:16px 0 6px}li{margin:2px 0}
</style></head><body><header><h1 id="title"></h1><div class="muted" id="meta"></div></header>
<main><div><div id="bar"></div><img id="big" alt=""><div id="strip"></div><section id="chapters"></section>
<section id="answers"></section></div><div><input id="q" placeholder="Filter transcript and on-screen text…">
<div id="rows"></div></div></main>
<script type="application/json" id="data">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent);const $=s=>document.querySelector(s);
const mm=t=>{t=Math.floor(t);return String(Math.floor(t/60)).padStart(2,'0')+':'+String(t%60).padStart(2,'0')};
const el=(tag,cls,txt)=>{const e=document.createElement(tag);if(cls)e.className=cls;if(txt!=null)e.textContent=txt;return e};
$('#title').textContent=D.title;$('#meta').textContent=D.meta;document.title=D.title;
const frames=D.frames.filter(f=>f.src);const thumbs=[];
frames.forEach(f=>{const i=el('img');i.src=f.src;i.alt=mm(f.t);i.title=mm(f.t);i.onclick=()=>go(f.t);$('#strip').append(i);thumbs.push(i)});
const total=D.duration||1;D.chapters.forEach((c,k)=>{const b=el('div',null,(k+1)+'. '+c.title);
b.style.flex=Math.max(c.t_end-c.t_start,1)/total;b.title=mm(c.t_start)+' '+c.title;b.onclick=()=>go(c.t_start);$('#bar').append(b)});
const rows=[];D.segments.forEach(s=>rows.push({t:s.start,txt:(s.speaker?s.speaker+': ':'')+s.text,ocr:false}));
D.ocr.forEach(o=>rows.push({t:o.t,txt:o.text,ocr:true}));rows.sort((a,b)=>a.t-b.t);
rows.forEach(r=>{const d=el('div','r'+(r.ocr?' ocr':''));d.append(el('span','t',mm(r.t)),document.createTextNode(r.txt));
d.onclick=()=>go(r.t);r.el=d;$('#rows').append(d)});
$('#q').oninput=e=>{const q=e.target.value.toLowerCase();rows.forEach(r=>r.el.style.display=!q||r.txt.toLowerCase().includes(q)?'':'none')};
const ch=$('#chapters');if(D.chapters.length){ch.append(el('h2',null,'Chapters'));D.chapters.forEach((c,k)=>{
const h=el('div','r');h.append(el('span','t',mm(c.t_start)),document.createTextNode((k+1)+'. '+c.title));h.onclick=()=>go(c.t_start);
ch.append(h);const ul=el('ul');c.bullets.forEach(b=>{const li=el('li',null,b.text);li.style.cursor='pointer';li.onclick=()=>go(b.t);ul.append(li)});ch.append(ul)})}
const an=$('#answers');if(D.answers.length){an.append(el('h2',null,'Answers'));D.answers.forEach(a=>{an.append(el('div',null,'Q: '+a.question));
an.append(el('div','muted',a.answer))})}
function go(t){let best=-1,bd=1e18;frames.forEach((f,i)=>{const d=Math.abs(f.t-t);if(d<bd){bd=d;best=i}});
if(best>=0){$('#big').src=frames[best].src;$('#big').alt=mm(frames[best].t);thumbs.forEach((x,i)=>x.classList.toggle('on',i===best));
thumbs[best].scrollIntoView({block:'nearest',inline:'center'})}
let rb=null;rows.forEach(r=>{r.el.classList.remove('on');if(r.t<=t+0.5)rb=r});if(rb){rb.el.classList.add('on');rb.el.scrollIntoView({block:'nearest'})}
[...$('#bar').children].forEach((b,k)=>b.classList.toggle('on',t>=D.chapters[k].t_start&&t<D.chapters[k].t_end))}
go(0);
</script></body></html>
"""


def write_viewer(d: dict, out: Path, max_bytes: int = VIEWER_MAX_BYTES) -> Path:
    used = 0
    frames = []
    for f in d["frames"]:
        src = None
        if f["path"]:
            p = Path(f["path"])
            size = p.stat().st_size
            if used + size * 4 // 3 <= max_bytes:
                mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
                src = f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()
                used += len(src)
            else:
                src = os.path.relpath(p, out.parent).replace(os.sep, "/")
        frames.append({"t": f["t"], "src": src})
    payload = {"title": d["title"], "meta": _meta_line(d), "duration": d["duration"], "frames": frames,
               "chapters": [{k: c[k] for k in ("t_start", "t_end", "title", "bullets")} for c in d["chapters"]],
               "segments": [{"start": s["start"], "speaker": s.get("speaker"), "text": s["text"]} for s in d["segments"]],
               "ocr": d["ocr"], "answers": d["answers"]}
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    out.write_text(_VIEWER.replace("__TITLE__", html.escape(d["title"])).replace("__DATA__", data), encoding="utf-8")
    return out


# --------------------------------------------------------------------------- orchestration

def parse_formats(value: str | None) -> list[str]:
    """`"pdf, pptx"` → ["pdf", "pptx"]; unknown names are dropped with a notice."""
    out = []
    for f in (value or "").replace(";", ",").split(","):
        f = f.strip().lower()
        if not f or f in ("none", "0"):
            continue
        if f == "all":
            return list(FORMATS)
        if f in FORMATS and f not in out:
            out.append(f)
        elif f not in FORMATS:
            print(f"[watch] export: unknown format {f!r} ignored", file=sys.stderr)
    return out


def default_formats() -> list[str]:
    """WATCH_EXPORTS, else pdf+pptx when wordcraft and deckcraft are installed, else none."""
    env = setting("WATCH_EXPORTS")
    if env is not None:
        return parse_formats(env)
    return ["pdf", "pptx"] if find_cli("wordcraft") and find_cli("deckcraft") else []


def build(conn, key: str, formats: list[str], *, store_dir=None, out_dir: Path | None = None) -> dict:
    """Build the requested exports, plus notes.md always;
    {format: {"path": str} | {"skipped": why} | {"error": why}}."""
    results: dict[str, dict] = {}
    try:
        d = load(conn, key, store_dir)
    except Exception as exc:  # noqa: BLE001
        return {f: {"error": f"load failed: {exc}"} for f in formats}
    base = Path(out_dir) if out_dir else Path(d["store_dir"] or tempfile.mkdtemp(prefix="watch-export-")) / "exports"
    base.mkdir(parents=True, exist_ok=True)
    name = slug(d["title"])

    def put(fmt: str, src: Path, dest_name: str):
        dest = base / dest_name
        shutil.move(str(src), dest) if src.parent != base else None
        results[fmt] = {"path": str(dest)}

    def fail(fmt: str, exc: Exception):
        kind = "skipped" if isinstance(exc, FileNotFoundError) else "error"
        results[fmt] = {kind: str(exc)}
        print(f"[watch] export {fmt}: {kind} — {exc}", file=sys.stderr)

    with tempfile.TemporaryDirectory(prefix="watch-export-") as t:
        tmp = Path(t)
        if "json" in formats:
            try:
                put("json", write_json(d, tmp / "run.json"), "run.json")
            except Exception as exc:  # noqa: BLE001
                fail("json", exc)
        if "viewer" in formats:
            try:
                results["viewer"] = {"path": str(write_viewer(d, base / "viewer.html"))}
            except Exception as exc:  # noqa: BLE001
                fail("viewer", exc)
        report: dict[str, Path] = {}
        if "docx" in formats or "pdf" in formats:
            try:
                report = write_report(d, tmp, "docx" in formats, "pdf" in formats)
                if "docx" in report:
                    put("docx", report["docx"], "report.docx")
            except Exception as exc:  # noqa: BLE001
                for f in ("docx", "pdf"):
                    if f in formats:
                        fail(f, exc)
        deck: dict[str, Path] = {}
        if "pptx" in formats or ("pdf" in formats and report.get("pdf")):
            try:
                deck = write_deck(d, tmp, "pptx" in formats, "pdf" in formats)
                if "pptx" in deck:
                    put("pptx", deck["pptx"], "storyboard.pptx")
            except Exception as exc:  # noqa: BLE001
                if "pptx" in formats:
                    fail("pptx", exc)
        if "pdf" in formats and report.get("pdf"):
            merged = tmp / f"watch-{name}.pdf"
            try:
                if deck.get("pdf"):
                    merge_pdfs([report["pdf"], deck["pdf"]], merged)
                    put("pdf", merged, merged.name)
                else:
                    put("pdf", report["pdf"], f"watch-{name}.pdf")
            except Exception as exc:  # noqa: BLE001 - merge failed: keep the report alone
                print(f"[watch] export pdf: merge failed ({exc}); report only", file=sys.stderr)
                put("pdf", report["pdf"], f"watch-{name}.pdf")
        if "xlsx" in formats:
            try:
                put("xlsx", write_xlsx(d, tmp), "timeline.xlsx")
            except Exception as exc:  # noqa: BLE001
                fail("xlsx", exc)
        if "contact" in formats:
            try:
                results["contact"] = {"path": str(write_contact(d, tmp, base / "contact-sheet.png"))}
            except Exception as exc:  # noqa: BLE001
                fail("contact", exc)
    notes = base / "notes.md"
    try:
        notes.write_text(extract.notes_md(conn, key), encoding="utf-8")
        results["notes"] = {"path": str(notes)}
    except Exception as exc:  # noqa: BLE001
        fail("notes", exc)
    return results
