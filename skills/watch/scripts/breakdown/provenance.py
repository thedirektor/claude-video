"""File-level provenance: camera tags, generator / editor signatures, AIGC labels, C2PA boxes.

Metadata is strong evidence when it is there (a phone make/model tag means camera footage
entered the pipeline; an AIGC label means a generator declared itself), but platforms and
re-encodes strip it, so its absence proves nothing: ``declared`` stays ``None`` then.
Only ffprobe tags and the first / last few MB of the file are read; nothing is guessed.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

SCAN_BYTES = 4 * 1024 * 1024  # MP4 metadata (moov / uuid / udta boxes) sits at one end of the file
CAMERA_KEYS = ("com.apple.quicktime.make", "com.apple.quicktime.model", "com.android.manufacturer",
               "com.android.model", "com.android.version", "make", "model",
               "com.apple.quicktime.location.iso6709", "location")
AIGC_KEYS = ("aigc", "ai_generated", "aigc_label", "contentproducer", "produceid")
# (pattern on tag values, kind, display name). Short tokens are word-bounded: "veo" also occurs by chance.
SIGNATURES = (
    (r"\bkling\b|kuaishou", "generator", "Kling"),
    (r"\bveo\b|deepmind|synthid", "generator", "Google Veo"),
    (r"\bsora\b", "generator", "OpenAI Sora"),
    (r"\brunway(ml)?\b", "generator", "Runway"),
    (r"\bpika\b", "generator", "Pika"),
    (r"\bluma\b|dream ?machine", "generator", "Luma"),
    (r"\bhailuo\b|minimax", "generator", "Hailuo / MiniMax"),
    (r"seedance|doubao|jimeng|即梦", "generator", "Seedance / Jimeng"),
    (r"\bhiggsfield\b", "generator", "Higgsfield"),
    (r"\bheygen\b|\bsynthesia\b|\bhedra\b", "generator", "Avatar generator"),
    (r"capcut|剪映|jianying", "editor", "CapCut"),
    (r"premiere|adobe", "editor", "Adobe"),
    (r"davinci|resolve", "editor", "DaVinci Resolve"),
    (r"final ?cut|\bfcp\b", "editor", "Final Cut"),
    (r"\bimovie\b", "editor", "iMovie"),
    (r"remotion", "renderer", "Remotion"),
    (r"blender", "renderer", "Blender"),
    (r"obs ?studio|\bobs\b", "screen", "OBS"),
)
AI_SOURCE_TYPES = (b"trainedAlgorithmicMedia", b"compositeWithTrainedAlgorithmicMedia")
AIGC_MEANING = {"1": "AI-generated", "2": "possibly AI-generated", "3": "suspected AI-generated"}


def _tags(path: str) -> dict[str, str]:
    try:
        res = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
                              str(path)], capture_output=True, text=True, timeout=60)
        data = json.loads(res.stdout or "{}")
    except (OSError, ValueError, subprocess.SubprocessError):
        return {}
    out: dict[str, str] = {}
    for k, v in ((data.get("format") or {}).get("tags") or {}).items():
        out[f"format.{k}"] = str(v)
    for i, s in enumerate(data.get("streams") or []):
        for k, v in (s.get("tags") or {}).items():
            out[f"stream{i}.{k}"] = str(v)
    return out


def _ends(path: str) -> bytes:
    try:
        p = Path(path)
        size = p.stat().st_size
        with p.open("rb") as f:
            head = f.read(SCAN_BYTES)
            if size > 2 * SCAN_BYTES:
                f.seek(size - SCAN_BYTES)
                return head + f.read(SCAN_BYTES)
            return head + f.read()
    except OSError:
        return b""


def parse_aigc(value: str) -> dict | None:
    """The Chinese implicit AIGC label (GB 45438): a JSON-ish value with ``Label`` and ``ContentProducer``."""
    m = re.search(r'"?label"?\s*[:=]\s*"?([0-9])', value, re.I)
    if not m:
        return None
    prod = re.search(r'"?contentproducer"?\s*[:=]\s*"?([^",}\s]+)', value, re.I)
    return {"label": m.group(1), "meaning": AIGC_MEANING.get(m.group(1), "unknown value"),
            "content_producer": prod.group(1) if prod else None}


def probe(path: str) -> dict:
    """``{declared, camera, signatures, aigc, c2pa, evidence[]}``.

    ``declared`` is ``"ai_generated"`` (an AIGC label 1 or a C2PA AI source type), ``"camera"`` (camera
    make / model tags) or ``None`` (nothing declared; the usual case after a platform re-encode)."""
    tags = _tags(path)
    raw = _ends(path)
    ev: list[dict] = []
    camera = {k: v for k, v in tags.items() if k.split(".", 1)[-1].lower() in CAMERA_KEYS}
    if camera:
        ev.append({"kind": "camera_tags", "detail": ", ".join(f"{k.split('.', 1)[-1]}={v}"
                                                              for k, v in list(camera.items())[:4])})
    aigc = None
    for k, v in tags.items():
        if any(a in k.split(".", 1)[-1].lower() for a in AIGC_KEYS) or re.search(r"\baigc\b", v, re.I):
            aigc = parse_aigc(v) or {"label": None, "meaning": "could not be parsed", "content_producer": None}
            ev.append({"kind": "aigc_label", "detail": f"AIGC label {aigc['label']} ({aigc['meaning']})"})
            break
    if aigc is None:
        m = re.search(rb'\{[^{}]{0,400}"?Label"?\s*[:=]\s*"?[0-9][^{}]{0,400}\}', raw)
        if m and b"aigc" in raw[max(0, m.start() - 200):m.end() + 200].lower():
            aigc = parse_aigc(m.group(0).decode("utf-8", "replace"))
            if aigc:
                ev.append({"kind": "aigc_label", "detail": f"AIGC label {aigc['label']} ({aigc['meaning']}) in file boxes"})
    blob = " ".join(f"{k}={v}" for k, v in tags.items()).lower()
    sigs = []
    for rx, kind, name in SIGNATURES:
        if re.search(rx, blob, re.I) and not any(s["name"] == name for s in sigs):
            sigs.append({"kind": kind, "name": name})
    if sigs:
        ev.append({"kind": "signatures", "detail": ", ".join(f"{s['kind']}: {s['name']}" for s in sigs)})
    c2pa = None
    if b"c2pa" in raw or b"jumb" in raw:
        ai = any(t in raw for t in AI_SOURCE_TYPES)
        c2pa = {"present": True, "ai_source_type": ai}
        ev.append({"kind": "c2pa", "detail": "C2PA content credentials found"
                                             + ("; digital source type: trained algorithmic media" if ai else "")})
    declared = None
    if (aigc and aigc.get("label") == "1") or (c2pa and c2pa["ai_source_type"]):
        declared = "ai_generated"
    elif camera and not any(s["kind"] == "generator" for s in sigs):
        declared = "camera"
    return {"declared": declared, "camera": camera or None, "signatures": sigs, "aigc": aigc, "c2pa": c2pa,
            "encoder": tags.get("format.encoder"), "evidence": ev}
