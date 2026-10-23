"""On-screen text tracks from OCR (pytesseract) sampled every ``step`` seconds.

Per-line detections are merged into tracks (same normalised text, box IoU >= 0.6
on consecutive samples). In/out times are refined by bisecting between the last
sample that saw the text and the next one that did not. Font family is always
``"not determined"``; size (``height_rel``), colour, position and timing are
measured. The entrance is ``"cut"`` only when the first frame carrying the text
already shows it complete; anything else is ``"not determined"``.
"""
from __future__ import annotations

import re

NOT_DETERMINED = "not determined"
MIN_CONF = 60
IOU_MERGE = 0.6
REFINE_STEPS = 4


def _import_cv():
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    return cv2, np


def _load_ocr():
    """(pytesseract, PIL.Image) or raise RuntimeError naming the missing piece."""
    try:
        import pytesseract  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("pytesseract is not installed") from exc
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("Pillow is not installed") from exc
    import shutil  # noqa: PLC0415
    if not shutil.which("tesseract"):
        raise RuntimeError("tesseract binary not found on PATH")
    return pytesseract, Image


def _norm(text: str) -> str:
    return re.sub(r"[^0-9a-z]", "", text.lower())


def _iou(a, b) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _read(cap, cv2, t: float):
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, img = cap.read()
    return img if ok and img is not None else None


def _detect(ocr, img, cv2) -> list[dict]:
    """Text lines in one BGR frame: [{"text","norm","box"(0-1)}]."""
    pt, Image = ocr
    h, w = img.shape[:2]
    # grayscale: tesseract misses white-on-red text in colour but reads it in luma
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    d = pt.image_to_data(pil, output_type=pt.Output.DICT, config="--psm 11")
    lines: dict = {}
    for i, raw in enumerate(d["text"]):
        word = (raw or "").strip()
        try:
            conf = float(d["conf"][i])
        except (TypeError, ValueError):
            continue
        if not word or conf < MIN_CONF:
            continue
        key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
        x, y, bw, bh = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
        ln = lines.setdefault(key, {"words": [], "x0": x, "y0": y, "x1": x + bw, "y1": y + bh})
        ln["words"].append(word)
        ln["x0"], ln["y0"] = min(ln["x0"], x), min(ln["y0"], y)
        ln["x1"], ln["y1"] = max(ln["x1"], x + bw), max(ln["y1"], y + bh)
    out = []
    for ln in lines.values():
        text = " ".join(ln["words"])
        norm = _norm(text)
        if len(norm) < 3:
            continue
        out.append({"text": text, "norm": norm,
                    "box": [ln["x0"] / w, ln["y0"] / h, (ln["x1"] - ln["x0"]) / w, (ln["y1"] - ln["y0"]) / h]})
    return out


def _present(ocr, cv2, cap, t: float, norm: str, box) -> bool:
    img = _read(cap, cv2, t)
    if img is None:
        return False
    return any(d["norm"] == norm and _iou(d["box"], box) >= IOU_MERGE for d in _detect(ocr, img, cv2))


def _bisect(ocr, cv2, cap, t_seen: float, t_unseen: float, norm, box) -> float:
    """Boundary between a time the text is seen and one it is not."""
    lo, hi = t_seen, t_unseen
    for _ in range(REFINE_STEPS):
        mid = (lo + hi) / 2.0
        if _present(ocr, cv2, cap, mid, norm, box):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _color_hex(cv2, np, img, box) -> str:
    h, w = img.shape[:2]
    x0, y0 = int(box[0] * w), int(box[1] * h)
    x1, y1 = int((box[0] + box[2]) * w), int((box[1] + box[3]) * h)
    pad = max(2, int(box[3] * h * 0.3))
    ex0, ey0, ex1, ey1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
    reg = cv2.cvtColor(img[ey0:ey1, ex0:ex1], cv2.COLOR_BGR2RGB).reshape(-1, 3).astype(np.float64)
    inner = np.zeros((ey1 - ey0, ex1 - ex0), bool)
    inner[y0 - ey0:y1 - ey0, x0 - ex0:x1 - ex0] = True
    ring = reg[~inner.reshape(-1)] if (~inner).any() else reg
    bg = np.median(ring, axis=0)
    box_px = reg[inner.reshape(-1)]
    dist = np.linalg.norm(box_px - bg, axis=1)
    n = max(1, int(len(dist) * 0.1))
    top = box_px[np.argsort(-dist)[:n]]
    return "#%02x%02x%02x" % tuple(int(round(v)) for v in top.mean(axis=0))


def _entrance(ocr, cv2, cap, t_in: float, fps: float, norm, box) -> str:
    """'cut' iff the text is absent one frame before t_in and complete one frame after."""
    if not fps or fps <= 0:
        return NOT_DETERMINED
    dt = 1.0 / fps
    if t_in - 2 * dt < 0:
        return NOT_DETERMINED
    before = _present(ocr, cv2, cap, t_in - 2 * dt, norm, box)
    after = _present(ocr, cv2, cap, t_in + 2 * dt, norm, box)
    return "cut" if (not before and after) else NOT_DETERMINED


def text_tracks(path: str, duration: float, step: float = 0.5) -> tuple[list[dict], list[str]]:
    try:
        cv2, np = _import_cv()
    except Exception:
        return [], ["opencv (cv2) / numpy not installed; typography not determined"]
    try:
        ocr = _load_ocr()
    except RuntimeError as exc:
        return [], [f"{exc}; typography not determined (tesseract/pytesseract required)"]
    try:
        return _run(cv2, np, ocr, str(path), float(duration), float(step)), []
    except Exception as exc:  # never raise out of a command
        return [], [f"typography failed ({type(exc).__name__}); not determined"]


def _run(cv2, np, ocr, path: str, duration: float, step: float) -> list[dict]:
    cap = cv2.VideoCapture(path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        times, t = [], 0.0
        while t < duration - 1e-9:
            times.append(round(t, 6))
            t += step
        open_tracks: list[dict] = []
        done: list[dict] = []
        for t in times:
            img = _read(cap, cv2, t)
            dets = _detect(ocr, img, cv2) if img is not None else []
            still = []
            for tr in open_tracks:
                m = next((d for d in dets if d["norm"] == tr["norm"] and _iou(d["box"], tr["box"]) >= IOU_MERGE), None)
                if m:
                    dets.remove(m)
                    tr["last"], tr["box"], tr["frames"] = t, m["box"], tr["frames"] + 1
                    tr["mid_img"] = img
                    still.append(tr)
                else:
                    done.append(tr)
            for d in dets:
                still.append({"text": d["text"], "norm": d["norm"], "box": d["box"], "first": t, "last": t,
                              "frames": 1, "mid_img": img})
            open_tracks = still
        done.extend(open_tracks)
        out = []
        for tr in done:
            t_first, t_last = tr["first"], tr["last"]
            nxt = t_last + step
            if nxt < duration - 1e-9:
                t_out = _bisect(ocr, cv2, cap, t_last, nxt, tr["norm"], tr["box"])
            else:
                t_out = duration
            prv = t_first - step
            t_in = _bisect(ocr, cv2, cap, t_first, prv, tr["norm"], tr["box"]) if prv >= 0 else 0.0
            entrance = _entrance(ocr, cv2, cap, t_in, fps, tr["norm"], tr["box"])
            conf = min(0.95, 0.5 + 0.1 * tr["frames"])
            out.append({
                "text": tr["text"],
                "t_in": round(t_in, 3), "t_out": round(t_out, 3),
                "box": [round(v, 4) for v in tr["box"]],
                "height_rel": round(tr["box"][3], 4),
                "color_hex": _color_hex(cv2, np, tr["mid_img"], tr["box"]),
                "entrance": entrance,
                "font_family": NOT_DETERMINED,
                "evidence": {"t_start": round(t_in, 3), "t_end": round(t_out, 3), "method": "ocr",
                             "confidence": round(conf, 3), "samples": tr["frames"]},
            })
        out.sort(key=lambda r: (r["t_in"], r["box"][1], r["box"][0]))
        return out
    finally:
        cap.release()
