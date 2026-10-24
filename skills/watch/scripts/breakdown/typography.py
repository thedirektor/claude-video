"""On-screen text tracks from OCR (pytesseract) sampled every ``step`` seconds.

Per-line detections are merged into tracks (same normalised text on adjacent samples). In/out times are refined by bisecting between the last
sample that saw the text and the next one that did not. Font family is always
``"not determined"``; size (``height_rel``), colour, position and timing are
measured. Entrance and exit come from pixels: every frame around the text's
appearance is diffed against a text-free background frame and the changed
region's box and contrast ramp are classified (cut / fade / slide_* / scale);
anything unclear is ``"not determined"``. ``slide_left`` means the text moves
leftwards. The per-sample OCR boxes are kept in ``evidence["box_track"]``.
"""
from __future__ import annotations

import math
import re

NOT_DETERMINED = "not determined"
MIN_CONF = 60
STEP_S = 0.5  # OCR sampling interval
IOU_MERGE = 0.6
REFINE_STEPS = 4
WORK_W = 320
MASK_T = 12  # grey-level change that counts as text pixels
LEAD_S = 1.0  # window before / after a track when looking for its animation
TAIL_S = 0.3  # extra look-ahead past the first sample without text (backgrounds often change soon after)
CUT_FRAMES = 2
POST_S = 0.3  # seconds of settled text that must look identical to the settled frame
MIN_POST = 2
POST_MAX = 0.002  # share of crop pixels allowed to change across those frames
TOP_STD_MAX = 35.0  # spread of the text-colour pixels (per channel) above which they are not one colour
RING_STD_MAX = 30.0  # busy surroundings: a sampled colour would be a guess
COLOR_CONTRAST_MIN = 90.0  # RGB distance between text and background below which the colour is unreliable
BG_MOVE_MAX = 0.02  # share of off-band crop pixels allowed to change before text is 'animating'


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
        ln = lines.setdefault(key, {"words": [], "confs": [], "x0": x, "y0": y, "x1": x + bw, "y1": y + bh})
        ln["words"].append(word)
        ln["confs"].append(conf)
        ln["x0"], ln["y0"] = min(ln["x0"], x), min(ln["y0"], y)
        ln["x1"], ln["y1"] = max(ln["x1"], x + bw), max(ln["y1"], y + bh)
    out = []
    for ln in lines.values():
        text = " ".join(ln["words"])
        norm = _norm(text)
        if len(norm) < 3:
            continue
        out.append({"text": text, "norm": norm, "conf": sum(ln["confs"]) / len(ln["confs"]) / 100.0,
                    "box": [ln["x0"] / w, ln["y0"] / h, (ln["x1"] - ln["x0"]) / w, (ln["y1"] - ln["y0"]) / h]})
    return out


def _present(ocr, cv2, cap, t: float, norm: str) -> bool:
    img = _read(cap, cv2, t)
    if img is None:
        return False
    return any(d["norm"] == norm for d in _detect(ocr, img, cv2))


def _bisect(ocr, cv2, cap, t_seen: float, t_unseen: float, norm) -> float:
    """Boundary between a time the text is seen and one it is not."""
    lo, hi = t_seen, t_unseen
    for _ in range(REFINE_STEPS):
        mid = (lo + hi) / 2.0
        if _present(ocr, cv2, cap, mid, norm):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _track_conf(samples: int, mean_ocr: float) -> float:
    """Track confidence: sample count evidence scaled by how well tesseract read the words (0-1)."""
    return min(0.95, 0.5 + 0.1 * samples) * max(0.0, min(1.0, mean_ocr))


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
    if float(ring.std(axis=0).mean()) > RING_STD_MAX:
        return NOT_DETERMINED  # busy background: the 'text' colour would be whatever pixel differs most
    box_px = reg[inner.reshape(-1)]
    dist = np.linalg.norm(box_px - bg, axis=1)
    n = max(1, int(len(dist) * 0.1))
    top = box_px[np.argsort(-dist)[:n]]
    if float(top.std(axis=0).mean()) > TOP_STD_MAX:
        return NOT_DETERMINED  # the strongest-contrast pixels disagree: they are not one flat text colour
    if float(np.linalg.norm(top.mean(axis=0) - bg)) < COLOR_CONTRAST_MIN:
        return NOT_DETERMINED  # text and background too close to tell apart
    return "#%02x%02x%02x" % tuple(int(round(v)) for v in top.mean(axis=0))


def _gray_frames(cv2, cap, fps: float, t0: float, t1: float, box):
    """[(t, small gray)] for every source frame in [t0, t1], cropped around ``box``.

    The crop (the settled box plus one box-width / three box-heights of margin) keeps
    other on-screen text and background changes elsewhere out of the diff.
    """
    i0, i1 = int(t0 * fps + 1e-6), int(math.ceil(t1 * fps - 1e-6)) + 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, i0)
    out = []
    for i in range(i0, i1 + 1):
        ok, img = cap.read()
        if not ok or img is None:
            break
        h, w = img.shape[:2]
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hk = max(2, int(round(h * WORK_W / w)))
        g = cv2.resize(g, (WORK_W, hk), interpolation=cv2.INTER_AREA)
        x0 = max(0, int((box[0] - box[2]) * WORK_W))
        x1 = min(WORK_W, int(math.ceil((box[0] + 2 * box[2]) * WORK_W)))
        y0 = max(0, int((box[1] - 3 * box[3]) * hk))
        y1 = min(hk, int(math.ceil((box[1] + 4 * box[3]) * hk)))
        out.append((i / fps, g[y0:y1, x0:x1]))
    return out


def _bbox(np, m):
    ys, xs = np.where(m)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _classify(np, grays, fi: int | None = None) -> tuple[str, int | None]:
    """Entrance class for a frame sequence [text-free background, ..., settled text, ...].

    ``grays[fi]`` is the settled frame (default: the last one); frames after it must be
    unchanged from it over the whole crop (moving video behind the text breaks that and
    the result is "not determined"). Returns (class, index of the first frame showing
    text) or ("not determined", None).
    """
    nd = (NOT_DETERMINED, None)
    if fi is None:
        fi = len(grays) - 1
    post = grays[fi + 1:]
    if post:
        if len(post) < MIN_POST or any(
                float((np.abs(g.astype(np.int16) - grays[fi].astype(np.int16)) > MASK_T).mean()) > POST_MAX
                for g in post):
            return nd
    else:
        return nd  # nothing after the text settled: cannot tell it from background motion
    grays = grays[:fi + 1]
    if len(grays) < 3:
        return nd
    bg = grays[0].astype(np.int16)
    diffs = [np.abs(g.astype(np.int16) - bg) for g in grays]
    masks = [d > MASK_T for d in diffs]
    fm = masks[-1]
    area_f = int(fm.sum())
    h, w = fm.shape
    if area_f < 40 or area_f > 0.5 * h * w:
        return nd
    # a text animation only moves along the settled box's row/column band; the four corners of
    # the crop must stay put, otherwise the "appearance" is background motion, not text.
    band = np.zeros((h, w), bool)
    bx0, by0, bx1, by1 = _bbox(np, fm)
    band[max(0, by0 - 2):by1 + 2, :] = True
    band[:, max(0, bx0 - 2):bx1 + 2] = True
    corners = ~band
    if corners.sum() >= 50 and max(float(m[corners].mean()) for m in masks) > BG_MOVE_MAX:
        return nd
    e = [float(d[fm].mean()) for d in diffs]
    boxes = [_bbox(np, m) for m in masks]
    bf = boxes[-1]
    area = [int(m.sum()) for m in masks]
    thr = max(8, 0.03 * area_f)
    start = next((i for i in range(1, len(grays)) if area[i] >= thr and (i + 1 >= len(grays) or area[i + 1] >= thr)), None)
    if start is None:
        return nd

    def same(b):
        return (b is not None and abs(b[0] - bf[0]) <= 1.5 and abs(b[1] - bf[1]) <= 1.5
                and abs(b[2] - bf[2]) <= 1.5 and abs(b[3] - bf[3]) <= 1.5)

    settle = len(grays) - 1
    while settle - 1 >= start and same(boxes[settle - 1]) and e[settle - 1] >= 0.9 * e[-1]:
        settle -= 1
    if settle - start <= CUT_FRAMES - 1:
        return "cut", start
    seq = list(range(start, settle + 1))
    live = [i for i in seq if boxes[i] is not None]
    if len(live) < 3:
        return nd
    fw, fh = bf[2] - bf[0], bf[3] - bf[1]
    free = [i for i in live if boxes[i][0] > 0 and boxes[i][1] > 0 and boxes[i][2] < w and boxes[i][3] < h]
    cx = [(boxes[i][0] + boxes[i][2]) / 2 for i in live]
    cy = [(boxes[i][1] + boxes[i][3]) / 2 for i in live]
    # the text's own box is only trustworthy once it is a clear majority of the final one
    if len(free) >= 2 and all(
            abs((boxes[i][2] - boxes[i][0]) / fw - 1) > 0.15 and abs((boxes[i][3] - boxes[i][1]) / fh - 1) > 0.15
            for i in free[:max(1, len(free) // 2)]):
        return "scale", start
    dx, dy = cx[-1] - cx[0], cy[-1] - cy[0]
    if max(abs(dx), abs(dy)) >= 8:
        axis, delta, series = (0, dx, cx) if abs(dx) >= abs(dy) else (1, dy, cy)
        other = abs(dy) if axis == 0 else abs(dx)
        steps = [b - a for a, b in zip(series, series[1:])]
        sgn = 1 if delta > 0 else -1
        if other * 3 <= abs(delta) and all(sgn * st >= -1.5 for st in steps):
            if axis == 0:
                return ("slide_right" if delta > 0 else "slide_left"), start
            return ("slide_down" if delta > 0 else "slide_up"), start
        return nd
    ramp = [e[i] / e[-1] for i in live]
    still = all(abs(c - cx[-1]) <= 2 and abs(d - cy[-1]) <= 2 for c, d in zip(cx, cy))
    if still and len(set(round(r, 1) for r in ramp)) >= 3 and all(b >= a - 0.08 for a, b in zip(ramp, ramp[1:])):
        return "fade", start
    return nd


_FLIP = {"slide_left": "slide_right", "slide_right": "slide_left", "slide_up": "slide_down", "slide_down": "slide_up"}


_SETTLE_AT = (0.5, 0.25, 0.75)  # where in the stable-box span to look for a settled frame


def _entrance(cv2, np, cap, fps, t_first, t_in_set, t_out_set, duration, box) -> tuple[str, float | None]:
    """(class, animation start time) from the frames before the text settles."""
    if not fps or fps <= 0 or t_first <= 0:
        return NOT_DETERMINED, None
    w0 = max(0.0, t_first - LEAD_S)
    for frac in _SETTLE_AT:
        t_f = t_in_set + frac * (t_out_set - t_in_set)
        frames = _gray_frames(cv2, cap, fps, w0, min(duration, t_out_set, t_f + POST_S), box)
        fi = next((i for i, (t, _) in enumerate(frames) if t >= t_f - 1e-6), None)
        if fi is None:
            continue
        cls, i = _classify(np, [g for _, g in frames], fi)
        if i is not None:
            return cls, frames[i][0]
    return NOT_DETERMINED, None


def _exit(cv2, np, cap, fps, t_in_set, t_out_set, t_end, duration, box) -> tuple[str, float | None]:
    """(class, time the text is gone) from the frames after the text starts leaving."""
    if not fps or fps <= 0 or t_end >= duration - 1.0 / fps:
        return NOT_DETERMINED, None
    for frac in _SETTLE_AT:
        t_f = t_in_set + frac * (t_out_set - t_in_set)
        frames = _gray_frames(cv2, cap, fps, max(t_in_set, t_f - POST_S), min(duration, t_end + TAIL_S), box)
        frames.reverse()
        fi = next((i for i, (t, _) in enumerate(frames) if t <= t_f + 1e-6), None)
        if fi is None:
            continue
        cls, i = _classify(np, [g for _, g in frames], fi)
        if i is not None:
            return _FLIP.get(cls, cls), frames[i][0] + 1.0 / fps
    return NOT_DETERMINED, None


def _drop_fragments(tracks: list[dict]) -> list[dict]:
    """Drop a track whose text is a strict part of a longer track that overlaps it in time and
    box (OCR sometimes reads one word of a line as its own, confidently read, line)."""
    def inside(a, b):
        iw = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
        ih = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
        return a[2] * a[3] > 0 and iw * ih / (a[2] * a[3]) >= 0.5

    keep = []
    for t in tracks:
        n = _norm(t["text"])
        frag = any(o is not t and len(_norm(o["text"])) > len(n) and n in _norm(o["text"])
                   and min(t["t_out"], o["t_out"]) > max(t["t_in"], o["t_in"]) and inside(t["box"], o["box"])
                   for o in tracks)
        if not frag:
            keep.append(t)
    return keep


def _modal(boxes):
    """Indices of the largest group of samples whose boxes agree (IoU >= 0.9), later wins ties."""
    best = []
    for i, (_, b) in enumerate(boxes):
        grp = [j for j, (_, c) in enumerate(boxes) if _iou(b, c) >= 0.9]
        if len(grp) >= len(best):
            best = grp
    return best


def text_tracks(path: str, duration: float, step: float = STEP_S) -> tuple[list[dict], list[str]]:
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
                cands = [d for d in dets if d["norm"] == tr["norm"]]
                m = max(cands, key=lambda d: _iou(d["box"], tr["boxes"][-1][1])) if cands else None
                if m:
                    dets.remove(m)
                    tr["last"] = t
                    tr["boxes"].append((t, m["box"]))
                    tr["confs"].append(m["conf"])
                    still.append(tr)
                elif t - tr["last"] < 2 * step - 1e-6:  # tolerate one missed OCR sample
                    still.append(tr)
                else:
                    done.append(tr)
            for d in dets:
                still.append({"text": d["text"], "norm": d["norm"], "first": t, "last": t, "boxes": [(t, d["box"])],
                              "confs": [d["conf"]]})
            open_tracks = still
        done.extend(open_tracks)
        out = []
        for tr in done:
            grp = _modal(tr["boxes"])
            t_set_in, box = tr["boxes"][grp[0]]
            t_set_out = tr["boxes"][grp[-1]][0]
            t_first, t_last = tr["first"], tr["last"]
            nxt = t_last + step
            if nxt < duration - 1e-9:
                t_out = _bisect(ocr, cv2, cap, t_last, nxt, tr["norm"])
            else:
                t_out = duration
            prv = t_first - step
            t_in = _bisect(ocr, cv2, cap, t_first, prv, tr["norm"]) if prv >= 0 else 0.0
            entrance, a_in = _entrance(cv2, np, cap, fps, t_first, t_set_in, t_set_out, duration, box)
            if a_in is not None:
                t_in = a_in
            ex, a_out = _exit(cv2, np, cap, fps, t_set_in, t_set_out, t_last + step, duration, box) if nxt < duration - 1e-9 \
                else (NOT_DETERMINED, None)
            if a_out is not None:
                t_out = a_out
            ref = _read(cap, cv2, (t_set_in + t_set_out) / 2.0)  # middle of the stable span: fully opaque
            conf = _track_conf(len(tr["boxes"]), sum(tr["confs"]) / len(tr["confs"]))
            out.append({
                "text": tr["text"],
                "t_in": round(t_in, 3), "t_out": round(t_out, 3),
                "box": [round(v, 4) for v in box],
                "height_rel": round(box[3], 4),
                "color_hex": _color_hex(cv2, np, ref, box) if ref is not None else NOT_DETERMINED,
                "entrance": entrance,
                "exit": ex,
                "font_family": NOT_DETERMINED,
                "evidence": {"t_start": round(t_in, 3), "t_end": round(t_out, 3), "method": "ocr",
                             "confidence": round(conf, 3), "samples": len(tr["boxes"]),
                             "ocr_confidence": round(sum(tr["confs"]) / len(tr["confs"]), 3),
                             "box_track": [[round(tt, 3)] + [round(v, 4) for v in b] for tt, b in tr["boxes"]]},
            })
        out = _drop_fragments(out)
        out.sort(key=lambda r: (r["t_in"], r["box"][1], r["box"][0]))
        return out
    finally:
        cap.release()
