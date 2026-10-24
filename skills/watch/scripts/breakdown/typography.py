"""On-screen text tracks from OCR (pytesseract) sampled every ``step`` seconds.

The video is decoded once, sequentially. Every ``step`` seconds the frame is read by tesseract
(downscaled to at most 720 px tall; a frame that did not change since the last OCR'd one reuses
that result). Between those samples a cheap text-likeness scan (every 0.1 s: edge blobs shaped
like words) triggers an extra OCR when a new word-shaped blob appears and persists for two scan
frames, so punch-word titles shorter than ``step`` are found; the extras are budgeted.

Per-line detections are merged into tracks (same normalised text on adjacent samples). In/out
times are first bisected between the last sample that saw the text and the next one that did not
(the bisected boundary also bounds the settle window of a one-sample track), then replaced by the
pixel-accurate animation time when the entrance / exit can be measured. Font family is always
``"not determined"``; size (``height_rel``), colour, position and timing are measured. Entrance
and exit come from pixels: every frame around the text's appearance is diffed against a text-free
background frame and the changed region's box and contrast ramp are classified (cut / fade /
slide_* / scale); anything unclear is ``"not determined"``. ``slide_left`` means the text moves
leftwards. The per-sample OCR boxes are kept in ``evidence["box_track"]``.
"""
from __future__ import annotations

import math
import re
from bisect import bisect_right

NOT_DETERMINED = "not determined"
MIN_CONF = 60
STEP_S = 0.5  # OCR sampling interval
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
OCR_MAX_H = 720  # frames taller than this are downscaled before OCR
SCAN_S = 0.1  # text-likeness scan cadence between OCR samples
SCAN_W = 320  # scan / change-detection work width
UNCHANGED_MAX = 0.0002  # share of pixels changed (> MASK_T) below which a frame counts as unchanged
EXTRA_SHARE = 0.6  # extra OCR runs allowed per base sample (plus EXTRA_MIN) ...
EXTRA_MIN = 4
EXTRA_PER_INTERVAL = 2  # ... and at most this many inside one base interval
EXTRA_GAP_S = 0.15  # no extra OCR this soon after another OCR run
NEW_BLOB_MIN = 0.0015  # share of the frame covered by new, persistent word-shaped blobs to trigger an extra OCR
GRAD_T = 48  # morphological gradient above which a pixel is an edge
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
    # grayscale: tesseract misses white-on-red text in colour but reads it in luma
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray.shape[0] > OCR_MAX_H:  # OCR cost is quadratic in the frame size; boxes stay normalised
        gray = cv2.resize(gray, (int(round(gray.shape[1] * OCR_MAX_H / gray.shape[0])), OCR_MAX_H),
                          interpolation=cv2.INTER_AREA)
    h, w = gray.shape[:2]
    pil = Image.fromarray(gray)
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


PROBE_MARGIN = 0.6  # crop margin (as a share of the box size) around the text for bisect probes


def _crop(img, box, tight: bool = False):
    h, w = img.shape[:2]
    mx, my = (0.005, 0.005) if tight else (max(0.04, box[2] * PROBE_MARGIN), max(0.04, box[3] * PROBE_MARGIN * 2))
    x0, y0 = max(0, int((box[0] - mx) * w)), max(0, int((box[1] - my) * h))
    x1, y1 = min(w, int(math.ceil((box[0] + box[2] + mx) * w))), min(h, int(math.ceil((box[1] + box[3] + my) * h)))
    return img[y0:y1, x0:x1]


def _present(ocr, cv2, cap, t: float, norm: str, box=None) -> bool:
    """Is the line ``norm`` on screen at ``t``? With ``box`` only a crop around it is read: a
    bisect probe asks about one known line, not about the whole frame."""
    img = _read(cap, cv2, t)
    if img is None:
        return False
    if box is not None:
        crop = _crop(img, box)
        if crop.size:
            img = crop
    return any(d["norm"] == norm for d in _detect(ocr, img, cv2))


def _bisect(ocr, cv2, cap, t_seen: float, t_unseen: float, norm, box=None) -> float:
    """Boundary between a time the text is seen and one it is not."""
    lo, hi = t_seen, t_unseen
    for _ in range(REFINE_STEPS):
        mid = (lo + hi) / 2.0
        if _present(ocr, cv2, cap, mid, norm, box):
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


_BISECT_MARGIN = 0.04  # a bisected boundary is only known to about this much: stay inside the text


def _entrance(cv2, np, cap, fps, t_first, t_in_set, t_out_set, duration, box,
              t_out_cap: float | None = None) -> tuple[str, float | None]:
    """(class, animation start time) from the frames before the text settles.

    The settle window runs to ``t_out_cap`` (the bisected end of the text) when that is later than
    the last box sample, so a track seen by a single OCR sample still has frames after its text
    settled to prove it did not move."""
    if not fps or fps <= 0 or t_first <= 0:
        return NOT_DETERMINED, None
    w0 = max(0.0, t_first - LEAD_S)
    end = max(t_out_set, t_out_cap - 2.0 / fps) if t_out_cap is not None else t_out_set  # _gray_frames reads up to 2 frames past its end
    for frac in _SETTLE_AT:
        t_f = t_in_set + frac * (t_out_set - t_in_set)
        frames = _gray_frames(cv2, cap, fps, w0, min(duration, end, t_f + POST_S), box)
        fi = next((i for i, (t, _) in enumerate(frames) if t >= t_f - 1e-6), None)
        if fi is None:
            continue
        cls, i = _classify(np, [g for _, g in frames], fi)
        if i is not None:
            return cls, frames[i][0]
    return NOT_DETERMINED, None


def _exit(cv2, np, cap, fps, t_in_set, t_out_set, t_end, duration, box,
          t_in_cap: float | None = None) -> tuple[str, float | None]:
    """(class, time the text is gone) from the frames after the text starts leaving.
    ``t_in_cap`` (the bisected start of the text) bounds the settle window from below."""
    if not fps or fps <= 0 or t_end >= duration - 1.0 / fps:
        return NOT_DETERMINED, None
    start = min(t_in_set, t_in_cap + 1.0 / fps) if t_in_cap is not None else t_in_set  # ...and one frame before its start
    for frac in _SETTLE_AT:
        t_f = t_in_set + frac * (t_out_set - t_in_set)
        frames = _gray_frames(cv2, cap, fps, max(start, t_f - POST_S), min(duration, t_end + TAIL_S), box)
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
    box (OCR sometimes reads one word of a line as its own, confidently read, line), or that
    lasted only one or two samples entirely inside the longer track's lifetime."""
    def inside(a, b):
        iw = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
        ih = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
        return a[2] * a[3] > 0 and iw * ih / (a[2] * a[3]) >= 0.5

    keep = []
    for t in tracks:
        n = _norm(t["text"])
        # a short-lived piece of a longer line read while that line animates (a word sliding in
        # sits elsewhere than where the line settles), so the time span alone makes it a fragment
        brief = t.get("evidence", {}).get("samples", 99) <= 2

        def inside_in_time(o):
            return o["t_in"] - 0.05 <= t["t_in"] and t["t_out"] <= o["t_out"] + 0.05

        frag = any(o is not t and len(_norm(o["text"])) > len(n) and n in _norm(o["text"])
                   and min(t["t_out"], o["t_out"]) > max(t["t_in"], o["t_in"])
                   and (inside(t["box"], o["box"]) or (brief and inside_in_time(o)))
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


def _small_gray(cv2, img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = g.shape[:2]
    return cv2.resize(g, (SCAN_W, max(2, int(round(h * SCAN_W / w)))), interpolation=cv2.INTER_AREA)


def _textlike(cv2, np, g):
    """Binary mask of word-shaped edge blobs (wide, short, reasonably filled) in a small gray frame."""
    h = g.shape[0]
    grad = cv2.morphologyEx(g, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    edges = (grad >= GRAD_T).astype(np.uint8)
    joined = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    if n <= 1:
        return np.zeros(g.shape, bool)
    bw, bh, area = st[1:, cv2.CC_STAT_WIDTH], st[1:, cv2.CC_STAT_HEIGHT], st[1:, cv2.CC_STAT_AREA]
    ok = (bh >= 3) & (bh <= 0.2 * h) & (bw >= 1.8 * bh) & (bw <= 0.9 * g.shape[1]) & (area >= 0.25 * bw * bh)
    ids = np.nonzero(ok)[0] + 1
    return np.isin(lab, ids) if len(ids) else np.zeros(g.shape, bool)


def _new_blob(cv2, np, m0, m1, m2) -> bool:
    """A word-shaped blob present in the last two scan frames (``m1``, ``m2``) but not before (``m0``)."""
    if m1 is None or m2 is None:
        return False
    d = lambda m, k: cv2.dilate(m.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)  # noqa: E731
    new = m2 & d(m1, 5) & ~(d(m0, 7) if m0 is not None else np.zeros_like(m2))
    return float(new.mean()) >= NEW_BLOB_MIN


def _stamps_ok(cv2, path: str, n: int = 40) -> bool:
    """Do the first frames carry strictly increasing presentation times? Sequential decoding then
    stamps every frame with its own time (constant or variable frame rate alike)."""
    cap = cv2.VideoCapture(str(path))
    try:
        last, got = -1.0, 0
        for _ in range(n):
            if not cap.grab():
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if t <= last:
                return False
            last, got = t, got + 1
        return got >= 2
    finally:
        cap.release()


def scan_available(path: str, duration: float) -> bool:
    """Whether :func:`text_tracks` will run the short-title scan on this file."""
    try:
        cv2, _ = _import_cv()
    except Exception:
        return False
    return _stamps_ok(cv2, path)


ISO_DIFF = 40  # grey-level change that marks a pixel as newly appeared
ISO_MIN_PX = 30


def _iso_mask(cv2, np, g, gp):
    """Small boolean mask of word-shaped blobs among the pixels that changed against ``gp`` (a
    couple of scan steps earlier) and are bright or dark: the trigger signal for a short title
    over a moving picture, where whole-frame edge blobs are drowned in clutter."""
    changed = np.abs(g.astype(np.int16) - gp.astype(np.int16)) > ISO_DIFF
    m = (changed & (g > 200) & (g > gp)).astype(np.uint8)  # bright and newly brighter: an appearance, never a removal
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    h, w = m.shape
    small = cv2.resize(m * 255, (SCAN_W, max(2, int(round(h * SCAN_W / w)))), interpolation=cv2.INTER_AREA) > 90
    joined = cv2.morphologyEx(small.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    if n <= 1:
        return np.zeros(small.shape, bool)
    bw, bh, area = st[1:, cv2.CC_STAT_WIDTH], st[1:, cv2.CC_STAT_HEIGHT], st[1:, cv2.CC_STAT_AREA]
    ok = (bh >= 3) & (bh <= 0.2 * small.shape[0]) & (bw >= 2.5 * bh) & (bw <= 0.9 * small.shape[1]) & (area >= 0.3 * bw * bh)
    ids = np.nonzero(ok)[0] + 1
    return np.isin(lab, ids) if len(ids) else np.zeros(small.shape, bool)


def _isolated(cv2, np, g, gp) -> list:
    """Text-isolated copies of the gray frame ``g`` for OCR: only pixels that changed against ``gp``
    (a couple of scan steps earlier) and are newly bright survive (drawn white-on-black), the rest is flattened. A short title over a moving picture is
    unreadable in the whole frame but reads cleanly once the moving background is removed."""
    changed = np.abs(g.astype(np.int16) - gp.astype(np.int16)) > ISO_DIFF
    out = []
    for sel, ink, paper in (((g > 200) & (g > gp), 255, 0),):  # bright text only: a dark selector reads the ghost of a removed title
        m = (changed & sel).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        if int(m.sum()) >= ISO_MIN_PX:
            out.append(cv2.cvtColor(np.where(m > 0, ink, paper).astype(np.uint8), cv2.COLOR_GRAY2BGR))
    return out


def _merge_lines(*groups) -> list:
    seen, out = set(), []
    for grp in groups:
        for d in grp:
            if d["norm"] not in seen:
                seen.add(d["norm"])
                out.append(d)
    return out


def _sample(cv2, np, ocr, path: str, duration: float, step: float, fps: float):
    """``(samples, mode, frame_times)``. ``samples`` is ``[(t, detections, is_extra)]`` in time order: one entry
    per base sample (every ``step``), plus the extra OCR runs the text-likeness scan asked for.
    ``mode`` is ``"cfr"`` (frame times follow index / fps), ``"vfr"`` (they drift from it: a
    variable frame rate, so seeking cannot be trusted) or ``"seek"`` (the decoder gives no usable
    frame times: sampled by seeking, no extras).

    One sequential decode, each frame stamped with its own presentation time. A base sample whose
    frame did not change since the last OCR run reuses that run's detections."""
    nbase = int(math.ceil(duration / step - 1e-9)) if duration > 0 else 0
    if not _stamps_ok(cv2, path):
        cap = cv2.VideoCapture(path)
        try:
            out = []
            for k in range(nbase):
                t = round(k * step, 6)
                img = _read(cap, cv2, t)
                out.append((t, _detect(ocr, img, cv2) if img is not None else [], False))
            return out, "seek", []
        finally:
            cap.release()
    cap = cv2.VideoCapture(path)
    try:
        sub = max(1, int(math.ceil(step / SCAN_S - 1e-9)))
        scan = step / sub
        tol = min(0.5 / fps, 0.5 * scan) if fps and fps > 0 else 0.5 * scan
        out: list = []
        stamps_all: list = []
        idx, k = 0, 0  # k: next scan slot (slot k is due at k * scan)
        drift_lo, drift_hi = 1e9, -1e9
        last_g = None
        last_dets: list = []
        last_ocr_t = -1e9
        masks: list = [None, None]
        frames: list = [None, None]  # gray scan frames k-2, k-1
        iso_prev = None
        pending = False  # trigger fired: also read the next scan frame (the mid-span of a short title)
        extras_total, extras_here = 0, 0
        budget = int(EXTRA_SHARE * nbase) + EXTRA_MIN
        while True:
            if not cap.grab():
                break
            t_frame = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if fps and fps > 0:
                drift_lo, drift_hi = min(drift_lo, t_frame - idx / fps), max(drift_hi, t_frame - idx / fps)
            idx += 1
            if t_frame >= duration - 1e-9:
                break
            stamps_all.append(t_frame)
            if t_frame < k * scan - tol:
                continue
            ok, img = cap.retrieve()
            kk = k
            k = max(k + 1, int(math.floor(t_frame / scan + 0.5)) + 1)  # a slow source skips slots
            if not ok or img is None:
                continue
            ts = round(t_frame, 6)
            g = _small_gray(cv2, img)
            m = _textlike(cv2, np, g)
            gfull = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            iso_m = _iso_mask(cv2, np, gfull, frames[0]) if frames[0] is not None else None
            iso_hit = (iso_m is not None and iso_prev is not None
                       and float((iso_m & iso_prev).mean()) >= NEW_BLOB_MIN * 0.5)
            base = kk % sub == 0 or (k - kk) > 1 and any(j % sub == 0 for j in range(kk, k))
            if base:
                extras_here = 0
            run = False
            is_extra = False
            if base:
                run = last_g is None or last_g.shape != g.shape or float(
                    (np.abs(g.astype(np.int16) - last_g.astype(np.int16)) > MASK_T).mean()) > UNCHANGED_MAX
                if not run:
                    out.append((ts, list(last_dets), False))
            elif (extras_total < budget and extras_here < EXTRA_PER_INTERVAL and ts - last_ocr_t >= EXTRA_GAP_S
                    and (iso_hit or _new_blob(cv2, np, masks[0], masks[1], m))):
                run = True
                is_extra = True
                extras_total += 1
                extras_here += 1
            if run:
                found = _detect(ocr, img, cv2)
                known = {d["norm"] for d in last_dets}
                unread = not any(d["norm"] not in known for d in found)
                if unread and frames[0] is not None and (
                        is_extra or (iso_m is not None and float(iso_m.mean()) >= NEW_BLOB_MIN * 0.5)):
                    # something new showed up but the whole frame reads nothing new: read the text-isolated copies
                    isos = [d for iso in _isolated(cv2, np, gfull, frames[0]) for d in _detect(ocr, iso, cv2)]
                    found = _merge_lines(found, isos)
                    pending = is_extra
                last_dets = found
                last_g, last_ocr_t = g, ts
                out.append((ts, list(last_dets), is_extra))
            elif pending:
                pending = False
                if not base:
                    out.append((ts, _detect(ocr, img, cv2), True))
                    last_ocr_t = ts
            masks = [masks[1], m]
            frames = [frames[1], gfull]
            iso_prev = iso_m
        vfr = not fps or fps <= 0 or (drift_hi - drift_lo) > min(1.5 / fps, 0.25)
        return out, "vfr" if vfr else "cfr", stamps_all
    finally:
        cap.release()


AMBIG_TIMING_CONF = 0.5  # ceiling when a VFR boundary is only known between two frames
START_COVER = 0.7  # share of the text's pixels that must already be in place to call a frame the start


def _refine_start(cv2, np, cap, fps, t_prev: float, t_first: float, box):
    """Frame time at which a text first shows, from pixels, between the last sample without it
    (``t_prev``) and the first with it: the text's own pixels are the bright / dark ones that
    changed between those two frames; the start is the first frame (that stays so for the next
    one) where >= START_COVER of them already have their final value. None when unclear."""
    if not fps or fps <= 0 or t_first <= t_prev:
        return None
    i0, i1 = int(t_prev * fps + 1e-6), int(round(t_first * fps))
    if i1 - i0 < 2:
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, i0)
    crops = []
    for _ in range(i0, i1 + 1):
        ok, img = cap.read()
        if not ok or img is None:
            return None
        crops.append(_crop(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), box, True).astype(np.int16))
    base, final = crops[0], crops[-1]
    if base.size == 0 or base.shape != final.shape:
        return None
    t_mask = (np.abs(final - base) > ISO_DIFF) & ((final > 200) | (final < 55))
    t_mask = cv2.morphologyEx(t_mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
    if int(t_mask.sum()) < ISO_MIN_PX:
        return None
    cover = [float((np.abs(c - final)[t_mask] <= 30).mean()) for c in crops]
    for j in range(1, len(crops) - 1):
        if cover[j] >= START_COVER and cover[j + 1] >= START_COVER:
            return (i0 + j) / fps
    return None


LATE_START_CONF = 0.4  # track confidence ceiling when the start time is only approximate


def _busy_around(cv2, np, cap, fps, t_first: float, box) -> bool:
    """Does the picture around the text move just before it was first seen?"""
    if not fps or fps <= 0:
        return True
    frames = _gray_frames(cv2, cap, fps, max(0.0, t_first - 0.4), t_first, box)
    if len(frames) < 2:
        return True
    h, w = frames[0][1].shape
    band = np.zeros((h, w), bool)
    # the crop is box +- margins: the band holding the text itself is the middle third of the rows
    band[h // 3: 2 * h // 3, :] = True
    moved = [float((np.abs(a[1].astype(np.int16) - b[1].astype(np.int16)) > MASK_T)[~band].mean())
             for a, b in zip(frames, frames[1:]) if a[1].shape == b[1].shape]
    return bool(moved) and float(np.mean(moved)) > BG_MOVE_MAX


def _run(cv2, np, ocr, path: str, duration: float, step: float) -> list[dict]:
    cap = cv2.VideoCapture(path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        samples, mode, frame_times = _sample(cv2, np, ocr, path, duration, step, fps)
        vfr = mode == "vfr"
        stamps = [x[0] for x in samples]
        open_tracks: list[dict] = []
        done: list[dict] = []
        for t, found, _x in samples:
            dets = list(found)
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
            i_next = bisect_right(stamps, t_last + 1e-9)
            nxt = stamps[i_next] if i_next < len(stamps) else None  # first sample after the text's last sighting
            i_prev = bisect_right(stamps, t_first - 1e-9) - 1
            prv = stamps[i_prev - 1] if i_prev >= 1 else None  # last sample before its first sighting
            # the bisected boundaries are the fallback times AND bound the settle window of a
            # track that only one or two OCR samples saw
            timing_note = None
            vfr_capped = False
            entrance_refined = False
            if vfr:
                # a frame is held until the next one: when no decoded frame lies between the last
                # sighting and the next sample the text ended exactly at that sample (and began at
                # its first sighting when none lies before it); otherwise only a midpoint is known
                entrance = ex = NOT_DETERMINED
                gap_in = prv is not None and any(prv < f < t_first - 1e-9 for f in frame_times)
                gap_out = nxt is not None and any(t_last + 1e-9 < f < nxt - 1e-9 for f in frame_times)
                t_in = (prv + t_first) / 2.0 if gap_in else t_first
                t_out = ((t_last + nxt) / 2.0 if gap_out else nxt) if nxt is not None else t_last
                if gap_in or gap_out:
                    timing_note = "variable-frame-rate source: start or end only known to between two frames"
                    vfr_capped = True
            else:
                t_out_b = _bisect(ocr, cv2, cap, t_last, nxt, tr["norm"], box) if nxt is not None else duration
                t_in_b = _bisect(ocr, cv2, cap, t_first, prv, tr["norm"], box) if prv is not None else 0.0
                t_in, t_out = t_in_b, t_out_b
                entrance, a_in = _entrance(cv2, np, cap, fps, t_first, t_set_in, t_set_out, duration, box,
                                           t_out_b - _BISECT_MARGIN)
                if a_in is not None:
                    t_in = a_in
                # the exit window runs a full OCR step past the last sighting, as before the scan added
                # closer samples (they can still sit inside a slow fade)
                ex, a_out = _exit(cv2, np, cap, fps, t_set_in, t_set_out, max(nxt, t_last + step) if nxt is not None else None, duration, box,
                                  t_in_b + _BISECT_MARGIN) if nxt is not None and max(nxt, t_last + step) < duration - 1e-9 \
                    else (NOT_DETERMINED, None)
                if a_out is not None:
                    t_out = a_out
            ref = _read(cap, cv2, (t_set_in + t_set_out) / 2.0) if not vfr else None  # middle of the stable span: fully opaque
            conf = _track_conf(len(tr["boxes"]), sum(tr["confs"]) / len(tr["confs"]))
            if vfr_capped:
                conf = min(conf, AMBIG_TIMING_CONF)
            if not vfr and entrance == NOT_DETERMINED and prv is not None and t_first - t_in <= 2 * SCAN_S + 1e-6:
                refined = _refine_start(cv2, np, cap, fps, prv, t_first, box)
                if refined is not None and prv <= refined <= t_first:
                    t_in = refined  # measured from the text's own pixels: no longer approximate
                    entrance_refined = True
            if (not vfr and entrance == NOT_DETERMINED and not entrance_refined
                    and t_first - t_in <= 2 * SCAN_S + 1e-6 and _busy_around(cv2, np, cap, fps, t_first, box)):
                # over a moving picture with no measurable animation the start is only known to
                # the sampling grid: it can sit a scan step or two late
                conf = min(conf, LATE_START_CONF)
                timing_note = "start time approximate (busy background)"
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
                             "box_track": [[round(tt, 3)] + [round(v, 4) for v in b] for tt, b in tr["boxes"]],
                             **({"timing_note": timing_note} if timing_note else {})},
            })
        out = _drop_fragments(out)
        out.sort(key=lambda r: (r["t_in"], r["box"][1], r["box"][0]))
        return out
    finally:
        cap.release()
