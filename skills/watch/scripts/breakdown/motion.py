"""Motion of one region over a segment: track, split into phases, fit easing.

``analyze`` follows a region (a normalised box, an OCR-resolved text, or the
largest moving area) frame by frame at the source fps, splits the movement
into anticipation / action / recovery / hold from its velocity, and fits each
move to the Godot 4 Tween TRANS x EASE set (plus a CSS cubic-bezier and Unity
keys, see ``easing``). ``snippets`` turns the result into generic engine code.

Nothing is forced: a lost track, a flat template or a move that is not one
clean motion leaves phases / curves ``"not determined"`` with a note, and a
missing dependency (cv2 / numpy / pytesseract) is a note, never an exception.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from config import memory_paths

from . import easing, media

SCHEMA = "watch.motion/1"
NOT_DETERMINED = "not determined"
WORK_W = 960  # tracking resolution (frames wider than this are downscaled)
MAX_FRAMES = 1800
MIN_SCORE = 0.45  # template-match score below which the region counts as lost
FLAT_STD = 3.0  # a template flatter than this cannot be told apart from its surroundings
MIN_MOVE_PX = 1.5  # total travel below this is "no motion"
NOISE_PX = 0.2  # per-frame travel treated as tracking noise
GAP_FRAMES = 2  # still frames bridged inside one move
HOLD_MIN_FRAMES = 3
MERGE_CLEAR_RMS = 0.005  # ...and either this tight or clearly below the split pair's RMS
MERGE_PEAK_TOL = 0.15  # measured vs fitted overshoot peak, relative
MERGE_MAX_RMS = 0.02  # a move + its return merge into one overshoot curve only when this tight
PERP_MAX = 0.5  # path wider than this share of its length is not one straight move


def _import_cv():
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    return cv2, np


def _skeleton(start: float, end: float, fps: float, video_key: str | None, ppu) -> dict:
    from .assemble import CONTRACT  # noqa: PLC0415 - deferred: assemble is heavier than motion
    return {"schema": SCHEMA, "contract": CONTRACT, "video_key": video_key,
            "segment": {"start": round(float(start), 4), "end": round(float(end), 4), "fps": round(float(fps), 4)},
            "region": {"box": [0.0, 0.0, 1.0, 1.0], "method": "whole_frame"},
            "px_per_unit": ppu, "track": [], "phases": [], "notes": []}


def _clamp_box(b):
    x, y, w, h = [float(c) for c in b]
    x = min(max(x, 0.0), 1.0)
    y = min(max(y, 0.0), 1.0)
    return [x, y, min(max(w, 0.0), 1.0 - x), min(max(h, 0.0), 1.0 - y)]


def parse_region(region):
    """-> ("box", [x,y,w,h]) | ("text", str) | (None, None)."""
    if region is None:
        return None, None
    if isinstance(region, (list, tuple)) and len(region) == 4:
        try:
            return "box", _clamp_box(region)
        except (TypeError, ValueError):
            return None, None
    text = str(region).strip()
    if not text:
        return None, None
    parts = [p.strip() for p in text.split(",")]
    if len(parts) == 4:
        try:
            return "box", _clamp_box([float(p) for p in parts])
        except ValueError:
            pass
    return "text", text


# ---- frames ----------------------------------------------------------------

def _read_frames(cv2, np, path: str, start: float, end: float, fps: float):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    try:
        i0 = max(0, int(round(start * fps)))
        i1 = max(i0 + 1, int(round(end * fps)))
        notes = []
        if i1 - i0 > MAX_FRAMES:
            i1 = i0 + MAX_FRAMES
            notes.append(f"segment longer than {MAX_FRAMES} frames; analysed the first {MAX_FRAMES}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, i0)
        times, grays = [], []
        W = H = 0
        sc = 1.0
        for i in range(i0, i1 + 1):
            ok, img = cap.read()
            if not ok or img is None:
                break
            if not W:
                H, W = img.shape[:2]
                sc = min(1.0, WORK_W / float(W))
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if sc < 1.0:
                g = cv2.resize(g, (int(round(W * sc)), int(round(H * sc))), interpolation=cv2.INTER_AREA)
            times.append(i / fps)
            grays.append(g)
        return {"times": times, "grays": grays, "W": W, "H": H, "sc": sc, "notes": notes}
    finally:
        cap.release()


def _resolve_text(cv2, path: str, text: str, start: float, end: float, fps: float):
    """OCR the segment for ``text`` -> (box, t, note|None). Needs pytesseract."""
    from . import typography as ty  # noqa: PLC0415
    try:
        ocr = ty._load_ocr()
    except RuntimeError as exc:
        return None, None, f"{exc}; text region not resolved"
    qn = ty._norm(text)
    if not qn:
        return None, None, "empty text region"
    cap = cv2.VideoCapture(str(path))
    try:
        n = 6
        ts = [start + (end - start) * k / (n - 1) for k in range(n)] if end > start else [start]
        best = None
        for t in ts:
            img = ty._read(cap, cv2, t)
            if img is None:
                continue
            for d in ty._detect(ocr, img, cv2):
                if d["norm"] == qn:
                    return d["box"], t, None
                if qn in d["norm"] and best is None:
                    best = (d["box"], t)
        if best:
            return best[0], best[1], None
    finally:
        cap.release()
    return None, None, f"text {text!r} not found by OCR in the segment"


def _auto_region(cv2, np, grays):
    """Box (work px) of the largest moving area at the frame pair with the most change, plus its frame index."""
    n = len(grays)
    if n < 3:
        return None
    h, w = grays[0].shape
    best_i, best_area, best_mask = None, 0, None
    for i in range(1, n):
        d = cv2.absdiff(grays[i], grays[i - 1])
        d = cv2.GaussianBlur(d, (5, 5), 0)
        mask = (d > 25).astype(np.uint8)
        area = int(mask.sum())
        if area > best_area:
            best_i, best_area, best_mask = i, area, mask
    if best_i is None or best_area < max(30, 0.0003 * w * h):
        return None
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.dilate(best_mask, k)
    cnt, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    comps = [(stats[j, cv2.CC_STAT_AREA], j) for j in range(1, cnt)]
    if not comps:
        return None
    top = max(a for a, _ in comps)
    keep = [j for a, j in comps if a >= 0.2 * top]
    x0 = min(stats[j, cv2.CC_STAT_LEFT] for j in keep)
    y0 = min(stats[j, cv2.CC_STAT_TOP] for j in keep)
    x1 = max(stats[j, cv2.CC_STAT_LEFT] + stats[j, cv2.CC_STAT_WIDTH] for j in keep)
    y1 = max(stats[j, cv2.CC_STAT_TOP] + stats[j, cv2.CC_STAT_HEIGHT] for j in keep)
    pad = 3
    x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
    return (x0, y0, x1 - x0, y1 - y0), best_i


# ---- tracking ----------------------------------------------------------------

def _subpixel(res, loc):
    x, y = loc
    dx = dy = 0.0
    if 0 < x < res.shape[1] - 1:
        a, b, c = float(res[y, x - 1]), float(res[y, x]), float(res[y, x + 1])
        den = a - 2 * b + c
        if abs(den) > 1e-9:
            dx = max(-0.5, min(0.5, 0.5 * (a - c) / den))
    if 0 < y < res.shape[0] - 1:
        a, b, c = float(res[y - 1, x]), float(res[y, x]), float(res[y + 1, x])
        den = a - 2 * b + c
        if abs(den) > 1e-9:
            dy = max(-0.5, min(0.5, 0.5 * (a - c) / den))
    return dx, dy


def _track_template(cv2, np, grays, r: int, box):
    """Track ``box`` (work px, taken from frame ``r``) both ways. -> (positions, note).

    ``positions[i]`` is ``(x, y, score)`` or None where the region was lost.
    """
    n = len(grays)
    h, w = grays[0].shape
    x, y, bw, bh = [int(v) for v in box]
    bw, bh = max(bw, 4), max(bh, 4)
    x, y = max(0, min(x, w - bw)), max(0, min(y, h - bh))
    tpl = grays[r][y:y + bh, x:x + bw]
    if float(tpl.std()) < FLAT_STD:
        return None, "region has no texture to follow (flat colour); tracking not determined"
    pos = [None] * n
    pos[r] = (float(x), float(y), 1.0)
    margin = int(max(2 * max(bw, bh), 0.15 * max(w, h)))
    for step in (1, -1):
        px, py = float(x), float(y)
        i = r + step
        while 0 <= i < n:
            x0, y0 = max(0, int(px) - margin), max(0, int(py) - margin)
            x1, y1 = min(w, int(px) + bw + margin), min(h, int(py) + bh + margin)
            win = grays[i][y0:y1, x0:x1]
            if win.shape[0] < bh or win.shape[1] < bw:
                break
            res = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
            res = np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0)
            _, mx, _, loc = cv2.minMaxLoc(res)
            if mx < MIN_SCORE:
                break
            dx, dy = _subpixel(res, loc)
            px, py = x0 + loc[0] + dx, y0 + loc[1] + dy
            pos[i] = (px, py, float(mx))
            i += step
    return pos, None


# ---- phases ------------------------------------------------------------------

def _runs(v, np):
    """Moves as (a, b, sign): v[a..b] is one stretch of same-sign motion."""
    peak = float(np.max(np.abs(v))) if len(v) else 0.0
    seed = max(0.05 * peak, 0.3)
    low = max(0.01 * peak, NOISE_PX)
    sign = np.sign(v)
    runs = []
    covered = -1
    for i in range(len(v)):
        if i <= covered or abs(v[i]) <= seed:
            continue
        s = sign[i]

        def reach(start, direction):
            cur = start
            while True:
                nxt = None
                for g in range(1, GAP_FRAMES + 2):
                    j = cur + direction * g
                    if 0 <= j < len(v) and abs(v[j]) > low and sign[j] == s:
                        nxt = j
                        break
                if nxt is None:
                    return cur
                cur = nxt
        a, b = reach(i, -1), reach(i, 1)
        runs.append([a, b, int(s)])
        covered = b
    # merge overlapping same-sign runs (a later seed can reach back into an earlier run)
    merged = []
    for r in runs:
        if merged and merged[-1][2] == r[2] and r[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], r[1])
        else:
            merged.append(r)
    return merged


def _split(P, np):
    """Principal-axis progress ``q`` and its moves; None when there is no motion."""
    D = P - P[0]
    c = P - P.mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    d = vt[0]
    q = D @ d
    perp = D @ np.array([-d[1], d[0]])
    along = float(q.max() - q.min())
    if along < MIN_MOVE_PX:
        return None
    v = np.diff(q)
    runs = _runs(v, np)
    return q, float(perp.max() - perp.min()) / along, runs


def _mean_conf(scores) -> float:
    return max(0.0, min(1.0, sum(scores) / len(scores))) if scores else 0.0


def _fit_span(T, q, f, ia, ib, prev_end, next_start) -> dict:
    """Fit the move over samples ia..ib, searching its exact start and end.

    The window runs on past ``ib`` (up to the next move) because an ease tail is
    below the velocity threshold; ``fit_bounds`` picks the end that fits best.
    """
    n = len(T)
    wa = max(prev_end, ia - 2)
    we = min(n - 1, max(ib, min(next_start, ib + max(8, 2 * (ib - ia)))))
    tt = [T[i] for i in range(wa, we + 1)]
    qq = [float(q[i]) for i in range(wa, we + 1)]
    s_c = [T[ia] + j * f / 4.0 for j in range(-4, 5)]
    e_c = [T[ib] + j * f / 4.0 for j in range(0, int((T[we] - T[ib]) / (f / 4.0) + 1e-9) + 1)] or [T[ib]]
    return easing.fit_bounds(tt, qq, s_c, e_c)


def _rms_of(fit) -> float:
    r = fit.get("error_rms") if isinstance(fit, dict) else None
    return float(r) if isinstance(r, (int, float)) and fit.get("godot") != NOT_DETERMINED else float("inf")


def _accept_merge(T, q, f, ph, nxt, span) -> bool:
    """A move + its return is one overshoot curve only when that is clearly the better story.

    The merged fit must be tight, clearly better than the split pair, and its own
    peak overshoot must match the measured one (a BACK curve cannot explain a
    separate return of a different size).
    """
    g = span.get("godot")
    merged = _rms_of(span)
    if g == NOT_DETERMINED or merged > MERGE_MAX_RMS:
        return False
    split = max(_rms_of(_fit_span(T, q, f, ph["ia"], ph["ib"], ph["prev_end"], nxt["ia"])),
                _rms_of(_fit_span(T, q, f, nxt["ia"], nxt["ib"], ph["ib"], nxt["next_start"])))
    if not (merged <= MERGE_CLEAR_RMS or merged < 0.5 * split):
        return False
    fc = easing.curve(g["trans"], g["ease"])
    fitted = max(fc(i / 200.0) for i in range(201)) - 1.0
    q0, q_end = float(q[max(ph["prev_end"], ph["ia"] - 2)]), float(q[nxt["ib"]])
    total = q_end - q0
    if abs(total) < 1e-9:
        return False
    peak = max(float(q[i]) for i in range(ph["ia"], nxt["ib"] + 1))
    measured = (peak - q_end) / total
    return abs(measured - fitted) <= max(MERGE_PEAK_TOL * max(fitted, 0.0), 0.005)


def _build_phases(T, P, scores, fps, ppu, W, H, notes):
    import numpy as np  # noqa: PLC0415
    n = len(T)
    sp = _split(P, np)
    if sp is None:
        notes.append(f"no motion detected (travel below {MIN_MOVE_PX} px); phases not determined")
        return []
    q, perp_ratio, runs = sp
    if not runs:
        notes.append("no move found above the noise floor; phases not determined")
        return []
    deltas = [float(q[b + 1] - q[a]) for a, b, _ in runs]
    main_i = max(range(len(runs)), key=lambda k: abs(deltas[k]))
    main_delta = abs(deltas[main_i])
    keep = [k for k in range(len(runs)) if k == main_i or abs(deltas[k]) >= max(1.5, 0.03 * main_delta)]
    runs = [runs[k] for k in keep]
    deltas = [deltas[k] for k in keep]
    main_i = keep.index(main_i)
    flip = -1.0 if deltas[main_i] < 0 else 1.0
    q = q * flip
    deltas = [d * flip for d in deltas]
    clean = True
    same_sign = [k for k, d in enumerate(deltas) if d > 0]
    if len(same_sign) > 1:
        clean = False
        notes.append("more than one move in the same direction; easing curves not determined (not one clean move)")
    if perp_ratio > PERP_MAX:
        clean = False
        notes.append("the path is not a straight move (it curves or changes direction); easing curves not determined")
    conf_track = _mean_conf(scores)
    f = 1.0 / fps if fps else (T[1] - T[0])

    phases = []
    for k, (a, b, _) in enumerate(runs):
        ia, ib = a, b + 1
        if k == main_i:
            name = "action"
        elif deltas[k] > 0:
            name = "action"
        elif k < main_i:
            name = "anticipation"
        else:
            name = "recovery"
        prev_end = runs[k - 1][1] + 1 if k > 0 else 0
        next_start = runs[k + 1][0] if k + 1 < len(runs) else n - 1
        phases.append({"name": name, "run": k, "ia": ia, "ib": ib, "prev_end": prev_end, "next_start": next_start})

    # an overshoot ease (BACK / ELASTIC / SPRING) reads as a move plus a smaller return; try the whole span as one
    merged_phases = []
    k = 0
    while k < len(phases):
        ph = phases[k]
        nxt = phases[k + 1] if k + 1 < len(phases) else None
        if (clean and nxt is not None and ph["name"] == "action" and nxt["name"] == "recovery"
                and nxt["ia"] - ph["ib"] <= GAP_FRAMES
                and abs(deltas[nxt["run"]]) < abs(deltas[ph["run"]])):
            span = _fit_span(T, q, f, ph["ia"], nxt["ib"], ph["prev_end"], nxt["next_start"])
            if _accept_merge(T, q, f, ph, nxt, span):
                notes.append("the move and its return were fitted as one overshoot curve")
                merged_phases.append({**ph, "ib": nxt["ib"], "next_start": nxt["next_start"], "span": span})
                k += 2
                continue
        merged_phases.append(ph)
        k += 1
    phases = merged_phases

    out = []
    refined_end = []
    for ph in phases:
        ia, ib = ph["ia"], ph["ib"]
        t_s, t_e = T[ia], T[ib]
        curve: object = NOT_DETERMINED
        cnote = None
        if clean and ib - ia + 1 >= easing.MIN_SAMPLES:
            curve = ph.get("span") or _fit_span(T, q, f, ia, ib, ph["prev_end"], ph["next_start"])
            if curve.get("godot") != NOT_DETERMINED:
                t_s, t_e = curve["t_start"], curve["t_end"]
        elif clean:
            cnote = "too few frames to fit a curve"
        ie = ib
        if curve is not NOT_DETERMINED and curve.get("godot") != NOT_DETERMINED:
            ie = max(ib, min(n - 1, int(math.ceil((t_e - T[0]) * fps - 1e-6))))
        dx_px, dy_px = float(P[ie][0] - P[ia][0]), float(P[ie][1] - P[ia][1])
        out.append(_phase_dict(ph["name"], t_s, t_e, dx_px, dy_px, W, H, ppu, curve, cnote, conf_track))
        refined_end.append(t_e)
    # holds: still stretches between moves and after the last move, measured from the refined end of the move
    holds = []
    for k, ph in enumerate(phases):
        stop = phases[k + 1]["ia"] if k + 1 < len(phases) else n - 1
        h_end = T[stop]
        if k + 1 < len(out):
            h_end = min(h_end, out[k + 1]["t_start"])
        if h_end - refined_end[k] >= HOLD_MIN_FRAMES * f - 1e-9:
            holds.append(_phase_dict("hold", refined_end[k], h_end, 0.0, 0.0, W, H, ppu,
                                     "not applicable", None, conf_track))
    out_all = out + holds
    out_all.sort(key=lambda d: d["t_start"])
    if out_all[0]["t_start"] - T[0] > 2 * f:
        notes.append(f"region at rest until t={out_all[0]['t_start']:.3f}s")
    return out_all


def _phase_dict(name, t_s, t_e, dx_px, dy_px, W, H, ppu, curve, cnote, conf_track) -> dict:
    px = math.hypot(dx_px, dy_px)
    disp = {"x": round(dx_px / W, 5) if W else 0.0, "y": round(dy_px / H, 5) if H else 0.0, "px": round(px, 2)}
    if ppu:
        disp["units"] = round(px / ppu, 4)
        disp["units_x"] = round(dx_px / ppu, 4)
        disp["units_y"] = round(dy_px / ppu, 4)
    ev = {"t_start": round(t_s, 4), "t_end": round(t_e, 4), "method": "measured", "confidence": round(conf_track, 3)}
    if cnote:
        ev["note"] = cnote
    d = {"name": name, "t_start": round(t_s, 4), "t_end": round(t_e, 4), "duration": round(t_e - t_s, 4),
         "displacement": disp, "curve": curve, "evidence": ev}
    return d


# ---- public API -----------------------------------------------------------------

def analyze(conn, video_key: str, start: float, end: float, region=None, px_per_unit: float | None = None,
            store_dir=None, fetch=None) -> dict:
    out = _skeleton(start, end, 0.0, video_key, px_per_unit)
    notes = out["notes"]
    try:
        _analyze(conn, video_key, float(start), float(end), region, px_per_unit, store_dir, fetch, out)
    except Exception as exc:  # noqa: BLE001 - a command never raises
        notes.append(f"motion analysis failed ({type(exc).__name__}: {exc}); not determined")
        out["track"], out["phases"] = [], []
    return out


def _analyze(conn, video_key, start, end, region, ppu, store_dir, fetch, out):
    notes = out["notes"]
    if end <= start:
        notes.append("segment end must be after its start; not determined")
        return
    root = Path(store_dir) if store_dir is not None else Path(memory_paths()["store_dir"])
    m = media.ensure_media(conn, video_key, root, fetch=fetch if fetch is not None else (lambda *_: None))
    if m.get("path"):
        media.pin(root, video_key)  # secured: keep it even if the analysis below fails
    if not m.get("path"):
        notes.append("only stored frames are available (no full-fps media); motion not determined")
        return
    fps = float(m.get("fps") or 0.0)
    out["segment"]["fps"] = round(fps, 4)
    if fps <= 0:
        notes.append("source frame rate unknown; motion not determined")
        return
    dur = float(m.get("duration") or 0.0)
    if dur and start >= dur:
        notes.append("segment starts after the end of the video; not determined")
        return
    if dur and end > dur:
        end = dur
        out["segment"]["end"] = round(end, 4)
        notes.append("segment end clamped to the video duration")
    try:
        cv2, np = _import_cv()
    except Exception:  # noqa: BLE001
        notes.append("opencv (cv2) / numpy are not installed; motion not determined")
        return
    fr = _read_frames(cv2, np, m["path"], start, end, fps)
    if not fr or len(fr["grays"]) < 4:
        notes.append("could not read enough frames from the segment; not determined")
        return
    notes.extend(fr["notes"])
    grays, T, W, H, sc = fr["grays"], fr["times"], fr["W"], fr["H"], fr["sc"]
    n = len(grays)
    wh, ww = grays[0].shape

    kind, val = parse_region(region)
    ref_i = 0
    box_px = None
    if kind == "box":
        out["region"] = {"box": [round(c, 5) for c in val], "method": "given"}
        box_px = (val[0] * ww, val[1] * wh, val[2] * ww, val[3] * wh)
    elif kind == "text":
        box, t_found, why = _resolve_text(cv2, m["path"], val, start, end, fps)
        if box is not None:
            out["region"] = {"box": [round(c, 5) for c in box], "method": "ocr", "text": val}
            box_px = (box[0] * ww, box[1] * wh, box[2] * ww, box[3] * wh)
            ref_i = min(n - 1, max(0, int(round((t_found - T[0]) * fps))))
        else:
            out["region"] = {"box": [0.0, 0.0, 1.0, 1.0], "method": "whole_frame",
                             "note": f"{why}; whole frame used, following the largest moving area"}
    elif region is not None:
        out["region"] = {"box": [0.0, 0.0, 1.0, 1.0], "method": "whole_frame",
                         "note": "region not understood; whole frame used, following the largest moving area"}
    mode = out["region"]["method"]  # given | ocr | whole_frame (unresolved) -> auto when nothing was asked
    if region is None:
        mode = "auto"
    cands = [(box_px, ref_i)] if box_px is not None else []
    if box_px is None:
        auto = _auto_region(cv2, np, grays)
        if auto is None:
            notes.append("no moving area found in the segment; motion not determined")
            out["region"] = {k: v for k, v in out["region"].items() if k != "tracked_box"}
            return
        box_px, p = auto
        # the template may reach past the frame edge in the other frames: try both frames of the pair
        cands = [(box_px, p - 1), (box_px, p)]
    best = None
    for bpx, ri in cands:
        pos_c, why = _track_template(cv2, np, grays, ri, bpx)
        if pos_c is None:
            best = best or (None, why, bpx, ri)
            continue
        got = [q for q in pos_c if q is not None]
        key = (len(got), sum(q[2] for q in got))
        if best is None or best[0] is None or key > best[4]:
            best = (pos_c, None, bpx, ri, key)
    pos, why, box_px, ref_i = best[0], best[1], best[2], best[3]
    if pos is None:
        notes.append(why)
        return
    if mode in ("auto", "whole_frame"):
        bx = [box_px[0] / ww, box_px[1] / wh, box_px[2] / ww, box_px[3] / wh]
        if mode == "whole_frame":
            out["region"]["tracked_box"] = [round(c, 5) for c in bx]
            out["region"]["tracked_method"] = "auto_motion"
        else:
            out["region"] = {"box": [round(c, 5) for c in bx], "method": "auto_motion",
                             "note": f"largest moving area at t={T[ref_i]:.3f}s"}
    valid = [i for i, p in enumerate(pos) if p is not None]
    lost = n - len(valid)
    bw, bh = int(max(box_px[2], 4)), int(max(box_px[3], 4))
    track = []
    for i in valid:
        x, y, s = pos[i]
        track.append({"t": round(T[i], 5), "x": round(x / sc / W, 5), "y": round(y / sc / H, 5),
                      "w": round(bw / sc / W, 5), "h": round(bh / sc / H, 5), "score": round(s, 3)})
    out["track"] = track
    if lost:
        first_bad = next(i for i, p in enumerate(pos) if p is None)
        notes.append(f"tracking lost the region at t={T[first_bad]:.3f}s ({lost} of {n} frames); "
                     "phases and curves not determined")
        return
    Tn = list(T)
    P = np.array([[pos[i][0] / sc, pos[i][1] / sc] for i in range(n)], dtype=float)
    scores = [pos[i][2] for i in range(n)]
    out["phases"] = _build_phases(Tn, P, scores, fps, ppu, W, H, notes)


# ---- snippets -----------------------------------------------------------------------

def _fmt(x, d=3) -> str:
    return f"{x:.{d}f}".rstrip("0").rstrip(".") if isinstance(x, (int, float)) else str(x)


def _godot_line(ph: dict) -> list[str]:
    c = ph["curve"]
    d = ph["displacement"]
    dur = _fmt(ph["duration"], 3)
    lines = [f"# {ph['name']}: {ph['t_start']}s -> {ph['t_end']}s, moves {_fmt(d['px'], 1)} px "
             f"(x {_fmt(d['x'], 4)}, y {_fmt(d['y'], 4)} of the frame)" + (f" = {_fmt(d['units'], 3)} game units" if "units" in d else "")]
    if ph["name"] == "hold":
        lines.append(f"tween.tween_interval({dur})")
        return lines
    if not isinstance(c, dict) or not isinstance(c.get("godot"), dict):
        lines.append(f"# curve not determined for this {ph['name']}; pick one by eye, nothing is assumed")
        lines.append(f"tween.tween_property(node, \"<property>\", <to>, {dur})")
        return lines
    g = c["godot"]
    dx, dy = d.get("units_x", None), d.get("units_y", None)
    ax, ay = abs(d["x"]), abs(d["y"])
    if ay <= 0.2 * ax:
        prop, sign_note = "position:x", f"<from_x> {'+' if (dx if dx is not None else d['x']) >= 0 else '-'} " \
                                         f"{_fmt(abs(dx) if dx is not None else abs(d['x']), 4)}"
    elif ax <= 0.2 * ay:
        prop, sign_note = "position:y", f"<from_y> {'+' if (dy if dy is not None else d['y']) >= 0 else '-'} " \
                                         f"{_fmt(abs(dy) if dy is not None else abs(d['y']), 4)}"
    else:
        prop, sign_note = "position", "<from> + Vector2(" + _fmt(dx if dx is not None else d['x'], 4) + ", " + \
                          _fmt(dy if dy is not None else d['y'], 4) + ")"
    unit = "game units" if "units" in d else "normalised frame units"
    lines.append(f"# <to> = {sign_note}  ({unit}; frame y grows downward)")
    ease = f".set_ease(Tween.EASE_{g['ease']})" if g["trans"] != "LINEAR" else ""
    lines.append(f"tween.tween_property(node, \"{prop}\", <to>, {dur})"
                 f".set_trans(Tween.TRANS_{g['trans']}){ease}")
    ru = c.get("runner_up")
    if c.get("ambiguous") and ru:
        lines.append(f"# nearly indistinguishable from TRANS_{ru['trans']} / EASE_{ru['ease']} "
                     f"(confidence {c['confidence']})")
    return lines


def snippets(m: dict) -> dict:
    """Generic engine snippets: Godot 4 Tween, Unity keys JSON, markdown table."""
    phases = m.get("phases") or []
    seg = m.get("segment", {})
    gd = ["# Generic Godot 4 Tween sketch measured from a reference clip.",
          f"# Segment {seg.get('start')}s - {seg.get('end')}s at {seg.get('fps')} fps. Replace `node` and the <placeholders>.",
          "var tween := create_tween()"]
    if not phases:
        gd.append("# no phases determined; nothing to emit")
    lead = (phases[0]["t_start"] - float(seg.get("start") or 0.0)) if phases else 0.0
    if lead > 0.05:
        gd.append(f"# at rest for the first {_fmt(lead, 3)}s of the segment")
        gd.append(f"tween.tween_interval({_fmt(lead, 3)})")
    for ph in phases:
        gd.extend(_godot_line(ph))
    unity_phases = []
    for ph in phases:
        c = ph["curve"]
        if ph["name"] == "hold" or not isinstance(c, dict):
            keys = NOT_DETERMINED
            note = "not applicable" if ph["name"] == "hold" else NOT_DETERMINED
        else:
            keys = c.get("unity_keys", NOT_DETERMINED)
            note = c.get("godot") if isinstance(c.get("godot"), dict) else NOT_DETERMINED
        unity_phases.append({"phase": ph["name"], "duration": ph["duration"], "keys": keys, "curve": note,
                             "displacement_px": ph["displacement"]["px"]})
    main = next((u for u in unity_phases if u["phase"] == "action" and isinstance(u["keys"], list)), None)
    unity = {"schema": "watch.motion.unity/1",
             "note": "time and value are normalised 0..1 per phase; scale value by the displacement and time by the duration",
             "keys": main["keys"] if main else NOT_DETERMINED, "phases": unity_phases}
    has_units = any("units" in ph["displacement"] for ph in phases)
    head = "| phase | start s | duration ms | displacement px | displacement (frame units x, y) |"
    sep = "|---|---|---|---|---|"
    if has_units:
        head += " game units |"
        sep += "---|"
    head += " curve | CSS cubic-bezier | confidence |"
    sep += "---|---|---|"
    rows = [head, sep]
    for ph in phases:
        c, d = ph["curve"], ph["displacement"]
        if isinstance(c, dict) and isinstance(c.get("godot"), dict):
            g = c["godot"]
            cur = f"{g['trans']}/{g['ease']}"
            if c.get("ambiguous") and c.get("runner_up"):
                cur += f" (or {c['runner_up']['trans']}/{c['runner_up']['ease']})"
            bez = "cubic-bezier(" + ", ".join(_fmt(v) for v in c["bezier"]) + ")" if isinstance(c.get("bezier"), list) else NOT_DETERMINED
            conf = _fmt(c.get("confidence", 0.0), 2)
        else:
            cur = "n/a" if ph["name"] == "hold" else NOT_DETERMINED
            bez, conf = ("n/a" if ph["name"] == "hold" else NOT_DETERMINED), "-"
        row = f"| {ph['name']} | {_fmt(ph['t_start'], 3)} | {_fmt(ph['duration'] * 1000.0, 0)} | {_fmt(d['px'], 1)} | " \
              f"{_fmt(d['x'], 4)}, {_fmt(d['y'], 4)} |"
        if has_units:
            row += f" {_fmt(d['units'], 3) if 'units' in d else '-'} |"
        row += f" {cur} | {bez} | {conf} |"
        rows.append(row)
    if not phases:
        rows.append("| _not determined_ |" + " |" * (len(head.split("|")) - 3))
    table = "\n".join(rows) + "\n"
    for note in m.get("notes", []):
        table += f"\n> {note}"
    return {"godot.gd": "\n".join(gd) + "\n", "unity.json": json.dumps(unity, indent=1) + "\n",
            "table.md": table.rstrip() + "\n"}
