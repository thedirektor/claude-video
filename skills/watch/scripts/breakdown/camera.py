"""Per-shot camera motion from dense optical flow (OpenCV Farneback).

Median flow gives pan / tilt, the radial component gives zoom, and a flow that
changes sign between sampled pairs is handheld. Content moves opposite to the
camera, so content moving left means the camera panned right. Without OpenCV
every field is ``"not determined"`` with a note; nothing is guessed.
"""
from __future__ import annotations

import re

NOT_DETERMINED = "not determined"
MAX_PAIRS = 8
WORK_W = 320
PAN_MIN = 0.04  # sustained translation, fraction of frame width per second
ZOOM_MIN = 0.04  # sustained radial expansion, fraction of frame per second
STATIC_PAIRS = 0.85
MIN_DIRECTIONAL_S = 0.4  # shorter shots can show a still camera, not a move (cuts and flashes sit at their edges)
MIN_DIRECTIONAL = 3  # frame pairs needed before a pan / tilt / zoom / handheld label (static needs 2)
SHAKE_MIN = 0.09  # jitter (frame fraction per second) that reads as handheld; measured fit noise on still shots reaches ~0.035, subject-driven fits ~0.08
PAIR_GAP_S = 0.12
INLIER_SHARE = 0.5  # share of tracked features the fitted camera transform must explain
MIN_INLIERS = 10
RANSAC_PX = 1.5
STATIC_PX = 0.4  # a tracked feature that moved less than this (work pixels) did not move
STATIC_SHARE = 0.35  # share of tracked features that must be at rest, spread over the picture, for a still camera
STATIC_COVER = 0.3  # features at rest must span this much of the picture (one still logo in a corner is not a still camera)
MIN_FEATURES = 12  # corners needed in a frame before its motion can be tracked
_CROP = re.compile(r"crop=(\d+):(\d+):(\d+):(\d+)")


def _import_cv():
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    return cv2, np


def _result(shot: dict, kind: str, conf: float, notes: list[str], **measurements) -> dict:
    """``evidence`` keeps the contract keys only; the numbers behind the label live in ``measurements``."""
    ev = {"t_start": round(shot.get("start", 0.0), 6), "t_end": round(shot.get("end", 0.0), 6),
          "method": "measured", "confidence": round(max(0.0, min(1.0, conf)), 3)}
    out = {"type": kind, "confidence": ev["confidence"], "evidence": ev, "notes": notes}
    if measurements:
        out["measurements"] = measurements
    return out


def _nd(shot: dict, note: str) -> dict:
    return _result(shot, NOT_DETERMINED, 0.0, [note])


def _parse_crop(crop: str | None) -> tuple[int, int, int, int] | None:
    """``crop=w:h:x:y`` (as ``shots.active_crop`` returns it) -> ``(w, h, x, y)``."""
    m = _CROP.fullmatch((crop or "").strip())
    return tuple(int(v) for v in m.groups()) if m else None


def camera_motion(path: str, shot: dict, crop: str | None = None) -> dict:
    """Camera motion of one shot. ``crop`` is the active picture (``shots.active_crop``): bars and
    pillars carry no features, only dilute the work resolution."""
    try:
        cv2, np = _import_cv()
    except Exception:  # ImportError, or a broken binary wheel
        return _nd(shot, "opencv (cv2) is not installed; camera motion not determined")
    try:
        return _measure(cv2, np, str(path), shot, _parse_crop(crop))
    except Exception as exc:  # never raise out of a command
        return _nd(shot, f"camera motion failed ({type(exc).__name__}); not determined")


def _gray(cv2, img, crop):
    if img is None:
        return None
    h, w = img.shape[:2]
    if crop:
        cw, ch, cx, cy = crop
        if cx + cw <= w and cy + ch <= h:  # a crop from another resolution is ignored, not guessed
            img = img[cy:cy + ch, cx:cx + cw]
            h, w = img.shape[:2]
    img = cv2.resize(img, (WORK_W, max(2, int(h * WORK_W / w))), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _textured(cv2, gray) -> bool:
    """Enough corners to tell motion from no motion. A flat or near-flat frame looks identical
    whether the camera moved or not."""
    pts = cv2.goodFeaturesToTrack(gray, maxCorners=60, qualityLevel=0.01, minDistance=7)
    return pts is not None and len(pts) >= MIN_FEATURES


def _read_at(cv2, cap, t: float):
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, img = cap.read()
    return img if ok else None


def _read_pair(cv2, cap, t0: float, steps: int):
    """Frame at ``t0`` and the one ``steps`` frames later: one seek, then plain sequential reads."""
    cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000.0)
    ok, a = cap.read()
    if not ok or a is None:
        return None, None
    for _ in range(steps - 1):
        if not cap.grab():
            return a, None
    ok, b = cap.read()
    return a, (b if ok else None)


def _spread(np, pts, w: int, h: int) -> float:
    """Smaller of the horizontal / vertical extent (10th-90th percentile) of ``pts`` (N x 2), as a
    share of the frame. One moving subject covers a fraction of the picture; a camera move does not."""
    if len(pts) < 2:
        return 0.0
    lo, hi = np.percentile(pts, 10, axis=0), np.percentile(pts, 90, axis=0)
    return float(min((hi[0] - lo[0]) / w, (hi[1] - lo[1]) / h))


def _pair(cv2, np, a, b, gap: float):
    """``(flow, static)`` between two gray frames.

    ``flow`` is (pan_x, pan_y, zoom) per second or None: corner features are tracked and a similarity
    transform fitted, so translation and scale are estimated separately (a pan never leaks into the
    zoom term). ``static`` is True when a large share of the tracked features, spread over the picture,
    did not move at all: a camera move displaces the whole frame, so a subject moving in front of
    a fixed background cannot fake that (and cannot defeat the fit's inlier test when it dominates).
    """
    pts = cv2.goodFeaturesToTrack(a, maxCorners=200, qualityLevel=0.01, minDistance=7)
    if pts is None or len(pts) < MIN_FEATURES:
        return None, False
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None, winSize=(21, 21), maxLevel=3)
    ok = st.reshape(-1) == 1
    if int(ok.sum()) < MIN_FEATURES:
        return None, False
    h, w = a.shape
    src, dst = pts[ok].reshape(-1, 2), nxt[ok].reshape(-1, 2)
    zero = np.hypot(*(dst - src).T) < STATIC_PX
    if (int(zero.sum()) >= max(MIN_INLIERS, STATIC_SHARE * len(src))
            and _spread(np, src[zero], w, h) >= STATIC_COVER):
        return (0.0, 0.0, 0.0), True
    m, inl = cv2.estimateAffinePartial2D(pts[ok], nxt[ok], method=cv2.RANSAC, ransacReprojThreshold=RANSAC_PX)
    if m is None or inl is None or int(inl.sum()) < MIN_INLIERS or int(inl.sum()) < INLIER_SHARE * int(ok.sum()):
        return None, False
    scale = float(np.hypot(m[0, 0], m[1, 0]))
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    ncx = m[0, 0] * cx + m[0, 1] * cy + m[0, 2]
    ncy = m[1, 0] * cx + m[1, 1] * cy + m[1, 2]
    return ((ncx - cx) / w / gap, (ncy - cy) / h / gap, (scale - 1.0) / gap), False


def _measure(cv2, np, path: str, shot: dict, crop=None) -> dict:
    start, end = float(shot["start"]), float(shot["end"])
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return _nd(shot, "video could not be opened; camera motion not determined")
    try:
        # stay inside the shot, away from the boundary frames (dissolves, flashes)
        margin = min(0.1, (end - start) * 0.1)
        lo, hi = start + margin, end - margin
        gap = min(PAIR_GAP_S, (hi - lo) / 3)
        if gap < 0.03:
            return _nd(shot, "shot too short to measure camera motion")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        steps = max(1, round(gap * fps)) if fps > 0 else 0
        if steps:
            gap = steps / fps  # the real spacing of the two frames
        count = min(MAX_PAIRS, max(2, int((hi - lo - gap) / gap) + 1))
        flows, still, flat, seen = [], 0, 0, 0
        for t0 in np.linspace(lo, hi - gap, count):
            if steps:
                ra, rb = _read_pair(cv2, cap, float(t0), steps)
                a, b = _gray(cv2, ra, crop), _gray(cv2, rb, crop)
            else:  # unknown frame rate: seek for each frame
                a, b = (_gray(cv2, _read_at(cv2, cap, float(t)), crop) for t in (t0, t0 + gap))
            if a is None or b is None:
                continue
            seen += 1
            if not _textured(cv2, a):  # nothing to track: no motion evidence either way
                flat += 1
                continue
            if float(np.abs(a.astype(np.float32) - b).mean()) < 0.5:
                still += 1
                flows.append((0.0, 0.0, 0.0))
                continue
            f, at_rest = _pair(cv2, np, a, b, gap)
            if at_rest:
                still += 1
            if f is not None:
                flows.append(f)
    finally:
        cap.release()
    if seen < 2:
        return _nd(shot, "too few readable frame pairs; camera motion not determined")
    if flat == seen:
        return _nd(shot, "picture is textureless (flat or blank); camera motion not determined")
    if still == seen:  # identical textured frames, or the background at rest in every pair: nothing moves
        return _result(shot, "static", 0.8, [], pairs=seen)
    if still >= STATIC_PAIRS * (seen - flat) and still >= 2:  # at rest in all but a stray pair
        return _result(shot, "static", 0.65, [], pairs=seen - flat, at_rest_pairs=still)
    if end - start < MIN_DIRECTIONAL_S:
        return _nd(shot, "shot too short to name a camera move; camera motion not determined")
    if len(flows) < MIN_DIRECTIONAL or len(flows) < 0.5 * seen:
        return _nd(shot, "not enough trackable features; camera motion not determined")
    arr = np.array(flows)
    dx, dy, zm = arr[:, 0], arr[:, 1], arr[:, 2]
    med = {"x": float(np.median(dx)), "y": float(np.median(dy)), "z": float(np.median(zm))}
    def _mad(v):  # robust spread: stray bad fits (moving subjects) must not read as shake
        return float(1.4826 * np.median(np.abs(v - np.median(v))))
    sd = {"x": _mad(dx), "y": _mad(dy), "z": _mad(zm)}
    jitter = max(sd["x"], sd["y"], sd["z"] * 0.5)
    stats = {"pairs": len(flows), "pan_x_per_s": round(med["x"], 4), "pan_y_per_s": round(med["y"], 4),
             "zoom_per_s": round(med["z"], 4), "jitter": round(jitter, 4)}
    comp = {"x": abs(med["x"]), "y": abs(med["y"]), "z": abs(med["z"]) * 0.5}  # zoom rate -> edge speed
    top = max(comp, key=comp.get)
    if top != "z" and comp["z"] >= 0.7 * comp[top]:
        top = "z"  # a scale change is unambiguous; zooming about an off-centre point also shifts the centre
    series_ = {"x": dx, "y": dy, "z": zm}[top]
    agree = float(np.mean(np.sign(series_) == np.sign(med[top])))
    # name a direction only when the sustained motion dominates its own jitter
    if (comp[top] >= (ZOOM_MIN * 0.5 if top == "z" else PAN_MIN) and agree >= 0.7
            and abs(med[top]) > 2.0 * sd[top]):
        if top == "x":
            kind = "pan_right" if med["x"] < 0 else "pan_left"  # content moves opposite the camera
        elif top == "y":
            kind = "tilt_up" if med["y"] > 0 else "tilt_down"
        else:
            kind = "zoom_in" if med["z"] > 0 else "zoom_out"
        conf = 0.85 * (1.0 - min(0.5, sd[top] / abs(med[top]))) if abs(med[top]) else 0.0
        return _result(shot, kind, conf, [], **stats)
    if jitter >= SHAKE_MIN:
        return _result(shot, "handheld", 0.6, [], **stats)
    if max(comp.values()) < PAN_MIN * 0.5:
        return _result(shot, "static", 0.8, [], **stats)
    return _nd(shot, "motion present but direction unreliable; camera motion not determined")
