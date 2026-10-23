"""Per-shot camera motion from dense optical flow (OpenCV Farneback).

Median flow gives pan / tilt, the radial component gives zoom, and a flow that
changes sign between sampled pairs is handheld. Content moves opposite to the
camera, so content moving left means the camera panned right. Without OpenCV
every field is ``"not determined"`` with a note; nothing is guessed.
"""
from __future__ import annotations

NOT_DETERMINED = "not determined"
MAX_PAIRS = 8
WORK_W = 320
PAN_MIN = 0.04  # sustained translation, fraction of frame width per second
ZOOM_MIN = 0.04  # sustained radial expansion, fraction of frame per second
SHAKE_MIN = 0.03  # jitter amplitude (frame fraction per second) that reads as handheld
PAIR_GAP_S = 0.12


def _import_cv():
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    return cv2, np


def _result(shot: dict, kind: str, conf: float, notes: list[str], **extra) -> dict:
    ev = {"t_start": round(shot.get("start", 0.0), 6), "t_end": round(shot.get("end", 0.0), 6),
          "method": "measured", "confidence": round(max(0.0, min(1.0, conf)), 3)}
    ev.update(extra)
    return {"type": kind, "confidence": ev["confidence"], "evidence": ev, "notes": notes}


def _nd(shot: dict, note: str) -> dict:
    return _result(shot, NOT_DETERMINED, 0.0, [note])


def camera_motion(path: str, shot: dict) -> dict:
    try:
        cv2, np = _import_cv()
    except Exception:  # ImportError, or a broken binary wheel
        return _nd(shot, "opencv (cv2) is not installed; camera motion not determined")
    try:
        return _measure(cv2, np, str(path), shot)
    except Exception as exc:  # never raise out of a command
        return _nd(shot, f"camera motion failed ({type(exc).__name__}); not determined")


def _frame(cv2, cap, t: float):
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, img = cap.read()
    if not ok or img is None:
        return None
    h, w = img.shape[:2]
    img = cv2.resize(img, (WORK_W, max(2, int(h * WORK_W / w))), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _pair(cv2, np, a, b, gap: float):
    """(pan_x, pan_y, zoom) per second between two gray frames, or None.

    Tracks corner features and fits a similarity transform, so translation and
    scale are estimated separately (a pan never leaks into the zoom term).
    """
    pts = cv2.goodFeaturesToTrack(a, maxCorners=200, qualityLevel=0.01, minDistance=7)
    if pts is None or len(pts) < 12:
        return None
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None, winSize=(21, 21), maxLevel=3)
    ok = st.reshape(-1) == 1
    if int(ok.sum()) < 12:
        return None
    m, inl = cv2.estimateAffinePartial2D(pts[ok], nxt[ok], method=cv2.RANSAC, ransacReprojThreshold=1.5)
    if m is None or inl is None or int(inl.sum()) < 10 or int(inl.sum()) < 0.5 * int(ok.sum()):
        return None
    h, w = a.shape
    scale = float(np.hypot(m[0, 0], m[1, 0]))
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    ncx = m[0, 0] * cx + m[0, 1] * cy + m[0, 2]
    ncy = m[1, 0] * cx + m[1, 1] * cy + m[1, 2]
    return ((ncx - cx) / w / gap, (ncy - cy) / h / gap, (scale - 1.0) / gap)


def _measure(cv2, np, path: str, shot: dict) -> dict:
    start, end = float(shot["start"]), float(shot["end"])
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return _nd(shot, "video could not be opened; camera motion not determined")
    try:
        # stay inside the shot, away from the boundary frames (dissolves, flashes)
        margin = min(0.1, (end - start) * 0.1)
        lo, hi = start + margin, end - margin
        gap = min(PAIR_GAP_S, (hi - lo) / 2)
        if gap < 0.03:
            return _nd(shot, "shot too short to measure camera motion")
        count = min(MAX_PAIRS, max(1, int((hi - lo - gap) / gap)))
        flows, still, seen = [], 0, 0
        for t0 in np.linspace(lo, hi - gap, count):
            a, b = _frame(cv2, cap, float(t0)), _frame(cv2, cap, float(t0 + gap))
            if a is None or b is None:
                continue
            seen += 1
            if float(np.abs(a.astype(np.float32) - b).mean()) < 0.5:
                still += 1
                flows.append((0.0, 0.0, 0.0))
                continue
            f = _pair(cv2, np, a, b, gap)
            if f is not None:
                flows.append(f)
    finally:
        cap.release()
    if seen < 2:
        return _nd(shot, "too few readable frame pairs; camera motion not determined")
    if still == seen:  # identical frames: nothing moves, whatever the texture
        return _result(shot, "static", 0.8, [], pairs=seen)
    if len(flows) < 2 or len(flows) < 0.5 * seen:
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
