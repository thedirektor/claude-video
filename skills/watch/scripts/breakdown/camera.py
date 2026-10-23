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
        starts = np.linspace(lo, hi - gap, count)
        flows, texture = [], []
        for t0 in starts:
            a, b = _frame(cv2, cap, float(t0)), _frame(cv2, cap, float(t0 + gap))
            if a is None or b is None:
                continue
            texture.append(float(a.std()))
            f = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            h, w = a.shape
            ys, xs = np.mgrid[0:h, 0:w]
            xc, yc = xs - (w - 1) / 2.0, ys - (h - 1) / 2.0
            # only textured pixels carry flow information (aperture problem)
            gx, gy = cv2.Sobel(a, cv2.CV_32F, 1, 0), cv2.Sobel(a, cv2.CV_32F, 0, 1)
            mask = (np.abs(gx) + np.abs(gy)) > 40.0
            if int(mask.sum()) < 0.01 * mask.size:
                flows.append((0.0, 0.0, 0.0))
                continue
            fx, fy, mx, my = f[..., 0][mask], f[..., 1][mask], xc[mask], yc[mask]
            r2 = mx ** 2 + my ** 2
            far = r2 > 400.0  # near the centre the radial ratio is noise
            zoom = float(np.median((fx[far] * mx[far] + fy[far] * my[far]) / r2[far])) if far.any() else 0.0
            flows.append((float(np.median(fx)) / w / gap, float(np.median(fy)) / h / gap, zoom / gap))
    finally:
        cap.release()
    if len(flows) < 2:
        return _nd(shot, "too few readable frame pairs; camera motion not determined")
    dx = np.array([f[0] for f in flows])
    dy = np.array([f[1] for f in flows])
    zm = np.array([f[2] for f in flows])
    mdx, mdy, mzm = float(np.median(dx)), float(np.median(dy)), float(np.median(zm))
    jitter = float(max(dx.std(), dy.std(), zm.std()))
    flat = float(np.median(texture)) < 4.0
    base = 0.55 if flat else 0.8
    stats = {"pairs": len(flows), "pan_x_per_s": round(mdx, 4), "pan_y_per_s": round(mdy, 4),
             "zoom_per_s": round(mzm, 4), "jitter": round(jitter, 4)}
    notes = ["low-texture frames: low confidence"] if flat else []
    comp = {"x": abs(mdx), "y": abs(mdy), "z": abs(mzm) * 0.5}  # zoom rate -> edge speed
    top = max(comp, key=comp.get)
    med = {"x": mdx, "y": mdy, "z": mzm}[top]
    series_ = {"x": dx, "y": dy, "z": zm}[top]
    agree = float(np.mean(np.sign(series_) == np.sign(med)))
    if comp[top] >= (ZOOM_MIN * 0.5 if top == "z" else PAN_MIN) and agree >= 0.7:
        if top == "x":
            kind = "pan_right" if mdx < 0 else "pan_left"  # content moves opposite the camera
        elif top == "y":
            kind = "tilt_up" if mdy > 0 else "tilt_down"
        else:
            kind = "zoom_in" if mzm > 0 else "zoom_out"
        return _result(shot, kind, base + min(0.15, comp[top]), notes, **stats)
    if jitter >= SHAKE_MIN:
        return _result(shot, "handheld", base - 0.1, notes, **stats)
    return _result(shot, "static", base, notes, **stats)
