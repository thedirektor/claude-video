"""Per-shot look: palette, brightness, contrast, saturation, temperature, letterbox.

Everything is measured on a handful of 64 px frames sampled inside the shot
(numpy k-means for the palette). Without OpenCV/numpy the shot's look is
``"not determined"`` with a note naming the dependency; nothing is guessed.
"""
from __future__ import annotations

NOT_DETERMINED = "not determined"
SAMPLES = 5
WORK_W = 64
PALETTE_K = 5
TEMP_DELTA = 0.06  # mean (R-B)/255 beyond which the shot reads warm / cool
BLACK_LUMA = 20  # 0-255 luma under which a row counts as a letterbox bar
BAR_MIN = 0.02  # a bar must cover at least this fraction of the height
BAR_MAX = 0.4  # more than this per side is a black frame, not a letterbox


def _import_cv():
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    return cv2, np


def _evidence(shot: dict, conf: float) -> dict:
    return {"t_start": round(float(shot.get("start", 0.0)), 6), "t_end": round(float(shot.get("end", 0.0)), 6),
            "method": "measured", "confidence": round(max(0.0, min(1.0, conf)), 3)}


def _nd(shot: dict, note: str) -> dict:
    return {"palette": NOT_DETERMINED, "brightness": NOT_DETERMINED, "contrast": NOT_DETERMINED,
            "saturation": NOT_DETERMINED, "temperature": NOT_DETERMINED, "letterbox": NOT_DETERMINED,
            "evidence": _evidence(shot, 0.0), "notes": [note]}


def shot_look(path: str, shot: dict) -> dict:
    try:
        cv2, np = _import_cv()
    except Exception:  # ImportError, or a broken binary wheel
        return _nd(shot, "opencv (cv2) / numpy not installed; look not determined")
    try:
        return _measure(cv2, np, str(path), shot)
    except Exception as exc:  # never raise out of a command
        return _nd(shot, f"look failed ({type(exc).__name__}); not determined")


def _frames(cv2, path: str, shot: dict) -> list:
    cap = cv2.VideoCapture(path)
    try:
        start, end = float(shot.get("start", 0.0)), float(shot.get("end", 0.0))
        span = max(end - start, 0.0)
        out = []
        for i in range(SAMPLES):
            t = start + span * (0.1 + 0.8 * i / (SAMPLES - 1))
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, img = cap.read()
            if not ok or img is None:
                continue
            h, w = img.shape[:2]
            out.append(cv2.resize(img, (WORK_W, max(2, int(round(h * WORK_W / w)))), interpolation=cv2.INTER_AREA))
        return out
    finally:
        cap.release()


def _srgb_to_lab(np, rgb):  # rgb float 0..1, shape (n, 3)
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], axis=1)


def _kmeans(np, lab, k: int):
    """Deterministic k-means in Lab: farthest-point init, 12 Lloyd iterations."""
    cent = [lab[0]]
    for _ in range(1, k):
        d = np.min(np.linalg.norm(lab[:, None, :] - np.array(cent)[None], axis=2), axis=1)
        if d.max() < 1.0:
            break
        cent.append(lab[int(d.argmax())])
    cent = np.array(cent)
    for _ in range(12):
        lbl = np.argmin(np.linalg.norm(lab[:, None, :] - cent[None], axis=2), axis=1)
        for j in range(len(cent)):
            sel = lab[lbl == j]
            if len(sel):
                cent[j] = sel.mean(axis=0)
    lbl = np.argmin(np.linalg.norm(lab[:, None, :] - cent[None], axis=2), axis=1)
    return cent, np.bincount(lbl, minlength=len(cent)) / len(lbl)



def _palette(np, rgb):  # rgb float 0..1 (n,3)
    lab = _srgb_to_lab(np, rgb)
    cent, share = _kmeans(np, lab, PALETTE_K)
    lbl = np.argmin(np.linalg.norm(lab[:, None, :] - cent[None], axis=2), axis=1)
    out = []
    for j in np.argsort(-share):
        if share[j] <= 0:
            continue
        mean_rgb = rgb[lbl == j].mean(axis=0) * 255
        out.append({"hex": "#%02x%02x%02x" % tuple(int(round(v)) for v in mean_rgb), "share": round(float(share[j]), 4)})
    return out[:PALETTE_K]


def _letterbox(np, grays):
    """{"top","bottom"} bar heights (fraction) shared by every sampled frame, else None."""
    h = grays[0].shape[0]
    black = np.all([(g.max(axis=1) < BLACK_LUMA) for g in grays if g.shape[0] == h], axis=0)
    top = 0
    while top < h and black[top]:
        top += 1
    bottom = 0
    while bottom < h - top and black[h - 1 - bottom]:
        bottom += 1
    tf, bf = top / h, bottom / h
    if top == h or tf > BAR_MAX or bf > BAR_MAX:
        return None
    if tf < BAR_MIN and bf < BAR_MIN:
        return None
    return {"top": round(tf, 4), "bottom": round(bf, 4)}


def _measure(cv2, np, path: str, shot: dict) -> dict:
    frames = _frames(cv2, path, shot)
    if not frames:
        return _nd(shot, "no frames could be decoded for this shot; look not determined")
    rgb = np.concatenate([cv2.cvtColor(f, cv2.COLOR_BGR2RGB).reshape(-1, 3) for f in frames]).astype(np.float64) / 255.0
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    luma = np.concatenate([g.reshape(-1) for g in grays]).astype(np.float64) / 255.0
    hsv = np.concatenate([cv2.cvtColor(f, cv2.COLOR_BGR2HSV).reshape(-1, 3) for f in frames])
    delta = float((rgb[:, 0] - rgb[:, 2]).mean())
    temp = "warm" if delta > TEMP_DELTA else "cool" if delta < -TEMP_DELTA else "neutral"
    conf = min(1.0, 0.5 + 0.1 * len(frames))
    return {
        "palette": _palette(np, rgb),
        "brightness": round(float(luma.mean()), 4),
        "contrast": round(float(luma.std()), 4),
        "saturation": round(float(hsv[:, 1].mean() / 255.0), 4),
        "temperature": temp,
        "letterbox": _letterbox(np, grays),
        "evidence": _evidence(shot, conf),
        "notes": [],
    }
