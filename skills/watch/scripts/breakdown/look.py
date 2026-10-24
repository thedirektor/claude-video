"""Per-shot look: palette, brightness, contrast, saturation, temperature, letterbox.

Everything is measured on a handful of 64 px frames sampled inside the shot
(numpy k-means for the palette); letterbox bars are read at full resolution. Without OpenCV/numpy the shot's look is
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
BAR_EDGE_SKIP = 6  # rows / columns next to the picture left out of the flatness test
BAR_FLAT = 8  # luma spread allowed inside a bar (codec noise); a gradient spreads wider
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


def _frames(cv2, path: str, shot: dict) -> tuple[list, list]:
    """``(small BGR frames, [(row max, column max) of the full-resolution luma])``. The bars are
    read from the full-resolution profiles so that their size is not quantised by the 64 px work
    width."""
    cap = cv2.VideoCapture(path)
    try:
        start, end = float(shot.get("start", 0.0)), float(shot.get("end", 0.0))
        span = max(end - start, 0.0)
        out, prof = [], []
        for i in range(SAMPLES):
            t = start + span * (0.1 + 0.8 * i / (SAMPLES - 1))
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, img = cap.read()
            if not ok or img is None:
                continue
            h, w = img.shape[:2]
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            prof.append((g.max(axis=1), g.max(axis=0), g.min(axis=1), g.min(axis=0)))
            out.append(cv2.resize(img, (WORK_W, max(2, int(round(h * WORK_W / w)))), interpolation=cv2.INTER_AREA))
        return out, prof
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


def _bars(np, black):
    n = len(black)
    lo = 0
    while lo < n and black[lo]:
        lo += 1
    hi = 0
    while hi < n - lo and black[n - 1 - hi]:
        hi += 1
    return lo / n, hi / n


def _letterbox(np, profiles):
    """Bar sizes (fraction of height / width) black in every sampled frame, else None.
    ``profiles`` are the full-resolution (row max, column max, row min, column min) luma of each frame."""
    ps = [p for p in profiles if p[0].shape == profiles[0][0].shape and p[1].shape == profiles[0][1].shape]
    rows = np.all([(p[0] < BLACK_LUMA) for p in ps], axis=0)
    cols = np.all([(p[1] < BLACK_LUMA) for p in ps], axis=0)
    if rows.all() or cols.all():
        return None
    (tf, bf), (lf, rf) = _bars(np, rows), _bars(np, cols)
    if max(tf, bf, lf, rf) > BAR_MAX:
        return None
    n_r, n_c = len(rows), len(cols)

    def uniform(frac, rng, n, side):  # a bar is flat black, not the dark end of a gradient
        k = int(round(frac * n))
        if k == 0:
            return True
        k -= BAR_EDGE_SKIP  # codec ringing / chroma bleed next to the picture reaches ~19 luma
        if k <= 0:
            return True
        sl = slice(0, k) if side == 0 else slice(n - k, n)
        hi = max(float(p[rng][sl].max()) for p in ps)
        lo = min(float(p[rng + 2][sl].min()) for p in ps)
        return hi - lo <= BAR_FLAT
    tf, bf = (tf if uniform(tf, 0, n_r, 0) else 0.0), (bf if uniform(bf, 0, n_r, 1) else 0.0)
    lf, rf = (lf if uniform(lf, 1, n_c, 0) else 0.0), (rf if uniform(rf, 1, n_c, 1) else 0.0)
    out = {k: (round(v, 4) if v >= BAR_MIN else 0.0) for k, v in
           (("top", tf), ("bottom", bf), ("left", lf), ("right", rf))}
    return out if any(out.values()) else None


def _measure(cv2, np, path: str, shot: dict) -> dict:
    frames, profiles = _frames(cv2, path, shot)
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
        "letterbox": _letterbox(np, profiles),
        "evidence": _evidence(shot, conf),
        "notes": [],
    }
