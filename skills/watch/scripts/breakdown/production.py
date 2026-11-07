"""Per-shot production method, measured: what can be told from pixels, and nothing more.

Labels (``label``):

* ``still``        - the picture does not change (a photo, a slide, a held frame).
* ``still_moved``  - a flat picture moved digitally as one rigid plane (push-in / pan on a photo, a
  scrolled screenshot): one similarity transform explains the whole frame, no parallax.
* ``synthetic``    - drawn or rendered smooth surfaces (code / template graphics, UI, slides, 3D):
  a large share of texture-free blocks that camera footage never has.
* ``ai_generated`` / ``camera`` - only when the FILE declares it (AIGC label / C2PA, camera tags;
  see ``provenance``): pixels after a platform re-encode cannot separate camera footage from AI video.
* ``"not determined"`` - textured footage with no declaration. ``candidates`` then ranks
  ``ai_video`` vs ``camera`` from the frame cadence: a source rate below the container rate
  (24 fps doubled into 30) is common in generated video, but film transfers do it too.

Measurements (always shipped): ``held_share`` (consecutive frames that repeat while the picture
moves), ``source_fps``, ``flat`` (share of mid-grey 8x8 blocks with no texture), ``rigid_residual``
(mean abs error after one similarity warp), ``still_diff``. Thresholds were set on labelled clips
(live phone footage, AI remakes, 3D explainer, code-UI promo, music video, talks); see
``tests/test_breakdown_production.py`` and the eval harness for how to re-measure them.
"""
from __future__ import annotations

NOT_DETERMINED = "not determined"
LABELS = ("still", "still_moved", "synthetic", "ai_generated", "camera")
WORK_W = 320
RUN_FRAMES = 24  # consecutive frames read for cadence (capped by the shot length)
DUP_DIFF = 0.6  # mean abs difference (0-255) under which two consecutive frames are the same frame
MOVE_DIFF = 1.5  # ... and above which the picture is moving
SPREAD_FRAMES = 5
STILL_DIFF = 0.8  # every spread frame within this of the first: nothing moves in the shot
FLAT_STD = 0.75  # an 8x8 block this flat has no grain and no texture
FLAT_LO, FLAT_HI = 8, 247  # crushed blacks / clipped whites are flat on camera too
CENTRE_SIDE = 512  # full-resolution centre crop for the flat-block measure
RIGID_RESIDUAL = 2.2  # mean abs residual (0-255) after one similarity warp: the picture moved as one plane
RIGID_MIN_MOTION = 1.0  # px of motion (work width) needed before "moved as one plane" means anything
RIGID_INLIERS = 0.85
FLAT_SYNTHETIC = 0.45  # labelled: 3D explainer 0.36-0.70, code UI up to 0.9 ...
WHITE_FILL = 0.35  # flat pure-white share of the frame: UI promo median 0.70, camera clips <= 0.05 (a huge blown-out sky could reach it)
FLAT_LEAN = 0.30  # ... but a denoised 4K music-video remaster reached 0.30-0.42: in between only leans
HELD_MIN = 0.15  # repeated-frame share that marks a lower source rate
CAP_UNDECLARED = 0.7  # pixel evidence alone never claims more than this


def _import_cv():
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    return cv2, np


def _evidence(shot: dict, conf: float, method: str = "measured") -> dict:
    return {"t_start": round(float(shot.get("start", 0.0)), 6), "t_end": round(float(shot.get("end", 0.0)), 6),
            "method": method, "confidence": round(max(0.0, min(1.0, conf)), 3)}


def _nd(shot: dict, note: str, **measurements) -> dict:
    return {"label": NOT_DETERMINED, "candidates": [], "measurements": measurements,
            "evidence": _evidence(shot, 0.0), "notes": [note] if note else []}


def shot_production(path: str, shot: dict, fps: float, provenance: dict | None = None) -> dict:
    try:
        cv2, np = _import_cv()
    except Exception:  # ImportError, or a broken binary wheel
        return _nd(shot, "opencv (cv2) / numpy not installed; production method not determined")
    try:
        m = measure(cv2, np, str(path), shot, fps)
    except Exception as exc:  # never raise out of a command
        return _nd(shot, f"production measurement failed ({type(exc).__name__}); not determined")
    return classify(shot, m, provenance)


# --------------------------------------------------------------------------- measurements

def _small(cv2, img):
    h, w = img.shape[:2]
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (WORK_W, max(2, int(round(h * WORK_W / w)))),
                      interpolation=cv2.INTER_AREA)


def _centre(cv2, img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    s = min(CENTRE_SIDE, h, w)
    y, x = (h - s) // 2, (w - s) // 2
    return g[y:y + s, x:x + s]


def flat_share(np, gray) -> float:
    h, w = (gray.shape[0] // 8) * 8, (gray.shape[1] // 8) * 8
    if h < 8 or w < 8:
        return 0.0
    b = gray[:h, :w].astype(np.float64).reshape(h // 8, 8, w // 8, 8).swapaxes(1, 2).reshape(-1, 64)
    mean, std = b.mean(axis=1), b.std(axis=1)
    mid = (mean > FLAT_LO) & (mean < FLAT_HI)
    if not mid.any():
        return 0.0
    return float(((std < FLAT_STD) & mid).sum() / max(1, mid.sum()))


def white_fill(np, gray) -> float:
    """Share of 8x8 blocks that are flat pure white (a drawn background; FLAT_HI leaves them out of ``flat``)."""
    h, w = (gray.shape[0] // 8) * 8, (gray.shape[1] // 8) * 8
    if h < 8 or w < 8:
        return 0.0
    b = gray[:h, :w].astype(np.float64).reshape(h // 8, 8, w // 8, 8).swapaxes(1, 2).reshape(-1, 64)
    return float(((b.mean(axis=1) >= FLAT_HI) & (b.std(axis=1) < FLAT_STD)).mean())


def _rigid(cv2, np, a, b) -> tuple[float, float] | None:
    """(mean abs residual after a similarity warp a->b, median feature motion px) or None (untrackable)."""
    pts = cv2.goodFeaturesToTrack(a, 200, 0.01, 6)
    if pts is None or len(pts) < 20:
        return None
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None)
    ok = st.reshape(-1) == 1
    if ok.sum() < 20:
        return None
    p0, p1 = pts[ok], nxt[ok]
    mat, inl = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=1.0)
    if mat is None or inl is None or inl.mean() < RIGID_INLIERS:
        return (99.0, float(np.median(np.linalg.norm((p1 - p0).reshape(-1, 2), axis=1))))
    warped = cv2.warpAffine(a, mat, (a.shape[1], a.shape[0]))
    mh, mw = a.shape[0] // 8, a.shape[1] // 8
    res = np.abs(warped.astype(np.float64) - b.astype(np.float64))[mh:-mh, mw:-mw]
    return float(res.mean()), float(np.median(np.linalg.norm((p1 - p0).reshape(-1, 2), axis=1)))


def measure(cv2, np, path: str, shot: dict, fps: float) -> dict:
    start, end = float(shot.get("start", 0.0)), float(shot.get("end", 0.0))
    span = max(end - start, 0.0)
    cap = cv2.VideoCapture(path)
    try:
        fps = fps or cap.get(cv2.CAP_PROP_FPS) or 0.0
        n_run = int(min(RUN_FRAMES, max(2, span * fps * 0.8))) if fps else 0
        run, centres = [], []
        mid = start + span / 2.0 - (n_run / fps / 2.0 if fps else 0.0)
        cap.set(cv2.CAP_PROP_POS_MSEC, max(start, mid) * 1000.0)
        for _ in range(n_run):
            ok, img = cap.read()
            if not ok or img is None:
                break
            run.append(_small(cv2, img))
            if len(centres) < 3 and len(run) in (1, n_run // 2 + 1, n_run):
                centres.append(_centre(cv2, img))
        spread = []
        for i in range(SPREAD_FRAMES):
            t = start + span * (0.08 + 0.84 * i / (SPREAD_FRAMES - 1))
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, img = cap.read()
            if ok and img is not None:
                spread.append(_small(cv2, img))
                if len(centres) < 6:
                    centres.append(_centre(cv2, img))
    finally:
        cap.release()
    if len(run) < 2 and len(spread) < 2:
        return {"frames": 0}
    diffs = [float(np.abs(run[i + 1].astype(np.int16) - run[i].astype(np.int16)).mean())
             for i in range(len(run) - 1)]
    moving = [d for d in diffs if d >= MOVE_DIFF]
    dup = sum(1 for d in diffs if d < DUP_DIFF)
    held = dup / len(diffs) if diffs and len(moving) >= max(2, len(diffs) // 4) else 0.0
    still_d = max((float(np.abs(s.astype(np.int16) - spread[0].astype(np.int16)).mean()) for s in spread[1:]),
                  default=0.0)
    rig = []
    n = len(run)
    # pairs spread over the run: a slow digital push moves < 1 px between neighbouring frames
    for i, j in {(0, n // 2), (n // 2, n - 1), (0, n - 1), (n // 4, (3 * n) // 4)}:
        if 0 <= i < j < n:
            r = _rigid(cv2, np, run[i], run[j])
            if r is not None:
                rig.append(r)
    moved = [r for r in rig if r[1] >= RIGID_MIN_MOTION]
    return {
        "frames": len(run) + len(spread),
        "held_share": round(held, 3),
        "source_fps": round(fps * (1.0 - held), 2) if held >= HELD_MIN and fps else None,
        "motion": round(float(np.median(diffs)), 3) if diffs else 0.0,
        "still_diff": round(still_d, 3),
        "flat": round(float(np.median([flat_share(np, c) for c in centres])), 3) if centres else None,
        "white_fill": round(float(np.median([white_fill(np, c) for c in centres])), 3) if centres else None,
        "rigid_residual": round(float(np.median([r[0] for r in moved])), 3) if moved else None,
        "rigid_pairs": len(moved),
    }


# --------------------------------------------------------------------------- classification

def classify(shot: dict, m: dict, provenance: dict | None = None) -> dict:
    """Label from the measurements (pure: unit-testable without decoding)."""
    if not m.get("frames"):
        return _nd(shot, "no frames could be decoded for this shot; production method not determined", **m)
    notes: list[str] = []
    flat = m.get("flat")
    declared = (provenance or {}).get("declared")
    still = m.get("still_diff", 99.0) < STILL_DIFF
    rigid = (m.get("rigid_residual") is not None and m["rigid_residual"] < RIGID_RESIDUAL
             and m.get("rigid_pairs", 0) >= 2)
    white = m.get("white_fill") or 0.0
    smooth = (flat is not None and flat >= FLAT_SYNTHETIC) or white >= WHITE_FILL
    held = bool(m.get("source_fps"))
    if held:
        notes.append(f"{m['held_share']:.0%} of consecutive frames repeat while the picture moves: source rate "
                     f"about {m['source_fps']:g} fps")
    if still:
        return _out(shot, m, "still", 0.85, [("still", 0.85)], notes)
    if rigid:
        notes.append("the whole picture moves as one rigid plane (digital push / pan / scroll, no parallax)")
        cands = [("still_moved", 0.7)] + ([("synthetic", 0.5)] if flat is not None and flat >= FLAT_LEAN else [])
        return _out(shot, m, "still_moved", 0.7, cands, notes)
    if smooth:
        if white >= WHITE_FILL:
            notes.append(f"{white:.0%} of the picture is a flat pure-white fill: a drawn background (UI, slide, "
                         "card); a very large blown-out sky would read the same")
        else:
            notes.append(f"{flat:.0%} of the picture is texture-free: drawn / rendered surfaces (graphics, UI, "
                         "slides or 3D), not camera footage")
        cands = [("synthetic", min(CAP_UNDECLARED, 0.35 + max(flat or 0.0, white)))]
        if declared == "ai_generated":
            cands.append(("ai_generated", 0.6))
        return _out(shot, m, "synthetic", cands[0][1], cands, notes)
    # textured, moving footage: pixels cannot separate camera from AI video
    if declared == "ai_generated":
        notes.append("the file declares AI-generated content (AIGC label / C2PA)")
        return _out(shot, m, "ai_generated", 0.9, [("ai_generated", 0.9)], notes)
    if declared == "camera":
        notes.append("camera make / model tags in the file")
        return _out(shot, m, "camera", 0.8, [("camera", 0.8)], notes)
    if flat is not None and flat >= FLAT_LEAN:
        notes.append(f"{flat:.0%} of the picture is texture-free: rendered / drawn surfaces or heavily denoised "
                     "footage; not enough to tell")
        cands = [("synthetic", 0.4), ("ai_video", 0.3), ("camera", 0.3)]
        return _out(shot, m, NOT_DETERMINED, 0.0, cands, notes)
    cands = [("ai_video", 0.45), ("camera", 0.3)] if held else [("camera", 0.4), ("ai_video", 0.35)]
    notes.append("textured footage with no provenance in the file: camera or AI video cannot be told apart "
                 "from pixels" + ("; the lowered source rate leans to generated video (film transfers do "
                                  "this too)" if held else ""))
    return _out(shot, m, NOT_DETERMINED, 0.0, cands, notes)


def _out(shot: dict, m: dict, label: str, conf: float, cands: list, notes: list[str]) -> dict:
    return {"label": label, "candidates": [{"label": lab, "confidence": round(c, 3)} for lab, c in cands],
            "measurements": m, "evidence": _evidence(shot, conf), "notes": notes}
