"""Frame-accurate shot detection and rhythm stats.

``frame_series`` makes one ffmpeg pass (160 px, every frame) and records the
scene score and mean luma per frame. ``detect_shots`` turns that series into
shots: a hard cut is a single-frame score spike, a gradual boundary is a run of
frames whose end frames differ a lot (tiny colour thumbnails) while the content on both sides of the window is stable; a flash frame is not a cut. Boundaries are frame
accurate because they come from per-frame timestamps, never from sampling.
"""
from __future__ import annotations

import math
import re
import statistics
import subprocess
import time

NOT_DETERMINED = "not determined"
CUT_FLOOR = 0.15  # brief says 0.3; ffmpeg's scene score for a real red->green cut is only 0.23
THUMB_SCALE = 2.5  # a thumbnail step of this many x the cut threshold counts as a full-strength cut
LUMA_SCALE = 60.0  # a luma jump of this many levels counts as a full-strength cut
CUT_K = 6.0
GRADUAL_MIN_FRAMES = 3

CROP_MIN_GAIN = 0.9  # crop only when the active picture is below this share of the frame
_CROP = re.compile(r"crop=(\d+):(\d+):(\d+):(\d+)")

_PTS = re.compile(r"pts_time:([0-9.]+)")
_SCENE = re.compile(r"lavfi\.scene_score=([0-9.eE+-]+)")
_YAVG = re.compile(r"lavfi\.signalstats\.YAVG=([0-9.eE+-]+)")


_PASSTHROUGH: list[str] | None = None


def _passthrough() -> list[str]:
    """One output frame per input frame: ``-fps_mode passthrough`` (ffmpeg >= 5.1), else the older
    ``-vsync 0`` that newer builds deprecate."""
    global _PASSTHROUGH
    if _PASSTHROUGH is None:
        try:
            out = subprocess.run(["ffmpeg", "-hide_banner", "-h", "full"], capture_output=True, text=True,
                                 timeout=60).stdout
        except (OSError, subprocess.SubprocessError):
            out = ""
        _PASSTHROUGH = ["-fps_mode", "passthrough"] if "-fps_mode" in out else ["-vsync", "0"]
    return _PASSTHROUGH


def _empty() -> dict:
    return {"t": [], "scene": [], "yavg": [], "thumb": []}


def active_crop(path: str) -> str | None:
    """ffmpeg ``crop=w:h:x:y`` for the active picture when black bars take a real share of the
    frame (a phone recording of a 16:9 video, a pillarboxed 4:3 film), else None. Black bars dilute
    every frame-difference measure, so cuts in a small picture would read as noise. Keyframes only:
    cheap, and ``reset=0`` keeps the union of everything that was ever lit."""
    def detect(args: list[str]) -> list:
        res = subprocess.run(["ffmpeg", "-hide_banner", *args], capture_output=True, text=True, timeout=900)
        return _CROP.findall(res.stderr)

    try:
        found = detect(["-skip_frame", "nokey", "-i", str(path), "-an",
                        "-vf", "cropdetect=limit=24:round=2:reset=0:skip=0", "-f", "null", "-"])
        if len(found) < 3:  # few keyframes (short clip): sample decoded frames instead
            found = detect(["-i", str(path), "-an", "-vf", "fps=2,cropdetect=limit=24:round=2:reset=0:skip=0",
                            "-f", "null", "-"])
        dims = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
             "-of", "csv=p=0", str(path)], capture_output=True, text=True, timeout=60).stdout.strip()
        fw, fh = (int(x) for x in dims.split(",")[:2])
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if not found or fw <= 0 or fh <= 0:
        return None
    w, h, x, y = (int(v) for v in found[-1])
    if w <= 16 or h <= 16 or w * h >= CROP_MIN_GAIN * fw * fh:
        return None
    return f"crop={w}:{h}:{x}:{y}"


def frame_series(path: str, fps: float) -> dict:
    """Per-frame ``{"t", "scene", "yavg", "thumb", "crop"}`` over the active picture (black bars cropped).

    Pass one measures scene score and mean luma per frame; a second pass decodes a 16x9 RGB thumbnail
    per frame (``bytes``, 432 per frame). Empty when fps is unknown (0), when ffmpeg fails (a
    truncated series would silently stretch the last shot to the media duration) or when the passes
    disagree on the frame count (no thumbnails: gradual boundaries and transitions are then not
    determined)."""
    if not fps or fps <= 0:
        return _empty()
    crop = active_crop(path)
    pre = f"{crop}," if crop else ""
    vf = pre + "scale=160:-2,signalstats,select='gte(scene,0)',metadata=print:file=-"
    try:
        res = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-an",
             "-vf", vf, *_passthrough(), "-f", "null", "-"],
            capture_output=True, text=True, timeout=1800)
    except (OSError, subprocess.SubprocessError):
        return _empty()
    if res.returncode != 0:  # decode error / killed: a partial series is worse than none
        return _empty()
    series = _empty()
    cur: dict = {}

    def flush() -> None:
        if "t" in cur:
            series["t"].append(cur["t"])
            series["scene"].append(cur.get("scene", 0.0))
            series["yavg"].append(cur.get("yavg", 0.0))

    for line in res.stdout.splitlines():
        if line.startswith("frame:"):
            flush()
            cur = {}
            m = _PTS.search(line)
            if m:
                cur["t"] = float(m.group(1))
        elif (m := _SCENE.search(line)):
            cur["scene"] = float(m.group(1))
        elif (m := _YAVG.search(line)):
            cur["yavg"] = float(m.group(1))
    flush()
    series["thumb"] = _thumbs(path, len(series["t"]), pre)
    series["crop"] = crop
    series["path"] = str(path)
    return series


def _thumbs(path: str, count: int, pre: str = "") -> list[bytes]:
    """Tiny RGB thumbnail per frame (second pass), as ``bytes`` slices (a third of the memory of int
    lists: ~370 MB/h at 30 fps otherwise); [] when the pass fails or the count disagrees."""
    size = THUMB_W * THUMB_H * 3
    try:
        res = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-an", "-vf",
             f"{pre}scale={THUMB_W}:{THUMB_H}:flags=area,format=rgb24", *_passthrough(), "-f", "rawvideo", "-"],
            capture_output=True, timeout=1800)
    except (OSError, subprocess.SubprocessError):
        return []
    raw = res.stdout
    if res.returncode != 0 or len(raw) != size * count:
        return []
    return [raw[i * size:(i + 1) * size] for i in range(count)]


def _cut_threshold(scene: list[float]) -> float:
    med = statistics.median(scene)
    mad = statistics.median([abs(s - med) for s in scene])
    return max(CUT_FLOOR, med + CUT_K * mad)


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


THUMB_W, THUMB_H = 16, 9
MAX_GRADUAL_S = 2.0
MIN_CROSS = 0.08  # content change across a gradual window (0..1 mean abs thumbnail diff)
MIN_SCORE = 3.0  # cross-window change vs the shots' own frame-to-frame variation
FLASH_FRAMES = 3
DARK = 40  # near-black thumbnail (mean channel value)
WHITE_MEAN = 215  # near-white thumbnail (mean channel value)
_LENGTHS = (4, 6, 9, 13, 19, 27, 38, 50)


def _dist(a, b) -> float:
    return sum(abs(x - y) for x, y in zip(a, b)) / (255.0 * len(a))


FLAT_STD = 10.0  # thumbnail spread below this is a blank, single-colour frame


def _flat_extreme(th) -> bool:
    """Near-black, near-white or single-colour thumbnail (the end of a fade, not content)."""
    m = sum(th) / len(th)
    if m < DARK or m > WHITE_MEAN:
        return True
    # per colour channel: a saturated single colour (red = 255, 0, 0 per pixel) is flat, not spread
    return max(statistics.pstdev(th[c::3]) for c in range(3)) < FLAT_STD


def _fps_of(t: list[float]) -> float:
    dts = sorted(t[i] - t[i - 1] for i in range(1, len(t)) if t[i] > t[i - 1])
    return 1.0 / dts[len(dts) // 2] if dts else 0.0


def _flash_cuts(cands: list[int], strength: list[float], series: dict) -> set[int]:
    """Cut pairs a few frames apart where the content returns (flash frames)."""
    thumb, yavg = series.get("thumb") or [], series["yavg"]
    drop: set[int] = set()
    for i in cands:
        for j in cands:
            if not (i < j <= i + FLASH_FRAMES) or i in drop:
                continue
            if thumb:
                before, after = _dist(thumb[i - 1], thumb[i]), _dist(thumb[i - 1], thumb[j])
            else:
                before, after = abs(yavg[i] - yavg[i - 1]), abs(yavg[j] - yavg[i - 1])
            if after < 0.3 * before:
                drop.update((i, j))
    return drop


HARD_STEP = 0.5  # share of a window's change that one frame may carry before it is a cut
HARD_STEP_CONF = 0.7
REFINE_ALPHA = 0.03  # a frame this close to either end frame (share of the blend) is still outside the dissolve
REFINE_CONTEXT = 3  # stable frames on each side of the window used as the pure end frames


def _refine_window(thumb: list, i0: int, i1: int) -> tuple[int, int]:
    """Tighten a quantised dissolve window onto the blend ramp.

    Projects every frame near the window onto the line between the (pure) frames just outside
    it: the dissolve starts where that share first leaves ~0 and ends where it reaches ~1.
    Falls back to the window itself when the ends are indistinguishable or the ramp is not
    monotone enough to read."""
    n = len(thumb)
    a, b = max(0, i0 - REFINE_CONTEXT), min(n - 1, i1 + REFINE_CONTEXT)
    ta, tb = thumb[a], thumb[b]
    d = [y - x for x, y in zip(ta, tb)]
    dd = sum(v * v for v in d)
    if dd == 0:
        return i0, i1
    alpha = [max(0.0, min(1.0, sum((fv - x) * dv for fv, x, dv in zip(thumb[k], ta, d)) / dd))
             for k in range(a, b + 1)]
    first = next((k for k, al in enumerate(alpha) if al > REFINE_ALPHA), None)
    last = next((k for k in range(len(alpha) - 1, -1, -1) if alpha[k] < 1 - REFINE_ALPHA), None)
    if first is None or last is None:
        return i0, i1
    r0, r1 = a + max(0, first - 1), a + min(len(alpha) - 1, last + 1)
    if r1 - r0 < GRADUAL_MIN_FRAMES:
        return i0, i1
    return r0, r1


SHIFT_MAX = 9  # cap on the frame-to-frame translation tried, in thumbnail columns: past ~60% of the
# width two frames no longer overlap enough to compare (faster moves: see shots._collapse_bursts)
SHIFT_TRAVEL = 1.5  # a move across the window cannot exceed this many frame widths: bounds the per-step shift
SHIFT_IMPROVE = 0.25  # a shift that explains a frame step this much better than no shift = motion
SHIFT_MEDIAN = 0.4  # median improvement above which the in-between frames are a translating picture
SHIFT_SHARE = 0.75  # share of frame steps that must improve for the same


def _gray(th) -> list[list[float]]:
    w = THUMB_W
    return [[(th[(r * w + x) * 3] + th[(r * w + x) * 3 + 1] + th[(r * w + x) * 3 + 2]) / 3.0
             for x in range(w)] for r in range(THUMB_H)]


def _step_error(a, b, dx: float, dy: int) -> float | None:
    """Mean abs difference of ``b`` against ``a`` shifted by (dx cols, dy rows); overlap only."""
    tot, cnt = 0.0, 0
    for r in range(THUMB_H):
        r2 = r - dy
        if not 0 <= r2 < THUMB_H:
            continue
        for x in range(THUMB_W):
            xs = x - dx
            if xs < 0 or xs > THUMB_W - 1:
                continue
            x0 = int(xs)
            fx = xs - x0
            x1 = min(THUMB_W - 1, x0 + 1)
            tot += abs(b[r][x] - (a[r2][x0] * (1 - fx) + a[r2][x1] * fx))
            cnt += 1
    return tot / cnt / 255.0 if cnt >= 0.4 * THUMB_W * THUMB_H else None


def translating(thumb: list, i0: int, i1: int) -> bool:
    """True when the frames of a window are one picture sliding: each step is explained far better
    by a small shift than by no shift. A dissolve only cross-fades in place, so shifting never helps.

    A pan over a soft scene also fits a straight blend (residual tiny), which is why the blend test
    alone would call a whip pan between held framings a confident dissolve."""
    g = [_gray(thumb[k]) for k in range(i0, i1 + 1)]
    # the range scales with the window: a short window has to cover the same ground in fewer steps
    reach = min(SHIFT_MAX, max(2, math.ceil(SHIFT_TRAVEL * THUMB_W / max(1, len(g) - 1))))
    gains = []
    for k in range(len(g) - 1):
        base = _step_error(g[k], g[k + 1], 0.0, 0)
        if not base or base < 1e-6:
            gains.append(0.0)
            continue
        best, at = base, (0.0, 0)
        for dy in (-1, 0, 1):  # coarse: whole columns
            for dx in range(-reach, reach + 1):
                if dx == 0 and dy == 0:
                    continue
                e = _step_error(g[k], g[k + 1], float(dx), dy)
                if e is not None and e < best:
                    best, at = e, (float(dx), dy)
        if at != (0.0, 0):  # fine: quarter columns around the best whole shift
            for step in (-0.75, -0.5, -0.25, 0.25, 0.5, 0.75):
                e = _step_error(g[k], g[k + 1], at[0] + step, at[1])
                if e is not None and e < best:
                    best = e
        gains.append(1.0 - best / base)
    if not gains:
        return False
    gains_sorted = sorted(gains)
    median = gains_sorted[len(gains_sorted) // 2]
    share = sum(1 for v in gains if v > SHIFT_IMPROVE) / len(gains)
    return median >= SHIFT_MEDIAN and share >= SHIFT_SHARE



def blend_residual(thumb: list, i0: int, i1: int) -> float | None:
    """Mean residual of in-between frames against the end-frame blend."""
    a, b = thumb[i0], thumb[i1]
    d = [y - x for x, y in zip(a, b)]
    dd = sum(v * v for v in d)
    if dd == 0 or i1 - i0 < 2:
        return None
    res = []
    for i in range(i0 + 1, i1):
        f = thumb[i]
        alpha = max(0.0, min(1.0, sum((fv - x) * dv for fv, x, dv in zip(f, a, d)) / dd))
        res.append(sum(abs(fv - (x + alpha * dv)) for fv, x, dv in zip(f, a, d)) / (255.0 * len(f)))
    return sum(res) / len(res)



PAN_MIN_PX = 1.2  # frame-to-frame translation (of a 160 px wide frame) that counts as moving
PAN_SHARE = 0.8  # share of the frame steps that must be the same slide
PAN_TOL = 0.35  # a step agrees with the slide within this share of its speed ...
PAN_TOL_PX = 1.5  # ... or this many px, whichever is larger
PAN_CONTEXT = 3  # steps either side of a cut that must slide with it
PAN_MIN_STEPS = 4


def _small_gray(cv2, np, img, crop, colour: bool = False):
    """160 px wide grey frame (float32); with ``colour`` also the 160 px colour frame."""
    h, w = img.shape[:2]
    m = re.fullmatch(r"crop=(\d+):(\d+):(\d+):(\d+)", crop or "")
    if m:
        cw, ch, cx, cy = (int(v) for v in m.groups())
        if cx + cw <= w and cy + ch <= h:
            img, h, w = img[cy:cy + ch, cx:cx + cw], ch, cw
    img = cv2.resize(img, (160, max(2, int(h * 160 / w))), interpolation=cv2.INTER_AREA)
    gray = np.float32(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return (gray, np.float32(img)) if colour else gray


PAN_OUTLIER = 3.0  # a step this many tolerances off the slide is a break in it, wherever it falls
PAN_BREAK_RATIO = 3.0  # a step whose slide-compensated residual exceeds this many x the median ...
PAN_BREAK_FLOOR = 0.02  # ... plus this (0..1 grey) is a picture the slide does not explain: a cut
SLIDE_SPAN_FRAC = 0.4  # residual spans cover at most this share of the frame width of travel
SLIDE_SPAN_MAX = 8
SLIDE_RATIO = 2.0  # a window changing this much more than the slide around it (after compensation)
SLIDE_FLOOR = 0.01  # ... plus this much (0..1 mean abs grey level) is a picture change, not the pan


class _PanReader:
    """One ``cv2.VideoCapture`` for every pan check of one ``detect_shots`` call (opening a capture per
    cut cost a container probe each time)."""

    def __init__(self, path: str):
        self.path, self.cap, self.failed = path, None, False

    def open(self, cv2):
        if self.cap is None and not self.failed:
            self.cap = cv2.VideoCapture(str(self.path))
            if not self.cap.isOpened():
                self.cap.release()
                self.cap, self.failed = None, True
        return self.cap

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def _pan_frames(cv2, np, cap, t: list[float], first: int, last: int, crop, colour: bool = False) -> list:
    """Grey 160 px frames ``first..last`` (``(grey, colour)`` pairs with ``colour``): one seek, then a
    sequential decode (shorter if it stops)."""
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t[first] - 0.5 / fps) * 1000.0)
    grays = []
    for _ in range(last - first + 1):
        ok, img = cap.read()
        if not ok or img is None:
            break
        grays.append(_small_gray(cv2, np, img, crop, colour))
    return grays


def _shifts_of(cv2, grays: list) -> list[tuple[float, float, float]]:
    """Phase-correlation ``(dx, dy, response)`` of each consecutive pair."""
    if len(grays) < 2:
        return []
    win = cv2.createHanningWindow(grays[0].shape[::-1], cv2.CV_32F)
    out = []
    for a, b in zip(grays, grays[1:]):
        # phaseCorrelate multiplies its inputs by the window in place: without copies every frame is
        # windowed twice (once as b, once as a) and the frames are no longer the picture afterwards
        (dx, dy), resp = cv2.phaseCorrelate(a.copy(), b.copy(), win)
        out.append((dx, dy, resp))
    return out


def _with_capture(series: dict, fn):
    """Run ``fn(cv2, np, cap)`` on the media; None when OpenCV, the path or the media is missing."""
    path = series.get("path")
    if not path:
        return None
    try:
        import cv2  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
    except Exception:
        return None
    reader = series.get("_pan_reader")
    if reader is not None:
        cap = reader.open(cv2)
        return None if cap is None else fn(cv2, np, cap)
    cap = cv2.VideoCapture(str(path))
    try:
        return fn(cv2, np, cap) if cap.isOpened() else None
    finally:
        cap.release()


class _Unclear:
    """Falsy verdict of ``continuous_pan``: the picture is moving through the window, but its shifts do
    not agree well enough to call it one slide (a pan over featureless detail, a move that changes)."""

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "UNCLEAR"


UNCLEAR = _Unclear()
PAN_UNCLEAR_NOTE = "picture moving through the window; one continuous pan could not be ruled out"


def _slide(steps: list) -> tuple | None:
    """(axis, speed, drift, tol) of a run of steps, or None when the picture is not moving."""
    axis = 0 if sum(abs(s[0]) for s in steps) >= sum(abs(s[1]) for s in steps) else 1
    speed = statistics.median(s[axis] for s in steps)
    drift = statistics.median(s[1 - axis] for s in steps)
    if abs(speed) < PAN_MIN_PX:
        return None
    return axis, speed, drift, max(PAN_TOL_PX, PAN_TOL * abs(speed))


def _residual(cv2, np, a, b, dx: float, dy: float) -> float | None:
    """Mean abs grey difference (0..1) of ``b`` against ``a`` moved by (dx, dy), on their overlap."""
    h, w = a.shape[:2]
    moved = cv2.warpAffine(a, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h))
    x0, x1 = int(max(0.0, dx)) + 1, int(min(w, w + dx)) - 1
    y0, y1 = int(max(0.0, dy)) + 1, int(min(h, h + dy)) - 1
    if x1 - x0 < 0.3 * w or y1 - y0 < 0.3 * h:
        return None
    return float(np.mean(np.abs(moved[y0:y1, x0:x1] - b[y0:y1, x0:x1]))) / 255.0


def continuous_pan(series: dict, i_before: int, i_after: int, colour: bool = True):
    """True when every frame step from ``i_before`` to ``i_after`` (clamped to the clip) is one picture
    sliding the same way at about the same speed: one continuous camera move, not an edit. A whip
    between held framings starts and ends at rest (False). A step whose measured shift is off the
    slide is re-checked by moving its first frame by the slide: a repeating texture can fool phase
    correlation, but the slide still explains the picture; a real cut, even between two pans the same
    way, does not. Any step that stays more than PAN_OUTLIER tolerances off breaks the slide wherever
    it falls, so the verdict does not depend on the seek landing on the exact frame. None when the
    check could not run (no OpenCV / media path / unreadable media); ``UNCLEAR`` (falsy) when the
    picture moves but is not one unbroken slide: callers keep the boundary and cap a gradual one."""
    t = series["t"]
    lo, hi = max(1, i_before), min(len(t) - 1, i_after)
    if hi - lo + 1 < PAN_MIN_STEPS:
        return False  # too little picture around the window to show a move

    def run(cv2, np, cap):
        pairs = _pan_frames(cv2, np, cap, t, lo - 1, hi, series.get("crop"), colour=True)
        grays, cols = [g for g, _ in pairs], [c for _, c in pairs]
        steps = _shifts_of(cv2, grays)
        if len(steps) < PAN_MIN_STEPS:
            return False
        # the agreement of the shifts is the evidence, not the peak height: a fast pan over smooth
        # detail gives weak peaks (0.02-0.08) that still land on the same shift step after step
        sl = _slide(steps)
        if sl is None:
            return False
        axis, speed, drift, tol = sl
        off = [max(abs(s[axis] - speed), abs(s[1 - axis] - drift)) for s in steps]
        dx, dy = (speed, drift) if axis == 0 else (drift, speed)
        # in colour: a cut to a regraded / hue-rotated copy keeps the grey picture and the shift
        frames = cols if colour else grays  # grey: does the picture's structure carry on (colour aside)?
        res = [_residual(cv2, np, frames[j], frames[j + 1], dx, dy) for j in range(len(steps))]
        known = [r for r in res if r is not None]
        if known and any(r is not None and r > PAN_BREAK_RATIO * statistics.median(known) + PAN_BREAK_FLOOR
                         for r in res):
            return UNCLEAR  # one step the slide cannot explain, wherever the seek put it: an edit
        ok = [r for r, o in zip(res, off) if r is not None and o <= tol]
        if ok:  # a wild shift the slide still explains: a repeating texture fooled the correlation
            # (only gross misreads are re-checked: a speed ramp or a stop stays a disagreement, since
            # on soft detail a wrong compensation leaves little residual too)
            ref = SLIDE_RATIO * statistics.median(ok) + SLIDE_FLOOR
            off = [0.0 if (o > PAN_OUTLIER * tol and r is not None and r <= ref) else o for o, r in zip(off, res)]
        if sum(o <= tol for o in off) < PAN_SHARE * len(off) or max(off) > PAN_OUTLIER * tol:
            return UNCLEAR  # moving, but not one unbroken slide
        return True

    return _with_capture(series, run)


def _span_residuals(cv2, np, grays: list, steps: list, speed: float):
    """(m, residuals): frames ``m`` apart (as far as the overlap allows, at most SLIDE_SPAN_MAX)
    compared after moving the first by the measured travel between them; None where no overlap."""
    w = grays[0].shape[1]
    m = max(1, min(SLIDE_SPAN_MAX, int(SLIDE_SPAN_FRAC * w / abs(speed))))
    res = []
    for j in range(len(grays) - m):
        dx = sum(s[0] for s in steps[j:j + m])
        dy = sum(s[1] for s in steps[j:j + m])
        res.append(_residual(cv2, np, grays[j], grays[j + m], dx, dy))
    return m, res


def slide_explains_window(series: dict, r0: int, r1: int):
    """For a gradual window that reads as one pan step by step: does the slide alone explain how the
    picture changes across it? Both layers of a dissolve between two pans the same way slide by the
    same step, so the per-step shifts cannot tell; the picture after compensating the slide can.
    True when the window's residual stays within SLIDE_RATIO x the residual of the slide around it;
    UNCLEAR when it does not or there is no slide context to compare with; None when it cannot run."""
    t = series["t"]
    n = len(t)

    def run(cv2, np, cap):
        first = max(0, r0 - SLIDE_SPAN_MAX - 3)
        last = min(n - 1, r1 + SLIDE_SPAN_MAX + 3)
        grays = _pan_frames(cv2, np, cap, t, first, last, series.get("crop"))
        steps = _shifts_of(cv2, grays)
        sl = _slide(steps) if steps else None
        if sl is None:
            return UNCLEAR
        m, res = _span_residuals(cv2, np, grays, steps, sl[1])
        inside = [r for j, r in enumerate(res) if r is not None and r0 <= first + j and first + j + m <= r1]
        outside = [r for j, r in enumerate(res) if r is not None and (first + j + m <= r0 or first + j >= r1)]
        if not inside or not outside:
            return UNCLEAR
        return statistics.mean(inside) <= SLIDE_RATIO * statistics.median(outside) + SLIDE_FLOOR or UNCLEAR

    return _with_capture(series, run)


DISSOLVE_MIN_RUN_S = 1.5  # a moving stretch this long (no cut, picture changing every frame) is scanned
DISSOLVE_GAP_S = 2.5  # quiet stretch (no thumbnail change) bridged between two moving runs
DISSOLVE_MOVING_SHARE = 0.35  # share of moving frames a stretch needs to be scanned
DISSOLVE_SCAN_BUDGET_S = 30.0  # wall-clock seconds the moving-dissolve scan may spend per media file
DISSOLVE_SCAN_CAPPED_NOTE = ("moving-dissolve scan budget reached: some moving stretches were not scanned, "
                             "dissolves between moving shots there can go undetected")
DISSOLVE_SCAN_MAX = 750  # frames decoded per scan chunk
DISSOLVE_K = 6.0  # spans whose slide residual exceeds this many x the stretch's median ...
DISSOLVE_MIN = 0.012  # ... and this level (0..1 grey) carry a picture change the slide does not explain
DISSOLVE_MIN_SPANS = 8  # consecutive such spans: a dissolve, not a glitch


def _moving_dissolves(series: dict, cut_idx: set[int], taken: list[tuple[int, int]]) -> list[dict]:
    """Dissolves between two moving shots, which the thumbnail test cannot see (the picture changes on
    every frame anyway): in a steady slide, frames compared after compensating the slide differ only
    by codec noise, while across a dissolve the incoming picture fades up and leaves a residual that
    rises for the whole cross-fade. Reported capped at UNCHECKED_CONF with PAN_UNCLEAR_NOTE: the
    window is measured, what happens inside it is not confirmed. Needs OpenCV and the media path."""
    t, th = series["t"], series.get("thumb") or []
    n = len(t)
    if len(th) != n or n < 8 or not series.get("path"):
        return []
    fps = _fps_of(t)
    d = _steps(series)
    # moving stretches: runs of changing thumbnails, merged across quiet gaps up to DISSOLVE_GAP_S
    # (a dissolve over featureless detail barely changes the thumbnails) and split at every cut
    active, k = [], 1
    while k < n:
        if d[k] >= BURST_ACTIVE and k not in cut_idx:
            s0 = k
            while k + 1 < n and d[k + 1] >= BURST_ACTIVE and k + 1 not in cut_idx:
                k += 1
            active.append((s0, k))
        k += 1
    runs: list[list[int]] = []
    for s0, e in active:
        if runs and t[s0] - t[runs[-1][1]] <= DISSOLVE_GAP_S and not any(runs[-1][1] < c <= s0 for c in cut_idx):
            runs[-1][1] = e
        else:
            runs.append([s0, e])
    # only stretches where a real share of frames moves (a dissolve over featureless detail leaves
    # 40-50% of the stretch nearly still in the thumbnails, so a median test would skip exactly those),
    # within a wall-clock budget, checked between scan chunks: the scan decodes every frame of a stretch
    runs = [(s0 - 1, e) for s0, e in runs if t[e] - t[s0 - 1] >= DISSOLVE_MIN_RUN_S
            and sum(x >= BURST_ACTIVE for x in d[s0:e + 1]) >= DISSOLVE_MOVING_SHARE * (e - s0 + 1)]
    started = time.monotonic()
    out: list[dict] = []
    for f_lo, f_hi in sorted(runs, key=lambda r: t[r[1]] - t[r[0]]):  # shortest first: most stretches
        for c0 in range(f_lo, f_hi, DISSOLVE_SCAN_MAX // 2):
            c1 = min(f_hi, c0 + DISSOLVE_SCAN_MAX)
            if time.monotonic() - started > DISSOLVE_SCAN_BUDGET_S:
                series["dissolve_scan_capped"] = True  # only when stretches were really left unscanned
                break

            def run(cv2, np, cap, c0=c0, c1=c1):
                grays = _pan_frames(cv2, np, cap, t, c0, c1, series.get("crop"))
                steps = _shifts_of(cv2, grays)
                sl = _slide(steps) if len(steps) >= PAN_MIN_STEPS else None
                if sl is None:
                    return []
                axis, speed, drift, tol = sl
                agree = sum(max(abs(s[axis] - speed), abs(s[1 - axis] - drift)) <= tol for s in steps)
                if agree < PAN_SHARE * len(steps):
                    return []  # not one steady slide: nothing to compensate
                m, res = _span_residuals(cv2, np, grays, steps, speed)
                vals = [r for r in res if r is not None]
                if len(vals) < 3 * DISSOLVE_MIN_SPANS:
                    return []
                thr = max(DISSOLVE_K * statistics.median(vals), DISSOLVE_MIN)
                found, j = [], 0
                while j < len(res):
                    if res[j] is not None and res[j] > thr:
                        j0 = j
                        while j + 1 < len(res) and res[j + 1] is not None and res[j + 1] > thr:
                            j += 1
                        if j - j0 + 1 >= DISSOLVE_MIN_SPANS and j0 >= m and j + 2 * m < len(res):
                            found.append((c0 + min(j0 + m, j), c0 + j))  # spans overlap the fade by m
                    j += 1
                return found

            for a0, a1 in _with_capture(series, run) or []:
                if a1 - a0 > MAX_GRADUAL_S * fps or any(a0 <= e + 1 and s0 - 1 <= a1 for s0, e in taken):
                    continue
                taken.append((a0, a1))
                out.append({"kind": "gradual", "t": round((t[a0] + t[a1]) / 2, 6), "frames": a1 - a0,
                            "t_start": t[a0], "t_end": t[a1], "confidence": UNCHECKED_CONF,
                            "note": PAN_UNCLEAR_NOTE})
    return out


def _count_pan_drop(series: dict, k: int = 1) -> None:
    series["pan_dropped"] = series.get("pan_dropped", 0) + k


UNCHECKED_CONF = 0.4
PAN_SKIPPED_NOTE = "opencv missing: continuous-pan check skipped"
PAN_UNREAD_NOTE = "media not readable: continuous-pan check skipped"


def _pan_skip_note() -> str:
    """Why ``continuous_pan`` returned None: OpenCV missing, else the media could not be read."""
    try:
        import cv2  # noqa: PLC0415,F401
        import numpy  # noqa: PLC0415,F401
    except Exception:
        return PAN_SKIPPED_NOTE
    return PAN_UNREAD_NOTE


def _gradual(series: dict, cut_idx: set[int]) -> list[dict]:
    """Dissolves: a window whose end frames differ a lot while the content on
    either side of the window is stable. Needs thumbnails; never guesses."""
    thumb, t = series.get("thumb") or [], series["t"]
    n = len(t)
    if len(thumb) != n or n < 8:
        return []
    fps = _fps_of(t)
    cap = int(MAX_GRADUAL_S * fps)
    flat_start, flat_end = _flat_extreme(thumb[0]), _flat_extreme(thumb[-1])
    cands = []
    for i0 in range(n):
        for length in _LENGTHS:
            if length > cap or length < GRADUAL_MIN_FRAMES:
                continue
            i1 = i0 + length
            pre_n = min(length, i0)
            post_n = min(length, n - 1 - i1)
            if i1 >= n or pre_n < 3 or post_n < 3:
                continue
            if any(i0 < c <= i1 for c in cut_idx):
                continue
            # a fade in from / out to black near a clip edge is not a shot boundary
            # (to black, white or any single colour: judged by the clip's own edge frame, since a
            # window can end a few frames before the picture has fully left)
            near = int(0.5 * fps)
            if (i0 < near and flat_start) or (n - 1 - i1 < near and flat_end):
                continue
            cross = _dist(thumb[i0], thumb[i1])
            if cross < MIN_CROSS:
                continue
            own = max(_dist(thumb[i0 - pre_n], thumb[i0]), _dist(thumb[i1], thumb[i1 + post_n]))
            cands.append((cross / (own + 0.02), cross, i0, i1))
    cands.sort(key=lambda c: (-c[0], c[3] - c[2]))
    out, taken = [], []
    for score, cross, i0, i1 in cands:
        if score < MIN_SCORE:
            break
        if any(i0 <= e + 1 and s0 - 1 <= i1 for s0, e in taken):
            continue
        taken.append((i0, i1))
        r0, r1 = _refine_window(thumb, i0, i1)
        steps = [_dist(thumb[k - 1], thumb[k]) for k in range(r0 + 1, r1 + 1)]
        if steps and max(steps) >= HARD_STEP * cross:
            # nearly all of the change is one frame: a hard cut between two moving pictures that the
            # score missed, not a gradual boundary
            k = r0 + 1 + steps.index(max(steps))
            out.append({"kind": "cut", "t": t[k], "frames": 1, "t_start": t[k - 1], "t_end": t[k],
                        "confidence": HARD_STEP_CONF})
            continue
        pan = continuous_pan(series, r0 - 3, r1 + 3)
        if pan:
            pan = slide_explains_window(series, r0, r1)
        if pan:
            _count_pan_drop(series)
            continue  # one camera move through the window, still moving the same way after it
        unchecked = pan is None or pan is UNCLEAR
        conf = _clamp(0.4 + 0.08 * score)
        bound = {"kind": "gradual", "t": round((t[r0] + t[r1]) / 2, 6), "frames": r1 - r0,
                 "t_start": t[r0], "t_end": t[r1], "confidence": round(conf, 3)}
        if unchecked:  # a continuous pan would look exactly like this: say so, claim less
            bound["confidence"] = min(bound["confidence"], UNCHECKED_CONF)
            bound["note"] = PAN_UNCLEAR_NOTE if pan is UNCLEAR else _pan_skip_note()
        out.append(bound)
    return out


def _boundaries(series: dict) -> list[dict]:
    t, scene, yavg = series["t"], series["scene"], series["yavg"]
    n = len(t)
    dy = [0.0] + [abs(yavg[i] - yavg[i - 1]) for i in range(1, n)]
    # cut strength: ffmpeg's scene score, or the luma jump when the score under-reads
    th = series.get("thumb") or []
    dt = _steps(series) if len(th) == n else [0.0] * n  # a big picture change the score under-reads
    cs = _chroma_steps(series) if len(th) == n else []
    # a colour-balance jump that stands out (a regrade / hue cut keeps the luma, so the score misses it)
    dc = [cs[i] / CHROMA_SCALE if _colour_edit(cs, i) else 0.0 for i in range(n)] if cs else [0.0] * n
    strength = [max(scene[i], dy[i] / LUMA_SCALE, dt[i] / THUMB_SCALE, dc[i]) for i in range(n)]
    thr = _cut_threshold(strength)
    cands = [i for i in range(1, n) if strength[i] >= thr]
    flash = _flash_cuts(cands, strength, series)
    cands = [i for i in cands if i not in flash]
    cut_idx: set[int] = set()
    out: list[dict] = []
    for i in cands:
        # adjacent candidates are one cut: keep the strongest (first on a tie)
        if i - 1 in cut_idx and strength[i] <= strength[i - 1]:
            continue
        if i - 1 in cut_idx:
            cut_idx.discard(i - 1)
            out = [b for b in out if b["t"] != t[i - 1]]
        cut_idx.add(i)
        out.append({"kind": "cut", "t": t[i], "frames": 1, "t_start": t[i - 1], "t_end": t[i],
                    "confidence": _clamp(0.5 + strength[i] / (2 * thr))})
    grads = _gradual(series, cut_idx)
    pos = {round(x, 6): i for i, x in enumerate(t)}
    taken = [(pos.get(round(g["t_start"], 6), 0), pos.get(round(g["t_end"], 6), 0)) for g in grads]
    out += grads + _moving_dissolves(series, cut_idx, taken)
    out.sort(key=lambda b: b["t"])
    return out


BURST_GAP_S = 0.4  # cuts closer than this form one candidate burst
BURST_ACTIVE = 0.015  # thumbnail change per frame below which the picture holds still
BURST_RUN = 4  # a move changes the picture for at least this many consecutive frames (accelerate, peak, brake)
BURST_RATIO = 0.4  # ... and the motion beside the cuts reaches this share of the weakest cut jump
BURST_HOLD = 3  # a held framing before and after the move: this many steps with no change
BURST_FLURRY = 3  # ... or this many cuts, each within BURST_FLURRY_GAP_S of the last (no edit cuts every 3 frames)
BURST_FLURRY_GAP_S = 0.16
BURST_MAX_S = 1.2  # a move is bounded by held frames: longer runs of change are ordinary footage
BURST_CONF = 0.4  # confidence cap of a collapsed motion boundary


def _steps(series: dict) -> list[float]:
    th = series.get("thumb") or []
    return [0.0] + [_dist(th[i - 1], th[i]) for i in range(1, len(th))]


CHROMA_EDIT = 0.05  # shift of the picture's colour balance (0..1, brightness removed) that is an edit ...
CHROMA_STAND_OUT = 3.0  # ... when it is this many x the colour drift of the frames around it
CHROMA_SCALE = 0.3  # a colour-balance jump of this size counts as a full-strength cut


def _chroma(th) -> tuple[float, float, float]:
    """Colour balance of a thumbnail: per-channel mean minus the mean of the three (brightness removed,
    so a lighting change or a flash of white does not count, a grade or hue change does)."""
    n = len(th) // 3
    m = [sum(th[c::3]) / n for c in range(3)]
    g = sum(m) / 3
    return tuple((v - g) / 255.0 for v in m)


def _chroma_steps(series: dict) -> list[float]:
    th = series.get("thumb") or []
    c = [_chroma(x) for x in th]
    return [0.0] + [max(abs(a - b) for a, b in zip(c[i - 1], c[i])) for i in range(1, len(c))]


def _colour_edit(cs: list[float], i: int) -> bool:
    """The colour balance jumps at frame ``i`` far beyond its own drift: a cut to a regraded or
    recoloured picture, however alike the structure (luma, shift) on either side."""
    if not cs or not 1 <= i < len(cs) or cs[i] < CHROMA_EDIT:
        return False
    near = sorted(cs[k] for k in range(max(1, i - 5), min(len(cs), i + 6)) if k != i)
    med = near[len(near) // 2] if near else 0.0
    # one frame, not a ramp: a pan sweeping into a new colour changes the balance over several steps
    adj = max(cs[k] for k in (i - 1, i + 1) if 1 <= k < len(cs)) if len(cs) > 2 else 0.0
    return cs[i] >= CHROMA_STAND_OUT * max(med, adj)


def _collapse_bursts(series: dict, bounds: list[dict]) -> list[dict]:
    """A fast whip / handheld move changes every frame, so it trips the cut detector once to a few
    times and leaves phantom 0.1 s shots. A real cut sits in a held picture: the frames around it
    barely change, and between two close cuts the picture holds still.

    Candidate groups (cuts < BURST_GAP_S apart; a lone cut is a group of one) are collapsed into one
    ``motion`` boundary when the picture is in motion on every step from before the first cut to
    after the last (a run of >= BURST_RUN changing frames) and that motion is not dwarfed by the cuts
    (in a montage of moving footage the cut jump is ~10x the motion around it). Needs thumbnails;
    otherwise the cuts stay."""
    t, th = series["t"], series.get("thumb") or []
    if len(th) != len(t) or not bounds:
        return bounds
    d = _steps(series)
    n = len(d)
    pos = {round(x, 6): i for i, x in enumerate(t)}
    cuts = [(pos.get(round(b["t"], 6)), b) for b in bounds if b["kind"] == "cut"]
    if any(i is None for i, _ in cuts):
        return bounds
    groups: list[list[tuple[int, dict]]] = []
    for i, b in cuts:
        if groups and t[i] - t[groups[-1][-1][0]] < BURST_GAP_S:
            groups[-1].append((i, b))
        else:
            groups.append([(i, b)])
    drop: set[int] = set()
    merged: list[dict] = []
    for g in groups:
        i0, i1 = g[0][0], g[-1][0]
        if any(d[k] < BURST_ACTIVE for k in range(i0, i1 + 1)):
            continue  # the picture holds still between two cuts
        lo, hi = i0, i1
        others = {i for i, _ in cuts} - {i for i, _ in g}  # a run never swallows another edit
        while lo - 1 > 0 and d[lo - 1] >= BURST_ACTIVE and lo - 1 not in others:
            lo -= 1
        while hi + 1 < n and d[hi + 1] >= BURST_ACTIVE and hi + 1 not in others:
            hi += 1
        cut_steps = {i for i, _ in g}
        beside = [d[k] for k in range(lo, hi + 1) if k not in cut_steps]
        jump = min(d[i] for i in cut_steps)
        if not BURST_RUN <= hi - lo + 1 or t[hi] - t[lo - 1] > BURST_MAX_S or not beside or max(beside) < BURST_RATIO * jump:
            continue
        if merged and t[lo - 1] <= merged[-1]["t_end"]:
            continue  # the same run already collapsed (two groups inside one move)
        held = (all(d[k] < BURST_ACTIVE for k in range(max(1, lo - BURST_HOLD), lo))
                and lo - BURST_HOLD >= 1
                and all(d[k] < BURST_ACTIVE for k in range(hi + 1, min(n, hi + 1 + BURST_HOLD)))
                and hi + BURST_HOLD < n)
        tight = all(t[b2] - t[a] <= BURST_FLURRY_GAP_S + 1e-6 for (a, _), (b2, _) in zip(g, g[1:]))
        flurry = len(g) >= BURST_FLURRY
        if len(g) > 1 and not tight:
            continue  # separate edits, however fast
        if not (held or flurry):
            continue  # ordinary footage in motion between two cuts
        keep: list[dict] = []
        first_b, last_b = g[0][1], g[-1][1]
        if len(g) >= 3:  # real edits can bracket a move: a cut that stands out of the motion and has a
            # held picture on its outer side stays a cut; only what lies between them is the move
            interior = sorted(d[k] for k in range(g[0][0] + 1, g[-1][0]))
            med = interior[len(interior) // 2] if interior else 0.0
            first, last = g[0], g[-1]

            def outer_steady(lo_i: int, hi_i: int) -> bool:
                # a held picture, or one steady slide (a cut from / into a pan): either way the cut's
                # outer side is one shot, so the cut is an edit and not part of the move (one frame of
                # margin from the cut: a seek landing a frame off must not pull the cut step in)
                return bool(continuous_pan(series, lo_i, hi_i))

            if d[first[0]] >= 2 * med and first[0] - 1 >= 1 and (
                    d[first[0] - 1] < BURST_ACTIVE or outer_steady(first[0] - 2 - 2 * PAN_CONTEXT, first[0] - 2)):
                keep.append(first[1])
                g = g[1:]
            if len(g) >= 2 and d[last[0]] >= 2 * med and last[0] + 1 < n and (
                    d[last[0] + 1] < BURST_ACTIVE or outer_steady(last[0] + 2, last[0] + 2 + 2 * PAN_CONTEXT)):
                keep.append(last[1])
                g = g[:-1]
            if not g:
                continue
            if len(keep) == 2:  # a camera move inside one shot between two real cuts: no boundary
                drop.update(id(b) for _, b in g)
                continue
            if keep and keep[0] is first[1]:
                lo = g[0][0]
            if keep and keep[-1] is last[1]:
                hi = g[-1][0]
        # a kept edge cut is an edit: the move is checked on its own side only
        p_lo = lo - 3
        p_hi = hi + 3
        if keep and keep[0] is first_b:
            p_lo = max(lo, pos[round(first_b["t"], 6)] + 2)
        if keep and keep[-1] is last_b:
            p_hi = min(hi, pos[round(last_b["t"], 6)] - 2)
        pan = continuous_pan(series, p_lo, p_hi)
        if not held or pan:
            # no held picture either side, or the move carries on in the same direction after it:
            # continuous motion, not an edit
            drop.update(id(b) for _, b in g)
            if pan:
                _count_pan_drop(series, len(g))
            continue
        drop.update(id(b) for _, b in g)
        merged.append({"kind": "motion", "t": round((t[lo - 1] + t[hi]) / 2, 6), "frames": hi - lo + 1,
                       "t_start": t[lo - 1], "t_end": t[hi],
                       "confidence": min(BURST_CONF, min(b["confidence"] for _, b in g)),
                       **({"note": _pan_skip_note()} if pan is None else {})})
    if not merged and not drop:
        return bounds
    out = [b for b in bounds if id(b) not in drop] + merged
    out.sort(key=lambda b: b["t"])
    return out


def correlation(a, b) -> float | None:
    """Mean per-colour-channel Pearson correlation of two RGB thumbnails (interleaved bytes); None
    when no channel has any spread. Per channel, because a single-colour frame is not "textured" just
    because its R, G and B levels differ."""
    cs = []
    for c in range(3):
        x, z = a[c::3], b[c::3]
        n = len(x)
        mx, mz = sum(x) / n, sum(z) / n
        vx = sum((v - mx) ** 2 for v in x)
        vz = sum((v - mz) ** 2 for v in z)
        if vx < 36.0 * n or vz < 36.0 * n:  # spread under ~6 levels: a flat channel, its pattern is noise
            continue
        cs.append(sum((v - mx) * (w - mz) for v, w in zip(x, z)) / (vx * vz) ** 0.5)
    return sum(cs) / len(cs) if cs else None


SAME_PICTURE_MAX_STEP = 0.1  # false steps on real footage were 0.055-0.067, real cuts >= 0.11; a big step is an edit however alike the frames
SAME_PICTURE = 0.8  # frames either side of a "cut" correlating this well are one picture
CUT_STAND_OUT = 2.5  # a cut's step must exceed the change around it by this factor


def _vet_cuts(series: dict, bounds: list[dict]) -> list[dict]:
    """Cuts need a jump that stands out of the picture's own motion. A step under CUT_STAND_OUT x the
    median change within +-5 frames is the same picture moving (a brightness or position nudge, a
    fast gesture): no cut. Correlated frames on either side with a standing-out jump may still be a
    jump cut inside one set, but frames that correlate that well are the same picture, so it is
    dropped as well (a 0.1 s sliver shot labelled cut was worse than the rare missed jump cut).
    Needs thumbnails."""
    t, th = series["t"], series.get("thumb") or []
    if len(th) != len(t):
        return bounds
    d = _steps(series)
    cs = _chroma_steps(series)
    pos = {round(x, 6): i for i, x in enumerate(t)}
    out = []
    merged_out = 0
    for b in bounds:
        i = pos.get(round(b["t"], 6)) if b["kind"] == "cut" else None
        if i is None or i < 1:
            out.append(b)
            continue
        y = series["yavg"]
        if 3 <= i:  # the luma has been falling / rising for several frames up to this one: a fade
            run = [y[i - k] - y[i - k - 1] for k in range(3)]
            if all(abs(v) >= 2.0 for v in run) and (all(v > 0 for v in run) or all(v < 0 for v in run)):
                out.append(b)  # transitions.classify owns fades
                continue
        if _colour_edit(cs, i):
            out.append(b)  # the colour balance jumps: a regrade / hue cut, not the same picture
            continue
        near = sorted(d[k] for k in range(max(1, i - 5), min(len(d), i + 6)) if k != i)
        med = near[len(near) // 2] if near else 0.0
        if med > 0 and d[i] < CUT_STAND_OUT * med:
            merged_out += 1
            continue
        corr = correlation(th[i - 1], th[i])
        if corr is not None and corr >= SAME_PICTURE and d[i] < SAME_PICTURE_MAX_STEP:
            merged_out += 1
            continue  # the same picture either side: a small step inside one shot, not an edit
        out.append(b)
    series["merged_same_picture"] = merged_out
    return out


def _drop_pan_cuts(series: dict, bounds: list[dict]) -> list[dict]:
    """A cut candidate inside one continuous pan is the slide itself (a fast pan over high-contrast detail
    jumps the score frame to frame), not an edit: dropped when ``continuous_pan`` shows the picture
    sliding the same way through it, the jump step included. The burst and gradual checks only see
    short runs between held framings, so a lone cut in a long pan reached the output at 1.0 before
    this pass. Without OpenCV a cut whose picture is moving on both sides is capped and says so."""
    t, th = series["t"], series.get("thumb") or []
    pos = {round(x, 6): i for i, x in enumerate(t)}
    d = _steps(series) if len(th) == len(t) else []
    out = []
    for b in bounds:
        i = pos.get(round(b["t"], 6)) if b["kind"] == "cut" else None
        if i is None:
            out.append(b)
            continue
        pan = continuous_pan(series, i - PAN_CONTEXT, i + PAN_CONTEXT)
        if pan:
            _count_pan_drop(series)
            continue
        moving = bool(d) and all(d[k] >= BURST_ACTIVE for k in range(max(1, i - PAN_CONTEXT), min(len(d), i + PAN_CONTEXT + 1)))
        if pan is None and moving:
            b = {**b, "confidence": min(b["confidence"], UNCHECKED_CONF), "note": _pan_skip_note()}
        out.append(b)
    return out


SINGLE_SHOT_CONF = 0.6  # no neighbouring boundary: a dissolve between two moving shots can go undetected


COLOUR_ONLY_NOTE = "colour change on the same picture: a cut or a lighting change, not determined"
RECURRING_NOTE = "recurring picture states (strobe, light cycle or rapid intercut): cuts not determined"
RECUR_FRAMES = 12  # a later cut this close that returns the picture to an earlier state
RECUR_RETURN = 0.3  # ... within this share of the first jump


def _grey_corr(a, b) -> float | None:
    """Pearson correlation of two thumbnails' grey levels; None when either is flat."""
    ga = [(a[k] + a[k + 1] + a[k + 2]) / 3.0 for k in range(0, len(a), 3)]
    gb = [(b[k] + b[k + 1] + b[k + 2]) / 3.0 for k in range(0, len(b), 3)]
    n = len(ga)
    ma, mb = sum(ga) / n, sum(gb) / n
    va = sum((v - ma) ** 2 for v in ga)
    vb = sum((v - mb) ** 2 for v in gb)
    if va < 36.0 * n or vb < 36.0 * n:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(ga, gb)) / (va * vb) ** 0.5


LUMA_SAME_MAX = 0.05  # largest slide-compensated luma residual (0..1) of a step on one picture
STRUCTURE_SAME = 0.75  # edge-map correlation (after the shift) above which two frames are one picture


def _grad(cv2, np, img):
    """Edge strength over luma and both chroma planes: a hue rotation turns the chroma plane, which
    keeps Cr/Cb edge strength, so a recoloured copy keeps its edges where a grey edge map would not
    (low-contrast colourful pictures have almost no grey edges)."""
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    tot = None
    for c in range(3):
        gx = cv2.Sobel(ycc[:, :, c], cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(ycc[:, :, c], cv2.CV_32F, 0, 1)
        tot = gx * gx + gy * gy if tot is None else tot + gx * gx + gy * gy
    return np.sqrt(tot)


def _same_structure(series: dict, i: int) -> bool | None:
    """True when the luma picture carries on through frame ``i``: every frame step around it, moved by
    the slide (median measured shift), leaves a residual within the pan break rule. A colour edit that
    keeps the luma (a hue / saturation change of the light or the grade) is then the same picture
    whatever its colour did; a real cut changes the luma picture. Every step is tested, so a seek
    landing a frame off changes nothing. None when it cannot run (no OpenCV / media)."""
    t = series["t"]
    lo, hi = max(1, i - PAN_CONTEXT), min(len(t) - 1, i + PAN_CONTEXT)

    def run(cv2, np, cap):
        grays = _pan_frames(cv2, np, cap, t, lo - 1, hi, series.get("crop"))
        steps = _shifts_of(cv2, grays)
        if len(steps) < 2:
            return None
        dx = statistics.median(s[0] for s in steps)
        dy = statistics.median(s[1] for s in steps)
        res = [_residual(cv2, np, grays[j], grays[j + 1], dx, dy) for j in range(len(steps))]
        known = [r for r in res if r is not None]
        if len(known) < 2:
            return None
        # relative (the break rule) and absolute: in a fast-moving shot the median is large, and a real
        # cut's residual (0.11-0.29 on the music video) would hide under 3x of it
        return max(known) <= min(LUMA_SAME_MAX, PAN_BREAK_RATIO * statistics.median(known) + PAN_BREAK_FLOOR)

    return _with_capture(series, run)


def _same_edges(series: dict, i: int) -> bool | None:
    """True when every frame step around frame ``i`` keeps its edges (luma + chroma edge strength,
    correlation after the measured shift >= STRUCTURE_SAME): a light that also changes the saturation
    alters the luma, but not where the edges are. The weakest step counts (seek-proof)."""
    t = series["t"]
    lo, hi = max(1, i - PAN_CONTEXT), min(len(t) - 1, i + PAN_CONTEXT)

    def run(cv2, np, cap):
        pairs = _pan_frames(cv2, np, cap, t, lo - 1, hi, series.get("crop"), colour=True)
        if len(pairs) < 2:
            return None
        edges = [_grad(cv2, np, c) for _, c in pairs]
        win = cv2.createHanningWindow(edges[0].shape[::-1], cv2.CV_32F)
        for a, b in zip(edges, edges[1:]):
            (dx, dy), _r = cv2.phaseCorrelate(a.copy(), b.copy(), win)
            h, w = a.shape
            moved = cv2.warpAffine(a, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h))
            x0, x1 = int(max(0.0, dx)) + 2, int(min(w, w + dx)) - 2
            y0, y1 = int(max(0.0, dy)) + 2, int(min(h, h + dy)) - 2
            if x1 - x0 < 0.3 * w or y1 - y0 < 0.3 * h:
                return False
            u, v = moved[y0:y1, x0:x1].ravel(), b[y0:y1, x0:x1].ravel()
            if float(u.std()) < 1e-3 or float(v.std()) < 1e-3 or float(np.corrcoef(u, v)[0, 1]) < STRUCTURE_SAME:
                return False
        return True

    return _with_capture(series, run)


def _cap(b: dict, note: str) -> dict:
    return {**b, "confidence": min(b["confidence"], UNCHECKED_CONF), "note": b.get("note") or note}


def _uncertain_colour_cuts(series: dict, bounds: list[dict]) -> list[dict]:
    """Cuts kept, but capped, when the colour is all that changed or the light keeps coming back.

    - Colour only: the colour balance jumps while the luma picture carries on (``_same_structure``;
      thumbnails' grey correlation without OpenCV). A cut to a regraded copy and a lighting
      change on the same picture measure exactly the same, so neither is claimed: capped at 0.4.
    - Recurring states: a later cut within RECUR_FRAMES returns the picture to where it was before an
      earlier one (a strobe, a light cycle). Flash frames (one return) are already dropped; a run of
      returns is capped at 0.4. Needs thumbnails."""
    t, th = series["t"], series.get("thumb") or []
    if len(th) != len(t):
        return bounds
    pos = {round(x, 6): i for i, x in enumerate(t)}
    cs = _chroma_steps(series)
    cut_at = {pos[round(b["t"], 6)]: b for b in bounds if b["kind"] == "cut" and round(b["t"], 6) in pos}
    note: dict[int, str] = {}
    for i in cut_at:
        if i < 1 or not _colour_edit(cs, i):
            continue
        same = _same_structure(series, i)
        if same is False:
            same = _same_edges(series, i)
        if same is None:  # no media check: fall back to the thumbnails' grey structure
            gc = _grey_corr(th[i - 1], th[i])
            same = gc is not None and gc >= SAME_PICTURE
        if same:
            note[i] = COLOUR_ONLY_NOTE
    idx = sorted(cut_at)
    for a, i in enumerate(idx):
        if i < 1:
            continue
        jump = _dist(th[i - 1], th[i])
        for j in idx[a + 1:]:
            if j - i > RECUR_FRAMES:
                break
            if jump > 0 and _dist(th[i - 1], th[j]) < RECUR_RETURN * jump:
                for k in idx[a:idx.index(j) + 1]:
                    note.setdefault(k, RECURRING_NOTE)
                break
    if not note:
        return bounds
    capped = {id(cut_at[i]): n for i, n in note.items()}
    return [_cap(b, capped[id(b)]) if id(b) in capped else b for b in bounds]


def detect_shots(series: dict, duration: float) -> list[dict]:
    """Shots from a frame series. Empty list when nothing can be measured."""
    if not series or len(series.get("t", [])) < 2 or not duration or duration <= 0:
        return []
    series["pan_dropped"] = 0
    reader = series["_pan_reader"] = _PanReader(series["path"]) if series.get("path") else None
    try:
        bounds = _drop_pan_cuts(series, _vet_cuts(series, _collapse_bursts(series, _boundaries(series))))
        bounds = _uncertain_colour_cuts(series, bounds)
    finally:
        series.pop("_pan_reader", None)
        if reader is not None:
            reader.close()
    return build_shots(bounds, duration)


def build_shots(bounds: list[dict], duration: float) -> list[dict]:
    """Shots between consecutive boundaries (see ``detect_shots`` for the boundary dict)."""
    edges = [0.0] + [b["t"] for b in bounds] + [float(duration)]
    shots = []
    for k in range(len(edges) - 1):
        start, end = edges[k], edges[k + 1]
        adj = [bounds[x] for x in (k - 1, k) if 0 <= x < len(bounds)]
        conf = min((b["confidence"] for b in adj), default=SINGLE_SHOT_CONF)
        shot = {
            "index": k, "start": round(start, 6), "end": round(end, 6),
            "duration": round(end - start, 6),
            "boundary": None,
            "evidence": {"t_start": round(start, 6), "t_end": round(end, 6),
                         "method": "measured", "confidence": round(conf, 3)},
        }
        if k > 0:
            b = bounds[k - 1]
            shot["boundary"] = {"kind": b["kind"], "t": round(b["t"], 6), "frames": b["frames"],
                                "evidence": {"t_start": b["t_start"], "t_end": b["t_end"],
                                             "method": "measured", "confidence": round(b["confidence"], 3)}}
            if b.get("note"):
                shot["boundary"]["note"] = b["note"]
        shots.append(shot)
    return shots


def rhythm(shots: list[dict]) -> dict:
    """Cut-rhythm stats; never divides by zero, single shot is its own window."""
    if not shots:
        return {"count": 0, "mean": NOT_DETERMINED, "median": NOT_DETERMINED,
                "min": NOT_DETERMINED, "max": NOT_DETERMINED,
                "fastest_window": NOT_DETERMINED, "slowest_window": NOT_DETERMINED,
                "evidence": {"t_start": 0.0, "t_end": 0.0, "method": "measured", "confidence": 0}}
    durs = [s["duration"] for s in shots]
    span = shots[-1]["end"] - shots[0]["start"]
    # a window is 3 shots on longer edits, fewer on short ones so fastest and slowest differ
    size = 3 if len(shots) >= 4 else max(1, len(shots) // 2)
    best = min(range(len(shots) - size + 1), key=lambda i: sum(durs[i:i + size]))
    win = shots[best:best + size]
    worst = max(range(len(shots) - size + 1), key=lambda i: sum(durs[i:i + size]))
    slow = shots[worst:worst + size]
    return {
        "count": len(shots),
        "mean": span / len(shots),
        "median": statistics.median(durs),
        "min": min(durs),
        "max": max(durs),
        "fastest_window": {"t_start": win[0]["start"], "t_end": win[-1]["end"],
                           "mean": sum(durs[best:best + size]) / size},
        "slowest_window": {"t_start": slow[0]["start"], "t_end": slow[-1]["end"],
                           "mean": sum(durs[worst:worst + size]) / size},
        "evidence": {"t_start": shots[0]["start"], "t_end": shots[-1]["end"],
                     "method": "measured", "confidence": round(min(s["evidence"]["confidence"] for s in shots), 3)},
    }
