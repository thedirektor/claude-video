"""``watch.py compare``: a reference against the user's own version (spec §2.3, Builder-check 4).

The candidate is transient: it is analysed and rendered next to the reference,
never indexed or stored. Alignment, in order: an explicit ``offset`` (the user
knows best), shared-audio cross-correlation of onset envelopes, then the first
visible motion in each clip. Periodic audio (a click track, a steady beat)
scores lags one beat apart alike; the picture is used to pick between them, and
when it cannot, the offset is reported with low confidence and a note. Cuts and
motion spans are mapped one-to-one inside the aligned overlap only; whatever
does not line up goes to ``unmatched`` instead of being forced onto a partner.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from . import audio, media, shots

NOT_DETERMINED = "not determined"
GRID = 0.01  # alignment resolution, seconds
MIN_OVERLAP_S = 2.0
AUDIO_MIN_SCORE = 0.5  # coverage-weighted correlation below this: the clips share no usable audio
VIS_MIN_SCORE = 0.5  # picture cross-correlation needed to align on the picture alone
VIS_CHECK = 0.2  # picture agreement an onset alignment must reach to keep its confidence
UNCONFIRMED_CONF = 0.2
ONSET_MAX_CONF = 0.6  # one event aligns the clips; its own timing difference is absorbed
AMBIG_PARTNER_S = 0.15  # a second partner this close in |delta| makes a pairing ambiguous
PEAK_ALIKE = 0.85  # peaks scoring within this share of the best are alternatives
PEAK_SEP = 0.05  # seconds: closer peaks are the same alignment
VIS_MARGIN = 0.1  # picture correlation lead needed to pick between alike audio peaks
VIS_MIN_LUMA_STD = 1.0  # grey levels: flatter luma carries no alignment information
VIS_MIN_CHANGE_STD = 0.2
AMBIG_CONF = 0.3
ONSET_FACTOR = 3.0  # motion onset: frame difference above this x its initial median
ONSET_FLOOR = 0.5  # ...and above this many grey levels (static synthetic clips have median 0)
SPAN_MIN_FRAMES = 3
SPAN_GAP_FRAMES = 2
MATCH_TOL_S = 0.5
NORM_HEIGHT = 360
THUMB_W, THUMB_H = 64, 36
_FONTS = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
          "/usr/share/fonts/dejavu/DejaVuSans.ttf", "/Library/Fonts/Arial.ttf", "C:/Windows/Fonts/arial.ttf")


class CompareError(ValueError):
    pass


def _ev(t0: float, t1: float, conf: float, method: str = "measured") -> dict:
    return {"t_start": round(t0, 4), "t_end": round(t1, 4), "method": method,
            "confidence": round(max(0.0, min(1.0, conf)), 3)}


# ---- inputs -----------------------------------------------------------------

def _resolve(source: str, tmp: Path, conn_factory) -> dict:
    """``{"path", "info", "key", "label"}`` for a local path, URL or stored video_key."""
    import download

    p = Path(source).expanduser()
    if p.is_file():
        info = media.probe(p)
        if not info:
            raise CompareError(f"not a readable video: {source}")
        return {"path": str(p), "info": info, "key": None, "label": p.name}
    if download.is_url(source):
        got = media.default_fetch(source, tmp)
        path = got.get("video_path") if isinstance(got, dict) else got
        if not path or not Path(path).is_file():
            raise CompareError(f"could not download {source}")
        info = media.probe(Path(path))
        if not info:
            raise CompareError(f"downloaded file is not a readable video: {source}")
        return {"path": str(path), "info": info, "key": None, "label": source}
    if p.suffix or "/" in source or "\\" in source:
        raise CompareError(f"file not found: {source}")
    from memory import retrieval

    conn = conn_factory()
    try:
        try:
            key = retrieval.resolve_video(conn, source)["video_key"]
        except (retrieval.VideoNotFound, retrieval.AmbiguousVideo) as exc:
            raise CompareError(str(exc)) from exc
        m = media.ensure_media(conn, key, fetch=media.default_fetch)
    finally:
        conn.close()
    if not m.get("path"):
        raise CompareError(f"{key}: full media not available (storyboard-only), cannot compare")
    info = {k: m[k] for k in ("fps", "fps_rational", "duration", "width", "height", "has_audio")}
    return {"path": m["path"], "info": info, "key": key, "label": key}


def _gray(path: str, np):
    """Per-frame ``(mean luma, mean abs difference to the previous frame)`` on a tiny grey copy."""
    try:
        res = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path, "-an",
             "-vf", f"scale={THUMB_W}:{THUMB_H},format=gray", "-vsync", "0", "-f", "rawvideo", "-"],
            capture_output=True, timeout=1800)
    except (OSError, subprocess.SubprocessError):
        return None, None
    n = len(res.stdout) // (THUMB_W * THUMB_H)
    if n < 2:
        return None, None
    f = np.frombuffer(res.stdout[: n * THUMB_W * THUMB_H], dtype=np.uint8).reshape(n, THUMB_H * THUMB_W)
    f = f.astype(np.float32)
    yavg = f.mean(axis=1)
    diff = np.concatenate([[0.0], np.abs(f[1:] - f[:-1]).mean(axis=1)]).astype(np.float32)
    return yavg, diff


def _to_grid(values, fps: float, duration: float, np):
    """Sample-and-hold a per-frame series onto the GRID clock."""
    g = np.arange(0.0, max(duration, GRID), GRID)
    idx = np.minimum((g * fps).astype(int), len(values) - 1)
    return values[idx]


def _impulses(values, fps: float, duration: float, np):
    """A per-frame change series as impulses at each frame's start time on the GRID clock, smoothed
    by the same 30 ms Gaussian for every clip (holding a 25 fps value for 40 ms but a 60 fps one
    for 17 ms would bias the cross-correlation peak)."""
    out = np.zeros(int(max(duration, GRID) / GRID) + 1, dtype=np.float64)
    idx = np.clip((np.arange(len(values)) / fps / GRID).round().astype(int), 0, len(out) - 1)
    np.maximum.at(out, idx, values)
    k = np.arange(-9, 10)
    g = np.exp(-0.5 * (k * GRID / 0.03) ** 2)
    return np.convolve(out, g / g.sum(), mode="same")


def _audio_grid(path: str, duration: float, np):
    x, note = audio._decode(path)
    if x is None:
        return None, note
    got = audio._onset_envelope(x, np)
    if got is None:
        return None, "audio too short"
    flux, t0, hop_s = got
    n = int(max(duration, len(x) / audio.SR) / GRID) + 1
    out = np.zeros(n, dtype=np.float64)
    idx = np.clip(((t0 + np.arange(len(flux)) * hop_s) / GRID).astype(int), 0, n - 1)
    np.maximum.at(out, idx, flux)
    return out, ""


# ---- alignment --------------------------------------------------------------

def _need(na: int, nb: int) -> int:
    """Minimum overlap in grid samples: half the shorter clip, and MIN_OVERLAP_S when the clips are
    long enough to afford it (short game clips are still comparable)."""
    short = min(na, nb)
    return max(int(0.5 * short), min(int(MIN_OVERLAP_S / GRID), int(0.8 * short)))


def _xcorr(a, b, np):
    """``(lags, score)``: overlap-normalised correlation, cand index = ref index + lag."""
    a = (a - a.mean()) / (a.std() or 1.0)
    b = (b - b.mean()) / (b.std() or 1.0)
    na, nb = len(a), len(b)
    size = 1 << (na + nb).bit_length()
    c = np.fft.irfft(np.fft.rfft(b, size) * np.conj(np.fft.rfft(a, size)), size)
    lags = np.arange(-(na - 1), nb)
    vals = np.concatenate([c[size - (na - 1):], c[:nb]]) if na > 1 else c[:nb]
    overlap = np.minimum(na, nb - lags) - np.maximum(0, -lags)
    ok = overlap >= _need(na, nb)
    # weight by how much of the shorter clip the overlap covers: dividing by a short overlap
    # inflates chance correlations at the edges of the search range
    score = np.where(ok, vals / np.maximum(overlap, 1) * overlap / min(na, nb), -np.inf)
    return lags, score


def _peaks(lags, score, np) -> list[tuple[float, float]]:
    """Local maxima ``(lag_s, score)``, best first, merged when closer than PEAK_SEP."""
    s = score
    finite = np.isfinite(s)
    left = np.concatenate([[-np.inf], s[:-1]])
    right = np.concatenate([s[1:], [-np.inf]])
    idx = np.where(finite & (s >= left) & (s >= right))[0]
    order = idx[np.argsort(-s[idx])]
    out: list[tuple[float, float]] = []
    for i in order:
        lag = float(lags[i]) * GRID
        if all(abs(lag - o[0]) > PEAK_SEP for o in out):
            if 0 < i < len(s) - 1 and np.isfinite(s[i - 1]) and np.isfinite(s[i + 1]):
                den = s[i - 1] - 2 * s[i] + s[i + 1]
                if den < 0:
                    lag += GRID * 0.5 * (s[i - 1] - s[i + 1]) / den
            out.append((lag, float(s[i])))
        if len(out) >= 8:
            break
    return out


def _pearson_at(a, b, lag_s: float, min_std: float) -> float | None:
    """Correlation over the overlap at a lag, weighted by how much of the shorter clip the overlap
    covers (a short overlap correlates well by chance). None when either side is flat or the
    overlap is too short to say anything."""
    k = int(round(lag_s / GRID))
    i0, i1 = max(0, -k), min(len(a), len(b) - k)
    if i1 - i0 < max(2, _need(len(a), len(b))):
        return None
    x, y = a[i0:i1], b[i0 + k:i1 + k]
    sx, sy = float(x.std()), float(y.std())
    if sx < min_std or sy < min_std:
        return None
    r = float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))
    return r * (i1 - i0) / min(len(a), len(b))


def _has_picture(rv, cv) -> bool:
    return all(float(x.std()) >= VIS_MIN_CHANGE_STD for x in (rv[1], cv[1]))


def _visual_alignment(rv, cv, np, notes: list[str]):
    """``(offset_s, confidence)`` from cross-correlating the frame-change series, or None."""
    if not _has_picture(rv, cv):
        return None
    lags, score = _xcorr(rv[1], cv[1], np)
    peaks = _peaks(lags, score, np)
    if not peaks or peaks[0][1] < VIS_MIN_SCORE:
        return None
    if len(peaks) > 1 and peaks[1][1] >= PEAK_ALIKE * peaks[0][1]:
        notes.append("picture: several alignments score alike")
        return None
    return peaks[0][0], min(0.85, peaks[0][1])


def _vis_score(rv, cv, lag_s: float) -> float:
    """Picture agreement: mean of luma and frame-change correlation over whichever of the two carries
    information (``rv``/``cv`` = (luma, change)); 0 when neither does."""
    got = [r for r in (_pearson_at(rv[0], cv[0], lag_s, VIS_MIN_LUMA_STD),
                       _pearson_at(rv[1], cv[1], lag_s, VIS_MIN_CHANGE_STD)) if r is not None]
    return sum(got) / len(got) if got else 0.0


def _audio_alignment(ra, ca, rv, cv, np, notes: list[str]):
    """``(offset_s, confidence, alternatives)`` from audio, or None when the audio is unusable.
    ``alternatives`` lists every alike offset when the choice is unresolved, else ``[]``."""
    lags, score = _xcorr(ra, ca, np)
    peaks = _peaks(lags, score, np)
    if not peaks or peaks[0][1] < AUDIO_MIN_SCORE:
        notes.append("audio: no shared audio pattern")
        return None
    best = peaks[0][1]
    alike = [p for p in peaks if p[1] >= PEAK_ALIKE * best]
    base_conf = min(0.95, best)
    if len(alike) == 1:
        if rv is not None and _has_picture(rv, cv) and _vis_score(rv, cv, peaks[0][0]) < VIS_CHECK:
            notes.append(f"audio: best match {peaks[0][0]:+.3f} s is contradicted by the picture; "
                         "treating the audio as not shared")
            return None
        return peaks[0][0], base_conf, []
    vis = sorted(((_vis_score(rv, cv, lag), lag) for lag, _ in alike), reverse=True) if rv is not None else []
    if vis and vis[0][0] >= 0.5 and vis[0][0] - vis[1][0] >= VIS_MARGIN:
        notes.append(f"audio: {len(alike)} alignments one period apart score alike; the picture picked "
                     f"{vis[0][1]:+.3f} s")
        return vis[0][1], min(base_conf, vis[0][0], 0.9), []
    return peaks[0][0], AMBIG_CONF, [lag for lag, _ in alike]


def _onset(diff, fps: float, np):
    """Time of the first frame whose change exceeds ONSET_FACTOR x the initial median, or None."""
    k0 = max(3, int(0.1 * fps))
    if len(diff) <= k0 + 1:
        return None
    thr = max(ONSET_FACTOR * float(np.median(diff[1:k0])), ONSET_FLOOR)
    hits = np.where(diff[1:] > thr)[0]
    if not len(hits) or hits[0] + 1 <= 1:
        return None
    i = int(hits[0]) + 1
    return {"t": i / fps, "strength": float(diff[i] / thr)}


# ---- mapping ----------------------------------------------------------------

def _match(ref: list[float], cand: list[float], off: float):
    """``(pairs, ambiguous_ref, ambiguous_cand)``: an order-preserving minimum-cost alignment of the
    two event lists (pair cost = |delta| within MATCH_TOL_S, skipping an event costs MATCH_TOL_S).
    A pair whose event has a neighbour almost as close on the other side is not trusted: both go to
    the ambiguous sets instead of being paired."""
    n, m, gap = len(ref), len(cand), MATCH_TOL_S
    inf = float("inf")
    cost = [[inf] * (m + 1) for _ in range(n + 1)]
    move = [[None] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            c0 = cost[i][j]
            if c0 == inf:
                continue
            if i < n and c0 + gap < cost[i + 1][j]:
                cost[i + 1][j], move[i + 1][j] = c0 + gap, "r"
            if j < m and c0 + gap < cost[i][j + 1]:
                cost[i][j + 1], move[i][j + 1] = c0 + gap, "c"
            if i < n and j < m:
                d = abs(cand[j] - off - ref[i])
                if d <= MATCH_TOL_S and c0 + d < cost[i + 1][j + 1]:
                    cost[i + 1][j + 1], move[i + 1][j + 1] = c0 + d, "m"
    pairs, i, j = [], n, m
    while i or j:
        mv = move[i][j]
        if mv == "m":
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif mv == "r":
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    keep, amb_r, amb_c = [], set(), set()
    for i, j in pairs:
        d = abs(cand[j] - off - ref[i])
        rival = any(0 <= k < m and k != j and abs(abs(cand[k] - off - ref[i]) - d) <= AMBIG_PARTNER_S
                    and abs(cand[k] - off - ref[i]) <= MATCH_TOL_S for k in (j - 1, j + 1))
        rival = rival or any(0 <= k < n and k != i and abs(abs(cand[j] - off - ref[k]) - d) <= AMBIG_PARTNER_S
                             and abs(cand[j] - off - ref[k]) <= MATCH_TOL_S for k in (i - 1, i + 1))
        if rival:
            amb_r.add(i)
            amb_c.add(j)
        else:
            keep.append((i, j))
    return keep, amb_r, amb_c


def _boundaries(path: str, info: dict) -> list[dict]:
    series = shots.frame_series(path, info["fps"])
    found = shots.detect_shots(series, info["duration"])
    return [dict(s["boundary"]) for s in found[1:] if s.get("boundary")]


def _spans(diff, fps: float, np) -> list[tuple[float, float]]:
    if diff is None or len(diff) < 4:
        return []
    k0 = max(3, int(0.1 * fps))
    thr = max(ONSET_FACTOR * float(np.median(diff[1:k0])), ONSET_FLOOR)
    active = diff > thr
    out, i, n = [], 1, len(diff)
    while i < n:
        if not active[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and (active[j + 1] or any(active[j + 1:j + 1 + SPAN_GAP_FRAMES + 1])):
            j += 1
        if j - i + 1 >= SPAN_MIN_FRAMES:
            out.append((i / fps, (j + 1) / fps))
        i = j + 1
    return out


def _in(t: float, lo: float, hi: float) -> bool:
    return lo - 1e-6 <= t <= hi + 1e-6


# ---- side by side -----------------------------------------------------------

def _even(x: float) -> int:
    return max(2, int(round(x / 2)) * 2)


def _font() -> str | None:
    return next((f for f in _FONTS if Path(f).is_file()), None)


def _render(ref: dict, cand: dict, off: float, a: float, b: float, slow: int, out: Path) -> dict:
    ri, ci = ref["info"], cand["info"]
    h = _even(min(NORM_HEIGHT, ri["height"] or NORM_HEIGHT))
    w = _even(h * (ri["width"] / ri["height"]) if ri["width"] and ri["height"] else h * 16 / 9)
    fps = ri["fps"] or ci["fps"] or 30.0
    font = _font()

    def label(text: str) -> str:
        if not font:
            return ""
        return (f",drawtext=fontfile={font}:expansion=none:text='{text}':x=10:y=10:fontsize={max(12, h // 18)}:"
                "fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=4")

    def leg(i: int, start: float, name: str) -> str:
        return (f"[{i}:v]trim=start={start:.4f}:duration={b - a:.4f},setpts=PTS-STARTPTS,fps={fps:.6f},"
                f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1"
                f"{label(name)},setpts={slow}*PTS,fps={fps:.6f}[{name}]")

    graph = ";".join([leg(0, a, "reference"), leg(1, a + off, "yours"), "[reference][yours]hstack=inputs=2[v]"])
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", ref["path"], "-i", cand["path"],
                    "-filter_complex", graph, "-map", "[v]", "-an", "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
                   check=True, capture_output=True, text=True, timeout=3600)
    return {"width": w, "height": h, "fps": round(fps, 4)}


# ---- entry point ------------------------------------------------------------

def _default_out(ref: dict) -> Path:
    from config import memory_paths

    root = Path(memory_paths()["store_dir"])
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if ref.get("key"):
        from memory import index

        from . import assemble

        conn = index.connect(memory_paths()["index_db"])
        try:
            return assemble.breakdown_dir(conn, root, ref["key"]) / f"compare-{stamp}"
        finally:
            conn.close()
    return root / "compare" / stamp


def _conn():
    from config import memory_paths
    from memory import index

    return index.connect(memory_paths()["index_db"])


def run(reference: str, candidate: str, *, offset: float | None = None, slow: int = 4,
        out_dir: Path | None = None, conn_factory=_conn) -> dict:
    slow = max(1, int(slow))
    tmp = Path(tempfile.mkdtemp(prefix="watch-compare-"))
    try:
        ref = _resolve(reference, tmp / "ref", conn_factory)
        cand = _resolve(candidate, tmp / "cand", conn_factory)
        out_dir = Path(out_dir) if out_dir else _default_out(ref)
        out_dir.mkdir(parents=True, exist_ok=True)
        res = _compare(ref, cand, offset, slow, out_dir)
        (out_dir / "compare.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        return res
    finally:
        shutil.rmtree(tmp, ignore_errors=True)  # a downloaded candidate is transient


def _compare(ref: dict, cand: dict, offset: float | None, slow: int, out_dir: Path) -> dict:
    ri, ci = ref["info"], cand["info"]
    notes: list[str] = []
    res = {"schema": "watch.compare/1",
           "reference": {"source": ref["label"], "duration": ri["duration"], "fps": ri["fps"]},
           "candidate": {"source": cand["label"], "duration": ci["duration"], "fps": ci["fps"], "transient": True},
           "alignment": None, "overlap": None, "normalised": None, "slow": slow,
           "shots": [], "shot_durations": [], "phases": [], "unmatched": [], "notes": notes, "video": None}
    try:
        import numpy as np  # noqa: PLC0415
    except ImportError:
        np = None
        notes.append("numpy missing: no automatic alignment or timing deltas (pip install numpy)")

    rv = cv = rdiff = cdiff = None
    if np is not None:
        ry, rdiff = _gray(ref["path"], np)
        cy, cdiff = _gray(cand["path"], np)
        if ry is not None and cy is not None:
            rv = (_to_grid(ry, ri["fps"], ri["duration"], np), _impulses(rdiff, ri["fps"], ri["duration"], np))
            cv = (_to_grid(cy, ci["fps"], ci["duration"], np), _impulses(cdiff, ci["fps"], ci["duration"], np))

    span_end = max(ri["duration"], ci["duration"])
    al = None
    if offset is not None:
        al = {"method": "offset", "offset_s": float(offset), "confidence": 1.0,
              "evidence": _ev(0.0, span_end, 1.0, "given")}
    alternatives: list[float] = []
    if al is None and np is not None and ri["has_audio"] and ci["has_audio"]:
        ra, n1 = _audio_grid(ref["path"], ri["duration"], np)
        ca, n2 = _audio_grid(cand["path"], ci["duration"], np)
        got = _audio_alignment(ra, ca, rv, cv, np, notes) if ra is not None and ca is not None else None
        if ra is None or ca is None:
            notes.append(f"audio: {n1 or n2}")
        if got is not None:
            lag, conf, alternatives = got
            al = {"method": "audio", "offset_s": round(lag, 4), "confidence": round(conf, 3),
                  "evidence": _ev(0.0, span_end, conf)}
    onset = None
    if (al is None or alternatives) and np is not None and rdiff is not None and cdiff is not None:
        o_r, o_c = _onset(rdiff, ri["fps"], np), _onset(cdiff, ci["fps"], np)
        if o_r and o_c:
            conf = 0.8 if min(o_r["strength"], o_c["strength"]) >= 2 else 0.5
            onset = (round(o_c["t"] - o_r["t"], 4), conf, o_r, o_c)
    if alternatives:
        tol = 2.0 / (ri["fps"] or 25.0)
        agree = [lag for lag in alternatives if onset and abs(lag - onset[0]) <= tol]
        if len(agree) == 1:
            notes.append(f"audio: {len(alternatives)} alignments one period apart score alike; the first "
                         f"visible motion picked {agree[0]:+.3f} s")
            al.update(offset_s=round(agree[0], 4), confidence=min(onset[1], 0.7),
                      evidence=_ev(0.0, span_end, min(onset[1], 0.7)))
        else:
            lags_txt = ", ".join(f"{lag:+.2f}" for lag in alternatives[:4])
            notes.append(f"audio alignment ambiguous: offsets {lags_txt} s score alike (periodic audio) "
                         "and the picture cannot tell them apart; pass --offset")
            al["alternatives_s"] = [round(x, 4) for x in alternatives]
    elif al is None and onset:
        off_s, conf, o_r, o_c = onset
        if rv is not None and _has_picture(rv, cv):
            here = _vis_score(rv, cv, off_s)
            near = max(_vis_score(rv, cv, off_s - 0.5), _vis_score(rv, cv, off_s + 0.5))
            if here < VIS_CHECK or here <= near:
                notes.append(f"first visible motion suggests {off_s:+.3f} s but the rest of the picture does "
                             "not agree (the two first motions may be different events)")
                conf = UNCONFIRMED_CONF
        if conf > UNCONFIRMED_CONF:
            al = {"method": "motion_onset", "offset_s": off_s, "confidence": min(conf, ONSET_MAX_CONF),
                  "evidence": _ev(o_r["t"], o_r["t"], min(conf, ONSET_MAX_CONF)),
                  "onsets": {"ref": o_r["t"], "cand": o_c["t"]}}
            notes.append("motion-onset alignment: deltas are relative to the first detected motion, which "
                         "absorbs that event's own timing difference")
    elif al is None and np is not None:
        notes.append("no motion onset found in both clips")
    if al is None and np is not None and rv is not None:
        got = _visual_alignment(rv, cv, np, notes)
        if got is not None:
            al = {"method": "visual", "offset_s": round(got[0], 4), "confidence": round(got[1], 3),
                  "evidence": _ev(0.0, span_end, got[1])}
    if al is None:
        al = {"method": NOT_DETERMINED, "offset_s": NOT_DETERMINED, "confidence": 0,
              "evidence": _ev(0.0, span_end, 0.0)}
        notes.append("alignment not determined: pass --offset SECONDS (candidate time = reference time + offset); "
                     "the side-by-side is unaligned and no timing deltas are reported")
    res["alignment"] = al
    aligned = al["method"] != NOT_DETERMINED and float(al["confidence"]) > UNCONFIRMED_CONF
    if al["method"] != NOT_DETERMINED and not aligned:
        notes.append("alignment unconfirmed: no timing deltas reported; pass --offset")
    off = float(al["offset_s"]) if al["method"] != NOT_DETERMINED else 0.0

    a = max(0.0, -off)
    b = min(ri["duration"], ci["duration"] - off)
    if b - a <= 0:
        raise CompareError(f"no overlap between the clips at offset {off:+.3f} s")
    res["overlap"] = {"ref_start": round(a, 4), "ref_end": round(b, 4)}

    if aligned:
        conf = float(al["confidence"])
        rb = [x for x in _boundaries(ref["path"], ri) if _in(x["t"], a, b)]
        cb = [x for x in _boundaries(cand["path"], ci) if _in(x["t"] - off, a, b)]
        pairs, amb_r, amb_c = _match([x["t"] for x in rb], [x["t"] for x in cb], off)
        for i, j in pairs:
            r, c = rb[i], cb[j]
            res["shots"].append({"ref": r["t"], "cand": c["t"], "kind_ref": r["kind"], "kind_cand": c["kind"],
                                 "delta_ms": round((c["t"] - off - r["t"]) * 1000, 1),
                                 "evidence": _ev(r["t"], r["t"], conf)})
        for (i0, j0), (i1, j1) in zip(pairs, pairs[1:]):
            rd, cd = rb[i1]["t"] - rb[i0]["t"], cb[j1]["t"] - cb[j0]["t"]
            res["shot_durations"].append({"ref": [rb[i0]["t"], rb[i1]["t"]], "cand": [cb[j0]["t"], cb[j1]["t"]],
                                          "delta_ms": round((cd - rd) * 1000, 1),
                                          "evidence": _ev(rb[i0]["t"], rb[i1]["t"], conf)})
        mi, mj = {i for i, _ in pairs}, {j for _, j in pairs}
        amb = "ambiguous: two partners almost equally close, not paired"
        res["unmatched"] += [{"side": "ref", "kind": x["kind"], "t": x["t"],
                              "reason": amb if k in amb_r else f"no candidate cut within {MATCH_TOL_S} s",
                              "evidence": _ev(x["t"], x["t"], conf)} for k, x in enumerate(rb) if k not in mi]
        res["unmatched"] += [{"side": "cand", "kind": x["kind"], "t": x["t"],
                              "reason": amb if k in amb_c else f"no reference cut within {MATCH_TOL_S} s",
                              "evidence": _ev(x["t"], x["t"], conf)} for k, x in enumerate(cb) if k not in mj]

        if np is not None:
            rs = [s for s in _spans(rdiff, ri["fps"], np) if _in(s[0], a, b)]
            cs = [s for s in _spans(cdiff, ci["fps"], np) if _in(s[0] - off, a, b)]
            sp, _, _ = _match([s[0] for s in rs], [s[0] for s in cs], off)
            for i, j in sp:
                (r0, r1), (c0, c1) = rs[i], cs[j]
                res["phases"].append({
                    "ref": {"t_start": round(r0, 4), "t_end": round(r1, 4)},
                    "cand": {"t_start": round(c0, 4), "t_end": round(c1, 4)},
                    "start_delta_ms": round((c0 - off - r0) * 1000, 1),
                    "duration_delta_ms": round(((c1 - c0) - (r1 - r0)) * 1000, 1),
                    "evidence": _ev(r0, r1, conf)})
            mi, mj = {i for i, _ in sp}, {j for _, j in sp}
            res["unmatched"] += [{"side": "ref", "kind": "motion_span", "t": round(s[0], 4),
                                  "reason": "no candidate motion span starts nearby",
                                  "evidence": _ev(s[0], s[1], conf)}
                                 for k, s in enumerate(rs) if k not in mi]
            res["unmatched"] += [{"side": "cand", "kind": "motion_span", "t": round(s[0], 4),
                                  "reason": "no reference motion span starts nearby",
                                  "evidence": _ev(s[0], s[1], conf)}
                                 for k, s in enumerate(cs) if k not in mj]
        if conf <= 0.5:
            notes.append(f"alignment confidence {conf:.2f}: treat every delta as approximate")

    try:
        res["normalised"] = _render(ref, cand, off, a, b, slow, out_dir / "compare.mp4")
        res["video"] = "compare.mp4"
    except (OSError, subprocess.SubprocessError) as exc:
        notes.append(f"side-by-side video failed: {type(exc).__name__}")
    return res


def summary_md(res: dict) -> str:
    al = res["alignment"]
    lines = [f"# Comparison: {res['reference']['source']} vs {res['candidate']['source']}", ""]
    if al["offset_s"] == NOT_DETERMINED:
        lines.append("- alignment: _not determined_ (pass --offset)")
    else:
        lines.append(f"- alignment: {al['method']}, candidate = reference {al['offset_s']:+.3f} s "
                     f"(confidence {al['confidence']})")
    if res["shots"]:
        lines += ["", "| ref cut (s) | your cut (s) | delta (ms) |", "|---|---|---|"]
        lines += [f"| {s['ref']:.3f} | {s['cand']:.3f} | {s['delta_ms']:+.0f} |" for s in res["shots"]]
    if res["phases"]:
        lines += ["", "| ref motion (s) | yours (s) | start delta (ms) | duration delta (ms) |", "|---|---|---|---|"]
        lines += [f"| {p['ref']['t_start']:.3f}-{p['ref']['t_end']:.3f} | {p['cand']['t_start']:.3f}-"
                  f"{p['cand']['t_end']:.3f} | {p['start_delta_ms']:+.0f} | {p['duration_delta_ms']:+.0f} |"
                  for p in res["phases"]]
    if res["unmatched"]:
        lines += ["", "Not matched (reported, not paired):"]
        lines += [f"- {u['side']} {u['kind']} at {u['t']:.3f} s: {u['reason']}" for u in res["unmatched"]]
    if res["notes"]:
        lines += ["", "Notes:"] + [f"- {n}" for n in res["notes"]]
    if res.get("video"):
        lines += ["", f"Side by side, {res['slow']}x slower: {res['video']}"]
    return "\n".join(lines)
