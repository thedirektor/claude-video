"""Measure the breakdown against hand labels: cut recall / precision / F1 and production accuracy.

Labels file (JSON; YAML too when PyYAML is installed), paths relative to the labels file::

    {"tolerance_frames": 2,
     "videos": [
       {"id": "phone_talking_head", "path": "media/v11.mp4",
        "cuts": [0.73, 1.50, 2.20],          # frame-accurate cut times, seconds
        "cuts_uncertain": [3.27],             # found or not, never scored
        "production": [[0.0, 1.38, "synthetic"], [1.38, 3.92, ["ai_video", "camera"]]]}
     ]}

``production`` labels use the breakdown vocabulary (``still``, ``still_moved``, ``synthetic``,
``camera``, ``ai_generated``) plus ``ai_video`` (generated footage that does not declare itself; the
breakdown can at most lean towards it). A list means any of them is right. Only labelled spans are
scored, weighted by duration, and an undetermined shot is never counted as right: the report gives the
determined share, the accuracy on what was determined, and how often the leaning (top candidate) of
undetermined shots was right. Runs on the media files directly: no video memory, no network.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from . import media, production, provenance, shots as shots_mod

NOT_DETERMINED = production.NOT_DETERMINED
# what a declared / measured label satisfies: AI that declared itself is also "ai_video"
SATISFIES = {"ai_generated": {"ai_generated", "ai_video"}, "camera": {"camera"}, "still": {"still"},
             "still_moved": {"still_moved"}, "synthetic": {"synthetic"}, "ai_video": {"ai_video"}}


class LabelsError(ValueError):
    pass


def load_labels(path: Path) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # noqa: PLC0415
        except ImportError as exc:
            raise LabelsError("YAML labels need PyYAML (pip install pyyaml), or write the file as JSON") from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict) or not isinstance(data.get("videos"), list):
        raise LabelsError("labels must be an object with a 'videos' list")
    for v in data["videos"]:
        if not isinstance(v, dict) or not v.get("path"):
            raise LabelsError("every video needs a 'path'")
    return data


def score_cuts(detected: list[float], truth: list[float], uncertain: list[float], tol: float) -> dict:
    """Greedy one-to-one matching within ``tol`` seconds; detections at uncertain times are not scored."""
    used: set[int] = set()
    hits = []
    for g in sorted(truth):
        best = None
        for k, d in enumerate(detected):
            if k not in used and abs(d - g) <= tol and (best is None or abs(d - g) < abs(detected[best] - g)):
                best = k
        if best is not None:
            used.add(best)
            hits.append(g)
    unscored = {k for k, d in enumerate(detected) if k not in used and any(abs(d - u) <= tol for u in uncertain)}
    fp = [round(d, 3) for k, d in enumerate(detected) if k not in used and k not in unscored]
    scored = len(detected) - len(unscored)
    rec = len(hits) / len(truth) if truth else None
    prec = (scored - len(fp)) / scored if scored else None
    f1 = 2 * rec * prec / (rec + prec) if rec and prec else (0.0 if rec is not None and prec is not None else None)
    return {"truth": len(truth), "detected": len(detected), "hits": len(hits), "false_positives": fp,
            "missed": [g for g in sorted(truth) if g not in hits], "recall": rec, "precision": prec, "f1": f1}


def _accepts(want, got: str) -> bool:
    wants = set(want) if isinstance(want, (list, tuple)) else {want}
    return bool(SATISFIES.get(got, {got}) & wants)


def score_production(shot_list: list[dict], spans: list) -> dict:
    lab_s = det_s = right_s = und_s = lean_right_s = 0.0
    confusion: dict[str, dict[str, float]] = {}
    for sp in spans:
        t0, t1, want = float(sp[0]), float(sp[1]), sp[2]
        key = want if isinstance(want, str) else "|".join(want)
        for s in shot_list:
            ov = min(t1, s["end"]) - max(t0, s["start"])
            if ov <= 0:
                continue
            p = s.get("production") or {}
            got = p.get("label", NOT_DETERMINED)
            lab_s += ov
            confusion.setdefault(key, {})
            confusion[key][got] = confusion[key].get(got, 0.0) + ov
            if got == NOT_DETERMINED:
                und_s += ov
                top = (p.get("candidates") or [{}])[0].get("label")
                if top and _accepts(want, top):
                    lean_right_s += ov
            else:
                det_s += ov
                if _accepts(want, got):
                    right_s += ov
    r = lambda a, b: round(a / b, 4) if b else None  # noqa: E731
    return {"labelled_s": round(lab_s, 3), "determined_share": r(det_s, lab_s),
            "accuracy_determined": r(right_s, det_s), "accuracy_overall": r(right_s, lab_s),
            "undetermined_lean_right": r(lean_right_s, und_s),
            "confusion_s": {k: {g: round(v, 3) for g, v in sorted(c.items(), key=lambda kv: -kv[1])}
                            for k, c in confusion.items()}}


def analyse(path: str) -> dict:
    """Shots + per-shot production of one media file (the breakdown's own measurement path)."""
    info = media.probe(Path(path))
    if not info or not info.get("fps"):
        raise LabelsError(f"cannot read video: {path}")
    series = shots_mod.frame_series(path, info["fps"])
    shot_list = shots_mod.detect_shots(series, info["duration"])
    prov = provenance.probe(path)
    for s in shot_list:
        s["production"] = production.shot_production(path, s, info["fps"], prov)
    return {"fps": info["fps"], "duration": info["duration"], "shots": shot_list}


def run(labels_path: Path, analyse_fn=analyse) -> dict:
    labels_path = Path(labels_path)
    data = load_labels(labels_path)
    tol_frames = float(data.get("tolerance_frames", 2))
    per = []
    agg_cut = {"truth": 0, "hits": 0, "detected_scored": 0, "fp": 0}
    agg_prod = {"lab": 0.0, "det": 0.0, "right": 0.0, "und": 0.0, "lean": 0.0}
    for v in data["videos"]:
        p = Path(v["path"])
        p = p if p.is_absolute() else (labels_path.parent / p)
        entry: dict = {"id": v.get("id") or p.stem, "path": str(p)}
        try:
            a = analyse_fn(str(p))
        except Exception as exc:  # noqa: BLE001 - one bad file never stops the run
            entry["error"] = f"{type(exc).__name__}: {exc}"
            per.append(entry)
            continue
        entry["shots"] = len(a["shots"])
        if "cuts" in v:
            det = [s["boundary"]["t"] for s in a["shots"][1:] if s.get("boundary")]
            c = score_cuts(det, [float(x) for x in v["cuts"]], [float(x) for x in v.get("cuts_uncertain") or []],
                           tol_frames / a["fps"])
            entry["cuts"] = c
            agg_cut["truth"] += c["truth"]
            agg_cut["hits"] += c["hits"]
            agg_cut["fp"] += len(c["false_positives"])
            agg_cut["detected_scored"] += c["hits"] + len(c["false_positives"])
        if v.get("production"):
            pr = score_production(a["shots"], v["production"])
            entry["production"] = pr
            lab = pr["labelled_s"]
            agg_prod["lab"] += lab
            agg_prod["det"] += lab * (pr["determined_share"] or 0)
            agg_prod["right"] += lab * (pr["accuracy_overall"] or 0)
            agg_prod["und"] += lab * (1 - (pr["determined_share"] or 0))
            agg_prod["lean"] += lab * (1 - (pr["determined_share"] or 0)) * (pr["undetermined_lean_right"] or 0)
        per.append(entry)
    rec = agg_cut["hits"] / agg_cut["truth"] if agg_cut["truth"] else None
    prec = agg_cut["hits"] / agg_cut["detected_scored"] if agg_cut["detected_scored"] else None
    r = lambda a, b: round(a / b, 4) if b else None  # noqa: E731
    return {"labels": str(labels_path), "tolerance_frames": tol_frames, "videos": per,
            "summary": {"cuts": {"truth": agg_cut["truth"], "recall": r(agg_cut["hits"], agg_cut["truth"]),
                                 "precision": prec and round(prec, 4),
                                 "f1": round(2 * rec * prec / (rec + prec), 4) if rec and prec else None,
                                 "false_positives": agg_cut["fp"]},
                        "production": {"labelled_s": round(agg_prod["lab"], 3),
                                       "determined_share": r(agg_prod["det"], agg_prod["lab"]),
                                       "accuracy_determined": r(agg_prod["right"], agg_prod["det"]),
                                       "accuracy_overall": r(agg_prod["right"], agg_prod["lab"]),
                                       "undetermined_lean_right": r(agg_prod["lean"], agg_prod["und"])}}}


def summary_md(res: dict) -> str:
    pct = lambda x: "n/a" if x is None else f"{x:.0%}"  # noqa: E731
    s = res["summary"]
    out = [f"# Breakdown evaluation: {Path(res['labels']).name}", "",
           f"Cut tolerance: {res['tolerance_frames']:g} frames.", "", "| video | shots | cut recall | cut precision "
           "| missed | false positives | production determined | right when determined | undetermined lean right |",
           "|---|---|---|---|---|---|---|---|---|"]
    for v in res["videos"]:
        if v.get("error"):
            out.append(f"| {v['id']} | error: {v['error']} | | | | | | | |")
            continue
        c, p = v.get("cuts") or {}, v.get("production") or {}
        out.append(f"| {v['id']} | {v['shots']} | {pct(c.get('recall'))} | {pct(c.get('precision'))} | "
                   f"{', '.join(f'{x:g}' for x in c.get('missed', [])) or '-'} | "
                   f"{', '.join(f'{x:g}' for x in c.get('false_positives', [])) or '-'} | "
                   f"{pct(p.get('determined_share'))} | {pct(p.get('accuracy_determined'))} | "
                   f"{pct(p.get('undetermined_lean_right'))} |")
    c, p = s["cuts"], s["production"]
    f1 = "n/a" if c["f1"] is None else f"{c['f1']:.3f}"
    out += ["", f"**Cuts:** recall {pct(c['recall'])}, precision {pct(c['precision'])}, F1 {f1} "
                f"over {c['truth']} labelled cuts.",
            f"**Production:** {pct(p['determined_share'])} of {p['labelled_s']:g} s determined, "
            f"{pct(p['accuracy_determined'])} right when determined, undetermined shots lean right "
            f"{pct(p['undetermined_lean_right'])}."]
    return "\n".join(out)


def main(argv: list[str]) -> int:
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(prog="watch evaluate", description="Score the breakdown against hand labels.")
    ap.add_argument("labels", help="labels file (.json, or .yaml with PyYAML)")
    ap.add_argument("--json", action="store_true", help="print the full JSON report")
    a = ap.parse_args(argv)
    try:
        res = run(Path(a.labels))
    except (OSError, ValueError) as exc:
        print(f"[watch] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(res, ensure_ascii=False, indent=1) if a.json else summary_md(res))
    return 0
