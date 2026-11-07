# Breakdown evaluation

`watch.py evaluate <labels.json>` measures `breakdown` against hand labels: cut recall / precision / F1
(one-to-one matching within `tolerance_frames`, default 2) and the per-shot production method
(duration-weighted; an undetermined shot is never counted as right). It runs on the media files
directly: no video memory, no network, no paid call. `--json` prints the full report.

Media is not committed (copyright): keep your clips next to your labels file, paths are resolved
relative to it. Format: see `labels.example.json`.

## Labelling

- `cuts`: the time of the first frame of each new shot, frame-accurate (step through the cut in a
  player). Put boundaries you are unsure of in `cuts_uncertain`: found or not, they are not scored.
- `production`: `[start, end, label]` spans, label one of `still`, `still_moved`, `synthetic`,
  `camera`, `ai_generated` (the file declares it), `ai_video` (generated, not declared), or a list
  when several are acceptable (a screen recording of a UI: `["synthetic", "still", "still_moved"]`).

## Seed results (2026-11-07)

Six clips: a phone TikTok (12 hand-labelled cuts), its AI remake, a 3D explainer, a code-UI promo,
a 131-shot music video and a one-shot clip. Cuts: recall 100%, precision 100%. Production: 24% of
410 s determined, 99% right when determined, undetermined shots lean right 93%.

**These numbers are not independent**: the thresholds were set on the same clips. Treat them as a
regression floor; only a set labelled after the thresholds were fixed measures accuracy.
