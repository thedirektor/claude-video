# claude-video — fork expansion design (0.3.0 → 0.6.0)

Date: 2026-07-06
Repo: `thedirektor/claude-video` (fork of `bradautomates/claude-video`)
Dev workspace: `~/claude-video` (branch `feat/v030-report-upstream`)
Status: approved by user (all four phases), pending spec review

## Context

- Fork is 21 commits ahead / 5 behind upstream. Fork adds (+3,319 lines): OpenRouter
  backend (vision + audio, model chain qwen3-asr-flash → voxtral-mini → groq),
  Gemini native-video backend (Files API upload + YouTube URL server-side),
  scene detection (`scenes.py`), OCR (`ocr.py`, default `spa+eng`),
  speech-window-weighted frame sampling (`speech.py`), local faster-whisper
  (`whisper_local.py`), AssemblyAI diarization (`whisper_assemblyai.py`),
  CI smoke workflow, argv hardening, Windows UTF-8/cp1252 fixes.
- Upstream 0.2.0 (the 5 missing commits) restructured the repo into a
  self-contained Agent Skills package: everything moved to `skills/watch/`,
  new `config.py`, rewritten 756-line `frames.py` with frame dedup, Whisper
  auto-chunking in `whisper.py`, a 9-file pytest suite under `tests/`, and a
  `WATCH_DETAIL` fallback fix. This layout also fixes Codex installs.
- Deployment topology (three clones of the fork on cort3x.me):
  1. `~/claude-video` — dev workspace (this repo). Work happens here only.
  2. `~/.claude/skills/watch` — live Claude Code skill (git clone, at `b5f6096`).
  3. `/opt/watch/claude-video` — bind-mounted **read-only** into the watch
     service container (currently at `27b839f`, one commit stale). Never
     branch-switch or dev here; a checkout changes what the container sees live.
- Ecosystem survey: 590 forks scanned, ~30 with real commits. Reference forks:
  danielfrey63 (resume/idempotency, pluggable STT), sciencemj (bundles,
  scene→slide, transcript cache), joweiser (chunking, --json), JoseBallestas
  (Whisper resilience), deebee37 (/edit), lutzleonhardt (transcript-first).
  Upstream open PRs #46/#42/#27/#21 (YouTube 403 family), #12/#26/#30
  (non-English subs), #10 (Whisper resilience), #40 + issue #37 (high-density).

## Goals

1. Re-base the fork onto upstream 0.2.0 so future syncs are cheap and the fork
   gains the pytest suite, frame dedup, and Whisper auto-chunking.
2. Make daily use reliable: YouTube 403s, Spanish captions, Whisper failures,
   crash resume.
3. Serve the owner's primary use case — debugging projects from screen
   recordings — with high-density frames, reusable bundles, and JSON output.
4. Add a colorist-oriented `/edit` command (ffmpeg operations, no GUI).

## Non-goals

- No GUI / interactive menu runners (deebee37's tkinter + easy_edit are out).
- No new third-party multimodal providers (TwelveLabs, Nemotron): the existing
  `--backend gemini` already covers native video understanding.
- No new pip dependencies in core paths — keep the upstream pure-stdlib policy;
  optional extras (faster-whisper, pytesseract) stay lazy-imported.
- No upstream PR submissions this round (personal fork first; cherry-pick
  upstreamable pieces later if desired).

## Phase 1 — Re-port onto upstream 0.2.0 (version 0.3.0)

**Strategy:** start from `upstream/main` and re-apply fork features as clean,
one-module-per-commit changes, rather than merging. The fork's history stays
on `main`; the new branch `feat/v030-report-upstream` is reset to
`upstream/main` and becomes the next `main` when done.

**Layout adoption (from upstream, unchanged):** `skills/watch/SKILL.md`,
`skills/watch/scripts/`, `config.py`, `tests/`, `.skillignore`, `AGENTS.md`,
`dev-sync.sh`, `.agents/plugins/marketplace.json`.

**Port order (one commit each, adapted to new layout + `config.py`):**
1. Windows UTF-8/cp1252 + argv hardening deltas that upstream still lacks
   (verify first — upstream may have absorbed some via #2/#4).
2. `download.py` deltas (URL handling additions).
3. `whisper.py` deltas — reconcile fork's Groq-fallback chain with upstream's
   new auto-chunking version; fork's chain logic layers on top.
4. `whisper_local.py` (faster-whisper) + `whisper_assemblyai.py` (diarization).
5. `scenes.py` + `ocr.py` + `speech.py`, re-wired into upstream's rewritten
   `frames.py` (dedup preserved; scene/speech sampling feeds the same
   candidate→dedup funnel).
6. `openrouter.py` backend + `gemini.py` backend + `watch.py` backend dispatch
   (`--backend claude|gemini|openrouter` and model flags).
7. SKILL.md content merge: upstream's new skill doc + fork's backend/OCR/
   diarization sections. `commands/watch.md` equivalent under new layout.
8. CI: adapt fork's `smoke.yml` to new paths; keep upstream `release.yml`.
9. Rebrand pass (thedirektor URLs) + CHANGELOG + bump `0.3.0`.

**Tests:** upstream suite must pass untouched. Add fork-module tests:
`test_scenes.py` (timestamp parsing, threshold math), `test_speech.py`
(window computation from segments), `test_openrouter.py` (payload shaping,
model-chain fallback order — mocked HTTP), `test_ocr.py` (confidence filter,
missing-binary graceful path).

**Verification (CLAUDE.md rule — evidence before assertions):**
- `pytest` green.
- Real e2e: `/watch` on one YouTube URL and one local file; confirm frames +
  transcript + report. `--backend gemini` and `--backend openrouter` each get
  one live run.
- CHECKPOINT with user before: push to origin, fast-forward `main`,
  `git pull` in `~/.claude/skills/watch`, `git pull` in
  `/opt/watch/claude-video` + `docker compose restart watch` (config-only;
  image not rebuilt because the mount is the code).

## Phase 2 — Reliability pack (version 0.4.0)

1. **YouTube 403/SABR family** (sources: PRs #46, #42, #27, #21):
   - Default `--extractor-args youtube:player_client=default,android,mweb`
     fallback chain in `download.py`.
   - Anti-bot flags (retries, sleep-interval jitter) on by default.
   - New `--cookies-from-browser <browser>` and `--cookies <file>` passthrough
     flags for login-walled sources. Off by default.
2. **Subtitle languages** (sources: PRs #12, #26, #30): replace hardcoded
   `en,en-US,en-GB,en-orig` with default
   `es,es-419,es-ES,en,en-US,en-GB,*-orig` plus a `--sub-lang <csv>` override.
   Auto-subs fallback (`--write-auto-subs`) when no manual track matches.
3. **Whisper resilience** (sources: PR #10, JoseBallestas fork):
   - Chunk-level retry caps (no infinite 5xx retry burning hourly quota).
   - Content-addressed chunk-transcript cache at `~/.cache/watch/transcripts/`
     keyed by sha256(audio-chunk bytes + model); re-runs skip uploaded chunks.
   - When `--start/--end` given, extract and transcribe only that audio window,
     not the full track.
4. **Resume/idempotency** (source: danielfrey63): persist per-stage
   intermediates (`download.json`, `transcript.json`, `frames.done`) in the
   work dir; a re-run with the same source + params resumes at the first
   incomplete stage. `--fresh` forces full re-run.

Acceptance: unplugging network mid-transcription then re-running completes
without re-uploading finished chunks; a Spanish-only YouTube video produces a
caption-based transcript with zero Whisper spend.

## Phase 3 — Debug-oriented features (version 0.5.0)

1. **High-density mode** (sources: issue #37, PR #40): `--density normal|high|max`.
   `high` raises the fps cap to 8 and the total frame cap to 200 on any source
   (on long sources that budget spreads thin — the tool warns and suggests a
   `--start/--end` range); `max` extracts every frame within an explicit
   `--start/--end` window ≤ 30 s (guard: refuse `max` without a range).
   Dedup stays on to absorb the burst.
2. **Bundle save/load** (source: sciencemj): `--save-bundle <name>` writes
   `~/.cache/watch/bundles/<name>/` containing `report.md`, `transcript.json`,
   `frames/`, `meta.json` (source URL, params, versions). New skill commands
   `/watch-save` and `/watch-load` documented in SKILL.md; `/watch-load` re-emits
   the report and frame paths so Claude can answer new questions with zero
   re-processing.
3. **`--json` output** (source: joweiser): machine-readable envelope
   `{source, meta, transcript: [...], frames: [{t, path, ocr}], report_path}`
   on stdout for Hermes/agent integration. Human report unchanged by default.
4. **Scene→slide grouping** (source: sciencemj): `--slides` groups scene cuts
   into slide spans, picks one representative frame per slide, groups the
   transcript per slide in the report. `--pdf` additionally emits
   `slides.pdf` (pure-stdlib JPEG→PDF writer).

## Phase 4 — `/edit` colorist command (version 0.6.0)

New `skills/edit/` skill + `skills/edit/scripts/edit.py` (source: deebee37,
scoped subset). Operations, all ffmpeg-based, chainable in one invocation:

- `--lut <file.cube>` and `--look <preset>` (a few built-in curves: warm, cool,
  bleach, teal-orange) — colorist core.
- `--letterbox <ratio>`, `--fps <n>`, `--trim <in> <out>` (stream copy) and
  `--trim-precise` (frame-accurate re-encode).
- `--blur <W:H:X:Y>` rectangular privacy blur.
- `--watermark-text <s>` / `--watermark-image <path>`.
- `--normalize-audio` (EBU R128 loudnorm), `--denoise`, `--sharpen`,
  `--stabilize` (two-pass vidstab when the ffmpeg build has it; detect and
  fail with a clear message otherwise).
- Export quality flags (`--crf`, `--preset`), `--strip-metadata`.

Safety: never overwrite the input; output defaults to `<name>.edit.<ext>`.
Print the composed ffmpeg command before running. Smoke harness
`tests/smoke_edit.py` (generated test clip via ffmpeg `testsrc`), wired into CI.

## Cross-cutting

- Pure stdlib in core; heavy deps stay optional + lazy.
- Windows: UTF-8 reconfigure on every entrypoint (fork convention).
- Secrets: `~/.config/watch/.env` only; `.env` gitignored (fork commit b5f6096).
- Each phase: own branch → tests + e2e verification → user checkpoint →
  push → deploy to the two live clones → Qdrant memory_store of key findings.
- Version bumps: 0.3.0 / 0.4.0 / 0.5.0 / 0.6.0 in `plugin.json`,
  `marketplace.json`, CHANGELOG.

## Error handling principles

- Every network path: bounded retries with caps, then a one-line actionable
  error (which backend failed, which env var or flag fixes it).
- Optional-dependency paths (tesseract, faster-whisper, vidstab): detect,
  degrade gracefully, say what was skipped and why.
- Late-stage crashes must not destroy earlier artifacts (danielfrey63's
  "report files survive late-stage crashes" behavior).

## Testing strategy

- Unit: pytest, pure functions mocked-HTTP only, runs offline in CI.
- Smoke: existing `smoke.yml` (real yt-dlp + ffmpeg on a tiny public video)
  adapted to new layout; `smoke_edit.py` added in Phase 4.
- E2E (manual, per phase, per CLAUDE.md): real invocations on the box before
  any deploy is declared done.

## Open items deferred (documented, not scheduled)

- Upstreaming selected pieces as PRs (subs languages, 403 fixes are welcome
  upstream per open PR traffic).
- Transcript-first default mode (lutzleonhardt) — revisit after Phase 3 usage.
- danielfrey63's pluggable STT abstraction (`stt.py`) — only if backend count
  keeps growing.
