# Changelog

All notable changes to `/watch` are documented here.

## [0.6.0] — 2026-10-23

Platform: the plugin side of durable jobs, costs, batch, central memory and integrations (service-side features — job queue, webhooks, subscriptions, `/runs/import`, n8n/Hermes/ingest wiring — live in the private service repo).

### Added
- **Cost plan + ceiling** (`cost.py`): `--plan`, `WATCH_COST_CEILING_USD` (default 1.00) enforced before any paid call, `--allow-cost`, `WATCH_PRICES` overrides; metadata probe (falls back to the proxy when blocked).
- **Actual spend** (`spend.py`): Gemini / OpenRouter usage tokens, Whisper audio seconds, TranscriptAPI credits, LLM chapter titles → `work/spend.json`.
- Metadata/captions fetch retries via `WATCH_YTDLP_FALLBACK_PROXY` when blocked; `run_ytdlp(capture=True)`.
- **Central memory push** (`memory/bundle.py`, `push.py`): run bundles, `WATCH_REMOTE_URL` + `WATCH_REMOTE_TOKEN`, outbox retry, `watch.py push [--all|--retry]`; idempotent import (`origin=import`).
- **Local MCP server** (`mcp_stdio.py`): `watch.py mcp` over stdio, no dependencies — for Claude Desktop on Windows/macOS/Linux.
- **Visual search** (`memory/visual.py`): `search --visual` — Immich CLIP over watch albums, mapped to (video_key, t); also in the local MCP.
- **Batch** (`playlist.py`): `watch.py batch <playlist|channel|file>` with `--limit`, `--force`, skipping indexed videos.

## [0.5.3] — 2026-10-23

Outputs from the video memory.

### Added
- **Chapters + cited notes** (`memory/extract.py`): deterministic boundaries (pauses, speaker changes, scene cuts) snapped to an even split, first-sentence titles (caption credit lines skipped), optional LLM titles; `notes(kind=chapter)` + `notes.md` with `(MM:SS)` from source rows.
- **Exports** (`exports.py`): `docx`/`pdf`/`pptx`/`xlsx` via the ArtCraft CLIs, merged searchable PDF (deck at 96 dpi + text layer), offline `viewer.html` (≤ 25 MB inlined frames), `json`, contact sheet (ffmpeg). Built from the index + store; `--exports`, `WATCH_EXPORTS`, `watch.py export`.
- **Library routing**: exports → Nextcloud folder, merged PDF → Paperless (replaces report.md), contact sheet → Immich album (`library.save_exports`).
- **Fallback proxy** (`WATCH_YTDLP_FALLBACK_PROXY`, `WATCH_FALLBACK_MAX_MB`): a blocked download is retried once through a residential HTTP proxy (≤480p, size cap, 20 s socket timeout); `doctor` probes it. Storyboards remain the next fallback.

## [0.5.2] — 2026-10-23

Acquisition resilience without accounts or paid proxies.

### Added
- **Storyboard frames** (`storyboard.py`): a blocked YouTube media download on a non-transcript run falls back to the public storyboard (`sb0`…`sb3`), split into tiles with ffmpeg `untile`, timed from the format's fps, honouring `--start/--end` and the frame cap; indexed as frame kind `storyboard`. No storyboard + a transcript → the run completes frameless instead of exiting.
- **yt-dlp self-heal** (`download.py`): breakage classification (extractor errors, not 403/bot walls), one upgrade per 6 h (`WATCH_YTDLP_PYLIB` target or `pip --user`; `WATCH_YTDLP_SELF_HEAL=0` off) and a single retry, for both downloads and caption fetches.
- **Gemini visual log** (`gemini.visual_log`, `memory/escalate.visual_log`): low-confidence answers on media-less YouTube videos get Gemini's timestamped on-screen log once per video, stored as `notes(kind=visual)`; `WATCH_VISUAL_LOG=0` off.
- **`watch.py doctor [--fix] [--json] [--offline]`**: dependency, canary, model, storage, endpoint and key-presence checks; safe repairs.

### Changed
- Personal hosts are no longer built-in defaults: `IMMICH_BASE_URL`, `NEXTCLOUD_URL`, `NEXTCLOUD_USER`, `PAPERLESS_URL` and `WATCH_LLM_BASE_URL` must be configured (a target without them is skipped).

### Not done (closed no-go)
- PO-token provider, Deno/EJS and self-hosted cobalt could not fetch most YouTube media from a datacenter IP without a logged-in account; the paid residential-proxy fallback was dropped (a self-hosted one arrived right after, see Unreleased).

## [0.5.1] — 2026-10-23

Grounded answers over the video memory.

### Added
- **`ask --answer`** (`memory/answer.py`): confidence (`high|medium|low` from top-hit strength — raw cosine or question-word coverage —, margin and cross-modal agreement), refusal floor (`No encontré esto en el video.` + 3 nearest moments, no LLM call), LLM answer over numbered evidence with a citation guard (only `[E#]` ids survive, rendered with the evidence row's timestamp; every other time string is stripped; no valid citation → not found).
- **Model-free escalation** (`memory/escalate.py`): on low confidence and stored media, 1 fps re-sample at 2048 px + re-OCR around the best hits; new text written back as `escalated` frames/OCR, evidence_version bumped; each time attempted once; `WATCH_ESCALATE_BUDGET_S` (45).
- **Answer cache** (schema v2 `answers`): exact normalised question, then cosine ≥ 0.92, per (video_key, evidence_version); `watch.py stats` reports answers, cache hits, tokens used/saved.
- **LLM chain** (`memory/llm.py`): `WATCH_LLM_CHAIN` = headroom (needs `WATCH_LLM_API_KEY`) → Gemini → OpenRouter; evidence-only when none answers. Never raises.
- Evidence items carry `cos` (raw cosine) for confidence.

## [0.5.0] — 2026-10-23

Video memory: every run is indexed, stored and queryable.

### Added
- **Identity.** `identity.video_key` (`yt:<id>`, `<extractor>:<id>` from info-json, `url:<hash>`, `file:<hash>`) and Windows-safe, collision-safe `store_dirname`; IPv6/default-port URL canonicalisation.
- **`memory/` package.** SQLite index (WAL, schema v1) with runs, merge rules (better transcript source wins, frames/OCR unioned by pHash/timestamp, notes append) and FTS; persistent store with LRU retention (`WATCH_STORE_MAX_GB`, media cap) and Qdrant inbox hand-off; speaker-aware transcript chunking; optional fastembed embeddings (multilingual MiniLM, 384-d, lazy import, FTS-only fallback); hybrid retrieval (0.45 BM25 + 0.55 cosine, OCR duplicate collapse); `resolve`/`ask`/`moment`/`list` evidence packs that cite only stored timestamps; `writer.remember`.
- **Every backend writes memory.** Claude, Gemini and OpenRouter runs are indexed; Gemini indexes yt-dlp captions (or Whisper only if `--whisper` is given); OpenRouter indexes the transcript the run already made (captions, else its audio model); both also index the backend answer as a note; neither calls TranscriptAPI or makes an extra paid transcript call. Never fails a run.
- **Flags.** `--save` (archive gemini/openrouter/ingest-transcript runs), `--no-index`, `--origin skill|service|ingest|import|backfill`. `--no-save` still indexes.
- **CLI subcommands.** `watch.py search | ask | moment | list | index` (`--backfill-paperless`, `--rebuild KEY`, `--forget KEY`, `--embed-missing`), all with `--json`.
- **`/watch-recall` searches the index first**, then Paperless, then Nextcloud.
- **Immich links** in archived report frame lines (t_ms-keyed refs); Paperless task id resolved to document id.
- **Config.** `WATCH_CONFIG_FILE` override; new settings `WATCH_INDEX_DB`, `WATCH_STORE_DIR`, `WATCH_MODEL_DIR`, `WATCH_EMBEDDINGS`, `WATCH_QDRANT_INBOX`, `WATCH_STORE_MAX_GB` (20), `WATCH_STORE_MEDIA_MAX_MB` (300), `WATCH_STORE_KEEP_MEDIA`.
- **Docs.** SKILL.md "Video memory" section with the rule: cite only evidence timestamps, never invent one.

### Fixed / minors closed
- Generic-extractor video keys and store dirname collisions; range-run clipping and empty-transcript guard in the index; boundary-correct chunk folding; hostile-query safety and embedder-failure fallback; race-safe `embed_pending`; `moment` rejects bad timestamps cleanly; backfill keeps report ranges and full source paths; atomic rebuild; CLI hardening; subprocess tests isolated from the real config.
- BACKLOG: "Report frame lines don't link to Immich" done; bundle save/load superseded by the store.

## [0.4.7] — 2026-10-23

### Added
- **YouTube transcript fallback via Gemini URL.** When TranscriptAPI and yt-dlp
  captions both miss, `gemini.transcribe_youtube` transcribes the video straight
  from its URL (reported as `via gemini (model)`). Opt out with
  `--no-gemini-transcript` or `WATCH_GEMINI_TRANSCRIPT=0`. Why: datacenter IP
  blocks on yt-dlp and TranscriptAPI credit exhaustion.

### Fixed (final review)
- **Timestamp plausibility guard.** `transcribe_youtube` now returns
  `(segments, model, approximate)`; > 250 words/min over the first→last span,
  or a last start past the known duration × 1.05, logs one stderr line and
  labels the source `gemini (model, approximate timestamps)` with a report note
  "Timestamps are model-generated and approximate". `watch.py` passes the
  info-json duration and the canonical URL.
- **Deterministic request.** `temperature=0`, `max_output_tokens=65536`
  (`gemini-3.1-flash-lite`'s `output_token_limit`). Media resolution stays
  `LOW`: live A/B on XUqrvbsTfck (talk ends ≈19:40) gave identical results for
  LOW and the API default — 156 segments, last start 19:40, 2953 words,
  154.7 wpm — so the cheaper LOW is kept.
- **Parser** accepts `- ` / `* ` bullets, `**` bold around the bracket and
  `[start - end]` ranges; an unparseable non-empty reply logs its first 80 chars.
- `--no-whisper` help notes Gemini still runs for YouTube.
- **Service** (`/opt/watch`): `transcribe` returns the full `transcript_source`
  label (nested parens no longer truncated); `watch`/`transcribe` MCP tools take
  `no_gemini_transcript`; `transcribe` docstring lists the real chain.

## [0.4.6] — 2026-10-22

### Added
- **Canonical source identity.** `identity.canonical_source` collapses URL
  spellings (e.g. the several YouTube forms) to one stable identity, so the same
  video always maps to the same job/video id.
- **Structured library refs + `library.json`.** `_post_document` now returns
  `(ok, task_id)`; `upload_to_immich` returns `album_id` + `assets`;
  `upload_to_paperless` returns `task_ids`; `save_artifacts(refs=…)` accepts and
  records them. `watch.py` writes `<work>/library.json` so downstream tools can
  resolve exactly which Immich assets a run produced and which Paperless
  consume **task ids** it queued (task ids, not document ids — Paperless assigns
  the document id asynchronously).
- **Service** (`/opt/watch`):
  - MCP report strips frame lines that carry OCR text and exposes `frames[].ocr`.
  - `transcribe` parses diarized segments (`speaker`, `end`).
  - `watch_frames` `resolution`: omitted = frames served exactly as extracted
    (no re-encode; text frames stay at the 1024 px hi-res width); an explicit
    value downscales to `min(resolution, 1024)`, never upscales.
  - REST `options` pass `no_save`, `timestamps`, `sub_lang`, `fresh`,
    `no_dedup`, `no_transcriptapi`; the MCP `watch` tool gained only `no_save`,
    `timestamps` and `max_frames`. `diarize=false` now reaches `watch.py` as
    `--no-diarize`.
  - `POST /analyze` reuses queued/processing/done jobs (not failed) like MCP,
    with YouTube URL spellings collapsed to one identity; `fresh=true` never
    reuses.
  - REST `sections.Transcript` no longer carries the plugin's work-dir footer
    (`markdown` stays raw).
  - Job results carry `library` refs; the service version is read from the plugin.
- **ingest**: MCP `extract` exposes watch options (`detail`, `whisper`,
  `diarize`, `start`, `end`, `no_ocr`, `watch_backend`, `watch_model`); ingest
  forwards `diarize=false` to watch (previously dropped), and a rejected submit
  surfaces watch's error detail.

## [0.4.5] — 2026-07-07

### Added
- **`/watch-recall <topic-or-title>` command.** Retrieve a previously saved
  `/watch` report so a later session can read, see, or learn from a video it
  never watched — the read side of the library storage added in 0.4.3/0.4.4.
  New skill `skills/watch-recall/` + `skills/watch/scripts/recall.py`. Retrieval
  order: **Paperless-ngx full-text search** (`GET /api/documents/?query=…` →
  fetch each match's full `content`) → **Nextcloud folder-title match** fallback
  (lists `Watch/`, matches the query against folder names, reads `report.md`) —
  so recall still works when Paperless is unconfigured or its search index is
  stale. Every match points at the original video URL, the Paperless doc +
  Nextcloud path, and the **Immich album** (pull frames only when visual detail
  is needed). `--full` prints each report's whole text; `--limit N` caps
  matches. Best-effort: `recall()` never raises. Reuses `library.py`'s config +
  Nextcloud/SSL helpers; read-only (GET/PROPFIND, never writes).

## [0.4.4] — 2026-07-07

### Added
- **Paperless-ngx as a third library-storage target (searchable index).**
  Every `/watch` run now also uploads `report.md` (and any other non-media
  artifact) to a Paperless-ngx instance as a `watch`-tagged document, additive
  with the Nextcloud copy. Nextcloud stores the report; Paperless full-text
  *indexes* it, so a no-vision agent can answer "did I already watch something
  about X?" with a single `GET /api/documents/?query=X` across every report
  ever saved. Configured via `PAPERLESS_URL` (default
  `https://paperless.example.com`) + `PAPERLESS_TOKEN` in `~/.config/watch/.env`;
  needs the Paperless Tika/Gotenberg pipeline to ingest markdown as text.
  Best-effort and never fatal, exactly like Immich/Nextcloud — a missing token
  skips only Paperless, and `--no-save` skips all three. `setup.py --json` now
  reports `paperless` under `library_targets`.
- The `watch` tag is auto-created in Paperless on first upload if absent.

### Notes
- **Re-run dedup relies on Paperless's own content-hash dedup, not delete.** An
  identical `report.md` is auto-rejected by Paperless as a duplicate (the
  original survives). The uploader deliberately never deletes prior docs:
  Paperless keeps deleted documents (and their content hash) in a 30-day trash,
  so a delete-then-reupload of identical bytes would be rejected as a duplicate
  and leave *zero* active docs. A materially changed `report.md` creates a
  second document with the same title.

## [0.4.3] — 2026-07-07

### Added
- **Library storage (Immich + Nextcloud).** Every `/watch` run now saves its
  own artifacts after the report is printed, unless `--no-save` is passed.
  Artifacts are split by who consumes them: the downloaded video + a capped
  set of representative frames go to an **Immich** album (`<title>
  (<video-id>)`), while `report.md` (the full transcript + OCR + frame index)
  goes to a **Nextcloud** folder (`Watch/<title> (<video-id>)/`) — so a
  no-vision agent can later learn from a video it never watched by reading
  `report.md`. New pure-stdlib `library.py`: content-type router, a
  representative-frame selector (`WATCH_SAVE_FRAME_CAP`, default 20, `0` =
  all; priority OCR-significant → transcript-cue → scene-representative →
  even-spacing fill), an Immich uploader (multipart `/api/assets` + album
  create/reuse, deduped via a stable `deviceAssetId`), and a Nextcloud WebDAV
  uploader (`MKCOL` + `PUT`, idempotent). New config keys `IMMICH_BASE_URL`,
  `IMMICH_API_KEY`, `NEXTCLOUD_URL`, `NEXTCLOUD_USER`, `NEXTCLOUD_PASS`,
  `WATCH_SAVE_FRAME_CAP` (`NEXTCLOUD_PASS` must be a Nextcloud **app
  password** — Settings → Security → Devices & Sessions). Best-effort: any
  failure (missing config, network, bad response) logs a one-line stderr note
  and skips only that target — it never fails the `/watch` run, and the
  other target still runs. `setup.py` documents the 6 keys in its env
  template and reports which targets are configured (`library_targets`,
  informational only — never gates `can_proceed`, same treatment as
  TranscriptAPI).

## [0.4.2] — 2026-07-07

### Added
- **TranscriptAPI YouTube backend.** YouTube URLs now fetch their transcript
  from [TranscriptAPI](https://transcriptapi.com) before falling back to yt-dlp
  captions → Whisper. Their servers clear YouTube's bot-gate / PO-token / nsig
  walls that block yt-dlp from datacenter IPs, so a Spanish (or any-language)
  YouTube video yields a caption-quality transcript with zero Whisper spend and
  no video download for `--detail transcript`. New pure-stdlib `transcriptapi.py`
  backend, `TRANSCRIPTAPI_API_KEY` env/config key, and a `--no-transcriptapi`
  opt-out. Language preference follows `--sub-lang`. Best-effort: any miss (no
  key, no credits, no transcript, network) falls through with a logged reason.

## [0.4.1] — 2026-07-07

### Fixed
- **Subtitle fetch was broken by an invalid regex in the default languages.**
  yt-dlp compiles each `--sub-langs` token as a regex; the `*-orig` token
  shipped in 0.4.0 is a malformed regex (bare `*`) and made yt-dlp abort with
  `Wrong regex for subtitlelangs`, silently killing caption download for every
  video. Corrected to `.*-orig`. Added a test that compiles every default
  sub-lang token so no invalid pattern can ship again (the mocked argv tests
  never invoke yt-dlp, so they could not catch this).

## [0.4.0] — 2026-07-07

### Added
- **YouTube reliability.** yt-dlp now runs a `player_client=default,android,mweb`
  fallback chain with anti-bot retry/jitter by default, so SABR/403 blocks on
  public videos self-heal without cookies. New `--cookies-from-browser <browser>`
  and `--cookies <file>` flags unlock login-walled sources (off by default).
- **Spanish-first subtitles.** Default subtitle preference is now
  `es,es-419,es-ES,en,en-US,en-GB,.*-orig`; override with `--sub-lang <csv>`. The
  report picks the first available track in that order. Never requests `all`.
- **Whisper chunk cache.** Transcribed audio chunks are cached at
  `~/.cache/watch/transcripts/` keyed by `sha256(bytes + model)`, so re-runs and
  crash-resumes skip already-uploaded chunks instead of re-spending quota.
- **Windowed transcription.** With `--start/--end`, only that audio window is
  extracted and transcribed (not the full track), then stitched back to source
  time — zero Whisper spend outside the requested range.
- **Crash resume.** With `--out-dir`, the download and transcript stages persist
  to `stage_*.json` and a matching re-run resumes past them. `--fresh` forces a
  full re-run and bypasses the chunk cache.

### Changed
- Chunk-level Whisper retries remain capped (no infinite 5xx quota burn); now
  covered by a regression test.

## [0.3.0] — 2026-07-06

### Changed
- Rebased the fork onto upstream 0.2.0: `skills/watch/` self-contained Agent
  Skills layout, detail levels (`WATCH_DETAIL`), perceptual frame dedup,
  Whisper auto-chunking, transcript-cue frames, and the upstream pytest suite.
- `/watch` slash command now comes from SKILL.md frontmatter (no `commands/`
  wrapper); scripts resolve via `${SKILL_DIR}`.
- Transcript resolution (captions → Whisper) now runs fully **before** frame
  extraction, not just for caption-bearing URLs — so two-pass speech-aware
  sampling is available for local files and Whisper-only transcripts too, not
  only videos with native captions.

### Added (re-ported fork features)
- Multi-backend: `--backend claude|gemini|openrouter` (Gemini native video incl.
  YouTube pass-through; OpenRouter vision + audio with qwen3-asr → voxtral →
  groq fallback chain).
- Transcription backends: `--whisper local` (faster-whisper GPU) and
  `--whisper assemblyai` (+ `--diarize` speaker labels).
- Two-pass speech-aware frame sampling (`--two-pass`, default on),
  PySceneDetect option (`--scene-threshold`), OCR pass with hi-res re-extract
  (`--no-ocr` to skip), external audio track via `--audio`.
- cp1252-tolerant config reads; UTF-8 stdout/stderr on Windows.

## [0.2.0] — 2026-06-29

### Added
- **`--detail` dial** with four modes — `transcript` (captions only, no frames), `efficient` (fast keyframe pass, cap 50), `balanced` (scene-aware, cap 100, default), and `token-burner` (scene-aware, uncapped). Set the default with `WATCH_DETAIL` in `~/.config/watch/.env`.
- **Frame deduplication** (default on; `--no-dedup` to disable). Before the budget cap, a pass downscales each frame to a 16×16 grayscale thumbnail and drops frames whose mean per-pixel difference from the last *kept* frame is within threshold — so the budget goes to distinct content instead of held slides and static recordings. The **Frames** report line shows how many near-duplicates were dropped.
- **Whisper auto-chunking.** Audio over the 25 MB upload cap is split into evenly sized chunks, transcribed per chunk, with segment timestamps shifted back into source time. Partial failures are tolerated — transcription only fails if *every* chunk fails, so length alone no longer breaks it.
- **`--timestamps T1,T2,…`** — grab a frame at each absolute timestamp; reserved against the cap, and the only frames produced under `--detail transcript`.
- **`--no-whisper`** — disable transcription entirely (frames only).
- pytest suite covering config, dedup, download, fixtures, frames, setup, timestamps, watch, and whisper (no network; ffmpeg-synthesized clips).

### Changed
- **Restructured into a self-contained `skills/watch/` package** so `SKILL.md` and its `scripts/` runtime are siblings in one folder. This fixes installs on Codex, Cursor, Copilot, and other Agent Skills hosts: `npx skills add` now copies the skill as a working unit instead of grabbing the root `SKILL.md` without its scripts.
- **Harness-agnostic path resolution** — `SKILL.md` resolves `$SKILL_DIR` from where it was Read instead of the Claude-Code-only `${CLAUDE_SKILL_DIR}`, so script calls work on every host.
- `/watch` is now derived from `SKILL.md` frontmatter; the separate `commands/watch.md` wrapper was dropped to avoid a duplicate slash command.
- `balanced` now full-decodes to detect every scene cut across the whole video. The previous early-exit was faster but kept only the first cuts and dropped the tail of long videos.
- `token-burner` is exempt from the long-video "sparse scan" warning, since it keeps every scene-change frame.
- `--max-frames` is now an override on top of each mode's default cap, rather than a fixed default of 80.

### Fixed
- Non-Claude installs (`npx skills add`) were dead on arrival — the installer copied `SKILL.md` without the `scripts/` it shells out to. The self-contained package layout resolves this.

### Removed
- `V2_PLAN.md` and `V2_CONCERNS.md` planning docs.

## [0.1.3] — 2026-05-09

### Fixed
- Windows: `video.info.json` is read as UTF-8 (#4). Previously `Path.read_text()` defaulted to cp1252 on Windows and crashed on yt-dlp's UTF-8 output, silently dropping Title/Uploader from the report. Same fix applied to `.env` reads/writes in `whisper.py` and `setup.py`.
- `download.py` now logs info.json parse failures to stderr instead of swallowing them.

### Security
- Hardened subprocess argv against option injection (#2): inserted `--` before the URL in the yt-dlp argv, and tightened `is_url` to reject `-`-prefixed sources and require a non-empty netloc. Resolved video/audio paths to absolute via `Path.resolve()` before passing to `ffmpeg`/`ffprobe`, so a relative path starting with `-` can't be misinterpreted as a flag.

## [0.1.2] — 2026-04-24

### Fixed
- Windows console crash: removed the emoji from the long-video warning in `watch.py`; cp1252 consoles couldn't encode it.
- `setup.py` now prints `winget` / `pip` install commands on Windows instead of "unsupported platform" — matches what the README already promised.

### Changed
- `SKILL.md` notes that on Windows the scripts must be invoked with `python`, not `python3` (the latter is the Microsoft Store stub on Windows).

## [0.1.1] — 2026-04-24

### Fixed
- Added `commands/watch.md` shim so `/watch` is callable when installed as a Claude Code plugin. Without it, the plugin loaded but the skill wasn't exposed as a slash command.
- `scripts/build-skill.sh` now strips `commands/` from the claude.ai `.skill` bundle alongside `hooks/` and `.claude-plugin/`.

## [0.1.0] — 2026-04-24

Initial marketplace release.

### Added
- `/watch <url-or-path> [question]` slash command.
- yt-dlp download with native caption extraction (manual + auto-subs).
- ffmpeg frame extraction with auto-scaled fps (≤2 fps, ≤100 frames, duration-aware budget).
- `--start` / `--end` focused mode with denser frame budget and transcript range filtering.
- Whisper fallback (Groq preferred, OpenAI secondary) for videos without captions.
- `setup.py` preflight: silent `--check`, structured `--json`, and installer that auto-runs `brew install` on macOS.
- Session-start hook that prints a one-line status on first run / partial config.
- `.skill` bundle packaging for claude.ai upload via `scripts/build-skill.sh`.
