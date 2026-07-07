# Changelog

All notable changes to `/watch` are documented here.

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
