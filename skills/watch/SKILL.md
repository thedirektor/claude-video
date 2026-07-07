---
name: watch
version: "0.4.2"
description: Watch a video (URL or local path) with a Claude, Gemini, or OpenRouter backend. Downloads with yt-dlp, extracts auto-scaled frames with ffmpeg via scene detection + OCR, pulls the transcript from captions or Whisper (local GPU via faster-whisper, Groq, OpenAI, or AssemblyAI for speaker diarization), and hands the result to Claude so it can answer questions about what's in the video.
argument-hint: "<video-url-or-path> [question]"
allowed-tools: Bash, Read, AskUserQuestion
homepage: https://github.com/thedirektor/claude-video
repository: https://github.com/thedirektor/claude-video
author: thedirektor
license: MIT
user-invocable: true
---

# /watch

You don't have a video input; this skill gives you one. A Python script gets captions first, optionally downloads the video, extracts frames as JPEGs (scene-aware, or fast keyframes at `efficient` detail), gets a timestamped transcript (native captions first, then Whisper API as fallback), and prints frame paths. You then `Read` each frame path to see the images and combine them with the transcript to answer the user. This is the default `--backend claude` pipeline described in the rest of this document; pass `--backend gemini` or `--backend openrouter` to hand the whole video to a hosted multimodal model instead — see "Backends" below.

## Resolve `SKILL_DIR` (do this before any command)

Every `python3 ...` command below runs a bundled script under `SKILL_DIR/scripts/`. Set `SKILL_DIR` to the **absolute path of the directory containing THIS SKILL.md you just Read** — your harness told you that path in the Read result. The scripts are always a direct sibling of this file (`SKILL_DIR/scripts/watch.py`), in every install layout:

```
Read ~/.claude/plugins/cache/claude-video/watch/<ver>/skills/watch/SKILL.md → SKILL_DIR=…/skills/watch
Read ~/.codex/skills/watch/SKILL.md                                          → SKILL_DIR=~/.codex/skills/watch
Read ~/.agents/skills/watch/SKILL.md                                         → SKILL_DIR=~/.agents/skills/watch
```

Substitute that literal path for `${SKILL_DIR}` in every command. This works on every harness (Claude Code, Codex, Cursor, Gemini CLI, …) without relying on any harness-specific environment variable. Guard once at the start of a run:

```bash
SKILL_DIR="<absolute path of the directory containing the SKILL.md you Read>"
if [ ! -f "$SKILL_DIR/scripts/watch.py" ]; then
  echo "ERROR: scripts/watch.py not found under SKILL_DIR=$SKILL_DIR" >&2
  echo "Re-check the directory of the SKILL.md you Read and substitute it as SKILL_DIR." >&2
  exit 1
fi
```

## Step 0 — Setup preflight (runs every `/watch` invocation, silent on success)

**Python interpreter:** every `python3 ...` command in this skill is for macOS/Linux. On **Windows**, substitute `python` — the `python3` command on Windows is the Microsoft Store stub and will not run the script.

On the first `/watch` invocation in a session, use structured preflight so you can detect first-run setup:

```bash
python3 "${SKILL_DIR}/scripts/setup.py" --json
```

Branch on two fields:

- **`can_proceed: true` and `first_run: false`** → setup is already done (the user may have deliberately skipped a Whisper key — that's allowed). Proceed to Step 1 without comment.
- **`first_run: true`** → genuine first-time setup. Do these in order:
  1. If `missing_binaries` is non-empty, run the installer first (it auto-installs on macOS / prints commands elsewhere — see below) and confirm the binaries land. **Do not skip this and jump to preferences.**
  2. Run the installer once more if needed so it scaffolds `~/.config/watch/.env` (it only writes the template when the file is absent, so let it create the file *before* you write any values into it).
  3. Encourage a Whisper API key and ask the watch-preference questions below, then write the selected values into `~/.config/watch/.env` and set `SETUP_COMPLETE=true`.
- **`can_proceed: false` and `first_run: false`** → setup was finished before but the environment regressed (e.g. `missing_binaries` after an OS change). Run the installer to remediate, then proceed. Don't re-ask preferences.

A missing Whisper key is *encouraged to fix, not required*: on a genuine first run `status` will read `needs_key` even when binaries are present — that's your cue to encourage a key, not a blocker.

On follow-up `/watch` calls in the same session, use the silent check:

```bash
python3 "${SKILL_DIR}/scripts/setup.py" --check
```

This is a <100ms lookup. Exit 0 means /watch can run — this **includes a user who finished setup without a Whisper key** (keyless is allowed). On exit 0 the script emits **nothing** — proceed to Step 1 without comment. **Do NOT announce "setup is complete" to the user** — they don't need a status message on every turn. The only acceptable user-visible output from Step 0 is when remediation is required.

On non-zero exit, follow the table:

| Exit | Meaning | Action |
|------|---------|--------|
| `2` | Missing binaries (`ffmpeg` / `ffprobe` / `yt-dlp`) | Run installer |
| `3` | Genuine first run with no Whisper API key | Run installer to scaffold `.env`, then encourage a key (the user may decline — proceed with `--no-whisper`) |
| `4` | Both missing | Run installer, then encourage a key |

Exit `3` only fires before the user has completed setup. Once `SETUP_COMPLETE=true` is written, a keyless install returns exit 0 and is never nagged again.

The installer is idempotent — safe to re-run:

```bash
python3 "${SKILL_DIR}/scripts/setup.py"
```

On macOS with Homebrew, it auto-installs `ffmpeg` and `yt-dlp`. On Linux/Windows, it prints the exact install commands for the user to run. It scaffolds `~/.config/watch/.env` with commented placeholders and default watch settings at `0600` perms.

**If an API key is still missing after install:** use `AskUserQuestion` to ask the user whether they have a Groq API key (preferred — cheaper, faster) or an OpenAI key. Then write it into `~/.config/watch/.env` — set the matching `GROQ_API_KEY=...` or `OPENAI_API_KEY=...` line. If they don't want to set up Whisper, proceed with `--no-whisper` and tell them videos without native captions will come back frames-only.

**First-run watch preference:** after the installer has scaffolded `~/.config/watch/.env`, use `AskUserQuestion` to ask one question:

- Default detail (one dial). Present these as `AskUserQuestion` options in this exact order — lightest to heaviest — and keep `(recommended)` on `balanced` even though it is not first (do **not** reorder to put the recommended option first):
  - `transcript` — no frames at all, transcript only (skips video download when captions exist).
  - `efficient` — fast keyframe pass (cap 50).
  - `balanced` (recommended) — scene-aware frames (cap 100, default).
  - `token-burner` — scene-aware, uncapped (maximum fidelity; high token cost).

Write the answer directly into `~/.config/watch/.env` by setting the bare key on its own line — **no trailing inline comment** (a `# note` after the value can break parsing):

```bash
WATCH_DETAIL=balanced
```

Use the user's selected value. If they skip the question, keep the recommended default. Once dependencies, the API-key choice, and this preference are handled, write or update `SETUP_COMPLETE=true` in the same file. Do not ask this preference question again when `SETUP_COMPLETE=true`.

**Structured mode (optional):** `python3 "${SKILL_DIR}/scripts/setup.py" --json` emits `{status, can_proceed, first_run, setup_complete, missing_binaries, whisper_backend, has_api_key, has_transcriptapi_key, config_file, watch_detail, platform}` where `status` is one of `ready | needs_install | needs_key | needs_install_and_key`. `status` describes the *ideal* state (a key is encouraged, so a keyless first run reads `needs_key`); `can_proceed` is the operational gate (binaries present AND a key is set OR setup was already completed). Branch on `can_proceed`/`first_run` to decide whether to run; use `status` to decide what to encourage. `has_transcriptapi_key` reports whether `TRANSCRIPTAPI_API_KEY` is set (optional — TranscriptAPI is best-effort and never gates `can_proceed`).

Within a single session, you can skip Step 0 on follow-up `/watch` calls — once `--check` returned 0, nothing about the environment changes between turns.

## When to use

- User pastes a video URL (YouTube, Vimeo, X, TikTok, Twitch clip, most yt-dlp-supported sites) and asks about it.
- User points at a local video file (`.mp4`, `.mov`, `.mkv`, `.webm`, etc.) and asks about it.
- User types `/watch <url-or-path> [question]`.

## Recommended limits

- **Best accuracy: videos under 10 minutes.** Frame coverage scales inversely with duration.
- **Universal rate cap: 2 fps.** The script never samples faster than 2 fps, even when a budget or `--fps` would imply more.
- **The frame ceiling is set by the detail mode** (`WATCH_DETAIL` in `~/.config/watch/.env`, or `--detail`), not a single global cap:
  - `transcript` → no frames
  - `efficient` → up to **50** (keyframes)
  - `balanced` (default) → up to **100** (scene-aware)
  - `token-burner` → **uncapped** (scene-aware; a soft warning prints past 250 frames) — except under two-pass sampling (see below), where the budget is always the duration-scaled auto-fps target
  - `--max-frames N` overrides whichever cap the mode would otherwise use.
- **Full-video frame budget by duration.** Token cost grows with frame count, so the script targets a budget by duration. This budget sets the fps and the uniform-sampling fallback; scene-aware selection can fill up to the detail cap above, whichever is lower:
  - ≤30s → ~12-30 frames
  - 30s-1min → ~40 frames
  - 1-3min → ~60 frames
  - 3-10min → ~80 frames
  - \>10min → up to the detail cap, sparsely spaced (warning printed)
- If the user hands you a long video, consider asking whether they want a specific section before burning tokens on a sparse scan.

## How to invoke

**Step 1 — parse the user input.** Separate the video source (URL or path) from any question the user asked. Example: `/watch https://youtu.be/abc what language is this in?` → source = `https://youtu.be/abc`, question = `what language is this in?`.

**Step 2 — run the watch script.** Pass the source verbatim. Do not shell-escape it yourself beyond normal quoting:

```bash
python3 "${SKILL_DIR}/scripts/watch.py" "<source>"
```

Optional flags:

**Backend**
- `--backend claude|gemini|openrouter` — pick the orchestrator. Default `claude` runs the local frame pipeline described in the rest of this document. `gemini` and `openrouter` each hand the extracted material to a hosted multimodal model and print its response directly instead of leaving frames for Claude to `Read`. See "Backends" below.
- `--gemini-model NAME` — Gemini model when `--backend gemini`. Choices: `gemini-3.1-flash-lite | gemini-2.5-flash | gemini-2.5-pro`. Default `gemini-3.1-flash-lite`. Ignored otherwise.
- `--openrouter-vision-model NAME` — OpenRouter model that analyzes frames + transcript when `--backend openrouter` (default `google/gemini-2.5-flash`). Ignored otherwise.
- `--openrouter-audio-model NAME` — OpenRouter model used to transcribe audio when `--backend openrouter` and no captions are found (default `qwen/qwen3-asr-flash-2026-02-10`). Ignored otherwise.
- *(positional)* `question` — required for `--backend gemini` / `--backend openrouter`, passed to the model as the prompt. Ignored for `--backend claude`, where Claude takes the question from the chat instead.

**Range / budget**
- `--detail transcript|efficient|balanced|token-burner` — fidelity/speed dial. `transcript` = no frames (transcript only, skips video download when captions exist); `efficient` = fast keyframes (cap 50); `balanced` = scene-aware frames (cap 100); `token-burner` = scene-aware, uncapped.
- `--timestamps T1,T2,…` — grab a frame at each of these absolute timestamps (`SS`, `MM:SS`, or `HH:MM:SS`). Use this after reading the transcript to capture deictic moments the presenter flags ("look here", "as you can see", "notice this") that visual selection alone may miss. See "Transcript-cue frames" below.
- `--max-frames N` — override the preset cap for tighter token budget (e.g. `--max-frames 40`)
- `--resolution W` — change frame width in px (default 512; bump to 1024 only if the user needs to read on-screen text)
- `--fps F` — override auto-fps (clamped to 2 fps max). Disables two-pass speech-aware sampling, which relies on the auto-fps-informed budget — the `balanced`/`token-burner` detail engines' own scene-aware extraction still runs, just at the forced rate.
- `--out-dir DIR` — keep working files somewhere specific (default: an auto-generated tmp dir)

**Download / resume**
- `--sub-lang CSV` — subtitle language preference for yt-dlp (default `es,es-419,es-ES,en,en-US,en-GB,.*-orig`). The report uses the first available track in this order. Never pass `all`.
- `--cookies-from-browser BROWSER` — load yt-dlp cookies from a browser profile (`chrome|firefox|edge|brave|safari|...`) for login-walled or age-gated sources. Off by default.
- `--cookies FILE` — path to a Netscape `cookies.txt` for yt-dlp. Off by default.
- `--fresh` — with `--out-dir`, ignore any saved resume state and re-run every stage from scratch (also bypasses the Whisper chunk cache).

**Frame sampling**
- `--no-dedup` — keep near-duplicate frames. By default a frame-delta pass drops frames that are visually near-identical to the previous kept one (held slides, static screen recordings, paused video) so the frame budget goes to distinct content; the report's **Frames** line notes how many were dropped. Pass this only if the user needs every sampled frame (e.g. judging subtle frame-to-frame motion).
- `--no-scene-detect` — skip PySceneDetect (`scenes.py`) when computing scene boundaries for two-pass sampling; falls back to even spacing across speech/silent windows instead. This only affects two-pass — the `balanced`/`token-burner` detail engines run their own independent ffmpeg scene-cut detection regardless of this flag.
- `--scene-threshold F` — PySceneDetect `ContentDetector` threshold used by two-pass (default `27.0`). Lower = more cuts detected. Bump to `35-40` for low-cut talking heads, drop to `20` for fast-cut promo content.
- `--two-pass` / `--no-two-pass` — distribute the frame budget proportionally to speech windows from the transcript (70% inside speech, 30% outside), using scene cuts (unless `--no-scene-detect`) to pick good frames within each window. **Default ON** whenever a timed transcript is available — captions or Whisper, for local files just as much as caption-bearing URLs, since the transcript is now fully resolved before frame extraction runs (see "Transcription" below). Automatically inert when `--detail transcript` is used, `--fps` is set, or no transcript is available. Under two-pass the budget is always the duration-scaled auto-fps target, not the flat detail cap — so `token-burner`'s "uncapped" promise is finite here too. Pass `--no-two-pass` if you want `token-burner`'s uncapped scene dump instead.

**Audio / transcription**
- `--audio FILE` — separate audio file (mp3/wav/m4a) to transcribe instead of the video's own audio track. Use when the video is muted and the voiceover ships as a separate ElevenLabs/recorded file. Implies Whisper transcription of that file; cannot combine with `--no-whisper`.
- `--whisper groq|openai|local|assemblyai` — force a specific Whisper backend.
  - `groq` — `whisper-large-v3` via Groq API. Cheap, fast, needs `GROQ_API_KEY`.
  - `openai` — `whisper-1` via OpenAI API. Needs `OPENAI_API_KEY`.
  - `local` — runs faster-whisper on the local GPU. No API key, no upload, no rate limit — but needs an NVIDIA GPU and faster-whisper installed (see "Local Whisper" below).
  - `assemblyai` — paid (~$0.37/hr with diarization, $50 free credits). The only backend that returns speaker labels (`[Speaker A]`, `[Speaker B]`, …). Needs `ASSEMBLYAI_API_KEY`. See "Speaker diarization" below.
  - **Default:** auto-pick `local` if available, else `groq`, else `openai`. AssemblyAI is never auto-picked — request it explicitly when you need diarization.
- `--whisper-model NAME` — faster-whisper model size for the local backend. Choices: `tiny | base | small | medium | large-v2 | large-v3`. Default `large-v3`. Ignored for groq / openai / assemblyai. See the model table under "Local Whisper" below.
- `--diarize` / `--no-diarize` — request speaker labels when the backend supports them (currently only AssemblyAI). Default ON; pass `--no-diarize` for sentence-level segments without speaker tags. Ignored for local / groq / openai.
- `--no-whisper` — disable the Whisper fallback entirely (frames-only if no captions). Cannot combine with `--audio`.
- `--no-transcriptapi` — disable the TranscriptAPI YouTube-transcript backend. By default, YouTube URLs fetch their transcript from [TranscriptAPI](https://transcriptapi.com) first (needs `TRANSCRIPTAPI_API_KEY`), which succeeds from server IPs where YouTube blocks yt-dlp; this flag forces the yt-dlp caption / Whisper path instead.

**OCR**
- `--no-ocr` — disable the OCR pass. By default, after frames are extracted the script runs Tesseract over them (lang=`spa+eng`) and re-extracts text-heavy frames at 1024px so on-screen text stays legible. Disable when text doesn't matter (silent action footage, abstract content) to save a few seconds.

### Focusing on a section (higher frame rate)

When the user asks about a specific moment — "what happens at the 2 minute mark?", "zoom into 0:45 to 1:00", "the first 10 seconds" — pass `--start` and/or `--end`. The script switches to focused-mode budgets, which are denser than full-video budgets (still capped at 2 fps, and still bounded by the detail-mode cap — the counts below assume the default `balanced` cap of 100; `efficient` tops out at 50):

- ≤5s → 2 fps (up to 10 frames)
- 5-15s → 2 fps (up to 30 frames)
- 15-30s → ~2 fps (up to 60 frames)
- 30-60s → ~1.3 fps (up to 80 frames)
- 60-180s → ~0.6 fps (100 frames, capped)

Focused mode is the right call for:
- Any moment/range the user names explicitly ("around 2:30", "the intro", "the last 30 seconds").
- Any video longer than ~10 minutes where the user's question is about a specific part — running focused on the relevant section is far more useful than a sparse scan of the whole thing.
- Re-runs after a full scan didn't have enough detail in some region.

Transcript is auto-filtered to the same range. Frame timestamps are absolute (real video timeline, not offset-from-start).

Examples:
```bash
# Last 10 seconds of a 1 minute video
python3 "${SKILL_DIR}/scripts/watch.py" video.mp4 --start 50 --end 60

# Zoom into 2:15 → 2:45 at 2 fps (60 frames)
python3 "${SKILL_DIR}/scripts/watch.py" "$URL" --start 2:15 --end 2:45 --fps 2

# From 1h12m to the end of the video
python3 "${SKILL_DIR}/scripts/watch.py" "$URL" --start 1:12:00
```

**Step 3 — Read every frame path the script lists.** The Read tool renders JPEGs directly as images for you. Read all frames in a single message (parallel tool calls) so you see them together. The frames are in chronological order with a `t=MM:SS` timestamp so you can align them to the transcript.

**Step 4 — answer the user.** You now have two streams of evidence:
- **Frames** — what's on screen at each timestamp
- **Transcript** — what's said at each timestamp. The report's header shows the source (`captions` = yt-dlp pulled native subs; `whisper (groq)` or `whisper (openai)` = transcribed by API).

If the user asked a specific question, answer it directly citing timestamps. If they didn't ask anything, summarize what happens in the video — structure, key moments, notable visuals, spoken content.

This holds for `transcript` detail too: even with no frames, produce a **summary** like the other modes — do not paste the full transcript into chat. Synthesize structure, key moments, and spoken content with timestamps; quote only the lines that matter. Offer the raw transcript only if the user explicitly asks for it.

**Step 5 — clean up.** The script prints a working directory at the end. If the user isn't going to ask follow-ups about this video, delete it with `rm -rf <dir>`. If they might, leave it in place.

## Backends

Three top-level backends, picked with `--backend`:

- **`--backend claude`** *(default)* — the pipeline described in the rest of this document. Download → frame extraction → OCR → transcript → Claude `Read`s each frame and answers from its own context. Best for: any case where you want Claude to reason over the video, ask follow-ups in the same session, or compare with prior conversation.
- **`--backend gemini`** — skip frame extraction, OCR, and Whisper entirely. Hand the whole video (or a YouTube URL) to Gemini's multimodal model and print its response. Best for: one-shot full-video analyses ("timestamp all on-screen text", "summarize this 30-minute keynote") where Gemini's native video understanding is what you want.
- **`--backend openrouter`** — run the same local frame-extraction + transcript pipeline as `claude` (including OCR, two-pass, detail modes), but instead of leaving frame paths for Claude to `Read`, base64-encode the frames and POST them together with the transcript and question to an OpenRouter vision model, printing its response. Best for: routing the analysis through a specific OpenRouter-hosted model rather than Claude or Gemini.

`gemini` and `openrouter` both require the question as the trailing positional argument, since there's no follow-up conversation turn for them to pull it from:

```bash
# Local file: uploaded to Gemini Files API, polled until ACTIVE, then sent
python3 "${SKILL_DIR}/scripts/watch.py" video.mp4 --backend gemini "Describe this video"

# YouTube URL: passed straight to Gemini, no yt-dlp download — fetched server-side
python3 "${SKILL_DIR}/scripts/watch.py" "https://youtu.be/abc" --backend gemini "Summarize"

# OpenRouter: local frame extraction + transcript, then POST to a vision model
python3 "${SKILL_DIR}/scripts/watch.py" video.mp4 --backend openrouter "What is on screen at the end?"
```

Non-YouTube URLs (Vimeo, TikTok, etc.) are downloaded with yt-dlp first and then uploaded to Gemini's Files API, since the model only natively fetches YouTube.

### `--gemini-model` picker

| Model | Notes | When to pick |
|-------|-------|--------------|
| `gemini-3.1-flash-lite` | **Default.** Stable May 7 2026. 1M-token input / 65k output, multimodal (text/image/video/audio/PDF). Free-tier quota available. | General default — fastest, cheapest, still video-capable. |
| `gemini-2.5-flash` | Balanced. | When `3.1-flash-lite`'s quality isn't enough but `2.5-pro` is overkill — mid-length videos that need careful reasoning over the visuals. |
| `gemini-2.5-pro` | Highest quality, longest context (~2M tokens). | Very long videos or deep reasoning with extensive output. |

`gemini-2.0-flash` and `gemini-1.5-pro` are not in the picker — 1.5-pro returns 404 on the current `v1beta` API, and 2.0-flash has no free-tier quota on most accounts. Override with `--gemini-model` if you need to point at a specific model that's been added later.

### OpenRouter models

- `--openrouter-vision-model NAME` (default `google/gemini-2.5-flash`) — analyzes the extracted frames + transcript and answers the question.
- `--openrouter-audio-model NAME` (default `qwen/qwen3-asr-flash-2026-02-10`) — transcribes audio when no captions are found. Falls back automatically down a chain when a leg errors: the requested model → `mistralai/voxtral-mini-transcribe` on OpenRouter → Groq's `whisper-large-v3` (if `GROQ_API_KEY` is set) — so a single model outage doesn't kill the transcript. If every leg fails, the run proceeds with no transcript rather than aborting (only the vision call is fatal).

## Detail and frames

Default behavior comes from `~/.config/watch/.env`:

- `WATCH_DETAIL=transcript|efficient|balanced|token-burner` (default: `balanced`)

At `transcript` detail, captions are enough to return a report without downloading video. If captions are missing, the script downloads audio only and tries Whisper. If no transcript can be produced, it reports the limitation clearly; re-run with `--detail balanced` for frames.

At `efficient` detail, the script downloads the video and extracts **keyframes only** (`ffmpeg -skip_frame nokey`) — a near-instant pass that lands frames on scene cuts. If a clip has fewer than 4 keyframes it falls back to uniform sampling.

At `balanced` / `token-burner` detail, the script extracts **scene-aware** frames: ffmpeg scene-change selection first, falling back to uniform sampling only when the video is effectively static. `balanced` caps at 100 frames; `token-burner` is uncapped. Frame report lines include both timestamp and selection reason. Extracted images are clamped to a maximum 1998px height for Claude Read compatibility.

When a timed transcript is available, two-pass sampling (see "Frame sampling" above) takes over from the detail engine described here and distributes the duration-scaled auto-fps budget across speech vs. silent windows instead — not the flat detail cap, so `token-burner` is finite (not uncapped) whenever two-pass runs. Pass `--no-two-pass` for the uncapped scene dump.

## Transcript-cue frames

Visual frame selection (scene/keyframe) can miss the moments a presenter explicitly flags — "look here", "as you can see", "notice this", "watch what happens" — because pointing at a slide is often a *low* visual change. `--timestamps` lets you force a frame at those exact moments. **You** decide which moments matter, by reading the transcript:

1. Run once at `--detail transcript` (or any detail) to get the timestamped transcript.
2. Scan it for deictic cues — phrases where the speaker directs attention to something on screen. This is a judgment call (ignore rhetorical "look, the point is…"); that's why it's done by you, not a regex.
3. Re-run with `--timestamps 4:32,7:10,9:55` (absolute source times). For a URL, point the second run at the **downloaded local file** in the work dir so it doesn't re-download.

Behavior:
- **Additive by default.** Cue frames (`reason=transcript-cue`) are merged into whatever `--detail` (or two-pass) already selected, in chronological order.
- **Pinned and counted first.** Cue frames are reserved against the frame cap before the detail/two-pass engine runs, so they're never evicted by even-sampling.
- **Honors focus mode.** With `--start/--end`, any cue timestamp outside the window is dropped (reported in the summary). Coordinates are always absolute source time.
- **Cue-only frames.** `--detail transcript --timestamps …` skips scene/keyframe/two-pass sampling and returns *only* the cue frames (it will download the video to do so, since frames need pixels).

## YouTube transcripts (TranscriptAPI)

For **YouTube URLs**, `/watch` resolves the transcript in this order: **TranscriptAPI → yt-dlp captions → Whisper**. TranscriptAPI (`transcriptapi.com`) runs the extraction on its own servers, so it clears YouTube's "confirm you're not a bot" / PO-token / nsig walls that block yt-dlp from datacenter/VPS IPs — with **zero Whisper spend** and no video download for `--detail transcript`. Set `TRANSCRIPTAPI_API_KEY` in `~/.config/watch/.env` (1 credit per transcript; video metadata is free). Language preference follows `--sub-lang` (default Spanish→English→auto). Pass `--no-transcriptapi` to force the yt-dlp path. YouTube *frames* still require the (often gated) video download — TranscriptAPI covers the transcript only.

## Transcription

The script gets a timestamped transcript via up to four backends, in priority order, resolved **before** frame extraction runs:

1. **Native captions (free, preferred).** yt-dlp pulls manual or auto-generated subtitles from the source platform if available.
2. **Local Whisper via faster-whisper (no API call).** If no captions came back AND faster-whisper + a CUDA GPU are available, the script runs Whisper directly on the user's GPU. No upload, no key, no rate limit. See "Local Whisper" below.
3. **Whisper API fallback.** If local isn't available, the script extracts audio (`ffmpeg -vn -ac 1 -ar 16000 -b:a 64k`, ~0.5 MB/min) and uploads it to whichever Whisper API has a key configured:
   - **Groq** — `whisper-large-v3`. Preferred API default: cheaper, faster. Get a key at console.groq.com/keys.
   - **OpenAI** — `whisper-1`. Fallback. Get a key at platform.openai.com/api-keys.

   Audio over the API's 25 MB upload cap is split into chunks, transcribed per chunk, and stitched back together with timestamps shifted into source time; a chunk-level failure is tolerated (transcription only fails if *every* chunk fails).
4. **AssemblyAI (paid, opt-in).** Only used when explicitly requested via `--whisper assemblyai`. The differentiator is **automatic speaker diarization** — multi-speaker transcripts come back tagged `[Speaker A]`, `[Speaker B]`, etc. See "Speaker diarization" below.

API keys live in `~/.config/watch/.env`. Auto-selection priority: `local` → `groq` → `openai`. AssemblyAI is never auto-picked — request it explicitly when diarization is the goal. Use `--no-whisper` to skip transcription entirely.

Because the transcript now fully resolves before any frame is extracted — for local files and Whisper-produced transcripts, not just caption-bearing URLs — two-pass speech-aware sampling (see "Frame sampling" above) is available in every one of these cases, not only when captions exist up front.

The report header shows the backend used:
- `via transcriptapi (lang)` — YouTube-only; fetched from TranscriptAPI before yt-dlp captions were attempted (see "YouTube transcripts (TranscriptAPI)" above)
- `via captions` — yt-dlp pulled native subs
- `via whisper (local, large-v3)` — local GPU
- `via whisper (groq)` / `via whisper (openai)` — API
- `via whisper (assemblyai, diarized)` — AssemblyAI with speaker labels (or `assemblyai` if `--no-diarize`)
- `via whisper (..., --audio)` suffix — `--audio FILE` was used to transcribe a separate VO track

### Speaker diarization

When `--whisper assemblyai` is used (with `--diarize`, the default), each transcript segment carries a speaker tag and the formatted output looks like:

```
[Speaker A] (0:00-0:05) Welcome back to the show. Today we're talking with…
[Speaker B] (0:05-0:09) Thanks for having me, glad to be here.
[Speaker A] (0:09-0:18) Let's jump in — when did you first realize…
```

When to reach for it:
- **Interviews / podcasts** — separating host vs guest dialogue is the obvious win.
- **Multi-speaker UGC** — vlog conversations, panel clips, gameplay commentary with two voices.
- **Ads with presenter + voiceover** — keeps the on-camera dialogue separate from the off-camera narration.
- **Single-speaker content** — still works; the structure is just `[Speaker A]` for everything. Useful when you want the per-utterance start/end timing surfaced.

The other Whisper backends (local / groq / openai) do not produce speaker labels — Whisper itself doesn't diarize. Pricing on AssemblyAI is roughly **$0.37 per hour of audio** with diarization enabled (text-only transcription is cheaper); new accounts get **$50 of free credits**, which covers ~135 hours of diarized audio.

> **Roadmap note:** Local diarization via [WhisperX](https://github.com/m-bain/whisperX) (which combines faster-whisper with pyannote speaker embeddings) is on the roadmap pending upstream Python 3.14 support — the current WhisperX release pins to ≤ 3.13. Until then, AssemblyAI is the only diarized path that ships in this skill.

### Local Whisper (faster-whisper on GPU)

Local Whisper runs the model directly on the user's NVIDIA GPU via [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and CTranslate2. No API call, no 25 MB upload limit, no rate limit. Tested at ~13× realtime on an RTX 2080 Ti with `large-v3`.

**Prerequisites:**
- NVIDIA GPU with CUDA 12 support and ≥ enough VRAM for the chosen model (see table)
- Python 3.10+ (3.13 / 3.14 confirmed working)
- `faster-whisper`, `nvidia-cublas-cu12`, `nvidia-cudnn-cu12` installed via pip

**Install (Windows / Linux / macOS-with-NVIDIA):**

```bash
pip install faster-whisper nvidia-cublas-cu12 nvidia-cudnn-cu12
```

**Windows DLL discovery (handled automatically).** `whisper_local.py` runs `os.add_dll_directory()` at import time on Windows to register `<sys.prefix>\Lib\site-packages\nvidia\cublas\bin` and `…\nvidia\cudnn\bin` with the loader, so `import ctranslate2` finds the cuBLAS / cuDNN wheels without any environment changes from the user. No manual `PATH` editing required for the standard `pip install` layout.

Linux / macOS users don't hit this in the first place — pip's RPATH metadata handles DLL discovery on those platforms.

<details>
<summary>Manual PATH fallback (only if auto-registration fails — non-standard install layout, conda site-packages outside <code>sys.prefix</code>, or running <code>ctranslate2</code> from a tool that imports before <code>whisper_local</code>)</summary>

```powershell
$prefix = (python -c "import sys; print(sys.prefix)")
$cublas = "$prefix\Lib\site-packages\nvidia\cublas\bin"
$cudnn  = "$prefix\Lib\site-packages\nvidia\cudnn\bin"
[Environment]::SetEnvironmentVariable("PATH", "$cublas;$cudnn;" + [Environment]::GetEnvironmentVariable("PATH","User"), "User")
# Restart the terminal afterwards so the new PATH is picked up.
```
</details>

**Verify the install:**

```bash
python -c "import ctranslate2; print('CUDA devices:', ctranslate2.get_cuda_device_count())"
```

A successful install prints `CUDA devices: 1` (or higher). Anything else — `0`, an `OSError`, an `ImportError` — means the runtime can't reach a GPU and `--whisper local` will fall through to the API backends in auto mode (or hard-error if you forced `--whisper local`).

**Model picker (`--whisper-model`).** First run downloads the model into the HuggingFace cache (`~/.cache/huggingface/hub`). `large-v3` is ~3 GB and takes a minute or two on a typical connection; subsequent runs are instant.

| Model | Parameters | VRAM (fp16) | Relative speed | When to use |
|-------|-----------:|-------------|----------------|-------------|
| `tiny` | 39 M | ~1 GB | ~32× | Toy / smoke tests; clean speech only |
| `base` | 74 M | ~1 GB | ~16× | Lightweight, low-VRAM GPUs |
| `small` | 244 M | ~2 GB | ~6× | Decent quality, fits 4 GB cards |
| `medium` | 769 M | ~5 GB | ~2× | Solid quality, fits 8 GB cards |
| `large-v2` | 1550 M | ~10 GB | 1× | High quality, older v2 dataset |
| `large-v3` | 1550 M | ~10 GB | 1× | **Default.** Best quality; needs ≥ 10 GB VRAM |

Speed numbers are approximate ratios — actual realtime multiplier depends heavily on the GPU. Drop a tier if you hit OOM, or if `large-v3` is overkill for the content (e.g. short clean voiceover transcribes fine with `medium`).

**Failure modes for `--whisper local`:**
- faster-whisper / ctranslate2 not installed → falls through to Groq/OpenAI in auto mode; hard-errors when forced.
- CUDA DLLs missing → same fall-through behavior; the install hint with the pip command is printed.
- Model fails to load (OOM, corrupted cache) → re-download by deleting the model directory under `~/.cache/huggingface/hub`, or pick a smaller `--whisper-model`.

## Resume and caching

Whisper chunk transcripts are cached at `~/.cache/watch/transcripts/`
(`sha256(audio bytes + model)`), so re-asking about the same video — or resuming
after a crash — never re-uploads chunks it already transcribed. When you pass
`--out-dir DIR`, the download and transcript stages also persist there; a re-run
with the same source and options resumes past them. Pass `--fresh` to ignore all
of this and start clean. With `--start/--end`, only that time window's audio is
transcribed, so focusing on a section costs nothing outside it.

## Failure modes and handling

- **Setup preflight failed** → run `python3 "${SKILL_DIR}/scripts/setup.py"` (auto-installs ffmpeg/yt-dlp via brew on macOS, scaffolds the `.env`). For API key, ask the user via `AskUserQuestion` and write it to `~/.config/watch/.env`.
- **No transcript available** → captions missing AND no Whisper backend usable (no GPU + faster-whisper, no API key, or every backend failed). Script prints a hint pointing to setup. Proceed frames-only and tell the user.
- **Long video warning printed** → acknowledge it in your answer. Offer to re-run focused on a specific section via `--start`/`--end` rather than a sparse full-video scan.
- **Download fails** → yt-dlp's error goes to stderr. If it's a login-required or region-locked video, tell the user plainly; do not keep retrying.
- **Whisper API request fails** → the error is printed to stderr (likely: invalid key or rate limit — the 25 MB upload cap alone won't fail it, since oversized audio is auto-chunked). If some chunks fail the transcript is partial and the dropped chunks are noted on stderr; the report says "none available" only if every chunk fails. Retry options: `--whisper openai` if Groq failed (or vice versa), `--whisper local` if a GPU is available, or `--whisper assemblyai` if you want speaker labels and have credits.
- **`--whisper assemblyai` errors** → missing key prints the install hint with the signup URL; an upstream error from AssemblyAI is surfaced verbatim (typically billing / quota / unsupported audio). Retry with `--no-diarize` if it was the diarization step that timed out — the cheaper non-diarized path is more lenient.
- **`--whisper local` requested but unavailable** → the script hard-errors with the install hint (`pip install faster-whisper nvidia-cublas-cu12 nvidia-cudnn-cu12`). Auto-mode falls through to Groq/OpenAI silently instead.
- **OCR unavailable** → `pytesseract` or the `tesseract` binary missing. `run_ocr` degrades to no-op — the report's `OCR:` line is simply omitted rather than showing an error. Either install Tesseract (Windows: `C:\Program Files\Tesseract-OCR` plus the `spa` language pack) or pass `--no-ocr` to skip the pass explicitly.
- **`--backend gemini` errors** →
  - *Missing key:* `GEMINI_API_KEY` not set. Add it to `~/.config/watch/.env` or the environment; get a key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
  - *SDK missing:* `pip install google-genai` (the script prints the exact line).
  - *429 RESOURCE_EXHAUSTED with `limit: 0`:* the chosen model has no free-tier quota on this account — retry with `--gemini-model gemini-3.1-flash-lite` (the default) or enable billing on the Google Cloud project.
  - *404 NOT_FOUND on a model:* the model was retired server-side. Pass an explicit `--gemini-model` from the current picker.
  - *Empty response:* check stderr for the `finish_reason` — usually safety blocking or a malformed prompt. Rephrase the question.
- **`--backend openrouter` errors** →
  - *Missing key:* `OPENROUTER_API_KEY` not set. Add it to `~/.config/watch/.env` or export it.
  - *Audio transcription chain exhausted:* all three legs (requested model → `mistralai/voxtral-mini-transcribe` → Groq `whisper-large-v3`) failed — the run proceeds with "Transcript: none" rather than aborting; only the vision call is fatal.
  - *Vision call fails (HTTP error, non-JSON response, or empty response):* the script hard-errors. Retry with a different `--openrouter-vision-model`, or fall back to `--backend claude` / `--backend gemini`.

## Workflow examples

**Muted product video + separate ElevenLabs voiceover** — the case `/watch` was extended to support. The video is silent stock footage; the VO ships as a separate `.mp3` (often AI-generated). `--audio` retargets transcription to the VO file, two-pass sampling is auto-enabled, and frames concentrate on the moments the VO is actually narrating:

```bash
python3 "${SKILL_DIR}/scripts/watch.py" \
    "/path/to/Product Video.mp4" \
    --audio "/path/to/ElevenLabs_VO.mp3"
```

The report header reads `Transcript: 12 segments (via whisper (local, large-v3, --audio))` and frames inside speech windows get the `[speech]` tag while frames outside get `[silent]`. Two-pass distribution (default 70/30 speech/silent) is what makes the frame budget land where the narration actually is, instead of evenly across the video.

**Talking head, GPU-only transcription, smaller model:**
```bash
python3 "${SKILL_DIR}/scripts/watch.py" speaker.mp4 \
    --whisper local --whisper-model medium --no-scene-detect
```
`--no-scene-detect` is appropriate here because a single-camera talking-head video has no real cuts, so PySceneDetect would just fall through to even spacing anyway. `medium` transcribes a clean voice as well as `large-v3` and uses ~half the VRAM.

**Fast-cut promo / ad creative — bias toward more frames at scene boundaries:**
```bash
python3 "${SKILL_DIR}/scripts/watch.py" promo.mp4 \
    --scene-threshold 20 --no-ocr
```
Lowering the threshold below the 27.0 default captures more cuts; `--no-ocr` saves a couple seconds when the content is visual (no slides, no UI text).

## Token efficiency

This skill burns tokens primarily on frames. Order of magnitude:
- 80 frames at 512px wide is roughly 50-80k image tokens depending on aspect ratio.
- The transcript is cheap (a few thousand tokens at most for a 10-minute video).
- Bumping `--resolution` to 1024 roughly quadruples the image tokens per frame. Only do it when necessary.

If you already watched a video this session and the user asks a follow-up, do **not** re-run the script — you already have the frames and transcript in context. Just answer from what you have.

## Windows compatibility

This fork has been tested on **Windows 11 + Python 3.14** (and the Bash tool's PowerShell shell). Notes:

- **UTF-8 encoding fix is already applied to all scripts.** Each Python file in `scripts/` reconfigures `sys.stdout` / `sys.stderr` to UTF-8 at startup, so non-ASCII content (Spanish transcripts, em-dashes, accented filenames) doesn't crash with `UnicodeEncodeError` on Windows's default cp1252 console.
- **Use `python` not `python3`.** On Windows the `python3` command typically resolves to the Microsoft Store stub; the skill's docs use `python3` for Unix conventions but on Windows you should substitute `python`.
- **Tesseract for OCR** must be installed separately. Default install path on Windows is `C:\Program Files\Tesseract-OCR` — make sure that directory is on `PATH` so `pytesseract` can find `tesseract.exe`. Install via `winget install UB-Mannheim.TesseractOCR` or grab the [installer](https://github.com/UB-Mannheim/tesseract/wiki). For Spanish OCR you also need the `spa.traineddata` language pack — bundled by default in the Mannheim installer.
- **Local Whisper DLL discovery is automatic.** `whisper_local.py` calls `os.add_dll_directory()` for the bundled cuBLAS / cuDNN wheels at import time, so `--whisper local` works out of the box after `pip install faster-whisper nvidia-cublas-cu12 nvidia-cudnn-cu12`. The manual PATH edit is only needed as a fallback for non-standard layouts (see "Local Whisper" above).
- **Long paths.** Some yt-dlp downloads produce long filenames; if you hit "filename too long" errors, enable Win32 long paths via the Group Policy editor or pass `--out-dir` to a short path like `D:\w`.

## Security & Permissions

**What this skill does:**
- Runs `yt-dlp` locally to download the video and pull native captions when the source supports them (public data; the request goes directly to whatever host the URL points at)
- For **YouTube URLs**, unless `--no-transcriptapi` is passed, sends the video URL to TranscriptAPI (`transcriptapi.com/api/v2`) when `TRANSCRIPTAPI_API_KEY` is set, requesting the transcript before falling back to yt-dlp captions/Whisper — see "YouTube transcripts (TranscriptAPI)" above. No audio or video bytes are sent, only the URL and language preference
- Runs `ffmpeg` / `ffprobe` locally to extract frames as JPEGs and, when an API Whisper backend is used, a mono 16 kHz audio clip
- When `--whisper local` is selected (or auto-picked because a GPU is available), runs faster-whisper / CTranslate2 entirely on the user's machine — no network call, no upload
- Sends the extracted audio clip to Groq's Whisper API (`api.groq.com/openai/v1/audio/transcriptions`) when `GROQ_API_KEY` is set and the local backend isn't used
- Sends the extracted audio clip to OpenAI's audio transcription API (`api.openai.com/v1/audio/transcriptions`) when `OPENAI_API_KEY` is set and the local backend isn't used, or when `--whisper openai` is forced
- When `--whisper assemblyai` is forced, uploads the extracted audio clip to AssemblyAI (`api.assemblyai.com`) and requests automatic speaker diarization + language detection. Only runs when the user explicitly requests this backend
- When `--backend gemini` is used, uploads the entire video file to Google's Gemini Files API (`generativelanguage.googleapis.com`) and sends a `generateContent` request with the user's question. For YouTube URLs, the URL is passed to Gemini and Google fetches the video server-side instead of uploading it ourselves
- When `--backend openrouter` is used, base64-encodes the extracted frames and sends them (with the transcript and question) to OpenRouter's chat completions endpoint (`openrouter.ai`) for the vision model, and — when no captions are found — sends the extracted audio clip to OpenRouter's audio model, falling back to `mistralai/voxtral-mini-transcribe` (also via OpenRouter) and then Groq's Whisper API if both OpenRouter legs fail
- Runs Tesseract locally (via `pytesseract`) over the extracted frames for OCR text detection (no network call); disable with `--no-ocr`
- Writes the downloaded video, frames, audio, and an intermediate transcript to a working directory under the system temp dir (or `--out-dir` if specified) so Claude can `Read` them
- Reads / creates `~/.config/watch/.env` (mode `0600`) to store API key(s) and a `SETUP_COMPLETE` marker. As a fallback, also reads `.env` in the current working directory
- On first use of `--whisper local`, downloads the chosen faster-whisper model from HuggingFace into the user's HuggingFace cache (`~/.cache/huggingface/hub`)

**What this skill does NOT do:**
- Does not upload the video itself to any API in `--backend claude` mode — only the extracted audio goes out, and only when an API Whisper backend is in use
- Does not upload anything when `--whisper local` is the active backend — the audio stays on disk and is processed entirely on-device
- The full video only leaves the machine when `--backend gemini` is explicitly selected (Files API upload, or a YouTube URL passed through for Google to fetch server-side); `--backend openrouter` sends extracted frames + audio, never the raw video file
- Does not access any platform account (no login, no session cookies, no posting)
- Does not share API keys between providers (Groq → `api.groq.com`, OpenAI → `api.openai.com`, AssemblyAI → `api.assemblyai.com`, Gemini → `generativelanguage.googleapis.com`, OpenRouter → `openrouter.ai`, TranscriptAPI → `transcriptapi.com`)
- Does not log, cache, or write API keys to stdout, stderr, or output files
- Does not persist anything outside the working directory, `~/.config/watch/.env`, and the HuggingFace model cache (when local Whisper is used) — clean up the working directory when you're done (Step 5)

**Bundled scripts:** `scripts/watch.py` (entry point), `scripts/config.py` (shared config — `~/.config/watch/.env`), `scripts/download.py` (yt-dlp wrapper), `scripts/frames.py` (ffmpeg frame extraction + auto-fps + dedup), `scripts/scenes.py` (PySceneDetect wrapper + midpoint picker), `scripts/speech.py` (speech-window detection + two-pass sampling), `scripts/ocr.py` (Tesseract OCR over frames), `scripts/transcribe.py` (VTT caption parsing + speaker-aware formatting), `scripts/transcriptapi.py` (TranscriptAPI YouTube-transcript client), `scripts/whisper.py` (Groq / OpenAI clients + backend resolver + auto-chunking), `scripts/whisper_local.py` (faster-whisper / GPU client), `scripts/whisper_assemblyai.py` (AssemblyAI client with speaker diarization), `scripts/gemini.py` (Gemini multimodal video client — `--backend gemini`), `scripts/openrouter.py` (OpenRouter vision + audio client — `--backend openrouter`), `scripts/setup.py` (preflight + installer)

Review scripts before first use to verify behavior.
