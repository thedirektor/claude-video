---
name: watch
description: Watch a video (URL or local path). Downloads with yt-dlp, extracts auto-scaled frames with ffmpeg via scene detection + OCR, pulls the transcript from captions or Whisper (local GPU via faster-whisper, Groq, or OpenAI), and hands the result to Claude so it can answer questions about what's in the video.
argument-hint: "<video-url-or-path> [question]"
allowed-tools: Bash, Read, AskUserQuestion
homepage: https://github.com/bradautomates/claude-video
repository: https://github.com/bradautomates/claude-video
author: bradautomates
license: MIT
user-invocable: true
---

# /watch — Claude watches a video

You don't have a video input; this skill gives you one. A Python script downloads the video, extracts frames as JPEGs, gets a timestamped transcript (native captions first, then Whisper API as fallback), and prints frame paths. You then `Read` each frame path to see the images and combine them with the transcript to answer the user.

## Step 0 — Setup preflight (runs every `/watch` invocation, silent on success)

**Python interpreter:** every `python3 ...` command in this skill is for macOS/Linux. On **Windows**, substitute `python` — the `python3` command on Windows is the Microsoft Store stub and will not run the script.

Before every `/watch` run, verify that dependencies and an API key are in place:

```bash
python3 "${CLAUDE_SKILL_DIR}/scripts/setup.py" --check
```

This is a <100ms lookup. On exit 0, the script emits **nothing** — proceed to Step 1 without comment. **Do NOT announce "setup is complete" to the user** — they don't need a status message on every turn. The only acceptable user-visible output from Step 0 is when remediation is required.

On non-zero exit, follow the table:

| Exit | Meaning | Action |
|------|---------|--------|
| `2` | Missing binaries (`ffmpeg` / `ffprobe` / `yt-dlp`) | Run installer |
| `3` | No Whisper API key | Run installer to scaffold `.env`, then ask user for a key |
| `4` | Both missing | Run installer, then ask for a key |

The installer is idempotent — safe to re-run:

```bash
python3 "${CLAUDE_SKILL_DIR}/scripts/setup.py"
```

On macOS with Homebrew, it auto-installs `ffmpeg` and `yt-dlp`. On Linux/Windows, it prints the exact install commands for the user to run. It scaffolds `~/.config/watch/.env` with commented placeholders at `0600` perms, and writes `SETUP_COMPLETE=true` once deps + a key are in place so the next session knows this user has already been through the wizard.

**If an API key is still missing after install:** use `AskUserQuestion` to ask the user whether they have a Groq API key (preferred — cheaper, faster) or an OpenAI key. Then write it into `~/.config/watch/.env` — set the matching `GROQ_API_KEY=...` or `OPENAI_API_KEY=...` line. If they don't want to set up Whisper, proceed with `--no-whisper` and tell them videos without native captions will come back frames-only.

**Structured mode (optional):** `python3 "${CLAUDE_SKILL_DIR}/scripts/setup.py" --json` emits `{status, first_run, missing_binaries, whisper_backend, has_api_key, config_file, platform}` where `status` is one of `ready | needs_install | needs_key | needs_install_and_key`. Use this when you need to branch on specifics (e.g. "is this the user's very first run?" → `first_run: true`).

Within a single session, you can skip Step 0 on follow-up `/watch` calls — once `--check` returned 0, nothing about the environment changes between turns.

## When to use

- User pastes a video URL (YouTube, Vimeo, X, TikTok, Twitch clip, most yt-dlp-supported sites) and asks about it.
- User points at a local video file (`.mp4`, `.mov`, `.mkv`, `.webm`, etc.) and asks about it.
- User types `/watch <url-or-path> [question]`.

## Recommended limits

- **Best accuracy: videos under 10 minutes.** Frame coverage scales inversely with duration.
- **Hard caps: 100 frames total and 2 fps.** Token cost grows with frame count, so the script targets a frame budget by duration (and never exceeds 2 fps even when the budget would imply more):
  - ≤30s → ~1-2 fps (up to 30 frames)
  - 30s-1min → ~40 frames
  - 1-3min → ~60 frames
  - 3-10min → ~80 frames
  - \>10min → 100 frames, sparsely spaced (warning printed)
- If the user hands you a long video, consider asking whether they want a specific section before burning tokens on a sparse scan.

## How to invoke

**Step 1 — parse the user input.** Separate the video source (URL or path) from any question the user asked. Example: `/watch https://youtu.be/abc what language is this in?` → source = `https://youtu.be/abc`, question = `what language is this in?`.

**Step 2 — run the watch script.** Pass the source verbatim. Do not shell-escape it yourself beyond normal quoting:

```bash
python3 "${CLAUDE_SKILL_DIR}/scripts/watch.py" "<source>"
```

Optional flags:

**Range / budget**
- `--start T` / `--end T` — focus on a section. Accepts `SS`, `MM:SS`, or `HH:MM:SS`. When either is set, fps auto-scales denser (see "Focusing on a section" below).
- `--max-frames N` — lower the cap for tighter token budget (e.g. `--max-frames 40`).
- `--resolution W` — frame width in px (default 512; bump to 1024 only if the user needs to read on-screen text).
- `--fps F` — override auto-fps (clamped to 2 fps max). **Disables scene detection and two-pass sampling** since both rely on auto-fps.
- `--out-dir DIR` — keep working files somewhere specific (default: auto-generated tmp dir).

**Frame sampling**
- `--no-scene-detect` — skip PySceneDetect and use fixed-fps extraction (the pre-scene-detect behavior). Use when scenes look like a poor proxy for "interesting moments" — talking-head video with no cuts, screen recordings of static UIs.
- `--scene-threshold F` — ContentDetector threshold (default `27.0`). Lower = more cuts. Bump to `35-40` for low-cut talking heads, drop to `20` for fast-cut promo content.
- `--two-pass` / `--no-two-pass` — distribute the frame budget proportionally to speech windows from the transcript (70% inside speech, 30% outside). **Default ON** when a transcript is available. Two-pass is what makes a muted-video + separate-VO workflow concentrate frames on the moments the VO is actually talking about.

**Audio / transcription**
- `--audio FILE` — separate audio file (mp3/wav/m4a) to transcribe instead of the video's own audio track. Use when the video is muted and the voiceover ships as a separate ElevenLabs/recorded file. Cannot combine with `--no-whisper`.
- `--whisper groq|openai|local` — force a specific Whisper backend.
  - `groq` — `whisper-large-v3` via Groq API. Cheap, fast, needs `GROQ_API_KEY`.
  - `openai` — `whisper-1` via OpenAI API. Needs `OPENAI_API_KEY`.
  - `local` — runs faster-whisper on the local GPU. No API key, no upload, no rate limit — but needs an NVIDIA GPU and faster-whisper installed (see "Local Whisper" below).
  - **Default:** auto-pick `local` if available, else `groq`, else `openai`.
- `--whisper-model NAME` — faster-whisper model size for the local backend. Choices: `tiny | base | small | medium | large-v2 | large-v3`. Default `large-v3`. Ignored for groq/openai. See the model table below.
- `--no-whisper` — disable Whisper entirely (frames-only if no captions).

**OCR**
- `--no-ocr` — disable the OCR pass. By default, after frames are extracted the script runs Tesseract over them (lang=`spa+eng`) and re-extracts text-heavy frames at 1024px so on-screen text stays legible. Disable when text doesn't matter (silent action footage, abstract content) to save a few seconds.

### Focusing on a section (higher frame rate)

When the user asks about a specific moment — "what happens at the 2 minute mark?", "zoom into 0:45 to 1:00", "the first 10 seconds" — pass `--start` and/or `--end`. The script switches to focused-mode budgets, which are denser than full-video budgets (still capped at 2 fps):

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
python3 "${CLAUDE_SKILL_DIR}/scripts/watch.py" video.mp4 --start 50 --end 60

# Zoom into 2:15 → 2:45 at 3 fps (90 frames)
python3 "${CLAUDE_SKILL_DIR}/scripts/watch.py" "$URL" --start 2:15 --end 2:45 --fps 3

# From 1h12m to the end of the video
python3 "${CLAUDE_SKILL_DIR}/scripts/watch.py" "$URL" --start 1:12:00
```

**Step 3 — Read every frame path the script lists.** The Read tool renders JPEGs directly as images for you. Read all frames in a single message (parallel tool calls) so you see them together. The frames are in chronological order with a `t=MM:SS` timestamp so you can align them to the transcript.

**Step 4 — answer the user.** You now have two streams of evidence:
- **Frames** — what's on screen at each timestamp
- **Transcript** — what's said at each timestamp. The report's header shows the source (`captions` = yt-dlp pulled native subs; `whisper (groq)` or `whisper (openai)` = transcribed by API).

If the user asked a specific question, answer it directly citing timestamps. If they didn't ask anything, summarize what happens in the video — structure, key moments, notable visuals, spoken content.

**Step 5 — clean up.** The script prints a working directory at the end. If the user isn't going to ask follow-ups about this video, delete it with `rm -rf <dir>`. If they might, leave it in place.

## Transcription

The script gets a timestamped transcript via three possible backends, in priority order:

1. **Native captions (free, preferred).** yt-dlp pulls manual or auto-generated subtitles from the source platform if available.
2. **Local Whisper via faster-whisper (no API call).** If no captions came back AND faster-whisper + a CUDA GPU are available, the script runs Whisper directly on the user's GPU. No upload, no key, no rate limit. See "Local Whisper" below for setup.
3. **Whisper API fallback.** If local isn't available, the script extracts audio (`ffmpeg -vn -ac 1 -ar 16000 -b:a 64k`, ~0.5 MB/min) and uploads it to whichever Whisper API has a key configured:
   - **Groq** — `whisper-large-v3`. Preferred API default: cheaper, faster. Get a key at console.groq.com/keys.
   - **OpenAI** — `whisper-1`. Fallback. Get a key at platform.openai.com/api-keys.

API keys live in `~/.config/watch/.env`. Auto-selection priority: `local` → `groq` → `openai`. Force a specific backend with `--whisper local|groq|openai`. Use `--no-whisper` to skip transcription entirely.

The report header shows the backend used:
- `via captions` — yt-dlp pulled native subs
- `via whisper (local, large-v3)` — local GPU
- `via whisper (groq)` / `via whisper (openai)` — API
- `via whisper (..., --audio)` suffix — `--audio FILE` was used to transcribe a separate VO track

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

## Workflow examples

**Muted product video + separate ElevenLabs voiceover** — the case `/watch` was extended to support. The video is silent stock footage; the VO ships as a separate `.mp3` (often AI-generated). `--audio` retargets transcription to the VO file, two-pass sampling is auto-enabled, and frames concentrate on the moments the VO is actually narrating:

```bash
python3 "${CLAUDE_SKILL_DIR}/scripts/watch.py" \
    "/path/to/Product Video.mp4" \
    --audio "/path/to/ElevenLabs_VO.mp3"
```

The report header reads `Transcript: 12 segments (via whisper (local, large-v3, --audio))` and frames inside speech windows get the `[speech]` tag while frames outside get `[silent]`. Two-pass distribution (default 70/30 speech/silent) is what makes the frame budget land where the narration actually is, instead of evenly across the video.

**Talking head, GPU-only transcription, smaller model:**
```bash
python3 "${CLAUDE_SKILL_DIR}/scripts/watch.py" speaker.mp4 \
    --whisper local --whisper-model medium --no-scene-detect
```
`--no-scene-detect` is appropriate here because a single-camera talking-head video has no real cuts, so PySceneDetect would just fall through to fps anyway. `medium` transcribes a clean voice as well as `large-v3` and uses ~half the VRAM.

**Fast-cut promo / ad creative — bias toward more frames at scene boundaries:**
```bash
python3 "${CLAUDE_SKILL_DIR}/scripts/watch.py" promo.mp4 \
    --scene-threshold 20 --no-ocr
```
Lowering the threshold below the 27.0 default captures more cuts; `--no-ocr` saves a couple seconds when the content is visual (no slides, no UI text).

## Windows compatibility

This fork has been tested on **Windows 11 + Python 3.14** (and the Bash tool's PowerShell shell). Notes:

- **UTF-8 encoding fix is already applied to all scripts.** Each Python file in `scripts/` reconfigures `sys.stdout` / `sys.stderr` to UTF-8 at startup, so non-ASCII content (Spanish transcripts, em-dashes, accented filenames) doesn't crash with `UnicodeEncodeError` on Windows's default cp1252 console.
- **Use `python` not `python3`.** On Windows the `python3` command typically resolves to the Microsoft Store stub; the skill's docs use `python3` for Unix conventions but on Windows you should substitute `python`.
- **Tesseract for OCR** must be installed separately. Default install path on Windows is `C:\Program Files\Tesseract-OCR` — make sure that directory is on `PATH` so `pytesseract` can find `tesseract.exe`. Install via `winget install UB-Mannheim.TesseractOCR` or grab the [installer](https://github.com/UB-Mannheim/tesseract/wiki). For Spanish OCR you also need the `spa.traineddata` language pack — bundled by default in the Mannheim installer.
- **Local Whisper DLL discovery is automatic.** `whisper_local.py` calls `os.add_dll_directory()` for the bundled cuBLAS / cuDNN wheels at import time, so `--whisper local` works out of the box after `pip install faster-whisper nvidia-cublas-cu12 nvidia-cudnn-cu12`. The manual PATH edit is only needed as a fallback for non-standard layouts (see "Local Whisper" above).
- **Long paths.** Some yt-dlp downloads produce long filenames; if you hit "filename too long" errors, enable Win32 long paths via the Group Policy editor or pass `--out-dir` to a short path like `D:\w`.

## Failure modes and handling

- **Setup preflight failed** → run `python3 "${CLAUDE_SKILL_DIR}/scripts/setup.py"` (auto-installs ffmpeg/yt-dlp via brew on macOS, scaffolds the `.env`). For API key, ask the user via `AskUserQuestion` and write it to `~/.config/watch/.env`.
- **No transcript available** → captions missing AND no Whisper backend usable (no GPU + faster-whisper, no API key, or all three failed). Script prints a hint pointing to setup. Proceed frames-only and tell the user.
- **Long video warning printed** → acknowledge it in your answer. Offer to re-run focused on a specific section via `--start`/`--end` rather than a sparse full-video scan.
- **Download fails** → yt-dlp's error goes to stderr. If it's a login-required or region-locked video, tell the user plainly; do not keep retrying.
- **Whisper API request fails** → the error is printed to stderr (likely: invalid key, rate limit, or 25 MB upload limit on a very long video). The report will say "none available" for transcript. Retry options: `--whisper openai` if Groq failed (or vice versa), or `--whisper local` if a GPU is available.
- **`--whisper local` requested but unavailable** → the script hard-errors with the install hint (`pip install faster-whisper nvidia-cublas-cu12 nvidia-cudnn-cu12`). Auto-mode falls through to Groq/OpenAI silently instead.
- **OCR unavailable** → `pytesseract` or the `tesseract` binary missing. Report shows `OCR: unavailable (...)`. Either install Tesseract (Windows: `C:\Program Files\Tesseract-OCR` plus the `spa` language pack) or pass `--no-ocr` to silence the warning.

## Token efficiency

This skill burns tokens primarily on frames. Order of magnitude:
- 80 frames at 512px wide is roughly 50-80k image tokens depending on aspect ratio.
- The transcript is cheap (a few thousand tokens at most for a 10-minute video).
- Bumping `--resolution` to 1024 roughly quadruples the image tokens per frame. Only do it when necessary.

If you already watched a video this session and the user asks a follow-up, do **not** re-run the script — you already have the frames and transcript in context. Just answer from what you have.

## Security & Permissions

**What this skill does:**
- Runs `yt-dlp` locally to download the video and pull native captions when the source supports them (public data; the request goes directly to whatever host the URL points at)
- Runs `ffmpeg` / `ffprobe` locally to extract frames as JPEGs and, when an API Whisper backend is used, a mono 16 kHz audio clip
- When `--whisper local` is selected (or auto-picked because a GPU is available), runs faster-whisper / CTranslate2 entirely on the user's machine — no network call, no upload
- Sends the extracted audio clip to Groq's Whisper API (`api.groq.com/openai/v1/audio/transcriptions`) when `GROQ_API_KEY` is set and the local backend isn't used
- Sends the extracted audio clip to OpenAI's audio transcription API (`api.openai.com/v1/audio/transcriptions`) when `OPENAI_API_KEY` is set and the local backend isn't used, or when `--whisper openai` is forced
- Runs Tesseract locally (via `pytesseract`) over the extracted frames for OCR text detection (no network call); disable with `--no-ocr`
- Writes the downloaded video, frames, audio, and an intermediate transcript to a working directory under the system temp dir (or `--out-dir` if specified) so Claude can `Read` them
- Reads / creates `~/.config/watch/.env` (mode `0600`) to store the Whisper API key(s) and a `SETUP_COMPLETE` marker. As a fallback, also reads `.env` in the current working directory
- On first use of `--whisper local`, downloads the chosen faster-whisper model from HuggingFace into the user's HuggingFace cache (`~/.cache/huggingface/hub`)

**What this skill does NOT do:**
- Does not upload the video itself to any API — only the extracted audio goes out, and only when an API Whisper backend is in use
- Does not upload anything when `--whisper local` is the active backend — the audio stays on disk and is processed entirely on-device
- Does not access any platform account (no login, no session cookies, no posting)
- Does not share API keys between providers (Groq key only goes to `api.groq.com`, OpenAI key only goes to `api.openai.com`)
- Does not log, cache, or write API keys to stdout, stderr, or output files
- Does not persist anything outside the working directory, `~/.config/watch/.env`, and the HuggingFace model cache (when local Whisper is used) — clean up the working directory when you're done (Step 5)

**Bundled scripts:** `scripts/watch.py` (entry point), `scripts/download.py` (yt-dlp wrapper), `scripts/frames.py` (ffmpeg frame extraction + auto-fps), `scripts/scenes.py` (PySceneDetect wrapper + midpoint picker), `scripts/speech.py` (speech-window detection + two-pass sampling), `scripts/ocr.py` (Tesseract OCR over frames), `scripts/transcribe.py` (VTT caption parsing), `scripts/whisper.py` (Groq / OpenAI clients + backend resolver), `scripts/whisper_local.py` (faster-whisper / GPU client), `scripts/setup.py` (preflight + installer)

Review scripts before first use to verify behavior.
