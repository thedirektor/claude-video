# /watch

**Give Claude the ability to watch any video.**

Claude Code:
```
/plugin marketplace add thedirektor/claude-video
/plugin install watch@claude-video
```

claude.ai (web): [download `watch.skill`](https://github.com/thedirektor/claude-video/releases/latest) and drop it into Settings → Capabilities → Skills.

Codex / generic skills:
```bash
git clone https://github.com/thedirektor/claude-video.git ~/.codex/skills/watch
```

Zero config to start — `yt-dlp` and `ffmpeg` install on first run via `brew` on macOS (Linux/Windows print exact commands). Captions cover most public videos for free. For caption-less videos, the script picks the best available Whisper backend automatically: a **local GPU via faster-whisper** if one is set up, then **Groq** (`whisper-large-v3`), then **OpenAI** (`whisper-1`).

---

Claude can read a webpage, run a script, browse a repo. What it can't do, out of the box, is *watch a video*. You paste a YouTube link and it has to either guess from the title or pull a transcript that's missing 90% of what's on screen.

With Claude Video `/watch` you can paste a URL or a local path, ask a question, and Claude downloads the video, extracts frames at an auto-scaled rate, pulls a timestamped transcript (free captions when available, Whisper API as fallback), and `Read`s every frame as an image. By the time it answers, it has *seen* the video and *heard* the audio.

```
/watch https://youtu.be/dQw4w9WgXcQ what happens at the 30 second mark?
```

## Why this exists

I built this because I'm constantly using video to keep up with content. If I see a YouTube video that's blowing up, I want to know how the creator structured the hook — what's on screen in the first 3 seconds, what they said, why it worked. That used to mean watching it myself with a notepad. Now I just paste the URL and ask.

The other half is summarization. Most YouTube videos don't deserve 20 minutes of my attention. I hand the URL to Claude, it pulls the transcript, and tells me what actually happened. If the visual matters, frames come along too. If it's a podcast or a talking head, transcript is enough.

Claude is great at reading and synthesizing — but until now, video was the one input I couldn't hand it. Pasting a YouTube link got you nothing useful. `/watch` closes that gap.

## What people actually use it for

**Analyze someone else's content.** `/watch https://youtu.be/<viral-video> what hook did they open with?` Claude looks at the first frames, reads the opening transcript, breaks down the structure. Same for ad creative, competitor launches, podcast intros, anything where the *how* matters as much as the *what*.

**Diagnose a bug from a video.** Someone sends you a screen recording of something broken. `/watch bug-repro.mov what's going wrong?` Claude watches the recording, finds the frame where the issue appears, describes what's on screen, often catches the cause without you ever opening the file.

**Summarize a video.** `/watch https://youtu.be/<long-thing> summarize this` does the obvious thing — pulls the structure, the key moments, what was actually said and shown. Faster than watching at 2x.

## How it works

1. **You paste a video and a question.** URL (anything yt-dlp supports — YouTube, Loom, TikTok, X, Instagram, plus a few hundred more) or a local path (`.mp4`, `.mov`, `.mkv`, `.webm`).
2. **`yt-dlp` downloads it.** For URLs, into a temp working directory. For local files, no download — just probed in place.
3. **`ffmpeg` extracts frames at an auto-scaled rate.** The frame budget is duration-aware: ≤30s gets ~30 frames, 30-60s gets ~40, 1-3min gets ~60, 3-10min gets ~80, longer gets 100 sparsely. Hard ceilings: 2 fps, 100 frames. JPEGs at 512px wide by default — bump with `--resolution 1024` if Claude needs to read on-screen text.
4. **The transcript comes from one of three places.** First try: `yt-dlp` pulls native captions (manual or auto-generated) from the source — free, instant, accurate-ish. Fallback: run Whisper. The script auto-selects the best available backend: **local GPU via faster-whisper** if installed (no API call, no upload, ~13× realtime on an RTX 2080 Ti), then **Groq** (`whisper-large-v3`), then **OpenAI** (`whisper-1`). Force a specific one with `--whisper local|groq|openai`. When transcription operates on a separate voiceover file (a muted product video plus an ElevenLabs `.mp3`, say), pass `--audio FILE` and the transcript is pulled from the VO instead of the video's own track.
5. **Frames + transcript are handed to Claude.** The script prints frame paths with `t=MM:SS` markers and the transcript with timestamps. Claude `Read`s each frame in parallel — JPEGs render directly as images in its context.
6. **Claude answers grounded in what's actually on screen and in the audio.** Not "based on the description" or "according to the title." It saw the frames. It heard the transcript. It answers the way someone who watched the video would.
7. **Cleanup.** The script prints a working directory at the end. If you're not asking follow-ups, Claude removes it.

## Frame budget — why it matters

Token cost is dominated by frames. Every frame is an image; image tokens add up fast. The script's auto-fps logic exists so you don't blow your context budget on a sparse scan of a 30-minute video that would have been better answered by a focused 30-second window.

| Duration | Default frame budget | What you get |
|----------|---------------------|--------------|
| ≤30 s | ~30 frames | Dense — basically every key moment |
| 30 s - 1 min | ~40 frames | Still dense |
| 1 - 3 min | ~60 frames | Comfortable |
| 3 - 10 min | ~80 frames | Sparse but workable |
| > 10 min | 100 frames | "Sparse scan" warning — re-run focused |

When the user names a moment ("around 2:30", "the last 30 seconds", "from 0:45 to 1:00"), pass `--start` / `--end`. Focused mode gets denser per-second budgets, capped at 2 fps. Far more useful than a sparse pass over the whole thing.

## Install

| Surface | Install |
|---------|---------|
| **Claude Code** | `/plugin marketplace add thedirektor/claude-video` then `/plugin install watch@claude-video` |
| **claude.ai** (web) | [Download `watch.skill`](https://github.com/thedirektor/claude-video/releases/latest) → Settings → Capabilities → Skills → `+` |
| **Codex** | `git clone https://github.com/thedirektor/claude-video.git ~/.codex/skills/watch` |
| **Manual / dev** | `git clone https://github.com/thedirektor/claude-video.git ~/.claude/skills/watch` |

### Claude Code

```
/plugin marketplace add thedirektor/claude-video
/plugin install watch@claude-video
```

Update later with `/plugin update watch@claude-video`.

### claude.ai (web)

1. [Download `watch.skill`](https://github.com/thedirektor/claude-video/releases/latest) from the latest release.
2. Go to Settings → Capabilities → Skills.
3. Click `+` and drop the file in.

Enable "Code execution and file creation" under Capabilities first — the skill shells out to `ffmpeg` and `yt-dlp`, so it won't run without it.

### Codex

```bash
git clone https://github.com/thedirektor/claude-video.git ~/.codex/skills/watch
```

### Manual (developer)

```bash
git clone https://github.com/thedirektor/claude-video.git ~/.claude/skills/watch
```

## First run

On the first `/watch` call, the skill runs `scripts/setup.py --check`. If `ffmpeg` / `yt-dlp` aren't on your PATH, or no Whisper API key is set, it walks you through fixing it:

- **macOS** — auto-runs `brew install ffmpeg yt-dlp`.
- **Linux** — prints the exact `apt` / `dnf` / `pipx` commands.
- **Windows** — prints the `winget` / `pip` commands.
- **API key** — scaffolds `~/.config/watch/.env` (mode `0600`) with commented placeholders for `GROQ_API_KEY` (preferred) and `OPENAI_API_KEY`.

After setup, preflight is silent and `/watch` just works. The check is a sub-100ms lookup, so it doesn't slow you down on subsequent runs.

## Bring your own keys (or run Whisper locally)

Captions cover the majority of public videos for free. The Whisper fallback only kicks in when a video genuinely has no caption track — typically local files, TikToks, some Vimeos, and the occasional caption-less YouTube upload.

| Capability | What you need | Cost |
|------------|---------------|------|
| Download + native captions | `yt-dlp` + `ffmpeg` | Free |
| Whisper, **local GPU** (preferred when available) | NVIDIA GPU + `pip install faster-whisper nvidia-cublas-cu12 nvidia-cudnn-cu12` | Free, no API call. ~13× realtime on RTX 2080 Ti with `large-v3` |
| Whisper, **Groq API** | [Groq API key](https://console.groq.com/keys) — `whisper-large-v3` | Cheap, fast |
| Whisper, **OpenAI API** | [OpenAI API key](https://platform.openai.com/api-keys) — `whisper-1` | Standard pricing |
| Whisper, **AssemblyAI** (speaker diarization) | [AssemblyAI key](https://www.assemblyai.com/dashboard/signup) + `pip install assemblyai` | ~$0.37/hr with diarization, $50 free credits to start |
| Disable Whisper entirely | `--no-whisper` | Free, frames-only when no captions |
| Native Gemini video backend (`--backend gemini`) | [Gemini API key](https://aistudio.google.com/apikey) + `pip install google-genai` | Free tier on `gemini-3.1-flash-lite`; paid for `2.5-flash` / `2.5-pro` |

The Whisper auto-selection priority is `local → groq → openai`. Force a specific Whisper backend with `--whisper local|groq|openai|assemblyai`. AssemblyAI is never auto-picked — request it explicitly when you need speaker diarization. See [Local Whisper](#local-whisper-faster-whisper-on-gpu) for the GPU setup and [Speaker diarization](#speaker-diarization-via-assemblyai) for the AssemblyAI flow. The Gemini backend is independent — picked with `--backend gemini` and described in its own section.

## Usage

```
/watch https://youtu.be/dQw4w9WgXcQ what happens at the 30 second mark?
/watch https://www.tiktok.com/@user/video/123 summarize this
/watch ~/Movies/screen-recording.mp4 when does the UI break?
/watch https://vimeo.com/123 what tools does she mention?
```

Focused on a specific section — denser frame budget, lower token cost:
```
/watch https://youtu.be/abc --start 2:15 --end 2:45
/watch video.mp4 --start 50 --end 60
/watch "$URL" --start 1:12:00            # from 1h12m to end
```

Muted product video + separate ElevenLabs voiceover (the case this fork was extended for) — `--audio` retargets transcription to the VO file, two-pass sampling auto-enables, and frames concentrate on the moments the VO is actually narrating:
```
/watch "Product Video.mp4" --audio "ElevenLabs_VO.mp3"
```

### Flags

All flags are forwarded to `scripts/watch.py`. The full set:

**Backend**

| Flag | Purpose |
|------|---------|
| `--backend claude\|gemini` | Pick the orchestrator. `claude` (default): local frame pipeline. `gemini`: short-circuit and send the video to Gemini's multimodal model. See [Native Gemini backend](#native-gemini-backend---backend-gemini). |
| `--gemini-model NAME` | Gemini model when `--backend gemini`. Choices: `gemini-3.1-flash-lite \| gemini-2.5-flash \| gemini-2.5-pro`. Default `gemini-3.1-flash-lite`. |
| *(positional)* `question` | Required for `--backend gemini` — the prompt to send to the model. Ignored for `--backend claude`. |

**Range / budget**

| Flag | Purpose |
|------|---------|
| `--start T` / `--end T` | Focus on a section. Accepts `SS`, `MM:SS`, or `HH:MM:SS`. |
| `--max-frames N` | Lower the frame cap for a tighter token budget (hard ceiling 100). |
| `--resolution W` | Frame width in px (default 512; bump to 1024 for slides / terminals / on-screen text). |
| `--fps F` | Override auto-fps (still capped at 2 fps). Disables scene detection and two-pass sampling. |
| `--out-dir DIR` | Keep working files somewhere specific (default: auto-generated tmp dir). |

**Frame sampling**

| Flag | Purpose |
|------|---------|
| `--no-scene-detect` | Skip PySceneDetect; use fixed-fps extraction. Right call for talking-head video with no cuts. |
| `--scene-threshold F` | ContentDetector threshold (default `27.0`). Lower = more cuts. |
| `--two-pass` / `--no-two-pass` | Distribute the frame budget proportionally to speech windows from the transcript (70 % inside speech, 30 % outside). Default ON when a transcript is available. |

**Audio / transcription**

| Flag | Purpose |
|------|---------|
| `--audio FILE` | Separate audio file (mp3/wav/m4a) to transcribe instead of the video's own audio track. Use for muted videos with separate VO. Cannot combine with `--no-whisper`. |
| `--whisper local\|groq\|openai\|assemblyai` | Force a specific Whisper backend. Default: auto-pick `local`, then `groq`, then `openai`. AssemblyAI is opt-in; see [Speaker diarization](#speaker-diarization-via-assemblyai). |
| `--whisper-model NAME` | Local-backend model size: `tiny\|base\|small\|medium\|large-v2\|large-v3` (default `large-v3`). Ignored for Groq / OpenAI / AssemblyAI. |
| `--diarize` / `--no-diarize` | Request speaker labels when the backend supports them (currently only AssemblyAI). Default ON. Ignored for local / groq / openai. |
| `--no-whisper` | Disable transcription entirely; frames only. |

**OCR**

| Flag | Purpose |
|------|---------|
| `--no-ocr` | Disable the OCR pass (Tesseract over each frame, lang=`spa+eng`, with adaptive 1024 px re-extraction of text-heavy frames). |

## Local Whisper (faster-whisper on GPU)

The `local` backend runs Whisper directly on an NVIDIA GPU via [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and CTranslate2. No API call, no 25 MB upload limit, no rate limit. Tested at ~13× realtime on an RTX 2080 Ti with `large-v3`.

**Prerequisites**

- NVIDIA GPU with CUDA 12 support and ≥ enough VRAM for the chosen model (see table below)
- Python 3.10+ (3.13 / 3.14 confirmed working)
- `faster-whisper`, `nvidia-cublas-cu12`, `nvidia-cudnn-cu12` installed via pip

**Install**

```bash
pip install faster-whisper nvidia-cublas-cu12 nvidia-cudnn-cu12
```

**Windows DLL discovery (handled automatically).** `scripts/whisper_local.py` runs `os.add_dll_directory()` at import time on Windows to register `<sys.prefix>\Lib\site-packages\nvidia\cublas\bin` and `…\nvidia\cudnn\bin` with the loader, so `import ctranslate2` finds the cuBLAS / cuDNN wheels with no environment changes. **No manual PATH editing required** for the standard `pip install` layout.

Linux / macOS-with-NVIDIA users don't hit this in the first place — pip's RPATH metadata handles DLL discovery on those platforms.

<details>
<summary>Manual PATH fallback (only if the auto-registration fails — non-standard install layout, conda site-packages outside <code>sys.prefix</code>, or importing <code>ctranslate2</code> from a tool that loads before <code>whisper_local</code>)</summary>

```powershell
$prefix = (python -c "import sys; print(sys.prefix)")
$cublas = "$prefix\Lib\site-packages\nvidia\cublas\bin"
$cudnn  = "$prefix\Lib\site-packages\nvidia\cudnn\bin"
[Environment]::SetEnvironmentVariable("PATH", "$cublas;$cudnn;" + [Environment]::GetEnvironmentVariable("PATH","User"), "User")
# Restart the terminal afterwards so the new PATH is picked up.
```
</details>

**Verify**

```bash
python -c "import ctranslate2; print('CUDA devices:', ctranslate2.get_cuda_device_count())"
```

A successful install prints `CUDA devices: 1` (or higher). Anything else — `0`, an `OSError`, an `ImportError` — means the runtime can't reach a GPU and `--whisper local` will fall through to the API backends in auto mode (or hard-error if you forced `--whisper local`).

**First-run model download.** On first use of `--whisper local`, the chosen faster-whisper model is downloaded into `~/.cache/huggingface/hub`. `large-v3` is ~3 GB; subsequent runs reuse the cached weights instantly.

**Model picker — `--whisper-model`**

| Model | Parameters | VRAM (fp16) | Relative speed | When to use |
|-------|-----------:|-------------|----------------|-------------|
| `tiny` | 39 M | ~1 GB | ~32× | Toy / smoke tests; clean speech only |
| `base` | 74 M | ~1 GB | ~16× | Lightweight, low-VRAM GPUs |
| `small` | 244 M | ~2 GB | ~6× | Decent quality, fits 4 GB cards |
| `medium` | 769 M | ~5 GB | ~2× | Solid quality, fits 8 GB cards |
| `large-v2` | 1550 M | ~10 GB | 1× | High quality, older v2 dataset |
| `large-v3` | 1550 M | ~10 GB | 1× | **Default.** Best quality; needs ≥ 10 GB VRAM |

Speed numbers are approximate ratios — actual realtime multiplier depends heavily on the GPU. Drop a tier if you hit OOM, or if `large-v3` is overkill for the content (a clean voiceover transcribes fine with `medium`).

## Speaker diarization (via AssemblyAI)

The Whisper-family backends (`local`, `groq`, `openai`) all do speech-to-text but none of them split a transcript by speaker — Whisper itself doesn't diarize. When you need `[Speaker A]` / `[Speaker B]` / … turns in the output, the script ships a fourth backend: `--whisper assemblyai`.

**Setup:** add `ASSEMBLYAI_API_KEY` to `~/.config/watch/.env` (get a key at [assemblyai.com/dashboard/signup](https://www.assemblyai.com/dashboard/signup) — $50 of free credits, ~135 hours of diarized audio) and install the SDK:

```bash
pip install assemblyai
```

**Usage:**

```bash
# Default behavior: diarization on, language auto-detected
python scripts/watch.py interview.mp4 --whisper assemblyai

# Skip diarization for sentence-level segments without speaker tags
python scripts/watch.py interview.mp4 --whisper assemblyai --no-diarize
```

The transcript output switches to a speaker-tagged layout when diarization is on:

```
[Speaker A] (0:00-0:05) Welcome back to the show. Today we're talking with…
[Speaker B] (0:05-0:09) Thanks for having me, glad to be here.
[Speaker A] (0:09-0:18) Let's jump in — when did you first realize…
```

**When it's worth it:**

| Content type | Why diarization helps |
|--------------|-----------------------|
| Interviews / podcasts | Separating host vs guest dialogue is the obvious win. |
| Multi-speaker UGC | Vlog conversations, panel clips, gameplay commentary with two voices. |
| Ads with presenter + voiceover | Keeps on-camera dialogue separate from off-camera narration. |
| Single-speaker content | Still works — everything tagged `[Speaker A]`. Useful when you want per-utterance start/end timing surfaced. |

Pricing: roughly **$0.37 per hour of audio** with diarization (text-only is cheaper). The `$50` of free credits covers ~135 hours of diarized audio at that rate, which is plenty for evaluation.

**Local diarization roadmap:** [WhisperX](https://github.com/m-bain/whisperX) combines faster-whisper with pyannote speaker embeddings for fully-local diarization, but the current release pins to Python ≤ 3.13. Adding it to this skill is on the roadmap pending upstream Python 3.14 support — until then, AssemblyAI is the only diarized path that ships here.

## Native Gemini backend (`--backend gemini`)

The default `claude` backend extracts frames + transcript locally so Claude can `Read` each frame in its own context. The `gemini` backend is the alternative: skip frame extraction entirely, hand the *whole* video (or a YouTube URL) to Gemini's multimodal model, and print Gemini's response directly. Best for one-shot full-video analyses where you want Gemini's native video understanding rather than Claude's frame-by-frame interpretation.

**Setup:** add `GEMINI_API_KEY` to `~/.config/watch/.env` (get a key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)) and install the SDK:

```bash
pip install google-genai
```

**Usage** — pass the question as the trailing positional argument:

```bash
# Local file: uploaded to Gemini Files API, polled until ACTIVE, then sent
python scripts/watch.py video.mp4 --backend gemini "Describe this video and timestamp on-screen text"

# YouTube URL: passed straight to Gemini, no yt-dlp download — fetched server-side
python scripts/watch.py "https://youtu.be/dQw4w9WgXcQ" --backend gemini "What happens here?"
```

Non-YouTube URLs (Vimeo, TikTok, etc.) are downloaded with yt-dlp first and uploaded via the Files API, since Gemini only ingests YouTube URLs natively.

### Model picker (`--gemini-model`)

| Model | Notes | When to pick |
|-------|-------|--------------|
| `gemini-3.1-flash-lite` | **Default.** Stable May 7 2026. 1M-token input / 65k output, multimodal (text/image/video/audio/PDF). Free-tier quota available. | General default — fastest, cheapest, still video-capable. |
| `gemini-2.5-flash` | Balanced. | When `3.1-flash-lite`'s quality isn't enough but `2.5-pro` is overkill — mid-length videos that need careful reasoning over the visuals. |
| `gemini-2.5-pro` | Highest quality, longest context (~2M tokens). | Very long videos or deep reasoning with extensive output. |

`gemini-2.0-flash` and `gemini-1.5-pro` are not in the picker — `1.5-pro` returns 404 on the current `v1beta` API, and `2.0-flash` has zero free-tier quota on most accounts.

**Output:** the script prints a report header with `Backend / Mode / Question`, then Gemini's full response. The frame-extraction / OCR / Whisper pipeline does not run.

## Windows compatibility

This fork has been tested on **Windows 11 + Python 3.14**. Notes:

- **UTF-8 encoding fix is already applied to all scripts.** Each Python file in `scripts/` reconfigures `sys.stdout` / `sys.stderr` to UTF-8 at startup, so non-ASCII content (Spanish transcripts, em-dashes, accented filenames) doesn't crash with `UnicodeEncodeError` on the default cp1252 console.
- **Use `python` not `python3`.** On Windows the `python3` command typically resolves to the Microsoft Store stub. The skill docs use `python3` for Unix conventions but on Windows substitute `python`.
- **Tesseract for OCR** must be installed separately. Default install path is `C:\Program Files\Tesseract-OCR`; make sure that directory is on `PATH` so `pytesseract` can find `tesseract.exe`. Install via `winget install UB-Mannheim.TesseractOCR` or grab the [installer](https://github.com/UB-Mannheim/tesseract/wiki). For Spanish OCR you also need the `spa.traineddata` language pack — bundled by default in the Mannheim installer.
- **Local Whisper DLL discovery is automatic.** `scripts/whisper_local.py` calls `os.add_dll_directory()` for the bundled cuBLAS / cuDNN wheels at import time, so `--whisper local` works out of the box after `pip install faster-whisper nvidia-cublas-cu12 nvidia-cudnn-cu12`. The manual PATH edit shown under [Local Whisper](#local-whisper-faster-whisper-on-gpu) is only needed as a fallback for non-standard install layouts.
- **Long paths.** Some yt-dlp downloads produce long filenames; if you hit "filename too long" errors, enable Win32 long paths via Group Policy or pass `--out-dir` to a short path like `D:\w`.

## Limits

- **Best accuracy: under 10 minutes.** Past that the script prints a "sparse scan" warning — re-run focused on the part you actually care about with `--start`/`--end`.
- **Hard caps: 2 fps, 100 frames.** Frame count drives token cost; the script enforces this even when the auto-fps math would imply higher.
- **Whisper upload limit: 25 MB.** At mono 16 kHz that's about 50 minutes of audio. Longer videos need either captions or `--start`/`--end` to a smaller window.
- **No private platforms.** This skill doesn't log into anything. Public URLs and local files only. If yt-dlp can't reach it without auth, neither can `/watch`.

## Structure

```
.
├── SKILL.md                 # skill contract — loaded by all three surfaces
├── scripts/
│   ├── watch.py             # entry point — orchestrates download → frames → transcript
│   ├── download.py          # yt-dlp wrapper
│   ├── frames.py            # ffmpeg frame extraction + auto-fps logic
│   ├── scenes.py            # PySceneDetect wrapper + scene-midpoint picker
│   ├── speech.py            # speech-window detection + two-pass sampling
│   ├── ocr.py               # Tesseract OCR over frames (lang=spa+eng) + adaptive upscale
│   ├── transcribe.py        # VTT caption parsing + dedupe
│   ├── whisper.py           # Groq / OpenAI HTTP clients + backend resolver
│   ├── whisper_local.py     # faster-whisper / GPU client (no network)
│   ├── whisper_assemblyai.py # AssemblyAI client with speaker diarization
│   ├── gemini.py            # Gemini multimodal video client (--backend gemini)
│   ├── setup.py             # preflight + installer
│   └── build-skill.sh       # build dist/watch.skill for claude.ai upload
├── hooks/                   # SessionStart status hook (Claude Code only)
├── .claude-plugin/          # plugin.json + marketplace.json (Claude Code)
├── .codex-plugin/           # codex packaging
└── .github/workflows/       # release.yml — auto-builds watch.skill on tag push
```

## Develop

```bash
# Build the claude.ai upload bundle:
bash scripts/build-skill.sh      # → dist/watch.skill
```

Releasing: tag `vX.Y.Z`, push the tag. The workflow builds `dist/watch.skill` and attaches it to the GitHub release.

See [CHANGELOG.md](CHANGELOG.md) for version history.

## Open source

MIT license.

Built on `yt-dlp`, `ffmpeg`, `pytesseract`, `PySceneDetect`, and Claude's multimodal `Read` tool. Whisper transcription via [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (local GPU), [Groq](https://groq.com), or [OpenAI](https://openai.com). Speaker diarization via [AssemblyAI](https://www.assemblyai.com). Native video understanding via [Gemini](https://ai.google.dev) when `--backend gemini` is selected.

---

[github.com/thedirektor/claude-video](https://github.com/thedirektor/claude-video) · [LICENSE](LICENSE)
