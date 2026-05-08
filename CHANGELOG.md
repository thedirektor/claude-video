# Changelog

All notable changes to `/watch` are documented here.

## [0.2.0] — 2026-05-08

Fork release covering everything added on top of upstream `bradautomates/claude-video v0.1.2`. Themes: smarter frame sampling (scenes, OCR, speech-aware two-pass), a separate-voiceover workflow, a local-GPU Whisper backend, and a real Windows story.

### Added

- **Scene-aware frame extraction.** `scripts/scenes.py` wraps PySceneDetect; the default sampler picks a frame at the midpoint of each detected scene instead of evenly distributing across the timeline. Falls back to fixed-fps when no cuts are detected.
- **OCR on frames.** `scripts/ocr.py` runs Tesseract (`lang=spa+eng`) over each extracted frame and includes detected text in the report. Frames flagged as text-heavy are re-extracted at 1024 px so on-screen text (slides, terminals, UI copy) stays legible when Claude `Read`s them.
- **Two-pass speech-aware sampling.** `scripts/speech.py` reads the transcript timing, builds speech windows, and distributes the frame budget 70 % inside speech / 30 % outside. The default behavior when a transcript is available; makes muted-video + voiceover content sample frames where the narration actually is.
- **`--audio FILE` flag.** Separate audio file (mp3/wav/m4a) to transcribe instead of the video's own track. Solves the muted product video + separate ElevenLabs VO workflow. Cannot combine with `--no-whisper`.
- **Local Whisper backend (`--whisper local`).** New `scripts/whisper_local.py` runs faster-whisper / CTranslate2 directly on an NVIDIA GPU — no API call, no upload, no rate limit, no 25 MB ceiling. Tested at ~13× realtime on an RTX 2080 Ti with `large-v3`. Auto-selected over Groq / OpenAI when the GPU is reachable.
- **Auto-registration of CUDA DLLs on Windows.** `whisper_local.py` calls `os.add_dll_directory()` on the bundled `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` wheels at import time, so `import ctranslate2` finds the runtime without any user `PATH` edits.
- **`--whisper-model NAME` flag.** Picks the faster-whisper model size for the local backend: `tiny | base | small | medium | large-v2 | large-v3` (default `large-v3`). Ignored for Groq / OpenAI.
- **`--no-scene-detect` / `--scene-threshold F`.** Disable PySceneDetect or tune the ContentDetector threshold (default `27.0`). Right call for talking-head video with no real cuts, or fast-cut promo content that needs a lower threshold.
- **`--two-pass` / `--no-two-pass`.** Toggle speech-aware budget distribution. ON by default when a transcript is available.
- **`--no-ocr` flag.** Skip the OCR pass + adaptive upscale for visual content where on-screen text doesn't matter.
- **`scripts/whisper.py::resolve_backend()`.** Three-way backend resolver — local / groq / openai — that handles forced selection (hard-error if unavailable) and auto-selection (silent fall-through).
- **Comprehensive Windows compatibility guide** in `SKILL.md` and `README.md`: tested config (Windows 11 + Python 3.14), `python` vs `python3` note, Tesseract install path with the `spa.traineddata` requirement, long-path workaround, and the local-Whisper auto-DLL registration story.

### Changed

- **Default Whisper backend selection** is now `local → groq → openai` (was `groq → openai`). `--whisper` accepts `local` as a third choice and the report header reflects the chosen backend (`whisper (local, large-v3)`, `whisper (groq, --audio)`, etc.).
- **Default sampling pipeline** is `two-pass → scenes → fps` (was `fps`). The two-pass branch only kicks in when a transcript is available; the scenes branch only when scene detection finds cuts; otherwise behavior matches the upstream fps fallback.
- **Frame report** annotates two-pass frames as `[speech]` / `[silent]` and includes any OCR-detected text inline so Claude can correlate visuals with on-screen captions / slides / UI copy.

### Fixed

- **UTF-8 encoding on all scripts.** Every Python file in `scripts/` now reconfigures `sys.stdout` / `sys.stderr` to UTF-8 at startup, preventing `UnicodeEncodeError` crashes on Windows's default cp1252 console when the transcript or filenames contain non-ASCII content (Spanish, em-dashes, accented paths). Builds on the v0.1.2 emoji-removal fix to cover all output paths.

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
