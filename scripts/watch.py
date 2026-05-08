#!/usr/bin/env python3
"""/watch entry point: download video, extract frames, parse transcript.

Prints a markdown report to stdout listing frame paths + transcript. Claude
then Reads each frame path to see the video.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

# Force UTF-8 on stdout/stderr so non-ASCII output (transcripts, accented
# paths, em-dashes) doesn't crash on Windows where the default is cp1252.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from download import download, is_url  # noqa: E402
from frames import (  # noqa: E402
    MAX_FPS,
    auto_fps,
    auto_fps_focus,
    extract,
    extract_at_timestamps,
    format_time,
    get_metadata,
    parse_time,
    reextract_frame,
)
from gemini import DEFAULT_MODEL as GEMINI_DEFAULT_MODEL  # noqa: E402
from gemini import VALID_MODELS as GEMINI_MODELS  # noqa: E402
from ocr import is_significant, run_ocr  # noqa: E402
from scenes import DEFAULT_THRESHOLD, detect_scenes, pick_midpoints  # noqa: E402
from speech import (  # noqa: E402
    DEFAULT_SPEECH_SHARE,
    compute_speech_windows,
    format_windows,
    two_pass_sample,
)
from transcribe import filter_range, format_transcript, parse_vtt  # noqa: E402
from whisper import resolve_backend, transcribe_video  # noqa: E402
from whisper_local import DEFAULT_MODEL as WHISPER_LOCAL_DEFAULT_MODEL  # noqa: E402
from whisper_local import VALID_MODELS as WHISPER_LOCAL_MODELS  # noqa: E402


HIRES_WIDTH = 1024


def _run_gemini_backend(args, work: Path) -> int:
    """Hand the entire video to Gemini for native multimodal analysis.

    Skips frame extraction, Whisper, OCR, scene detection, and two-pass
    sampling — Gemini ingests the video directly and answers the user's
    question. For YouTube URLs we pass the URL through to the model
    (Gemini fetches it server-side); for everything else we download (if
    needed) and upload to the Gemini Files API.
    """
    from gemini import generate_with_video, is_youtube_url

    question = " ".join(args.question).strip()
    if not question:
        raise SystemExit(
            "--backend gemini requires a question. Pass it as the trailing "
            'positional argument: `watch.py video.mp4 --backend gemini "Describe this video"`'
        )

    if args.audio:
        print(
            "[watch] --audio is ignored for --backend gemini (Gemini transcribes the "
            "video's own audio track natively).",
            file=sys.stderr,
        )

    source = args.source
    youtube = is_youtube_url(source)

    if youtube:
        gemini_source = source
        mode_label = "native YouTube URL (no download)"
    elif is_url(source):
        print("[watch] downloading via yt-dlp before Gemini upload…", file=sys.stderr)
        dl = download(source, work / "download")
        gemini_source = dl["video_path"]
        mode_label = "yt-dlp download → Gemini Files API upload"
    else:
        gemini_source = source
        mode_label = "local file → Gemini Files API upload"

    response_text = generate_with_video(
        source=gemini_source,
        question=question,
        model_name=args.gemini_model,
        is_youtube=youtube,
    )

    print()
    print("# watch: video report (Gemini backend)")
    print()
    print(f"- **Source:** {source}")
    print(f"- **Backend:** Gemini ({args.gemini_model})")
    print(f"- **Mode:** {mode_label}")
    print(f"- **Question:** {question}")
    print()
    print("## Gemini response")
    print()
    print(response_text)
    print()
    print("---")
    print(f"_Work dir: `{work}` — delete when done._")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="watch",
        description="Download a video, extract auto-scaled frames, and surface the transcript.",
    )
    ap.add_argument("source", help="Video URL or local file path")
    ap.add_argument(
        "question",
        nargs="*",
        default=[],
        help="Optional trailing question. Required for --backend gemini "
        "(passed to the model as the prompt). Ignored for --backend claude — "
        "Claude handles the question via its own conversation context.",
    )
    ap.add_argument(
        "--backend",
        choices=["claude", "gemini"],
        default="claude",
        help="claude (default): extract frames + transcript locally so Claude can `Read` them. "
        "gemini: skip frame extraction and Whisper, hand the whole video to Gemini's "
        "native multimodal model and print its response.",
    )
    ap.add_argument(
        "--gemini-model",
        choices=list(GEMINI_MODELS),
        default=GEMINI_DEFAULT_MODEL,
        help=f"Gemini model for --backend gemini (default {GEMINI_DEFAULT_MODEL}). "
        "Use gemini-2.5-flash for harder reasoning over the visual content, or "
        "gemini-2.5-pro for very long videos / extensive output. "
        "Ignored for --backend claude.",
    )
    ap.add_argument("--max-frames", type=int, default=80, help="Cap on frame count (default 80, hard max 100)")
    ap.add_argument("--resolution", type=int, default=512, help="Frame width in pixels (default 512)")
    ap.add_argument("--fps", type=float, default=None, help="Override auto-fps")
    ap.add_argument("--start", type=str, default=None, help="Range start (SS, MM:SS, or HH:MM:SS)")
    ap.add_argument("--end", type=str, default=None, help="Range end (SS, MM:SS, or HH:MM:SS)")
    ap.add_argument("--out-dir", type=str, default=None, help="Working directory (default: tmp)")
    ap.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Separate audio file (mp3/wav/m4a) to transcribe instead of the video's audio track. "
        "Useful when the video is muted and the VO ships as a separate file.",
    )
    ap.add_argument(
        "--no-whisper",
        action="store_true",
        help="Disable Whisper fallback. Report frames-only if no captions available.",
    )
    ap.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable OCR pass. Skips text detection on frames and the high-res re-extract.",
    )
    ap.add_argument(
        "--no-scene-detect",
        action="store_true",
        help="Skip PySceneDetect and use fixed-fps extraction (the pre-scene-detect behavior).",
    )
    ap.add_argument(
        "--scene-threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"ContentDetector threshold (default {DEFAULT_THRESHOLD}). Lower = more cuts.",
    )
    ap.add_argument(
        "--two-pass",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Distribute frame budget proportionally to speech windows from the transcript "
        "(70%% inside speech, 30%% outside). Default ON when a transcript is available.",
    )
    ap.add_argument(
        "--whisper",
        choices=["groq", "openai", "local", "assemblyai"],
        default=None,
        help="Force a specific Whisper backend. 'local' runs faster-whisper on the GPU "
        "(needs faster-whisper + CUDA). 'assemblyai' is paid (~$0.37/hr with diarization, "
        "$50 free credits) and adds automatic speaker labels to the transcript. "
        "Default: auto-pick local if available, else Groq, else OpenAI. "
        "AssemblyAI is never auto-picked — request it explicitly when you need diarization.",
    )
    ap.add_argument(
        "--whisper-model",
        choices=list(WHISPER_LOCAL_MODELS),
        default=WHISPER_LOCAL_DEFAULT_MODEL,
        help=f"faster-whisper model for the local backend (default {WHISPER_LOCAL_DEFAULT_MODEL}). "
        "Smaller models are faster but less accurate. Ignored for groq / openai / assemblyai.",
    )
    ap.add_argument(
        "--diarize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Request speaker diarization when the backend supports it (currently AssemblyAI). "
        "Default ON. Pass --no-diarize to skip speaker labels and get sentence-level segments. "
        "Ignored for local / groq / openai (none of those expose diarization here).",
    )
    args = ap.parse_args()

    max_frames = min(args.max_frames, 100)

    if args.audio and args.no_whisper:
        raise SystemExit(
            "--audio implies Whisper transcription of that file; cannot combine with --no-whisper. "
            "Drop one of the two flags."
        )
    audio_override: Path | None = None
    if args.audio:
        audio_override = Path(args.audio).expanduser().resolve()
        if not audio_override.exists():
            raise SystemExit(f"--audio file not found: {audio_override}")

    if args.out_dir:
        work = Path(args.out_dir).expanduser().resolve()
    else:
        work = Path(tempfile.mkdtemp(prefix="watch-"))
    work.mkdir(parents=True, exist_ok=True)
    print(f"[watch] working dir: {work}", file=sys.stderr)

    if args.backend == "gemini":
        return _run_gemini_backend(args, work)

    print(
        "[watch] downloading via yt-dlp…" if is_url(args.source) else "[watch] using local file…",
        file=sys.stderr,
    )
    dl = download(args.source, work / "download")
    video_path = dl["video_path"]

    meta = get_metadata(video_path)
    full_duration = meta["duration_seconds"]

    start_sec = parse_time(args.start)
    end_sec = parse_time(args.end)

    if start_sec is not None and start_sec < 0:
        raise SystemExit("--start must be non-negative")
    if end_sec is not None and start_sec is not None and end_sec <= start_sec:
        raise SystemExit("--end must be greater than --start")
    if full_duration > 0 and start_sec is not None and start_sec >= full_duration:
        raise SystemExit(f"--start {start_sec:.1f}s is past end of video ({full_duration:.1f}s)")

    effective_start = start_sec if start_sec is not None else 0.0
    effective_end = end_sec if end_sec is not None else full_duration
    effective_duration = max(0.0, effective_end - effective_start)
    focused = start_sec is not None or end_sec is not None

    if focused:
        fps, target = auto_fps_focus(effective_duration, max_frames=max_frames)
    else:
        fps, target = auto_fps(effective_duration, max_frames=max_frames)
    if args.fps is not None:
        fps = min(args.fps, MAX_FPS)
        target = max(1, int(round(fps * effective_duration)))

    scope = (
        f"{format_time(effective_start)}-{format_time(effective_end)} ({effective_duration:.1f}s)"
        if focused else f"full {effective_duration:.1f}s"
    )

    # ──────────────────────────────────────────────────────────────────
    # Transcript first: speech windows drive two-pass sampling, so we
    # need timing info before we pick frame timestamps.
    # ──────────────────────────────────────────────────────────────────
    transcript_segments: list[dict] = []
    transcript_text: str | None = None
    transcript_source: str | None = None

    if audio_override is None and dl.get("subtitle_path"):
        try:
            all_segments = parse_vtt(dl["subtitle_path"])
            transcript_segments = filter_range(all_segments, start_sec, end_sec) if focused else all_segments
            transcript_text = format_transcript(transcript_segments)
            transcript_source = "captions"
        except Exception as exc:
            print(f"[watch] subtitle parse failed: {exc}", file=sys.stderr)

    if not transcript_segments and not args.no_whisper:
        backend, api_key, error_hint = resolve_backend(args.whisper)
        if backend:
            whisper_input = str(audio_override) if audio_override else video_path
            audio_out = work / ("audio_override.mp3" if audio_override else "audio.mp3")
            try:
                if audio_override:
                    print(
                        f"[watch] transcribing separate audio file via {backend}: "
                        f"{audio_override.name}",
                        file=sys.stderr,
                    )
                all_segments, used_backend = transcribe_video(
                    whisper_input,
                    audio_out,
                    backend=backend,
                    api_key=api_key,
                    model_name=args.whisper_model if backend == "local" else None,
                    enable_diarization=args.diarize if backend == "assemblyai" else False,
                )
                transcript_segments = (
                    filter_range(all_segments, start_sec, end_sec) if focused else all_segments
                )
                transcript_text = format_transcript(transcript_segments)
                # Build a label like "whisper (local, large-v3, --audio)",
                # "whisper (assemblyai, diarized)", or "whisper (groq)".
                # Per-backend extras are folded into a single comma list so
                # the transcript header tells you exactly what produced it.
                if used_backend == "local":
                    backend_label = f"local, {args.whisper_model}"
                elif used_backend == "assemblyai":
                    backend_label = (
                        "assemblyai, diarized" if args.diarize else "assemblyai"
                    )
                else:
                    backend_label = used_backend
                transcript_source = (
                    f"whisper ({backend_label}, --audio)"
                    if audio_override
                    else f"whisper ({backend_label})"
                )
            except SystemExit as exc:
                print(f"[watch] whisper transcription failed: {exc}", file=sys.stderr)
        else:
            hint = error_hint or "no subtitles and no Whisper backend available"
            setup_py = SCRIPT_DIR / "setup.py"
            print(
                f"[watch] {hint} — run `python3 {setup_py}` to enable the Whisper fallback",
                file=sys.stderr,
            )

    # Speech windows are clipped to the focus range so two-pass distribution
    # respects --start/--end.
    speech_windows = compute_speech_windows(
        transcript_segments,
        range_start=effective_start,
        range_end=effective_end,
    )

    # ──────────────────────────────────────────────────────────────────
    # Sampling decision: two-pass > scene-detect > fps.
    # ──────────────────────────────────────────────────────────────────
    use_scenes = not args.no_scene_detect and args.fps is None
    scenes: list[tuple[float, float]] = []
    scene_count: int | None = None
    sampling_mode = "fps"
    speech_frame_count = 0
    non_speech_frame_count = 0
    frames: list[dict] = []

    if use_scenes:
        print(
            f"[watch] detecting scenes (threshold={args.scene_threshold}) over {scope}…",
            file=sys.stderr,
        )
        scenes = detect_scenes(
            video_path,
            threshold=args.scene_threshold,
            start_seconds=start_sec,
            end_seconds=end_sec,
        )
        scene_count = len(scenes)

    two_pass_active = (
        args.two_pass
        and bool(speech_windows)
        and args.fps is None
    )

    if two_pass_active:
        plan = two_pass_sample(
            range_start=effective_start,
            range_end=effective_end,
            speech_windows=speech_windows,
            scenes=scenes,
            max_frames=max_frames,
            speech_share=DEFAULT_SPEECH_SHARE,
        )
        timestamps = plan["timestamps"]
        speech_frame_count = plan["speech_count"]
        non_speech_frame_count = plan["non_speech_count"]
        if timestamps:
            print(
                f"[watch] two-pass sampling: {speech_frame_count} frames in speech "
                f"windows, {non_speech_frame_count} outside (max {max_frames})…",
                file=sys.stderr,
            )
            frames = extract_at_timestamps(
                video_path,
                work / "frames",
                timestamps=timestamps,
                resolution=args.resolution,
            )
            sampling_mode = "two-pass"

    if not frames and use_scenes and scenes:
        timestamps = pick_midpoints(scenes, max_frames=max_frames)
        print(
            f"[watch] {scene_count} scenes detected; extracting "
            f"{len(timestamps)} mid-scene frames…",
            file=sys.stderr,
        )
        frames = extract_at_timestamps(
            video_path,
            work / "frames",
            timestamps=timestamps,
            resolution=args.resolution,
        )
        sampling_mode = "scenes"

    if not frames:
        if use_scenes and scene_count == 0 and not two_pass_active:
            print(
                "[watch] no scene cuts detected — falling back to fixed-fps extraction.",
                file=sys.stderr,
            )
        print(f"[watch] extracting ~{target} frames at {fps:.3f} fps over {scope}…", file=sys.stderr)
        frames = extract(
            video_path,
            work / "frames",
            fps=fps,
            resolution=args.resolution,
            max_frames=max_frames,
            start_seconds=start_sec,
            end_seconds=end_sec,
        )
        sampling_mode = "fps"

    # ──────────────────────────────────────────────────────────────────
    # OCR + adaptive 1024px upscale on text-heavy frames (unchanged).
    # ──────────────────────────────────────────────────────────────────
    ocr_text: dict[str, str] = {}
    hires_count = 0
    if not args.no_ocr and frames:
        print(f"[watch] running OCR on {len(frames)} frames (lang=spa+eng)…", file=sys.stderr)
        ocr_text = run_ocr([f["path"] for f in frames])
        if ocr_text:
            hires_targets = [f for f in frames if is_significant(ocr_text.get(f["path"], ""))]
            if hires_targets and args.resolution < HIRES_WIDTH:
                print(
                    f"[watch] re-extracting {len(hires_targets)} text-heavy frames "
                    f"at {HIRES_WIDTH}px…",
                    file=sys.stderr,
                )
                for f in hires_targets:
                    if reextract_frame(
                        video_path,
                        Path(f["path"]),
                        f["timestamp_seconds"],
                        resolution=HIRES_WIDTH,
                    ):
                        hires_count += 1
            ocr_json_path = work / "ocr.json"
            ocr_json_path.write_text(
                json.dumps(ocr_text, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"[watch] OCR results saved to {ocr_json_path}", file=sys.stderr)

    info = dl.get("info") or {}

    print()
    print("# watch: video report")
    print()
    print(f"- **Source:** {args.source}")
    if audio_override is not None:
        print(f"- **Audio source:** `{audio_override}` (separate file via `--audio`)")
    if info.get("title"):
        print(f"- **Title:** {info['title']}")
    if info.get("uploader"):
        print(f"- **Uploader:** {info['uploader']}")
    print(f"- **Duration:** {format_time(full_duration)} ({full_duration:.1f}s)")
    if focused:
        print(
            f"- **Focus range:** {format_time(effective_start)} → {format_time(effective_end)} "
            f"({effective_duration:.1f}s)"
        )
    if meta.get("width") and meta.get("height"):
        print(f"- **Resolution:** {meta['width']}x{meta['height']} ({meta.get('codec') or 'unknown codec'})")

    mode = "focused" if focused else "full"
    if sampling_mode == "two-pass":
        scene_note = ""
        if scene_count and scene_count > 0:
            scene_note = f", {scene_count} scenes informing pick"
        print(
            f"- **Frames:** {len(frames)} extracted via two-pass "
            f"({speech_frame_count} in speech windows, {non_speech_frame_count} outside; "
            f"speech share {int(DEFAULT_SPEECH_SHARE * 100)}%{scene_note}, "
            f"max {max_frames}, {mode} mode)"
        )
    elif sampling_mode == "scenes" and scene_count is not None:
        suffix = ""
        if args.two_pass and not speech_windows and not args.no_whisper:
            suffix = " — two-pass requested but no transcript with timing was available"
        print(
            f"- **Frames:** {scene_count} scenes detected, "
            f"{len(frames)} extracted at scene midpoints "
            f"(max {max_frames}, threshold {args.scene_threshold}, {mode} mode){suffix}"
        )
    else:
        suffix = ""
        if not args.no_scene_detect and args.fps is None and scene_count == 0:
            suffix = " — scene-detect found no cuts, fell back to fps"
        print(
            f"- **Frames:** {len(frames)} @ {fps:.3f} fps, {mode} mode "
            f"(budget {target}, max {max_frames}){suffix}"
        )

    if hires_count:
        print(
            f"- **Frame size:** {args.resolution}px wide "
            f"({hires_count} text-heavy frames upscaled to {HIRES_WIDTH}px)"
        )
    else:
        print(f"- **Frame size:** {args.resolution}px wide")
    if args.no_ocr:
        print("- **OCR:** disabled (`--no-ocr`)")
    elif ocr_text:
        text_frames = sum(1 for v in ocr_text.values() if v.strip())
        print(f"- **OCR:** {text_frames}/{len(frames)} frames had detected text (lang=spa+eng)")
    else:
        print("- **OCR:** unavailable (pytesseract or tesseract binary missing — see stderr)")
    if transcript_segments:
        in_range = " in range" if focused else ""
        print(
            f"- **Transcript:** {len(transcript_segments)} segments{in_range} "
            f"(via {transcript_source or 'captions'})"
        )
        if speech_windows:
            speech_total = sum(e - s for s, e in speech_windows)
            denom = effective_duration if focused else full_duration
            print(
                f"- **Speech windows:** "
                f"{format_windows(speech_windows, speech_total=speech_total, full_duration=denom)}"
            )
    else:
        print("- **Transcript:** none available")

    if not focused and full_duration > 600:
        mins = int(full_duration // 60)
        print()
        print(
            f"> **Warning:** This is a {mins}-minute video. Frame coverage is sparse at this length — "
            "accuracy degrades noticeably on anything over 10 minutes. For better results, "
            "re-run with `--start HH:MM:SS --end HH:MM:SS` to zoom into a specific section."
        )

    print()
    print("## Frames")
    print()
    print(f"Frames live at: `{work / 'frames'}`")
    print()
    print(
        "**Read each frame path below with the Read tool to view the image.** "
        "Frames are in chronological order; `t=MM:SS` is the absolute timestamp in the source video."
    )
    if ocr_text:
        print()
        print(
            "Each frame line includes any text detected by OCR (Spanish + English). "
            "Use it to correlate visuals with on-screen captions, slides, UI text, etc. "
            "Frames where OCR found significant text were re-extracted at "
            f"{HIRES_WIDTH}px so the text is legible when you Read them."
        )
    print()
    speech_set = [(s, e) for s, e in speech_windows]
    for frame in frames:
        ts = format_time(frame["timestamp_seconds"])
        line = f"- `{frame['path']}` (t={ts})"
        if sampling_mode == "two-pass" and speech_set:
            in_speech = any(s <= frame["timestamp_seconds"] <= e for s, e in speech_set)
            line += " [speech]" if in_speech else " [silent]"
        text = ocr_text.get(frame["path"], "").strip() if ocr_text else ""
        if text:
            collapsed = " ".join(text.split())
            line += f" — OCR: {collapsed}"
        print(line)

    print()
    print("## Transcript")
    print()
    if transcript_text:
        label = transcript_source or "captions"
        if focused:
            print(f"_Source: {label}. Filtered to {format_time(effective_start)} → {format_time(effective_end)}:_")
        else:
            print(f"_Source: {label}._")
        print()
        print("```")
        print(transcript_text)
        print("```")
    elif focused and dl.get("subtitle_path"):
        print(f"_No transcript lines fell inside {format_time(effective_start)} → {format_time(effective_end)}._")
    else:
        setup_py = SCRIPT_DIR / "setup.py"
        print(
            "_No transcript available — proceed with frames only. "
            "Captions were missing and the Whisper fallback was unavailable "
            "(no API key set, or `--no-whisper` was used). "
            f"Run `python3 {setup_py}` to enable Whisper, then re-run._"
        )

    print()
    print("---")
    print(f"_Work dir: `{work}` — delete when done._")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
