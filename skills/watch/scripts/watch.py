#!/usr/bin/env python3
"""/watch entry point: download video, extract frames, parse transcript.

Prints a markdown report to stdout listing frame paths + transcript. Claude
then Reads each frame path to see the video.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from config import frame_cap, get_config, memory_paths  # noqa: E402
from download import DEFAULT_SUB_LANGS, download, fetch_captions, is_url  # noqa: E402
import library  # noqa: E402
import state  # noqa: E402
from frames import (  # noqa: E402
    MAX_FPS,
    auto_fps,
    auto_fps_focus,
    dedupe_perceptual,
    extract,
    extract_at_timestamps,
    extract_keyframes,
    extract_scene_or_uniform,
    format_time,
    get_metadata,
    merge_frames,
    parse_time,
    parse_timestamps,
    reextract_frame,
)
from transcribe import filter_range, format_transcript, parse_vtt  # noqa: E402
import transcriptapi  # noqa: E402
import gemini  # noqa: E402
import identity  # noqa: E402
import storyboard  # noqa: E402
import doctor  # noqa: E402
import exports  # noqa: E402
from memory import writer as memory_writer  # noqa: E402
from memory.index import RunRecord, frame_kind  # noqa: E402
from ocr import is_significant, run_ocr  # noqa: E402
from scenes import DEFAULT_THRESHOLD, detect_scenes  # noqa: E402
from speech import DEFAULT_SPEECH_SHARE, compute_speech_windows, format_windows, two_pass_sample  # noqa: E402
from whisper import extract_audio, resolve_backend, transcribe_video  # noqa: E402
from whisper_local import DEFAULT_MODEL as WHISPER_LOCAL_DEFAULT_MODEL  # noqa: E402
from whisper_local import VALID_MODELS as WHISPER_LOCAL_MODELS  # noqa: E402
from gemini import DEFAULT_MODEL as GEMINI_DEFAULT_MODEL  # noqa: E402
from gemini import VALID_MODELS as GEMINI_MODELS  # noqa: E402
from openrouter import DEFAULT_AUDIO_MODEL as OR_AUDIO_DEFAULT  # noqa: E402
from openrouter import DEFAULT_VISION_MODEL as OR_VISION_DEFAULT  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


HIRES_WIDTH = 1024

ORIGINS = ("skill", "service", "ingest", "import", "backfill")


def should_save(*, backend: str, origin: str, detail: str, save: bool, no_save: bool,
                env_no_save: bool) -> bool:
    """Whether this run is archived to the library (Immich/Nextcloud/Paperless +
    Qdrant inbox). Indexing is independent of this (only --no-index skips it)."""
    if no_save or env_no_save:
        return False
    if save:
        return True
    if backend in ("gemini", "openrouter"):
        return False
    if origin == "ingest" and detail == "transcript":
        return False
    return True


def _env_no_save() -> bool:
    # WATCH_NO_SAVE (any non-empty, non-"0"/"false"/"no" value) force-disables
    # library saving even without --no-save. The test suite sets it so e2e runs
    # never upload fixtures to the real Immich/Nextcloud/Paperless library.
    return os.environ.get("WATCH_NO_SAVE", "").strip().lower() not in ("", "0", "false", "no")


def _run_should_save(args, backend: str, detail: str) -> bool:
    return should_save(
        backend=backend, origin=args.origin, detail=detail, save=args.save,
        no_save=args.no_save, env_no_save=_env_no_save(),
    )


def _library_name(args, info: dict) -> tuple[str, str]:
    """(title, video_id) naming this run's library entry (album / folder / document)."""
    video_id = info.get("id") or hashlib.sha1(args.source.encode("utf-8")).hexdigest()[:8]
    return info.get("title") or Path(args.source).name, video_id


def _planned_exports(args) -> list[str]:
    """The export formats this run will build (none without the index)."""
    if args.no_index:
        return []
    return exports.parse_formats(args.exports) if args.exports is not None else exports.default_formats()


def _save_library(args, work: Path, *, media_path: str | None, frames: list[dict],
                  ocr_text: dict, info: dict) -> dict:
    """Push this run's artifacts to the library; returns the refs (may be empty).
    Never raises — save must never fail the run. Paperless is left to the exports
    step when this run will build a PDF (it replaces report.md there)."""
    title, video_id = _library_name(args, info)
    refs: dict = {}
    try:
        summary = library.save_artifacts(
            media_path, frames, str(work / "report.md"), ocr_text, title, video_id,
            refs=refs, paperless="pdf" not in _planned_exports(args),
        )
        for line in summary or []:
            print(f"[watch] {line}", file=sys.stderr)
        if refs:
            (work / "library.json").write_text(json.dumps(refs, indent=2), encoding="utf-8")
    except Exception as exc:  # belt: save must never fail the run
        print(f"[watch] library save skipped ({exc})", file=sys.stderr)
    return refs


def _blocked_media_fallback(args, work: Path, exc: SystemExit, *, detail: str,
                            has_transcript: bool, max_frames: int | None) -> dict:
    """A YouTube media download failed (usually the datacenter bot wall).

    Non-transcript runs fetch the public storyboard as low-res frames. With neither
    storyboard frames nor a transcript the original failure is re-raised; otherwise
    the run continues (frameless when the storyboard is unavailable).
    """
    if not transcriptapi.is_youtube_url(args.source):
        raise exc
    print(f"[watch] media download failed: {exc}", file=sys.stderr)
    board: dict | None = None
    info: dict = {}
    if detail != "transcript":
        try:
            frames, meta, info = storyboard.fetch(
                args.source, work / "storyboard",
                start=parse_time(args.start), end=parse_time(args.end), max_frames=max_frames,
            )
            if frames:
                board = {"frames": frames, "meta": meta}
        except Exception as sb_exc:  # noqa: BLE001 - fallback of a fallback
            print(f"[watch] storyboard unavailable: {sb_exc}", file=sys.stderr)
    if board is None and not has_transcript:
        raise exc
    return {"video_path": None, "subtitle_path": None, "info": info or {"url": args.source},
            "downloaded": False, "storyboard": board, "blocked": str(exc)}


def _memory_title(source: str, info: dict) -> str | None:
    """Title for the index: metadata title, else the file name for local
    files; None for URLs without metadata (yt-dlp is often bot-gated)."""
    title = (info or {}).get("title")
    if title:
        return str(title)
    return None if is_url(source) else Path(source).name


def _remember(args, work: Path, *, backend: str, detail: str, info: dict,
              transcript_source: str | None, segments: list[dict], frames: list[dict],
              notes: list[dict], media_path: str | None, rng: tuple[float, float] | None,
              refs: dict, archive: bool) -> dict | None:
    """Index this run (unless --no-index) and write work/memory.json. Never raises."""
    if args.no_index:
        return None
    try:
        info = info or {}
        run = RunRecord(
            video_key=identity.video_key(args.source, info),
            source=args.source,
            origin=args.origin,
            backend=backend,
            detail=detail,
            title=_memory_title(args.source, info),
            url=identity.canonical_source(args.source) if is_url(args.source) else None,
            duration=info.get("duration"),
            channel=info.get("channel") or info.get("uploader"),
            upload_date=info.get("upload_date"),
            transcript_source=transcript_source if segments else None,
            range=rng,
            segments=list(segments or []),
            frames=list(frames or []),
            notes=list(notes or []),
            report_path=str(work / "report.md") if (work / "report.md").is_file() else None,
            media_path=media_path,
            library=refs or None,
        )
        result = memory_writer.remember(run, archive=archive)
        if result is not None:
            (work / "memory.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(
                f"[watch] video memory: {result['video_key']} "
                f"(evidence v{result['evidence_version']}, {result['mode']})",
                file=sys.stderr,
            )
        return result
    except Exception as exc:  # noqa: BLE001 - memory must never fail a run
        print(f"[watch] memory: not indexed ({type(exc).__name__}: {exc})", file=sys.stderr)
    return None


def _outputs(args, work: Path, result: dict | None, *, saved: bool, info: dict) -> None:
    """After indexing: chapters (always), cited notes + the requested exports, and —
    when the run is archived — their library routing. Never raises."""
    if not result:
        return
    try:
        from memory import extract
        from memory import index as memory_index

        key = result["video_key"]
        formats = _planned_exports(args)
        conn = memory_index.connect(memory_paths()["index_db"])
        try:
            extract.store_chapters(conn, key, llm_titles=True if args.llm_titles else None)
            built = exports.build(conn, key, formats)
        finally:
            conn.close()
        notes_src = built.get("notes", {}).get("path")
        if notes_src and Path(notes_src).is_file():
            shutil.copy2(notes_src, work / "notes.md")
        (work / "exports.json").write_text(json.dumps(built, indent=2), encoding="utf-8")
        made = [f"{k} → {Path(v['path']).name}" for k, v in built.items() if v.get("path")]
        if made:
            print(f"[watch] outputs: {', '.join(made)}", file=sys.stderr)
        if saved:
            title, video_id = _library_name(args, info)
            pdf = built.get("pdf", {}).get("path")
            paperless_doc = pdf or (str(work / "report.md") if "pdf" in formats else None)
            files = [v["path"] for k, v in built.items() if v.get("path") and k != "contact"]
            for line in library.save_exports(files, title, video_id, paperless_doc=paperless_doc,
                                             contact=built.get("contact", {}).get("path")):
                print(f"[watch] {line}", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001 - outputs must never fail a run
        print(f"[watch] outputs skipped ({type(exc).__name__}: {exc})", file=sys.stderr)


def _finish_backend_run(args, work: Path, *, backend_label: str, answer: str,
                        segments: list[dict], transcript_source: str | None, info: dict,
                        media_path: str | None, frames: list[dict]) -> None:
    """Shared tail of the Gemini/OpenRouter backends: write work/report.md (the
    stdout report plus a transcript section — stdout itself stays answer-only),
    archive only when should_save allows, then index the run."""
    question = " ".join(args.question).strip()
    lines = [
        f"# watch: video report ({backend_label} backend)",
        "",
        f"- **Source:** {args.source}",
        f"- **Question:** {question}",
        "",
        f"## {backend_label} response",
        "",
        answer or "",
        "",
        "## Transcript",
        "",
    ]
    if segments:
        lines += [f"_Source: {transcript_source or 'captions'}._", "", "```",
                  format_transcript(segments), "```"]
    else:
        lines.append("_No transcript available._")
    (work / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    backend = backend_label.lower()
    detail = getattr(args, "resolved_detail", None) or args.detail or "balanced"
    saved = _run_should_save(args, backend, detail)
    refs: dict = {}
    if saved:
        refs = _save_library(args, work, media_path=media_path,
                             frames=[f for f in frames if f.get("path")], ocr_text={}, info=info)
    note = {"kind": "backend_answer", "t_start": None, "t_end": None,
            "text": f"Q: {question}\n\n{answer or ''}"}
    result = _remember(
        args, work, backend=backend, detail=detail, info=info,
        transcript_source=transcript_source, segments=segments,
        frames=[{"t": f["timestamp_seconds"], "path": f.get("path"), "kind": frame_kind(f), "ocr": None}
                for f in frames],
        notes=[note], media_path=media_path, rng=getattr(args, "memory_range", None),
        refs=refs, archive=saved,
    )
    _outputs(args, work, result, saved=saved, info=info)


def _gemini_free_transcript(args, work: Path, *, youtube: bool, dl: dict | None,
                            media_path: str | None) -> tuple[list[dict], str | None, dict, str | None]:
    """Free transcript for the Gemini backend: yt-dlp captions only (no
    TranscriptAPI, no extra Gemini call); Whisper only with an explicit
    --whisper. Returns (segments, source, info, media_path). Never raises."""
    info: dict = dict((dl or {}).get("info") or {})
    subtitle_path = (dl or {}).get("subtitle_path")
    try:
        if youtube:
            caps = fetch_captions(
                args.source,
                work / "download",
                sub_langs=args.sub_lang or DEFAULT_SUB_LANGS,
                cookies_from_browser=args.cookies_from_browser,
                cookies_file=args.cookies,
            )
            info = dict(caps.get("info") or {}) or info
            subtitle_path = caps.get("subtitle_path")
        if subtitle_path:
            segs = parse_vtt(subtitle_path)
            if segs:
                return segs, "captions", info, media_path
    except (Exception, SystemExit) as exc:  # noqa: BLE001 - best effort
        print(f"[watch] caption fetch for memory failed: {exc}", file=sys.stderr)
    if args.whisper and not args.no_whisper:
        try:
            if media_path is None and is_url(args.source):
                dl2 = download(
                    args.source, work / "download", audio_only=True,
                    sub_langs=args.sub_lang or DEFAULT_SUB_LANGS,
                    cookies_from_browser=args.cookies_from_browser, cookies_file=args.cookies,
                )
                media_path = dl2.get("video_path")
                info = info or dict(dl2.get("info") or {})
            backend, api_key, hint = resolve_backend(args.whisper)
            if backend and media_path:
                segs, used = transcribe_video(
                    media_path, work / "audio.mp3", backend=backend, api_key=api_key,
                    model_name=args.whisper_model if backend == "local" else None,
                    enable_diarization=args.diarize if backend == "assemblyai" else False,
                    use_cache=not args.fresh,
                )
                if used == "local":
                    label = f"local, {args.whisper_model}"
                elif used == "assemblyai":
                    label = "assemblyai, diarized" if args.diarize else "assemblyai"
                else:
                    label = used
                return segs, f"whisper ({label})", info, media_path
            if hint:
                print(f"[watch] {hint}", file=sys.stderr)
        except (Exception, SystemExit) as exc:  # noqa: BLE001 - best effort
            print(f"[watch] whisper transcription failed: {exc}", file=sys.stderr)
    return [], None, info, media_path


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
    dl: dict | None = None
    media_path: str | None = None

    if youtube:
        gemini_source = source
        mode_label = "native YouTube URL (no download)"
    elif is_url(source):
        print("[watch] downloading via yt-dlp before Gemini upload…", file=sys.stderr)
        dl = download(source, work / "download")
        gemini_source = dl["video_path"]
        media_path = dl["video_path"]
        mode_label = "yt-dlp download → Gemini Files API upload"
    else:
        gemini_source = source
        media_path = source
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
    sys.stdout.flush()

    # Memory side (stdout above is the whole contract for ingest/Hermes): only a
    # FREE transcript (yt-dlp captions) is fetched, and only when the run is
    # indexed or archived — never TranscriptAPI or a second Gemini call.
    detail = getattr(args, "resolved_detail", None) or args.detail or "balanced"
    segments: list[dict] = []
    tx_source: str | None = None
    info: dict = {}
    if not args.no_index or _run_should_save(args, "gemini", detail):
        segments, tx_source, info, media_path = _gemini_free_transcript(
            args, work, youtube=youtube, dl=dl, media_path=media_path
        )
    _finish_backend_run(
        args, work, backend_label="Gemini", answer=response_text, segments=segments,
        transcript_source=tx_source, info=info, media_path=media_path, frames=[],
    )
    return 0


def _run_openrouter_backend(args, work: Path) -> int:
    """Extract frames, transcribe via OpenRouter audio, then query OpenRouter vision.

    Follows the same download → transcript → extract pipeline as the claude
    backend, but instead of printing frame paths for Claude to Read, encodes
    all frames as base64 and POSTs them together with the transcript and
    question to the OpenRouter chat completions endpoint.
    """
    from openrouter import analyze_with_frames, transcribe_audio

    question = " ".join(args.question).strip()
    if not question:
        raise SystemExit(
            "--backend openrouter requires a question. Pass it as the trailing "
            'positional argument: `watch.py video.mp4 --backend openrouter "Describe this video"`'
        )

    if args.audio:
        print(
            "[watch] --audio is ignored for --backend openrouter (use the default audio "
            "pipeline which extracts audio from the video).",
            file=sys.stderr,
        )

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
    effective_start = start_sec if start_sec is not None else 0.0
    effective_end = end_sec if end_sec is not None else full_duration
    effective_duration = max(0.0, effective_end - effective_start)
    focused = start_sec is not None or end_sec is not None

    max_frames = min(args.max_frames if args.max_frames is not None else 80, 100)
    if focused:
        fps, target = auto_fps_focus(effective_duration, max_frames=max_frames)
    else:
        fps, target = auto_fps(effective_duration, max_frames=max_frames)
    if args.fps is not None:
        fps = min(args.fps, MAX_FPS)
        target = max(1, int(round(fps * effective_duration)))

    # Transcript: captions first, then OpenRouter audio model
    transcript_segments: list[dict] = []
    transcript_text: str | None = None
    transcript_source: str | None = None

    if dl.get("subtitle_path"):
        try:
            all_segs = parse_vtt(dl["subtitle_path"])
            transcript_segments = filter_range(all_segs, start_sec, end_sec) if focused else all_segs
            transcript_text = format_transcript(transcript_segments)
            transcript_source = "captions"
        except Exception as exc:
            print(f"[watch] subtitle parse failed: {exc}", file=sys.stderr)

    if not transcript_segments and not args.no_whisper:
        audio_out = work / "audio.mp3"
        print(
            f"[watch] extracting audio for OpenRouter ({args.openrouter_audio_model})…",
            file=sys.stderr,
        )
        try:
            audio_path = extract_audio(video_path, audio_out)
            size_kb = audio_path.stat().st_size / 1024
            print(
                f"[watch] audio: {size_kb:.0f} kB — transcribing via OpenRouter…",
                file=sys.stderr,
            )
            transcript_segments = transcribe_audio(audio_path, model=args.openrouter_audio_model)
            if focused:
                transcript_segments = filter_range(transcript_segments, start_sec, end_sec)
            transcript_text = format_transcript(transcript_segments)
            transcript_source = f"openrouter ({args.openrouter_audio_model})"
            print(
                f"[watch] transcribed {len(transcript_segments)} segments via OpenRouter",
                file=sys.stderr,
            )
        except SystemExit as exc:
            print(f"[watch] OpenRouter audio transcription failed: {exc}", file=sys.stderr)

    # Frame extraction (uniform + fps, mirroring the fork's OpenRouter path)
    print(f"[watch] extracting ~{target} frames at {fps:.3f} fps…", file=sys.stderr)
    frames = extract(
        video_path,
        work / "frames",
        fps=fps,
        resolution=args.resolution,
        max_frames=max_frames,
        start_seconds=start_sec,
        end_seconds=end_sec,
    )

    frame_paths = [f["path"] for f in frames]
    print(
        f"[watch] sending {len(frame_paths)} frames to OpenRouter ({args.openrouter_vision_model})…",
        file=sys.stderr,
    )

    response_text = analyze_with_frames(
        frame_paths=frame_paths,
        transcript_text=transcript_text,
        question=question,
        vision_model=args.openrouter_vision_model,
    )

    info = dl.get("info") or {}
    scope = (
        f"{format_time(effective_start)}–{format_time(effective_end)}"
        if focused
        else "full"
    )

    print()
    print("# watch: video report (OpenRouter backend)")
    print()
    print(f"- **Source:** {args.source}")
    if info.get("title"):
        print(f"- **Title:** {info['title']}")
    if info.get("uploader"):
        print(f"- **Uploader:** {info['uploader']}")
    print(f"- **Duration:** {format_time(full_duration)} ({full_duration:.1f}s)")
    if focused:
        print(f"- **Focus:** {scope}")
    print(f"- **Vision model:** {args.openrouter_vision_model}")
    print(f"- **Frames:** {len(frame_paths)} @ {fps:.3f} fps ({scope})")
    if transcript_segments:
        print(f"- **Transcript:** {len(transcript_segments)} segments (via {transcript_source})")
    else:
        print("- **Transcript:** none")
    print(f"- **Question:** {question}")
    print()
    print("## OpenRouter response")
    print()
    print(response_text)
    print()
    print("---")
    print(f"_Work dir: `{work}` — delete when done._")
    sys.stdout.flush()

    args.memory_range = (effective_start, effective_end) if focused else None
    _finish_backend_run(
        args, work, backend_label="OpenRouter", answer=response_text,
        segments=transcript_segments, transcript_source=transcript_source, info=info,
        media_path=video_path, frames=frames,
    )
    return 0


MEMORY_COMMANDS = {"search", "ask", "moment", "list", "index", "stats", "export"}


def main() -> int:
    if sys.argv[1:2] == ["doctor"]:
        return doctor.main(sys.argv[2:])
    # Video-memory read path: `watch.py search|ask|moment|list|index ...`, unless the
    # word is an existing path (a file literally named "search" is still watched).
    if len(sys.argv) > 1 and sys.argv[1] in MEMORY_COMMANDS and not os.path.exists(sys.argv[1]):
        from memory import cli as memory_cli

        return memory_cli.main(sys.argv[1:])
    ap = argparse.ArgumentParser(
        prog="watch",
        description="Download a video, extract auto-scaled frames, and surface the transcript.",
    )
    ap.add_argument("source", help="Video URL or local file path")
    ap.add_argument(
        "question",
        nargs="*",
        default=[],
        help="Optional trailing question. Required for --backend gemini / openrouter "
        "(passed to the model as the prompt). Ignored for --backend claude — "
        "Claude handles the question via its own conversation context.",
    )
    ap.add_argument(
        "--backend",
        choices=["claude", "gemini", "openrouter"],
        default="claude",
        help="claude (default): extract frames + transcript locally so Claude can `Read` them. "
        "gemini: skip frame extraction and Whisper, hand the whole video to Gemini's "
        "native multimodal model and print its response. "
        "openrouter: extract frames, transcribe via OpenRouter audio model, then POST "
        "everything to an OpenRouter vision model and print its response.",
    )
    ap.add_argument(
        "--gemini-model",
        choices=list(GEMINI_MODELS),
        default=GEMINI_DEFAULT_MODEL,
        help=f"Gemini model for --backend gemini (default {GEMINI_DEFAULT_MODEL}). "
        "Ignored for --backend claude / openrouter.",
    )
    ap.add_argument(
        "--openrouter-vision-model",
        default=OR_VISION_DEFAULT,
        help=f"OpenRouter model for vision/chat analysis (default {OR_VISION_DEFAULT}). "
        "Only applies to --backend openrouter.",
    )
    ap.add_argument(
        "--openrouter-audio-model",
        default=OR_AUDIO_DEFAULT,
        help=f"OpenRouter model for audio transcription (default {OR_AUDIO_DEFAULT}). "
        "Used when no captions are found and --no-whisper is not set. "
        "Only applies to --backend openrouter.",
    )
    ap.add_argument("--max-frames", type=int, default=None, help="Override frame cap")
    ap.add_argument("--resolution", type=int, default=512, help="Frame width in pixels (default 512)")
    ap.add_argument("--fps", type=float, default=None, help="Override auto-fps")
    ap.add_argument(
        "--detail",
        choices=["transcript", "efficient", "balanced", "token-burner"],
        default=None,
        help="Fidelity/speed dial: transcript (no frames), efficient (fast keyframes, cap 50), "
             "balanced (scene, cap 100), token-burner (scene, uncapped).",
    )
    ap.add_argument(
        "--timestamps",
        type=str,
        default=None,
        help="Comma-separated absolute timestamps (SS, MM:SS, HH:MM:SS) to grab a frame at, "
             "e.g. transcript-flagged 'look here' moments. Added on top of the detail frames "
             "(reserved against the cap); with --detail transcript these become the only frames.",
    )
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
        "--cookies-from-browser",
        type=str,
        default=None,
        help="Load yt-dlp cookies from a browser profile (chrome|firefox|edge|brave|safari|...) "
        "for login-walled or age-gated sources. Off by default.",
    )
    ap.add_argument(
        "--cookies",
        type=str,
        default=None,
        help="Path to a Netscape cookies.txt for yt-dlp. Off by default.",
    )
    ap.add_argument(
        "--sub-lang",
        type=str,
        default=None,
        help="Comma-separated subtitle language preference for yt-dlp "
        f"(default: {DEFAULT_SUB_LANGS}). yt-dlp fetches every match; the report "
        "uses the first available in this order. Never pass 'all'.",
    )
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore any resume state in --out-dir and re-run every stage from "
        "scratch (also bypasses the Whisper chunk cache for this run).",
    )
    ap.add_argument(
        "--no-whisper",
        action="store_true",
        help="Disable Whisper fallback. Report frames-only if no captions available. "
        "The Gemini URL transcript still runs for YouTube (disable it with "
        "--no-gemini-transcript).",
    )
    ap.add_argument(
        "--no-transcriptapi",
        action="store_true",
        help="Disable the TranscriptAPI YouTube-transcript backend (transcriptapi.com). "
        "By default, YouTube URLs fetch their transcript from TranscriptAPI first "
        "(needs TRANSCRIPTAPI_API_KEY); this forces the yt-dlp caption / Whisper path.",
    )
    ap.add_argument(
        "--no-gemini-transcript",
        action="store_true",
        help="Disable the Gemini YouTube-transcript fallback. By default, a YouTube URL "
        "with no TranscriptAPI/caption transcript is transcribed by Gemini from the URL "
        "(needs GEMINI_API_KEY); also disabled by WATCH_GEMINI_TRANSCRIPT=0.",
    )
    ap.add_argument(
        "--no-save",
        action="store_true",
        help="Do not push this run's artifacts to the library (Immich/Nextcloud). "
        "Saving is on by default.",
    )
    ap.add_argument(
        "--save",
        action="store_true",
        help="Archive this run to the library even where it is off by default "
        "(--backend gemini/openrouter, ingest transcript runs). --no-save / "
        "WATCH_NO_SAVE still win.",
    )
    ap.add_argument(
        "--exports",
        default=None,
        metavar="FORMATS",
        help="Comma list of exports built from the video memory after the run: "
        "docx,pdf,pptx,xlsx,viewer,json,contact (or all/none). Default: WATCH_EXPORTS, "
        "else pdf,pptx when wordcraft-cli and deckcraft-cli are installed, else none. "
        "Chapters and notes.md are always written.",
    )
    ap.add_argument(
        "--llm-titles",
        action="store_true",
        help="Rewrite chapter titles with one LLM call (also WATCH_LLM_TITLES=1).",
    )
    ap.add_argument(
        "--no-index",
        action="store_true",
        help="Do not record this run in the video memory (index + store). "
        "Indexing is on by default, even with --no-save.",
    )
    ap.add_argument(
        "--origin",
        choices=list(ORIGINS),
        default="skill",
        help="Who started this run (recorded in the video memory; ingest "
        "transcript runs are indexed but not archived). Default: skill.",
    )
    ap.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable OCR pass. Skips text detection on frames and the high-res re-extract.",
    )
    ap.add_argument(
        "--no-scene-detect",
        action="store_true",
        help="Skip PySceneDetect when informing two-pass sampling (use even spacing instead).",
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
        "(70%% inside speech, 30%% outside). Default ON when a timed transcript is available.",
    )
    ap.add_argument(
        "--whisper",
        choices=["groq", "openai", "local", "assemblyai"],
        default=None,
        help="Force a specific Whisper backend. 'local' runs faster-whisper on the GPU "
        "(needs faster-whisper + CUDA). 'assemblyai' is paid and adds automatic speaker "
        "labels to the transcript. Default: auto-pick local if available, else Groq, else "
        "OpenAI. AssemblyAI is never auto-picked — request it explicitly for diarization.",
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
        "Default ON. Pass --no-diarize to skip speaker labels. "
        "Ignored for local / groq / openai (none of those expose diarization here).",
    )
    ap.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable near-duplicate frame removal. Keeps visually identical "
             "frames (static screen recordings, held slides) instead of collapsing them.",
    )
    # parse_intermixed_args (not parse_args) so a trailing question survives options
    # placed after it, e.g. `watch.py video.mp4 --backend gemini "Describe this"`.
    # Plain parse_args rejects that ("unrecognized arguments") because the nargs="*"
    # question positional can't span an interleaved optional.
    args = ap.parse_intermixed_args()

    config = get_config()
    detail = args.detail or str(config["detail"])
    configured_cap = frame_cap(detail)
    if args.max_frames is not None:
        max_frames = args.max_frames
    else:
        max_frames = configured_cap
    if max_frames is not None and max_frames < 1:
        raise SystemExit("--max-frames must be greater than zero")
    budget_cap = max_frames if max_frames is not None else 100
    cue_timestamps = parse_timestamps(args.timestamps)

    if args.out_dir:
        work = Path(args.out_dir).expanduser().resolve()
    else:
        work = Path(tempfile.mkdtemp(prefix="watch-"))
    work.mkdir(parents=True, exist_ok=True)
    print(f"[watch] working dir: {work}", file=sys.stderr)

    # Resume is only meaningful with a stable work dir. With the default tmp dir
    # each run gets a fresh path, so load_stage always misses (harmless no-op).
    resume = bool(args.out_dir) and not args.fresh
    resume_sig = state.params_signature(
        args.source,
        {
            "detail": args.detail,
            "start": args.start,
            "end": args.end,
            "timestamps": args.timestamps,
            "audio": args.audio,
            "sub_lang": args.sub_lang,
            "whisper": args.whisper,
            "whisper_model": args.whisper_model,
            "diarize": args.diarize,
        },
    )
    if args.out_dir and args.fresh:
        state.clear_stages(work)
        print("[watch] --fresh: cleared resume state", file=sys.stderr)

    args.resolved_detail = detail
    if args.backend == "gemini":
        return _run_gemini_backend(args, work)
    if args.backend == "openrouter":
        return _run_openrouter_backend(args, work)

    # --audio supplies an external voice-over track to transcribe in place of the
    # video's own audio; it implies Whisper, so it cannot combine with --no-whisper.
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

    url_source = is_url(args.source)
    dl: dict = {"subtitle_path": None, "info": {}, "downloaded": False}
    transcript_segments: list[dict] = []
    transcript_text: str | None = None
    transcript_source: str | None = None
    video_path: str | None = None

    # Resume: reuse a previously-resolved transcript BEFORE any (paid) TranscriptAPI
    # call or caption parse, so a matching --out-dir re-run costs nothing.
    cached_tx = state.load_stage(work, "transcript", resume_sig) if resume else None
    if cached_tx and cached_tx.get("segments"):
        transcript_segments = cached_tx["segments"]
        transcript_source = cached_tx.get("source") or "captions"
        transcript_text = format_transcript(transcript_segments)
        print(
            f"[watch] resume: reusing transcript ({len(transcript_segments)} segments)",
            file=sys.stderr,
        )

    if url_source:
        print("[watch] checking metadata/captions via yt-dlp…", file=sys.stderr)
        dl = fetch_captions(
            args.source,
            work / "download",
            sub_langs=args.sub_lang or DEFAULT_SUB_LANGS,
            cookies_from_browser=args.cookies_from_browser,
            cookies_file=args.cookies,
        )
        # TranscriptAPI first (YouTube URLs): fetch the transcript from
        # transcriptapi.com, whose servers clear YouTube's bot-gate / PO-token /
        # nsig walls that block yt-dlp caption fetch from a datacenter IP. Runs
        # BEFORE the caption parse + the download decision below, so transcript
        # mode skips the (gated) audio download. Best-effort — any miss (no key,
        # no credits, no transcript, network) falls through to yt-dlp captions /
        # Whisper. Skipped for --audio (that external VO is not the video's own).
        if (
            not transcript_segments
            and not args.no_transcriptapi
            and audio_override is None
            and transcriptapi.is_youtube_url(args.source)
        ):
            ta_key = transcriptapi.load_api_key()
            if ta_key:
                ta_langs = transcriptapi.languages_from_sub_langs(args.sub_lang or DEFAULT_SUB_LANGS)
                print(f"[watch] fetching transcript via TranscriptAPI (lang: {ta_langs})…", file=sys.stderr)
                ta_result = transcriptapi.fetch_transcript(args.source, ta_key, language=ta_langs)
                if ta_result is not None:
                    transcript_segments, _resolved_lang = ta_result
                    transcript_text = format_transcript(transcript_segments)
                    transcript_source = (
                        f"transcriptapi ({_resolved_lang})" if _resolved_lang else "transcriptapi"
                    )
                    print(
                        f"[watch] TranscriptAPI: {len(transcript_segments)} segments ({_resolved_lang})",
                        file=sys.stderr,
                    )
            else:
                print("[watch] no TRANSCRIPTAPI_API_KEY set — skipping TranscriptAPI", file=sys.stderr)
        # --audio supplies an external track that must win over the video's own
        # captions, so skip caption parsing entirely when it is set.
        if dl.get("subtitle_path") and audio_override is None and not transcript_segments:
            try:
                transcript_segments = parse_vtt(dl["subtitle_path"])
                transcript_text = format_transcript(transcript_segments)
                transcript_source = "captions"
            except Exception as exc:
                print(f"[watch] subtitle parse failed: {exc}", file=sys.stderr)
                transcript_segments = []
        # Gemini URL transcript (YouTube only): last resort before the (often
        # gated) download + Whisper. Gemini fetches the video itself from the URL,
        # so it works from datacenter IPs. Best-effort (never raises); opt out with
        # --no-gemini-transcript or WATCH_GEMINI_TRANSCRIPT=0. Skipped for --audio.
        if (
            not transcript_segments
            and not args.no_gemini_transcript
            and os.environ.get("WATCH_GEMINI_TRANSCRIPT", "").strip().lower() not in ("0", "false", "no")
            and audio_override is None
            and transcriptapi.is_youtube_url(args.source)
        ):
            _gm_duration = (dl.get("info") or {}).get("duration") or None
            gm_result = gemini.transcribe_youtube(
                identity.canonical_source(args.source), duration=_gm_duration
            )
            if gm_result is not None:
                transcript_segments, _gm_model, _gm_approx = gm_result
                transcript_text = format_transcript(transcript_segments)
                transcript_source = (
                    f"gemini ({_gm_model}, approximate timestamps)"
                    if _gm_approx
                    else f"gemini ({_gm_model})"
                )

    # Persist a pre-download transcript (TranscriptAPI / captions) now, so a later
    # gated video-download failure doesn't discard it and a resume reuses it without
    # re-billing TranscriptAPI. The post-Whisper save below still covers the
    # local-file / caption-less path.
    if args.out_dir and transcript_segments:
        state.save_stage(
            work,
            "transcript",
            {"segments": transcript_segments, "source": transcript_source},
            resume_sig,
        )

    # --timestamps needs the video for frame grabs, so it overrides the
    # transcript-mode download skip (and forces a full, not audio-only, fetch).
    audio_only = detail == "transcript" and not cue_timestamps
    if detail == "transcript" and transcript_segments and not cue_timestamps:
        video_path = None
    else:
        cached_dl = state.load_stage(work, "download", resume_sig) if resume else None
        if (
            cached_dl
            and cached_dl.get("video_path")
            and Path(cached_dl["video_path"]).exists()
        ):
            print("[watch] resume: reusing previously downloaded video", file=sys.stderr)
            dl = cached_dl
        else:
            if url_source:
                print(
                    "[watch] downloading audio via yt-dlp…" if audio_only
                    else "[watch] downloading video via yt-dlp…",
                    file=sys.stderr,
                )
                try:
                    dl = download(
                        args.source,
                        work / "download",
                        audio_only=audio_only,
                        sub_langs=args.sub_lang or DEFAULT_SUB_LANGS,
                        cookies_from_browser=args.cookies_from_browser,
                        cookies_file=args.cookies,
                    )
                except SystemExit as exc:
                    dl = _blocked_media_fallback(
                        args, work, exc, detail=detail,
                        has_transcript=bool(transcript_segments), max_frames=budget_cap,
                    )
            else:
                print("[watch] using local file…", file=sys.stderr)
                dl = download(args.source, work / "download")
            if args.out_dir:
                state.save_stage(work, "download", dl, resume_sig)
        video_path = dl["video_path"]

    meta = get_metadata(video_path) if video_path else {
        "duration_seconds": float((dl.get("info") or {}).get("duration") or 0),
        "width": None,
        "height": None,
        "codec": None,
        "has_audio": False,
    }
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
        fps, target = auto_fps_focus(effective_duration, max_frames=budget_cap)
    else:
        fps, target = auto_fps(effective_duration, max_frames=budget_cap)
    if args.fps is not None:
        fps = min(args.fps, MAX_FPS)
        target = max(1, int(round(fps * effective_duration)))

    if transcript_segments and focused:
        transcript_segments = filter_range(transcript_segments, start_sec, end_sec)
        transcript_text = format_transcript(transcript_segments)

    # Transcript-first: finish resolving the transcript (download-discovered captions,
    # then the Whisper fallback) BEFORE frame extraction so the two-pass gate below
    # sees Whisper transcripts too — local files and caption-less URLs — not just the
    # captions available pre-frames. A failed Whisper call falls through, leaving the
    # frame pipeline (without two-pass) to run exactly as it would with no transcript.
    if not transcript_segments and dl.get("subtitle_path") and audio_override is None:
        try:
            all_segments = parse_vtt(dl["subtitle_path"])
            transcript_segments = filter_range(all_segments, start_sec, end_sec) if focused else all_segments
            transcript_text = format_transcript(transcript_segments)
            transcript_source = "captions"
        except Exception as exc:
            print(f"[watch] subtitle parse failed: {exc}", file=sys.stderr)

    # Whisper fallback: resolve_backend now spans local / groq / openai / assemblyai
    # and reports a hint when nothing is usable. --audio, when present, supplies an
    # external track and bypasses the video's own has_audio check.
    whisper_ready = bool(audio_override) or (video_path is not None and meta.get("has_audio"))
    if not transcript_segments and not args.no_whisper and whisper_ready:
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
                    start_seconds=(effective_start if (focused and not audio_override) else None),
                    end_seconds=(effective_end if (focused and not audio_override) else None),
                    use_cache=not args.fresh,
                )
                transcript_segments = (
                    filter_range(all_segments, start_sec, end_sec) if focused else all_segments
                )
                transcript_text = format_transcript(transcript_segments)
                # Fold per-backend extras into a single label so the transcript header
                # says exactly what produced it: "local, large-v3", "assemblyai,
                # diarized", plain "groq", and a "--audio" suffix for external tracks.
                if used_backend == "local":
                    backend_label = f"local, {args.whisper_model}"
                elif used_backend == "assemblyai":
                    backend_label = "assemblyai, diarized" if args.diarize else "assemblyai"
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
    elif not transcript_segments and video_path and not meta.get("has_audio") and not audio_override:
        print("[watch] no audio stream found — proceeding without transcription", file=sys.stderr)

    if args.out_dir and transcript_segments:
        state.save_stage(
            work,
            "transcript",
            {"segments": transcript_segments, "source": transcript_source},
            resume_sig,
        )

    scope = (
        f"{format_time(effective_start)}-{format_time(effective_end)} ({effective_duration:.1f}s)"
        if focused else f"full {effective_duration:.1f}s"
    )
    frames: list[dict] = []
    frame_meta: dict = {"engine": "none", "candidate_count": 0, "selected_count": 0, "fallback": False}
    cue_frames: list[dict] = []
    cue_meta: dict = {}

    # Transcript cues are pinned: extracted first and counted against the cap so
    # the detail engine never evicts the moments the user explicitly asked for.
    if cue_timestamps and video_path:
        cue_frames, cue_meta = extract_at_timestamps(
            video_path,
            work / "frames",
            cue_timestamps,
            resolution=args.resolution,
            max_frames=max_frames,
            start_seconds=start_sec,
            end_seconds=end_sec,
        )
        if cue_meta.get("dropped_out_of_window"):
            print(
                f"[watch] {cue_meta['dropped_out_of_window']} cue timestamp(s) outside the "
                "focus range — dropped",
                file=sys.stderr,
            )

    detail_budget = max_frames if max_frames is None else max(0, max_frames - len(cue_frames))
    # Overridden below when two-pass runs, so the final report's cap reflects the
    # budget actually applied (duration-scaled target) instead of the flat detail cap.
    report_cap = detail_budget

    # Two-pass sampling grafts in ABOVE the detail engines: when a timed transcript
    # is already in hand (captions or Whisper, now resolved before frame extraction), distribute
    # the frame budget 70/30 across speech vs silent windows instead of running the
    # scene/keyframe engine. Precedence: cue frames (pinned) → two-pass → detail
    # engines → uniform fps. An explicit --fps override disables it (that flag means
    # "sample at this rate", not "let the transcript drive the picks").
    speech_windows: list[tuple[float, float]] = []
    two_pass_eligible = (
        args.two_pass
        and bool(transcript_segments)
        and video_path is not None
        and detail != "transcript"
        and args.fps is None
        and detail_budget != 0
    )
    if two_pass_eligible:
        speech_windows = compute_speech_windows(
            transcript_segments, range_start=effective_start, range_end=effective_end
        )
    use_two_pass = two_pass_eligible and bool(speech_windows)

    if use_two_pass:
        scene_spans: list[tuple[float, float]] = []
        if not args.no_scene_detect:
            print(
                f"[watch] detecting scenes (threshold={args.scene_threshold}) to inform two-pass…",
                file=sys.stderr,
            )
            scene_spans = detect_scenes(
                video_path,
                threshold=args.scene_threshold,
                start_seconds=start_sec,
                end_seconds=end_sec,
            )
            if not scene_spans:
                print(
                    "[watch] no scene cuts detected; two-pass will use even spacing",
                    file=sys.stderr,
                )
        # two_pass_sample needs a concrete cap: use the duration-scaled auto_fps
        # target (not the flat detail cap) so two-pass honors the same ~2fps
        # invariant as every other engine. min() still respects a detail_budget
        # that cue frames have shrunk below the target, and token-burner's
        # uncapped None falls back to the target itself (no cap to intersect).
        two_pass_budget = min(target, detail_budget) if detail_budget is not None else target
        report_cap = two_pass_budget
        plan = two_pass_sample(
            effective_start,
            effective_end,
            speech_windows,
            scene_spans,
            two_pass_budget,
            speech_share=DEFAULT_SPEECH_SHARE,
        )
        timestamps = plan["timestamps"]
        print(
            f"[watch] two-pass sampling: {plan['speech_count']} frame(s) in speech windows, "
            f"{plan['non_speech_count']} outside (cap {two_pass_budget})…",
            file=sys.stderr,
        )
        # extract_at_timestamps wipes cue_*.jpg in its out_dir, so when cue frames are
        # already pinned in work/frames we route two-pass picks to a sibling directory
        # to avoid clobbering them; absolute paths in the report keep both readable.
        two_pass_dir = (work / "frames_two_pass") if cue_frames else (work / "frames")
        frames, _ts_meta = extract_at_timestamps(
            video_path,
            two_pass_dir,
            timestamps,
            resolution=args.resolution,
            max_frames=detail_budget,
            start_seconds=start_sec,
            end_seconds=end_sec,
        )
        deduped_count = 0
        if not args.no_dedup:
            frames, deduped_count = dedupe_perceptual(frames)
        for f in frames:
            f["reason"] = "two-pass"
        frame_meta = {
            "engine": "two-pass",
            "candidate_count": len(timestamps),
            "selected_count": len(frames),
            "fallback": False,
        }
        if deduped_count:
            frame_meta["deduped_count"] = deduped_count
    elif detail != "transcript" and video_path and detail_budget != 0:
        cap_label = "unlimited" if detail_budget is None else str(detail_budget)
        engine_label = "keyframes" if detail == "efficient" else "scene-aware frames"
        print(
            f"[watch] extracting {engine_label} over {scope} "
            f"(target {target}, cap {cap_label})…",
            file=sys.stderr,
        )
        if detail == "efficient":
            frames, frame_meta = extract_keyframes(
                video_path,
                work / "frames",
                resolution=args.resolution,
                max_frames=detail_budget,
                start_seconds=start_sec,
                end_seconds=end_sec,
                dedup=not args.no_dedup,
            )
        else:  # balanced, token-burner
            frames, frame_meta = extract_scene_or_uniform(
                video_path,
                work / "frames",
                fps=fps,
                target_frames=target,
                resolution=args.resolution,
                max_frames=detail_budget,
                start_seconds=start_sec,
                end_seconds=end_sec,
                dedup=not args.no_dedup,
            )

    elif detail != "transcript" and dl.get("storyboard"):
        frames = list(dl["storyboard"]["frames"])
        frame_meta = dict(dl["storyboard"]["meta"])
        if not args.no_dedup:
            frames, deduped_count = dedupe_perceptual(frames)
            if deduped_count:
                frame_meta["deduped_count"] = deduped_count
                frame_meta["selected_count"] = len(frames)

    if cue_frames:
        frames = merge_frames(frames, cue_frames)

    # OCR pass runs on the FINAL frame list (post merge_frames). Frames whose text is
    # significant get re-extracted at HIRES_WIDTH so the on-screen text is legible when
    # Claude Reads them. run_ocr degrades to {} when pytesseract/tesseract is missing,
    # so this is a no-op (and invisible in the report) on hosts without OCR installed.
    ocr_text: dict[str, str] = {}
    if not args.no_ocr and frames:
        print(f"[watch] running OCR on {len(frames)} frame(s) (lang=spa+eng)…", file=sys.stderr)
        raw_ocr = run_ocr([f["path"] for f in frames])
        for f in frames:
            text = raw_ocr.get(f["path"], "")
            if is_significant(text) and args.resolution < HIRES_WIDTH and video_path:
                hires = Path(f["path"]).with_name(Path(f["path"]).stem + "_hires.jpg")
                if reextract_frame(video_path, hires, f["timestamp_seconds"], HIRES_WIDTH):
                    f["path"] = str(hires)
            if text.strip():
                # Key by the FINAL path so the report loop's ocr_text.get(frame["path"])
                # still matches after any hi-res re-extract swapped the path.
                ocr_text[f["path"]] = text

    info = dl.get("info") or {}

    # Report lines are both printed (stdout, unchanged) AND captured so the
    # exact same text can be written to work/report.md below.
    report_lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        report_lines.append(s)

    emit()
    emit("# watch: video report")
    emit()
    emit(f"- **Source:** {args.source}")
    if info.get("title"):
        emit(f"- **Title:** {info['title']}")
    if info.get("uploader"):
        emit(f"- **Uploader:** {info['uploader']}")
    emit(f"- **Duration:** {format_time(full_duration)} ({full_duration:.1f}s)")
    if focused:
        emit(
            f"- **Focus range:** {format_time(effective_start)} → {format_time(effective_end)} "
            f"({effective_duration:.1f}s)"
        )
    if meta.get("width") and meta.get("height"):
        emit(f"- **Resolution:** {meta['width']}x{meta['height']} ({meta.get('codec') or 'unknown codec'})")
    range_mode = "focused" if focused else "full"
    emit(f"- **Detail:** {detail}")
    detail_count = frame_meta.get("selected_count", 0)
    if detail != "transcript":
        cap_label = "unlimited" if report_cap is None else str(report_cap)
        engine = frame_meta.get("engine", "scene")
        fallback = " with uniform fallback" if frame_meta.get("fallback") else ""
        deduped = frame_meta.get("deduped_count", 0)
        dedup_note = f", {deduped} near-duplicate{'s' if deduped != 1 else ''} dropped" if deduped else ""
        emit(
            f"- **Frames:** {detail_count} selected from {frame_meta.get('candidate_count', detail_count)} "
            f"candidates ({engine}{fallback}{dedup_note}, {range_mode} range, budget {target}, cap {cap_label})"
        )
    elif not cue_frames:
        emit("- **Frames:** skipped (transcript detail)")
    if dl.get("via_fallback_proxy"):
        emit("- **Download:** via the fallback proxy (≤480p) — the normal egress was blocked")
    if frame_meta.get("engine") == "storyboard":
        step = frame_meta.get("interval_s")
        every = f", one every ~{step:g}s" if step else ""
        emit(
            f"- **Frame source:** YouTube storyboard ({frame_meta.get('tile')} tiles{every}) — "
            "the media download was blocked, so frames are low-res seek previews"
        )
    elif dl.get("blocked") and detail != "transcript":
        emit("- **Frames:** none — media download blocked and no storyboard available")
    if cue_frames:
        dropped = cue_meta.get("dropped_out_of_window", 0)
        drop_note = f", {dropped} dropped outside range" if dropped else ""
        emit(
            f"- **Cue frames:** {len(cue_frames)} at transcript-flagged timestamps "
            f"(transcript-cue{drop_note})"
        )
    if frames:
        emit(f"- **Frame size:** max {args.resolution}px wide, max 1998px tall")
    # OCR status: report "disabled" only when the user asked, and a count only when
    # OCR actually produced text. Staying silent when OCR ran but found nothing (or
    # was unavailable) keeps the default zero-fork-flags output byte-equivalent.
    if args.no_ocr:
        emit("- **OCR:** disabled (`--no-ocr`)")
    elif ocr_text:
        text_frames = sum(1 for v in ocr_text.values() if v.strip())
        emit(f"- **OCR:** {text_frames}/{len(frames)} frame(s) had detected text (lang=spa+eng)")
    if transcript_segments:
        in_range = " in range" if focused else ""
        emit(
            f"- **Transcript:** {len(transcript_segments)} segments{in_range} "
            f"(via {transcript_source or 'captions'})"
        )
        if "approximate timestamps" in (transcript_source or ""):
            emit("- **Note:** Timestamps are model-generated and approximate")
        if use_two_pass and speech_windows:
            speech_total = sum(e - s for s, e in speech_windows)
            denom = effective_duration if focused else full_duration
            emit(
                f"- **Speech windows:** "
                f"{format_windows(speech_windows, speech_total=speech_total, full_duration=denom)}"
            )
    else:
        emit("- **Transcript:** none available")

    if detail == "token-burner" and len(frames) > 250:
        emit()
        emit(
            f"> **Warning:** token-burner detail selected {len(frames)} frames. "
            "This may use a large number of image tokens."
        )

    if not focused and full_duration > 600 and detail not in ("transcript", "token-burner"):
        mins = int(full_duration // 60)
        emit()
        emit(
            f"> **Warning:** This is a {mins}-minute video. Frame coverage is sparse at this length "
            f"under `{detail}` detail — its cap spreads thin across the full clip. For better results, "
            "re-run with `--start HH:MM:SS --end HH:MM:SS` to zoom into a section, or use "
            "`--detail token-burner` to keep every scene-change frame across the whole video."
        )

    emit()
    emit("## Frames")
    emit()
    if frames:
        emit(f"Frames live at: `{work / 'frames'}`")
        emit()
        emit(
            "**Read each frame path below with the Read tool to view the image.** "
            "Frames are in chronological order; `t=MM:SS` is the absolute timestamp in the source video."
        )
        if ocr_text:
            emit()
            emit(
                "Frame lines include any OCR-detected text (Spanish + English). "
                f"Frames with significant text were re-extracted at {HIRES_WIDTH}px for legibility."
            )
        emit()
        # Under two-pass, tag each frame [speech]/[silent] by speech-window membership;
        # keep upstream's reason= annotation and append inline OCR text when present.
        speech_set = list(speech_windows) if use_two_pass else []
        for frame in frames:
            line = (
                f"- `{frame['path']}` "
                f"(t={format_time(frame['timestamp_seconds'])}, reason={frame.get('reason', 'selected')})"
            )
            if speech_set:
                in_speech = any(s <= frame["timestamp_seconds"] <= e for s, e in speech_set)
                line += " [speech]" if in_speech else " [silent]"
            text = ocr_text.get(frame["path"], "").strip() if ocr_text else ""
            if text:
                line += f" — OCR: {' '.join(text.split())}"
            emit(line)
    else:
        emit("_No frames extracted._")

    emit()
    emit("## Transcript")
    emit()
    if transcript_text:
        label = transcript_source or "captions"
        if focused:
            emit(f"_Source: {label}. Filtered to {format_time(effective_start)} → {format_time(effective_end)}:_")
        else:
            emit(f"_Source: {label}._")
        emit()
        emit("```")
        emit(transcript_text)
        emit("```")
    elif detail == "transcript":
        emit(
            "_No transcript available at transcript detail. Captions were missing and Whisper was "
            "unavailable or failed, so there is no visual fallback here. Re-run with "
            "`--detail balanced` for frames._"
        )
    elif focused and dl.get("subtitle_path"):
        emit(f"_No transcript lines fell inside {format_time(effective_start)} → {format_time(effective_end)}._")
    else:
        setup_py = SCRIPT_DIR / "setup.py"
        emit(
            "_No transcript available — proceed with frames only. "
            "Captions were missing and the Whisper fallback was unavailable "
            "(no API key set, or `--no-whisper` was used). "
            f"Run `python3 {setup_py}` to enable Whisper, then re-run._"
        )

    emit()
    emit("---")
    emit(f"_Work dir: `{work}` — delete when done._")

    (work / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    saved = _run_should_save(args, "claude", detail)
    refs: dict = {}
    if saved:
        refs = _save_library(args, work, media_path=video_path, frames=frames,
                             ocr_text=ocr_text, info=info)

    result = _remember(
        args, work, backend="claude", detail=detail, info=info,
        transcript_source=transcript_source, segments=transcript_segments,
        frames=[
            {"t": f["timestamp_seconds"], "path": f.get("path"), "kind": frame_kind(f),
             "ocr": ocr_text.get(f.get("path"))}
            for f in frames
        ],
        notes=[], media_path=video_path,
        rng=(effective_start, effective_end) if focused else None,
        refs=refs, archive=saved,
    )
    _outputs(args, work, result, saved=saved, info=info)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
