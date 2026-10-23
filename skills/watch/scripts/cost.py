#!/usr/bin/env python3
"""Cost plan and ceiling for one /watch run (no side effects).

``plan(args, probe)`` estimates what a run will spend from its options and a cheap
metadata probe (duration, captions, audio): frames, transcript route, $ per paid
provider, what leaves the box. Only money /watch itself spends counts toward
``WATCH_COST_CEILING_USD`` (default 1.00); the calling client's image tokens for the
``claude`` backend are informational. Prices are list prices per provider (verified
2026-10-23) and can be overridden with ``WATCH_PRICES`` (JSON, merged per key).
"""
from __future__ import annotations

import json
import math

from config import frame_cap, setting
from frames import auto_fps, auto_fps_focus, parse_time

# USD. Token prices are per 1M tokens; transcription per hour of audio.
DEFAULT_PRICES: dict[str, dict] = {
    "gemini:gemini-3.1-flash-lite": {"in": 0.125, "in_audio": 0.25, "out": 0.75},
    "gemini:gemini-2.5-flash": {"in": 0.30, "in_audio": 1.00, "out": 2.50},
    "gemini:gemini-2.5-flash-lite": {"in": 0.10, "in_audio": 0.30, "out": 0.40},
    "gemini:gemini-3.1-pro-preview": {"in": 2.00, "in_audio": 2.00, "out": 12.00},
    "openrouter:google/gemini-2.5-flash": {"in": 0.30, "out": 2.50},
    "openrouter:qwen/qwen3-asr-flash-2026-02-10": {"per_hour": 0.12},
    "groq:whisper-large-v3": {"per_hour": 0.111},
    "groq:whisper-large-v3-turbo": {"per_hour": 0.04},
    "openai:whisper-1": {"per_hour": 0.36},
    "assemblyai:universal": {"per_hour": 0.15},
    "local:whisper": {"per_hour": 0.0},
}
FALLBACK_TOKEN_PRICE = {"in": 1.00, "in_audio": 1.00, "out": 4.00}

GEMINI_LOW_VIDEO_TOKENS_S = 66     # per second of video at LOW media resolution (1 fps)
GEMINI_VIDEO_TOKENS_S = 258        # default media resolution
GEMINI_AUDIO_TOKENS_S = 32
GEMINI_TRANSCRIPT_OUT_TOKENS = 8000
BACKEND_OUT_TOKENS = 2000
TITLES_TOKENS = (2500, 300)        # one LLM call for chapter titles


class CostCeilingExceeded(SystemExit):
    pass


def prices() -> dict[str, dict]:
    table = {k: dict(v) for k, v in DEFAULT_PRICES.items()}
    raw = setting("WATCH_PRICES")
    if raw:
        try:
            for k, v in json.loads(raw).items():
                table.setdefault(k, {}).update(v)
        except (ValueError, AttributeError):
            pass
    return table


def ceiling() -> float:
    try:
        return float(setting("WATCH_COST_CEILING_USD") or "1.00")
    except ValueError:
        return 1.00


def _has_key(name: str) -> bool:
    return bool(setting(name))


def _price(key: str) -> dict:
    return prices().get(key) or FALLBACK_TOKEN_PRICE


def _tokens_usd(key: str, tin: float = 0, tin_audio: float = 0, tout: float = 0) -> float:
    p = _price(key)
    return (tin * p.get("in", 0) + tin_audio * p.get("in_audio", p.get("in", 0)) + tout * p.get("out", 0)) / 1e6


def _hours_usd(key: str, seconds: float) -> float:
    return _price(key).get("per_hour", 0.0) * seconds / 3600


def _whisper_route(args) -> str | None:
    choice = (args.whisper or "").strip().lower() or None
    keys = {"groq": "GROQ_API_KEY", "openai": "OPENAI_API_KEY", "assemblyai": "ASSEMBLYAI_API_KEY"}
    if choice and choice != "auto":
        return choice if choice == "local" or _has_key(keys.get(choice, "")) else None
    try:
        import whisper
        return whisper.resolve_backend(None)[0]
    except Exception:  # noqa: BLE001
        return next((b for b in ("groq", "openai") if _has_key(keys[b])), None)


def _whisper_cost(route: str, seconds: float, diarize: bool) -> tuple[str, str, float]:
    if route == "local":
        return "local", "local:whisper", 0.0
    if route == "groq":
        return "groq", "groq:whisper-large-v3", _hours_usd("groq:whisper-large-v3", seconds)
    if route == "openai":
        return "openai", "openai:whisper-1", _hours_usd("openai:whisper-1", seconds)
    return "assemblyai", "assemblyai:universal", _hours_usd("assemblyai:universal", seconds)


def plan(args, probe: dict) -> dict:
    """Estimate one run. ``probe``: {duration, has_captions, youtube, has_audio}."""
    full = float(probe.get("duration") or 0)
    start, end = parse_time(args.start), parse_time(args.end)
    a = start or 0.0
    b = end if end is not None else full
    dur = max(0.0, b - a) if full or end is not None else 0.0
    focused = start is not None or end is not None
    detail = args.detail or "balanced"
    cap = args.max_frames if args.max_frames is not None else frame_cap(detail)
    if detail == "transcript":
        frames = 0
    else:
        _, target = (auto_fps_focus if focused else auto_fps)(dur, max_frames=cap or 100)
        frames = target if cap is None else min(target, cap)

    costs: list[dict] = []
    credits: dict[str, int] = {}
    leaves = []
    youtube = bool(probe.get("youtube"))
    backend = args.backend or "claude"
    if youtube:
        leaves.append("YouTube (yt-dlp: metadata, captions, media) via the configured egress")

    route = "none"
    if backend == "claude":
        if youtube and not args.no_transcriptapi and _has_key("TRANSCRIPTAPI_API_KEY"):
            route = "transcriptapi"
            credits["transcriptapi"] = 1
            leaves.append("transcriptapi.com (video URL)")
        elif probe.get("has_captions"):
            route = "captions"
        elif youtube and not args.no_gemini_transcript and _has_key("GEMINI_API_KEY"):
            route = "gemini"
            key = "gemini:gemini-3.1-flash-lite"
            costs.append({"item": "transcript (Gemini watches the URL, low resolution)", "provider": "gemini",
                          "usd": _tokens_usd(key, dur * GEMINI_LOW_VIDEO_TOKENS_S, dur * GEMINI_AUDIO_TOKENS_S,
                                             GEMINI_TRANSCRIPT_OUT_TOKENS),
                          "basis": f"{dur:.0f}s × {GEMINI_LOW_VIDEO_TOKENS_S + GEMINI_AUDIO_TOKENS_S} tok/s"})
            leaves.append("Google Gemini API (video URL)")
        elif not args.no_whisper and (probe.get("has_audio", True) or args.audio):
            w = _whisper_route(args)
            if w:
                route = f"whisper:{w}"
                provider, key, usd = _whisper_cost(w, dur, bool(args.diarize))
                costs.append({"item": "transcript (Whisper)", "provider": provider, "usd": usd,
                              "basis": f"{dur / 60:.1f} min of audio"})
                if provider != "local":
                    leaves.append(f"{provider} (audio upload)")
        claude_tokens = frames * math.ceil(args.resolution * (args.resolution * 9 / 16) / 750)
    else:
        claude_tokens = 0

    if backend == "gemini":
        model = args.gemini_model or "gemini-3.1-flash-lite"
        key = f"gemini:{model}"
        costs.append({"item": f"answer ({model} watches the video)", "provider": "gemini",
                      "usd": _tokens_usd(key, dur * GEMINI_VIDEO_TOKENS_S, dur * GEMINI_AUDIO_TOKENS_S,
                                         BACKEND_OUT_TOKENS),
                      "basis": f"{dur:.0f}s × {GEMINI_VIDEO_TOKENS_S + GEMINI_AUDIO_TOKENS_S} tok/s"})
        leaves.append("Google Gemini API (" + ("video URL" if youtube else "video upload") + ")")
        route = "captions (free, indexed)"
    elif backend == "openrouter":
        model = args.openrouter_vision_model or "google/gemini-2.5-flash"
        per_frame = math.ceil(args.resolution * (args.resolution * 9 / 16) / 750)
        costs.append({"item": f"answer ({model}, {frames} frames)", "provider": "openrouter",
                      "usd": _tokens_usd(f"openrouter:{model}", frames * per_frame + dur * 3, 0, BACKEND_OUT_TOKENS),
                      "basis": f"{frames} frames × {per_frame} tok + transcript"})
        leaves.append("OpenRouter (frames + transcript)")
        route = "captions, else OpenRouter audio model"

    titles = args.llm_titles or (setting("WATCH_LLM_TITLES") or "0").strip() in ("1", "true", "yes")
    if titles and _has_key("GEMINI_API_KEY"):
        costs.append({"item": "chapter titles (one LLM call)", "provider": "gemini",
                      "usd": _tokens_usd("gemini:gemini-3.1-flash-lite", TITLES_TOKENS[0], 0, TITLES_TOKENS[1]),
                      "basis": "~2.8k tokens"})

    total = round(sum(c["usd"] for c in costs), 4)
    for c in costs:
        c["usd"] = round(c["usd"], 4)
    lim = ceiling()
    return {"source": args.source, "duration_s": round(dur, 1), "backend": backend, "detail": detail,
            "frames_est": frames, "transcript_route": route, "costs": costs, "credits": credits,
            "total_usd": total, "ceiling_usd": lim, "within_ceiling": lim <= 0 or total <= lim,
            "informational": {"claude_image_tokens": claude_tokens},
            "leaves_box": leaves}


def enforce(p: dict, allow: bool = False) -> None:
    """Fail fast (before any paid call) when the plan exceeds the ceiling."""
    if allow or p["within_ceiling"]:
        return
    raise CostCeilingExceeded(
        f"estimated cost ${p['total_usd']:.2f} exceeds WATCH_COST_CEILING_USD ${p['ceiling_usd']:.2f} — "
        f"re-run with --allow-cost (or raise the ceiling).\n{render(p)}")


def render(p: dict) -> str:
    lines = [f"# watch: cost plan",
             f"- source: {p['source']}",
             f"- duration: {p['duration_s']:.0f}s · backend: {p['backend']} · detail: {p['detail']} · "
             f"frames ≈ {p['frames_est']}",
             f"- transcript route: {p['transcript_route']}"]
    for c in p["costs"]:
        lines.append(f"- {c['item']} [{c['provider']}]: ${c['usd']:.4f} ({c['basis']})")
    for k, n in p["credits"].items():
        lines.append(f"- {k}: {n} credit(s) (prepaid)")
    lines.append(f"- total: ${p['total_usd']:.2f} (ceiling ${p['ceiling_usd']:.2f})")
    if p["informational"]["claude_image_tokens"]:
        lines.append(f"- informational: ~{p['informational']['claude_image_tokens']:,} image tokens for the "
                     "calling Claude session (not counted)")
    if p["leaves_box"]:
        lines.append("- leaves the box: " + "; ".join(p["leaves_box"]))
    return "\n".join(lines)
