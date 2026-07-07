# claude-video / watch — backlog (deferred TODOs)

Captured 2026-07-07 (end of the v0.4.0→v0.4.3 session). Newest first.

## Near-term polish
- **Library entry titles from TranscriptAPI metadata.** `watch.py` names the Immich album + Nextcloud folder from yt-dlp's `dl["info"]["title"]`, which is empty for YouTube URLs on this box (bot-gated), so they fall back to the URL basename (e.g. `watch v=7kSKQLd6WLs`). TranscriptAPI returns `title`/`author` (seen via `get_youtube_video_info`) — thread that into `video_id`/`title` at the save call so YouTube library entries get real names. Small change in `watch.py` (and maybe surface title from `transcriptapi.fetch_transcript`).
- **Test the untested-but-correct `library.py` branches** (final-review minors): `_find_album` album-found branch; non-dict-JSON resilience in `upload_to_immich`. Add regression tests.

## YouTube frames on the box (bigger)
- YouTube *video/audio download* (needed for frames) is bot-gated from this datacenter IP. Beating it needs cookies (`--cookies`) **plus** a POT provider (bgutil-ytdlp-pot-provider, node script-mode — node is present) **plus** a JS runtime (Deno/EJS) for the nsig `n` challenge. Alternatively a MeTube-with-cookies fallback (`http://metube:8081` `/upload-cookies` + `/add`), or run `/watch` from a residential IP. Transcript already works via TranscriptAPI; only frames are blocked.

## Library — later enhancements (from the v0.4.3 spec "out of scope")
- Resolvable remote frame links inside `report.md` (so a later agent can pull the exact Immich assets referenced), instead of relying on the report's text alone.
- Register a folder as an Immich **External Library** (scan-based) as an alternative to direct API upload.
- Retention/cleanup of old library entries.
- Immich→watch: none. Nextcloud `occ files:scan` not needed (WebDAV PUT auto-indexes).

## Cross-cutting minors (from SDD final reviews, non-blocking)
- `--backend gemini`/`openrouter` return early **before** the TranscriptAPI block, so they don't get the TranscriptAPI transcript for YouTube (openrouter still needs the gated audio download). Document the backend scope, or wire TranscriptAPI in for openrouter.
- Frame-stage resume (`frames.done`) was deferred in v0.4.0 (chunk cache + transcript-stage save cover the acceptance case).
- Whisper/TranscriptAPI retry backoff can add a few seconds on a persistent outage (best-effort).

## Roadmap (original 0.3.0→0.6.0 design)
- **Phase 3 → v0.5.0:** debug-oriented features (`--density`, bundle save/load, `--json` output, scene→slide grouping).
- **Phase 4 → v0.6.0:** `/edit` colorist command (ffmpeg LUT/look/letterbox/trim/blur/watermark/loudnorm, smoke_edit.py).

## Separate (infra)
- watch→**Immich** storage is done (v0.4.3). A deeper Immich *integration* (albums-as-collections, dedup across the whole library) is not planned.
- Dev hygiene: running `watch.py` with the config present uploads to the REAL library — use `--no-save` in tests. Consider a dev/env guard.
