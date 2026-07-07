# /watch → library storage (Immich + Nextcloud) — design

Date: 2026-07-07
Repo: `thedirektor/claude-video` (the `/watch` skill)
Status: approved (design), pending spec review
Version: **0.4.3** (keeps v0.5.0 for the planned Phase-3 debug features)

## Purpose

`/watch` turns a video into a text-first report (timestamped transcript + OCR of
on-screen text + representative frames). This feature **persists each watch into a
browsable, re-consumable library** so an agent — especially one **without video
vision** (Claude Code, other agents) — can later *learn from* a video it never
"watched": read the transcript, read the on-screen text/code (OCR), and pull the
key frames when visual detail is needed. Use cases: debugging from a screen
recording, implementing from a tutorial, retaining reference material.

The library is therefore split by **who consumes what**:
- **`report.md` (transcript + OCR + frame index) → Nextcloud.** This is the
  primary artifact a **no-vision agent reads** to understand the video. It is
  self-sufficient text.
- **video + representative frames → Immich.** The **visual source** — human
  gallery browsing, Immich's own search, and a vision-capable step when needed.

## Non-goals

- No new pip dependencies (stdlib `urllib` HTTP, like `whisper.py`/`transcriptapi.py`).
- Not a sync/dedup engine across the whole library; per-run best-effort upload only.
- No rewriting of the report's frame paths to remote URLs (v1): the report's value
  to a no-vision agent is its **text** (transcript + inline OCR), which is complete
  on its own. Resolvable remote frame links are a possible later enhancement.
- No Immich **External Library** scan path — direct API upload only (instant,
  matches the "just watched a video" workflow).

## Architecture

A new pure-stdlib module `skills/watch/scripts/library.py`:
- `save_artifacts(...)` — the **content-type router**: given the run's artifacts
  (video path or None, frame dicts, `report.md` path, title, video id/hash) and
  config, routes each file to exactly one target and returns a short result
  summary (what landed where) for the report footer.
- `upload_to_immich(...)` — video + representative frames → Immich album.
- `upload_to_nextcloud(...)` — non-media files (`report.md`, …) → Nextcloud folder.
- `load_config()` — reads the target endpoints/keys from env → `~/.config/watch/.env`
  → `./.env` (mirrors `transcriptapi.load_api_key`).

`watch.py` calls `library.save_artifacts(...)` **once, after the report has been
printed**, unless `--no-save`. It is **best-effort**: wrapped so any failure logs a
one-line warning to stderr and returns — it must NEVER fail or alter the `/watch`
run (the report is already on stdout). The report ends with a footer line naming
the Immich album and Nextcloud folder (or the reason a target was skipped).

Each unit is independently testable: the router (pure routing logic), each uploader
(mocked HTTP), and config loading.

## Content router (no duplicates)

Route every artifact by extension to exactly one target:
- **Immich media set:** `.mp4 .mkv .webm .mov .m4v .avi` (video) and
  `.jpg .jpeg .png .webp .gif` (frames) → **Immich**.
- **Everything else** (`.md .txt .json .vtt .srt` …) → **Nextcloud**.

Artifacts a run produces: the downloaded **video** (only when a real download
happened — absent for TranscriptAPI-only YouTube runs and for `--detail transcript`),
the selected **frames**, and **`report.md`** (written to the work dir; see below).
A run with no media (transcript-only) simply uploads `report.md` to Nextcloud and
does nothing on Immich — the router handles empty sets.

### Representative frame cap

`/watch`'s pipeline already yields a scene-aware, deduped, detail-capped frame set
(efficient≤50, balanced≤100). For the library we upload a **further-reduced
representative subset** so Immich isn't flooded and the agent gets the *informative*
moments. Selection, capped at `WATCH_SAVE_FRAME_CAP` (default **20**, `0` = all,
configurable in `~/.config/watch/.env`):
1. Always keep **OCR-significant** frames (those with detected on-screen text — the
   hi-res re-extracted ones; critical for tutorials/code/debugging).
2. Then **transcript-cue** frames (moments the transcript flagged).
3. Then **scene-representative** frames.
4. If still under the cap, fill by **even time spacing** across the remaining
   selected frames.
5. If the count already ≤ cap, keep all; if a category alone exceeds the cap, keep
   it in timestamp order truncated to the cap (OCR-significant wins).
The frame dicts at the save point already carry `path`, `timestamp_seconds`,
`reason`, and OCR is known via the report's `ocr_text` map — enough to classify
without re-processing.

## Immich uploader

Reuse the **proven, already-in-production** pattern on this box:
`~/.hermes-personal/scripts/immich_upload.py` (a hermes hook that uploads
generated media into Immich albums today). Flow:
1. `POST {IMMICH_BASE_URL}/api/assets` — multipart: `assetData` (file),
   `deviceAssetId` (stable per file → dedup: re-runs of the same video+frame do
   not duplicate; use e.g. `watch-<videoid>-<basename>`), `deviceId`
   (`"watch-skill"`), `fileCreatedAt`/`fileModifiedAt` (file mtime, ISO8601).
   Header `x-api-key: {IMMICH_API_KEY}`. Returns `{id, status}` (status
   `created`|`duplicate`).
2. `POST /api/albums` `{"albumName": "<title> (<videoid>)"}` (or look up existing by
   name) → album id.
3. `PUT /api/albums/{id}/assets` `{"ids": [assetIds]}` → group video + frames.
Config: `IMMICH_BASE_URL` (default `https://immich.cort3x.me`), `IMMICH_API_KEY`
(**already validated** on the box; in `~/.hermes/.env` and to be added to watch
config). The external `/api` router bypasses Authelia (verified), so scripted calls
work. Best-effort: partial failures (one frame fails) log and continue.

## Nextcloud uploader

WebDAV over stdlib `urllib` with Basic auth:
1. `MKCOL {NEXTCLOUD_URL}/remote.php/dav/files/{NEXTCLOUD_USER}/Watch/` then
   `.../Watch/<title (videoid)>/` (idempotent — `405 Method Not Allowed` on an
   existing collection is treated as success).
2. `PUT .../Watch/<title (videoid)>/report.md` (+ any other non-media file).
   `PUT` overwrites → idempotent, no duplicates.
Config: `NEXTCLOUD_URL` (default `https://nextcloud.cort3x.me`), `NEXTCLOUD_USER`
(default `thedirektor`), `NEXTCLOUD_PASS` (a **Nextcloud app password** — Settings →
Security → Devices & Sessions — **the user must generate this**; none exists on the
box). The `nextcloud.cort3x.me` router has no Authelia middleware, so Basic-Auth
WebDAV clients aren't redirected to a login page.

### report.md at the save point

`watch.py` currently prints the report to **stdout** and does not write a
`report.md` file. This feature writes the report text to `work/report.md` (so it
can be uploaded) in addition to printing it. The written report is the **full**
report (transcript + OCR + frame index) — the complete text a no-vision agent needs.

## Trigger, config, partial behavior

- **Always-on**, `--no-save` skips. (No per-target flags — the router decides
  target per file.)
- **Graceful partials:** each target is attempted only if its required config is
  present. No `NEXTCLOUD_PASS` → skip Nextcloud with a one-line note, still do
  Immich. No `IMMICH_API_KEY` → skip Immich, still do Nextcloud. A target's HTTP
  failure logs and is skipped; the run is unaffected.
- **Config keys** (`~/.config/watch/.env` host + `/opt/watch/secrets/.env`
  container): `IMMICH_BASE_URL`, `IMMICH_API_KEY`, `NEXTCLOUD_URL`,
  `NEXTCLOUD_USER`, `NEXTCLOUD_PASS`, `WATCH_SAVE_FRAME_CAP`. `setup.py` documents
  them in its env template + status.
- **Deploy note:** container secret changes need `docker compose up -d watch`
  (recreate), not `restart` (env_file is read at create).

## Error handling

- Every network path: bounded retries where sensible, then a one-line actionable
  stderr note (which target, why skipped) — never a raise into `main()`.
- Missing/partial config degrades gracefully (skip that target, say so).
- Late-stage: the report is printed BEFORE saving, so a save failure never costs
  the user the report.

## Testing

- **Unit (offline, mocked HTTP):** router classification (which file → which
  target, empty-media run); representative-frame selection (OCR-significant kept,
  cap honored, `0`=all); Immich upload builds correct multipart + album calls
  (mock `urlopen`, assert endpoints/headers/dedup id); Nextcloud MKCOL/PUT argv +
  `405`-is-ok; config loading precedence; `--no-save` skips entirely;
  missing-config skips one target but not the other.
- **E2E (manual, on the box):** a real `/watch` of a short clip → verify the Immich
  album appears with video + capped frames, and the Nextcloud `Watch/<title>/`
  folder holds `report.md`; re-run → no duplicates.

## Out of scope (documented)

- Remote/resolvable frame links inside `report.md` (v1 relies on the report's text).
- Immich External Library scan path.
- Nextcloud `occ files:scan` (WebDAV PUT auto-indexes).
- Retention/cleanup of old library entries.
