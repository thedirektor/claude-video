# /watch → library storage (Immich + Nextcloud) v0.4.3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]` checkboxes.

**Goal:** After each `/watch` run, persist artifacts to a browsable/re-consumable library — video + representative frames → Immich (album per video), `report.md` (transcript+OCR, the no-vision-agent artifact) → Nextcloud (folder per video) — always-on, `--no-save` to skip, best-effort. Ship as v0.4.3.

**Architecture:** New pure-stdlib `library.py` (config + content router + representative-frame selection + Immich uploader + Nextcloud WebDAV uploader). `watch.py` writes `report.md` to the work dir and calls `library.save_artifacts(...)` after printing the report, unless `--no-save`; best-effort (never fails the run). Spec: `docs/superpowers/specs/2026-07-07-watch-library-storage-design.md`.

**Tech:** Python 3.11 stdlib (`urllib`, `json`, `base64`, `ssl`, `hashlib`) — no new deps. pytest with mocked HTTP.

## Global Constraints
- Dev repo `~/claude-video`, branch `feat/v0.4.3-library` from `main` (v0.4.2). NOT in `/opt/watch/claude-video`.
- Pure stdlib; no new pip deps. Windows UTF-8 header on new entrypoint. `encoding="utf-8", errors="replace"` on all reads.
- **Best-effort:** `save_artifacts` and every uploader must NEVER raise into `main()` — on any failure log a one-line stderr note and continue; a partial/missing-config target is skipped, the other still runs.
- Segment/artifact contracts: frame dicts carry `path` (str), `timestamp_seconds` (float), `reason` (str); OCR significance is known from the report's `ocr_text` map (path→text). `{start,end,text}` transcript unchanged.
- Config keys (`~/.config/watch/.env` host + `/opt/watch/secrets/.env` container): `IMMICH_BASE_URL` (default `https://immich.cort3x.me`), `IMMICH_API_KEY`, `NEXTCLOUD_URL` (default `https://nextcloud.cort3x.me`), `NEXTCLOUD_USER` (default `thedirektor`), `NEXTCLOUD_PASS`, `WATCH_SAVE_FRAME_CAP` (default `20`, `0`=all). Secrets NEVER committed; `.env` gitignored.
- Version `0.4.3`. Upstream conventions (no `commands/`, `${SKILL_DIR}`, version synced across SKILL.md + both plugin.json).
- Tests offline (mock `urlopen`). NO push / NO live-clone touch until the Task 5 checkpoint.
- Reference implementation to reuse for Immich: `~/.hermes-personal/scripts/immich_upload.py` (proven on this box) — implementer reads it for the exact `/api/assets` multipart + album flow.

---

### Task 1: Branch setup
- [ ] **Step 1:** `cd ~/claude-video && git checkout main && git status -s` (clean; version 0.4.2). Carry the spec+plan.
```bash
git checkout -b feat/v0.4.3-library main
git add docs/superpowers/plans/2026-07-07-watch-library-storage-0.4.3.md 2>/dev/null || true
git commit -q -m "docs: carry library-storage plan onto v0.4.3 branch" || true
```
- [ ] **Step 2:** `python3 -m pytest -q` → baseline green (152).

---

### Task 2: `library.py` — config, router, frame selection, both uploaders

**Files:** Create `skills/watch/scripts/library.py`; Test `tests/test_library.py`.

**Interfaces (Task 3 depends on these):**
- `library.load_config() -> dict` — reads the 6 keys (env → `~/.config/watch/.env` → `./.env`), applies defaults.
- `library.select_representative_frames(frames: list[dict], ocr_text: dict, cap: int) -> list[dict]` — priority OCR-significant → cue → scene → even-spacing; `cap<=0` returns all; stable timestamp order.
- `library.is_immich_media(path: str) -> bool` — extension in the media set.
- `library.upload_to_immich(files, album_name, cfg) -> dict` — returns `{"album": name, "uploaded": n, "url": album_url_or_None, "error": None|str}`; never raises.
- `library.upload_to_nextcloud(files, folder, cfg) -> dict` — returns `{"folder": path, "uploaded": n, "url": None, "error": None|str}`; never raises.
- `library.save_artifacts(video_path: str|None, frames: list[dict], report_path: str|None, ocr_text: dict, title: str, video_id: str, cfg: dict|None=None) -> list[str]` — the router; returns human-readable summary lines for the report footer; NEVER raises.

- [ ] **Step 1: Write failing tests** `tests/test_library.py` (mock `library.urlopen`):
  - `test_is_immich_media`: `.mp4/.jpg/.png/.webp`→True; `.md/.txt/.json/.vtt`→False.
  - `test_select_frames_keeps_ocr_and_caps`: given 40 frames, 5 with OCR text, cap 10 → returns 10, all 5 OCR-significant included, timestamp-ordered.
  - `test_select_frames_cap_zero_returns_all`.
  - `test_router_splits_no_dup`: `save_artifacts` with a video(.mp4), 3 frames(.jpg), report(.md) and stubbed uploaders → media (video+frames) go to immich, report.md to nextcloud, each file exactly once.
  - `test_router_transcript_only`: video_path=None, no frames, report.md → only nextcloud called.
  - `test_immich_upload_builds_asset_and_album`: mock `urlopen`, assert `POST /api/assets` (x-api-key header, multipart, deviceAssetId stable), `POST /api/albums`, `PUT /api/albums/{id}/assets`.
  - `test_nextcloud_mkcol_put`: mock `urlopen`, assert `MKCOL` folder then `PUT report.md`; a `405` on MKCOL is treated success.
  - `test_missing_config_skips_target`: no `NEXTCLOUD_PASS` → nextcloud skipped (note in summary), immich still attempted; no `IMMICH_API_KEY` → immich skipped.
  - `test_save_artifacts_never_raises`: uploader stubbed to raise → `save_artifacts` returns a summary with the error, does not propagate.

- [ ] **Step 2:** Run → FAIL (no module).
- [ ] **Step 3: Implement `library.py`.** Pure stdlib. Key structure:
  - Windows UTF-8 header; `import base64, hashlib, json, mimetypes, os, ssl, sys, time, urllib.*; from urllib.request import Request, urlopen`.
  - `IMMICH_MEDIA_EXT = {".mp4",".mkv",".webm",".mov",".m4v",".avi",".jpg",".jpeg",".png",".webp",".gif"}`; `is_immich_media` = suffix in set (lowercased).
  - `load_config()`: dotenv reader (copy `transcriptapi.load_api_key`'s parser), return dict with the 6 keys + defaults (`IMMICH_BASE_URL`, `NEXTCLOUD_URL`, `NEXTCLOUD_USER`, `WATCH_SAVE_FRAME_CAP=20`).
  - `select_representative_frames(frames, ocr_text, cap)`: classify each frame — `ocr` if `ocr_text.get(path,"").strip()`, else by `reason` (`cue`/`transcript-cue` → cue; `two-pass`/`scene`/`selected` → scene). Priority order OCR → cue → scene → rest; within a tier keep timestamp order. Take until `cap` (if `cap>0`); if OCR tier alone exceeds cap, truncate to cap. Fill remaining slots by even-spaced sampling of the leftover. Return list preserving overall timestamp order.
  - `_multipart(fields: dict, filename, filebytes, mimetype) -> (body, boundary)`: copy the multipart builder from `whisper.py::_build_multipart` (proven).
  - `upload_to_immich(files, album_name, cfg)`: for each file `POST {IMMICH_BASE_URL}/api/assets` multipart with fields `deviceAssetId=f"watch-{video_id}-{basename}"`, `deviceId="watch-skill"`, `fileCreatedAt`/`fileModifiedAt`=ISO8601 of mtime, header `x-api-key`. Collect returned asset ids. Then `POST /api/albums {"albumName": album_name}` (or reuse existing id if `409`/duplicate — follow the template's approach), `PUT /api/albums/{id}/assets {"ids": [...]}`. Return summary dict. Wrap each network op; on failure record error, continue. Read `~/.hermes-personal/scripts/immich_upload.py` first and match its exact field names/flow.
  - `upload_to_nextcloud(files, folder, cfg)`: Basic auth header `base64(user:pass)`. `MKCOL {NEXTCLOUD_URL}/remote.php/dav/files/{user}/Watch/` then `.../Watch/{folder}/` (treat `405`/`301` on existing as OK). `PUT .../Watch/{folder}/{basename}` with file bytes for each. Return summary dict.
  - `save_artifacts(...)`: build `album/folder = f"{sanitize(title)} ({video_id})"`; split files by `is_immich_media`; select representative frames for the immich media (video always kept, frames via `select_representative_frames`); call each uploader only if its config present (`IMMICH_API_KEY` / `NEXTCLOUD_PASS`), catch everything, return summary lines like `"Immich: album '…' (N assets)"` / `"Nextcloud: Watch/… (N files)"` / `"Nextcloud: skipped (no NEXTCLOUD_PASS)"`.
  - `sanitize(title)`: strip/replace filesystem/URL-unsafe chars, cap length ~80.
- [ ] **Step 4:** Run tests → PASS. Full suite green.
- [ ] **Step 5:** Commit `feat(library): Immich+Nextcloud storage router with representative-frame cap (stdlib, best-effort)`.

---

### Task 3: Wire into `watch.py` — write report.md, save, `--no-save`

**Files:** Modify `skills/watch/scripts/watch.py`; Test `tests/test_watch_library.py`.

**Interfaces:** consumes `library.save_artifacts`. Produces: `report.md` written to `work/report.md`; a save call after the report prints; `--no-save` flag.

- [ ] **Step 1: Failing test** `tests/test_watch_library.py`: for a local clip run (mock `library.save_artifacts` to record its call), assert (a) `work/report.md` exists with the report text, (b) `save_artifacts` called once with the video/frames/report/title/id, (c) `--no-save` → `save_artifacts` NOT called. Keep offline (silent clip, `--no-whisper`, mock save).
- [ ] **Step 2:** Run → FAIL (`--no-save` unknown / no report.md / save not called).
- [ ] **Step 3: Implement.**
  - `import library` in the imports block; add `--no-save` argparse (`store_true`, after `--no-transcriptapi`): help "Do not push this run's artifacts to the library (Immich/Nextcloud). Saving is on by default."
  - `watch.py` currently `print()`s the report. Refactor the report emission to build the report **string** (accumulate lines into a list, or capture) and both `print()` it AND write it to `work / "report.md"` (encoding utf-8). Minimal approach: collect the printed lines into a `report_lines: list[str]`, `print("\n".join(...))` once, then `(work/"report.md").write_text("\n".join(report_lines)+"\n", encoding="utf-8")`. Keep stdout output byte-identical.
  - After writing report.md (end of `main`, before `return 0`), add:
    ```python
    if not args.no_save:
        video_id = (dl.get("info") or {}).get("id") or _short_hash(args.source)
        title = (dl.get("info") or {}).get("title") or Path(args.source).name
        try:
            summary = library.save_artifacts(
                video_path, frames, str(work / "report.md"), ocr_text, title, video_id,
            )
            for line in summary:
                print(f"[watch] {line}", file=sys.stderr)
        except Exception as exc:  # belt: save must never fail the run
            print(f"[watch] library save skipped ({exc})", file=sys.stderr)
    ```
    (`_short_hash(source)` = first 8 of `hashlib.sha1`. `video_id` from yt-dlp info when present; verify `info` carries `id` — if not, fall back to hash. `frames` and `ocr_text` are in scope at end of main.)
- [ ] **Step 4:** Run tests → PASS; full suite green; `import watch` clean.
- [ ] **Step 5:** Commit `feat(watch): write report.md + push artifacts to the library after each run (--no-save opts out)`.

---

### Task 4: setup.py + SKILL.md + CHANGELOG + v0.4.3
- [ ] Add the 6 library keys to `setup.py` env template + a status line (library configured? which targets). Bump `0.4.2`→`0.4.3` in SKILL.md + both plugin.json. SKILL.md: document `--no-save`, the always-on library save, the Immich/Nextcloud split, `WATCH_SAVE_FRAME_CAP`, and the Nextcloud app-password requirement. CHANGELOG `## [0.4.3]` entry. Run suite + `setup.py --check`/`--json`. Commit `docs: v0.4.3 — library storage docs, setup keys, changelog, version bump`.

---

### Task 5: Suite + E2E + gated deploy (NO deploy without approval)
- [ ] Full suite green; `import watch, library` clean; CI paths intact.
- [ ] E2E on the box: a real `/watch` of a short clip WITH a working `NEXTCLOUD_PASS` (app password) → verify the Immich album (video + ≤20 frames) exists via `GET /api/albums?...`, and the Nextcloud `Watch/<title>/report.md` exists via WebDAV `PROPFIND`. Re-run → Immich shows `duplicate` (no dup), Nextcloud PUT overwrites. `--no-save` → nothing uploaded.
- [ ] STOP — user checkpoint. On approval: merge→main, tag v0.4.3, push; add the 6 keys to `/opt/watch/secrets/.env`; pull both clones; `docker compose up -d watch` (recreate — env change); verify a live in-container save; `memory_store`.

## Self-Review
- Spec coverage: router+frames+uploaders (T2) → watch wiring+report.md+`--no-save` (T3) → setup/docs/version (T4) → verify+E2E+deploy (T5). Best-effort never-raises, no-dup routing, representative-cap, partial-config all in constraints + tasks.
- Type consistency: `save_artifacts(video_path, frames, report_path, ocr_text, title, video_id, cfg=None) -> list[str]`; frame dicts `{path,timestamp_seconds,reason}`; config dict keys fixed. Consistent across library.py, tests, watch.py.
- No placeholders in shipped tasks.
