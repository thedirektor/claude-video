# Phase 2 — Reliability pack (v0.4.0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `/watch` survive real-world friction — YouTube 403/SABR blocks, Spanish-first captions, Whisper rate-limit/crash mid-run, and interrupted long jobs — shipping as v0.4.0.

**Architecture:** Four independent reliability features, each landing as its own commit(s) with tests, on top of the released 0.3.0 layout. Feature 1 & 2 harden `download.py`'s two yt-dlp argv builders. Feature 3 adds a content-addressed chunk cache and window-only audio extraction to `whisper.py`. Feature 4 adds a new pure `state.py` module and wires download+transcript stage resume into `watch.py`. No new pip dependencies — all stdlib (`hashlib`, `json`).

**Tech Stack:** Python 3.11 stdlib core; `yt-dlp`/`ffmpeg`/`ffprobe` binaries; pytest with ffmpeg-synthesized clips and mocked HTTP (offline in CI).

## Global Constraints

- **Dev repo:** `~/claude-video` (the canonical dev workspace). Do **all** work there. Do NOT dev in `/opt/watch/claude-video` — that is the read-only container bind-mount (`./claude-video:/app/claude-video:ro` in `/opt/watch/docker-compose.yml`); a branch switch there changes what the live `watch` service sees.
- **Working branch:** `feat/v0.4.0-reliability`, created in Task 1 from the clean `0.3.0` base. All commits land there.
- Pure stdlib in core paths; optional heavy deps stay lazy-imported. **No new pip dependencies** (spec non-goal).
- Windows UTF-8 pattern — every new script entrypoint keeps this exact header block after imports:
  ```python
  for _stream in (sys.stdout, sys.stderr):
      try:
          _stream.reconfigure(encoding="utf-8", errors="replace")
      except (AttributeError, OSError):
          pass
  ```
- All `~/.config/watch/.env` and cache reads use `encoding="utf-8", errors="replace"`.
- Upstream conventions are law (`AGENTS.md`): NO `commands/` directory; `${SKILL_DIR}` in SKILL.md, never `${CLAUDE_SKILL_DIR}`; the skill folder `skills/watch/` stays self-contained; version synced across `skills/watch/SKILL.md` frontmatter, `.claude-plugin/plugin.json`, `.codex-plugin/plugin.json`.
- **Version this phase:** `0.4.0` (bumped in Task 10, along with `CHANGELOG.md`).
- **Error-handling principles (spec):** every network path has bounded retries with a cap, then one actionable one-line error; optional-dep paths degrade gracefully and say what was skipped; late-stage crashes must not destroy earlier artifacts (this is exactly what Feature 4 formalizes).
- **Cache/state locations:** transcript chunk cache at `~/.cache/watch/transcripts/`; resume stage files inside the run's `--out-dir` work dir as `stage_*.json`. Both are best-effort — a cache/state failure must never fail the run.
- Tests: run from repo root with `python3 -m pytest -q`. ffmpeg/ffprobe required locally (present on the box). All new tests run offline (mock HTTP, ffmpeg-synthesized or stubbed subprocess).
- **NO pushing to origin, NO touching `~/.claude/skills/watch` or `/opt/watch/claude-video`** — deployment happens only after the user checkpoint in Task 12.

---

### Task 1: Branch setup

**Files:**
- Create: branch `feat/v0.4.0-reliability` in `~/claude-video` from the `0.3.0` base commit
- Carry over: `docs/superpowers/specs/` + `docs/superpowers/plans/` (this plan)

**Interfaces:**
- Produces: the working branch every later task commits to.

- [ ] **Step 1: Confirm the dev repo base is clean 0.3.0**

```bash
cd ~/claude-video
git status -s                       # expect: clean (or only untracked docs)
grep '"version"' .claude-plugin/plugin.json   # expect: "0.3.0"
```
Expected: version `0.3.0`. If the tree is dirty with unrelated work, STOP and report — do not branch over uncommitted changes.

- [ ] **Step 2: Create the phase-2 branch from the 0.3.0 release state**

```bash
cd ~/claude-video
git fetch origin
# Base the branch on origin/main (the released 0.3.0 line). If origin/main is
# not at 0.3.0, base on the local commit whose plugin.json reads 0.3.0 and STOP
# to confirm with the user before proceeding.
git checkout -b feat/v0.4.0-reliability origin/main
```
Expected: new branch created. `grep '"version"' .claude-plugin/plugin.json` still reads `0.3.0`.

- [ ] **Step 3: Carry this plan + spec onto the branch**

```bash
cd ~/claude-video
mkdir -p docs/superpowers/plans docs/superpowers/specs
cp /opt/watch/claude-video/docs/superpowers/plans/2026-07-07-phase2-reliability-0.4.0.md docs/superpowers/plans/
cp /opt/watch/claude-video/docs/superpowers/specs/2026-07-06-claude-video-0.3.0-design.md docs/superpowers/specs/ 2>/dev/null || true
git add docs/
git commit -m "docs: carry phase-2 reliability plan onto v0.4.0 branch"
```

- [ ] **Step 4: Verify the baseline test suite is green**

```bash
python3 -m pytest -q
```
Expected: all tests pass. If ffmpeg-dependent tests fail, STOP — the box has ffmpeg, so a failure means the base is wrong.

---

### Task 2: yt-dlp hardening — extractor fallback chain + anti-bot (Feature 1, part 1)

**Files:**
- Modify: `skills/watch/scripts/download.py` (add `HARDENING_ARGS`, splice into both argv builders)
- Test: `tests/test_download.py` (add hardening assertions)

**Interfaces:**
- Produces: module constant `HARDENING_ARGS: list[str]` and its presence in the argv built by `fetch_captions` and `download_url`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_download.py` (the file already has `_capture_argv`, `URL`, and the `SCRIPTS_DIR` sys.path insert):

```python
def _has_flag_value(argv: list[str], flag: str, value: str) -> bool:
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv) and argv[i + 1] == value:
            return True
    return False


def test_fetch_captions_uses_player_client_chain(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch)
    download.fetch_captions(URL, tmp_path / "download")
    argv = calls[0]
    assert _has_flag_value(argv, "--extractor-args", "youtube:player_client=default,android,mweb")
    assert "--retries" in argv and "--extractor-retries" in argv


def test_download_url_uses_player_client_chain(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch)
    with pytest.raises(SystemExit):
        download.download_url(URL, tmp_path / "download")
    argv = calls[0]
    assert _has_flag_value(argv, "--extractor-args", "youtube:player_client=default,android,mweb")
    assert "--sleep-interval" in argv and "--max-sleep-interval" in argv
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_download.py -k player_client -v`
Expected: FAIL — `--extractor-args` not in argv.

- [ ] **Step 3: Add `HARDENING_ARGS` and splice it in**

In `skills/watch/scripts/download.py`, add this constant right after `VIDEO_EXTS = {...}` (near line 22):

```python
# yt-dlp hardening. YouTube's SABR/403 rollout breaks the default `web` client;
# a player_client fallback chain (tried in order) keeps most public videos
# reachable without cookies. Anti-bot retry + request jitter absorb transient
# 403/429s. Sources: upstream PRs #46, #42, #27, #21.
HARDENING_ARGS = [
    "--extractor-args", "youtube:player_client=default,android,mweb",
    "--retries", "10",
    "--fragment-retries", "10",
    "--extractor-retries", "3",
    "--sleep-requests", "1",
    "--sleep-interval", "1",
    "--max-sleep-interval", "5",
]
```

In `fetch_captions`, insert `*HARDENING_ARGS,` into the `cmd` list immediately before `"-o", output_template,`:

```python
        "--no-playlist",
        "--ignore-errors",
        *HARDENING_ARGS,
        "-o", output_template,
        "--",
        url,
    ]
```

In `download_url`, insert the same `*HARDENING_ARGS,` line immediately before `"-o", output_template,` in that function's `cmd` list.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_download.py -v`
Expected: PASS (new tests + existing tests still green).

- [ ] **Step 5: Commit**

```bash
git add skills/watch/scripts/download.py tests/test_download.py
git commit -m "feat(download): yt-dlp player_client fallback chain + anti-bot retry/jitter"
```

---

### Task 3: Cookie passthrough flags (Feature 1, part 2)

**Files:**
- Modify: `skills/watch/scripts/download.py` (add `_cookie_args`, thread params through `fetch_captions`, `download_url`, `download`)
- Modify: `skills/watch/scripts/watch.py` (add `--cookies-from-browser` / `--cookies` argparse + thread to calls)
- Test: `tests/test_download.py`

**Interfaces:**
- Consumes: `HARDENING_ARGS` (Task 2).
- Produces: `download.download(source, out_dir, audio_only=False, sub_langs=..., cookies_from_browser=None, cookies_file=None)` signature (sub_langs added in Task 4; add the two cookie params now). `fetch_captions` and `download_url` gain the same two cookie keyword params.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_download.py`:

```python
def test_no_cookie_flags_by_default(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch)
    download.fetch_captions(URL, tmp_path / "download")
    argv = calls[0]
    assert "--cookies" not in argv
    assert "--cookies-from-browser" not in argv


def test_cookies_from_browser_passed_through(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch)
    download.fetch_captions(URL, tmp_path / "download", cookies_from_browser="firefox")
    assert _has_flag_value(calls[0], "--cookies-from-browser", "firefox")


def test_cookies_file_passed_through(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch)
    with pytest.raises(SystemExit):
        download.download_url(URL, tmp_path / "download", cookies_file="/tmp/c.txt")
    assert _has_flag_value(calls[0], "--cookies", "/tmp/c.txt")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_download.py -k cookie -v`
Expected: FAIL — `fetch_captions()` got an unexpected keyword argument `cookies_from_browser`.

- [ ] **Step 3: Add `_cookie_args` and thread the params**

In `download.py`, add after `HARDENING_ARGS`:

```python
def _cookie_args(cookies_from_browser: str | None, cookies_file: str | None) -> list[str]:
    """Build the yt-dlp cookie flags. Off by default — both None yields []."""
    args: list[str] = []
    if cookies_from_browser:
        args += ["--cookies-from-browser", cookies_from_browser]
    if cookies_file:
        args += ["--cookies", cookies_file]
    return args
```

Change `fetch_captions` signature to:

```python
def fetch_captions(
    url: str,
    out_dir: Path,
    cookies_from_browser: str | None = None,
    cookies_file: str | None = None,
) -> dict:
```

and in its `cmd` list, add `*_cookie_args(cookies_from_browser, cookies_file),` immediately after the `*HARDENING_ARGS,` line.

Change `download_url` signature to:

```python
def download_url(
    url: str,
    out_dir: Path,
    audio_only: bool = False,
    cookies_from_browser: str | None = None,
    cookies_file: str | None = None,
) -> dict:
```

and add the same `*_cookie_args(...)` line after `*HARDENING_ARGS,` in its `cmd`.

Change `download` to thread them:

```python
def download(
    source: str,
    out_dir: Path,
    audio_only: bool = False,
    cookies_from_browser: str | None = None,
    cookies_file: str | None = None,
) -> dict:
    if is_url(source):
        return download_url(
            source,
            out_dir,
            audio_only=audio_only,
            cookies_from_browser=cookies_from_browser,
            cookies_file=cookies_file,
        )
    return resolve_local(source)
```

- [ ] **Step 4: Wire the argparse flags in watch.py**

In `skills/watch/scripts/watch.py`, add these arguments in `main()` right after the `--audio` argument block (near line 341):

```python
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
```

Then update the two `download(...)` calls and the `fetch_captions(...)` call in `main()` to pass the cookies through. The `fetch_captions` call (near line 452):

```python
        dl = fetch_captions(
            args.source,
            work / "download",
            cookies_from_browser=args.cookies_from_browser,
            cookies_file=args.cookies,
        )
```

The URL `download(...)` call (near line 476):

```python
            dl = download(
                args.source,
                work / "download",
                audio_only=audio_only,
                cookies_from_browser=args.cookies_from_browser,
                cookies_file=args.cookies,
            )
```

(The local-file `download(args.source, work / "download")` call near line 483 needs no change — cookies are irrelevant for local files.)

- [ ] **Step 5: Run tests**

Run: `python3 -m pytest tests/test_download.py -v && python3 -c "import sys; sys.path.insert(0,'skills/watch/scripts'); import watch"`
Expected: PASS; watch.py imports cleanly.

- [ ] **Step 6: Commit**

```bash
git add skills/watch/scripts/download.py skills/watch/scripts/watch.py tests/test_download.py
git commit -m "feat(download): --cookies-from-browser / --cookies passthrough (off by default)"
```

---

### Task 4: Spanish-first subtitle languages + `--sub-lang` override (Feature 2)

**Files:**
- Modify: `skills/watch/scripts/download.py` (`DEFAULT_SUB_LANGS`, `sub_langs` param, language-aware `_pick_subtitle`)
- Modify: `skills/watch/scripts/watch.py` (`--sub-lang` argparse + thread to calls)
- Test: `tests/test_download.py` — **rewrite the two English-only regression tests** (their premise is now inverted) and add override + pick-order tests

**Interfaces:**
- Consumes: `_cookie_args`, `HARDENING_ARGS`.
- Produces: `download.DEFAULT_SUB_LANGS: str` (`"es,es-419,es-ES,en,en-US,en-GB,*-orig"`); `sub_langs: str` keyword on `fetch_captions`, `download_url`, `download`; `_pick_subtitle(out_dir, sub_langs=DEFAULT_SUB_LANGS)`.

- [ ] **Step 1: Rewrite the regression guard + add new failing tests**

The existing `tests/test_download.py` guard asserts **English-only** sub-langs. Phase 2 inverts that: the default is Spanish-first, and the only invariant left is "bounded, never `all`". **Replace** the `_assert_english_only` helper and the two `*_requests_english_only` tests with:

```python
DEFAULT_LANGS = "es,es-419,es-ES,en,en-US,en-GB,*-orig"


def _assert_bounded(langs: str) -> None:
    tokens = [t.strip() for t in langs.split(",")]
    assert "all" not in tokens, f"sub-langs must never request all languages, got {langs!r}"
    assert tokens, f"sub-langs must be non-empty, got {langs!r}"


def test_fetch_captions_default_langs_are_spanish_first_and_bounded(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch)
    download.fetch_captions(URL, tmp_path / "download")
    langs = _sub_langs(calls[0])
    _assert_bounded(langs)
    assert langs == DEFAULT_LANGS


def test_download_url_default_langs_are_spanish_first_and_bounded(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch)
    with pytest.raises(SystemExit):
        download.download_url(URL, tmp_path / "download")
    langs = _sub_langs(calls[0])
    _assert_bounded(langs)
    assert langs == DEFAULT_LANGS


def test_sub_langs_override_is_honored(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch)
    download.fetch_captions(URL, tmp_path / "download", sub_langs="de,fr")
    langs = _sub_langs(calls[0])
    _assert_bounded(langs)
    assert langs == "de,fr"
```

Add a pure `_pick_subtitle` ordering test (no network — just touch files):

```python
def test_pick_subtitle_prefers_language_order(tmp_path):
    d = tmp_path / "download"
    d.mkdir()
    for name in ("video.en.vtt", "video.es.vtt", "video.fr.vtt"):
        (d / name).write_text("WEBVTT\n", encoding="utf-8")
    picked = download._pick_subtitle(d)  # default es-first
    assert picked is not None and picked.name == "video.es.vtt"


def test_pick_subtitle_falls_back_to_orig(tmp_path):
    d = tmp_path / "download"
    d.mkdir()
    (d / "video.en-orig.vtt").write_text("WEBVTT\n", encoding="utf-8")
    picked = download._pick_subtitle(d)
    assert picked is not None and picked.name == "video.en-orig.vtt"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_download.py -k "spanish or override or pick_subtitle" -v`
Expected: FAIL — `DEFAULT_SUB_LANGS` not defined / `sub_langs` kwarg unexpected / pick order wrong.

- [ ] **Step 3: Implement Spanish-first defaults + language-aware pick**

In `download.py`, add after `VIDEO_EXTS` (and before/after `HARDENING_ARGS` — order among constants does not matter):

```python
# Default subtitle language preference. Spanish first (owner's primary content),
# then English, then any original-language track. yt-dlp tries each in order and
# writes every match; _pick_subtitle then selects by this same preference.
# `*-orig` catches the uploader's original auto-caption track when localized
# tracks are absent. Never "all" — that pulls YouTube's hundreds of
# auto-translated tracks and stalls the run. Sources: PRs #12, #26, #30.
DEFAULT_SUB_LANGS = "es,es-419,es-ES,en,en-US,en-GB,*-orig"
```

Replace `_pick_subtitle` with a language-ordered version:

```python
def _pick_subtitle(out_dir: Path, sub_langs: str = DEFAULT_SUB_LANGS) -> Path | None:
    candidates = sorted(out_dir.glob("video*.vtt"))
    if not candidates:
        return None
    # Preference follows the requested language order. For "es,en,*-orig" we look
    # for video.es.vtt, then video.en.vtt, then any *-orig track, then anything.
    markers: list[str] = []
    for lang in sub_langs.split(","):
        lang = lang.strip()
        if not lang:
            continue
        if lang.endswith("-orig"):
            markers.append("-orig.")
        else:
            markers.append(f".{lang}.")
    for marker in markers:
        for c in candidates:
            if marker in c.name:
                return c
    return candidates[0]
```

Add `sub_langs: str = DEFAULT_SUB_LANGS` to `fetch_captions` and `download_url` signatures, replace their hardcoded `"--sub-langs", "en.*",` with `"--sub-langs", sub_langs,`, and change their `_pick_subtitle(out_dir)` calls to `_pick_subtitle(out_dir, sub_langs)`. Thread `sub_langs` through `download` to `download_url`:

```python
def download(
    source: str,
    out_dir: Path,
    audio_only: bool = False,
    sub_langs: str = DEFAULT_SUB_LANGS,
    cookies_from_browser: str | None = None,
    cookies_file: str | None = None,
) -> dict:
    if is_url(source):
        return download_url(
            source,
            out_dir,
            audio_only=audio_only,
            sub_langs=sub_langs,
            cookies_from_browser=cookies_from_browser,
            cookies_file=cookies_file,
        )
    return resolve_local(source)
```

(Give `download_url` the `sub_langs` param too, positioned after `audio_only` and before the cookie params, matching the call above.)

- [ ] **Step 4: Wire `--sub-lang` in watch.py**

In `watch.py`, change the download import (line 19) to include the default:

```python
from download import DEFAULT_SUB_LANGS, download, fetch_captions, is_url  # noqa: E402
```

Add the argument after the `--cookies` block:

```python
    ap.add_argument(
        "--sub-lang",
        type=str,
        default=None,
        help="Comma-separated subtitle language preference for yt-dlp "
        f"(default: {DEFAULT_SUB_LANGS}). yt-dlp fetches every match; the report "
        "uses the first available in this order. Never pass 'all'.",
    )
```

Thread `sub_langs=args.sub_lang or DEFAULT_SUB_LANGS` into the `fetch_captions(...)` call and the URL `download(...)` call in `main()` (both edited in Task 3). For example the `fetch_captions` call becomes:

```python
        dl = fetch_captions(
            args.source,
            work / "download",
            sub_langs=args.sub_lang or DEFAULT_SUB_LANGS,
            cookies_from_browser=args.cookies_from_browser,
            cookies_file=args.cookies,
        )
```

and the URL `download(...)` call gains the same `sub_langs=args.sub_lang or DEFAULT_SUB_LANGS,` argument.

- [ ] **Step 5: Run tests**

Run: `python3 -m pytest tests/test_download.py -v`
Expected: PASS — all download tests green, including the rewritten guard.

- [ ] **Step 6: Commit**

```bash
git add skills/watch/scripts/download.py skills/watch/scripts/watch.py tests/test_download.py
git commit -m "feat(download): Spanish-first sub-langs default + --sub-lang override, language-ordered pick"
```

---

### Task 5: Content-addressed Whisper chunk cache (Feature 3, part 1)

**Files:**
- Modify: `skills/watch/scripts/whisper.py` (cache helpers + `use_cache` threaded through `_transcribe_file` / `transcribe_video`)
- Test: `tests/test_transcript_cache.py` (new)

**Interfaces:**
- Produces: `whisper.TRANSCRIPT_CACHE_DIR: Path`; `whisper._chunk_cache_key(audio_path: Path, model: str) -> str`; `whisper._cache_load(key) -> list[dict] | None`; `whisper._cache_save(key, segments, model) -> None`. `_transcribe_file(backend, api_key, audio_path, use_cache=True)` and `transcribe_video(..., use_cache=True)` gain the flag.

- [ ] **Step 1: Write the failing test**

Create `tests/test_transcript_cache.py`:

```python
"""Content-addressed Whisper chunk cache: hit skips the upload, key tracks model."""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import whisper  # noqa: E402


def _fake_audio(tmp_path: Path, data: bytes = b"AUDIODATA") -> Path:
    p = tmp_path / "chunk_000.mp3"
    p.write_bytes(data)
    return p


def test_key_changes_with_model(tmp_path):
    audio = _fake_audio(tmp_path)
    k1 = whisper._chunk_cache_key(audio, "whisper-large-v3")
    k2 = whisper._chunk_cache_key(audio, "whisper-1")
    assert k1 != k2


def test_key_changes_with_bytes(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    ka = whisper._chunk_cache_key(_fake_audio(tmp_path / "a", b"AAA"), "m")
    kb = whisper._chunk_cache_key(_fake_audio(tmp_path / "b", b"BBB"), "m")
    assert ka != kb


def test_cache_miss_then_hit_skips_upload(tmp_path, monkeypatch):
    monkeypatch.setattr(whisper, "TRANSCRIPT_CACHE_DIR", tmp_path / "cache")
    audio = _fake_audio(tmp_path)

    calls = {"n": 0}
    segs = [{"start": 0.0, "end": 1.0, "text": "hello"}]

    def fake_post(endpoint, api_key, model, audio_path):
        calls["n"] += 1
        return {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}]}

    monkeypatch.setattr(whisper, "_post_whisper", fake_post)

    first = whisper._transcribe_file("groq", "key", audio, use_cache=True)
    assert first == segs
    assert calls["n"] == 1  # miss → uploaded

    second = whisper._transcribe_file("groq", "key", audio, use_cache=True)
    assert second == segs
    assert calls["n"] == 1  # hit → no second upload


def test_use_cache_false_always_uploads(tmp_path, monkeypatch):
    monkeypatch.setattr(whisper, "TRANSCRIPT_CACHE_DIR", tmp_path / "cache")
    audio = _fake_audio(tmp_path)
    calls = {"n": 0}

    def fake_post(endpoint, api_key, model, audio_path):
        calls["n"] += 1
        return {"segments": [{"start": 0.0, "end": 1.0, "text": "x"}]}

    monkeypatch.setattr(whisper, "_post_whisper", fake_post)
    whisper._transcribe_file("groq", "key", audio, use_cache=False)
    whisper._transcribe_file("groq", "key", audio, use_cache=False)
    assert calls["n"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_transcript_cache.py -v`
Expected: FAIL — `whisper` has no attribute `_chunk_cache_key` / `_transcribe_file` got unexpected kwarg `use_cache`.

- [ ] **Step 3: Add cache helpers**

In `whisper.py`, add `import hashlib` to the import block (top of file), then add near the other module constants (after `MAX_UPLOAD_BYTES`):

```python
# Content-addressed chunk-transcript cache. Keyed by sha256(chunk bytes + model),
# so a re-run (crash resume, repeated question) reuses already-uploaded chunks
# instead of burning the hourly Whisper quota again. Best-effort: any cache I/O
# failure is swallowed and the chunk simply re-uploads. Source: PR #10.
TRANSCRIPT_CACHE_DIR = Path.home() / ".cache" / "watch" / "transcripts"


def _chunk_cache_key(audio_path: Path, model: str) -> str:
    h = hashlib.sha256()
    h.update(audio_path.read_bytes())
    h.update(b"\x00")
    h.update(model.encode("utf-8"))
    return h.hexdigest()


def _cache_load(key: str) -> list[dict] | None:
    path = TRANSCRIPT_CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        blob = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return None
    segments = blob.get("segments")
    return segments if isinstance(segments, list) else None


def _cache_save(key: str, segments: list[dict], model: str) -> None:
    try:
        TRANSCRIPT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (TRANSCRIPT_CACHE_DIR / f"{key}.json").write_text(
            json.dumps({"model": model, "segments": segments}), encoding="utf-8"
        )
    except OSError:
        pass  # cache is best-effort; never fail transcription over it
```

- [ ] **Step 4: Wire cache into `_transcribe_file`**

Replace `_transcribe_file` with:

```python
def _transcribe_file(
    backend: str, api_key: str, audio_path: Path, use_cache: bool = True
) -> list[dict]:
    """Upload one audio file and return its 0-based segments, with an optional cache."""
    if backend == "groq":
        endpoint, model = GROQ_ENDPOINT, GROQ_MODEL
    elif backend == "openai":
        endpoint, model = OPENAI_ENDPOINT, OPENAI_MODEL
    else:
        raise SystemExit(f"Unknown whisper backend: {backend}")

    key = _chunk_cache_key(audio_path, model) if use_cache else None
    if key is not None:
        cached = _cache_load(key)
        if cached is not None:
            print(f"[watch] cache hit for {audio_path.name} ({model})", file=sys.stderr)
            return cached

    response = _post_whisper(endpoint, api_key, model, audio_path)
    segments = _segments_from_response(response)
    if key is not None:
        _cache_save(key, segments, model)
    return segments
```

- [ ] **Step 5: Thread `use_cache` through `transcribe_video`**

In `transcribe_video`, add `use_cache: bool = True` as the last keyword parameter of the signature, and change the `transcribe_one` closure (near line 573) to forward it:

```python
    def transcribe_one(path: Path) -> list[dict]:
        return _transcribe_file(backend, api_key, path, use_cache=use_cache)
```

- [ ] **Step 6: Run tests**

Run: `python3 -m pytest tests/test_transcript_cache.py tests/test_whisper.py -v`
Expected: PASS (new cache tests + existing whisper tests green).

- [ ] **Step 7: Commit**

```bash
git add skills/watch/scripts/whisper.py tests/test_transcript_cache.py
git commit -m "feat(whisper): content-addressed chunk-transcript cache (sha256 of bytes+model)"
```

---

### Task 6: Window-only audio extraction for `--start/--end` (Feature 3, part 2)

**Files:**
- Modify: `skills/watch/scripts/whisper.py` (`extract_audio` window params; `transcribe_video` range params + final key-preserving shift)
- Modify: `skills/watch/scripts/watch.py` (pass range into `transcribe_video`)
- Test: `tests/test_whisper_window.py` (new)

**Interfaces:**
- Consumes: `use_cache` plumbing (Task 5).
- Produces: `extract_audio(video_path, out_path, start_seconds=None, duration_seconds=None)`; `whisper._shift_all(segments, offset) -> list[dict]` (preserves all keys, incl. `speaker`); `transcribe_video(..., start_seconds=None, end_seconds=None, use_cache=True)` returning source-absolute segments.

- [ ] **Step 1: Write the failing test**

Create `tests/test_whisper_window.py`:

```python
"""Windowed transcription: extract only [start,end], then shift to source time."""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import whisper  # noqa: E402


def test_extract_audio_passes_ss_and_t(monkeypatch, tmp_path):
    captured = {}

    class _Result:
        returncode = 0
        stderr = ""

    def fake_run(cmd, *a, **k):
        captured["cmd"] = list(cmd)
        # Materialize a non-empty output so extract_audio's post-checks pass.
        out = Path(cmd[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"MP3")
        return _Result()

    monkeypatch.setattr(whisper.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(whisper.subprocess, "run", fake_run)

    whisper.extract_audio("in.mp4", tmp_path / "a.mp3", start_seconds=30.0, duration_seconds=15.0)
    cmd = captured["cmd"]
    assert "-ss" in cmd and cmd[cmd.index("-ss") + 1] == "30.000"
    assert "-t" in cmd and cmd[cmd.index("-t") + 1] == "15.000"
    # -ss must precede -i for fast input seek → 0-based output timestamps.
    assert cmd.index("-ss") < cmd.index("-i")


def test_shift_all_preserves_speaker(tmp_path):
    segs = [{"start": 1.0, "end": 2.0, "text": "hi", "speaker": "Speaker A"}]
    out = whisper._shift_all(segs, 30.0)
    assert out == [{"start": 31.0, "end": 32.0, "text": "hi", "speaker": "Speaker A"}]


def test_shift_all_noop_on_zero(tmp_path):
    segs = [{"start": 1.0, "end": 2.0, "text": "hi"}]
    assert whisper._shift_all(segs, 0.0) is segs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_whisper_window.py -v`
Expected: FAIL — `extract_audio()` got an unexpected keyword argument `start_seconds` / no `_shift_all`.

- [ ] **Step 3: Add window params to `extract_audio`**

Replace the `extract_audio` body's `cmd` construction so seek/duration are optional (keep the ffmpeg/existence guards unchanged):

```python
def extract_audio(
    video_path: str,
    out_path: Path,
    start_seconds: float | None = None,
    duration_seconds: float | None = None,
) -> Path:
    """Extract mono 16kHz 64kbps mp3 — ~480 kB/min, fits any Whisper limit.

    With start_seconds/duration_seconds, extract only that window (fast input
    seek before -i, so the output's timestamps start at 0).
    """
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not installed. Install with: brew install ffmpeg")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if start_seconds is not None and start_seconds > 0:
        cmd += ["-ss", f"{start_seconds:.3f}"]
    cmd += ["-i", str(Path(video_path).resolve())]
    if duration_seconds is not None and duration_seconds > 0:
        cmd += ["-t", f"{duration_seconds:.3f}"]
    cmd += [
        "-vn",
        "-acodec", "libmp3lame",
        "-ar", "16000",
        "-ac", "1",
        "-b:a", "64k",
        str(out_path.resolve()),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg audio extraction failed: {result.stderr.strip()}")
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise SystemExit("ffmpeg produced no audio — video may have no audio track")
    return out_path
```

- [ ] **Step 4: Add `_shift_all` and window logic to `transcribe_video`**

Add near `shift_segments` (which stays as-is for chunk offsets):

```python
def _shift_all(segments: list[dict], offset_seconds: float) -> list[dict]:
    """Shift start/end by offset, preserving every other key (e.g. `speaker`).

    Used to lift window-relative Whisper timestamps back to source-absolute time
    after windowed extraction. Unlike shift_segments, it keeps diarization tags.
    """
    if not offset_seconds:
        return segments
    out: list[dict] = []
    for seg in segments:
        moved = dict(seg)
        moved["start"] = round(seg["start"] + offset_seconds, 2)
        moved["end"] = round(seg["end"] + offset_seconds, 2)
        out.append(moved)
    return out
```

Change `transcribe_video`'s signature to add `start_seconds` / `end_seconds` (before `use_cache`):

```python
def transcribe_video(
    video_path: str,
    audio_out: Path,
    backend: str | None = None,
    api_key: str | None = None,
    model_name: str | None = None,
    enable_diarization: bool = False,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    use_cache: bool = True,
) -> tuple[list[dict], str]:
```

Compute the window and pass it to `extract_audio`. Replace the `audio_path = extract_audio(video_path, audio_out)` line (near line 534) with:

```python
    window_offset = start_seconds or 0.0
    duration_seconds = None
    if start_seconds is not None and end_seconds is not None and end_seconds > start_seconds:
        duration_seconds = end_seconds - start_seconds
    audio_path = extract_audio(
        video_path, audio_out, start_seconds=start_seconds, duration_seconds=duration_seconds
    )
```

The extracted audio is window-relative, so every backend below returns window-relative segments. Lift them to source time at each return. There are three `return ..., <backend>` points in `transcribe_video` (local, assemblyai, and the final groq/openai). Wrap each returned `segments` with `_shift_all(segments, window_offset)`:

- local branch:
  ```python
      segments = _shift_all(segments, window_offset)
      print(f"[watch] transcribed {len(segments)} segments via local", file=sys.stderr)
      return segments, "local"
  ```
- assemblyai branch:
  ```python
      segments = _shift_all(segments, window_offset)
      print(f"[watch] transcribed {len(segments)} segments via assemblyai", file=sys.stderr)
      return segments, "assemblyai"
  ```
- final groq/openai path — after the `if not segments: raise ...` guard and before the final `return segments, backend`:
  ```python
      segments = _shift_all(segments, window_offset)
      print(f"[watch] transcribed {len(segments)} segments via {backend}", file=sys.stderr)
      return segments, backend
  ```

- [ ] **Step 5: Pass the range from watch.py**

In `watch.py`'s `transcribe_video(...)` call (near line 552), add the window and cache args. Only window the **video's own** track — an external `--audio` file is not aligned to video timestamps, so pass `None` there:

```python
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
```

`args.fresh` is added in Task 9; if executing Task 6 before Task 9, temporarily use `use_cache=True` and revert to `not args.fresh` in Task 9. The subsequent `filter_range(all_segments, start_sec, end_sec)` call stays — it is now idempotent (segments are already within the window) and still trims the `--audio` path, which was not windowed.

- [ ] **Step 6: Run tests**

Run: `python3 -m pytest tests/test_whisper_window.py tests/test_whisper.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add skills/watch/scripts/whisper.py skills/watch/scripts/watch.py tests/test_whisper_window.py
git commit -m "feat(whisper): window-only audio extraction for --start/--end (no full-track Whisper spend)"
```

---

### Task 7: Lock chunk-level retry cap (regression guard) (Feature 3, part 3)

**Files:**
- Test: `tests/test_whisper_retry.py` (new) — no production change; this locks the spec's "no infinite 5xx retry" guarantee that `_post_whisper`'s `MAX_ATTEMPTS` already provides.

**Interfaces:**
- Consumes: `whisper._post_whisper`, `whisper.MAX_ATTEMPTS`.

- [ ] **Step 1: Write the failing-if-regressed test**

Create `tests/test_whisper_retry.py`:

```python
"""Regression lock: a persistent 5xx stops after MAX_ATTEMPTS — never loops forever."""
from __future__ import annotations

import sys
import urllib.error
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import whisper  # noqa: E402


def test_persistent_500_is_bounded(monkeypatch, tmp_path):
    audio = tmp_path / "chunk.mp3"
    audio.write_bytes(b"MP3")

    attempts = {"n": 0}

    def always_500(request, timeout=None, context=None):
        attempts["n"] += 1
        raise urllib.error.HTTPError(
            url="http://x", code=500, msg="err", hdrs=None, fp=None
        )

    monkeypatch.setattr(whisper, "urlopen", always_500)
    monkeypatch.setattr(whisper.time, "sleep", lambda *_a, **_k: None)  # no real backoff

    with pytest.raises(SystemExit):
        whisper._post_whisper("http://x", "key", "model", audio)

    # Bounded: exactly MAX_ATTEMPTS uploads, then it gives up. No infinite loop.
    assert attempts["n"] == whisper.MAX_ATTEMPTS
```

- [ ] **Step 2: Run the test**

Run: `python3 -m pytest tests/test_whisper_retry.py -v`
Expected: PASS immediately (the cap already exists). If it FAILS or hangs, the retry cap has regressed — fix `_post_whisper` before proceeding.

- [ ] **Step 3: Commit**

```bash
git add tests/test_whisper_retry.py
git commit -m "test(whisper): lock bounded 5xx retry (no infinite quota burn)"
```

---

### Task 8: `state.py` — params signature + stage load/save (Feature 4, part 1)

**Files:**
- Create: `skills/watch/scripts/state.py`
- Test: `tests/test_state.py` (new)

**Interfaces:**
- Produces:
  - `state.params_signature(source: str, params: dict) -> str` — stable 16-hex signature over source + params.
  - `state.save_stage(work: Path, name: str, data, sig: str) -> None` — writes `work/stage_<name>.json`.
  - `state.load_stage(work: Path, name: str, sig: str) -> object | None` — returns saved `data` only if the file exists and its stored sig matches; else `None`.
  - `state.clear_stages(work: Path) -> None` — deletes every `work/stage_*.json`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_state.py`:

```python
"""Resume state: params signature stability + stage roundtrip + sig gating."""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import state  # noqa: E402


def test_signature_is_stable_and_order_independent():
    a = state.params_signature("url", {"detail": "balanced", "start": "0:30"})
    b = state.params_signature("url", {"start": "0:30", "detail": "balanced"})
    assert a == b
    assert len(a) == 16


def test_signature_changes_with_params():
    a = state.params_signature("url", {"detail": "balanced"})
    b = state.params_signature("url", {"detail": "efficient"})
    c = state.params_signature("url2", {"detail": "balanced"})
    assert a != b and a != c


def test_stage_roundtrip(tmp_path):
    sig = state.params_signature("url", {"detail": "balanced"})
    payload = {"video_path": "/x/video.mp4", "downloaded": True}
    state.save_stage(tmp_path, "download", payload, sig)
    loaded = state.load_stage(tmp_path, "download", sig)
    assert loaded == payload


def test_load_returns_none_on_sig_mismatch(tmp_path):
    state.save_stage(tmp_path, "download", {"a": 1}, "sigA")
    assert state.load_stage(tmp_path, "download", "sigB") is None


def test_load_returns_none_when_absent(tmp_path):
    assert state.load_stage(tmp_path, "download", "sig") is None


def test_load_tolerates_corrupt_file(tmp_path):
    (tmp_path / "stage_download.json").write_text("{not json", encoding="utf-8")
    assert state.load_stage(tmp_path, "download", "sig") is None


def test_clear_stages_removes_only_stage_files(tmp_path):
    (tmp_path / "stage_download.json").write_text("{}", encoding="utf-8")
    (tmp_path / "stage_transcript.json").write_text("{}", encoding="utf-8")
    (tmp_path / "video.info.json").write_text("{}", encoding="utf-8")
    state.clear_stages(tmp_path)
    assert not (tmp_path / "stage_download.json").exists()
    assert not (tmp_path / "stage_transcript.json").exists()
    assert (tmp_path / "video.info.json").exists()  # non-stage file untouched
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_state.py -v`
Expected: FAIL — no module named `state`.

- [ ] **Step 3: Implement `state.py`**

Create `skills/watch/scripts/state.py`:

```python
#!/usr/bin/env python3
"""Per-stage resume state for /watch.

A run with a stable work dir (--out-dir) persists each expensive stage's result
as `stage_<name>.json` alongside a signature of the source + output-affecting
params. A later run with the same signature reloads finished stages and skips
them; a changed signature (or --fresh) ignores the cache. Late-stage crashes
therefore never destroy earlier network work. Source: danielfrey63 fork.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

for _stream in (__import__("sys").stdout, __import__("sys").stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


def params_signature(source: str, params: dict) -> str:
    """Stable 16-hex signature over source + output-affecting params.

    Key order does not matter (sort_keys). Values are coerced to str so None,
    ints, and strings all serialize deterministically.
    """
    normalized = {k: ("" if v is None else str(v)) for k, v in params.items()}
    payload = json.dumps({"source": source, "params": normalized}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _stage_path(work: Path, name: str) -> Path:
    return work / f"stage_{name}.json"


def save_stage(work: Path, name: str, data, sig: str) -> None:
    """Persist a stage result. Best-effort — never raises on I/O failure."""
    try:
        work.mkdir(parents=True, exist_ok=True)
        _stage_path(work, name).write_text(
            json.dumps({"sig": sig, "data": data}), encoding="utf-8"
        )
    except (OSError, TypeError):
        pass  # unserializable / unwritable → just don't cache this stage


def load_stage(work: Path, name: str, sig: str):
    """Return the saved `data` iff the stage file exists and its sig matches."""
    path = _stage_path(work, name)
    if not path.exists():
        return None
    try:
        blob = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(blob, dict) or blob.get("sig") != sig:
        return None
    return blob.get("data")


def clear_stages(work: Path) -> None:
    """Delete every stage_*.json (used by --fresh). Leaves other files alone."""
    try:
        for f in work.glob("stage_*.json"):
            f.unlink()
    except OSError:
        pass
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_state.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add skills/watch/scripts/state.py tests/test_state.py
git commit -m "feat(state): params-signature stage load/save for crash resume"
```

---

### Task 9: Wire download + transcript resume into watch.py + `--fresh` (Feature 4, part 2)

**Files:**
- Modify: `skills/watch/scripts/watch.py` (import `state`; `--fresh` arg; resume the download + transcript stages; save each on completion)
- Test: `tests/test_resume.py` (new — integration on a silent ffmpeg clip)

**Interfaces:**
- Consumes: `state.params_signature`, `state.load_stage`, `state.save_stage`, `state.clear_stages` (Task 8).
- Produces: resume behavior when `--out-dir` is set and `--fresh` is not; `args.fresh` boolean consumed by Task 6's `use_cache=not args.fresh`.

**Scope note:** resume covers the two network-heavy stages — **download** (skip re-download when the file is still on disk) and **transcript** (skip re-transcription). Frame extraction is local, deterministic, and cheap, so `frames.done` from the spec is **deferred** (see Deferred section) — the chunk cache (Task 5) plus transcript-stage save already satisfy the spec's acceptance test (unplug network mid-transcription → re-run completes without re-uploading finished chunks).

- [ ] **Step 1: Write the failing test**

Create `tests/test_resume.py`:

```python
"""Resume: a second run with the same --out-dir reuses the downloaded video."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import watch  # noqa: E402


def _silent_clip(path: Path) -> None:
    # A short silent clip: no audio track (whisper skipped), a few keyframes.
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-t", "1.5", "-i", "color=c=blue:s=160x120:r=10",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )


def test_second_run_skips_download(tmp_path, monkeypatch, capsys):
    clip = tmp_path / "clip.mp4"
    _silent_clip(clip)
    work = tmp_path / "work"

    calls = {"n": 0}
    real_download = watch.download

    def counting_download(*args, **kwargs):
        calls["n"] += 1
        return real_download(*args, **kwargs)

    monkeypatch.setattr(watch, "download", counting_download)

    argv = [str(clip), "--out-dir", str(work), "--detail", "efficient", "--no-ocr"]
    monkeypatch.setattr(sys, "argv", ["watch.py", *argv])
    assert watch.main() == 0
    monkeypatch.setattr(sys, "argv", ["watch.py", *argv])
    assert watch.main() == 0

    # Local resolve counts as a "download" call; the second run must reuse the
    # saved stage (video file still present) rather than resolving again.
    assert calls["n"] == 1


def test_fresh_forces_redownload(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    _silent_clip(clip)
    work = tmp_path / "work"

    calls = {"n": 0}
    real_download = watch.download

    def counting_download(*args, **kwargs):
        calls["n"] += 1
        return real_download(*args, **kwargs)

    monkeypatch.setattr(watch, "download", counting_download)

    base = [str(clip), "--out-dir", str(work), "--detail", "efficient", "--no-ocr"]
    monkeypatch.setattr(sys, "argv", ["watch.py", *base])
    assert watch.main() == 0
    monkeypatch.setattr(sys, "argv", ["watch.py", *base, "--fresh"])
    assert watch.main() == 0
    assert calls["n"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_resume.py -v`
Expected: FAIL — `--fresh` unrecognized / download called twice (no resume yet).

- [ ] **Step 3: Import `state` and add `--fresh`**

In `watch.py`, add to the imports block (after the `from download import ...` line):

```python
import state  # noqa: E402
```

Add the argument after the `--sub-lang` block:

```python
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore any resume state in --out-dir and re-run every stage from "
        "scratch (also bypasses the Whisper chunk cache for this run).",
    )
```

- [ ] **Step 4: Compute the signature and handle `--fresh`**

Right after `work.mkdir(parents=True, exist_ok=True)` and the working-dir print (near line 423), insert:

```python
    # Resume is only meaningful with a stable work dir. With the default tmp dir
    # each run gets a fresh path, so load_stage always misses (harmless no-op).
    resume = bool(args.out_dir) and not args.fresh
    resume_sig = state.params_signature(
        args.source,
        {
            "detail": args.detail,
            "start": args.start,
            "end": args.end,
            "audio": args.audio,
            "sub_lang": args.sub_lang,
            "whisper": args.whisper,
            "resolution": args.resolution,
        },
    )
    if args.out_dir and args.fresh:
        state.clear_stages(work)
        print("[watch] --fresh: cleared resume state", file=sys.stderr)
```

Place this **before** the `if args.backend == "gemini"` dispatch so the gemini/openrouter backends (which return early) are unaffected — they simply never read `resume`.

- [ ] **Step 5: Resume the download stage**

Find the download-for-frames block (near lines 469–484). Wrap the URL/local `download(...)` calls so a cached stage with a still-present video file is reused. Replace:

```python
    else:
        if url_source:
            print(...)
            dl = download(args.source, work / "download", audio_only=audio_only, sub_langs=..., cookies_from_browser=..., cookies_file=...)
        else:
            print("[watch] using local file…", file=sys.stderr)
            dl = download(args.source, work / "download")
        video_path = dl["video_path"]
```

with (keep the exact `download(...)` argument lists you have after Tasks 3–4):

```python
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
                dl = download(
                    args.source,
                    work / "download",
                    audio_only=audio_only,
                    sub_langs=args.sub_lang or DEFAULT_SUB_LANGS,
                    cookies_from_browser=args.cookies_from_browser,
                    cookies_file=args.cookies,
                )
            else:
                print("[watch] using local file…", file=sys.stderr)
                dl = download(args.source, work / "download")
            if args.out_dir:
                state.save_stage(work, "download", dl, resume_sig)
        video_path = dl["video_path"]
```

- [ ] **Step 6: Resume the transcript stage**

The transcript is resolved across the caption block and the Whisper-fallback block (lines ~522–588). Add a load at the top of that region and a save after it settles.

Immediately **before** the caption re-parse block that begins `if not transcript_segments and dl.get("subtitle_path")...` (near line 527), insert:

```python
    cached_tx = state.load_stage(work, "transcript", resume_sig) if resume else None
    if cached_tx and cached_tx.get("segments"):
        transcript_segments = cached_tx["segments"]
        transcript_source = cached_tx.get("source") or "captions"
        transcript_text = format_transcript(transcript_segments)
        print(
            f"[watch] resume: reusing transcript ({len(transcript_segments)} segments)",
            file=sys.stderr,
        )
```

Then guard the caption re-parse and the whole Whisper-fallback block so they only run on a cache miss — change their leading conditions from `if not transcript_segments and ...` to also require no cached transcript. The simplest robust form: wrap both blocks in `if not transcript_segments:` (they already start with `if not transcript_segments and ...`, so the reuse above short-circuits them naturally — no further edit needed, because setting `transcript_segments` from cache makes those `if not transcript_segments` guards false).

Finally, **after** the Whisper-fallback block and its `elif not transcript_segments ...` (near line 588, right before the `scope = (...)` assignment), save the resolved transcript:

```python
    if args.out_dir and transcript_segments:
        state.save_stage(
            work,
            "transcript",
            {"segments": transcript_segments, "source": transcript_source},
            resume_sig,
        )
```

- [ ] **Step 7: Run tests**

Run: `python3 -m pytest tests/test_resume.py -v`
Expected: PASS — `test_second_run_skips_download` sees exactly 1 download call; `test_fresh_forces_redownload` sees 2.

- [ ] **Step 8: Run the full suite to catch regressions in main()**

Run: `python3 -m pytest -q`
Expected: all green.

- [ ] **Step 9: Commit**

```bash
git add skills/watch/scripts/watch.py tests/test_resume.py
git commit -m "feat(watch): resume download + transcript stages via --out-dir state; --fresh forces full re-run"
```

---

### Task 10: Version bump 0.4.0 + CHANGELOG + SKILL.md docs

**Files:**
- Modify: `skills/watch/SKILL.md` (frontmatter version + new flag docs)
- Modify: `.claude-plugin/plugin.json`, `.codex-plugin/plugin.json` (version)
- Modify: `CHANGELOG.md` (new `[0.4.0]` section)
- Modify: `.claude-plugin/marketplace.json` (version, only if it carries one)

**Interfaces:** none (docs/metadata only).

- [ ] **Step 1: Bump versions to 0.4.0**

```bash
cd ~/claude-video
sed -i 's/version: "0.3.0"/version: "0.4.0"/' skills/watch/SKILL.md
sed -i 's/"version": "0.3.0"/"version": "0.4.0"/' .claude-plugin/plugin.json .codex-plugin/plugin.json
grep -rl '"version": "0.3.0"' .claude-plugin/marketplace.json .agents/plugins/marketplace.json 2>/dev/null \
  | xargs -r sed -i 's/"version": "0.3.0"/"version": "0.4.0"/'
```

Verify:

```bash
grep -rn '0.4.0' skills/watch/SKILL.md .claude-plugin/plugin.json .codex-plugin/plugin.json
```
Expected: all three read `0.4.0`.

- [ ] **Step 2: Add the CHANGELOG entry**

Insert this block in `CHANGELOG.md` directly above the `## [0.3.0] — 2026-07-06` line:

```markdown
## [0.4.0] — 2026-07-07

### Added
- **YouTube reliability.** yt-dlp now runs a `player_client=default,android,mweb`
  fallback chain with anti-bot retry/jitter by default, so SABR/403 blocks on
  public videos self-heal without cookies. New `--cookies-from-browser <browser>`
  and `--cookies <file>` flags unlock login-walled sources (off by default).
- **Spanish-first subtitles.** Default subtitle preference is now
  `es,es-419,es-ES,en,en-US,en-GB,*-orig`; override with `--sub-lang <csv>`. The
  report picks the first available track in that order. Never requests `all`.
- **Whisper chunk cache.** Transcribed audio chunks are cached at
  `~/.cache/watch/transcripts/` keyed by `sha256(bytes + model)`, so re-runs and
  crash-resumes skip already-uploaded chunks instead of re-spending quota.
- **Windowed transcription.** With `--start/--end`, only that audio window is
  extracted and transcribed (not the full track), then stitched back to source
  time — zero Whisper spend outside the requested range.
- **Crash resume.** With `--out-dir`, the download and transcript stages persist
  to `stage_*.json` and a matching re-run resumes past them. `--fresh` forces a
  full re-run and bypasses the chunk cache.

### Changed
- Chunk-level Whisper retries remain capped (no infinite 5xx quota burn); now
  covered by a regression test.
```

- [ ] **Step 3: Document the new flags in SKILL.md**

In the "How to invoke" flag list (near lines 132–180, where `--out-dir`, `--whisper`, etc. are documented), add bullets:

```markdown
- `--sub-lang CSV` — subtitle language preference for yt-dlp (default `es,es-419,es-ES,en,en-US,en-GB,*-orig`). The report uses the first available track in this order. Never pass `all`.
- `--cookies-from-browser BROWSER` — load yt-dlp cookies from a browser profile (`chrome|firefox|edge|brave|safari|...`) for login-walled or age-gated sources. Off by default.
- `--cookies FILE` — path to a Netscape `cookies.txt` for yt-dlp. Off by default.
- `--fresh` — with `--out-dir`, ignore any saved resume state and re-run every stage from scratch (also bypasses the Whisper chunk cache).
```

And add a short section after the "Transcription" area explaining resume:

```markdown
## Resume and caching

Whisper chunk transcripts are cached at `~/.cache/watch/transcripts/`
(`sha256(audio bytes + model)`), so re-asking about the same video — or resuming
after a crash — never re-uploads chunks it already transcribed. When you pass
`--out-dir DIR`, the download and transcript stages also persist there; a re-run
with the same source and options resumes past them. Pass `--fresh` to ignore all
of this and start clean. With `--start/--end`, only that time window's audio is
transcribed, so focusing on a section costs nothing outside it.
```

- [ ] **Step 4: Commit**

```bash
git add skills/watch/SKILL.md .claude-plugin/plugin.json .codex-plugin/plugin.json CHANGELOG.md .claude-plugin/marketplace.json .agents/plugins/marketplace.json 2>/dev/null
git commit -m "docs: v0.4.0 — reliability pack changelog, flag docs, version bump"
```

---

### Task 11: Full suite + smoke sanity

**Files:** none (verification only).

- [ ] **Step 1: Run the full test suite from repo root**

Run: `python3 -m pytest -q`
Expected: all tests pass, including the rewritten `test_download.py` guard and the new `test_transcript_cache.py`, `test_whisper_window.py`, `test_whisper_retry.py`, `test_state.py`, `test_resume.py`.

- [ ] **Step 2: Confirm every entrypoint still imports**

```bash
cd ~/claude-video
python3 -c "import sys; sys.path.insert(0,'skills/watch/scripts'); import watch, download, whisper, state, transcribe; print('imports OK')"
```
Expected: `imports OK`.

- [ ] **Step 3: Confirm the smoke workflow references still resolve**

```bash
grep -rn 'skills/watch' .github/workflows/*.yml
```
Expected: paths point at `skills/watch/...` (unchanged by this phase). No edit needed unless a path is stale — if so, fix and note it.

- [ ] **Step 4: Commit (only if Step 3 required a fix)**

```bash
git add .github/workflows/
git commit -m "ci: repath smoke workflow for v0.4.0 (if needed)"
```

---

### Task 12: E2E verification + user checkpoint (NO deploy without approval)

**Files:** none (manual verification on the box). **Per CLAUDE.md: evidence before assertions; no deploy until the user approves.**

- [ ] **Step 1: Spanish-caption E2E (Feature 2) — zero Whisper spend**

Run against a Spanish-only YouTube video (owner supplies a URL), from `~/claude-video`:

```bash
python3 skills/watch/scripts/watch.py "<spanish-youtube-url>" --detail transcript --out-dir /tmp/watch-es
```
Expected: report `Transcript: … (via captions)` with Spanish text; **no** `[watch] extracting audio for Whisper` line (captions won, Whisper never ran).

- [ ] **Step 2: Windowed transcription E2E (Feature 3)**

```bash
python3 skills/watch/scripts/watch.py "<caption-less-url-or-local-file>" --start 0:30 --end 0:45 --whisper groq --out-dir /tmp/watch-win
```
Expected: stderr shows the audio extracted is the ~15s window (small kB), transcript timestamps fall in `0:30`–`0:45` (source-absolute), and `~/.cache/watch/transcripts/` gains a JSON.

- [ ] **Step 3: Resume E2E (Feature 4)**

```bash
# First run to completion:
python3 skills/watch/scripts/watch.py "<url>" --detail balanced --out-dir /tmp/watch-resume
ls /tmp/watch-resume/stage_*.json   # expect stage_download.json (+ stage_transcript.json if a transcript resolved)
# Second run — should print resume lines and skip download/transcription:
python3 skills/watch/scripts/watch.py "<url>" --detail balanced --out-dir /tmp/watch-resume
```
Expected: second run prints `[watch] resume: reusing previously downloaded video` and (if applicable) `[watch] resume: reusing transcript …`, and finishes markedly faster. A `--fresh` third run re-does everything.

- [ ] **Step 4: YouTube 403 hardening spot-check (Feature 1)**

```bash
python3 skills/watch/scripts/watch.py "<a-normally-403-prone-youtube-url>" --detail efficient --out-dir /tmp/watch-403
```
Expected: downloads succeed (player_client chain), no fatal 403.

- [ ] **Step 5: STOP — user checkpoint**

Present the E2E evidence to the user. **Do not deploy.** Ask for approval to:
1. Push `feat/v0.4.0-reliability` and merge to `main` (tag `v0.4.0`).
2. Deploy to the two live clones:
   - `~/.claude/skills/watch` — `git pull` (live Claude Code skill).
   - `/opt/watch/claude-video` — `git pull` (bind-mounted `:ro` into the `watch` container; scripts run per-invocation so **no image rebuild and no restart needed** — the next `/watch` job reads the new files. Restart only if the MCP server caches script state).
3. `memory_store` (qdrant-mem) the key Phase 2 findings (Spanish-first default, chunk-cache location, resume/`--fresh` semantics, the `:ro`-but-live bind-mount deploy fact).

Only after explicit approval, execute the chosen deploy steps and verify with a real `watch.cort3x.me` MCP call.

---

## Deferred (explicitly NOT in this phase)

- **`frames.done` frame-stage resume** (spec Feature 4 listed it). Frame extraction is local, deterministic, and cheap; resuming it means re-validating every frame file path and reconstructing `frame_meta` for the report — fragile for little gain. The spec's acceptance test (network unplug mid-transcription → resume without re-uploading) is already met by the chunk cache (Task 5) + transcript-stage save (Task 9). Revisit if a user hits slow frame extraction on very long focused runs.
- **Auto-subs-only fallback signaling.** yt-dlp already fetches both manual and auto tracks (`--write-subs --write-auto-subs`); `_pick_subtitle` selects by language order. A distinct "manual missing, used auto" note in the report is cosmetic and deferred.
- **Upstreaming** subs-language / 403 fixes as PRs (spec open item) — personal fork first.

## Self-Review

- **Spec coverage:** Feature 1 → Tasks 2–3; Feature 2 → Task 4; Feature 3 (retry cap / chunk cache / windowed) → Tasks 7 / 5 / 6; Feature 4 (resume + `--fresh`) → Tasks 8–9. Acceptance criteria: Spanish caption zero-Whisper run → Task 12 Step 1; resume-without-re-upload → chunk cache (5) + transcript save (9), proven in Task 12 Step 3. `frames.done` consciously deferred with rationale above.
- **Type consistency:** `sub_langs: str`, `cookies_from_browser: str | None`, `cookies_file: str | None`, `use_cache: bool`, `start_seconds/end_seconds: float | None` used identically across `download.py`, `whisper.py`, and the `watch.py` call sites. `state.params_signature/save_stage/load_stage/clear_stages` signatures match Task 8 definitions and Task 9 call sites. `_shift_all` (key-preserving) is distinct from the existing `shift_segments` (chunk offsets) — intentional, tested in Task 6.
- **No placeholders:** every code and test step carries complete code; every run step has an exact command and expected result.
