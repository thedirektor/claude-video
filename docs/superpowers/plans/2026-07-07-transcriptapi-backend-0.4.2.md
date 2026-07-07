# TranscriptAPI YouTube backend (v0.4.2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Give `/watch` a TranscriptAPI backend so YouTube URLs get their transcript from transcriptapi.com (whose servers beat YouTube's bot-gate / PO-token / nsig walls) instead of yt-dlp captions, which fail from this datacenter IP — shipping as v0.4.2.

**Architecture:** A new pure-stdlib module `transcriptapi.py` (mirrors the `whisper.py`/`gemini.py` backend pattern) fetches the transcript over TranscriptAPI's REST API. `watch.py`'s transcript resolution gains a new highest-precedence source for YouTube URLs: **TranscriptAPI → yt-dlp captions → Whisper**. It is best-effort — any TranscriptAPI failure (no key, no credits, no transcript, network) logs a reason and falls through to the existing path, so nothing regresses. Transcript-only scope; YouTube *frame* extraction still needs the (gated) video download and is unchanged.

**Tech Stack:** Python 3.11 stdlib (`urllib`, `json`, `ssl`) — no new pip deps. TranscriptAPI REST `https://transcriptapi.com/api/v2`. pytest with mocked HTTP.

## Global Constraints

- **Dev repo:** `~/claude-video`, branch `feat/v0.4.2-transcriptapi` from `main` (currently v0.4.1 @ the deployed head). Do NOT dev in `/opt/watch/claude-video` (the `:ro` container mount).
- Pure stdlib in core paths; **no new pip dependencies**. Follow `whisper.py`'s hand-rolled `urllib` HTTP + `load_api_key` dotenv pattern (env → `~/.config/watch/.env` → `./.env`).
- Windows UTF-8 header block on the new entrypoint (after imports):
  ```python
  for _stream in (sys.stdout, sys.stderr):
      try:
          _stream.reconfigure(encoding="utf-8", errors="replace")
      except (AttributeError, OSError):
          pass
  ```
- All dotenv/config reads use `encoding="utf-8", errors="replace"`.
- **Best-effort backend:** TranscriptAPI failures must NEVER raise out of `fetch_transcript` — return `None` and log a one-line reason to stderr so `watch.py` falls back to captions/Whisper.
- Segment shape across the pipeline is `{"start": float, "end": float, "text": str}` (same as `transcribe.parse_vtt` / `whisper`), so `filter_range`, `format_transcript`, and two-pass all consume it unchanged. TranscriptAPI returns `{text, start, duration}` → map `end = start + duration`.
- Upstream conventions: NO `commands/` dir; `${SKILL_DIR}` in SKILL.md; version synced across `skills/watch/SKILL.md`, `.claude-plugin/plugin.json`, `.codex-plugin/plugin.json`.
- **Version this phase:** `0.4.2`.
- Tests run offline from repo root: `python3 -m pytest -q` (mock `urlopen`; no real network in unit tests). E2E (Task 6) makes one real credited call.
- **Secret:** `TRANSCRIPTAPI_API_KEY` (a `sk_…` bearer). Lives in `~/.config/watch/.env` (host) and `/opt/watch/secrets/.env` (container). NEVER commit it; `.env` is gitignored. Deploy places it (Task 6), not the code.
- NO pushing / NO touching live clones until the user checkpoint in Task 6.

---

### Task 1: Branch setup

**Files:** Create branch `feat/v0.4.2-transcriptapi` from `main`; carry this plan.

- [ ] **Step 1: Confirm base is clean v0.4.1**
```bash
cd ~/claude-video && git checkout main && git status -s
grep '"version"' .claude-plugin/plugin.json   # expect 0.4.1
```
Expected: clean tree, version `0.4.1`. If not, STOP and report.

- [ ] **Step 2: Branch + carry plan**
```bash
cd ~/claude-video
git checkout -b feat/v0.4.2-transcriptapi main
cp /opt/watch/claude-video/docs/superpowers/plans/2026-07-07-transcriptapi-backend-0.4.2.md docs/superpowers/plans/ 2>/dev/null || true
git add docs/superpowers/plans/2026-07-07-transcriptapi-backend-0.4.2.md 2>/dev/null || true
git commit -q -m "docs: carry transcriptapi backend plan onto v0.4.2 branch" || true
```

- [ ] **Step 3: Baseline green**
```bash
python3 -m pytest -q
```
Expected: all pass (134 at v0.4.1).

---

### Task 2: `transcriptapi.py` module

**Files:**
- Create: `skills/watch/scripts/transcriptapi.py`
- Test: `tests/test_transcriptapi.py`

**Interfaces:**
- Produces:
  - `transcriptapi.is_youtube_url(source: str) -> bool`
  - `transcriptapi.load_api_key() -> str | None` (env `TRANSCRIPTAPI_API_KEY` → `~/.config/watch/.env` → `./.env`)
  - `transcriptapi.languages_from_sub_langs(sub_langs: str) -> str` (regex sub-langs → TranscriptAPI priority list)
  - `transcriptapi.fetch_transcript(video_url: str, api_key: str, language: str | None = None) -> tuple[list[dict], str] | None` — `(segments, resolved_language)` on success; `None` on any failure (logged). Segments are `{start,end,text}`. NEVER raises.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_transcriptapi.py`:

```python
"""TranscriptAPI backend: URL detection, language mapping, HTTP mapping, fallback."""
from __future__ import annotations

import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import transcriptapi  # noqa: E402


class _Resp:
    def __init__(self, payload, status=200):
        self._b = json.dumps(payload).encode()
        self.status = status
    def read(self):
        return self._b
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


def test_is_youtube_url():
    assert transcriptapi.is_youtube_url("https://www.youtube.com/watch?v=abc12345678")
    assert transcriptapi.is_youtube_url("https://youtu.be/abc12345678")
    assert transcriptapi.is_youtube_url("http://m.youtube.com/watch?v=x")
    assert not transcriptapi.is_youtube_url("https://vimeo.com/12345")
    assert not transcriptapi.is_youtube_url("/local/file.mp4")
    assert not transcriptapi.is_youtube_url("notayoutubeimpostor.com")


def test_language_mapping_from_default_sub_langs():
    # region-stripped, deduped, drops the .*-orig regex, appends asr fallback
    out = transcriptapi.languages_from_sub_langs("es,es-419,es-ES,en,en-US,en-GB,.*-orig")
    assert out == "es,en,asr"


def test_language_mapping_custom():
    assert transcriptapi.languages_from_sub_langs("de,fr") == "de,fr,asr"
    assert transcriptapi.languages_from_sub_langs("all") == "asr"


def test_fetch_transcript_maps_segments(monkeypatch):
    payload = {"video_id": "x", "language": "es",
               "transcript": [{"text": "hola", "start": 0.0, "duration": 2.0},
                              {"text": "  ", "start": 2.0, "duration": 1.0},
                              {"text": "mundo", "start": 2.0, "duration": 1.5}]}
    monkeypatch.setattr(transcriptapi, "urlopen", lambda *a, **k: _Resp(payload))
    result = transcriptapi.fetch_transcript("vid", "sk_key", language="es,en,asr")
    assert result is not None
    segs, lang = result
    assert lang == "es"
    # empty-text segment dropped; end = start + duration
    assert segs == [{"start": 0.0, "end": 2.0, "text": "hola"},
                    {"start": 2.0, "end": 3.5, "text": "mundo"}]


def test_fetch_transcript_404_returns_none(monkeypatch, capsys):
    def raise404(*a, **k):
        raise urllib.error.HTTPError("u", 404, "no", None,
                                     io.BytesIO(b'{"detail":"none"}'))
    monkeypatch.setattr(transcriptapi, "urlopen", raise404)
    assert transcriptapi.fetch_transcript("vid", "sk_key") is None  # → caller falls back


def test_fetch_transcript_402_returns_none_with_warning(monkeypatch, capsys):
    def raise402(*a, **k):
        raise urllib.error.HTTPError("u", 402, "pay", None,
                                     io.BytesIO(b'{"detail":"no credits"}'))
    monkeypatch.setattr(transcriptapi, "urlopen", raise402)
    assert transcriptapi.fetch_transcript("vid", "sk_key") is None
    assert "credit" in capsys.readouterr().err.lower()


def test_fetch_transcript_network_error_returns_none(monkeypatch):
    def boom(*a, **k):
        raise urllib.error.URLError("dns")
    monkeypatch.setattr(transcriptapi, "urlopen", boom)
    monkeypatch.setattr(transcriptapi.time, "sleep", lambda *_a, **_k: None)
    assert transcriptapi.fetch_transcript("vid", "sk_key") is None  # never raises
```

- [ ] **Step 2: Run — verify fail**

Run: `python3 -m pytest tests/test_transcriptapi.py -v`
Expected: FAIL — no module `transcriptapi`.

- [ ] **Step 3: Implement `transcriptapi.py`**

Create `skills/watch/scripts/transcriptapi.py`:

```python
#!/usr/bin/env python3
"""Fetch a YouTube transcript from TranscriptAPI (https://transcriptapi.com).

TranscriptAPI's servers handle YouTube's bot-gate / PO-token / nsig challenges,
so this works from datacenter IPs where yt-dlp caption fetch is blocked. Pure
stdlib HTTP; best-effort (any failure returns None so the caller falls back to
yt-dlp captions / Whisper).
"""
from __future__ import annotations

import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
from pathlib import Path
from urllib.request import Request, urlopen

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

API_BASE = "https://transcriptapi.com/api/v2"
MAX_ATTEMPTS = 3
RETRYABLE = {408, 429, 500, 503}


def is_youtube_url(source: str) -> bool:
    """True only for real YouTube hosts (not 'notyoutube.com' impostors)."""
    try:
        p = urllib.parse.urlparse(source)
    except ValueError:
        return False
    if p.scheme not in ("http", "https"):
        return False
    host = (p.netloc or "").lower().split(":")[0]
    return (
        host == "youtu.be"
        or host.endswith(".youtu.be")
        or host == "youtube.com"
        or host.endswith(".youtube.com")
    )


def load_api_key() -> str | None:
    """TRANSCRIPTAPI_API_KEY from env, then ~/.config/watch/.env, then ./.env."""
    def _from_env() -> str | None:
        v = os.environ.get("TRANSCRIPTAPI_API_KEY")
        return v.strip() if v else None

    def _from_dotenv(path: Path) -> str | None:
        if not path.exists():
            return None
        try:
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                if key.strip() != "TRANSCRIPTAPI_API_KEY":
                    continue
                value = value.strip()
                if len(value) >= 2 and value[0] in ('"', "'") and value[-1] == value[0]:
                    value = value[1:-1]
                return value or None
        except OSError:
            return None
        return None

    value = _from_env()
    if value:
        return value
    for candidate in (Path.home() / ".config" / "watch" / ".env", Path.cwd() / ".env"):
        value = _from_dotenv(candidate)
        if value:
            return value
    return None


def languages_from_sub_langs(sub_langs: str) -> str:
    """Map watch's regex --sub-langs to a TranscriptAPI priority list.

    TranscriptAPI ignores region and takes <=10 codes; `asr` = auto captions.
    'es,es-419,es-ES,en,en-US,en-GB,.*-orig' -> 'es,en,asr'. Drops regex tokens
    ('.*-orig'), 'all', and empties; region-strips ('es-419'->'es'); dedups;
    always appends 'asr' as the auto-generated fallback.
    """
    out: list[str] = []
    for tok in sub_langs.split(","):
        tok = tok.strip().lower()
        if not tok or tok == "all" or tok.endswith("-orig") or not tok[0].isalpha():
            continue
        base = tok.split("-")[0]
        if base and base not in out:
            out.append(base)
    if "asr" not in out:
        out.append("asr")
    return ",".join(out[:10])


def _retry_after(exc: urllib.error.HTTPError) -> float | None:
    header = exc.headers.get("Retry-After") if getattr(exc, "headers", None) else None
    if not header:
        return None
    try:
        return float(header)
    except ValueError:
        return None


def _get(path: str, params: dict) -> tuple[int, dict | None]:
    """GET with bounded retry on transient codes. Returns (status, json|None)."""
    api_key = params.pop("_api_key")
    url = f"{API_BASE}{path}?{urllib.parse.urlencode(params)}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "watch-skill/1.0 (+claude-code; python-urllib)",
        "Accept": "application/json",
    }
    context = ssl.create_default_context()
    for attempt in range(MAX_ATTEMPTS):
        request = Request(url, headers=headers, method="GET")
        try:
            with urlopen(request, timeout=60, context=context) as response:
                body = response.read().decode("utf-8", errors="replace")
                return getattr(response, "status", 200), json.loads(body or "{}")
        except urllib.error.HTTPError as exc:
            if exc.code in RETRYABLE and attempt < MAX_ATTEMPTS - 1:
                time.sleep(_retry_after(exc) or 2.0 * (attempt + 1))
                continue
            try:
                detail = json.loads(exc.read().decode("utf-8", errors="replace") or "{}")
            except Exception:
                detail = None
            return exc.code, detail
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(2.0 * (attempt + 1))
                continue
            return 0, None  # network give-up → treated as failure by caller
    return 0, None


def fetch_transcript(
    video_url: str, api_key: str, language: str | None = None
) -> tuple[list[dict], str] | None:
    """Return (segments, resolved_language) or None on any failure (logged).

    Segments are {start, end, text}; end = start + duration. NEVER raises —
    TranscriptAPI is a best-effort source, so the caller can fall back.
    """
    params: dict = {
        "_api_key": api_key,
        "video_url": video_url,
        "format": "json",
        "include_timestamp": "true",
    }
    if language:
        params["language"] = language

    status, data = _get("/youtube/transcript", params)

    if status == 200 and isinstance(data, dict):
        segments: list[dict] = []
        for item in data.get("transcript") or []:
            text = (item.get("text") or "").strip()
            if not text:
                continue
            start = round(float(item.get("start") or 0.0), 2)
            duration = float(item.get("duration") or 0.0)
            segments.append({"start": start, "end": round(start + duration, 2), "text": text})
        if segments:
            return segments, str(data.get("language") or language or "")
        print("[watch] TranscriptAPI returned an empty transcript — falling back", file=sys.stderr)
        return None

    reason = ""
    if isinstance(data, dict):
        reason = str(data.get("detail") or "")
    if status == 401:
        msg = "invalid/missing TRANSCRIPTAPI_API_KEY"
    elif status == 402:
        msg = "no TranscriptAPI credits remaining (transcriptapi.com/billing)"
    elif status == 404:
        msg = f"no transcript available{': ' + reason if reason else ''}"
    elif status == 0:
        msg = "network error / no response"
    else:
        msg = f"HTTP {status}{': ' + reason if reason else ''}"
    print(f"[watch] TranscriptAPI unavailable ({msg}) — falling back to captions/Whisper",
          file=sys.stderr)
    return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: transcriptapi.py <youtube-url-or-id> [lang-csv]", file=sys.stderr)
        raise SystemExit(2)
    key = load_api_key()
    if not key:
        raise SystemExit("TRANSCRIPTAPI_API_KEY not set")
    lang = sys.argv[2] if len(sys.argv) > 2 else None
    res = fetch_transcript(sys.argv[1], key, language=lang)
    if res is None:
        raise SystemExit("no transcript")
    segs, resolved = res
    print(json.dumps({"language": resolved, "segments": segs}, ensure_ascii=False, indent=2))
```

- [ ] **Step 4: Run — verify pass**

Run: `python3 -m pytest tests/test_transcriptapi.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**
```bash
git add skills/watch/scripts/transcriptapi.py tests/test_transcriptapi.py
git commit -m "feat(transcriptapi): stdlib YouTube-transcript backend (bot-gate-proof, best-effort)"
```

---

### Task 3: Wire TranscriptAPI into watch.py (precedence + `--no-transcriptapi`)

**Files:**
- Modify: `skills/watch/scripts/watch.py`
- Test: `tests/test_watch_transcriptapi.py`

**Interfaces:**
- Consumes: `transcriptapi.is_youtube_url`, `.load_api_key`, `.languages_from_sub_langs`, `.fetch_transcript`; `DEFAULT_SUB_LANGS` from `download`.
- Produces: for a YouTube URL, transcript resolves via TranscriptAPI first; `transcript_source == "transcriptapi (<lang>)"`; the yt-dlp caption parse + Whisper are skipped (they are already guarded by `if not transcript_segments`). `--no-transcriptapi` opts out. On any TranscriptAPI miss, the existing path runs unchanged.

- [ ] **Step 1: Write the failing test**

Create `tests/test_watch_transcriptapi.py`:

```python
"""watch.main() prefers TranscriptAPI for YouTube URLs; falls back on miss."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import watch  # noqa: E402


YT = "https://www.youtube.com/watch?v=m0Jk8aIDiJ8"


def _no_frames(monkeypatch):
    # Keep main() cheap + offline: transcript detail, no captions/whisper/frames.
    monkeypatch.setattr(watch, "fetch_captions",
                        lambda *a, **k: {"video_path": None, "subtitle_path": None,
                                         "info": {"title": "T", "duration": 30}, "downloaded": False})


def test_youtube_uses_transcriptapi(monkeypatch, tmp_path, capsys):
    _no_frames(monkeypatch)
    called = {"n": 0}
    def fake_fetch(url, key, language=None):
        called["n"] += 1
        return ([{"start": 0.0, "end": 2.0, "text": "hola"}], "es")
    monkeypatch.setattr(watch.transcriptapi, "fetch_transcript", fake_fetch)
    monkeypatch.setattr(watch.transcriptapi, "load_api_key", lambda: "sk_test")
    # If TranscriptAPI wins, parse_vtt must never run:
    monkeypatch.setattr(watch, "parse_vtt", lambda *a, **k: (_ for _ in ()).throw(AssertionError("captions used")))
    monkeypatch.setattr(sys, "argv", ["watch.py", YT, "--detail", "transcript", "--no-whisper", "--out-dir", str(tmp_path)])
    assert watch.main() == 0
    assert called["n"] == 1
    out = capsys.readouterr().out
    assert "transcriptapi" in out.lower()
    assert "hola" in out


def test_no_transcriptapi_flag_skips_it(monkeypatch, tmp_path):
    _no_frames(monkeypatch)
    def boom(*a, **k):
        raise AssertionError("TranscriptAPI called despite --no-transcriptapi")
    monkeypatch.setattr(watch.transcriptapi, "fetch_transcript", boom)
    monkeypatch.setattr(watch.transcriptapi, "load_api_key", lambda: "sk_test")
    monkeypatch.setattr(sys, "argv", ["watch.py", YT, "--detail", "transcript",
                                      "--no-whisper", "--no-transcriptapi", "--out-dir", str(tmp_path)])
    assert watch.main() == 0  # no transcript, but no crash


def test_transcriptapi_miss_falls_back(monkeypatch, tmp_path):
    _no_frames(monkeypatch)
    monkeypatch.setattr(watch.transcriptapi, "load_api_key", lambda: "sk_test")
    monkeypatch.setattr(watch.transcriptapi, "fetch_transcript", lambda *a, **k: None)  # miss
    hit = {"n": 0}
    def fake_vtt(*a, **k):
        hit["n"] += 1
        return [{"start": 0.0, "end": 1.0, "text": "caption fallback"}]
    # Give the caption path a subtitle to parse so the fallback is exercised:
    monkeypatch.setattr(watch, "fetch_captions",
                        lambda *a, **k: {"video_path": None, "subtitle_path": "x.vtt",
                                         "info": {"title": "T", "duration": 30}, "downloaded": False})
    monkeypatch.setattr(watch, "parse_vtt", fake_vtt)
    monkeypatch.setattr(sys, "argv", ["watch.py", YT, "--detail", "transcript", "--no-whisper", "--out-dir", str(tmp_path)])
    assert watch.main() == 0
    assert hit["n"] >= 1  # fell back to captions
```

- [ ] **Step 2: Run — verify fail**

Run: `python3 -m pytest tests/test_watch_transcriptapi.py -v`
Expected: FAIL — `watch` has no attribute `transcriptapi` / `--no-transcriptapi` unrecognized.

- [ ] **Step 3: Import + argparse**

In `watch.py`, add to the imports block (after `from transcribe import ...`):
```python
import transcriptapi  # noqa: E402
```

Add the argparse flag after the `--no-whisper` block:
```python
    ap.add_argument(
        "--no-transcriptapi",
        action="store_true",
        help="Disable the TranscriptAPI YouTube-transcript backend (transcriptapi.com). "
        "By default, YouTube URLs fetch their transcript from TranscriptAPI first "
        "(needs TRANSCRIPTAPI_API_KEY); this forces the yt-dlp caption / Whisper path.",
    )
```

- [ ] **Step 4: Resolve TranscriptAPI before captions/Whisper**

`watch.py`'s transcript resolution (the caption re-parse block near the `if not transcript_segments and dl.get("subtitle_path")` line, ~line 555 at v0.4.1) is where captions are parsed and, below it, Whisper. Insert a TranscriptAPI attempt **immediately before** that region so it wins for YouTube URLs. It sets `transcript_segments`, which short-circuits the existing `if not transcript_segments` guards (captions + Whisper) — do NOT rewrite those guards.

Insert (place it right after the resume `cached_tx` block if present, else right before the caption re-parse block):
```python
    # TranscriptAPI: for YouTube URLs, fetch the transcript from transcriptapi.com
    # (their servers clear YouTube's bot-gate / PO-token / nsig walls that block
    # yt-dlp caption fetch from a datacenter IP). Highest precedence; best-effort —
    # any miss falls through to the yt-dlp captions / Whisper path below. Skipped
    # when --audio supplies an external track (that VO is not the video's own).
    if (
        not transcript_segments
        and not args.no_transcriptapi
        and audio_override is None
        and url_source
        and transcriptapi.is_youtube_url(args.source)
    ):
        ta_key = transcriptapi.load_api_key()
        if ta_key:
            ta_langs = transcriptapi.languages_from_sub_langs(args.sub_lang or DEFAULT_SUB_LANGS)
            print(f"[watch] fetching transcript via TranscriptAPI (lang: {ta_langs})…", file=sys.stderr)
            ta_result = transcriptapi.fetch_transcript(args.source, ta_key, language=ta_langs)
            if ta_result is not None:
                all_segments, resolved_lang = ta_result
                transcript_segments = (
                    filter_range(all_segments, start_sec, end_sec) if focused else all_segments
                )
                transcript_text = format_transcript(transcript_segments)
                transcript_source = f"transcriptapi ({resolved_lang})" if resolved_lang else "transcriptapi"
                print(f"[watch] TranscriptAPI: {len(transcript_segments)} segments ({resolved_lang})", file=sys.stderr)
        else:
            print("[watch] no TRANSCRIPTAPI_API_KEY set — skipping TranscriptAPI, using captions/Whisper", file=sys.stderr)
```

Note: `start_sec`, `end_sec`, `focused`, `url_source`, `audio_override`, `args.sub_lang`, `DEFAULT_SUB_LANGS`, `filter_range`, `format_transcript` are all already in scope at that point (verify by reading the surrounding lines). If the resume `cached_tx` reuse block exists just above, place this AFTER it so a resumed transcript still wins and this becomes a no-op (its `if not transcript_segments` guard is false).

- [ ] **Step 5: Save resumed TranscriptAPI transcript too**

The resume stage-save (`if args.out_dir and transcript_segments: state.save_stage(... "transcript" ...)`) already runs after the whole transcript region, so a TranscriptAPI transcript is persisted for `--out-dir` resume automatically — confirm it is below this block; no new code needed.

- [ ] **Step 6: Run tests**

Run: `python3 -m pytest tests/test_watch_transcriptapi.py tests/test_resume.py -v`
Expected: PASS. Then confirm `python3 -c "import sys; sys.path.insert(0,'skills/watch/scripts'); import watch"`.

- [ ] **Step 7: Commit**
```bash
git add skills/watch/scripts/watch.py tests/test_watch_transcriptapi.py
git commit -m "feat(watch): TranscriptAPI is the top transcript source for YouTube URLs (--no-transcriptapi opts out)"
```

---

### Task 4: setup.py env template + status, SKILL.md docs, CHANGELOG, v0.4.2

**Files:**
- Modify: `skills/watch/scripts/setup.py` (add `TRANSCRIPTAPI_API_KEY` to the `.env` template + preflight status line)
- Modify: `skills/watch/SKILL.md` (frontmatter `0.4.1`→`0.4.2`; document the backend + `--no-transcriptapi` + the key)
- Modify: `.claude-plugin/plugin.json`, `.codex-plugin/plugin.json` (`0.4.2`)
- Modify: `CHANGELOG.md` (`## [0.4.2]` entry)
- Test: extend `tests/test_setup.py` if it asserts the env-template key set (follow its existing pattern; otherwise no test change)

- [ ] **Step 1: setup.py env template + status**

In `setup.py`, wherever the `.env` template keys are defined (the block listing `GROQ_API_KEY` / `OPENAI_API_KEY` / `ASSEMBLYAI_API_KEY` / OpenRouter / Gemini), add a commented `TRANSCRIPTAPI_API_KEY=` line with a one-line comment: `# TranscriptAPI (transcriptapi.com) — YouTube transcripts that beat the bot-gate`. In the preflight/status output that reports which backends are configured, add a line reporting whether `TRANSCRIPTAPI_API_KEY` is set (mirror the existing per-key status formatting). Match the file's existing style exactly; read it first.

- [ ] **Step 2: Bump versions**
```bash
cd ~/claude-video
sed -i 's/version: "0.4.1"/version: "0.4.2"/' skills/watch/SKILL.md
sed -i 's/"version": "0.4.1"/"version": "0.4.2"/' .claude-plugin/plugin.json .codex-plugin/plugin.json
grep -rn '0.4.2' skills/watch/SKILL.md .claude-plugin/plugin.json .codex-plugin/plugin.json
```
Expected: all three read `0.4.2`.

- [ ] **Step 3: SKILL.md docs**

In the flag list, add:
```markdown
- `--no-transcriptapi` — disable the TranscriptAPI YouTube-transcript backend. By default, YouTube URLs fetch their transcript from [TranscriptAPI](https://transcriptapi.com) first (needs `TRANSCRIPTAPI_API_KEY`), which succeeds from server IPs where YouTube blocks yt-dlp; this flag forces the yt-dlp caption / Whisper path instead.
```
Add a short section near "Transcription":
```markdown
## YouTube transcripts (TranscriptAPI)

For **YouTube URLs**, `/watch` resolves the transcript in this order: **TranscriptAPI → yt-dlp captions → Whisper**. TranscriptAPI (`transcriptapi.com`) runs the extraction on its own servers, so it clears YouTube's "confirm you're not a bot" / PO-token / nsig walls that block yt-dlp from datacenter/VPS IPs — with **zero Whisper spend** and no video download for `--detail transcript`. Set `TRANSCRIPTAPI_API_KEY` in `~/.config/watch/.env` (1 credit per transcript; video metadata is free). Language preference follows `--sub-lang` (default Spanish→English→auto). Pass `--no-transcriptapi` to force the yt-dlp path. YouTube *frames* still require the (often gated) video download — TranscriptAPI covers the transcript only.
```

- [ ] **Step 4: CHANGELOG**

Insert above the latest entry:
```markdown
## [0.4.2] — 2026-07-07

### Added
- **TranscriptAPI YouTube backend.** YouTube URLs now fetch their transcript
  from [TranscriptAPI](https://transcriptapi.com) before falling back to yt-dlp
  captions → Whisper. Their servers clear YouTube's bot-gate / PO-token / nsig
  walls that block yt-dlp from datacenter IPs, so a Spanish (or any-language)
  YouTube video yields a caption-quality transcript with zero Whisper spend and
  no video download for `--detail transcript`. New pure-stdlib `transcriptapi.py`
  backend, `TRANSCRIPTAPI_API_KEY` env/config key, and a `--no-transcriptapi`
  opt-out. Language preference follows `--sub-lang`. Best-effort: any miss (no
  key, no credits, no transcript, network) falls through with a logged reason.
```

- [ ] **Step 5: Run suite + commit**
```bash
python3 -m pytest -q
git add -A
git commit -m "docs: v0.4.2 — TranscriptAPI backend docs, setup key, changelog, version bump"
```
Expected: green (version-consistency held across manifests).

---

### Task 5: Full suite + import sanity

- [ ] **Step 1:** `python3 -m pytest -q` — all green (adds test_transcriptapi + test_watch_transcriptapi).
- [ ] **Step 2:** `python3 -c "import sys; sys.path.insert(0,'skills/watch/scripts'); import watch, transcriptapi; print('ok')"`.
- [ ] **Step 3:** CI paths unchanged — `grep -rn 'skills/watch' .github/workflows/*.yml` still resolves. Fix + commit only if stale.

---

### Task 6: E2E + user checkpoint (NO deploy without approval)

**Per CLAUDE.md: evidence before assertions; deploy only after approval.**

- [ ] **Step 1: Place the key (host, for the E2E)**

The `TRANSCRIPTAPI_API_KEY` (`sk_…`) already exists on the box in `~/.hermes-personal/config.yaml` / `~/.claude/.mcp.json`. Add it to watch's config (mode 0600, not committed):
```bash
touch ~/.config/watch/.env && chmod 600 ~/.config/watch/.env
grep -q '^TRANSCRIPTAPI_API_KEY=' ~/.config/watch/.env || printf 'TRANSCRIPTAPI_API_KEY=%s\n' '<sk_… from ~/.claude/.mcp.json>' >> ~/.config/watch/.env
```

- [ ] **Step 2: Real E2E — the Spanish video that failed via yt-dlp**
```bash
python3 skills/watch/scripts/watch.py "https://www.youtube.com/watch?v=m0Jk8aIDiJ8" \
  --detail transcript --out-dir /tmp/watch-ta
```
Expected: `[watch] TranscriptAPI: N segments (es)`; report `Transcript: … (via transcriptapi (es))` with Spanish text; NO Whisper line; NO video download. Also verify `--no-transcriptapi` on the same URL falls back (and, from this box, fails to fetch — proving the default path is what makes it work).

- [ ] **Step 3: STOP — user checkpoint**

Present evidence. On approval: merge `feat/v0.4.2-transcriptapi` → `main`, tag `v0.4.2`, push; add `TRANSCRIPTAPI_API_KEY` to `/opt/watch/secrets/.env` (container env) and `~/.claude/skills/watch` host is fine via `~/.config/watch/.env`; pull both live clones; `docker compose restart watch`; verify a live `/watch` YouTube transcript in-container; `memory_store` the backend + key-location facts.

---

## Self-Review

- **Spec coverage:** module (Task 2) → wiring/precedence + opt-out (Task 3) → setup/docs/version (Task 4) → verification (Task 5) → real E2E + gated deploy (Task 6). Best-effort fallback, `{start,end,text}` shape, no-new-deps, and secret handling all in Global Constraints and enforced per task.
- **Type consistency:** `fetch_transcript(video_url, api_key, language=None) -> tuple[list[dict], str] | None`; `is_youtube_url(str)->bool`; `load_api_key()->str|None`; `languages_from_sub_langs(str)->str` — identical across `transcriptapi.py`, its tests, and the `watch.py` call site. Segment dicts are `{start,end,text}` everywhere.
- **No placeholders:** every code/test step carries complete code; run steps have exact commands + expected output.
