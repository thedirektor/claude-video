# Phase 1 — Re-port fork onto upstream 0.2.0 (v0.3.0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the fork's seven modules and pipeline features on top of upstream 0.2.0's `skills/watch/` layout, gaining upstream's pytest suite, frame dedup, Whisper auto-chunking, and detail levels — shipping as v0.3.0.

**Architecture:** Start a fresh branch from `upstream/main`. Copy fork-only modules 1:1 into `skills/watch/scripts/` (flat `scripts/*.py` map directly), then graft the fork's backend dispatch, whisper backends, two-pass sampling, and OCR onto upstream's rewritten `watch.py`/`whisper.py`/`frames.py`. Upstream history stays intact; every fork feature is a clean commit on top.

**Tech Stack:** Python 3.11 stdlib core; lazy optional deps (PySceneDetect, pytesseract+PIL, faster-whisper, assemblyai SDK, google-genai SDK); ffmpeg/ffprobe/yt-dlp binaries; pytest.

## Global Constraints

- Repo: `/home/thedirektor/claude-video`. Git refs: `upstream/main` = upstream 0.2.0; `main` = fork (flat layout). Read fork sources via `git show main:scripts/<file>` — never switch back to `main` mid-work.
- Working branch: `port/v0.3.0`, created in Task 1. All commits land there.
- Pure stdlib in core paths; optional heavy deps stay lazy-imported (import inside function, or guarded module import).
- Windows UTF-8 pattern — every script entrypoint keeps/gains this exact header block after imports:
  ```python
  for _stream in (sys.stdout, sys.stderr):
      try:
          _stream.reconfigure(encoding="utf-8", errors="replace")
      except (AttributeError, OSError):
          pass
  ```
- All `~/.config/watch/.env` reads use `encoding="utf-8", errors="replace"` (cp1252 tolerance, fork 0.2.2).
- Upstream conventions are law (from `AGENTS.md`): NO `commands/` directory (SKILL.md frontmatter provides `/watch`); `${SKILL_DIR}` in SKILL.md, never `${CLAUDE_SKILL_DIR}`; the skill folder `skills/watch/` stays self-contained; version synced across `skills/watch/SKILL.md` frontmatter, `.claude-plugin/plugin.json`, `.codex-plugin/plugin.json`.
- Version this phase: `0.3.0`.
- Do NOT re-port argv hardening / `encoding="utf-8"` config writes — those are upstream #2/#4 and already present in 0.2.0. Only the stdout/stderr reconfigure block and `errors="replace"` reads are fork-specific.
- Tests: run from repo root with `python3 -m pytest -q`. ffmpeg/ffprobe required locally (present on the box). All new tests must run offline (mock HTTP).
- NO pushing to origin, NO touching `~/.claude/skills/watch` or `/opt/watch/claude-video` — deployment happens only after the user checkpoint in Task 12.

---

### Task 1: Branch setup

**Files:**
- Create: branch `port/v0.3.0` from `upstream/main`
- Carry over: `docs/superpowers/` (spec + this plan) from `feat/v030-report-upstream`

**Interfaces:**
- Produces: the working branch every later task commits to.

- [ ] **Step 1: Create branch and carry docs**

```bash
cd ~/claude-video
git checkout -b port/v0.3.0 upstream/main
git checkout feat/v030-report-upstream -- docs/
git add docs/
git commit -m "docs: carry design spec and plan onto 0.2.0 port branch"
```

- [ ] **Step 2: Verify upstream baseline is green**

```bash
python3 -m pytest -q
```
Expected: all upstream tests pass (test_config, test_dedup, test_download, test_fixtures, test_frames, test_setup, test_timestamps, test_watch, test_whisper). If ffmpeg-dependent tests fail, stop and report — the box has ffmpeg, so failures mean something else is wrong.

- [ ] **Step 3: Confirm layout facts**

```bash
ls skills/watch/scripts/   # expect: build-skill.sh config.py download.py frames.py setup.py transcribe.py watch.py whisper.py
ls commands/ 2>/dev/null   # expect: No such file or directory
```

---

### Task 2: Windows UTF-8 headers + cp1252-tolerant env reads

**Files:**
- Modify: `skills/watch/scripts/watch.py`, `download.py`, `transcribe.py`, `whisper.py`, `setup.py`, `frames.py`, `config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: upstream `config.read_env_file(path: Path | None = None) -> dict[str, str]`.
- Produces: same signatures; behavior change only (`errors="replace"` reads, UTF-8 stdout/stderr).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_config.py`:

```python
def test_read_env_file_tolerates_cp1252_bytes(tmp_path):
    env = tmp_path / ".env"
    env.write_bytes(b"GROQ_API_KEY=gsk_test\x97key\n")  # \x97 = cp1252 em dash, invalid UTF-8
    values = config.read_env_file(env)
    assert values["GROQ_API_KEY"].startswith("gsk_test")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_config.py::test_read_env_file_tolerates_cp1252_bytes -v`
Expected: FAIL with `UnicodeDecodeError` (or KeyError from the OSError swallow path returning `{}`).

- [ ] **Step 3: Implement**

In `skills/watch/scripts/config.py`, inside `read_env_file`, change the file read to:

```python
        text = path.read_text(encoding="utf-8", errors="replace")
```

Apply the same `errors="replace"` to every `.read_text(encoding="utf-8")` on `CONFIG_FILE`/dotenv paths in `whisper.py` (`load_api_key`) and `setup.py` (`_read_env_key`). Check for others:

```bash
grep -rn 'read_text(encoding="utf-8")' skills/watch/scripts/
```
Every hit on a config/.env path gets `errors="replace"`. (File *writes* stay plain `encoding="utf-8"`.)

Then add the UTF-8 reconfigure header (exact block from Global Constraints) to each of `watch.py`, `download.py`, `transcribe.py`, `whisper.py`, `setup.py`, `frames.py` — placed right after the import block, before any constants. `config.py` prints nothing; skip it.

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_config.py -v && python3 -m py_compile skills/watch/scripts/*.py`
Expected: all PASS, clean compile.

- [ ] **Step 5: Commit**

```bash
git add skills/watch/scripts/ tests/test_config.py
git commit -m "fix: UTF-8 stdout/stderr + cp1252-tolerant .env reads (fork 0.2.0/0.2.2)"
```

---

### Task 3: Speaker-aware transcript formatting

**Files:**
- Modify: `skills/watch/scripts/transcribe.py`
- Test: `tests/test_transcribe.py` (create)

**Interfaces:**
- Consumes: segments `list[dict]` with keys `start, end, text` and optional `speaker`.
- Produces: `format_transcript(segments: list[dict]) -> str` — unchanged signature; emits `[Speaker A] (M:SS-M:SS) text` lines when any segment has `speaker`, else legacy `[MM:SS] text`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_transcribe.py`:

```python
"""Tests for transcribe.py formatting."""
import transcribe


def test_format_transcript_plain_segments():
    segs = [{"start": 0.0, "end": 2.0, "text": "hello"},
            {"start": 65.0, "end": 67.0, "text": "world"}]
    out = transcribe.format_transcript(segs)
    assert "[0:00] hello" in out
    assert "[1:05] world" in out
    assert "Speaker" not in out


def test_format_transcript_with_speakers():
    segs = [{"start": 0.0, "end": 2.5, "text": "hi", "speaker": "Speaker A"},
            {"start": 3.0, "end": 5.0, "text": "yo", "speaker": "Speaker B"}]
    out = transcribe.format_transcript(segs)
    assert "[Speaker A] (0:00-0:02) hi" in out
    assert "[Speaker B] (0:03-0:05) yo" in out
```

Note: check the exact legacy timestamp format first — read `format_transcript` in `skills/watch/scripts/transcribe.py` and match the assertion to its real `[M:SS]` output (upstream uses minutes without zero-pad; adjust assertions to the actual format before committing the test).

- [ ] **Step 2: Run test to verify the speaker test fails**

Run: `python3 -m pytest tests/test_transcribe.py -v`
Expected: plain test PASSES (upstream behavior), speaker test FAILS (no speaker support yet).

- [ ] **Step 3: Port the fork implementation**

View the fork version and transplant the speaker logic into upstream's `format_transcript`:

```bash
git show main:scripts/transcribe.py | sed -n '/def format_transcript/,/^def \|^if __name__/p'
```

The fork logic: compute `has_speakers = any("speaker" in s for s in segments)`; when true, each line renders `[{seg['speaker']}] ({fmt(start)}-{fmt(end)}) {text}` using the same minute:second formatter; when false, output is byte-identical to upstream. Copy it verbatim, adjusting only if upstream's formatter helper is named differently.

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_transcribe.py -v`
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add skills/watch/scripts/transcribe.py tests/test_transcribe.py
git commit -m "feat: speaker-aware transcript formatting (AssemblyAI diarization support)"
```

---

### Task 4: Port whisper_local.py and whisper_assemblyai.py

**Files:**
- Create: `skills/watch/scripts/whisper_local.py`, `skills/watch/scripts/whisper_assemblyai.py`
- Test: `tests/test_whisper_backends.py` (create)

**Interfaces:**
- Produces (consumed by Task 5's `resolve_backend`/`transcribe_video`):
  ```python
  # whisper_local.py
  VALID_MODELS = ("tiny", "base", "small", "medium", "large-v2", "large-v3")
  DEFAULT_MODEL = "large-v3"
  INSTALL_HINT: str
  def is_available() -> tuple[bool, str]
  def transcribe_local(audio_path, language=None, model_name=DEFAULT_MODEL,
                       compute_type="float16", device="cuda") -> list[dict]
  # whisper_assemblyai.py
  INSTALL_HINT: str
  def load_api_key() -> str | None            # ASSEMBLYAI_API_KEY
  def transcribe_assemblyai(audio_path, enable_diarization=True) -> list[dict]  # segments may carry "speaker"
  ```

- [ ] **Step 1: Copy modules verbatim from the fork**

```bash
git show main:scripts/whisper_local.py > skills/watch/scripts/whisper_local.py
git show main:scripts/whisper_assemblyai.py > skills/watch/scripts/whisper_assemblyai.py
python3 -m py_compile skills/watch/scripts/whisper_local.py skills/watch/scripts/whisper_assemblyai.py
```
No path adaptations needed — both are leaf modules with no intra-repo imports. Confirm:
```bash
grep -n "^from \|^import " skills/watch/scripts/whisper_local.py skills/watch/scripts/whisper_assemblyai.py
```
Expected: stdlib + lazy third-party only; no `from scripts.` / relative imports.

- [ ] **Step 2: Write tests (graceful-degradation paths — no GPU/SDK required)**

Create `tests/test_whisper_backends.py`:

```python
"""Tests for the optional whisper backends' graceful-degradation paths."""
import whisper_assemblyai
import whisper_local


def test_local_is_available_returns_tuple():
    ok, reason = whisper_local.is_available()
    assert isinstance(ok, bool)
    assert isinstance(reason, str)
    if not ok:
        assert reason  # a missing dep must explain itself


def test_local_install_hint_mentions_faster_whisper():
    assert "faster-whisper" in whisper_local.INSTALL_HINT


def test_assemblyai_load_api_key_env(monkeypatch):
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "aai_test123")
    assert whisper_assemblyai.load_api_key() == "aai_test123"


def test_assemblyai_load_api_key_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
    monkeypatch.setattr(whisper_assemblyai, "CONFIG_FILE", tmp_path / ".env", raising=False)
    assert whisper_assemblyai.load_api_key() is None
```

Note: before committing, read `whisper_assemblyai.py::load_api_key` — if the config-file path constant has a different name (or is inlined), adapt the monkeypatch in the last test to whatever it actually reads, or drop that test if the file path is hardcoded to `Path.home()` (then instead monkeypatch `HOME` via `monkeypatch.setenv("HOME", str(tmp_path))`).

- [ ] **Step 3: Run tests**

Run: `python3 -m pytest tests/test_whisper_backends.py -v`
Expected: all PASS (box has no CUDA GPU — `is_available()` returns `(False, <reason>)`, which the test accepts).

- [ ] **Step 4: Commit**

```bash
git add skills/watch/scripts/whisper_local.py skills/watch/scripts/whisper_assemblyai.py tests/test_whisper_backends.py
git commit -m "feat: local faster-whisper + AssemblyAI diarization backends"
```

---

### Task 5: whisper.py — resolve_backend + local/assemblyai dispatch on top of chunking

**Files:**
- Modify: `skills/watch/scripts/whisper.py`
- Test: `tests/test_whisper.py` (extend)

**Interfaces:**
- Consumes: Task 4 modules; upstream `load_api_key`, `extract_audio`, `plan_chunks`, `split_audio`, `transcribe_chunks`, `_transcribe_file`.
- Produces (consumed by Task 9's watch.py):
  ```python
  def resolve_backend(preferred: str | None = None) -> tuple[str | None, str | None, str | None]
      # (backend, api_key, error_hint); backend in {"local","assemblyai","groq","openai",None}
  def transcribe_video(video_path: str, audio_out: Path, backend: str | None = None,
                       api_key: str | None = None, model_name: str | None = None,
                       enable_diarization: bool = False) -> tuple[list[dict], str]
  ```
  Upstream's chunking path (groq/openai) is untouched; local/assemblyai transcribe the single extracted mp3 (no chunking — local has no upload limit; AssemblyAI's SDK handles size).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_whisper.py`:

```python
import sys
import types


def _fake_local(ok=True, reason=""):
    mod = types.ModuleType("whisper_local")
    mod.is_available = lambda: (ok, reason)
    mod.INSTALL_HINT = "pip install faster-whisper"
    mod.DEFAULT_MODEL = "large-v3"
    mod.VALID_MODELS = ("tiny", "large-v3")
    return mod


def _fake_assemblyai(key="aai_k"):
    mod = types.ModuleType("whisper_assemblyai")
    mod.load_api_key = lambda: key
    mod.INSTALL_HINT = "pip install assemblyai"
    return mod


def test_resolve_backend_auto_prefers_local(monkeypatch):
    monkeypatch.setitem(sys.modules, "whisper_local", _fake_local(ok=True))
    monkeypatch.setenv("GROQ_API_KEY", "gsk_x")
    backend, key, hint = whisper.resolve_backend(None)
    assert backend == "local"
    assert hint is None


def test_resolve_backend_auto_falls_to_groq_without_gpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "whisper_local", _fake_local(ok=False, reason="no CUDA"))
    monkeypatch.setenv("GROQ_API_KEY", "gsk_x")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    backend, key, hint = whisper.resolve_backend(None)
    assert backend == "groq"
    assert key == "gsk_x"


def test_resolve_backend_auto_never_picks_assemblyai(monkeypatch):
    monkeypatch.setitem(sys.modules, "whisper_local", _fake_local(ok=False, reason="no CUDA"))
    monkeypatch.setitem(sys.modules, "whisper_assemblyai", _fake_assemblyai())
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    backend, key, hint = whisper.resolve_backend(None)
    assert backend is None  # paid diarization is opt-in only


def test_resolve_backend_explicit_assemblyai(monkeypatch):
    monkeypatch.setitem(sys.modules, "whisper_assemblyai", _fake_assemblyai(key="aai_k"))
    backend, key, hint = whisper.resolve_backend("assemblyai")
    assert (backend, key) == ("assemblyai", "aai_k")


def test_resolve_backend_explicit_missing_key_gives_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "whisper_assemblyai", _fake_assemblyai(key=None))
    backend, key, hint = whisper.resolve_backend("assemblyai")
    assert backend is None
    assert hint  # actionable message
```

Also account for `.env` fallback: `resolve_backend`'s groq/openai leg goes through upstream `load_api_key`, which reads `~/.config/watch/.env` — tests that assert "no key" must also isolate HOME: add `monkeypatch.setenv("HOME", str(tmp_path))` and the `tmp_path` fixture argument to `test_resolve_backend_auto_never_picks_assemblyai` and any test that deletes env keys.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_whisper.py -v -k resolve_backend`
Expected: FAIL with `AttributeError: module 'whisper' has no attribute 'resolve_backend'`.

- [ ] **Step 3: Port resolve_backend + extend transcribe_video**

Reference the fork implementation:

```bash
git show main:scripts/whisper.py | sed -n '/def resolve_backend/,/^def /p'
git show main:scripts/whisper.py | sed -n '/def transcribe_video/,$p'
```

Add to upstream `whisper.py` (below `load_api_key`), adapted:

```python
def resolve_backend(preferred: str | None = None) -> tuple[str | None, str | None, str | None]:
    """Pick a transcription backend. Returns (backend, api_key, error_hint).

    Auto order: local (free, needs GPU) -> groq -> openai. AssemblyAI is
    paid diarization and is only used when explicitly requested.
    """
    sys.path.insert(0, str(Path(__file__).parent.resolve()))
    if preferred == "local":
        try:
            from whisper_local import INSTALL_HINT, is_available
        except ImportError:
            return None, None, "faster-whisper is not installed. pip install faster-whisper"
        ok, reason = is_available()
        if ok:
            return "local", None, None
        return None, None, f"Local Whisper unavailable: {reason}\n{INSTALL_HINT}"
    if preferred == "assemblyai":
        try:
            from whisper_assemblyai import INSTALL_HINT, load_api_key as _aai_key
        except ImportError:
            return None, None, "assemblyai SDK is not installed. pip install assemblyai"
        key = _aai_key()
        if key:
            return "assemblyai", key, None
        return None, None, "ASSEMBLYAI_API_KEY missing. Add it to ~/.config/watch/.env"
    if preferred in ("groq", "openai"):
        backend, key = load_api_key(preferred)
        if backend:
            return backend, key, None
        return None, None, f"No {preferred} API key found. Run setup.py"
    # auto
    try:
        from whisper_local import is_available
        ok, _reason = is_available()
        if ok:
            return "local", None, None
    except ImportError:
        pass
    backend, key = load_api_key(None)
    if backend:
        return backend, key, None
    return None, None, None
```

**Cross-check against the fork source from Step 3's `git show` and keep the fork's exact hint strings where they differ** — the fork's wording is the tested UX. Then extend upstream `transcribe_video`:

```python
def transcribe_video(video_path, audio_out, backend=None, api_key=None,
                     model_name=None, enable_diarization=False):
    if backend is None:
        backend, api_key, hint = resolve_backend(None)
        if backend is None:
            raise SystemExit(hint or "No Whisper backend available. Run setup.py")
    audio_path = extract_audio(video_path, audio_out)
    if backend == "local":
        from whisper_local import DEFAULT_MODEL, transcribe_local
        segments = transcribe_local(audio_path, model_name=model_name or DEFAULT_MODEL)
        return segments, "local"
    if backend == "assemblyai":
        from whisper_assemblyai import transcribe_assemblyai
        segments = transcribe_assemblyai(audio_path, enable_diarization=enable_diarization)
        return segments, "assemblyai"
    # groq / openai: keep upstream's existing size-check -> plan_chunks -> split_audio
    # -> transcribe_chunks flow EXACTLY as-is from here down.
    ...
```
Keep the entire remaining upstream body (chunking) untouched — only the two new dispatch branches and the resolve fallback are added. Preserve upstream's existing signature default compatibility: existing upstream callers pass `(video_path, audio_out, backend, api_key)` positionally, which still works.

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_whisper.py -v`
Expected: all PASS (old chunking tests + new resolve tests).

- [ ] **Step 5: Commit**

```bash
git add skills/watch/scripts/whisper.py tests/test_whisper.py
git commit -m "feat: resolve_backend + local/assemblyai dispatch atop upstream chunking"
```

---

### Task 6: Port speech.py, scenes.py, ocr.py

**Files:**
- Create: `skills/watch/scripts/speech.py`, `skills/watch/scripts/scenes.py`, `skills/watch/scripts/ocr.py`
- Test: `tests/test_speech.py`, `tests/test_scenes.py`, `tests/test_ocr.py` (create)

**Interfaces:**
- Produces (consumed by Task 9):
  ```python
  # speech.py
  DEFAULT_SPEECH_SHARE = 0.7
  def compute_speech_windows(segments, gap_threshold=2.0, range_start=None, range_end=None) -> list[tuple[float, float]]
  def two_pass_sample(range_start, range_end, speech_windows, scenes, max_frames, speech_share=0.7) -> dict
  def format_windows(windows, speech_total=None, full_duration=None) -> str
  # scenes.py
  DEFAULT_THRESHOLD = 27.0
  def detect_scenes(video_path, threshold=27.0, start=None, end=None) -> list[tuple[float, float]]  # lazy PySceneDetect
  def pick_midpoints(scenes, max_frames) -> list[float]
  # ocr.py
  DEFAULT_LANG = "spa+eng"
  def run_ocr(frame_paths, lang="spa+eng", min_conf=50) -> dict        # {path: text}; empty dict when tesseract missing
  def is_significant(text: str, min_chars: int = 10) -> bool
  ```

- [ ] **Step 1: Copy modules verbatim**

```bash
for f in speech.py scenes.py ocr.py; do git show main:scripts/$f > skills/watch/scripts/$f; done
python3 -m py_compile skills/watch/scripts/speech.py skills/watch/scripts/scenes.py skills/watch/scripts/ocr.py
grep -n "^from \|^import " skills/watch/scripts/speech.py skills/watch/scripts/scenes.py skills/watch/scripts/ocr.py
```
Expected: leaf modules, stdlib + lazy third-party imports only.

- [ ] **Step 2: Write tests**

Create `tests/test_speech.py`:

```python
"""Tests for speech-window computation and two-pass frame budgeting."""
import speech


def _segs(*pairs):
    return [{"start": a, "end": b, "text": "x"} for a, b in pairs]


def test_windows_merge_small_gaps():
    # 1.0s gap < 2.0 threshold -> single window
    wins = speech.compute_speech_windows(_segs((0, 5), (6, 10)))
    assert wins == [(0.0, 10.0)]


def test_windows_split_on_large_gaps():
    wins = speech.compute_speech_windows(_segs((0, 5), (10, 15)))
    assert len(wins) == 2


def test_windows_respect_range():
    wins = speech.compute_speech_windows(_segs((0, 5), (10, 15)), range_start=8.0, range_end=20.0)
    assert all(s >= 8.0 for s, _ in wins)


def test_two_pass_sample_returns_timestamps_within_range():
    wins = [(10.0, 20.0)]
    plan = speech.two_pass_sample(0.0, 30.0, wins, scenes=[], max_frames=10)
    ts = plan["timestamps"] if isinstance(plan, dict) else plan
    assert ts, "must produce timestamps"
    assert all(0.0 <= t <= 30.0 for t in ts)
    assert len(ts) <= 10


def test_two_pass_majority_of_budget_in_speech():
    wins = [(10.0, 20.0)]
    plan = speech.two_pass_sample(0.0, 100.0, wins, scenes=[], max_frames=10, speech_share=0.7)
    ts = plan["timestamps"] if isinstance(plan, dict) else plan
    in_speech = [t for t in ts if 10.0 <= t <= 20.0]
    assert len(in_speech) >= len(ts) * 0.5
```

**Before finalizing:** run `git show main:scripts/speech.py | sed -n '/def two_pass_sample/,/return/p'` and pin the tests to the REAL return shape (the fork returns a dict; assert its actual keys — e.g. `plan["timestamps"]`, and whatever speech/silent annotation key it carries — instead of the `isinstance` hedge above; replace the hedge with exact-key assertions once read).

Create `tests/test_scenes.py`:

```python
"""Tests for scene-span utilities (pure parts; PySceneDetect not required)."""
import scenes


def test_pick_midpoints_returns_span_centers():
    spans = [(0.0, 10.0), (10.0, 30.0)]
    mids = scenes.pick_midpoints(spans, max_frames=10)
    assert mids == [5.0, 20.0]


def test_pick_midpoints_caps_at_max_frames():
    spans = [(float(i), float(i + 1)) for i in range(50)]
    mids = scenes.pick_midpoints(spans, max_frames=10)
    assert len(mids) == 10


def test_detect_scenes_without_dependency_errors_cleanly(tmp_path):
    import importlib.util
    if importlib.util.find_spec("scenedetect") is not None:
        return  # dependency present; the graceful path can't be exercised
    try:
        scenes.detect_scenes(str(tmp_path / "nope.mp4"))
        raised = False
    except (SystemExit, ImportError):
        raised = True
    assert raised
```

Create `tests/test_ocr.py`:

```python
"""Tests for OCR helpers (no tesseract binary required)."""
import ocr


def test_is_significant_rejects_short_noise():
    assert not ocr.is_significant("ab")
    assert not ocr.is_significant("")


def test_is_significant_accepts_real_text():
    assert ocr.is_significant("Error: connection refused at line 42")


def test_run_ocr_missing_tesseract_returns_empty(monkeypatch):
    monkeypatch.setattr(ocr, "find_tesseract", lambda: None)
    assert ocr.run_ocr(["/nonexistent/frame.jpg"]) == {}
```

**Before finalizing** `test_run_ocr_missing_tesseract_returns_empty`: read `git show main:scripts/ocr.py | sed -n '/def run_ocr/,/^def /p'` — if `run_ocr` gates on `_load_pytesseract()` rather than `find_tesseract()`, monkeypatch that instead (`monkeypatch.setattr(ocr, "_load_pytesseract", lambda: (None, None, "missing"))`).

- [ ] **Step 3: Run tests, fix pins**

Run: `python3 -m pytest tests/test_speech.py tests/test_scenes.py tests/test_ocr.py -v`
Expected: all PASS. If a return-shape assertion fails, read the module source and fix the TEST pin (the modules are ported verbatim and already battle-tested — do not modify module code to satisfy a mis-pinned test).

- [ ] **Step 4: Commit**

```bash
git add skills/watch/scripts/speech.py skills/watch/scripts/scenes.py skills/watch/scripts/ocr.py tests/test_speech.py tests/test_scenes.py tests/test_ocr.py
git commit -m "feat: speech-aware sampling, PySceneDetect scenes, OCR modules"
```

---

### Task 7: frames.py — reextract_frame for OCR hi-res upgrades

**Files:**
- Modify: `skills/watch/scripts/frames.py`
- Test: `tests/test_frames.py` (extend)

**Interfaces:**
- Consumes: upstream `_scale_filter(resolution)`.
- Produces (consumed by Task 9): `def reextract_frame(video_path: str, out_path: Path, timestamp_seconds: float, resolution: int) -> bool`

Upstream already has `extract_at_timestamps` (returns `tuple[list[dict], dict]`) — the fork's variant is NOT ported; Task 9 uses upstream's.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_frames.py`:

```python
def test_reextract_frame_produces_higher_res(cut_clip, tmp_path):
    out = tmp_path / "hires.jpg"
    ok = frames.reextract_frame(str(cut_clip), out, 1.0, resolution=640)
    assert ok is True
    assert out.exists() and out.stat().st_size > 0


def test_reextract_frame_bad_timestamp_returns_false(cut_clip, tmp_path):
    out = tmp_path / "nope.jpg"
    ok = frames.reextract_frame(str(cut_clip), out, 99999.0, resolution=640)
    assert ok is False
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_frames.py -v -k reextract`
Expected: FAIL with `AttributeError: module 'frames' has no attribute 'reextract_frame'`.

- [ ] **Step 3: Port from fork, adapted to upstream helpers**

Reference: `git show main:scripts/frames.py | sed -n '/def reextract_frame/,/^def /p'`. Add to `skills/watch/scripts/frames.py` (near `extract_at_timestamps`), reusing upstream's `_scale_filter`:

```python
def reextract_frame(video_path: str, out_path: Path, timestamp_seconds: float, resolution: int) -> bool:
    """Re-extract a single frame at higher resolution (OCR hi-res upgrade)."""
    resolved = str(Path(video_path).resolve())
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", f"{timestamp_seconds:.3f}", "-i", resolved,
        "-frames:v", "1", "-vf", _scale_filter(resolution),
        "-q:v", "2", "-y", str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
        if proc.stderr:
            print(f"[frames] re-extract failed at {timestamp_seconds:.1f}s: {proc.stderr.strip()}",
                  file=sys.stderr)
        return False
    return True
```
Cross-check the fork body from the `git show` and prefer its exact ffmpeg arguments where they differ (it was tuned in production).

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_frames.py -v`
Expected: all PASS (upstream frame tests + the two new ones).

- [ ] **Step 5: Commit**

```bash
git add skills/watch/scripts/frames.py tests/test_frames.py
git commit -m "feat: reextract_frame for OCR hi-res frame upgrades"
```

---

### Task 8: Port openrouter.py and gemini.py backends

**Files:**
- Create: `skills/watch/scripts/openrouter.py`, `skills/watch/scripts/gemini.py`
- Test: `tests/test_openrouter.py`, `tests/test_gemini.py` (create)

**Interfaces:**
- Produces (consumed by Task 9):
  ```python
  # openrouter.py
  DEFAULT_VISION_MODEL = "google/gemini-2.5-flash"
  DEFAULT_AUDIO_MODEL = "qwen/qwen3-asr-flash-2026-02-10"
  def load_api_key() -> str | None                        # OPENROUTER_API_KEY
  def transcribe_audio(audio_path, model=DEFAULT_AUDIO_MODEL, api_key=None) -> list[dict]
  def analyze_with_frames(frame_paths, transcript_text, question,
                          vision_model=DEFAULT_VISION_MODEL, api_key=None) -> str
  # gemini.py
  DEFAULT_MODEL = "gemini-3.1-flash-lite"
  VALID_MODELS = ("gemini-3.1-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro")
  def load_api_key() -> str | None                        # GEMINI_API_KEY
  def is_youtube_url(source: str) -> bool
  def generate_with_video(source, question, model_name=DEFAULT_MODEL, is_youtube=False) -> str
  ```

- [ ] **Step 1: Copy modules verbatim**

```bash
git show main:scripts/openrouter.py > skills/watch/scripts/openrouter.py
git show main:scripts/gemini.py > skills/watch/scripts/gemini.py
python3 -m py_compile skills/watch/scripts/openrouter.py skills/watch/scripts/gemini.py
grep -n "^from \|^import " skills/watch/scripts/openrouter.py skills/watch/scripts/gemini.py
```
Expected: no intra-repo imports (leaf modules).

- [ ] **Step 2: Write tests (offline — mock urlopen / pure functions)**

Create `tests/test_gemini.py`:

```python
"""Pure-function tests for the Gemini backend module."""
import gemini


def test_is_youtube_url():
    assert gemini.is_youtube_url("https://www.youtube.com/watch?v=abc123")
    assert gemini.is_youtube_url("https://youtu.be/abc123")
    assert not gemini.is_youtube_url("https://vimeo.com/12345")
    assert not gemini.is_youtube_url("/home/user/video.mp4")


def test_default_model_is_valid():
    assert gemini.DEFAULT_MODEL in gemini.VALID_MODELS
```

Create `tests/test_openrouter.py`:

```python
"""Offline tests for the OpenRouter transcription fallback chain."""
import json
import types

import pytest

import openrouter


class _FakeResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode()
    def read(self):
        return self._body
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


def test_transcribe_audio_falls_back_through_chain(monkeypatch, tmp_path):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"\x00" * 128)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")

    calls = []

    def fake_urlopen(req, *a, **kw):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        calls.append(url)
        if "openrouter.ai" in url:
            import urllib.error
            raise urllib.error.HTTPError(url, 500, "boom", {}, None)
        return _FakeResponse({"segments": [{"start": 0.0, "end": 1.0, "text": "hola"}],
                              "text": "hola"})

    monkeypatch.setattr(openrouter, "urlopen", fake_urlopen, raising=False)
    segs = openrouter.transcribe_audio(audio)
    assert segs and segs[0]["text"] == "hola"
    assert any("openrouter.ai" in u for u in calls)   # tried primary + voxtral
    assert any("groq.com" in u for u in calls)        # landed on groq


def test_transcribe_audio_all_legs_fail_raises(monkeypatch, tmp_path):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"\x00" * 128)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))  # isolate ~/.config/watch/.env

    def fake_urlopen(req, *a, **kw):
        import urllib.error
        url = req.full_url if hasattr(req, "full_url") else str(req)
        raise urllib.error.HTTPError(url, 500, "boom", {}, None)

    monkeypatch.setattr(openrouter, "urlopen", fake_urlopen, raising=False)
    with pytest.raises(SystemExit):
        openrouter.transcribe_audio(audio)
```

**Before finalizing:** check how `openrouter.py` imports urlopen (`grep -n "urlopen" skills/watch/scripts/openrouter.py`). If it calls `urllib.request.urlopen(...)` (module-qualified) instead of a top-level `urlopen` name, monkeypatch `openrouter.urllib.request.urlopen` accordingly, or patch `urllib.request.urlopen` globally via `monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)`. Also confirm retry constants (`_MAX_ATTEMPTS = 3`) don't make this test slow — if the retry sleep is real, monkeypatch `openrouter.time.sleep` to a no-op (`monkeypatch.setattr(openrouter.time, "sleep", lambda s: None)`).

- [ ] **Step 3: Run tests**

Run: `python3 -m pytest tests/test_openrouter.py tests/test_gemini.py -v`
Expected: all PASS, total runtime < 10s (no real sleeps).

- [ ] **Step 4: Commit**

```bash
git add skills/watch/scripts/openrouter.py skills/watch/scripts/gemini.py tests/test_openrouter.py tests/test_gemini.py
git commit -m "feat: OpenRouter (vision+audio chain) and Gemini native-video backends"
```

---

### Task 9: watch.py — graft backends, two-pass sampling, and OCR onto upstream pipeline

**Files:**
- Modify: `skills/watch/scripts/watch.py`
- Test: `tests/test_watch.py` (extend)

**Interfaces:**
- Consumes: everything produced by Tasks 3-8, plus upstream `watch.py` internals (detail levels, cue frames, `extract_keyframes`, `extract_scene_or_uniform`, `merge_frames`, `fetch_captions` fast path).
- Produces: the final CLI. New flags added to upstream's set:
  ```
  question (positional, nargs="*", default=[])
  --backend {claude,gemini,openrouter}        default=claude
  --gemini-model {<VALID_MODELS>}             default=gemini-3.1-flash-lite
  --openrouter-vision-model                   default=google/gemini-2.5-flash
  --openrouter-audio-model                    default=qwen/qwen3-asr-flash-2026-02-10
  --audio PATH                                default=None
  --no-ocr                                    store_true
  --no-scene-detect                           store_true
  --scene-threshold FLOAT                     default=27.0
  --two-pass / --no-two-pass                  BooleanOptionalAction, default=True
  --whisper {groq,openai,local,assemblyai}    (extends upstream's {groq,openai})
  --whisper-model {<WHISPER_LOCAL_MODELS>}    default=large-v3
  --diarize / --no-diarize                    BooleanOptionalAction, default=True
  ```

This is the largest task. The fork's `watch.py` is the reference implementation — read it in full first:

```bash
git show main:scripts/watch.py > /tmp/fork-watch-reference.py
```

- [ ] **Step 1: Write the failing e2e tests**

Append to `tests/test_watch.py` (reuse its existing `_run(clip, *args, env_extra=...)` subprocess helper):

```python
def test_gemini_backend_requires_question(cut_clip, tmp_path):
    proc = _run_raw(cut_clip, "--backend", "gemini", "--out-dir", str(tmp_path))
    assert proc.returncode != 0
    assert "question" in (proc.stderr + proc.stdout).lower()


def test_unknown_backend_rejected(cut_clip, tmp_path):
    proc = _run_raw(cut_clip, "--backend", "bogus", "--out-dir", str(tmp_path))
    assert proc.returncode != 0


def test_claude_backend_default_pipeline_unchanged(cut_clip, tmp_path):
    # upstream flow must keep working with zero fork flags
    proc = _run_raw(cut_clip, "--no-whisper", "--no-ocr", "--out-dir", str(tmp_path))
    assert proc.returncode == 0
    assert "# watch: video report" in proc.stdout


def test_no_ocr_and_scene_threshold_accepted(cut_clip, tmp_path):
    proc = _run_raw(cut_clip, "--no-whisper", "--no-ocr", "--no-scene-detect",
                    "--scene-threshold", "30", "--no-two-pass", "--out-dir", str(tmp_path))
    assert proc.returncode == 0
```

Add the `_run_raw` helper next to the existing `_run` (same subprocess invocation but returning the completed process without asserting success — copy `_run`'s command construction exactly, dropping its success assertion; read `_run` first and mirror its env handling including `PYTHONIOENCODING=utf-8`).

- [ ] **Step 2: Run to verify failures**

Run: `python3 -m pytest tests/test_watch.py -v -k "gemini or bogus or no_ocr"`
Expected: FAIL — `--backend`/`--no-ocr` are unrecognized arguments (argparse exits 2, but stderr says "unrecognized arguments", not "question").

- [ ] **Step 3: Add imports, flags, and backend dispatch**

In `skills/watch/scripts/watch.py`:

(a) Extend the import block (after the existing `from whisper import ...` line — replace it):

```python
from ocr import is_significant, run_ocr
from scenes import DEFAULT_THRESHOLD, detect_scenes, pick_midpoints
from speech import DEFAULT_SPEECH_SHARE, compute_speech_windows, format_windows, two_pass_sample
from whisper import load_api_key, resolve_backend, transcribe_video
from whisper_local import DEFAULT_MODEL as WHISPER_LOCAL_DEFAULT_MODEL
from whisper_local import VALID_MODELS as WHISPER_LOCAL_MODELS
from gemini import DEFAULT_MODEL as GEMINI_DEFAULT_MODEL
from gemini import VALID_MODELS as GEMINI_MODELS
from openrouter import DEFAULT_AUDIO_MODEL as OR_AUDIO_DEFAULT
from openrouter import DEFAULT_VISION_MODEL as OR_VISION_DEFAULT
```

Add module constant `HIRES_WIDTH = 1024` next to the other constants.

(b) Add the argparse flags exactly as listed in **Interfaces** above. `question` is `ap.add_argument("question", nargs="*", default=[], help=...)` placed right after `source`. `--whisper` REPLACES upstream's `choices=["groq", "openai"]` with `choices=["groq", "openai", "local", "assemblyai"]`.

(c) Backend dispatch — in `main()`, immediately after the working-dir mkdir/print:

```python
    if args.backend == "gemini":
        return _run_gemini_backend(args, work)
    if args.backend == "openrouter":
        return _run_openrouter_backend(args, work)
```

(d) Transplant `_run_gemini_backend(args, work)` and `_run_openrouter_backend(args, work)` from `/tmp/fork-watch-reference.py` as complete functions, with these adaptations only:
   - Any call to the fork's `extract_at_timestamps(...)` (returns `list`) becomes upstream's `extract_at_timestamps(...)` (returns `tuple[list, dict]`) — unpack `frames_list, _meta = ...`.
   - The openrouter backend's frame-extraction step uses upstream's engines: replicate the same detail-based selection `main()` uses (`extract_keyframes` for `efficient`, else `extract_scene_or_uniform`), or simply call upstream's uniform `extract(...)` with the computed fps/target — match what the fork did (uniform + fps) unless it used scenes.
   - Keep the fork's question-required guard, `--audio`-ignored warnings, and report headers byte-for-byte.

- [ ] **Step 4: Graft whisper backend selection into the claude path**

In `main()`'s Whisper fallback block (upstream step 11: `load_api_key(args.whisper)` → `transcribe_video(...)`), replace with:

```python
        backend, api_key, hint = resolve_backend(args.whisper)
        if backend is None:
            if hint:
                print(f"[watch] {hint}", file=sys.stderr)
        else:
            try:
                model_name = args.whisper_model if backend == "local" else None
                diarize = args.diarize if backend == "assemblyai" else False
                segments, used = transcribe_video(
                    audio_source, work / "audio.mp3", backend, api_key,
                    model_name=model_name, enable_diarization=diarize)
                transcript = format_transcript(segments)
                if backend == "local":
                    transcript_source = f"whisper (local, {model_name})"
                elif backend == "assemblyai":
                    transcript_source = "whisper (assemblyai, diarized)" if diarize else "whisper (assemblyai)"
                else:
                    transcript_source = f"whisper ({used})"
            except SystemExit as exc:
                print(f"[watch] Whisper failed: {exc}", file=sys.stderr)
```
where `audio_source = args.audio if args.audio else video_path` (the fork's `--audio` external-file feature; also mirror the fork's guard that `--audio` + `--no-whisper` is an error — copy the exact guard from the reference). Preserve upstream's surrounding conditions (`not args.no_whisper and meta["has_audio"]`) but let `args.audio` bypass the `has_audio` check (external audio file supplies the sound).

- [ ] **Step 5: Graft two-pass sampling + OCR into the claude path**

Insert the two-pass branch into upstream's frame-extraction step (upstream step 9), ABOVE the detail-engine selection, so precedence is: cue frames (always) → two-pass (transcript + video + default on) → upstream detail engines → uniform fps. Transplant from `/tmp/fork-watch-reference.py` with adaptations:

```python
    use_two_pass = (args.two_pass and transcript_segments and video_path
                    and detail != "transcript" and args.fps is None and detail_budget != 0)
    if use_two_pass:
        speech_windows = compute_speech_windows(
            transcript_segments, range_start=effective_start, range_end=effective_end)
        scene_spans = []
        if not args.no_scene_detect:
            try:
                scene_spans = detect_scenes(video_path, threshold=args.scene_threshold,
                                            start=effective_start, end=effective_end)
            except (SystemExit, ImportError) as exc:
                print(f"[watch] scene detection unavailable ({exc}); "
                      "two-pass will use even spacing", file=sys.stderr)
        plan = two_pass_sample(effective_start or 0.0, effective_end or duration,
                               speech_windows, scene_spans, detail_budget,
                               speech_share=DEFAULT_SPEECH_SHARE)
        frames, ts_meta = extract_at_timestamps(video_path, work / "frames",
                                                plan_timestamps(plan), args.resolution,
                                                max_frames=detail_budget,
                                                start_seconds=effective_start,
                                                end_seconds=effective_end)
        engine_meta = {"engine": "two-pass", "candidate_count": len(plan_timestamps(plan)),
                       "selected_count": len(frames), "fallback": False}
```
**Adaptation contract:** `plan_timestamps(plan)` is a placeholder NAME for however the fork extracts the timestamp list + speech/silent labels from `two_pass_sample`'s return dict — copy the fork's exact unpacking and label-annotation code from the reference file (it annotates each frame `[speech]`/`[silent]` in the report). Keep variable names from upstream's `main()` (`transcript_segments`, `effective_start`, `effective_end`, `detail_budget`) — check their exact upstream names at the insertion point and use those. Apply `dedupe_perceptual` to the two-pass frames when `not args.no_dedup` (new behavior, cheap win — import it from `frames`).

Then the OCR pass, after ALL frame branches complete and `frames` is final (post `merge_frames`):

```python
    ocr_text: dict[str, str] = {}
    if not args.no_ocr and frames:
        ocr_text = run_ocr([f["path"] for f in frames])
        for f in frames:
            text = ocr_text.get(f["path"], "")
            if is_significant(text):
                hires = Path(f["path"]).with_name(Path(f["path"]).stem + "_hires.jpg")
                if reextract_frame(video_path, hires, f["timestamp_seconds"], HIRES_WIDTH):
                    f["path"] = str(hires)
```
(add `reextract_frame` to the `from frames import ...` list). Mirror the fork's report rendering: inline OCR text under each frame line — copy the fork's report-section code for OCR/speech annotations from the reference, adjusted to upstream's report loop variables.

- [ ] **Step 6: Run the full suite**

Run: `python3 -m pytest -q`
Expected: ALL tests pass — upstream's `test_watch.py` detail-engine tests (proves the default pipeline is untouched when fork flags are absent: no transcript → no two-pass → upstream engines run as before) plus the new backend tests.

- [ ] **Step 7: Manual smoke on a real clip**

```bash
cd ~/claude-video
ffmpeg -y -f lavfi -i "testsrc=duration=8:size=640x360:rate=10" -f lavfi -i "sine=frequency=440:duration=8" -shortest /tmp/claude-1000/-home-thedirektor/ec7d8186-1a61-42bd-ba7d-192b68c40707/scratchpad/smoke.mp4
python3 skills/watch/scripts/watch.py /tmp/claude-1000/-home-thedirektor/ec7d8186-1a61-42bd-ba7d-192b68c40707/scratchpad/smoke.mp4 --no-whisper --out-dir /tmp/claude-1000/-home-thedirektor/ec7d8186-1a61-42bd-ba7d-192b68c40707/scratchpad/watch-out
```
Expected: exit 0, report on stdout, frames in the out dir. OCR runs (tesseract present on box) or degrades with a stderr warning.

- [ ] **Step 8: Commit**

```bash
git add skills/watch/scripts/watch.py skills/watch/scripts/frames.py tests/test_watch.py
git commit -m "feat: multi-backend dispatch, two-pass sampling, OCR hi-res pass in watch.py"
```

---

### Task 10: setup.py — env template + status messaging for new backends

**Files:**
- Modify: `skills/watch/scripts/setup.py`
- Test: `tests/test_setup.py` (extend)

**Interfaces:**
- Consumes: upstream `_scaffold_env()`, `ENV_TEMPLATE`, `_status()`.
- Produces: `ENV_TEMPLATE` containing commented stanzas for `ASSEMBLYAI_API_KEY=`, `GEMINI_API_KEY=`, `OPENROUTER_API_KEY=` (the fork lacked the OpenRouter stanza — add it; `openrouter.py::load_api_key` already reads it from this file).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_setup.py` (reuse its `_run` fake-HOME helper):

```python
def test_scaffolded_env_mentions_all_backends(tmp_path):
    proc = _run(tmp_path)  # default install path scaffolds ~/.config/watch/.env
    env_file = tmp_path / ".config" / "watch" / ".env"
    assert env_file.exists()
    content = env_file.read_text(encoding="utf-8")
    for key in ("GROQ_API_KEY", "OPENAI_API_KEY", "ASSEMBLYAI_API_KEY",
                "GEMINI_API_KEY", "OPENROUTER_API_KEY"):
        assert key in content, f"missing {key} stanza"
```
**Before finalizing:** read `tests/test_setup.py`'s existing `_run` signature and mirror its exact invocation pattern (it runs `setup.py` via subprocess with a fake `$HOME`); adapt the call above to match.

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_setup.py -v -k all_backends`
Expected: FAIL — upstream template has only GROQ/OPENAI.

- [ ] **Step 3: Extend ENV_TEMPLATE**

Reference the fork's stanzas: `git show main:scripts/setup.py | grep -A4 'ASSEMBLYAI\|GEMINI'`. Append to upstream's `ENV_TEMPLATE` (keeping the 83da59f rule — no inline comments on value lines):

```python
# Optional: AssemblyAI for speaker diarization (--whisper assemblyai).
# ASSEMBLYAI_API_KEY=

# Optional: Gemini native-video backend (--backend gemini).
# GEMINI_API_KEY=

# Optional: OpenRouter backend (--backend openrouter) and audio transcription.
# OPENROUTER_API_KEY=
```
Also update `cmd_check`/`cmd_install` stderr messaging to mention `--whisper local` and `--backend gemini|openrouter` as keyless/alternative options where the fork did (see `git show main:scripts/setup.py` for the exact wording).

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_setup.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add skills/watch/scripts/setup.py tests/test_setup.py
git commit -m "feat: env template + setup messaging for assemblyai/gemini/openrouter"
```

---

### Task 11: SKILL.md merge, docs, version 0.3.0

**Files:**
- Modify: `skills/watch/SKILL.md`, `README.md`, `CHANGELOG.md`, `.claude-plugin/plugin.json`, `.claude-plugin/marketplace.json`, `.codex-plugin/plugin.json`, `.agents/plugins/marketplace.json`

**Interfaces:**
- Consumes: upstream SKILL.md section structure (11 sections, `${SKILL_DIR}` convention); fork SKILL.md fork-added sections.
- Produces: one merged SKILL.md; version `0.3.0` synced in SKILL.md frontmatter + both plugin.json files.

- [ ] **Step 1: Merge SKILL.md**

Reference: `git show main:SKILL.md > /tmp/fork-skill-reference.md`. Into upstream's `skills/watch/SKILL.md`, merge the fork-added sections, adapted:
- `## Backends` (with `--gemini-model` picker) — insert after upstream's `## How to invoke`.
- `### Speaker diarization` and `### Local Whisper (faster-whisper on GPU)` (with install/verify/model-picker/failure-modes subsections) — insert under upstream's `## Transcription`.
- `## Workflow examples` (three examples) — insert before upstream's `## Token efficiency`.
- `## Windows compatibility` — insert before `## Security & Permissions`; update its **Bundled scripts:** manifest line to the new 13-script list under `scripts/` (config, download, frames, gemini, ocr, openrouter, scenes, setup, speech, transcribe, watch, whisper, whisper_assemblyai, whisper_local — that's 14; count what's actually there with `ls skills/watch/scripts/*.py`).
- Every command in merged content uses `${SKILL_DIR}/scripts/...` — rewrite any `${CLAUDE_SKILL_DIR}` from the fork text. Verify zero leftovers: `grep -c CLAUDE_SKILL_DIR skills/watch/SKILL.md` → expected `0`.
- Frontmatter: `version: "0.3.0"`, `description` = fork's richer one-liner (scene detection + OCR + local whisper + diarization), `homepage`/`repository` → `https://github.com/thedirektor/claude-video`, `author: thedirektor`. Keep `allowed-tools: Bash, Read, AskUserQuestion` and `user-invocable: true`.
- Document the new flags in upstream's `## How to invoke` flag table: `--backend`, `--audio`, `--no-ocr`, `--no-scene-detect`, `--scene-threshold`, `--two-pass/--no-two-pass`, `--whisper local|assemblyai`, `--whisper-model`, `--diarize/--no-diarize`.

- [ ] **Step 2: Versions + manifests + README + CHANGELOG**

- `.claude-plugin/plugin.json`: `version: "0.3.0"`, author `thedirektor`, homepage/repository → fork URLs.
- `.codex-plugin/plugin.json`: `version: "0.3.0"` (keep its `skills` + `interface` structure).
- `.claude-plugin/marketplace.json` + `.agents/plugins/marketplace.json`: repoint `url` fields at `https://github.com/thedirektor/claude-video.git`; refresh descriptions to mention backends.
- `README.md`: take upstream's restructured README as base; re-add the fork's sections (backends, local whisper, Windows guide, install-from-fork URLs pointing at `thedirektor/claude-video`). Keep Brad credited as original author.
- `CHANGELOG.md`: new top entry:

```markdown
## [0.3.0] — 2026-07-06

### Changed
- Rebased the fork onto upstream 0.2.0: `skills/watch/` self-contained Agent
  Skills layout, detail levels (`WATCH_DETAIL`), perceptual frame dedup,
  Whisper auto-chunking, transcript-cue frames, and the upstream pytest suite.
- `/watch` slash command now comes from SKILL.md frontmatter (no `commands/`
  wrapper); scripts resolve via `${SKILL_DIR}`.

### Added (re-ported fork features)
- Multi-backend: `--backend claude|gemini|openrouter` (Gemini native video incl.
  YouTube pass-through; OpenRouter vision + audio with qwen3-asr → voxtral →
  groq fallback chain).
- Transcription backends: `--whisper local` (faster-whisper GPU) and
  `--whisper assemblyai` (+ `--diarize` speaker labels).
- Two-pass speech-aware frame sampling (`--two-pass`, default on),
  PySceneDetect option (`--scene-threshold`), OCR pass with hi-res re-extract
  (`--no-ocr` to skip), external audio track via `--audio`.
- cp1252-tolerant config reads; UTF-8 stdout/stderr on Windows.
```

- [ ] **Step 3: Verify + commit**

```bash
python3 - <<'EOF'
import json, re, sys
v1 = json.load(open(".claude-plugin/plugin.json"))["version"]
v2 = json.load(open(".codex-plugin/plugin.json"))["version"]
m = re.search(r'^version:\s*"?([\d.]+)"?', open("skills/watch/SKILL.md").read(), re.M)
v3 = m.group(1)
assert v1 == v2 == v3 == "0.3.0", (v1, v2, v3)
print("versions ok:", v1)
EOF
git add skills/watch/SKILL.md README.md CHANGELOG.md .claude-plugin/ .codex-plugin/ .agents/
git commit -m "docs: merge fork SKILL.md sections, rebrand, bump to 0.3.0"
```

---

### Task 12: CI repath + pytest job

**Files:**
- Create: `.github/workflows/smoke.yml` (fork's, repathed — upstream has only release.yml)

**Interfaces:**
- Consumes: fork smoke.yml structure (static / pipeline / build jobs).

- [ ] **Step 1: Port and repath the workflow**

```bash
git show main:.github/workflows/smoke.yml > .github/workflows/smoke.yml
```
Then repath every reference:
- `scripts/*.py` → `skills/watch/scripts/*.py` (py_compile step)
- `python scripts/setup.py` → `python skills/watch/scripts/setup.py`
- `python scripts/watch.py` → `python skills/watch/scripts/watch.py`
- `bash scripts/build-skill.sh` → `bash skills/watch/scripts/build-skill.sh` (output stays `dist/watch.skill`)
- Version-check step: compare CHANGELOG top entry against `.claude-plugin/plugin.json` AND `skills/watch/SKILL.md` frontmatter AND `.codex-plugin/plugin.json` (the AGENTS.md three-way sync rule).
- `hooks/hooks.json` and manifest-parse steps: paths unchanged (still repo root).

Add a `tests` job to the workflow:

```yaml
  tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install ffmpeg
        run: sudo apt-get update && sudo apt-get install -y ffmpeg
      - name: Install pytest
        run: pip install pytest
      - name: Run suite
        run: python3 -m pytest -q
```

- [ ] **Step 2: Validate YAML + run everything locally**

```bash
python3 -c "import yaml, sys; yaml.safe_load(open('.github/workflows/smoke.yml')); print('yaml ok')"
python3 -m pytest -q
python3 -m py_compile skills/watch/scripts/*.py
bash skills/watch/scripts/build-skill.sh && ls -la dist/watch.skill
```
Expected: yaml ok, suite green, compile clean, `dist/watch.skill` built. (If PyYAML is absent, `pip install --user pyyaml` or validate with `ruby -ryaml -e ...` — any parser works.)

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/smoke.yml
git commit -m "ci: repath smoke workflow to skills/watch layout, add pytest job"
```

---

### Task 13: E2E verification + user checkpoint (NO deploy without approval)

**Files:** none (verification only)

- [ ] **Step 1: Full local gate**

```bash
cd ~/claude-video && python3 -m pytest -q && python3 -m py_compile skills/watch/scripts/*.py
```

- [ ] **Step 2: Real e2e — claude backend, local file and YouTube**

```bash
python3 skills/watch/scripts/watch.py /tmp/claude-1000/-home-thedirektor/ec7d8186-1a61-42bd-ba7d-192b68c40707/scratchpad/smoke.mp4 --no-whisper --out-dir /tmp/claude-1000/-home-thedirektor/ec7d8186-1a61-42bd-ba7d-192b68c40707/scratchpad/e2e-local
python3 skills/watch/scripts/watch.py "https://www.youtube.com/watch?v=jNQXAC9IVRw" --out-dir /tmp/claude-1000/-home-thedirektor/ec7d8186-1a61-42bd-ba7d-192b68c40707/scratchpad/e2e-yt
```
Expected: both exit 0; the YouTube run ("Me at the zoo", 19s, has captions) produces a captions transcript and frames; report lists engine + dedup stats.

- [ ] **Step 3: Real e2e — gemini and openrouter backends (keys live in ~/.config/watch/.env)**

```bash
python3 skills/watch/scripts/watch.py "https://www.youtube.com/watch?v=jNQXAC9IVRw" --backend gemini "What is this video about?"
python3 skills/watch/scripts/watch.py /tmp/claude-1000/-home-thedirektor/ec7d8186-1a61-42bd-ba7d-192b68c40707/scratchpad/smoke.mp4 --backend openrouter "Describe the frames" --no-whisper
```
Expected: exit 0 and a model answer in each report. If a key is missing from `~/.config/watch/.env`, report WHICH backend couldn't be e2e-verified — do not fake success.

- [ ] **Step 4: STOP — user checkpoint**

Report to the user: test counts, e2e evidence (transcript source lines, frame counts, backend answers), and the deploy plan below. **Do not execute any of it without explicit approval:**

1. `git push origin port/v0.3.0`
2. Fast-forward or reset `main` to `port/v0.3.0`; tag the old flat-layout head first: `git tag v0.2.2-flat <old-main-sha>`; `git push origin main --tags`
3. `git -C ~/.claude/skills/watch pull` (live skill picks up 0.3.0)
4. `git -C /opt/watch/claude-video pull && docker compose -f /opt/watch/docker-compose.yml restart watch` — check first whether the watch service's code expects the flat `scripts/` layout (its Dockerfile/src may reference `scripts/watch.py`; if so, flag it BEFORE pulling — the layout change would break the container)
5. `memory_store` (Qdrant) the key port findings

---

## Deferred (explicitly NOT in this phase)

- Replacing PySceneDetect with upstream's ffmpeg scene detection inside two-pass (dep-drop candidate; revisit after Phase 3 usage).
- danielfrey63-style pluggable STT abstraction.
- Reliability pack items (YouTube 403 chain, `--sub-lang`, chunk cache, resume) → Phase 2 plan.
