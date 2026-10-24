"""watch.py doctor: structured health checks, safe --fix repairs, no secrets printed."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import doctor  # noqa: E402

SECRET = "sk-very-secret-value-123"


def _run_factory(outputs: dict[str, tuple[int, str]]):
    calls = []

    def run(cmd, *a, **k):
        calls.append(list(cmd))
        key = next((name for name in outputs if name in " ".join(cmd)), None)
        rc, out = outputs.get(key, (0, ""))
        return SimpleNamespace(returncode=rc, stdout=out, stderr="" if rc == 0 else out)
    run.calls = calls
    return run


@pytest.fixture
def env(monkeypatch, tmp_path):
    for k in ("IMMICH_BASE_URL", "NEXTCLOUD_URL", "PAPERLESS_URL", "WATCH_LLM_BASE_URL",
              "GEMINI_API_KEY", "TRANSCRIPTAPI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("WATCH_INDEX_DB", str(tmp_path / "index.db"))
    monkeypatch.setenv("WATCH_STORE_DIR", str(tmp_path / "store"))
    return monkeypatch


def _ok_run():
    return _run_factory({
        "ffmpeg -version": (0, "ffmpeg version 7.1"),
        "ffprobe -version": (0, "ffprobe version 7.1"),
        "--version": (0, "2026.10.21"),
        "--list-langs": (0, "List of available languages (3):\neng\nosd\nspa\n"),
        "--dump-single-json": (0, json.dumps({"id": "jNQXAC9IVRw", "title": "Me at the zoo"})),
    })


def _by_name(report):
    return {c["name"]: c for c in report["checks"]}


def test_all_good_offline(env):
    rep = doctor.run_checks(online=False, which=lambda n: f"/usr/bin/{n}", run=_ok_run(),
                            embedder_ok=lambda: True)
    c = _by_name(rep)
    assert c["ffmpeg"]["status"] == "ok" and "7.1" in c["ffmpeg"]["detail"]
    assert c["yt-dlp"]["status"] == "ok" and "2026.10.21" in c["yt-dlp"]["detail"]
    assert c["tesseract"]["status"] == "ok"
    assert "yt-dlp canary" not in c  # offline
    assert rep["ok"] is True


def test_missing_ffmpeg_fails(env):
    rep = doctor.run_checks(online=False, which=lambda n: None if n == "ffmpeg" else f"/usr/bin/{n}",
                            run=_ok_run(), embedder_ok=lambda: True)
    assert _by_name(rep)["ffmpeg"]["status"] == "fail" and rep["ok"] is False


def test_tesseract_without_spanish_warns(env):
    run = _run_factory({"--list-langs": (0, "List of available languages (2):\neng\nosd\n")})
    rep = doctor.run_checks(online=False, which=lambda n: f"/usr/bin/{n}", run=run, embedder_ok=lambda: True)
    t = _by_name(rep)["tesseract"]
    assert t["status"] == "warn" and "spa" in t["detail"]


def test_canary_breakage_suggests_fix(env):
    run = _run_factory({"--dump-single-json": (1, "ERROR: Unable to extract initial player response"),
                        "--version": (0, "2026.10.21")})
    rep = doctor.run_checks(online=True, which=lambda n: f"/usr/bin/{n}", run=run, embedder_ok=lambda: True,
                            http=lambda url: 200)
    c = _by_name(rep)["yt-dlp canary"]
    assert c["status"] == "fail" and c.get("fixable") is True


def test_endpoints_only_when_configured_and_secrets_never_shown(env):
    env.setenv("PAPERLESS_URL", "https://docs.example.com")
    env.setenv("GEMINI_API_KEY", SECRET)
    seen = []
    rep = doctor.run_checks(online=True, which=lambda n: f"/usr/bin/{n}", run=_ok_run(), embedder_ok=lambda: True,
                            http=lambda url: seen.append(url) or 401)
    c = _by_name(rep)
    assert seen == ["https://docs.example.com"]
    assert c["paperless"]["status"] == "ok"  # reachable (auth required is still reachable)
    assert "immich" not in c
    assert c["gemini key"]["status"] == "ok"
    assert SECRET not in json.dumps(rep) and SECRET not in doctor.render(rep)


def test_unreachable_endpoint_warns(env):
    env.setenv("IMMICH_BASE_URL", "https://photos.example.com")
    rep = doctor.run_checks(online=True, which=lambda n: f"/usr/bin/{n}", run=_ok_run(), embedder_ok=lambda: True,
                            http=lambda url: None)
    assert _by_name(rep)["immich"]["status"] == "warn"


def test_fix_upgrades_ytdlp_and_downloads_model(env, monkeypatch):
    run = _run_factory({"--dump-single-json": (1, "ERROR: Unable to extract initial player response")})
    healed, fetched = [], []
    monkeypatch.setattr(doctor.download, "self_heal", lambda **k: healed.append(k) or True)
    rep = doctor.run_checks(online=True, which=lambda n: f"/usr/bin/{n}", run=run, embedder_ok=lambda: False,
                            http=lambda url: 200)
    fixes = doctor.apply_fixes(rep, fetch_model=lambda: fetched.append(1) or True)
    assert healed == [{"force": True}] and fetched == [1]
    assert {f["name"] for f in fixes} == {"yt-dlp canary", "embedding model"}


def test_main_json_and_exit_code(env, monkeypatch, capsys):
    monkeypatch.setattr(doctor, "run_checks", lambda **k: {"ok": False, "checks": [
        {"name": "ffmpeg", "status": "fail", "detail": "not found"}]})
    assert doctor.main(["--json", "--offline"]) == 1
    assert json.loads(capsys.readouterr().out)["checks"][0]["name"] == "ffmpeg"


def test_watch_py_dispatches_doctor(monkeypatch):
    import watch
    got = []
    monkeypatch.setattr(watch.doctor, "main", lambda argv: got.append(argv) or 0)
    monkeypatch.setattr(sys, "argv", ["watch.py", "doctor", "--json"])
    assert watch.main() == 0 and got == [["--json"]]


def test_fallback_proxy_check(env):
    env.setenv("WATCH_YTDLP_FALLBACK_PROXY", "http://100.64.0.9:8888")
    common = dict(online=True, which=lambda n: f"/usr/bin/{n}", run=_ok_run(), embedder_ok=lambda: True,
                  http=lambda url: 200)
    assert _by_name(doctor.run_checks(**common, proxy_status=lambda p: 204))["fallback proxy"]["status"] == "ok"
    c = _by_name(doctor.run_checks(**common, proxy_status=lambda p: None))["fallback proxy"]
    assert c["status"] == "warn" and "storyboards" in c["detail"]


def test_no_fallback_proxy_no_check(env):
    env.delenv("WATCH_YTDLP_FALLBACK_PROXY", raising=False)
    rep = doctor.run_checks(online=True, which=lambda n: f"/usr/bin/{n}", run=_ok_run(), embedder_ok=lambda: True,
                            http=lambda url: 200, proxy_status=lambda p: pytest.fail("not configured"))
    assert "fallback proxy" not in _by_name(rep)


def test_breakdown_dependency_checks(env, monkeypatch):
    import importlib.util

    env.delenv("WATCH_YTDLP_FALLBACK_PROXY", raising=False)
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a: None if n == "cv2" else real(n, *a))
    checks = {c["name"]: c for c in doctor.run_checks(online=False, which=lambda b: None,
                                                      run=_run_factory({}), embedder_ok=lambda: None,
                                                      proxy_status=lambda p: None)["checks"]}
    assert checks["opencv"]["status"] == "warn" and "breakdown fields affected" in checks["opencv"]["detail"]
    assert "numpy" in checks and "breakdown fields affected" in checks["tesseract"]["detail"]
