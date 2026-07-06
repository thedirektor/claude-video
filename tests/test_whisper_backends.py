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
    # whisper_assemblyai.load_api_key() has no CONFIG_FILE constant to patch —
    # the ~/.config/watch/.env path is inlined via Path.home(). Redirect HOME
    # to an empty tmp_path (and cwd, since the cwd/.env fallback also runs)
    # so the probe can't pick up a real key from this machine's config.
    monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    assert whisper_assemblyai.load_api_key() is None
