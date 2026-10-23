import tempfile

import config


def test_setting_env_beats_file(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("WATCH_STORE_DIR=/from/file\n")
    monkeypatch.setattr(config, "CONFIG_FILE", f)
    monkeypatch.setenv("WATCH_STORE_DIR", "/from/env")
    assert config.setting("WATCH_STORE_DIR") == "/from/env"
    monkeypatch.delenv("WATCH_STORE_DIR")
    assert config.setting("WATCH_STORE_DIR") == "/from/file"


def test_conftest_isolates_memory():  # Review Focus 1
    p = config.memory_paths()
    assert str(p["index_db"]).startswith(tempfile.gettempdir())
    assert not str(p["index_db"]).startswith("/opt/watch")
    assert config.embeddings_enabled() is False
    assert config.setting("WATCH_QDRANT_INBOX") is None and config.setting("WATCH_MODEL_DIR") is None


def test_each_test_gets_own_index(tmp_path):
    assert config.memory_paths()["index_db"].is_relative_to(tmp_path)
