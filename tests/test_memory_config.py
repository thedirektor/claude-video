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


def test_subprocess_does_not_read_host_config(tmp_path):
    import os
    import subprocess
    import sys
    home = tmp_path / "home"
    (home / ".config" / "watch").mkdir(parents=True)
    (home / ".config" / "watch" / ".env").write_text("WATCH_QDRANT_INBOX=/should/not/leak\n")
    code = (
        "import config;"
        "print(config.setting('WATCH_QDRANT_INBOX'));"
        "print(config.memory_paths()['index_db'])"
    )
    r = subprocess.run(
        [sys.executable, "-I", "-c", "import sys;sys.path.insert(0, %r);" % str(config.__file__.rsplit("/", 1)[0]) + code],
        env={**os.environ, "HOME": str(home)}, capture_output=True, text=True,
    )
    out = r.stdout.split()
    assert r.returncode == 0, r.stderr
    assert out[0] == "None"
    assert out[1] == os.environ["WATCH_INDEX_DB"]
