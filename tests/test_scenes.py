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

    # Pinned from scripts/scenes.py: detect_scenes catches the missing-import
    # case itself, warns to stderr, and returns [] — it never raises. Callers
    # are expected to fall back to fps-based extraction on an empty list.
    assert scenes.detect_scenes(str(tmp_path / "nope.mp4")) == []
