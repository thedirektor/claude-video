"""Pure-function tests for the Gemini backend module."""
import gemini


def test_is_youtube_url():
    assert gemini.is_youtube_url("https://www.youtube.com/watch?v=abc123")
    assert gemini.is_youtube_url("https://youtu.be/abc123")
    assert not gemini.is_youtube_url("https://vimeo.com/12345")
    assert not gemini.is_youtube_url("/home/user/video.mp4")


def test_default_model_is_valid():
    assert gemini.DEFAULT_MODEL in gemini.VALID_MODELS
