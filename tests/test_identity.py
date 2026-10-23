import pytest
from identity import canonical_source, store_dirname, video_key

YT = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

@pytest.mark.parametrize("src", [
    "https://youtu.be/dQw4w9WgXcQ",
    "https://youtu.be/dQw4w9WgXcQ?si=abc&t=30",
    "http://youtube.com/watch?v=dQw4w9WgXcQ&list=PLx&index=2",
    "https://m.youtube.com/watch?feature=share&v=dQw4w9WgXcQ",
    "https://music.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://www.youtube.com/shorts/dQw4w9WgXcQ",
    "https://www.youtube.com/live/dQw4w9WgXcQ?si=x",
    "https://www.youtube.com/embed/dQw4w9WgXcQ",
    "  https://www.youtube.com/watch?v=dQw4w9WgXcQ#t=10  ",
])
def test_youtube_variants_collapse(src):
    assert canonical_source(src) == YT

def test_non_youtube_keeps_path_case_and_query():
    assert canonical_source("HTTPS://Vimeo.COM/Abc/123?h=Zz#x") == "https://vimeo.com/Abc/123?h=Zz"

def test_local_path_untouched():
    assert canonical_source(" /app/data/uploads/Clip One.mp4 ") == "/app/data/uploads/Clip One.mp4"

def test_youtube_without_id_falls_back():
    assert canonical_source("https://www.youtube.com/@channel") == "https://www.youtube.com/@channel"


def test_video_key_youtube_spellings():
    assert video_key("https://youtu.be/dQw4w9WgXcQ?si=x") == "yt:dQw4w9WgXcQ"

def test_video_key_extractor_and_url_fallback():
    assert video_key("https://vimeo.com/1", {"extractor_key": "Vimeo", "id": "1"}) == "vimeo:1"
    k = video_key("https://example.com/v.mp4")
    assert k.startswith("url:") and len(k) == 20 and k == video_key("HTTPS://Example.com:443/v.mp4#x")

def test_video_key_file_by_content(tmp_path):
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    a.write_bytes(b"x" * 10)
    b.write_bytes(b"x" * 10)
    k = video_key(str(a))
    assert k == video_key(str(b)) and k.startswith("file:") and len(k) == 21

def test_store_dirname_windows_safe():
    assert store_dirname("yt:dQw4w9WgXcQ") == "yt_dQw4w9WgXcQ"
    assert store_dirname("generic:a/b?c") == "generic_a_b_c"

def test_canonical_ipv6_and_ports():
    assert canonical_source("http://[::1]:8080/v") == "http://[::1]:8080/v"
    assert canonical_source("http://x.com:80/v") == "http://x.com/v"
    assert canonical_source("http://x.com:0/v") == "http://x.com:0/v"

@pytest.mark.parametrize("bad", ["", "http://", "https://[bad", "http://x:99999"])
def test_canonical_never_raises(bad):
    assert canonical_source(bad) == bad.strip()
