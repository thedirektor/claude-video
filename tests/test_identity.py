import pytest
from identity import canonical_source

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
