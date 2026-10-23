from memory.textnorm import normalize_for_search


def test_normalize_spanish_and_digits():
    assert normalize_for_search("  Canción   ÑANDÚ ٢٠٢٦ ") == "cancion nandu 2026"
    assert normalize_for_search("Straße") == "strasse"
