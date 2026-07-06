"""Tests for OCR helpers (no tesseract binary required)."""
import ocr


def test_is_significant_rejects_short_noise():
    assert not ocr.is_significant("ab")
    assert not ocr.is_significant("")


def test_is_significant_accepts_real_text():
    assert ocr.is_significant("Error: connection refused at line 42")


def test_run_ocr_missing_tesseract_returns_empty(monkeypatch):
    # Pinned from scripts/ocr.py: run_ocr gates on _load_pytesseract() (which
    # returns (pytesseract, Image, tess_path) or (None, None, None)), not on
    # find_tesseract() directly — patch the actual gate it calls.
    monkeypatch.setattr(ocr, "_load_pytesseract", lambda: (None, None, "missing"))
    assert ocr.run_ocr(["/nonexistent/frame.jpg"]) == {}
