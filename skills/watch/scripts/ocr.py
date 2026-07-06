#!/usr/bin/env python3
"""OCR a list of extracted frames with pytesseract.

Best-effort: if pytesseract or the tesseract binary is missing, we log a
warning and return an empty mapping rather than crashing. Confidence-filtered
output keeps low-quality noise (≈ random pixel matches in textures) out of
Claude's context.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Optional

# Force UTF-8 on stdout/stderr so non-ASCII output (Spanish accents) doesn't
# crash on Windows where the default is cp1252.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


DEFAULT_LANG = "spa+eng"
DEFAULT_MIN_CONF = 50
SIGNIFICANT_TEXT_MIN_CHARS = 10


def find_tesseract() -> Optional[str]:
    """Locate the tesseract binary. Falls back to known Windows install paths."""
    found = shutil.which("tesseract")
    if found:
        return found
    if sys.platform == "win32":
        candidates = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
        for c in candidates:
            if Path(c).exists():
                return c
    return None


def _load_pytesseract() -> tuple[object | None, object | None, str | None]:
    """Returns (pytesseract, PIL.Image, tesseract_path) or (None, None, None)."""
    try:
        import pytesseract  # type: ignore
    except ImportError:
        print(
            "[ocr] pytesseract is not installed — skipping OCR. "
            "Install with: pip install pytesseract pillow",
            file=sys.stderr,
        )
        return None, None, None
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        print(
            "[ocr] Pillow is not installed — skipping OCR. "
            "Install with: pip install pillow",
            file=sys.stderr,
        )
        return None, None, None

    tess_path = find_tesseract()
    if not tess_path:
        print(
            "[ocr] tesseract binary not found on PATH — skipping OCR. "
            "On Windows, install from https://github.com/UB-Mannheim/tesseract/wiki "
            "and ensure the eng+spa language packs are selected.",
            file=sys.stderr,
        )
        return None, None, None
    pytesseract.pytesseract.tesseract_cmd = tess_path
    return pytesseract, Image, tess_path


def _extract_words(data: dict, min_conf: int) -> str:
    """Pull confidence-filtered words from pytesseract.image_to_data output."""
    words: list[str] = []
    for word, conf in zip(data.get("text", []), data.get("conf", [])):
        try:
            c = float(conf)
        except (TypeError, ValueError):
            continue
        if c < min_conf:
            continue
        token = (word or "").strip()
        if token:
            words.append(token)
    return " ".join(words).strip()


def run_ocr(
    frame_paths: list[str],
    lang: str = DEFAULT_LANG,
    min_conf: int = DEFAULT_MIN_CONF,
) -> dict[str, str]:
    """Run OCR over each frame. Returns {frame_path: detected_text}.

    On any setup failure (missing pytesseract, Pillow, or tesseract binary),
    returns an empty dict and warns to stderr — the caller should treat OCR
    as opt-in best-effort, never as required.
    """
    pytesseract, Image, _ = _load_pytesseract()
    if pytesseract is None:
        return {}

    results: dict[str, str] = {}
    for path in frame_paths:
        try:
            with Image.open(path) as img:
                data = pytesseract.image_to_data(
                    img, lang=lang, output_type=pytesseract.Output.DICT
                )
            results[path] = _extract_words(data, min_conf)
        except Exception as exc:
            print(f"[ocr] failed on {path}: {exc}", file=sys.stderr)
            results[path] = ""
    return results


def is_significant(text: str, min_chars: int = SIGNIFICANT_TEXT_MIN_CHARS) -> bool:
    """True if OCR text is long enough to justify re-extracting at higher res."""
    return len((text or "").replace(" ", "")) >= min_chars


if __name__ == "__main__":
    import json

    if len(sys.argv) < 2:
        print("usage: ocr.py <frame_path> [<frame_path> ...]", file=sys.stderr)
        raise SystemExit(2)

    out = run_ocr(sys.argv[1:])
    json.dump(out, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
