from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class PdfExtractionError(Exception):
    """Raised when a PDF file cannot be parsed (corrupt/unreadable)."""


@dataclass
class ExtractedDocument:
    text: str
    content_hash: str


def extract_pdf(path: Path) -> ExtractedDocument:
    raw_bytes = path.read_bytes()

    try:
        reader = PdfReader(path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except (PdfReadError, ValueError) as exc:
        raise PdfExtractionError(f"Failed to read PDF: {path.name}") from exc

    content_hash = hashlib.sha256(raw_bytes).hexdigest()
    return ExtractedDocument(text=text, content_hash=content_hash)
