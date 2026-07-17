from pathlib import Path

import pytest
from reportlab.pdfgen import canvas

from app.pdf_extraction import PdfExtractionError, extract_pdf


def make_pdf(path: Path, lines: list[str]) -> None:
    c = canvas.Canvas(str(path), pagesize=(300, 300))
    y = 260
    for line in lines:
        c.drawString(30, y, line)
        y -= 20
    c.showPage()
    c.save()


def test_extract_pdf_returns_text_and_stable_hash(tmp_path: Path):
    pdf_path = tmp_path / "faq.pdf"
    make_pdf(pdf_path, ["Como funciona a renegociacao?", "O cliente pode renegociar parcelas."])

    result = extract_pdf(pdf_path)

    assert "renegociacao" in result.text.lower()
    assert len(result.content_hash) == 64  # sha-256 hex digest

    # Re-extracting the same unchanged file yields the same hash.
    result2 = extract_pdf(pdf_path)
    assert result2.content_hash == result.content_hash


def test_extract_pdf_different_content_yields_different_hash(tmp_path: Path):
    path_a = tmp_path / "a.pdf"
    path_b = tmp_path / "b.pdf"
    make_pdf(path_a, ["Conteudo A"])
    make_pdf(path_b, ["Conteudo B, bem diferente"])

    result_a = extract_pdf(path_a)
    result_b = extract_pdf(path_b)

    assert result_a.content_hash != result_b.content_hash


def test_extract_pdf_corrupt_file_raises_typed_error(tmp_path: Path):
    corrupt_path = tmp_path / "corrupt.pdf"
    corrupt_path.write_bytes(b"this is not a real pdf file")

    with pytest.raises(PdfExtractionError):
        extract_pdf(corrupt_path)
