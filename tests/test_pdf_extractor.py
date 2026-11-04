from pathlib import Path

import pytest
from pypdf import PdfWriter

from cluster_c.unstructured.pdf_extractor import PdfExtractionError, extract_pdf_pages


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PDF = (
    ROOT
    / "data/unstructured/input/guidance/pdf/ea_general_guidance_sample_v1.pdf"
)


def test_extract_sample_guidance_page_by_page() -> None:
    pages = extract_pdf_pages(SAMPLE_PDF)

    assert [page.page_number for page in pages] == [1, 2, 3]
    assert "SAMPLE EA THREE-LENS DECISION GUIDANCE" in pages[0].text
    assert "GROWTH-G-001" in pages[1].text
    assert "SIMPLICITY-S-002" in pages[2].text


def test_image_only_or_blank_pdf_reports_ocr_required(tmp_path: Path) -> None:
    path = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with path.open("wb") as output:
        writer.write(output)

    with pytest.raises(PdfExtractionError, match="ocr_required") as error:
        extract_pdf_pages(path)

    assert error.value.status == "ocr_required"


def test_pdf_without_pages_reports_empty_pdf(tmp_path: Path) -> None:
    path = tmp_path / "empty.pdf"
    writer = PdfWriter()
    with path.open("wb") as output:
        writer.write(output)

    with pytest.raises(PdfExtractionError, match="empty_pdf") as error:
        extract_pdf_pages(path)

    assert error.value.status == "empty_pdf"


def test_encrypted_pdf_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "encrypted.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.encrypt("secret")
    with path.open("wb") as output:
        writer.write(output)

    with pytest.raises(PdfExtractionError, match="encrypted_pdf") as error:
        extract_pdf_pages(path)

    assert error.value.status == "encrypted_pdf"


def test_corrupt_pdf_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.pdf"
    path.write_bytes(b"this is not a PDF")

    with pytest.raises(PdfExtractionError, match="corrupt_or_unreadable") as error:
        extract_pdf_pages(path)

    assert error.value.status == "corrupt_or_unreadable"
