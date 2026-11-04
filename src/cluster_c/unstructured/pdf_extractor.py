"""Page-level text extraction for general EA guidance PDFs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError


@dataclass(frozen=True)
class ExtractedPdfPage:
    """Text extracted from one physical PDF page."""

    page_number: int
    text: str


class PdfExtractionError(ValueError):
    """A PDF could not be accepted by the text-only guidance pipeline."""

    def __init__(self, status: str, path: Path, detail: str) -> None:
        self.status = status
        self.path = path
        self.detail = detail
        super().__init__(f"{status}: {path} - {detail}")


def extract_pdf_pages(path: Path) -> list[ExtractedPdfPage]:
    """Extract text with one-based page provenance from a text-based PDF.

    OCR is intentionally outside the first implementation. A PDF whose pages
    contain no extractable text receives the explicit ``ocr_required`` status.
    """
    try:
        reader = PdfReader(path)
    except (OSError, PdfReadError, ValueError) as exc:
        raise PdfExtractionError("corrupt_or_unreadable", path, str(exc)) from exc

    if reader.is_encrypted:
        raise PdfExtractionError(
            "encrypted_pdf",
            path,
            "Encrypted guidance PDFs are not supported",
        )
    if not reader.pages:
        raise PdfExtractionError("empty_pdf", path, "PDF contains no pages")

    pages: list[ExtractedPdfPage] = []
    try:
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            pages.append(ExtractedPdfPage(page_number=page_number, text=text))
    except (PdfReadError, TypeError, ValueError) as exc:
        raise PdfExtractionError("text_extraction_failed", path, str(exc)) from exc

    if not any(page.text for page in pages):
        raise PdfExtractionError(
            "ocr_required",
            path,
            "PDF pages contain no extractable text; OCR is not enabled",
        )
    return pages
