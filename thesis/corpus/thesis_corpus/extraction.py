"""PDF open/page-count checks (PyMuPDF) and text conversion (Docling).

PyMuPDF is used only for the preliminary open/page-count/basic-checks step
(catches corrupted or password-protected files before Docling even tries).
Docling owns the actual text extraction, both the native pass and the OCR
fallback -- it is a single pipeline that can be run with or without OCR
enabled, rather than two separate tools.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

# Single obvious place to change OCR languages (tesseract language codes).
# fra/deu require `brew install tesseract-lang` in addition to the base
# tesseract install -- see thesis_corpus/README.md.
OCR_LANGUAGES = ["eng", "fra", "deu"]

# Below this average characters-per-page, native extraction is treated as
# implausible (empty or garbled) and the OCR pass is triggered instead.
MIN_CHARS_PER_PAGE = 20


class PdfOpenError(Exception):
    """The PDF could not be opened at all (corrupted, encrypted, etc.)."""


def get_page_count(pdf_path: Path) -> int:
    try:
        with pymupdf.open(pdf_path) as doc:
            if doc.needs_pass:
                raise PdfOpenError(f"password-protected: {pdf_path.name}")
            return doc.page_count
    except PdfOpenError:
        raise
    except Exception as e:
        raise PdfOpenError(str(e)) from e


def _build_converter(ocr: bool) -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = ocr
    if ocr:
        pipeline_options.ocr_options = TesseractCliOcrOptions(lang=OCR_LANGUAGES)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


@dataclass
class ConversionOutcome:
    markdown: str
    pages: list[tuple[int, str]]  # (page_number, text), 1-indexed, only pages with text


def convert_with_docling(pdf_path: Path, ocr: bool) -> ConversionOutcome:
    """Run one Docling conversion pass (native or OCR) and return both the
    full markdown export and a best-effort per-page text reconstruction
    built from Docling's own provenance metadata."""
    converter = _build_converter(ocr)
    result = converter.convert(str(pdf_path))
    doc = result.document

    markdown = doc.export_to_markdown()

    pages_text: dict[int, list[str]] = {}
    for item, _level in doc.iterate_items():
        text = getattr(item, "text", None)
        if not text:
            continue
        prov = getattr(item, "prov", None) or []
        if not prov:
            continue
        page_no = prov[0].page_no
        pages_text.setdefault(page_no, []).append(text)

    pages = [(pn, "\n".join(chunks)) for pn, chunks in sorted(pages_text.items())]
    return ConversionOutcome(markdown=markdown, pages=pages)


def is_text_plausible(pages: list[tuple[int, str]], page_count: int) -> bool:
    """Heuristic: empty or far too little text for the page count signals a
    scanned/image-only PDF that needs OCR rather than a genuine short doc."""
    if not pages or page_count == 0:
        return False
    total_chars = sum(len(text) for _, text in pages)
    return (total_chars / max(page_count, 1)) >= MIN_CHARS_PER_PAGE
