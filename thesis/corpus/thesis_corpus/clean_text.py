"""PDF-to-clean-text extraction pipeline -- first version.

Finds PDFs recursively under SOURCE_DIR, checks whether each can be opened
(PyMuPDF) and its text extracted (Docling, native pass first, OCR retry if
the native pass looks implausible), writes one output directory per document
under OUTPUT_DIR plus a corpus-level manifest, and logs every outcome. One
broken PDF never stops the batch.

Deliberately out of scope for this stage: chapter selection, Zotero
integration, entity extraction, LLM processing, embeddings, semantic search,
reference removal, or any other text analysis -- see the thesis Methods
chapter (corpus construction protocol) for where those happen instead.

Usage (from thesis/corpus/):
    python -m thesis_corpus.clean_text            # full batch
    python -m thesis_corpus.clean_text --limit 5   # first 5 PDFs only

Never modifies, moves, renames, or overwrites the source PDFs -- every write
target is under OUTPUT_DIR.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from thesis_corpus.extraction import (
    ConversionOutcome,
    PdfOpenError,
    convert_with_docling,
    get_page_count,
    is_text_plausible,
)
from thesis_corpus.ids import make_document_id
from thesis_corpus.writer import DocumentResult, ManifestWriter, PageRecord, write_document_outputs

CORPUS_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = CORPUS_DIR / "litterature with pdfs" / "Generated Corpus" / "files"
OUTPUT_DIR = CORPUS_DIR / "processed"
DOCUMENTS_DIR = OUTPUT_DIR / "documents"
MANIFEST_PATH = OUTPUT_DIR / "corpus_manifest.csv"
LOG_PATH = OUTPUT_DIR / "pipeline.log"

logger = logging.getLogger("thesis_corpus.clean_text")


def setup_logging() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def discover_pdfs(source_dir: Path) -> list[Path]:
    return sorted(source_dir.rglob("*.pdf"))


def process_one(pdf_path: Path, document_id: str) -> DocumentResult:
    source_relative_path = str(pdf_path.relative_to(SOURCE_DIR))
    base = dict(
        document_id=document_id,
        source_relative_path=source_relative_path,
        source_filename=pdf_path.name,
        page_count=None,
        extraction_method=None,
        ocr_used=False,
        processing_status="failed",
        error_message="",
    )

    try:
        page_count = get_page_count(pdf_path)
    except PdfOpenError as e:
        logger.warning("[%s] unreadable: %s", document_id, e)
        return DocumentResult(**{**base, "processing_status": "unreadable", "error_message": str(e)})

    base["page_count"] = page_count

    try:
        outcome = convert_with_docling(pdf_path, ocr=False)
        used_ocr = False
        if not is_text_plausible(outcome.pages, page_count):
            logger.info("[%s] native extraction implausible, retrying with OCR", document_id)
            outcome = convert_with_docling(pdf_path, ocr=True)
            used_ocr = True
    except Exception as e:
        logger.error("[%s] extraction failed: %s", document_id, e)
        return DocumentResult(**{**base, "processing_status": "failed", "error_message": str(e)})

    method = "ocr" if used_ocr else "native"
    pages = [
        PageRecord(
            document_id=document_id,
            page_number=page_no,
            text=text,
            extraction_method=method,
            character_count=len(text),
            warnings=[],
        )
        for page_no, text in outcome.pages
    ]

    # Pages Docling produced no text item for at all (e.g. a blank page)
    # still get a record, per the one-record-per-PDF-page requirement.
    seen_pages = {p.page_number for p in pages}
    for page_no in range(1, page_count + 1):
        if page_no not in seen_pages:
            pages.append(PageRecord(
                document_id=document_id, page_number=page_no, text="",
                extraction_method=method, character_count=0,
                warnings=["no text extracted for this page"],
            ))
    pages.sort(key=lambda p: p.page_number)

    status = "processed_with_ocr" if used_ocr else "processed"
    return DocumentResult(
        **{**base, "extraction_method": method, "ocr_used": used_ocr,
           "processing_status": status, "error_message": ""},
        pages=pages,
        markdown=outcome.markdown,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                         help="Process only the first N discovered PDFs (for testing).")
    args = parser.parse_args()

    setup_logging()

    if not SOURCE_DIR.exists():
        raise SystemExit(f"Source directory not found: {SOURCE_DIR}")

    pdfs = discover_pdfs(SOURCE_DIR)
    logger.info("Discovered %d PDFs under %s", len(pdfs), SOURCE_DIR)
    if args.limit is not None:
        pdfs = pdfs[:args.limit]
        logger.info("Limiting to first %d PDFs", len(pdfs))

    used_ids: set[str] = set()
    manifest = ManifestWriter(MANIFEST_PATH)
    counts = {"processed": 0, "processed_with_ocr": 0, "unreadable": 0, "failed": 0}

    try:
        for i, pdf_path in enumerate(pdfs, 1):
            document_id = make_document_id(pdf_path, SOURCE_DIR, used_ids)
            logger.info("[%d/%d] %s -> %s", i, len(pdfs), pdf_path.name, document_id)
            result = process_one(pdf_path, document_id)
            doc_dir = DOCUMENTS_DIR / document_id
            write_document_outputs(result, doc_dir)
            manifest.add(result, str(doc_dir.relative_to(OUTPUT_DIR)))
            counts[result.processing_status] += 1
    finally:
        manifest.close()

    logger.info("Done. %s", counts)
    print(f"\nProcessed {len(pdfs)} PDFs: {counts}")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
