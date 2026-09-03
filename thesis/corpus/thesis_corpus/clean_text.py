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
# Defaults -- the literature corpus. --source-dir/--output-dir override both
# for other corpora (e.g. MIVILUDES); every sub-path below is always derived
# the same way, just rooted at whichever output_dir is in effect.
SOURCE_DIR = CORPUS_DIR / "litterature with pdfs" / "Generated Corpus" / "files"
OUTPUT_DIR = CORPUS_DIR / "processed" / "literature"
DOCUMENTS_DIR = OUTPUT_DIR / "documents"
MANIFEST_PATH = OUTPUT_DIR / "corpus_manifest.csv"
LOG_DIR = OUTPUT_DIR / "logs"
LOG_PATH = LOG_DIR / "pipeline.log"

logger = logging.getLogger("thesis_corpus.clean_text")


def setup_logging(log_dir: Path, log_path: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def discover_pdfs(source_dir: Path) -> list[Path]:
    return sorted(source_dir.rglob("*.pdf"))


def process_one(pdf_path: Path, document_id: str, source_dir: Path) -> DocumentResult:
    source_relative_path = str(pdf_path.relative_to(source_dir))
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
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR,
                         help="Directory to search recursively for PDFs (default: the literature corpus).")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                         help="Directory to write documents/, corpus_manifest.csv, and logs/ under "
                              "(default: the literature corpus's processed/).")
    args = parser.parse_args()

    source_dir = args.source_dir
    output_dir = args.output_dir
    documents_dir = output_dir / "documents"
    manifest_path = output_dir / "corpus_manifest.csv"
    log_dir = output_dir / "logs"
    log_path = log_dir / "pipeline.log"

    setup_logging(log_dir, log_path)
    logger.info("Source dir resolved to: %s", source_dir)
    logger.info("Output dir resolved to: %s", output_dir)

    if not source_dir.exists():
        raise SystemExit(f"Source directory not found: {source_dir}")

    pdfs = discover_pdfs(source_dir)
    logger.info("Discovered %d PDFs under %s", len(pdfs), source_dir)
    if args.limit is not None:
        pdfs = pdfs[:args.limit]
        logger.info("Limiting to first %d PDFs", len(pdfs))

    used_ids: set[str] = set()
    manifest = ManifestWriter(manifest_path)
    counts = {"processed": 0, "processed_with_ocr": 0, "unreadable": 0, "failed": 0}

    try:
        for i, pdf_path in enumerate(pdfs, 1):
            document_id = make_document_id(pdf_path, source_dir, used_ids)
            logger.info("[%d/%d] %s -> %s", i, len(pdfs), pdf_path.name, document_id)
            result = process_one(pdf_path, document_id, source_dir)
            doc_dir = documents_dir / document_id
            write_document_outputs(result, doc_dir)
            manifest.add(result, str(doc_dir.relative_to(output_dir)))
            counts[result.processing_status] += 1
    finally:
        manifest.close()

    logger.info("Done. %s", counts)
    print(f"\nProcessed {len(pdfs)} PDFs: {counts}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
