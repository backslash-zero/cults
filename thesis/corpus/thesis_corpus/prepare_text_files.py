"""Converts standalone plain-text documents into the pages.jsonl shape
Stage 2 (extract_and_embed) expects.

For text sources that are neither a PDF (clean_text.py) nor a structured,
database-backed batch (prepare_interviews.py) -- e.g. a single reference
document like MIVILUDES's "Comment identifier une derive sectaire". Finds
every *.txt file under --source-dir and treats each one as a single page
(document_id derived from its filename, same slug convention as
clean_text.py): these are standalone documents with no natural page
structure to preserve, the same reasoning prepare_interviews.py uses.

Usage (from thesis/corpus/):
    python -m thesis_corpus.prepare_text_files \
        --source-dir "MIVILUDES/Sectarian Drifts" \
        --output-dir "processed/miviludes"
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from thesis_corpus.ids import normalize_filename

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("thesis_corpus.prepare_text_files")


def discover_text_files(source_dir: Path) -> list[Path]:
    return sorted(source_dir.rglob("*.txt"))


def process_one(text_path: Path, source_dir: Path, output_dir: Path, used_ids: set[str]) -> str:
    base = normalize_filename(text_path.name)
    document_id = base
    if document_id in used_ids:
        document_id = normalize_filename(str(text_path.relative_to(source_dir)))
    used_ids.add(document_id)

    text = text_path.read_text(encoding="utf-8")
    doc_dir = output_dir / "documents" / document_id
    doc_dir.mkdir(parents=True, exist_ok=True)

    page = {
        "document_id": document_id,
        "page_number": 1,
        "text": text,
        "extraction_method": "text_file",
        "character_count": len(text),
        "warnings": [],
    }
    with open(doc_dir / "pages.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(page, ensure_ascii=False) + "\n")

    metadata = {
        "document_id": document_id,
        "source_relative_path": str(text_path.relative_to(source_dir)),
        "source_filename": text_path.name,
        "page_count": 1,
        "extraction_method": "text_file",
        "processing_status": "processed",
        "error_message": "",
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }
    (doc_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return document_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=Path, required=True,
                         help="Directory to search recursively for .txt files.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if not args.source_dir.exists():
        raise SystemExit(f"Source directory not found: {args.source_dir}")

    text_files = discover_text_files(args.source_dir)
    logger.info("Discovered %d .txt files under %s", len(text_files), args.source_dir)

    used_ids: set[str] = set()
    for text_path in text_files:
        document_id = process_one(text_path, args.source_dir, args.output_dir, used_ids)
        logger.info("%s -> %s", text_path.name, document_id)

    print(f"\nProcessed {len(text_files)} text files -> {args.output_dir}")


if __name__ == "__main__":
    main()
