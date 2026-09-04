"""Converts interview transcripts into the pages.jsonl shape Stage 2
(extract_and_embed) expects, so the same annotate/embed/reduce pipeline
works on interviews unmodified.

Interviews aren't PDFs and have no natural page structure the way a book or
article does, so each interview is treated as a single page (page_number 1)
containing its whole transcript. Uses each interview's original-language
transcript.txt, not translation_en.txt, even where a translation exists:
mixing original-language and translated text would be an inconsistent basis
for comparison, and bge-m3's multilingual embeddings (already used for the
French MIVILUDES corpus) are exactly why each interview can stay in its own
language rather than needing translation first.

There is deliberately no corpus_manifest.csv here (unlike clean_text.py) --
there's no meaningful native/OCR/failed distinction for already-transcribed
text, and the proofreading/correction pass already happened outside this
pipeline (see each interview's corrections.log). build_registry.py already
handles a corpus with no manifest by reporting stage1_status "n/a".

Usage (from thesis/corpus/):
    python -m thesis_corpus.prepare_interviews
    python -m thesis_corpus.prepare_interviews --limit 5
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_DIR = CORPUS_DIR / "interviews" / "cleaned"
DEFAULT_DATABASE_PATH = CORPUS_DIR / "interviews" / "metadata" / "database.json"
DEFAULT_OUTPUT_DIR = CORPUS_DIR / "processed" / "interviews"

# Some transcripts end with a "--- \n\n TRANSCRIPTION / EDITING NOTES" section:
# the transcriber's own commentary on ambiguous names, ASR corrections, etc.
# -- not interview content. Left in place, this gets chunked and annotated
# like real dialogue (confirmed on 3/26 transcripts: b2-aug13-1832,
# b3-aug22-2057, b3-aug22-2101 -- one bad extracted item, "cercle solaire",
# traced back to this section's own commentary rather than the interviewee's
# actual words). Stripped here so it never reaches the chunker.
_NOTES_SECTION_RE = re.compile(
    r"\n-{3,}\s*\n+TRANSCRIPTION\s*/\s*EDITING NOTES\b.*", re.DOTALL,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("thesis_corpus.prepare_interviews")


def load_entries(database_path: Path) -> list[dict]:
    return json.loads(database_path.read_text(encoding="utf-8"))["interviews"]


def process_one(entry: dict, source_dir: Path, output_dir: Path) -> str:
    document_id = entry["id"]
    transcript_path = source_dir / document_id / "transcript.txt"
    if not transcript_path.exists():
        logger.warning("[%s] no transcript.txt at %s -- skipping", document_id, transcript_path)
        return "missing"

    text = transcript_path.read_text(encoding="utf-8")
    text = _NOTES_SECTION_RE.sub("", text).rstrip() + "\n"
    doc_dir = output_dir / "documents" / document_id
    doc_dir.mkdir(parents=True, exist_ok=True)

    page = {
        "document_id": document_id,
        "page_number": 1,
        "text": text,
        "extraction_method": "transcript",
        "character_count": len(text),
        "warnings": [],
    }
    with open(doc_dir / "pages.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(page, ensure_ascii=False) + "\n")

    metadata = {
        "document_id": document_id,
        "source_relative_path": str(transcript_path.relative_to(source_dir)),
        "source_filename": transcript_path.name,
        "language": entry.get("language"),
        "translated": entry.get("translated", False),
        "page_count": 1,
        "extraction_method": "transcript",
        "processing_status": "processed",
        "error_message": "",
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }
    (doc_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return "processed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR,
                         help="Directory containing one <id>/transcript.txt per interview.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH,
                         help="interviews_source database.json (for the interview id list and language metadata).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None,
                         help="Process only the first N interviews (for testing).")
    args = parser.parse_args()

    if not args.database.exists():
        raise SystemExit(f"Database not found: {args.database}")
    if not args.source_dir.exists():
        raise SystemExit(f"Source directory not found: {args.source_dir}")

    entries = load_entries(args.database)
    logger.info("Loaded %d interviews from %s", len(entries), args.database)
    if args.limit is not None:
        entries = entries[:args.limit]

    counts = {"processed": 0, "missing": 0}
    for entry in entries:
        status = process_one(entry, args.source_dir, args.output_dir)
        counts[status] += 1

    logger.info("Done. %s", counts)
    print(f"\nProcessed {len(entries)} interviews: {counts}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
