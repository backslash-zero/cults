"""Cross-corpus registry: one row per document, across every corpus.

Scans processed/<corpus>/ for every corpus subdirectory and derives its
status from that corpus's own existing manifests -- this is generated
output, not a hand-maintained ledger, so it can never drift out of sync
with the corpus-specific files it summarizes:

  - Stage 1 status: processed/<corpus>/corpus_manifest.csv, if present
    (corpora without a PDF-extraction stage, e.g. future interview or
    survey corpora fed some other way, simply have no Stage 1 row -- their
    documents show stage1_status "n/a").
  - Stage 2 status: processed/<corpus>/annotated_documents.txt (annotated)
    and processed/<corpus>/criterion_expressions.jsonl (embedded, with
    item counts) -- a document annotated but with zero relevant items is
    "annotated_only", not "embedded".

Usage (from thesis/corpus/):
    python -m thesis_corpus.build_registry
    python -m thesis_corpus.build_registry --processed-root <path>
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PROCESSED_ROOT = CORPUS_DIR / "processed"

REGISTRY_FIELDS = ["corpus", "document_id", "stage1_status", "stage2_status", "item_count", "output_dir"]

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("thesis_corpus.build_registry")


def read_stage1_statuses(corpus_dir: Path) -> dict[str, str]:
    manifest_path = corpus_dir / "corpus_manifest.csv"
    if not manifest_path.exists():
        return {}
    statuses = {}
    with open(manifest_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            statuses[row["document_id"]] = row["processing_status"]
    return statuses


def read_annotated_ids(corpus_dir: Path) -> set[str]:
    path = corpus_dir / "annotated_documents.txt"
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def read_item_counts(corpus_dir: Path) -> Counter:
    path = corpus_dir / "criterion_expressions.jsonl"
    counts = Counter()
    if not path.exists():
        return counts
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                counts[json.loads(line)["document_id"]] += 1
            except (json.JSONDecodeError, KeyError):
                continue
    return counts


def build_corpus_rows(corpus_name: str, corpus_dir: Path) -> list[dict]:
    stage1 = read_stage1_statuses(corpus_dir)
    annotated_ids = read_annotated_ids(corpus_dir)
    item_counts = read_item_counts(corpus_dir)

    documents_dir = corpus_dir / "documents"
    document_ids = set(stage1) | annotated_ids | set(item_counts)
    if documents_dir.exists():
        document_ids |= {d.name for d in documents_dir.iterdir() if d.is_dir()}

    rows = []
    for document_id in sorted(document_ids):
        count = item_counts.get(document_id, 0)
        if count > 0:
            stage2_status = "embedded"
        elif document_id in annotated_ids:
            stage2_status = "annotated_only"
        else:
            stage2_status = "not_started"

        rows.append({
            "corpus": corpus_name,
            "document_id": document_id,
            "stage1_status": stage1.get(document_id, "n/a"),
            "stage2_status": stage2_status,
            "item_count": count,
            "output_dir": f"{corpus_name}/documents/{document_id}",
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    args = parser.parse_args()

    if not args.processed_root.exists():
        raise SystemExit(f"Processed root not found: {args.processed_root}")

    all_rows = []
    for corpus_dir in sorted(args.processed_root.iterdir()):
        if not corpus_dir.is_dir():
            continue
        rows = build_corpus_rows(corpus_dir.name, corpus_dir)
        logger.info("%s: %d documents", corpus_dir.name, len(rows))
        all_rows.extend(rows)

    registry_path = args.processed_root / "registry.csv"
    with open(registry_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    logger.info("Total documents across %d corpora: %d", len({r['corpus'] for r in all_rows}), len(all_rows))
    print(f"Registry: {registry_path}")


if __name__ == "__main__":
    main()
