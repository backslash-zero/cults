"""Filters the manually-reviewed dictionaries/conceptnet_concepts_candidates.csv
down to just the rows kept on hand review (is_generic == "true" -- see
extract_conceptnet_concepts.py's row-write comment for the convention:
every row defaults to "true", hub/generic terms get hand-flipped to
"false" on review), writing dictionaries/conceptnet_concepts_kept.csv.

Run this on the Ollama-serving machine right before embedding, so
embed_concept_backbone.py only spends Ollama calls on the concepts
actually meant for the shared space, not the excluded hub words:

    python -m thesis_corpus.filter_conceptnet_concepts
    python -m thesis_corpus.embed_concept_backbone \\
        --input dictionaries/conceptnet_concepts_kept.csv \\
        --output dictionaries/conceptnet_concepts_embedded.jsonl
"""
from __future__ import annotations

import csv
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = CORPUS_DIR / "dictionaries" / "conceptnet_concepts_candidates.csv"
OUTPUT_PATH = CORPUS_DIR / "dictionaries" / "conceptnet_concepts_kept.csv"


def main() -> None:
    with open(INPUT_PATH, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["is_generic"] == "true"]

    if not rows:
        raise SystemExit(f"No rows with is_generic=='true' found in {INPUT_PATH}")

    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)} kept rows -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
