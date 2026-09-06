"""Second pass over the cached ConceptNet dump (no re-download): for every
term that surfaced as a candidate in fetch_conceptnet_related.py's output,
counts its TOTAL connectivity across all of English ConceptNet -- not just
to our 1,500 structural_concepts seeds.

Why this exists: ranking candidates purely by "how strongly/broadly they
connect to our seeds" floods the result with generic hub words ("change",
"person", "activity", "be", "act") that loosely connect to almost
anything in ConceptNet, seed or not -- confirmed empirically (raising the
per-edge weight threshold to 2.0 didn't fix it; the same hub words stayed
on top). This is the standard IDF-style correction for that: a term whose
connections to our seeds make up a large *fraction* of its total
ConceptNet connectivity is genuinely specific to this domain; a term
connected to everything gets penalized regardless of its raw seed-edge
count. See extract_conceptnet_concepts.py for how this degree figure is
actually used (specificity = seed_weight / total_degree).

Usage (from thesis/corpus/):
    python -m thesis_corpus.compute_conceptnet_degree
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
from collections import defaultdict
from pathlib import Path

from thesis_corpus.fetch_conceptnet_related import DUMP_PATH, concept_term

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("thesis_corpus.compute_conceptnet_degree")

CORPUS_DIR = Path(__file__).resolve().parent.parent
RELATED_RAW_PATH = CORPUS_DIR / "dictionaries" / "conceptnet_related_raw.jsonl"
OUTPUT_PATH = CORPUS_DIR / "dictionaries" / "conceptnet_term_degree.jsonl"

# Same relation exclusions as extract_conceptnet_concepts.py -- degree must
# be measured on the same "meaningful relations" basis as the seed-edge
# weight it's being compared against, or the specificity ratio isn't
# apples-to-apples.
EXCLUDED_RELATIONS = {
    "RelatedTo", "HasContext", "DerivedFrom", "FormOf",
    "EtymologicallyRelatedTo", "EtymologicallyDerivedFrom",
    "Antonym", "DistinctFrom", "NotDesires", "NotHasProperty",
    "NotCapableOf", "ExternalURL",
}


def load_tracked_terms(path: Path) -> set[str]:
    terms = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                terms.add(json.loads(line)["related"])
    return terms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--related-raw", type=Path, default=RELATED_RAW_PATH)
    parser.add_argument("--dump-path", type=Path, default=DUMP_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    if not args.dump_path.exists():
        raise SystemExit(f"Missing {args.dump_path} -- run fetch_conceptnet_related.py first.")

    tracked_terms = load_tracked_terms(args.related_raw)
    logger.info("%d candidate terms to compute total ConceptNet degree for", len(tracked_terms))

    degree: dict[str, dict] = defaultdict(lambda: {"total_weight": 0.0, "total_edges": 0})
    n_lines = 0
    with gzip.open(args.dump_path, "rt", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            n_lines += 1
            if n_lines % 5_000_000 == 0:
                logger.info("  ... %dM lines scanned", n_lines // 1_000_000)
            if len(row) < 5:
                continue
            _uri, relation, start_uri, end_uri, metadata_json = row[:5]
            relation = relation.removeprefix("/r/")
            if relation in EXCLUDED_RELATIONS:
                continue
            start_term = concept_term(start_uri)
            end_term = concept_term(end_uri)
            if start_term is None or end_term is None:
                continue

            try:
                weight = json.loads(metadata_json).get("weight", 1.0)
            except (json.JSONDecodeError, AttributeError):
                weight = 1.0

            for term in (start_term, end_term):
                if term in tracked_terms:
                    entry = degree[term]
                    entry["total_weight"] += weight
                    entry["total_edges"] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for term, info in degree.items():
            f.write(json.dumps({"term": term, **info}, ensure_ascii=False) + "\n")

    print(f"\nDone. {n_lines} lines scanned, degree computed for {len(degree)} terms -> {args.output}")


if __name__ == "__main__":
    main()
