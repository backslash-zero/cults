"""Stage 2 of the ConceptNet-expanded reference set: turns
fetch_conceptnet_related.py's raw edge list into a ranked candidate list,
the same shape as extract_structural_concepts.py's own output, for
embedding and pooling as a fourth (all-reference) point-set.

This set is seeded from -- but distinct from -- `structural_concepts`:
it's the neighborhood ConceptNet associates with those 1,500 corpus-derived
words, not a repeat of the words themselves (already-present terms are
excluded, see EXISTING_TERMS below) and not restricted to exactly what the
corpus happens to say. It stays thematically anchored to the same
social/relational domain as its seeds (so it is NOT topic-neutral the way
`concept_backbone` is -- that one is chosen with zero reference to this
thesis's subject matter at all) -- this is a smoother, less
frequency-biased *generalization* of `structural_concepts`, not a second
independent yardstick.

Ranking: tried three things, in order, before landing here -- worth
recording since none of them cleanly separated domain-specific concepts
from generic ones and this candidate list is NOT assumed clean as a
result (see "Known limitations" below):
  1. Raw seed-connectivity (sum of edge weight to any seed): floods the
     top with hub words ("change", "person", "activity", "be", "make")
     that loosely connect to almost everything in ConceptNet.
  2. Specificity alone (seed-weight / total ConceptNet degree, from
     compute_conceptnet_degree.py): fixes the hub-word problem but
     over-corrects into small-sample noise -- a term with only 5-12 total
     ConceptNet edges needs just 3 of them to be seeds to hit a "perfect"
     specificity=1.0, without being remotely central to the domain.
  3. **n_seed_connections x specificity (what's actually used)**: rewards
     terms that are both reasonably broadly connected to our seeds AND
     disproportionately so relative to their overall ConceptNet degree.
     Better than either alone, but a handful of hub words (e.g. "activity",
     "be", "make") still rank highly enough to need hand-pruning -- same
     situation `structural_concepts` was already in with its own
     frequency-based ranking, not a new problem this set introduces.

Filtering (per the resolved design): a ConceptNet-surfaced candidate is
kept only if it's ALSO a valid Open English WordNet lemma in the same
in-domain lexicographer files `structural_concepts` already restricts to
(reusing that script's own INCLUDE_LEXFILES/domain_synset logic directly,
not a reimplementation) -- this filters out ConceptNet's often noisier
crowd-sourced edges the same way it already filters proper nouns and
off-domain senses for structural_concepts. A gloss-pattern sweep also
catches WordNet's own incomplete instance-marking for minor historical
figures (the same gap structural_concepts hit for "hubbard"/"iskcon" --
found again here for "town", whose first noun sense is genuinely a
19th-century architect with no instance_hyponym relation recorded).

Usage (from thesis/corpus/):
    python -m thesis_corpus.extract_conceptnet_concepts
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import wn

from thesis_corpus.extract_structural_concepts import (
    EXCLUDE_WORDS, INCLUDE_LEXFILES, WORDNET_LEXICON, domain_synset,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("thesis_corpus.extract_conceptnet_concepts")

CORPUS_DIR = Path(__file__).resolve().parent.parent
RELATED_RAW_PATH = CORPUS_DIR / "dictionaries" / "conceptnet_related_raw.jsonl"
DEGREE_PATH = CORPUS_DIR / "dictionaries" / "conceptnet_term_degree.jsonl"
STRUCTURAL_CANDIDATES_PATH = CORPUS_DIR / "dictionaries" / "structural_concepts_candidates.csv"
CONCEPT_BACKBONE_PATH = CORPUS_DIR / "dictionaries" / "concept_backbone_embedded.jsonl"
OUTPUT_PATH = CORPUS_DIR / "dictionaries" / "conceptnet_concepts_candidates.csv"

TARGET_SIZE = 1500
# Ranking by raw seed-connectivity alone floods the result with hub words
# ("change", "person", "activity", "be") loosely connected to almost
# everything in ConceptNet -- confirmed empirically, a stricter per-edge
# weight threshold didn't fix it either. Ranking by SPECIFICITY instead
# (what fraction of a term's *total* ConceptNet connectivity is accounted
# for by our seeds -- see compute_conceptnet_degree.py) fixes the actual
# cause rather than a symptom. A minimum connection count guards against a
# single fluke edge producing a perfect but meaningless specificity=1.0.
MIN_SEED_CONNECTIONS = 3
MIN_TOKEN_LEN = 3

# ConceptNet's relations span from precise/structural to loose/co-occurrence.
# Checked directly: RelatedTo alone is 63% of all matched edges (299,043 of
# 475,700) at the lowest average weight of any major relation (0.91) --
# it's ConceptNet's catch-all "these two things appeared together
# somewhere" relation, and ranking by it floods the result with generic
# hub words ("make", "have", "big", "like") that connect to nearly
# anything, not concepts specific to this domain. HasContext (topical
# co-occurrence tagging), DerivedFrom/FormOf/EtymologicallyRelatedTo
# (morphological, not conceptual) are excluded for the same reason.
# Antonym/DistinctFrom/NotDesires are excluded because they mean the
# opposite of "related" -- including them would pull in oppositional
# concepts under a "relatedness" ranking.
EXCLUDED_RELATIONS = {
    "RelatedTo", "HasContext", "DerivedFrom", "FormOf",
    "EtymologicallyRelatedTo", "EtymologicallyDerivedFrom",
    "Antonym", "DistinctFrom", "NotDesires", "NotHasProperty",
    "NotCapableOf", "ExternalURL",
}

# Same stopword rationale as extract_structural_concepts.py: WordNet
# contains entries for several of these, so membership alone isn't enough.
STOPWORDS = {
    "the", "a", "an", "of", "in", "to", "and", "or", "is", "was", "were", "be", "been", "being",
    "that", "this", "it", "its", "one", "two", "first", "new", "old", "way", "thing", "something",
}

# domain_synset() already excludes a synset with an instance_hyponym
# relation, but WordNet's own instance-marking turns out to be incomplete
# for minor historical figures (documented in extract_structural_concepts.py
# after "hubbard"/"iskcon" leaked through the same way) -- confirmed here
# directly: "town"'s first noun.person synset is "United States architect
# who was noted for his design and construction of truss bridges
# (1784-1844)" with an *empty* instance_hyponym/instance_hypernym list, so
# the existing check lets it straight through. A gloss-pattern sweep for
# the recurring WordNet biographical-entry template catches what the
# relation-based check misses.
_BIOGRAPHICAL_GLOSS_RE = re.compile(
    r"\(\d{4}\s*-\s*\d{0,4}\)|"
    r"\b(united states|american|english|british|french|german|italian|"
    r"russian|scottish|irish|canadian)\b\s+\w+\s+who\b",
    re.IGNORECASE,
)


def load_existing_terms() -> set[str]:
    """Every term already present in structural_concepts or concept_backbone
    -- excluded here so this set is additive, not a repeat."""
    terms = set()
    with open(STRUCTURAL_CANDIDATES_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            terms.add(row["concept_en"].strip().lower())
    with open(CONCEPT_BACKBONE_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                terms.add(json.loads(line)["concept_en"].strip().lower())
    return terms


def load_degree(path: Path) -> dict[str, dict]:
    degree = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                degree[d["term"]] = d
    return degree


def aggregate_edges(path: Path) -> dict[str, dict]:
    """related_term -> {total_weight, seeds: set, relations: set, n_edges}."""
    agg: dict[str, dict] = defaultdict(lambda: {"total_weight": 0.0, "seeds": set(), "relations": set(), "n_edges": 0})
    n_skipped_relation = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            edge = json.loads(line)
            if edge["relation"] in EXCLUDED_RELATIONS:
                n_skipped_relation += 1
                continue
            entry = agg[edge["related"]]
            entry["total_weight"] += edge["weight"]
            entry["seeds"].add(edge["seed"])
            entry["relations"].add(edge["relation"])
            entry["n_edges"] += 1
    logger.info("Skipped %d edges with an excluded (loose/morphological) relation type", n_skipped_relation)
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=RELATED_RAW_PATH)
    parser.add_argument("--degree", type=Path, default=DEGREE_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--target-size", type=int, default=TARGET_SIZE)
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Missing {args.input} -- run fetch_conceptnet_related.py first.")
    if not args.degree.exists():
        raise SystemExit(f"Missing {args.degree} -- run compute_conceptnet_degree.py first.")

    existing_terms = load_existing_terms()
    logger.info("%d terms already in structural_concepts/concept_backbone (excluded here)", len(existing_terms))

    logger.info("Aggregating %s ...", args.input)
    agg = aggregate_edges(args.input)
    logger.info("%d unique related terms surfaced by ConceptNet", len(agg))

    logger.info("Loading total ConceptNet degree from %s ...", args.degree)
    degree = load_degree(args.degree)

    # specificity = what fraction of a term's TOTAL ConceptNet connectivity
    # is accounted for by our seeds -- always in (0, 1] since seed-weight is
    # a subset of total weight. A term connected to everything in ConceptNet
    # scores low even with a high raw seed-weight; a term whose connections
    # are disproportionately to our seeds scores high regardless of its
    # absolute popularity.
    scored = []
    for term, info in agg.items():
        n_seeds = len(info["seeds"])
        if n_seeds < MIN_SEED_CONNECTIONS:
            continue
        total_degree = degree.get(term, {}).get("total_weight", info["total_weight"])
        specificity = info["total_weight"] / total_degree if total_degree > 0 else 0.0
        combined_score = n_seeds * specificity
        scored.append((term, info, specificity, total_degree, combined_score))

    ranked = sorted(scored, key=lambda t: -t[4])

    en = wn.Wordnet(WORDNET_LEXICON)
    candidates = []
    n_excluded_existing, n_excluded_domain, n_excluded_stop, n_excluded_biographical = 0, 0, 0, 0

    for term, info, specificity, total_degree, combined_score in ranked:
        term_lower = term.strip().lower()
        if term_lower in STOPWORDS or term_lower in EXCLUDE_WORDS or len(term_lower) < MIN_TOKEN_LEN:
            n_excluded_stop += 1
            continue
        if term_lower in existing_terms:
            n_excluded_existing += 1
            continue
        synset = domain_synset(en, term_lower)
        if synset is None:
            n_excluded_domain += 1
            continue
        if _BIOGRAPHICAL_GLOSS_RE.search(synset.definition()):
            n_excluded_biographical += 1
            continue
        candidates.append({
            "concept_en": term_lower,
            "gloss_en": synset.definition(),
            "combined_score": round(combined_score, 4),
            "specificity": round(specificity, 4),
            "total_weight": round(info["total_weight"], 3),
            "total_degree": round(total_degree, 3),
            "n_seed_connections": len(info["seeds"]),
            "n_edges": info["n_edges"],
            "seed_terms": ";".join(sorted(info["seeds"])[:10]),
            "relations": ";".join(sorted(info["relations"])),
        })
        if len(candidates) >= args.target_size:
            break

    logger.info(
        "%d candidates kept, ranked by seed-specificity (in-domain WordNet lexfile match; excluded "
        "%d already-existing terms, %d stopwords/too-short, %d without an in-domain WordNet sense, "
        "%d biographical-gloss leaks)",
        len(candidates), n_excluded_existing, n_excluded_stop, n_excluded_domain, n_excluded_biographical,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "concept_id", "concept_en", "gloss_en", "combined_score", "specificity", "total_weight",
        "total_degree", "n_seed_connections", "n_edges", "seed_terms", "relations", "is_generic",
    ]
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, c in enumerate(candidates, 1):
            writer.writerow({
                "concept_id": f"cn_{i:04d}",
                "concept_en": c["concept_en"],
                "gloss_en": c["gloss_en"],
                "combined_score": c["combined_score"],
                "specificity": c["specificity"],
                "total_weight": c["total_weight"],
                "total_degree": c["total_degree"],
                "n_seed_connections": c["n_seed_connections"],
                "n_edges": c["n_edges"],
                "seed_terms": c["seed_terms"],
                "relations": c["relations"],
                # NOT yet manually reviewed -- same convention as
                # structural_concepts_candidates.csv: every row defaults to
                # "true" (already domain-filtered), meant to be flipped to
                # "false" per-row on hand review, hub words like "activity"/
                # "be"/"make" expected among the ones needing a flip.
                "is_generic": "true",
            })

    print(f"\nDone. {len(candidates)} candidates -> {args.output}")
    print("NOT yet manually reviewed -- expect hub words near the top needing is_generic=false on review.")


if __name__ == "__main__":
    main()
