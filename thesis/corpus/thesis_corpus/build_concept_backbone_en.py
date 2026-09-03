"""Build a generic, low-bias English concept backbone from Open English WordNet.

Produces a list of abstract, high-centrality English concepts to serve as
neutral anchor points for structuring the shared embedding space across
this project's corpora (literature, MIVILUDES, interviews). Deliberately
seed-free: it does not start from any cult/sect-related term list. Instead
it walks the *entire* noun graph of Open English WordNet (oewn:2024, via
the `wn` library) and filters/ranks it structurally:

  1. Every noun synset is a candidate -- no topical starting points.
  2. Abstractness: kept only if its hypernym chain leads to the WordNet
     root synset `abstraction.n.06` ("abstraction, abstract entity"), as
     opposed to `physical_entity.n.01` or other roots. This is WordNet's
     own top-level entity/abstraction split, not a hand-picked distinction.
  3. Named-entity exclusion: dropped if the synset itself is an
     `instance_hyponym` of something (WordNet's mechanism for individuals
     like "Paris", as opposed to ordinary class hyponymy), or if its
     primary lemma is capitalized.
  4. Taxonomy exclusion: dropped if the synset's lexicographer file is
     `noun.animal` or `noun.plant` -- biological classification terms
     (e.g. "bird genus", "asterid dicot genus") pass the abstraction test
     structurally (a genus is a hyponym of `group`) but have very high
     hyponym counts purely from enumerating species, which dominates
     degree-based ranking without being a general concept.
  5. Centrality: node degree -- the total count of all WordNet relations
     (hypernym, hyponym, meronym, holonym, attribute, similar_to, ...)
     touching the synset.
  6. The top --target-size synsets by degree are kept, sorted descending.

Prerequisite: the oewn:2024 lexicon downloaded once via `wn`:
    python -c "import wn; wn.download('oewn:2024')"

Usage (from thesis/corpus/):
    python -m thesis_corpus.build_concept_backbone_en
    python -m thesis_corpus.build_concept_backbone_en --target-size 5000
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import wn

CORPUS_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = CORPUS_DIR / "dictionaries" / "concept_backbone_omw_en.csv"

LEXICON = "oewn:2024"
ABSTRACTION_ROOT_ID = "oewn-00002137-n"  # "abstraction, abstract entity"
TAXONOMY_LEXFILES = {"noun.animal", "noun.plant"}
SOURCE_LABEL = f"OMW ({LEXICON} via wn)"

FIELDNAMES = ["concept_id", "concept_en", "gloss_en", "source", "centrality_score"]

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("thesis_corpus.build_concept_backbone_en")


def is_abstract(synset: wn.Synset, ancestor_cache: dict[str, set[str]]) -> bool:
    """True if `synset`'s hypernym chain includes the abstraction root."""
    if synset.id == ABSTRACTION_ROOT_ID:
        return True
    return ABSTRACTION_ROOT_ID in ancestor_ids(synset, ancestor_cache)


def ancestor_ids(synset: wn.Synset, cache: dict[str, set[str]]) -> set[str]:
    """All hypernym-chain ancestor synset IDs, transitively, memoized."""
    sid = synset.id
    if sid in cache:
        return cache[sid]
    cache[sid] = set()  # cycle guard: seen-but-incomplete while recursing
    ancestors: set[str] = set()
    for parent in synset.hypernyms():
        ancestors.add(parent.id)
        ancestors |= ancestor_ids(parent, cache)
    cache[sid] = ancestors
    return ancestors


def is_named_entity(synset: wn.Synset) -> bool:
    if synset.get_related("instance_hypernym"):
        return True
    lemmas = synset.lemmas()
    return bool(lemmas) and lemmas[0][:1].isupper()


def is_taxonomy(synset: wn.Synset) -> bool:
    return synset.lexfile() in TAXONOMY_LEXFILES


def degree(synset: wn.Synset) -> int:
    return sum(len(related) for related in synset.relations().values())


def concept_id(synset: wn.Synset) -> str:
    return synset.ili if synset.ili else synset.id


def build_backbone(target_size: int) -> list[dict]:
    w = wn.Wordnet(LEXICON)
    nouns = w.synsets(pos="n")
    logger.info("Noun synsets inspected: %d", len(nouns))

    ancestor_cache: dict[str, set[str]] = {}
    candidates = []
    n_abstract = 0
    n_named_entity_excluded = 0
    n_taxonomy_excluded = 0
    for synset in nouns:
        if not is_abstract(synset, ancestor_cache):
            continue
        n_abstract += 1
        if is_named_entity(synset):
            n_named_entity_excluded += 1
            continue
        if is_taxonomy(synset):
            n_taxonomy_excluded += 1
            continue
        candidates.append(synset)

    logger.info("Passing abstractness filter: %d", n_abstract)
    logger.info("Excluded as named entities: %d", n_named_entity_excluded)
    logger.info("Excluded as biological taxonomy: %d", n_taxonomy_excluded)
    logger.info("Passing all filters: %d", len(candidates))

    ranked = sorted(candidates, key=degree, reverse=True)
    selected = ranked[:target_size]
    logger.info("Selected (target size %d): %d", target_size, len(selected))

    rows = []
    seen_ids = set()
    for synset in selected:
        cid = concept_id(synset)
        if cid in seen_ids:
            continue
        seen_ids.add(cid)
        lemmas = synset.lemmas()
        rows.append({
            "concept_id": cid,
            "concept_en": lemmas[0] if lemmas else "",
            "gloss_en": synset.definition() or "",
            "source": SOURCE_LABEL,
            "centrality_score": degree(synset),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target-size", type=int, default=3000,
                         help="Number of top-ranked concepts to keep (default: 3000).")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH,
                         help="Output CSV path (default: dictionaries/concept_backbone_omw_en.csv).")
    args = parser.parse_args()

    if LEXICON not in {f"{lex.id}:{lex.version}" for lex in wn.lexicons()}:
        raise SystemExit(
            f"Lexicon {LEXICON} not found locally. Download it first:\n"
            f'    python -c "import wn; wn.download(\'{LEXICON}\')"'
        )

    rows = build_backbone(args.target_size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Done. %d concepts written -> %s", len(rows), args.output)


if __name__ == "__main__":
    main()
