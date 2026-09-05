"""Build one shared, cross-corpus embedding space (PCA on a pooled matrix).

Every prior embedding step (extract_and_embed's three corpora,
embed_miviludes_criteria, embed_concept_backbone) produces vectors in the
same raw 1024-d bge-m3 space, but reduce_embeddings.py's downsampling fits
an INDEPENDENT PCA per corpus -- literature's own 100-d space, MIVILUDES's
own, interviews' own. Those are unrelated coordinate systems: a point's
position in one is not comparable to a point's position in another. This
script instead pools every vector from every dataset, standardizes, and
fits ONE PCA on the pooled matrix -- so every item, from every dataset,
ends up in the same shared coordinate system, comparable to every other.

This is what operationalizes the concept backbone's stated purpose (see
Methods.tex, "A Generic Concept Backbone"): it is not a static reference
list sitting outside the analysis, but an active participant in fitting
this shared space, alongside every corpus item and the MIVILUDES criteria.

Every point additionally carries a `point_role`, distinguishing three kinds
of thing this space pools together (see Methods.tex, "A Shared Cross-Corpus
Space"):
  - `"expression"`: a criterion expression extracted from a text (the three
    corpora) or a MIVILUDES criterion -- something a source actually said.
  - `"reference"`: a backdrop vocabulary point, not itself a claim any
    source makes. Two distinct subsets share this role, deliberately kept
    separate rather than merged into one, since they buy different things:
    `concept_backbone` (WordNet, topic-neutral -- not derived from any
    corpus, a fixed independent yardstick) and `structural_concepts`
    (corpus-derived generic/structural vocabulary -- extracted from the
    corpora's own expression text, geometrically closer to the data by
    construction, but not topic-neutral: it exists *because* the corpora
    use this vocabulary). Use `concept_backbone` when independence from the
    corpus matters; use `structural_concepts` when proximity/interpretive
    relevance matters more than neutrality.
  - `"emergent"`: a named entity/group/concept mentioned BY the corpora
    themselves (emergent entities, below) -- corpus-derived like an
    expression point, but a recurring reference object rather than a claim
    about one.

Pooling (seven source_dataset values; total point count is logged at
runtime, not asserted against a hardcoded constant -- it changes whenever
the corpus grows or the emergent-entity threshold below is adjusted):
  - Each corpus item (literature/miviludes/interviews) contributes ONE
    point (`point_role="expression"`): its embedding_vector, plus its
    claim_mode/epistemic_status/attribution tags carried through unchanged
    for later faceting -- except exact-duplicate expressions within the
    same document (keeping the first occurrence) and expressions under
    MIN_EXPRESSION_WORDS words, both filtered here at pooling time rather
    than in the archive (see load_corpus_points). Interview items
    additionally carry response_rank:
    interviews open with a free-listing prompt ("what comes to mind when
    you hear the word cult?"), and order of mention is a standard
    cognitive-salience proxy in prototype theory (first-mentioned = most
    prototypical) -- this is the position of the item within its own
    document, in the order the archive already lists them (1-indexed).
  - Each MIVILUDES criterion contributes ONE point (`point_role="expression"`):
    its French embedding (the official original). The English translation is
    kept only as a display label (`label_en`) on the same point, not
    embedded separately -- translation fidelity is instead checked directly,
    once, via raw-space cosine similarity between the French and English
    embeddings (see `check_miviludes_translation_fidelity`), rather than by
    spending a second near-duplicate point in the shared analytical space.
  - Each concept-backbone entry contributes ONE point
    (`point_role="reference"`, `source_dataset="concept_backbone"`): its
    embedding_vector.
  - Each structural-concept entry (see extract_structural_concepts.py)
    contributes ONE point (`point_role="reference"`,
    `source_dataset="structural_concepts"`): its embedding_vector, plus
    `mention_distribution` (per-corpus mention counts, same provenance
    role as on emergent-entity points) reconstructed from the CSV's
    mention-count columns.
  - Each emergent entity mentioned at least `--entity-anchor-min-mentions`
    times across all three corpora contributes ONE point
    (`point_role="emergent"`): a per-unique (normalized) entity-anchor
    embedding already computed in Stage 2 (`entity_anchor_vectors`), never
    before pooled into any space. Gives named entities/dimensions (e.g.
    "Scientology", "charismatic leader") an actual position relative to
    corpus expressions and the concept backbone. Carries
    `mention_distribution` (per-corpus mention counts) as provenance
    metadata, not used in the PCA fit.

Preprocessing: StandardScaler (zero mean, unit variance per dimension)
before PCA. Every vector already comes from the same embedding model, but
the sources differ a lot in register (academic prose, government French,
casual interview speech, bare word+gloss dictionary entries) and could
plausibly carry different per-dimension distributions -- this is a
defensive measure against any one dataset dominating the fit purely due to
scale, not a claim that such an imbalance is known to exist.

Dimensionality is not a fixed constant: PCA is first fit at full rank to
get the complete explained-variance curve (saved as a diagnostic, not just
asserted), and the smallest k reaching 95% cumulative variance is chosen
from that curve.

This needs no Ollama -- only numpy/scikit-learn/matplotlib on data that's
already local. Never modifies any of the source files; only ever writes new
files under processed/.

Usage (from thesis/corpus/):
    python -m thesis_corpus.build_shared_space
    python -m thesis_corpus.build_shared_space --entity-anchor-min-mentions 5
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

CORPUS_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = CORPUS_DIR / "processed"
# All cross-corpus (not per-corpus) outputs live under their own
# processed/shared_space/ subdirectory, as a sibling of processed/<corpus>/
# -- keeps "one corpus's own pipeline output" and "output that spans every
# corpus" visually and structurally distinct, and gives any later step
# (e.g. a further 2D/3D reduction for visualization) an obvious home next
# to the space it would be reducing.
SHARED_SPACE_DIR = PROCESSED_DIR / "shared_space"

CORPUS_ARCHIVES = {
    "literature": PROCESSED_DIR / "literature" / "criterion_expressions.jsonl",
    "miviludes": PROCESSED_DIR / "miviludes" / "criterion_expressions.jsonl",
    "interviews": PROCESSED_DIR / "interviews" / "criterion_expressions.jsonl",
}
MIVILUDES_CRITERIA_PATH = CORPUS_DIR / "metadata" / "miviludes_criteria_embedded.jsonl"
CONCEPT_BACKBONE_PATH = CORPUS_DIR / "dictionaries" / "concept_backbone_embedded.jsonl"
# Produced by extract_structural_concepts.py + embedding on the Windows/Ollama
# machine (see thesis_corpus/README.md) -- python -m thesis_corpus.embed_concept_backbone
# --input dictionaries/structural_concepts_candidates.csv --output <this path>
# (the existing embed_concept_backbone.py is fully generic over its
# --input/--output CSV, no separate embed script needed).
STRUCTURAL_CONCEPTS_PATH = CORPUS_DIR / "dictionaries" / "structural_concepts_embedded.jsonl"

OUTPUT_PATH = SHARED_SPACE_DIR / "embedding_space.jsonl"
VARIANCE_CSV_PATH = SHARED_SPACE_DIR / "variance_curve.csv"
VARIANCE_PLOT_PATH = SHARED_SPACE_DIR / "variance_curve.png"
VARIANCE_JSON_PATH = SHARED_SPACE_DIR / "variance_curve.json"

VARIANCE_THRESHOLD = 0.95
EMBEDDING_DIM = 1024
ENTITY_ANCHOR_MIN_MENTIONS = 3
# Below this, a FR/EN pair is treated as a likely mistranslation worth
# inspecting by hand. NOT 0.90: the MIVILUDES criteria check's own first
# run flagged crit-legal-disputes at 0.50 under a flat <0.90 rule, and
# hand inspection found it was actually an accurate translation -- short
# official phrases just embed less stably cross-lingually than length
# alone would suggest. 0.70 is calibrated to catch genuine mistranslations
# without flagging that kind of expected short-phrase noise.
COSINE_FLAG_THRESHOLD = 0.70
# Expressions shorter than this (naive whitespace word count) are dropped
# when pooling -- see MIN_EXPRESSION_WORDS' use in load_corpus_points.
MIN_EXPRESSION_WORDS = 5

_WHITESPACE_RE = re.compile(r"\s+")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("thesis_corpus.build_shared_space")


def normalize_anchor(anchor: str) -> str:
    """Lowercase, strip, and collapse internal whitespace runs to one space
    -- so "Charismatic  Leader" and "charismatic leader" merge to the same
    entity rather than becoming two near-duplicate points."""
    return _WHITESPACE_RE.sub(" ", anchor.strip().lower())


def load_corpus_points(corpus_name: str, path: Path) -> tuple[list[dict], dict[str, int]]:
    """Two known-and-deferred extraction issues (documented in Methods.tex,
    "Vectorising Scholarly Work") get filtered here, at pooling time, rather
    than by mutating the archive: the LLM occasionally emits the same
    expression twice within one chunk's response, and occasionally emits a
    short, low-content phrase -- already captured as an entity anchor on a
    fuller expression -- as though it were an independent expression.
    Filtering the pooled space (not the archive) matches this codebase's
    "never modify the source, only write new files" convention -- the
    archive stays the full, traceable record of what the LLM actually
    produced, flaws included.

    response_rank is computed BEFORE filtering, over every item in original
    archive order, so a dropped item doesn't shift the rank of items after
    it -- "3rd mentioned" should reflect what was actually said 3rd, not a
    renumbering among whatever survives the filter.

    Returns (points, removal_counts) -- removal_counts is
    {"duplicates": N, "short_fragments": M}, for the "X duplicates removed,
    Y short fragments removed" log line and Methods.tex documentation."""
    points = []
    response_rank_by_document: dict[str, int] = defaultdict(int)
    for_interviews = corpus_name == "interviews"
    seen_text_by_document: dict[str, set[str]] = defaultdict(set)
    duplicates_removed = 0
    short_fragments_removed = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            document_id = item["document_id"]
            text = item["embedding_text"]

            response_rank = None
            if for_interviews:
                response_rank_by_document[document_id] += 1
                response_rank = response_rank_by_document[document_id]

            if text in seen_text_by_document[document_id]:
                duplicates_removed += 1
                continue
            seen_text_by_document[document_id].add(text)

            if len(text.split()) < MIN_EXPRESSION_WORDS:
                short_fragments_removed += 1
                continue

            points.append({
                "source_dataset": corpus_name,
                "point_role": "expression",
                "key": f"{document_id}:{item['chunk_index']}",
                "label": text,
                "attribution": item.get("attribution"),
                "claim_mode": item.get("claim_mode"),
                "epistemic_status": item.get("epistemic_status"),
                "response_rank": response_rank,
                "vector": item["embedding_vector"],
            })
    return points, {"duplicates": duplicates_removed, "short_fragments": short_fragments_removed}


def _report_translation_fidelity(label: str, similarities: list[tuple[str, float]]) -> None:
    """Shared reporting for both the 17-criteria and (once wired in) the
    914-expression translation-fidelity checks: full distribution, not
    just mean/min -- a flat low-percentile number is expected for short
    phrases (see COSINE_FLAG_THRESHOLD) and shouldn't itself read as a
    problem. Only pairs below COSINE_FLAG_THRESHOLD are named individually,
    as candidates for manual inspection, not as confirmed errors."""
    values = np.array([c for _, c in similarities], dtype=np.float64)
    print(f"\n{label} FR vs EN raw-embedding cosine similarity "
          f"(diagnostic; only FR is pooled into the shared space):")
    print(f"  n={len(values)}  mean={values.mean():.4f}  median={np.median(values):.4f}  "
          f"p10={np.percentile(values, 10):.4f}  min={values.min():.4f}")
    flagged = sorted((k, c) for k, c in similarities if c < COSINE_FLAG_THRESHOLD)
    if flagged:
        print(f"  {len(flagged)} pair(s) below {COSINE_FLAG_THRESHOLD:.2f} -- candidates for manual "
              f"inspection (not automatically errors; short phrases embed noisily cross-lingually):")
        for key, cosine in flagged:
            print(f"    {key:<40} cosine={cosine:.4f}")
        logger.warning(
            "%d %s FR/EN pair(s) below the %.2f flag threshold -- inspect by hand before treating as an error.",
            len(flagged), label, COSINE_FLAG_THRESHOLD,
        )


def check_miviludes_translation_fidelity(path: Path) -> None:
    """Diagnostic only: cosine similarity between each criterion's raw
    (pre-PCA) French and English embeddings. Only the French embedding is
    pooled into the shared space (see load_miviludes_criteria_points) --
    this replaces the old approach of pooling both and comparing their
    shared-space distance from the origin, which spent a second
    near-duplicate point on a check obtainable directly from the raw
    vectors."""
    similarities = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            fr = np.array(item["embedding_vector_fr"], dtype=np.float64)
            en = np.array(item["embedding_vector_en"], dtype=np.float64)
            cosine = float(np.dot(fr, en) / (np.linalg.norm(fr) * np.linalg.norm(en)))
            similarities.append((item["id"], cosine))

    _report_translation_fidelity("MIVILUDES criteria", similarities)


def load_miviludes_criteria_points(path: Path) -> list[dict]:
    points = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            points.append({
                "source_dataset": "miviludes_criteria",
                "point_role": "expression",
                "key": item["id"],
                "label": item["criterion_fr"],
                "label_en": item["criterion_en"],
                "vector": item["embedding_vector_fr"],
            })
    return points


def load_concept_backbone_points(path: Path) -> list[dict]:
    points = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            points.append({
                "source_dataset": "concept_backbone",
                "point_role": "reference",
                "key": item["concept_id"],
                "label": item["concept_en"],
                "vector": item["embedding_vector"],
            })
    return points


def load_structural_concepts_points(path: Path) -> list[dict]:
    """Reads structural_concepts_embedded.jsonl (extract_structural_concepts.py's
    candidates CSV, embedded via the existing embed_concept_backbone.py --
    same "<term>: <gloss>" format, just a different input/output path). Also
    a `point_role="reference"` point-set, but corpus-derived rather than
    topic-neutral -- see the module docstring's "reference" bullet for the
    distinction from `concept_backbone`."""
    points = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            points.append({
                "source_dataset": "structural_concepts",
                "point_role": "reference",
                "key": item["concept_id"],
                "label": item["concept_en"],
                "mention_distribution": {
                    "literature": int(item.get("literature_mentions", 0)),
                    "miviludes": int(item.get("miviludes_mentions", 0)),
                    "interviews": int(item.get("interviews_mentions", 0)),
                },
                "vector": item["embedding_vector"],
            })
    return points


def load_emergent_entities(archive_paths: dict[str, Path], min_mentions: int) -> list[dict]:
    """Pools one point per unique (normalized) entity anchor mentioned at
    least `min_mentions` times across all three corpus archives, using the
    per-anchor embedding Stage 2 already computed (entity_anchor_vectors) --
    never before pooled into any space. First-seen vector per normalized
    anchor is kept (the embedding is a function of the literal anchor text
    alone, so occurrences of the same normalized string carry equivalent
    vectors modulo casing/whitespace, already normalized away here).

    These are "emergent entities" (`point_role="emergent"`): named
    entities/groups/concepts mentioned BY the corpora themselves, as
    distinct from an "expression" point (a claim a source makes) or a
    "reference" point (the corpus-independent concept backbone).

    Each point also carries `mention_distribution`: a per-corpus mention
    count (e.g. {"literature": 820, "miviludes": 15, "interviews": 12}) --
    provenance metadata only, not used in the PCA fit, so an anchor
    overwhelmingly mentioned in one corpus can be told apart from one
    mentioned evenly across all three."""
    vector_by_anchor: dict[str, list[float]] = {}
    mentions_by_corpus: dict[str, Counter[str]] = defaultdict(Counter)

    for corpus_name, path in archive_paths.items():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                raw_vectors = item.get("entity_anchor_vectors") or {}
                for raw_anchor, vector in raw_vectors.items():
                    key = normalize_anchor(raw_anchor)
                    if not key:
                        continue
                    mentions_by_corpus[corpus_name][key] += 1
                    vector_by_anchor.setdefault(key, vector)

    total_mentions: Counter[str] = Counter()
    for corpus_counts in mentions_by_corpus.values():
        total_mentions.update(corpus_counts)

    def distribution_for(anchor: str) -> dict[str, int]:
        return {corpus_name: mentions_by_corpus[corpus_name].get(anchor, 0) for corpus_name in archive_paths}

    print(f"\nEmergent entities: {len(total_mentions)} unique (normalized) across all corpora; "
          f"top 20 by mention count:")
    for anchor, count in total_mentions.most_common(20):
        print(f"  {anchor}: {count} total mentions -- {distribution_for(anchor)}")

    points = []
    for anchor, count in total_mentions.items():
        if count < min_mentions:
            continue
        points.append({
            "source_dataset": "emergent_entities",
            "point_role": "emergent",
            "key": anchor,
            "label": anchor,
            "mention_distribution": distribution_for(anchor),
            "vector": vector_by_anchor[anchor],
        })
    return points


def sanity_checks(points: list[dict], coords: np.ndarray) -> None:
    norms_by_dataset = defaultdict(list)
    for p, c in zip(points, coords):
        norms_by_dataset[p["source_dataset"]].append(float(np.linalg.norm(c)))

    print("\nMean shared-space vector norm by source_dataset (diagnostic, not a pass/fail check):")
    for dataset, norms in sorted(norms_by_dataset.items()):
        print(f"  {dataset:<24} n={len(norms):<6} mean_norm={sum(norms)/len(norms):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variance-threshold", type=float, default=VARIANCE_THRESHOLD)
    parser.add_argument("--entity-anchor-min-mentions", type=int, default=ENTITY_ANCHOR_MIN_MENTIONS,
                         help="Minimum times an entity anchor must be mentioned across all corpora to get its own point.")
    args = parser.parse_args()

    points: list[dict] = []
    counts: dict[str, int] = {}
    removed: dict[str, dict[str, int]] = {}

    for corpus_name, path in CORPUS_ARCHIVES.items():
        if not path.exists():
            raise SystemExit(f"Missing corpus archive: {path}")
        new_points, removal_counts = load_corpus_points(corpus_name, path)
        counts[corpus_name] = len(new_points)
        removed[corpus_name] = removal_counts
        points.extend(new_points)

    logger.info("Filtered during pooling (duplicates / short fragments, per corpus): %s", removed)
    total_duplicates = sum(r["duplicates"] for r in removed.values())
    total_short = sum(r["short_fragments"] for r in removed.values())
    print(f"\nFiltered during pooling: {total_duplicates} exact-duplicate expressions removed "
          f"(per document, keeping first occurrence), {total_short} short fragments removed "
          f"(under {MIN_EXPRESSION_WORDS} words). Per corpus: {removed}")

    if not MIVILUDES_CRITERIA_PATH.exists():
        raise SystemExit(f"Missing: {MIVILUDES_CRITERIA_PATH}")
    check_miviludes_translation_fidelity(MIVILUDES_CRITERIA_PATH)
    criteria_points = load_miviludes_criteria_points(MIVILUDES_CRITERIA_PATH)
    counts["miviludes_criteria"] = len(criteria_points)
    points.extend(criteria_points)

    if not CONCEPT_BACKBONE_PATH.exists():
        raise SystemExit(f"Missing: {CONCEPT_BACKBONE_PATH}")
    concept_points = load_concept_backbone_points(CONCEPT_BACKBONE_PATH)
    counts["concept_backbone"] = len(concept_points)
    points.extend(concept_points)

    if not STRUCTURAL_CONCEPTS_PATH.exists():
        raise SystemExit(
            f"Missing: {STRUCTURAL_CONCEPTS_PATH} -- run "
            "extract_structural_concepts.py, then embed it (on the Ollama "
            "machine) via embed_concept_backbone.py --input "
            "dictionaries/structural_concepts_candidates.csv --output "
            f"{STRUCTURAL_CONCEPTS_PATH}"
        )
    structural_concept_points = load_structural_concepts_points(STRUCTURAL_CONCEPTS_PATH)
    counts["structural_concepts"] = len(structural_concept_points)
    points.extend(structural_concept_points)

    emergent_entity_points = load_emergent_entities(CORPUS_ARCHIVES, args.entity_anchor_min_mentions)
    counts["emergent_entities"] = len(emergent_entity_points)
    points.extend(emergent_entity_points)

    logger.info("Pooled point counts: %s", counts)
    logger.info("Total pooled points: %d", len(points))

    vectors = np.array([p["vector"] for p in points], dtype=np.float64)
    for p in points:
        del p["vector"]
    if vectors.shape[1] != EMBEDDING_DIM:
        raise SystemExit(f"Expected {EMBEDDING_DIM}-d vectors, got {vectors.shape[1]}")

    logger.info("Standardizing pooled matrix (zero mean, unit variance per dimension) before PCA...")
    scaled = StandardScaler().fit_transform(vectors)

    logger.info("Fitting full-rank PCA to get the complete explained-variance curve...")
    full_pca = PCA(random_state=args.seed)
    full_coords = full_pca.fit_transform(scaled)
    cumulative_variance = np.cumsum(full_pca.explained_variance_ratio_)

    k = int(np.searchsorted(cumulative_variance, args.variance_threshold) + 1)
    k = min(k, len(cumulative_variance))
    variance_at_k = float(cumulative_variance[k - 1])
    logger.info("%.1f%% cumulative variance at k=%d (threshold: %.0f%%)", variance_at_k * 100, k, args.variance_threshold * 100)

    SHARED_SPACE_DIR.mkdir(parents=True, exist_ok=True)
    with open(VARIANCE_CSV_PATH, "w", encoding="utf-8") as f:
        f.write("n_components,cumulative_variance\n")
        for i, v in enumerate(cumulative_variance, 1):
            f.write(f"{i},{v}\n")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance)
    plt.axhline(args.variance_threshold, color="gray", linestyle="--", linewidth=1)
    plt.axvline(k, color="gray", linestyle="--", linewidth=1)
    plt.scatter([k], [variance_at_k], color="red", zorder=5, label=f"k={k}, {variance_at_k*100:.1f}%")
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title("Shared cross-corpus PCA: explained variance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(VARIANCE_PLOT_PATH, dpi=150)
    plt.close()

    VARIANCE_JSON_PATH.write_text(
        json.dumps({
            "curve": [float(v) for v in cumulative_variance],
            "chosen_k": k,
            "variance_at_k": variance_at_k,
            "threshold": args.variance_threshold,
        }, indent=2),
        encoding="utf-8",
    )

    shared_coords = full_coords[:, :k]

    logger.info("Writing %s ...", OUTPUT_PATH)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for p, coord in zip(points, shared_coords):
            out = {
                "source_dataset": p["source_dataset"],
                "point_role": p["point_role"],
                "key": p["key"],
                "label": p["label"],
                "label_en": p.get("label_en"),
                "label_fr": p.get("label_fr"),
                "attribution": p.get("attribution"),
                "claim_mode": p.get("claim_mode"),
                "epistemic_status": p.get("epistemic_status"),
                "response_rank": p.get("response_rank"),
                "mention_distribution": p.get("mention_distribution"),
                "shared_space_vector": coord.tolist(),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    sanity_checks(points, shared_coords)

    print(f"\nDone. {len(points)} points, k={k} ({variance_at_k*100:.1f}% variance).")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Variance curve: {VARIANCE_CSV_PATH}, {VARIANCE_PLOT_PATH}, {VARIANCE_JSON_PATH}")


if __name__ == "__main__":
    main()
