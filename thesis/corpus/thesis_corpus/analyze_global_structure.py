"""Global geometric structure of the shared cross-corpus embedding space:
per-source centroids/dispersion, a pairwise source-centroid distance
matrix, and (separately) how the expression corpora as a whole relate to
the three reference vocabularies and to emergent entities.

Three deliverables, kept in separate output files and never conflated:

  1. Pairwise source-centroid distance matrix (8x8): computed directly
     between individual, independently-computed source centroids --
     mode-independent, no combined/weighted intermediate involved at all.
     A corpus's own centroid/dispersion isn't an imbalance-sensitive
     statistic; only comparisons *to a combined reference* are (below).
  2. Combined-expression-centroid comparisons: distance from
     concept_backbone / structural_concepts / conceptnet_concepts /
     emergent_entities centroids to "the expression corpora as a whole",
     in three variants that are never merged -- full (raw pooled mean),
     reduced_literature (pooled mean using the 2,500-point literature
     subsample), and equal_weight (literature/MIVILUDES/interview
     centroids averaged with equal 1/3 weight, via
     balanced_analysis.weighted_centroid()). Plus, for each expression
     corpus's own centroid (full and reduced_literature only --
     equal_weight has no single-corpus analogue), the nearest 10 terms
     from each of the 3 reference vocabularies (concept_backbone,
     structural_concepts, conceptnet_concepts), one long-format table
     tagged by `reference_dataset` -- raw retrieval, NOT adjusted for the
     three sets' very different sizes (3,000/1,500/195).
  3. Equal-size-controlled reference comparison (new): the same
     combined-expression and per-corpus-centroid queries as deliverables
     2a/2b, but nearest-k distance into each reference set after
     down-sampling every set to the same size
     (geometric_analysis_common.equal_size_reference_comparison,
     n_reference = min of the 3 sizes, repeated B times, mean/std/95%-CI)
     -- controls for pool-size bias, kept in its own file, never merged
     with the raw retrieval table above. This controls cardinality only;
     it is not evidence the three vocabularies are otherwise
     interchangeable (they differ in source, curation, coverage, and
     purpose -- see ANALYSIS_OVERVIEW.md's "Why three reference subsets").

Euclidean distance is primary throughout; cosine is reported alongside as
a sensitivity column, never primary.

Never imports from or modifies build_shared_space.py/balanced_analysis.py
beyond reusing weighted_centroid(); never writes anywhere but its own
processed/analysis/<run-id>/analyze_global_structure/ subdirectory.

Usage (from thesis/corpus/):
    python -m thesis_corpus.analyze_global_structure
    python -m thesis_corpus.analyze_global_structure --run-id 20260101-120000
"""
from __future__ import annotations

import argparse
import logging

import numpy as np

from thesis_corpus import geometric_analysis_common as gac

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("thesis_corpus.analyze_global_structure")

MODULE_NAME = "analyze_global_structure"
REFERENCE_LIKE_DATASETS = ("concept_backbone", "structural_concepts", "conceptnet_concepts", "emergent_entities")
NEAREST_TERMS_K = 10
COMBINED_MODES = ("full", "reduced_literature", "equal_weight")
PER_CORPUS_MODES = ("full", "reduced_literature")


def pairwise_centroid_matrix(centroids: dict[str, dict]) -> list[dict]:
    """7x7 direct pairwise comparison between independent source centroids
    -- no combined/weighted reference involved. One row per ordered pair
    (a, b), a != b."""
    rows = []
    sources = list(centroids.keys())
    for a in sources:
        for b in sources:
            if a == b:
                continue
            ca, cb = centroids[a]["centroid"], centroids[b]["centroid"]
            euclidean = float(np.linalg.norm(ca - cb))
            cosine = float(gac.cosine_similarities(ca, cb[np.newaxis, :])[0])
            rows.append({
                "source_a": a, "source_b": b,
                "euclidean_distance": euclidean,
                "cosine_similarity": cosine,
            })
    return rows


def corpus_centroid_for_mode(shared_space: gac.SharedSpace, corpus: str, mode: str, per_source: dict) -> np.ndarray:
    """A single expression corpus's own centroid under `mode` (full or
    reduced_literature only). For MIVILUDES/interviews these two modes
    coincide (reduced_literature doesn't touch them) so the mode-
    independent per_source value is reused directly; for literature,
    reduced_literature recomputes from the 2,500-point subsample."""
    if mode == "full" or corpus != "literature":
        return per_source[corpus]["centroid"]
    lit_points, lit_vectors = gac.load_reduced_literature_points()
    return lit_vectors.mean(axis=0)


def nearest_terms(query: np.ndarray, candidate_points: list[dict], candidate_vectors: np.ndarray, k: int) -> list[dict]:
    euclidean = gac.euclidean_distances(query, candidate_vectors)
    cosine = gac.cosine_similarities(query, candidate_vectors)
    order = np.argsort(euclidean)[:k]
    return [
        {
            "rank": rank + 1,
            "key": candidate_points[i]["key"],
            "label": candidate_points[i]["label"],
            "euclidean_distance": float(euclidean[i]),
            "cosine_similarity": float(cosine[i]),
        }
        for rank, i in enumerate(order)
    ]


write_csv = gac.write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=gac.DEFAULT_SEED)
    parser.add_argument("--bootstrap-reps", type=int, default=gac.DEFAULT_BOOTSTRAP_REPS)
    args = parser.parse_args()

    logger.info("Loading %s ...", gac.EMBEDDING_SPACE_PATH)
    shared_space = gac.load_shared_space()
    logger.info("Loaded %d points, %d-d vectors", len(shared_space.points), shared_space.vectors.shape[1])

    run_id, run_dir = gac.get_or_create_run_dir(args.run_id)
    gac.init_run_manifest(run_dir, shared_space, defaults={"seed": args.seed})
    gac.update_run_manifest(run_dir, MODULE_NAME, "running")
    out_dir = gac.module_run_dir(run_dir, MODULE_NAME)

    try:
        # --- Deliverable 1: mode-independent per-source centroids + pairwise matrix ---
        logger.info("Deliverable 1: per-source centroids/dispersion + pairwise matrix (mode-independent)...")
        per_source = gac.per_source_centroids_and_dispersion(shared_space)
        centroid_rows = [
            {"source_dataset": s, "n": v["n"], "dispersion": v["dispersion"]}
            for s, v in per_source.items()
        ]
        write_csv(out_dir / "per_source_centroids_dispersion.csv", centroid_rows)

        pairwise_rows = pairwise_centroid_matrix(per_source)
        write_csv(out_dir / "pairwise_source_centroid_matrix.csv", pairwise_rows)

        # --- Deliverable 2a: reference/emergent centroids vs combined-expression reference ---
        logger.info("Deliverable 2a: reference-to-combined-expression-centroid distances (full/reduced_literature/equal_weight)...")
        combined_rows = []
        for mode in COMBINED_MODES:
            combined_ref = gac.combined_expression_reference(shared_space, mode)
            for ref_dataset in REFERENCE_LIKE_DATASETS:
                ref_centroid = per_source[ref_dataset]["centroid"]
                euclidean = float(np.linalg.norm(ref_centroid - combined_ref))
                cosine = float(gac.cosine_similarities(ref_centroid, combined_ref[np.newaxis, :])[0])
                combined_rows.append({
                    "mode": mode, "reference_dataset": ref_dataset,
                    "euclidean_distance_to_combined_expression": euclidean,
                    "cosine_similarity_to_combined_expression": cosine,
                })
        write_csv(out_dir / "reference_to_combined_expression_centroid.csv", combined_rows)

        # --- Deliverable 2b: nearest reference-vocabulary terms to each expression-corpus centroid (raw, pool-size-biased) ---
        logger.info(
            "Deliverable 2b: nearest %s terms to each expression-corpus centroid "
            "(full/reduced_literature; raw retrieval, NOT size-controlled -- see equal-size version below)...",
            ", ".join(gac.REFERENCE_VOCAB_DATASETS),
        )
        reference_points: dict[str, list[dict]] = {}
        reference_vectors: dict[str, np.ndarray] = {}
        for ref_dataset in gac.REFERENCE_VOCAB_DATASETS:
            idxs = gac.source_dataset_indices(shared_space.points, ref_dataset)
            reference_points[ref_dataset] = [shared_space.points[i] for i in idxs]
            reference_vectors[ref_dataset] = shared_space.vectors[idxs]

        nearest_reference_rows_raw = []
        for corpus in gac.EXPRESSION_CORPORA:
            for mode in PER_CORPUS_MODES:
                centroid = corpus_centroid_for_mode(shared_space, corpus, mode, per_source)
                for ref_dataset in gac.REFERENCE_VOCAB_DATASETS:
                    for row in nearest_terms(centroid, reference_points[ref_dataset], reference_vectors[ref_dataset], NEAREST_TERMS_K):
                        nearest_reference_rows_raw.append({
                            "expression_corpus": corpus, "mode": mode,
                            "reference_dataset": ref_dataset, **row,
                        })
        write_csv(out_dir / "nearest_reference_terms_raw.csv", nearest_reference_rows_raw)

        # --- Deliverable 2c (new): equal-size-controlled reference comparison ---
        # Raw nearest-term retrieval above is biased toward whichever
        # reference set is largest (3,000/1,500/195 candidates) purely by
        # pool size. This controls for that: every reference set is
        # down-sampled to the same n_reference (= min across the three,
        # currently 195) before nearest-k distance is computed, repeated
        # gac.DEFAULT_BOOTSTRAP_REPS times. Covers both deliverable 2a's
        # combined-expression-centroid queries and 2b's per-corpus-centroid
        # queries -- kept in its own file, never merged with the raw table.
        logger.info(
            "Deliverable 2c: equal-size reference comparison (n_reference=min of the "
            "3 reference-set sizes, B=%d reps)...", args.bootstrap_reps,
        )
        queries: dict[str, np.ndarray] = {}
        query_meta: dict[str, dict] = {}
        for corpus in gac.EXPRESSION_CORPORA:
            for mode in PER_CORPUS_MODES:
                key = f"corpus_centroid:{corpus}:{mode}"
                queries[key] = corpus_centroid_for_mode(shared_space, corpus, mode, per_source)
                query_meta[key] = {"query_type": "expression_corpus_centroid", "query_label": corpus, "mode": mode}
        for mode in COMBINED_MODES:
            key = f"combined_expression_centroid:{mode}"
            queries[key] = gac.combined_expression_reference(shared_space, mode)
            query_meta[key] = {"query_type": "combined_expression_centroid", "query_label": "combined_expression", "mode": mode}

        equal_size_result = gac.equal_size_reference_comparison(
            queries, reference_vectors, k=NEAREST_TERMS_K, seed=args.seed, reps=args.bootstrap_reps,
        )
        equal_size_rows = []
        for query_key, per_reference in equal_size_result.items():
            meta = query_meta[query_key]
            for reference_dataset, stats in per_reference.items():
                for distance_statistic in ("mean", "median"):
                    summary = stats[f"nearest_k_distance_{distance_statistic}"]
                    equal_size_rows.append({
                        "query_key": query_key, "query_label": meta["query_label"], "query_type": meta["query_type"],
                        "mode": meta["mode"], "reference_dataset": reference_dataset,
                        "n_reference": stats["n_reference"], "k": stats["k"],
                        "repetitions": args.bootstrap_reps, "seed": args.seed,
                        "distance_statistic": distance_statistic, **summary,
                    })
        write_csv(out_dir / "nearest_reference_terms_equal_size.csv", equal_size_rows)

        gac.write_module_config(
            out_dir,
            input_sha256=shared_space.input_sha256,
            n_points=len(shared_space.points),
            per_source_n={s: v["n"] for s, v in per_source.items()},
            seed=args.seed,
            metric="euclidean_primary_cosine_sensitivity",
            combined_modes=COMBINED_MODES,
            per_corpus_modes=PER_CORPUS_MODES,
            nearest_terms_k=NEAREST_TERMS_K,
            bootstrap_reps=args.bootstrap_reps,
            reference_vocab_datasets=gac.REFERENCE_VOCAB_DATASETS,
            git_commit=gac.git_commit_hash(),
        )
        gac.update_run_manifest(run_dir, MODULE_NAME, "completed", output_dir=out_dir)
    except Exception as e:
        gac.update_run_manifest(run_dir, MODULE_NAME, "failed", error=str(e))
        raise

    print(f"\nDone. Run {run_id} -> {out_dir}")


if __name__ == "__main__":
    main()
