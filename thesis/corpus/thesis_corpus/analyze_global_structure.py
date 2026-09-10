"""Global geometric structure of the shared cross-corpus embedding space:
per-source centroids/dispersion, a pairwise source-centroid distance
matrix, and (separately) how the expression corpora as a whole relate to
the three reference vocabularies and to emergent entities.

Four deliverables, kept in separate output files and never conflated:

  1. Pairwise source-centroid distance matrix (8x8): computed directly
     between individual, independently-computed source centroids --
     mode-independent, no combined/weighted intermediate involved at all.
     A corpus's own centroid/dispersion isn't an imbalance-sensitive
     statistic; only comparisons *to a combined reference* are (below).
     The only deliverable that respects --epistemic-status-filter
     (default "all" = every point, unchanged from prior behaviour;
     "asserted_qualified" restricts the three expression corpora's own
     points to asserted/qualified epistemic status before computing their
     centroid/dispersion/pairwise distances -- checks whether e.g. a
     negated "X is NOT a cult" statement sitting close to an asserted
     "X is a cult" one is distorting these numbers). Deliverables 2-4
     below are always computed unfiltered, regardless of this flag.
     Two sub-deliverables sit alongside it, always computing BOTH epistemic
     slices regardless of the CLI flag (which only controls the pairwise
     matrix itself): 1b, `source_centroid_separation_normalized.csv` --
     `full_source_dispersion`-based normalized separation
     (R_{A,B} = centroid distance / mean dispersion, numerator and
     denominator always from the same slice -- an interpretive baseline for
     "is this centroid distance large or small," not a significance test);
     and 1c, `equal_n_dispersion.csv` -- bootstrapped dispersion under
     equal-sized sampling, for a fair cross-corpus comparison of internal
     breadth given literature's much larger raw point count. `dispersion`,
     `equal_n_dispersion`, and `normalized_separation_ratio` are three
     conceptually and terminologically distinct outputs, kept in separate
     files, never blended into one number.
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
  4. Nearest actual expressions to each expression corpus's own centroid:
     for literature/miviludes/interviews, and for both the full source and
     an asserted+qualified-only slice of it, the 10 nearest real
     expressions FROM THAT SAME SOURCE to its own centroid -- "what does
     this source's center of gravity actually sound like," as opposed to
     deliverables 1-3's cross-corpus/cross-vocabulary distances. Always
     computes both slices, independent of --epistemic-status-filter. Each
     row carries full provenance (document_id, chunk_index, pooled_key,
     occurrence_key, attribution, claim_mode, epistemic_status) and the
     raw archive's context_window, resolved via
     geometric_analysis_common.resolve_context_windows (a strict,
     occurrence-aware join -- a missing context_window for a row actually
     selected here is a hard error, not a blank field; see that function's
     own docstring for the MIVILUDES French/English-translation caveat). A
     companion `source_centroid_nearest_expressions_diversity_audit.csv`
     reports, per (source, epistemic_slice) group, how many distinct
     documents and distinct expression texts appear in that top-10 -- these
     are qualitative illustrations of what is central in the CURRENT
     embedding representation, not the essential or definitive position of
     a source.

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
from pathlib import Path

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
    # Sibling of whichever embedding_space.jsonl this run loaded -- never a
    # hardcoded v1 path (see geometric_analysis_common.corpus_vectors_and_points'
    # identical fix; kept in sync with it deliberately).
    balanced_sample_path = shared_space.input_path.parent / "literature_balanced_sample.jsonl"
    lit_points, lit_vectors = gac.load_reduced_literature_points(balanced_sample_path)
    return lit_vectors.mean(axis=0)


write_csv = gac.write_csv

CI95_VALIDATION_ATOL = 1e-12


def validate_equal_n_dispersion_ci95(rows: list[dict], atol: float = CI95_VALIDATION_ATOL) -> dict:
    """Checks ci95_low <= mean <= ci95_high with a small, DOCUMENTED
    floating-point tolerance -- validation-only, never touches the CSV
    values themselves (no rounding/clipping/smoothing of mean/ci95_low/
    ci95_high in the output; this function only reads rows, never mutates
    them).

    Real-world trigger: whichever expression corpus is the equal-n
    LIMITING corpus (the smallest one, currently interviews) has every
    bootstrap rep draw its exact same full retained set -- there is no
    genuine sampling variation across reps for that corpus, so the only
    "variation" in its per-rep dispersion values is floating-point
    summation-order noise from the underlying BLAS reduction
    (np.linalg.norm/.mean over a large matrix), typically ~1e-14 in
    magnitude here. That can push `mean` fractionally outside its own
    [ci95_low, ci95_high] interval -- an artifact of the exact-equality
    edge case, not a real statistical or logical error.

    Raises AssertionError if any row's violation exceeds `atol` (a real
    invariant failure, not tolerated). Returns a diagnostic dict --
    intended to be logged and recorded in this module's config.json,
    never silently discarded, even when every row passes within
    tolerance.
    """
    violations = []
    for row in rows:
        mean, ci95_low, ci95_high = row["mean_dispersion"], row["ci95_low"], row["ci95_high"]
        low_violation = max(ci95_low - mean, 0.0)
        high_violation = max(mean - ci95_high, 0.0)
        if low_violation > 0.0 or high_violation > 0.0:
            violations.append({
                "source_dataset": row["source_dataset"], "epistemic_slice": row["epistemic_slice"],
                "mean_dispersion": mean, "ci95_low": ci95_low, "ci95_high": ci95_high,
                "raw_low_violation_magnitude": low_violation,
                "raw_high_violation_magnitude": high_violation,
                "within_documented_tolerance": low_violation <= atol and high_violation <= atol,
            })
    for v in violations:
        if not v["within_documented_tolerance"]:
            raise AssertionError(
                f"equal_n_dispersion: ci95 bound violates mean for "
                f"source_dataset={v['source_dataset']!r} epistemic_slice={v['epistemic_slice']!r} "
                f"beyond the documented tolerance atol={atol} -- "
                f"low_violation={v['raw_low_violation_magnitude']}, "
                f"high_violation={v['raw_high_violation_magnitude']}. This is a genuine "
                "invariant failure, not the known floating-point edge case -- investigate."
            )
    return {
        "validation_atol": atol,
        "violations": violations,
        "explanation": (
            "A logged violation here means ci95_low > mean or mean > ci95_high by a small "
            "floating-point amount, not a real invariant failure once the documented tolerance "
            "is applied. Expected specifically when source_dataset is the equal-n limiting "
            "corpus for that epistemic_slice: every bootstrap rep then draws the exact same "
            "full retained set, so the only apparent variation across reps is floating-point "
            "summation-order noise from the underlying BLAS reduction, not genuine sampling "
            "variation. CSV values (mean_dispersion/ci95_low/ci95_high) are never rounded, "
            "clipped, or otherwise altered by this check."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--shared-space-dir", type=Path, default=None,
                         help="Use a different pooled space (e.g. processed/shared_space_v2/) instead of v1's "
                              "processed/shared_space/; also switches the run-output root to a sibling "
                              "processed/analysis_v2/ directory so v1 and v2 runs are never mixed.")
    parser.add_argument("--seed", type=int, default=gac.DEFAULT_SEED)
    parser.add_argument("--bootstrap-reps", type=int, default=gac.DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument(
        "--epistemic-status-filter", choices=list(gac.EPISTEMIC_STATUS_FILTER_CHOICES), default="all",
        help=(
            "Restricts Deliverable 1's pairwise source-centroid matrix (only) "
            "to expression points with this epistemic status; 'all' (default) "
            "is the unfiltered, pre-existing behaviour. Deliverables 2a/2b/2c "
            "and 4 are unaffected by this flag."
        ),
    )
    args = parser.parse_args()
    epistemic_filter = gac.EPISTEMIC_STATUS_FILTER_CHOICES[args.epistemic_status_filter]

    embedding_space_path, interview_prototypes_path, analysis_root = gac.resolve_space_paths(args.shared_space_dir)
    logger.info("Loading %s ...", embedding_space_path)
    shared_space = gac.load_shared_space(embedding_space_path)
    logger.info("Loaded %d points, %d-d vectors", len(shared_space.points), shared_space.vectors.shape[1])

    run_id, run_dir = gac.get_or_create_run_dir(args.run_id, analysis_root)
    gac.init_run_manifest(run_dir, shared_space, defaults={"seed": args.seed})
    gac.update_run_manifest(run_dir, MODULE_NAME, "running")
    out_dir = gac.module_run_dir(run_dir, MODULE_NAME)

    try:
        # --- Deliverable 1: mode-independent per-source centroids + pairwise matrix ---
        # per_source (unfiltered "all" slice) and per_source_asserted_qualified
        # (always computed regardless of --epistemic-status-filter) together
        # drive per_source_centroids_dispersion.csv, which always reports BOTH
        # slices -- this is independent of the CLI flag, which only controls
        # which slice pairwise_source_centroid_matrix.csv uses.
        # per_source (unfiltered) also feeds deliverables 2a/2b/2c below, which
        # are deliberately NOT affected by --epistemic-status-filter.
        logger.info(
            "Deliverable 1: per-source centroids/dispersion (both epistemic slices, "
            "always) + pairwise matrix (mode-independent, epistemic_status_filter=%s)...",
            args.epistemic_status_filter,
        )
        per_source = gac.per_source_centroids_and_dispersion(shared_space)
        per_source_asserted_qualified = gac.per_source_centroids_and_dispersion(
            shared_space, epistemic_status_filter=gac.EPISTEMIC_STATUS_FILTER_CHOICES["asserted_qualified"],
        )
        per_source_for_matrix = per_source if epistemic_filter is None else per_source_asserted_qualified

        # per_source_centroids_dispersion.csv: the pre-existing "all"-slice
        # rows keep every previously-existing value exactly (same computation
        # as before this pass, just with epistemic_slice/median/p10/p90
        # columns added) -- never duplicated or overwritten. A second set of
        # rows is added for "asserted_qualified". ALL 8 source_datasets are
        # included in both slices (not just the 3 expression corpora) -- this
        # is a deliberate choice: it matches the filter's actual logic (which
        # only ever touches the 3 expression corpora) and gives a directly
        # checkable confirmation, rather than a mere assertion, that the 5
        # status-less datasets' centroid/dispersion are unaffected by the
        # filter -- their asserted_qualified-slice row must come out
        # byte-identical to their all-slice row.
        centroid_rows = []
        for slice_name, ps in (("all", per_source), ("asserted_qualified", per_source_asserted_qualified)):
            for s, v in ps.items():
                centroid_rows.append({
                    "source_dataset": s, "epistemic_slice": slice_name, "n": v["n"],
                    "dispersion": v["dispersion"], "dispersion_median": v["dispersion_median"],
                    "dispersion_p10": v["dispersion_p10"], "dispersion_p90": v["dispersion_p90"],
                })
        write_csv(out_dir / "per_source_centroids_dispersion.csv", centroid_rows)

        pairwise_rows = pairwise_centroid_matrix(per_source_for_matrix)
        write_csv(out_dir / "pairwise_source_centroid_matrix.csv", pairwise_rows)

        # --- Deliverable 1b (new): normalized separation, full-source only ---
        # R_{A,B} = d(centroid_A, centroid_B) / mean(D_A, D_B), both the
        # distance and both dispersions always drawn from the SAME slice
        # (per_source for "all", per_source_asserted_qualified for
        # "asserted_qualified") -- never mixed across slices. This is a
        # deterministic computation over full-source centroids/dispersions,
        # no random draw -- reproducibility here means exact byte-identical
        # rerun, not "under a fixed seed" (that applies to Deliverable 1c
        # below, which IS randomised).
        logger.info("Deliverable 1b: normalized source-centroid separation (full-source, both epistemic slices)...")
        normalized_separation_rows = []
        for slice_name, ps in (("all", per_source), ("asserted_qualified", per_source_asserted_qualified)):
            for row in gac.normalized_separation(ps):
                normalized_separation_rows.append({"epistemic_slice": slice_name, **row})
        write_csv(out_dir / "source_centroid_separation_normalized.csv", normalized_separation_rows)

        # --- Deliverable 1c (new): equal-n dispersion, bootstrapped ---
        # Kept in a SEPARATE file from Deliverable 1b's ratio table on
        # purpose, so it can never be misread as having been used in that
        # ratio (it wasn't -- Deliverable 1b is full-source only; an equal-n
        # version of the ratio itself is a distinct, more expensive
        # deliverable not built in this pass, since it would need centroid
        # distance AND dispersion recomputed from the same equal-n draw
        # within each bootstrap rep).
        logger.info(
            "Deliverable 1c: equal-n bootstrapped dispersion (B=%d reps, both epistemic slices)...",
            args.bootstrap_reps,
        )
        equal_n_dispersion_rows = []
        for slice_name, slice_filter in (("all", None), ("asserted_qualified", gac.EPISTEMIC_STATUS_FILTER_CHOICES["asserted_qualified"])):
            equal_n_result = gac.equal_n_dispersion(
                shared_space, seed=args.seed, reps=args.bootstrap_reps, epistemic_status_filter=slice_filter,
            )
            for corpus, stats in equal_n_result.items():
                equal_n_dispersion_rows.append({
                    "source_dataset": corpus, "epistemic_slice": slice_name,
                    "equal_n_draw_size": stats["n_per_draw"], "bootstrap_reps": stats["n_reps"], "seed": args.seed,
                    "mean_dispersion": stats["mean"], "std": stats["std"],
                    "ci95_low": stats["ci95_low"], "ci95_high": stats["ci95_high"],
                })
        write_csv(out_dir / "equal_n_dispersion.csv", equal_n_dispersion_rows)

        equal_n_dispersion_ci95_validation = validate_equal_n_dispersion_ci95(equal_n_dispersion_rows)
        if equal_n_dispersion_ci95_validation["violations"]:
            logger.info(
                "equal_n_dispersion CI95 validation: %d row(s) had a sub-tolerance "
                "(atol=%s) ci95/mean discrepancy -- see config.json's "
                "equal_n_dispersion_ci95_validation for details: %s",
                len(equal_n_dispersion_ci95_validation["violations"]),
                equal_n_dispersion_ci95_validation["validation_atol"],
                equal_n_dispersion_ci95_validation["violations"],
            )
        else:
            logger.info("equal_n_dispersion CI95 validation: all rows satisfy ci95_low <= mean <= ci95_high exactly.")

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
                    for row in gac.nearest_points(centroid, reference_points[ref_dataset], reference_vectors[ref_dataset], NEAREST_TERMS_K):
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

        # --- Deliverable 4 (enriched): nearest actual expressions to each
        # expression corpus's own centroid, raw and asserted+qualified-only.
        # Candidates are drawn from the SAME source only (not the full
        # expression pool) -- this answers "what does this source's own
        # center of gravity actually sound like", a different question from
        # deliverables 1-3's cross-corpus/cross-vocabulary distances. Always
        # computes both epistemic slices regardless of
        # --epistemic-status-filter (that flag only governs deliverable 1).
        #
        # These are qualitative illustrations of what is central in the
        # CURRENT embedding representation of a source -- not the essential
        # or definitive position of that source.
        #
        # context_window is resolved once per source (not per slice, not
        # per row) via gac.resolve_context_windows, reused for both epistemic
        # slices of that source. NOTE for MIVILUDES specifically: the
        # `embedding_text` column below is the pooled point's own `label`
        # (the ENGLISH translation actually embedded/used for this distance
        # calculation), while `context_window` is the ORIGINAL FRENCH
        # surrounding text from the raw archive -- a real, expected
        # consequence of MIVILUDES being pooled by its English translation
        # (see resolve_context_windows' own docstring), not a join defect.
        logger.info("Deliverable 4: nearest expressions to each source's own centroid (all / asserted_qualified)...")
        source_centroid_nearest_rows = []
        diversity_audit_rows = []
        context_resolutions: dict[str, gac.ContextWindowResolution] = {}
        archive_paths = gac.resolve_archive_paths(shared_space)
        for corpus in gac.EXPRESSION_CORPORA:
            context_resolutions[corpus] = gac.resolve_context_windows(shared_space, corpus, archive_paths[corpus])
            if corpus == "miviludes":
                logger.info(
                    "MIVILUDES source_centroid_nearest_expressions rows: document provenance "
                    "(document_id/chunk_index/context_window) retained explicitly, since the "
                    "corpus is only %d source documents -- always inspect provenance for this "
                    "source rather than treating any single nearest expression as representative.",
                    len({item["document_id"] for item in context_resolutions[corpus].archive_items}),
                )

        for corpus in gac.EXPRESSION_CORPORA:
            resolution = context_resolutions[corpus]
            for slice_name, slice_filter in (("all", None), ("asserted_qualified", gac.EPISTEMIC_STATUS_FILTER_CHOICES["asserted_qualified"])):
                idxs = gac.source_dataset_indices(shared_space.points, corpus, slice_filter)
                vecs = shared_space.vectors[idxs]
                centroid = vecs.mean(axis=0)
                cand_points = [shared_space.points[i] for i in idxs]

                # gac.nearest_points is reused for its rank/label/distance
                # formatting, but its own return rows don't carry the
                # original shared_space.points index (only the point's own
                # `key`, which is NOT guaranteed globally unique) -- so the
                # exact same argsort is replicated locally here to recover
                # the true global index per rank, without changing
                # nearest_points' validated contract. exclude_self=False
                # (the default, used here) makes this equivalent by
                # construction: nearest_points with no self-exclusion is
                # exactly a plain argsort-and-slice.
                euclidean = gac.euclidean_distances(centroid, vecs)
                rank_order_local = list(np.argsort(euclidean)[:NEAREST_TERMS_K])
                global_indices_in_rank_order = [idxs[i] for i in rank_order_local]
                rows = gac.nearest_points(centroid, cand_points, vecs, NEAREST_TERMS_K)
                assert len(rows) == len(global_indices_in_rank_order), (
                    f"{corpus}/{slice_name}: nearest_points returned {len(rows)} rows but "
                    f"the replicated argsort produced {len(global_indices_in_rank_order)} -- "
                    "ranking mismatch, stop and investigate rather than silently zip mismatched rows."
                )

                document_ids_in_group: list[str] = []
                embedding_texts_in_group: list[str] = []
                for row, global_idx in zip(rows, global_indices_in_rank_order):
                    point = shared_space.points[global_idx]
                    pooled_key = point["key"]
                    document_id = gac.key_document_id(pooled_key)
                    chunk_index = gac.key_chunk_index(pooled_key)
                    occurrence_key = resolution.occurrence_key_by_pooled_index.get(global_idx)
                    context_window = resolution.context_window_by_pooled_index.get(global_idx)

                    # Hard error, not a blank field: a selected row (one
                    # that made it into this deliverable) with no resolved
                    # context_window means the occurrence-aware join failed
                    # for a row we're actually relying on -- stop and report
                    # rather than continue silently.
                    if context_window is None or context_window == "":
                        raise SystemExit(
                            f"source_centroid_nearest_expressions: missing context_window for a "
                            f"SELECTED row -- source_dataset={corpus!r}, epistemic_slice={slice_name!r}, "
                            f"pooled_key={pooled_key!r}, occurrence_key={occurrence_key!r}. "
                            "This is a hard-fail per the fail-loud context-window resolution rule; "
                            "investigate the occurrence-aware join for this source before proceeding."
                        )

                    document_ids_in_group.append(document_id)
                    embedding_texts_in_group.append(point["label"])

                    source_centroid_nearest_rows.append({
                        "source_dataset": corpus, "epistemic_slice": slice_name, "n_in_slice": len(idxs),
                        "rank": row["rank"], "pooled_key": pooled_key, "occurrence_key": occurrence_key,
                        "document_id": document_id, "chunk_index": chunk_index,
                        "embedding_text": point["label"], "context_window": context_window,
                        "attribution": point.get("attribution"), "claim_mode": point.get("claim_mode"),
                        "epistemic_status": point.get("epistemic_status"),
                        "euclidean_distance": row["euclidean_distance"], "cosine_similarity": row["cosine_similarity"],
                    })

                # Diversity audit: how concentrated is this top-10 group in
                # a single document, and does it contain duplicate/
                # near-identical text (normalized-whitespace exact match --
                # a pragmatic operational definition, not NLP-based
                # near-duplicate detection).
                normalized_texts = [" ".join(t.split()).lower() for t in embedding_texts_in_group]
                n_unique_texts = len(set(normalized_texts))
                diversity_audit_rows.append({
                    "source_dataset": corpus, "epistemic_slice": slice_name,
                    "n_rows": len(embedding_texts_in_group),
                    "n_unique_document_ids": len(set(document_ids_in_group)),
                    "n_unique_embedding_texts": n_unique_texts,
                    "has_duplicate_or_near_identical_text": n_unique_texts < len(embedding_texts_in_group),
                })
        write_csv(out_dir / "source_centroid_nearest_expressions.csv", source_centroid_nearest_rows)
        write_csv(out_dir / "source_centroid_nearest_expressions_diversity_audit.csv", diversity_audit_rows)

        # --- Deliverable 4b: nearest EMERGENT ENTITIES to each source's own
        # (all-status) centroid -- the entity-layer counterpart to
        # Deliverable 4 above, same "what's actually central to this
        # source's centroid" question, answered against the entity pool
        # instead of the expression pool. No context_window resolution
        # needed here (entities aren't tied to one archive occurrence the
        # way an expression is), so this is a much shorter block than
        # Deliverable 4 -- a straight gac.nearest_points call per corpus.
        logger.info("Deliverable 4b: nearest emergent entities to each source's own centroid...")
        entity_idxs = gac.source_dataset_indices(shared_space.points, "emergent_entities")
        entity_points = [shared_space.points[i] for i in entity_idxs]
        entity_vectors = shared_space.vectors[entity_idxs]
        source_centroid_nearest_entity_rows = []
        for corpus in gac.EXPRESSION_CORPORA:
            centroid = per_source[corpus]["centroid"]
            rows = gac.nearest_points(centroid, entity_points, entity_vectors, NEAREST_TERMS_K)
            for row in rows:
                source_centroid_nearest_entity_rows.append({
                    "source_dataset": corpus, "rank": row["rank"],
                    "entity_key": row["key"], "entity_label": row["label"],
                    "euclidean_distance": row["euclidean_distance"], "cosine_similarity": row["cosine_similarity"],
                })
        write_csv(out_dir / "source_centroid_nearest_entities.csv", source_centroid_nearest_entity_rows)

        for corpus, resolution in context_resolutions.items():
            logger.info("resolve_context_windows(%s) stats: %s", corpus, resolution.stats)

        gac.write_module_config(
            out_dir,
            input_sha256=shared_space.input_sha256,
            n_points=len(shared_space.points),
            per_source_n={s: v["n"] for s, v in per_source.items()},
            per_source_n_for_matrix={s: v["n"] for s, v in per_source_for_matrix.items()},
            epistemic_status_filter=args.epistemic_status_filter,
            seed=args.seed,
            metric="euclidean_primary_cosine_sensitivity",
            combined_modes=COMBINED_MODES,
            per_corpus_modes=PER_CORPUS_MODES,
            nearest_terms_k=NEAREST_TERMS_K,
            bootstrap_reps=args.bootstrap_reps,
            reference_vocab_datasets=gac.REFERENCE_VOCAB_DATASETS,
            context_window_resolution_stats={c: r.stats for c, r in context_resolutions.items()},
            equal_n_dispersion_ci95_validation=equal_n_dispersion_ci95_validation,
            git_commit=gac.git_commit_hash(),
        )
        gac.update_run_manifest(run_dir, MODULE_NAME, "completed", output_dir=out_dir)
    except Exception as e:
        gac.update_run_manifest(run_dir, MODULE_NAME, "failed", error=str(e))
        raise

    print(f"\nDone. Run {run_id} -> {out_dir}")


if __name__ == "__main__":
    main()
