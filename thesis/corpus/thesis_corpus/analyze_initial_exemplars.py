"""Geometric analysis of the interview initial-exemplar prototype layer
(processed/shared_space/interview_prototypes.jsonl) -- the replacement for
rank-based prototype analysis, which is retired outright (see
audit_free_listing_rank.py).

Reads the prototype layer only -- all review-gating and virtual-key
resolution against the raw interview archive already happened in
build_interview_prototype_layer.py (run that first; it fails loudly on an
unreviewed CSV row, a missing initial_response_form, or a resolution
mismatch, so by the time this script runs, every prototype point is
already a manually-reviewed, verified exemplar). This script only computes
distances against the *existing* shared space (embedding_space.jsonl) --
it never re-validates initial_exemplars.csv itself.

n~=25 throughout: every output is exploratory/descriptive only, never
given an inferential-statistics treatment.

Usage (from thesis/corpus/):
    python -m thesis_corpus.analyze_initial_exemplars
"""
from __future__ import annotations

import argparse
import csv
import json
import logging

import numpy as np

from thesis_corpus import geometric_analysis_common as gac

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("thesis_corpus.analyze_initial_exemplars")

MODULE_NAME = "analyze_initial_exemplars"
NEAREST_K = 10
CENTROID_MODES = ("full", "reduced_literature", "equal_weight")
SAME_VECTOR_EPSILON = 1e-9  # for detecting a prototype's own point re-appearing in the pooled interview pool


def load_interview_prototypes(path) -> tuple[list[dict], np.ndarray]:
    points, vectors = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            points.append({k: v for k, v in item.items() if k != "shared_space_vector"})
            vectors.append(item["shared_space_vector"])
    return points, np.array(vectors, dtype=np.float64)


def nearest(
    query: np.ndarray, candidate_points: list[dict], candidate_vectors: np.ndarray, k: int,
    exclude_self: bool = False,
) -> list[dict]:
    """exclude_self drops any candidate numerically identical to `query` --
    relevant when the same underlying expression could independently exist
    in both the prototype layer and the ordinary pooled interview pool
    (whenever a reviewed exemplar happened to also survive the pooling
    filter): both are derived from the same raw vector through the same
    persisted transform, so they'd be exact duplicates, not a genuine
    nearest neighbour."""
    euclidean = gac.euclidean_distances(query, candidate_vectors)
    cosine = gac.cosine_similarities(query, candidate_vectors)
    order = np.argsort(euclidean)
    results = []
    for i in order:
        if exclude_self and euclidean[i] < SAME_VECTOR_EPSILON:
            continue
        results.append({
            "rank": len(results) + 1,
            "key": candidate_points[i]["key"],
            "label": candidate_points[i]["label"],
            "euclidean_distance": float(euclidean[i]),
            "cosine_similarity": float(cosine[i]),
        })
        if len(results) == k:
            break
    return results


write_csv = gac.write_csv


def assert_prototype_layer_current(shared_space: gac.SharedSpace) -> dict:
    """Part 1 of the post-ConceptNet-rebuild plan: confirm (never
    re-derive/rebuild here) that interview_prototypes.jsonl's provenance
    matches the *current* embedding_space.jsonl. If a future pipeline
    rebuild ever changes the PCA fit without reprojecting the prototype
    layer, every distance this module computes against it would be
    silently wrong -- this check turns that into a loud failure instead.
    Returns the transform metadata dict for recording in config.json."""
    if not gac.PCA_TRANSFORM_METADATA_PATH.exists():
        raise SystemExit(f"Missing {gac.PCA_TRANSFORM_METADATA_PATH} -- cannot verify prototype-layer currency.")
    metadata = json.loads(gac.PCA_TRANSFORM_METADATA_PATH.read_text(encoding="utf-8"))
    if metadata.get("embedding_space_sha256") != shared_space.input_sha256:
        raise SystemExit(
            f"pca_transform_metadata.json's embedding_space_sha256 "
            f"({metadata.get('embedding_space_sha256')}) does not match the current "
            f"embedding_space.jsonl ({shared_space.input_sha256}) -- the shared space "
            "was rebuilt since the persisted transform was fit. Re-run "
            "build_shared_space.py's persistence step, then "
            "build_interview_prototype_layer.py to reproject, before running this module."
        )
    logger.info(
        "Prototype-layer currency confirmed: pca_transform_metadata.json's "
        "embedding_space_sha256 matches the current embedding_space.jsonl (n_points_fit=%d).",
        metadata.get("n_points_fit"),
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--prototypes", type=type(gac.INTERVIEW_PROTOTYPES_PATH), default=gac.INTERVIEW_PROTOTYPES_PATH)
    parser.add_argument("--seed", type=int, default=gac.DEFAULT_SEED)
    parser.add_argument("--bootstrap-reps", type=int, default=gac.DEFAULT_BOOTSTRAP_REPS)
    args = parser.parse_args()

    if not args.prototypes.exists():
        raise SystemExit(
            f"No prototype layer at {args.prototypes} -- run "
            "`python -m thesis_corpus.build_interview_prototype_layer` first."
        )

    logger.info("Loading %s ...", args.prototypes)
    prototype_points, prototype_vectors = load_interview_prototypes(args.prototypes)
    logger.info("%d interview prototype points", len(prototype_points))

    logger.info("Loading %s ...", gac.EMBEDDING_SPACE_PATH)
    shared_space = gac.load_shared_space()
    transform_metadata = assert_prototype_layer_current(shared_space)

    run_id, run_dir = gac.get_or_create_run_dir(args.run_id)
    gac.init_run_manifest(run_dir, shared_space, defaults={})
    gac.update_run_manifest(run_dir, MODULE_NAME, "running")
    out_dir = gac.module_run_dir(run_dir, MODULE_NAME)

    try:
        criteria_idxs = gac.source_dataset_indices(shared_space.points, "miviludes_criteria")
        criteria_points = [shared_space.points[i] for i in criteria_idxs]
        criteria_vectors = shared_space.vectors[criteria_idxs]

        per_source = gac.per_source_centroids_and_dispersion(shared_space)
        combined_refs = {mode: gac.combined_expression_reference(shared_space, mode) for mode in CENTROID_MODES}

        lit_idxs = gac.source_dataset_indices(shared_space.points, "literature")
        lit_points = [shared_space.points[i] for i in lit_idxs]
        lit_vectors = shared_space.vectors[lit_idxs]

        miv_idxs = gac.source_dataset_indices(shared_space.points, "miviludes")
        miv_points = [shared_space.points[i] for i in miv_idxs]
        miv_vectors = shared_space.vectors[miv_idxs]

        interview_idxs = gac.source_dataset_indices(shared_space.points, "interviews")
        interview_points_all = [shared_space.points[i] for i in interview_idxs]
        interview_vectors_all = shared_space.vectors[interview_idxs]

        criterion_distance_rows = []
        reference_distance_rows = []
        centroid_distance_rows = []
        nearest_lit_rows, nearest_miv_rows, nearest_interview_rows = [], [], []
        exemplar_summary_rows = []

        for proto, query in zip(prototype_points, prototype_vectors):
            document_id = proto["document_id"]
            exemplar_type = proto["exemplar_type"]
            initial_response_form = proto["initial_response_form"]

            for cp, cv in zip(criteria_points, criteria_vectors):
                euclidean = float(np.linalg.norm(query - cv))
                cosine = float(gac.cosine_similarities(query, cv[np.newaxis, :])[0])
                criterion_distance_rows.append({
                    "document_id": document_id, "exemplar_type": exemplar_type,
                    "initial_response_form": initial_response_form,
                    "criterion_key": cp["key"], "criterion_label": cp.get("label_en") or cp["label"],
                    "euclidean_distance": euclidean, "cosine_similarity": cosine,
                })

            for ref_name in gac.REFERENCE_VOCAB_DATASETS:
                ref_centroid = per_source[ref_name]["centroid"]
                euclidean = float(np.linalg.norm(query - ref_centroid))
                reference_distance_rows.append({
                    "document_id": document_id, "exemplar_type": exemplar_type,
                    "initial_response_form": initial_response_form,
                    "reference_dataset": ref_name, "euclidean_distance": euclidean,
                })

            for mode in CENTROID_MODES:
                euclidean = float(np.linalg.norm(query - combined_refs[mode]))
                centroid_distance_rows.append({
                    "document_id": document_id, "exemplar_type": exemplar_type,
                    "initial_response_form": initial_response_form,
                    "mode": mode, "euclidean_distance_to_combined_expression_centroid": euclidean,
                })

            for r in nearest(query, lit_points, lit_vectors, NEAREST_K):
                nearest_lit_rows.append({"document_id": document_id, "exemplar_type": exemplar_type,
                                          "initial_response_form": initial_response_form, **r})
            for r in nearest(query, miv_points, miv_vectors, NEAREST_K):
                nearest_miv_rows.append({"document_id": document_id, "exemplar_type": exemplar_type,
                                          "initial_response_form": initial_response_form, **r})

            interview_neighbours = nearest(
                query, interview_points_all, interview_vectors_all, NEAREST_K, exclude_self=True,
            )
            if interview_neighbours:
                nearest_doc = gac.key_document_id(interview_neighbours[0]["key"])
                interview_neighbours[0]["same_document_as_exemplar"] = (nearest_doc == document_id)
            for r in interview_neighbours:
                nearest_interview_rows.append({"document_id": document_id, "exemplar_type": exemplar_type,
                                                "initial_response_form": initial_response_form, **r})

            exemplar_summary_rows.append({
                "document_id": document_id, "exemplar_type": exemplar_type,
                "initial_response_form": initial_response_form,
                "follow_up_examples": proto.get("follow_up_examples", ""),
                "status": "resolved", "source_expression_key": proto["source_expression_key"],
                "source_expression_label": proto["source_expression_label"],
            })

        # --- Equal-size-controlled reference comparison (Part 4) ---
        # reference_distances.csv above is a raw centroid distance (no
        # pool-size bias possible there -- a centroid summarizes the whole
        # set regardless of its size). This is the k-NN-based comparison,
        # which DOES need size control: all 25 prototypes are batched into
        # one gac.equal_size_reference_comparison call (one shared
        # per-repetition resample, not resampled per prototype).
        logger.info(
            "Equal-size reference comparison (n_reference=min of the 3 reference-set "
            "sizes, B=%d reps, batched over all %d prototypes)...",
            args.bootstrap_reps, len(prototype_points),
        )
        reference_vectors = {
            ref_name: shared_space.vectors[gac.source_dataset_indices(shared_space.points, ref_name)]
            for ref_name in gac.REFERENCE_VOCAB_DATASETS
        }
        proto_queries = {proto["document_id"]: vector for proto, vector in zip(prototype_points, prototype_vectors)}
        equal_size_result = gac.equal_size_reference_comparison(
            proto_queries, reference_vectors, k=NEAREST_K, seed=args.seed, reps=args.bootstrap_reps,
        )
        reference_comparison_equal_size_rows = []
        proto_by_document = {p["document_id"]: p for p in prototype_points}
        for document_id, per_reference in equal_size_result.items():
            proto = proto_by_document[document_id]
            for reference_dataset, stats in per_reference.items():
                for distance_statistic in ("mean", "median"):
                    summary = stats[f"nearest_k_distance_{distance_statistic}"]
                    reference_comparison_equal_size_rows.append({
                        "query_key": document_id, "query_label": document_id, "query_type": "prototype",
                        "document_id": document_id, "exemplar_type": proto["exemplar_type"],
                        "initial_response_form": proto["initial_response_form"],
                        "reference_dataset": reference_dataset,
                        "n_reference": stats["n_reference"], "k": stats["k"],
                        "repetitions": args.bootstrap_reps, "seed": args.seed,
                        "distance_statistic": distance_statistic, **summary,
                    })
        write_csv(out_dir / "reference_comparison_equal_size.csv", reference_comparison_equal_size_rows)

        # interview_prototypes.jsonl (and therefore prototype_points above)
        # structurally excludes every "unavailable" row -- build_interview_prototype_layer.py
        # never pads or substitutes for one. That means every OTHER table in
        # this module only ever sees the 25 resolved prototypes, and a
        # reader inferring "0 unavailable" purely from exemplar_summary.csv's
        # own contents would be wrong (there's nothing there to count in the
        # first place). Read initial_exemplars.csv directly, once, to add the
        # true unavailable/pending rows here -- exemplar_summary.csv becomes
        # the one place this module states the complete, honest picture.
        with open(gac.INITIAL_EXEMPLARS_CSV_PATH, encoding="utf-8") as f:
            all_csv_rows = list(csv.DictReader(f))
        for r in all_csv_rows:
            if r["review_status"] != "reviewed":
                exemplar_summary_rows.append({
                    "document_id": r["document_id"], "exemplar_type": r.get("exemplar_type", ""),
                    "initial_response_form": r.get("initial_response_form", ""),
                    "follow_up_examples": "", "status": r["review_status"],
                    "source_expression_key": "", "source_expression_label": "",
                })
        n_reviewed = sum(1 for r in all_csv_rows if r["review_status"] == "reviewed")
        n_unavailable = sum(1 for r in all_csv_rows if r["review_status"] == "unavailable")
        n_pending = sum(1 for r in all_csv_rows if r["review_status"] not in ("reviewed", "unavailable"))
        logger.info(
            "initial_exemplars.csv: %d reviewed, %d unavailable, %d pending (of %d total).",
            n_reviewed, n_unavailable, n_pending, len(all_csv_rows),
        )

        write_csv(out_dir / "exemplar_summary.csv", exemplar_summary_rows)
        write_csv(out_dir / "criterion_distances.csv", criterion_distance_rows)
        write_csv(out_dir / "reference_distances.csv", reference_distance_rows)
        write_csv(out_dir / "combined_expression_centroid_distances.csv", centroid_distance_rows)
        write_csv(out_dir / "nearest_literature.csv", nearest_lit_rows)
        write_csv(out_dir / "nearest_miviludes.csv", nearest_miv_rows)
        write_csv(out_dir / "nearest_other_interviews.csv", nearest_interview_rows)

        (out_dir / "EXPLORATORY_NOTICE.txt").write_text(
            f"{n_reviewed} reviewed / {n_unavailable} unavailable / {n_pending} pending "
            f"(of {len(all_csv_rows)} interviews total, per interviews/metadata/initial_exemplars.csv). "
            f"Only the {len(prototype_points)} reviewed rows have a geometric position (interview_prototypes.jsonl) "
            "and appear in every other table in this directory; unavailable/pending rows appear only in "
            "exemplar_summary.csv, with no distances computed for them. Every table here is "
            "exploratory/descriptive only -- no inferential statistics, no claim of representativeness. "
            "response_rank plays no role anywhere here; see audit_free_listing_rank.py for why. Reviewed "
            "points live in a separate prototype layer (build_interview_prototype_layer.py), not the "
            "ordinary pooled interview expressions -- some were excluded from the pooled space by its "
            "short-fragment filter but are legitimate here regardless.\n",
            encoding="utf-8",
        )

        gac.write_module_config(
            out_dir,
            input_sha256=shared_space.input_sha256,
            prototypes_path=str(args.prototypes),
            n_prototypes=len(prototype_points),
            nearest_k=NEAREST_K,
            centroid_modes=CENTROID_MODES,
            metric="euclidean",
            n_reviewed=n_reviewed,
            n_unavailable=n_unavailable,
            n_pending=n_pending,
            seed=args.seed,
            bootstrap_reps=args.bootstrap_reps,
            reference_vocab_datasets=gac.REFERENCE_VOCAB_DATASETS,
            transform_embedding_space_sha256=transform_metadata.get("embedding_space_sha256"),
            transform_n_points_fit=transform_metadata.get("n_points_fit"),
            git_commit=gac.git_commit_hash(),
        )
        gac.update_run_manifest(run_dir, MODULE_NAME, "completed", output_dir=out_dir)
    except Exception as e:
        gac.update_run_manifest(run_dir, MODULE_NAME, "failed", error=str(e))
        raise

    print(f"\nDone. Run {run_id} -> {out_dir}")


if __name__ == "__main__":
    main()
