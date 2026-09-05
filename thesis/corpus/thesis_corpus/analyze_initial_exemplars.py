"""Geometric analysis of the manually-reviewed interview initial exemplars
(thesis/corpus/interviews/metadata/initial_exemplars.csv) -- the
replacement for rank-based prototype analysis, which is retired outright
(see audit_free_listing_rank.py).

Refuses to run unless every row's `review_status` is "reviewed" or
"unavailable" -- a "pending" row means propose_initial_exemplars.py's
candidate hasn't been checked against the real transcript yet, and this
script must not treat an unreviewed guess as analytical input.

For each "reviewed" row, resolves `source_expression_key` (the virtual
document_id:chunk_index:occurrence key -- see
geometric_analysis_common.derive_interview_expression_keys) against the
*current* embedding_space.jsonl, and requires an exact match (after
whitespace normalization only -- no fuzzy matching, no
nearest-neighbour substitution) between the CSV's own
`source_expression_label` (what the pipeline's label read *at review
time*) and the resolved point's *current* label. This check is entirely
about pipeline-representation drift -- whether the pooled space still
contains what the reviewer actually looked at -- not about whether the
transcript wording matches the embedded text; those are allowed to differ
(`transcript_initial_exemplar_text` vs. `source_expression_label` are
deliberately separate fields; see initial_exemplars.csv's own header).

Two distinct failure modes, reported separately, either one halting the
run:
  (a) the virtual key doesn't exist in the current file at all -- most
      likely pooling-time filtering dropped that exact expression since
      the CSV was reviewed;
  (b) the virtual key exists but source_expression_label doesn't match the
      current label exactly -- possible drift, needs manual re-verification.

n~=26 throughout: every output is exploratory/descriptive only, never
given an inferential-statistics treatment.

Usage (from thesis/corpus/):
    python -m thesis_corpus.analyze_initial_exemplars
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re

import numpy as np

from thesis_corpus import geometric_analysis_common as gac

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("thesis_corpus.analyze_initial_exemplars")

MODULE_NAME = "analyze_initial_exemplars"
INITIAL_EXEMPLARS_CSV_PATH = gac.CORPUS_DIR / "interviews" / "metadata" / "initial_exemplars.csv"
NEAREST_K = 10
CENTROID_MODES = ("full", "reduced_literature", "equal_weight")

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.strip())


def load_initial_exemplars(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validate_review_status(rows: list[dict]) -> None:
    pending = [r["document_id"] for r in rows if r["review_status"] not in ("reviewed", "unavailable")]
    if pending:
        raise SystemExit(
            f"{len(pending)} row(s) in {INITIAL_EXEMPLARS_CSV_PATH} are not yet reviewed "
            f"(review_status must be 'reviewed' or 'unavailable'): {pending}. "
            "Run propose_initial_exemplars.py's output through manual review first -- "
            "this script refuses to treat an unreviewed candidate as analytical input."
        )


def resolve_reviewed_row(row: dict, points: list[dict], virtual_keys: dict[str, int]) -> int:
    key = row["source_expression_key"]
    if key not in virtual_keys:
        raise SystemExit(
            f"[{row['document_id']}] virtual key {key!r} does not resolve against the "
            "current embedding_space.jsonl -- most likely pooling-time "
            "deduplication/short-fragment filtering (build_shared_space.py) dropped this "
            "exact expression since the CSV was reviewed. Needs manual re-selection, not "
            "an automatic substitute."
        )
    index = virtual_keys[key]
    current_label = points[index]["label"]
    if _normalize(current_label) != _normalize(row["source_expression_label"]):
        raise SystemExit(
            f"[{row['document_id']}] virtual key {key!r} resolved to different text than "
            f"reviewed -- possible drift. CSV: {row['source_expression_label']!r}, "
            f"current: {current_label!r}. Needs manual re-verification, not a fuzzy match."
        )
    return index


def nearest(query: np.ndarray, candidate_points: list[dict], candidate_vectors: np.ndarray, k: int, exclude_index: int | None = None) -> list[dict]:
    euclidean = gac.euclidean_distances(query, candidate_vectors)
    cosine = gac.cosine_similarities(query, candidate_vectors)
    order = np.argsort(euclidean)
    results = []
    for i in order:
        if exclude_index is not None and i == exclude_index:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--input", type=type(INITIAL_EXEMPLARS_CSV_PATH), default=INITIAL_EXEMPLARS_CSV_PATH)
    args = parser.parse_args()

    logger.info("Loading %s ...", args.input)
    rows = load_initial_exemplars(args.input)
    validate_review_status(rows)
    logger.info("%d rows: %d reviewed, %d unavailable", len(rows),
                sum(1 for r in rows if r["review_status"] == "reviewed"),
                sum(1 for r in rows if r["review_status"] == "unavailable"))

    logger.info("Loading %s ...", gac.EMBEDDING_SPACE_PATH)
    shared_space = gac.load_shared_space()
    virtual_keys = gac.derive_interview_expression_keys(shared_space.points)

    run_id, run_dir = gac.get_or_create_run_dir(args.run_id)
    gac.init_run_manifest(run_dir, shared_space, defaults={})
    gac.update_run_manifest(run_dir, MODULE_NAME, "running")
    out_dir = gac.module_run_dir(run_dir, MODULE_NAME)

    try:
        criteria_idxs = gac.source_dataset_indices(shared_space.points, "miviludes_criteria")
        criteria_points = [shared_space.points[i] for i in criteria_idxs]
        criteria_vectors = shared_space.vectors[criteria_idxs]

        structural_idxs = gac.source_dataset_indices(shared_space.points, "structural_concepts")
        structural_points = [shared_space.points[i] for i in structural_idxs]
        structural_vectors = shared_space.vectors[structural_idxs]

        backbone_idxs = gac.source_dataset_indices(shared_space.points, "concept_backbone")
        backbone_points = [shared_space.points[i] for i in backbone_idxs]
        backbone_vectors = shared_space.vectors[backbone_idxs]

        lit_idxs = gac.source_dataset_indices(shared_space.points, "literature")
        lit_points = [shared_space.points[i] for i in lit_idxs]
        lit_vectors = shared_space.vectors[lit_idxs]

        miv_idxs = gac.source_dataset_indices(shared_space.points, "miviludes")
        miv_points = [shared_space.points[i] for i in miv_idxs]
        miv_vectors = shared_space.vectors[miv_idxs]

        interview_idxs = gac.source_dataset_indices(shared_space.points, "interviews")
        interview_points_all = [shared_space.points[i] for i in interview_idxs]
        interview_vectors_all = shared_space.vectors[interview_idxs]
        # nearest()'s exclude_index must be an index into this *subset* (0..203),
        # not into shared_space.points (0..44324) -- the exemplar's `index` below
        # is a full-array index, so it needs translating through this map.
        full_to_interview_local = {full_i: local_i for local_i, full_i in enumerate(interview_idxs)}

        per_source = gac.per_source_centroids_and_dispersion(shared_space)
        combined_refs = {mode: gac.combined_expression_reference(shared_space, mode) for mode in CENTROID_MODES}

        criterion_distance_rows = []
        reference_distance_rows = []
        centroid_distance_rows = []
        nearest_lit_rows, nearest_miv_rows, nearest_interview_rows = [], [], []
        exemplar_summary_rows = []
        n_unavailable = sum(1 for r in rows if r["review_status"] == "unavailable")

        for row in rows:
            if row["review_status"] == "unavailable":
                exemplar_summary_rows.append({
                    "document_id": row["document_id"], "exemplar_type": row.get("exemplar_type", "unclear"),
                    "status": "unavailable", "notes": row.get("notes", ""),
                })
                continue

            index = resolve_reviewed_row(row, shared_space.points, virtual_keys)
            query = shared_space.vectors[index]
            document_id = row["document_id"]
            exemplar_type = row["exemplar_type"]

            for cp, cv in zip(criteria_points, criteria_vectors):
                euclidean = float(np.linalg.norm(query - cv))
                cosine = float(gac.cosine_similarities(query, cv[np.newaxis, :])[0])
                criterion_distance_rows.append({
                    "document_id": document_id, "exemplar_type": exemplar_type,
                    "criterion_key": cp["key"], "criterion_label": cp.get("label_en") or cp["label"],
                    "euclidean_distance": euclidean, "cosine_similarity": cosine,
                })

            for ref_name, ref_centroid in (("structural_concepts", per_source["structural_concepts"]["centroid"]),
                                            ("concept_backbone", per_source["concept_backbone"]["centroid"])):
                euclidean = float(np.linalg.norm(query - ref_centroid))
                reference_distance_rows.append({
                    "document_id": document_id, "exemplar_type": exemplar_type,
                    "reference_dataset": ref_name, "euclidean_distance": euclidean,
                })

            for mode in CENTROID_MODES:
                euclidean = float(np.linalg.norm(query - combined_refs[mode]))
                centroid_distance_rows.append({
                    "document_id": document_id, "exemplar_type": exemplar_type,
                    "mode": mode, "euclidean_distance_to_combined_expression_centroid": euclidean,
                })

            for r in nearest(query, lit_points, lit_vectors, NEAREST_K):
                nearest_lit_rows.append({"document_id": document_id, "exemplar_type": exemplar_type, **r})
            for r in nearest(query, miv_points, miv_vectors, NEAREST_K):
                nearest_miv_rows.append({"document_id": document_id, "exemplar_type": exemplar_type, **r})

            interview_neighbours = nearest(
                query, interview_points_all, interview_vectors_all, NEAREST_K,
                exclude_index=full_to_interview_local.get(index),
            )
            if interview_neighbours:
                nearest_doc = gac.key_document_id(interview_neighbours[0]["key"])
                interview_neighbours[0]["same_document_as_exemplar"] = (nearest_doc == document_id)
            for r in interview_neighbours:
                nearest_interview_rows.append({"document_id": document_id, "exemplar_type": exemplar_type, **r})

            exemplar_summary_rows.append({
                "document_id": document_id, "exemplar_type": exemplar_type,
                "status": "resolved", "source_expression_key": row["source_expression_key"],
                "source_expression_label": row["source_expression_label"],
            })

        write_csv(out_dir / "exemplar_summary.csv", exemplar_summary_rows)
        write_csv(out_dir / "criterion_distances.csv", criterion_distance_rows)
        write_csv(out_dir / "reference_distances.csv", reference_distance_rows)
        write_csv(out_dir / "combined_expression_centroid_distances.csv", centroid_distance_rows)
        write_csv(out_dir / "nearest_literature.csv", nearest_lit_rows)
        write_csv(out_dir / "nearest_miviludes.csv", nearest_miv_rows)
        write_csv(out_dir / "nearest_other_interviews.csv", nearest_interview_rows)

        (out_dir / "EXPLORATORY_NOTICE.txt").write_text(
            "n~=26 interview exemplars. Every table in this directory is exploratory/"
            "descriptive only -- no inferential statistics, no claim of representativeness. "
            "response_rank plays no role anywhere here; see audit_free_listing_rank.py for why.\n",
            encoding="utf-8",
        )

        gac.write_module_config(
            out_dir,
            input_sha256=shared_space.input_sha256,
            initial_exemplars_csv=str(args.input),
            n_rows=len(rows),
            n_resolved=len(rows) - n_unavailable,
            n_unavailable=n_unavailable,
            nearest_k=NEAREST_K,
            centroid_modes=CENTROID_MODES,
            metric="euclidean",
            git_commit=gac.git_commit_hash(),
        )
        gac.update_run_manifest(run_dir, MODULE_NAME, "completed", output_dir=out_dir)
    except SystemExit as e:
        gac.update_run_manifest(run_dir, MODULE_NAME, "failed", error=str(e))
        raise
    except Exception as e:
        gac.update_run_manifest(run_dir, MODULE_NAME, "failed", error=str(e))
        raise

    print(f"\nDone. Run {run_id} -> {out_dir}")


if __name__ == "__main__":
    main()
