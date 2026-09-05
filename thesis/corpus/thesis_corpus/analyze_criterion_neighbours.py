"""Distance from each of the 17 MIVILUDES criteria to the three expression
corpora (literature, MIVILUDES, interviews), in the shared 394-D space.

Query pool is exactly the three expression corpora -- never
miviludes_criteria itself (a criterion must never appear as its own or
another criterion's "nearest neighbour"), never reference/emergent points,
never a self-match.

Two explicitly separate kinds of output, never merged into one table:

  1. Qualitative retrieval (`full` mode only): nearest 10 labels in each
     expression corpus separately, for manual reading. Not
     corpus-size-adjusted -- literature's 35,621 points naturally offer
     closer nearest-neighbours than interviews' 204 purely by density.
  2. Quantitative controlled comparison (`equal_n_expression` mode,
     bootstrapped): for each criterion x corpus, the repeated-sampling
     mean/std/95% interval of mean and median nearest-10 distance. This
     controlled statistic, not the raw full-mode distance, is what feeds
     the criterion x corpus heatmap. `equal_weight` is a third, separate
     column: criterion-to-corpus-*centroid* distance, never nearest-10.

Language-representation sensitivity audit: the 17 criteria are embedded
French in the shared space, while MIVILUDES's own pooled expressions are
now English (this session's translation fix) -- literature is English,
interviews mixed. Checked before building this: build_shared_space.py
fits StandardScaler/PCA with .fit_transform() only, never persisted (no
joblib/pickle anywhere in that file), so there's no fitted transform to
reuse for projecting the criteria's stored English embeddings into the
394-D space. Per the resolved design: this audit runs entirely in raw
1024-d bge-m3 space instead (confirmed available for every corpus --
literature/interviews' own embedding_vector field, MIVILUDES's own
translated embedding_vector_en) using cosine similarity as primary WITHIN
THIS AUDIT SPECIFICALLY -- a deliberate, scoped exception to the toolkit's
general Euclidean-primary rule, since raw bge-m3 space isn't
standardized/PCA'd. Same two-output-type structure, run a second time.
Outputs kept in separate files, every row tagged
language_representation=french_primary_shared_space or english_sensitivity
-- never merged into one heatmap or summary statistic. Only cross-comparison
made: whether a criterion's nearest-corpus ordering changes between the two.

Usage (from thesis/corpus/):
    python -m thesis_corpus.analyze_criterion_neighbours
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict

import numpy as np

from thesis_corpus import geometric_analysis_common as gac

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("thesis_corpus.analyze_criterion_neighbours")

MODULE_NAME = "analyze_criterion_neighbours"
NEAREST_K = 10

LITERATURE_ARCHIVE_PATH = gac.PROCESSED_DIR / "literature" / "criterion_expressions.jsonl"
INTERVIEWS_ARCHIVE_PATH = gac.PROCESSED_DIR / "interviews" / "criterion_expressions.jsonl"
MIVILUDES_TRANSLATIONS_PATH = gac.PROCESSED_DIR / "miviludes" / "expression_translations_embedded.jsonl"
CRITERIA_EMBEDDED_PATH = gac.CORPUS_DIR / "metadata" / "miviludes_criteria_embedded.jsonl"


# ---------------------------------------------------------------------------
# 1. French-primary analysis (394-D shared space, Euclidean primary)
# ---------------------------------------------------------------------------

def qualitative_retrieval(criteria_points, criteria_vectors, corpus_pool: dict[str, tuple[list[dict], np.ndarray]]) -> list[dict]:
    rows = []
    for c_point, c_vector in zip(criteria_points, criteria_vectors):
        for corpus, (points, vectors) in corpus_pool.items():
            euclidean = gac.euclidean_distances(c_vector, vectors)
            cosine = gac.cosine_similarities(c_vector, vectors)
            order = np.argsort(euclidean)[:NEAREST_K]
            for rank, i in enumerate(order):
                rows.append({
                    "criterion_key": c_point["key"], "criterion_label": c_point.get("label_en") or c_point["label"],
                    "corpus": corpus, "rank": rank + 1,
                    "neighbor_key": points[i]["key"], "neighbor_label": points[i]["label"],
                    "euclidean_distance": float(euclidean[i]), "cosine_similarity": float(cosine[i]),
                    "comparison_type": "qualitative_retrieval",
                    "language_representation": "french_primary_shared_space",
                })
    return rows


def controlled_comparison_shared_space(shared_space: gac.SharedSpace, criteria_points, criteria_vectors, seed: int, reps: int) -> list[dict]:
    per_rep: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: {"mean": [], "median": []})
    for rep in range(reps):
        draw = gac.corpus_vectors_and_points(shared_space, "equal_n_expression", seed=seed + rep)
        for c_point, c_vector in zip(criteria_points, criteria_vectors):
            for corpus, (_points, vectors) in draw.items():
                distances = gac.euclidean_distances(c_vector, vectors)
                nearest_distances = np.sort(distances)[:NEAREST_K]
                per_rep[(c_point["key"], corpus)]["mean"].append(float(nearest_distances.mean()))
                per_rep[(c_point["key"], corpus)]["median"].append(float(np.median(nearest_distances)))

    rows = []
    for c_point in criteria_points:
        for corpus in gac.EXPRESSION_CORPORA:
            values = per_rep[(c_point["key"], corpus)]
            mean_summary = gac.bootstrap_summary(values["mean"])
            median_summary = gac.bootstrap_summary(values["median"])
            rows.append({
                "criterion_key": c_point["key"], "criterion_label": c_point.get("label_en") or c_point["label"],
                "corpus": corpus, "comparison_type": "controlled_comparison",
                "language_representation": "french_primary_shared_space",
                "mean_nearest10_distance_mean": mean_summary["mean"], "mean_nearest10_distance_std": mean_summary["std"],
                "mean_nearest10_distance_ci95_low": mean_summary["ci95_low"], "mean_nearest10_distance_ci95_high": mean_summary["ci95_high"],
                "median_nearest10_distance_mean": median_summary["mean"], "median_nearest10_distance_std": median_summary["std"],
                "median_nearest10_distance_ci95_low": median_summary["ci95_low"], "median_nearest10_distance_ci95_high": median_summary["ci95_high"],
            })
    return rows


def equal_weight_centroid_distances(per_source: dict, criteria_points, criteria_vectors) -> list[dict]:
    rows = []
    for c_point, c_vector in zip(criteria_points, criteria_vectors):
        for corpus in gac.EXPRESSION_CORPORA:
            euclidean = float(np.linalg.norm(c_vector - per_source[corpus]["centroid"]))
            rows.append({
                "criterion_key": c_point["key"], "criterion_label": c_point.get("label_en") or c_point["label"],
                "corpus": corpus, "comparison_type": "equal_weight_centroid_distance",
                "language_representation": "french_primary_shared_space",
                "euclidean_distance_to_corpus_centroid": euclidean,
            })
    return rows


# ---------------------------------------------------------------------------
# 2. English-sensitivity audit (raw 1024-d bge-m3, cosine primary)
# ---------------------------------------------------------------------------

def load_raw_vectors_by_key(path) -> dict[tuple, list[float]]:
    lookup = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            lookup[(item["document_id"], item["chunk_index"], item["embedding_text"])] = item["embedding_vector"]
    return lookup


def load_miviludes_english_vectors_by_key(path) -> dict[tuple, list[float]]:
    lookup = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            lookup[(item["document_id"], item["chunk_index"], item["text_fr"])] = item["embedding_vector_en"]
    return lookup


def build_raw_sensitivity_pool(shared_space: gac.SharedSpace) -> dict[str, tuple[list[dict], np.ndarray]]:
    """Rebuilds the three expression corpora's raw 1024-d bge-m3 vectors,
    restricted to exactly the points that survived pooling into
    embedding_space.jsonl (so the sensitivity audit's candidate pool
    matches the primary analysis's population). Literature/interviews use
    their own archive's raw embedding_vector (English/mixed, unchanged by
    the MIVILUDES translation work); MIVILUDES uses its English
    translation's raw embedding_vector_en."""
    lit_raw = load_raw_vectors_by_key(LITERATURE_ARCHIVE_PATH)
    interviews_raw = load_raw_vectors_by_key(INTERVIEWS_ARCHIVE_PATH)
    miviludes_raw_en = load_miviludes_english_vectors_by_key(MIVILUDES_TRANSLATIONS_PATH)

    pool: dict[str, tuple[list[dict], np.ndarray]] = {}
    missing = 0
    for corpus, raw_lookup, label_field in (
        ("literature", lit_raw, "label"),
        ("interviews", interviews_raw, "label"),
        ("miviludes", miviludes_raw_en, "label_fr"),
    ):
        idxs = gac.source_dataset_indices(shared_space.points, corpus)
        points, vectors = [], []
        for i in idxs:
            p = shared_space.points[i]
            document_id = gac.key_document_id(p["key"])
            chunk_index = gac.key_chunk_index(p["key"])
            key = (document_id, chunk_index, p[label_field])
            if key not in raw_lookup:
                missing += 1
                continue
            points.append(p)
            vectors.append(raw_lookup[key])
        pool[corpus] = (points, np.array(vectors, dtype=np.float64))
    if missing:
        logger.warning(
            "%d pooled expression point(s) could not be matched back to a raw "
            "archive vector for the sensitivity audit (skipped, not substituted).",
            missing,
        )
    return pool


def qualitative_retrieval_sensitivity(criteria_keys, criteria_vectors_en, corpus_pool) -> list[dict]:
    rows = []
    for c_key, c_vector in zip(criteria_keys, criteria_vectors_en):
        for corpus, (points, vectors) in corpus_pool.items():
            cosine = gac.cosine_similarities(c_vector, vectors)
            order = np.argsort(-cosine)[:NEAREST_K]
            for rank, i in enumerate(order):
                rows.append({
                    "criterion_key": c_key, "criterion_label": c_key,
                    "corpus": corpus, "rank": rank + 1,
                    "neighbor_key": points[i]["key"], "neighbor_label": points[i]["label"],
                    "cosine_similarity": float(cosine[i]),
                    "comparison_type": "qualitative_retrieval",
                    "language_representation": "english_sensitivity",
                })
    return rows


def controlled_comparison_sensitivity(corpus_pool, criteria_keys, criteria_vectors_en, seed: int, reps: int) -> list[dict]:
    n = min(len(points) for points, _ in corpus_pool.values())
    per_rep: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: {"mean": [], "median": []})

    for rep in range(reps):
        draw = {}
        for corpus, (points, vectors) in corpus_pool.items():
            idxs = list(range(len(points)))
            if corpus == "literature":
                sampled = gac.stratified_sample_by_document(points, idxs, n, seed + rep)
            else:
                sampled = gac.simple_random_sample(idxs, n, seed + rep)
            draw[corpus] = vectors[sampled]

        for c_key, c_vector in zip(criteria_keys, criteria_vectors_en):
            for corpus, vectors in draw.items():
                cosine = gac.cosine_similarities(c_vector, vectors)
                nearest_sims = np.sort(cosine)[-NEAREST_K:]
                per_rep[(c_key, corpus)]["mean"].append(float(nearest_sims.mean()))
                per_rep[(c_key, corpus)]["median"].append(float(np.median(nearest_sims)))

    rows = []
    for c_key in criteria_keys:
        for corpus in gac.EXPRESSION_CORPORA:
            values = per_rep[(c_key, corpus)]
            mean_summary = gac.bootstrap_summary(values["mean"])
            median_summary = gac.bootstrap_summary(values["median"])
            rows.append({
                "criterion_key": c_key, "criterion_label": c_key,
                "corpus": corpus, "comparison_type": "controlled_comparison",
                "language_representation": "english_sensitivity",
                "mean_nearest10_cosine_mean": mean_summary["mean"], "mean_nearest10_cosine_std": mean_summary["std"],
                "mean_nearest10_cosine_ci95_low": mean_summary["ci95_low"], "mean_nearest10_cosine_ci95_high": mean_summary["ci95_high"],
                "median_nearest10_cosine_mean": median_summary["mean"], "median_nearest10_cosine_std": median_summary["std"],
                "median_nearest10_cosine_ci95_low": median_summary["ci95_low"], "median_nearest10_cosine_ci95_high": median_summary["ci95_high"],
            })
    return rows


def compare_nearest_corpus_ordering(french_controlled: list[dict], english_controlled: list[dict]) -> list[dict]:
    """For each criterion, which corpus is nearest under each language
    representation -- French uses smallest mean Euclidean distance,
    English uses largest mean cosine similarity (each representation's own
    natural "closer" direction). Flags a criterion language-sensitive if
    the two orderings disagree on which corpus is nearest."""
    def nearest_corpus(rows, value_field, best):
        by_criterion = defaultdict(dict)
        for r in rows:
            by_criterion[r["criterion_key"]][r["corpus"]] = r[value_field]
        return {
            criterion: best(corpus_values, key=corpus_values.get)
            for criterion, corpus_values in by_criterion.items()
        }

    french_nearest = nearest_corpus(french_controlled, "mean_nearest10_distance_mean", min)
    english_nearest = nearest_corpus(english_controlled, "mean_nearest10_cosine_mean", max)

    rows = []
    for criterion_key in french_nearest:
        f_corpus = french_nearest[criterion_key]
        e_corpus = english_nearest.get(criterion_key)
        rows.append({
            "criterion_key": criterion_key,
            "nearest_corpus_french_primary": f_corpus,
            "nearest_corpus_english_sensitivity": e_corpus,
            "language_sensitive": f_corpus != e_corpus,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=gac.DEFAULT_SEED)
    parser.add_argument("--bootstrap-reps", type=int, default=gac.DEFAULT_BOOTSTRAP_REPS)
    args = parser.parse_args()

    logger.info("Loading %s ...", gac.EMBEDDING_SPACE_PATH)
    shared_space = gac.load_shared_space()

    run_id, run_dir = gac.get_or_create_run_dir(args.run_id)
    gac.init_run_manifest(run_dir, shared_space, defaults={"seed": args.seed, "bootstrap_reps": args.bootstrap_reps})
    gac.update_run_manifest(run_dir, MODULE_NAME, "running")
    out_dir = gac.module_run_dir(run_dir, MODULE_NAME)

    try:
        criteria_idxs = gac.source_dataset_indices(shared_space.points, "miviludes_criteria")
        criteria_points = [shared_space.points[i] for i in criteria_idxs]
        criteria_vectors = shared_space.vectors[criteria_idxs]
        full_pool = gac.corpus_vectors_and_points(shared_space, "full")
        per_source = gac.per_source_centroids_and_dispersion(shared_space)

        logger.info("French-primary: qualitative retrieval (full mode)...")
        qual_rows = qualitative_retrieval(criteria_points, criteria_vectors, full_pool)
        gac.write_csv(out_dir / "qualitative_retrieval_french_primary.csv", qual_rows)

        logger.info("French-primary: controlled comparison (equal_n_expression, B=%d)...", args.bootstrap_reps)
        french_controlled = controlled_comparison_shared_space(shared_space, criteria_points, criteria_vectors, args.seed, args.bootstrap_reps)
        gac.write_csv(out_dir / "controlled_comparison_french_primary.csv", french_controlled)

        logger.info("French-primary: equal_weight centroid distances...")
        eqw_rows = equal_weight_centroid_distances(per_source, criteria_points, criteria_vectors)
        gac.write_csv(out_dir / "equal_weight_centroid_distances_french_primary.csv", eqw_rows)

        logger.info("English-sensitivity audit: building raw bge-m3 pool...")
        criteria_en = []
        criteria_en_vectors = []
        with open(CRITERIA_EMBEDDED_PATH, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                criteria_en.append(item["id"])
                criteria_en_vectors.append(item["embedding_vector_en"])
        criteria_en_vectors = np.array(criteria_en_vectors, dtype=np.float64)

        sensitivity_pool = build_raw_sensitivity_pool(shared_space)

        logger.info("English-sensitivity: qualitative retrieval...")
        sens_qual_rows = qualitative_retrieval_sensitivity(criteria_en, criteria_en_vectors, sensitivity_pool)
        gac.write_csv(out_dir / "qualitative_retrieval_english_sensitivity.csv", sens_qual_rows)

        logger.info("English-sensitivity: controlled comparison (B=%d)...", args.bootstrap_reps)
        english_controlled = controlled_comparison_sensitivity(sensitivity_pool, criteria_en, criteria_en_vectors, args.seed, args.bootstrap_reps)
        gac.write_csv(out_dir / "controlled_comparison_english_sensitivity.csv", english_controlled)

        logger.info("Comparing nearest-corpus ordering between language representations...")
        ordering_rows = compare_nearest_corpus_ordering(french_controlled, english_controlled)
        gac.write_csv(out_dir / "language_sensitivity_ordering_comparison.csv", ordering_rows)
        n_sensitive = sum(1 for r in ordering_rows if r["language_sensitive"])
        logger.info("%d/%d criteria show a different nearest-corpus under French vs. English representation.", n_sensitive, len(ordering_rows))

        gac.write_module_config(
            out_dir,
            input_sha256=shared_space.input_sha256,
            seed=args.seed,
            bootstrap_reps=args.bootstrap_reps,
            nearest_k=NEAREST_K,
            n_criteria=len(criteria_points),
            n_language_sensitive_criteria=n_sensitive,
            french_primary_metric="euclidean",
            english_sensitivity_metric="cosine",
            git_commit=gac.git_commit_hash(),
        )
        gac.update_run_manifest(run_dir, MODULE_NAME, "completed", output_dir=out_dir)
    except Exception as e:
        gac.update_run_manifest(run_dir, MODULE_NAME, "failed", error=str(e))
        raise

    print(f"\nDone. Run {run_id} -> {out_dir}")


if __name__ == "__main__":
    main()
