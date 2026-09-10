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

Language-representation sensitivity audit, THREE representations (was
two): the 17 criteria are embedded French in the shared space, while
MIVILUDES's own pooled expressions are now English (an earlier session's
translation fix) -- literature is English, interviews mixed.

  - `french_primary_shared_space`: the criteria as pooled, current 394-D
    shared PCA space. Authoritative throughout -- never replaced.
  - `english_sensitivity_shared_space` (new): now that
    build_shared_space.py persists its fitted StandardScaler+PCA
    (pca_transform.joblib, verified current -- see
    assert_english_criteria_compatible/project_criteria_to_shared_space
    below), the stored English criterion embeddings
    (embedding_vector_en) can be projected into the SAME 394-D coordinate
    system without refitting anything -- exactly the technique
    build_interview_prototype_layer.py already uses for the interview
    prototypes. This makes the sensitivity check apples-to-apples: same
    coordinates, same Euclidean metric, same corpus vectors, same
    controlled equal-n samples as French-primary -- unlike comparing
    French shared-space Euclidean distances to English raw-space cosine
    distances (the old, weaker comparison). Never added to
    embedding_space.jsonl, the PCA fit, or any point-count total --
    purely a run-local analysis input. This branch is REQUIRED for a
    complete run: any compatibility/projection failure here fails this
    whole module (not silently skipped), per the plan.
  - `english_raw_embedding_cosine` (was simply "english_sensitivity"
    before this representation existed -- renamed for disambiguation now
    that there are two different English representations): the original
    raw 1024-d bge-m3 cosine audit, retained as an optional SECONDARY
    embedding-level diagnostic, kept separate from both 394-D analyses
    (a reference-vocabulary comparison isn't meaningful here -- no
    reference-set points are ever projected into raw bge-m3 space).

Every output file name and `language_representation` tag identifies its
representation explicitly (never just a column in a file another
representation could accidentally overwrite or get grouped into) --
see main()'s file list. Query pool for every representation is exactly
the three expression corpora -- never miviludes_criteria itself, never
reference/emergent points, never a self-match. Two explicitly separate
kinds of output per representation, never merged into one table:

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

Also, for both 394-D representations (not the raw-cosine one): nearest
terms from each of the 3 reference vocabularies (concept_backbone,
structural_concepts, conceptnet_concepts) -- raw retrieval AND an
equal-size-controlled comparison (geometric_analysis_common's
equal_size_reference_comparison), kept in separate files per the toolkit's
"raw vs. controlled, never merged" convention.

Usage (from thesis/corpus/):
    python -m thesis_corpus.analyze_criterion_neighbours
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from collections import defaultdict

import joblib
import numpy as np

from thesis_corpus import geometric_analysis_common as gac

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("thesis_corpus.analyze_criterion_neighbours")

MODULE_NAME = "analyze_criterion_neighbours"
NEAREST_K = 10
NEAREST_K_ENTITIES = 5  # top-N emergent entities per criterion, for the compact report table/mini-map
COMPOSITION_K_VALUES = (10, 20)  # criterion_neighbour_composition's k values
EXPECTED_EMBEDDING_MODEL = "bge-m3"  # the only embedding model ever used in this pipeline

CRITERIA_EMBEDDED_PATH = gac.CORPUS_DIR / "metadata" / "miviludes_criteria_embedded.jsonl"


# ---------------------------------------------------------------------------
# 1. French-primary analysis (394-D shared space, Euclidean primary)
# ---------------------------------------------------------------------------

def qualitative_retrieval(
    criteria_points, criteria_vectors, corpus_pool: dict[str, tuple[list[dict], np.ndarray]],
    language_representation: str,
) -> list[dict]:
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
                    "language_representation": language_representation,
                })
    return rows


def controlled_comparison_shared_space(
    shared_space: gac.SharedSpace, criteria_points, criteria_vectors, seed: int, reps: int,
    language_representation: str,
) -> list[dict]:
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
                "language_representation": language_representation,
                "mean_nearest10_distance_mean": mean_summary["mean"], "mean_nearest10_distance_std": mean_summary["std"],
                "mean_nearest10_distance_ci95_low": mean_summary["ci95_low"], "mean_nearest10_distance_ci95_high": mean_summary["ci95_high"],
                "median_nearest10_distance_mean": median_summary["mean"], "median_nearest10_distance_std": median_summary["std"],
                "median_nearest10_distance_ci95_low": median_summary["ci95_low"], "median_nearest10_distance_ci95_high": median_summary["ci95_high"],
            })
    return rows


def criterion_neighbour_composition(
    shared_space: gac.SharedSpace, criteria_points, criteria_vectors,
    k_values: tuple[int, ...], seed: int, reps: int,
) -> list[dict]:
    """Equal-n, bootstrapped source-COMPOSITION of each criterion's k
    nearest neighbours -- the per-criterion analogue of
    analyze_cluster_structure.knn_composition's own equal_n_expression
    bootstrap loop, modeled directly on it, but querying with a fixed
    criterion vector each rep instead of a pool member.

    Distinct from controlled_comparison_shared_space above (which reports
    nearest-10 DISTANCE magnitude per corpus, each corpus's own equal-sized
    pool searched separately): this reports what FRACTION of a criterion's
    nearest neighbours, drawn from one COMBINED equal-sized pool across all
    three corpora, come from each source. A raw (non-equal-n) version of
    this composition metric would be dominated by literature purely by its
    ~97%-of-expression-points candidate-pool size, regardless of actual
    semantic content -- exactly the bias equal-n sampling exists to
    correct, same as analyze_cluster_structure's own headline numbers.

    French-primary representation only (`criteria_points`/`criteria_vectors`
    as loaded by main() -- authoritative elsewhere in this module already;
    computing this against all 3 language representations would triple
    the cost for no real gain).

    Fails loudly (ValueError) if any k in k_values exceeds a rep's actual
    equal-n candidate pool size, rather than silently truncating.

    Returns rows with NO automatic categorical label (no "folk-anchored"
    etc.) -- proportions stay continuous; criterion selection for any
    later write-up is a human judgment call made by inspecting this table,
    not something this function decides.
    """
    pool = gac.expression_pool_indices(shared_space.points)
    max_k = max(k_values)

    per_criterion_reps: dict[str, dict[int, dict[str, list[float]]]] = {
        c["key"]: {k: {corpus: [] for corpus in gac.EXPRESSION_CORPORA} for k in k_values}
        for c in criteria_points
    }
    equal_n_candidate_count: int | None = None

    for rep in range(reps):
        draw = gac.equal_n_expression_draw(shared_space.points, pool, seed + rep)
        all_vectors_parts, all_labels_parts = [], []
        for corpus, idxs in draw.items():
            all_vectors_parts.append(shared_space.vectors[idxs])
            all_labels_parts.append(np.full(len(idxs), corpus))
        all_vectors = np.concatenate(all_vectors_parts, axis=0)
        all_labels = np.concatenate(all_labels_parts, axis=0)
        n_candidates = len(all_labels)
        if equal_n_candidate_count is None:
            equal_n_candidate_count = n_candidates
        if max_k > n_candidates:
            raise ValueError(
                f"criterion_neighbour_composition: k={max_k} exceeds the equal-n candidate "
                f"pool size ({n_candidates}) at rep {rep} -- refusing to silently truncate; "
                "lower --bootstrap-... k or investigate why the equal-n pool shrank."
            )

        for c_point, c_vector in zip(criteria_points, criteria_vectors):
            euclidean = gac.euclidean_distances(c_vector, all_vectors)
            order = np.argsort(euclidean)[:max_k]
            ordered_labels = all_labels[order]
            for k in k_values:
                top_k_labels = ordered_labels[:k]
                rep_fractions = {
                    corpus: float(np.count_nonzero(top_k_labels == corpus)) / k
                    for corpus in gac.EXPRESSION_CORPORA
                }
                # Invariant: fractions must sum to 1 (within float tolerance)
                # per (criterion, rep, k) BEFORE aggregation -- every
                # candidate in the pool is from exactly one of the 3
                # expression corpora, so this must hold by construction.
                rep_sum = sum(rep_fractions.values())
                if abs(rep_sum - 1.0) > 1e-9:
                    raise AssertionError(
                        f"criterion={c_point['key']} k={k} rep={rep}: source fractions sum to "
                        f"{rep_sum}, not 1 -- stop and investigate rather than aggregate a broken rep."
                    )
                for corpus, fraction in rep_fractions.items():
                    per_criterion_reps[c_point["key"]][k][corpus].append(fraction)

    rows = []
    for c_point in criteria_points:
        for k in k_values:
            per_corpus_summary = {
                corpus: gac.bootstrap_summary(per_criterion_reps[c_point["key"]][k][corpus])
                for corpus in gac.EXPRESSION_CORPORA
            }
            # Invariant: reported MEAN fractions must also sum to ~1
            # (subject only to rounding) -- linearity of expectation over
            # per-rep sums that are each exactly 1.
            mean_sum = sum(s["mean"] for s in per_corpus_summary.values())
            if abs(mean_sum - 1.0) > 1e-6:
                raise AssertionError(
                    f"criterion={c_point['key']} k={k}: mean fractions sum to {mean_sum}, not ~1."
                )
            for corpus, summary in per_corpus_summary.items():
                rows.append({
                    "criterion_key": c_point["key"],
                    "criterion_label_fr": c_point["label"],
                    "criterion_label_en": c_point.get("label_en") or c_point["label"],
                    "k": k, "source": corpus,
                    "mean_fraction": summary["mean"], "ci95_low": summary["ci95_low"], "ci95_high": summary["ci95_high"],
                    "equal_n_candidate_count": equal_n_candidate_count,
                    "bootstrap_reps": reps, "seed": seed,
                })
    return rows


def equal_weight_centroid_distances(
    per_source: dict, criteria_points, criteria_vectors, language_representation: str,
) -> list[dict]:
    rows = []
    for c_point, c_vector in zip(criteria_points, criteria_vectors):
        for corpus in gac.EXPRESSION_CORPORA:
            euclidean = float(np.linalg.norm(c_vector - per_source[corpus]["centroid"]))
            rows.append({
                "criterion_key": c_point["key"], "criterion_label": c_point.get("label_en") or c_point["label"],
                "corpus": corpus, "comparison_type": "equal_weight_centroid_distance",
                "language_representation": language_representation,
                "euclidean_distance_to_corpus_centroid": euclidean,
            })
    return rows


def reference_comparison_for_representation(
    criteria_points, criteria_vectors, reference_pools: dict[str, tuple[list[dict], np.ndarray]],
    language_representation: str, seed: int, reps: int,
) -> tuple[list[dict], list[dict]]:
    """Nearest reference-vocabulary terms (raw retrieval) + equal-size-
    controlled comparison, for one 394-D representation (French-primary or
    English-sensitivity-shared-space -- never the raw-cosine
    representation, where no reference-set points exist in that space)."""
    raw_rows = []
    for c_point, c_vector in zip(criteria_points, criteria_vectors):
        for ref_dataset, (ref_points, ref_vectors) in reference_pools.items():
            euclidean = gac.euclidean_distances(c_vector, ref_vectors)
            cosine = gac.cosine_similarities(c_vector, ref_vectors)
            order = np.argsort(euclidean)[:NEAREST_K]
            for rank, i in enumerate(order):
                raw_rows.append({
                    "criterion_key": c_point["key"], "criterion_label": c_point.get("label_en") or c_point["label"],
                    "reference_dataset": ref_dataset, "rank": rank + 1,
                    "neighbor_key": ref_points[i]["key"], "neighbor_label": ref_points[i]["label"],
                    "euclidean_distance": float(euclidean[i]), "cosine_similarity": float(cosine[i]),
                    "comparison_type": "qualitative_retrieval",
                    "language_representation": language_representation,
                })

    queries = {c_point["key"]: c_vector for c_point, c_vector in zip(criteria_points, criteria_vectors)}
    reference_vectors_only = {name: vecs for name, (_pts, vecs) in reference_pools.items()}
    equal_size_result = gac.equal_size_reference_comparison(queries, reference_vectors_only, k=NEAREST_K, seed=seed, reps=reps)
    criterion_label_by_key = {c_point["key"]: c_point.get("label_en") or c_point["label"] for c_point in criteria_points}
    equal_size_rows = []
    for criterion_key, per_reference in equal_size_result.items():
        for reference_dataset, stats in per_reference.items():
            for distance_statistic in ("mean", "median"):
                summary = stats[f"nearest_k_distance_{distance_statistic}"]
                equal_size_rows.append({
                    "query_key": criterion_key, "query_label": criterion_label_by_key[criterion_key], "query_type": "criterion",
                    "criterion_key": criterion_key, "reference_dataset": reference_dataset,
                    "n_reference": stats["n_reference"], "k": stats["k"],
                    "repetitions": reps, "seed": seed, "distance_statistic": distance_statistic,
                    "language_representation": language_representation, **summary,
                })
    return raw_rows, equal_size_rows


# ---------------------------------------------------------------------------
# 1b. English-sensitivity-in-shared-space projection (new -- uses the now-
# persisted StandardScaler+PCA, same technique build_interview_prototype_layer.py
# uses for the interview prototypes)
# ---------------------------------------------------------------------------

def load_criteria_english_records(path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def assert_english_criteria_compatible(records: list[dict], scaler) -> None:
    """Required compatibility checks before projecting embedding_vector_en
    through the persisted transform -- fail loudly, never coerce/truncate/
    pad/silently substitute. This branch is REQUIRED for a complete run
    (see module docstring): any failure here raises, which (via main()'s
    surrounding try/except) fails this entire module and records it
    'failed' in RUN_MANIFEST.json -- never just skips this one branch
    while the rest of the module reports 'completed'."""
    if len(records) != 17:
        raise SystemExit(f"Expected 17 MIVILUDES criteria, found {len(records)} in {CRITERIA_EMBEDDED_PATH}")
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"Duplicate criterion id(s) in {CRITERIA_EMBEDDED_PATH}: {ids}")
    for r in records:
        vec = np.asarray(r["embedding_vector_en"], dtype=np.float64)
        if vec.shape[0] != scaler.n_features_in_:
            raise SystemExit(
                f"Criterion {r['id']!r}'s embedding_vector_en has dimensionality {vec.shape[0]}, "
                f"but the persisted scaler (pca_transform.joblib) expects {scaler.n_features_in_} "
                "-- refusing to project. Re-embed with the same model/pipeline the shared space "
                "was built from, or re-run build_shared_space.py's persistence step if the model changed."
            )
        if not np.all(np.isfinite(vec)):
            raise SystemExit(f"Criterion {r['id']!r}'s embedding_vector_en contains non-finite values.")
        if r.get("embedding_model") != EXPECTED_EMBEDDING_MODEL:
            raise SystemExit(
                f"Criterion {r['id']!r} was embedded with model {r.get('embedding_model')!r}, "
                f"expected {EXPECTED_EMBEDDING_MODEL!r} -- the model every other point in the "
                "shared space (and the persisted transform) was built from."
            )


def project_criteria_to_shared_space(
    records: list[dict], french_points_by_key: dict[str, dict], scaler, pca, k: int,
) -> tuple[list[dict], np.ndarray]:
    """Projects the stored English criterion embeddings into the SAME 394-D
    coordinate system the French-primary criteria already live in, via the
    current persisted scaler/pca -- never refitting anything. Analysis-only:
    the returned points/vectors are never written to embedding_space.jsonl,
    the PCA fit, or any point-count total."""
    raw_vectors = np.array([r["embedding_vector_en"] for r in records], dtype=np.float64)
    projected = pca.transform(scaler.transform(raw_vectors))[:, :k]
    points = []
    for r in records:
        fr_point = french_points_by_key[r["id"]]
        points.append({"key": r["id"], "label": fr_point["label"], "label_en": r.get("criterion_en") or fr_point.get("label_en")})
    return points, projected


# ---------------------------------------------------------------------------
# 2. English raw-embedding-cosine audit (raw 1024-d bge-m3, cosine primary) --
# an optional SECONDARY embedding-level diagnostic, separate from both 394-D
# representations above (no reference-vocabulary comparison here: no
# reference-set points are ever projected into raw bge-m3 space).
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
    # Never assume v1's fixed paths -- derive from whichever shared space was
    # actually loaded (see gac.resolve_archive_paths). The MIVILUDES
    # translations file lives alongside its archive in both v1 and v2
    # (expression_translations_embedded.jsonl next to criterion_expressions.jsonl),
    # so it's derived as that sibling rather than a separate hardcoded constant.
    archive_paths = gac.resolve_archive_paths(shared_space)
    miviludes_translations_path = archive_paths["miviludes"].parent / "expression_translations_embedded.jsonl"
    lit_raw = load_raw_vectors_by_key(archive_paths["literature"])
    interviews_raw = load_raw_vectors_by_key(archive_paths["interviews"])
    miviludes_raw_en = load_miviludes_english_vectors_by_key(miviludes_translations_path)

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
                    "language_representation": "english_raw_embedding_cosine",
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
                "language_representation": "english_raw_embedding_cosine",
                "mean_nearest10_cosine_mean": mean_summary["mean"], "mean_nearest10_cosine_std": mean_summary["std"],
                "mean_nearest10_cosine_ci95_low": mean_summary["ci95_low"], "mean_nearest10_cosine_ci95_high": mean_summary["ci95_high"],
                "median_nearest10_cosine_mean": median_summary["mean"], "median_nearest10_cosine_std": median_summary["std"],
                "median_nearest10_cosine_ci95_low": median_summary["ci95_low"], "median_nearest10_cosine_ci95_high": median_summary["ci95_high"],
            })
    return rows


def _nearest_corpus_by_criterion(rows: list[dict], value_field: str, best) -> dict[str, str]:
    by_criterion = defaultdict(dict)
    for r in rows:
        by_criterion[r["criterion_key"]][r["corpus"]] = r[value_field]
    return {
        criterion: best(corpus_values, key=corpus_values.get)
        for criterion, corpus_values in by_criterion.items()
    }


def compare_nearest_corpus_ordering(
    label_a: str, rows_a: list[dict], value_field_a: str, best_a,
    label_b: str, rows_b: list[dict], value_field_b: str, best_b,
) -> list[dict]:
    """For each criterion, which corpus is nearest under representation A
    vs. representation B -- each representation's own natural "closer"
    direction (smallest mean Euclidean distance for a 394-D shared-space
    representation, largest mean cosine similarity for the raw-cosine
    one). Flags a criterion language-sensitive if the two orderings
    disagree. Generic over which two representations are being compared
    so the same function serves both pairs main() computes (the primary,
    same-metric french_primary_shared_space vs.
    english_sensitivity_shared_space comparison, and the secondary
    french_primary_shared_space vs. english_raw_embedding_cosine one) --
    never averaging/merging the underlying distances, only comparing which
    corpus each independently identifies as nearest."""
    nearest_a = _nearest_corpus_by_criterion(rows_a, value_field_a, best_a)
    nearest_b = _nearest_corpus_by_criterion(rows_b, value_field_b, best_b)

    rows = []
    for criterion_key in nearest_a:
        corpus_a = nearest_a[criterion_key]
        corpus_b = nearest_b.get(criterion_key)
        rows.append({
            "criterion_key": criterion_key,
            "comparison_pair": f"{label_a}_vs_{label_b}",
            f"nearest_corpus_{label_a}": corpus_a,
            f"nearest_corpus_{label_b}": corpus_b,
            "language_sensitive": corpus_a != corpus_b,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--shared-space-dir", type=Path, default=None,
                         help="Use a different pooled space (e.g. processed/shared_space_v2/) instead of v1's "
                              "processed/shared_space/; also switches the run-output root to a sibling "
                              "processed/analysis_v2/ directory so v1 and v2 runs are never mixed.")
    parser.add_argument("--seed", type=int, default=gac.DEFAULT_SEED)
    parser.add_argument("--bootstrap-reps", type=int, default=gac.DEFAULT_BOOTSTRAP_REPS)
    args = parser.parse_args()

    embedding_space_path, interview_prototypes_path, analysis_root = gac.resolve_space_paths(args.shared_space_dir)
    logger.info("Loading %s ...", embedding_space_path)
    shared_space = gac.load_shared_space(embedding_space_path)

    run_id, run_dir = gac.get_or_create_run_dir(args.run_id, analysis_root)
    gac.init_run_manifest(run_dir, shared_space, defaults={"seed": args.seed, "bootstrap_reps": args.bootstrap_reps})
    gac.update_run_manifest(run_dir, MODULE_NAME, "running")
    out_dir = gac.module_run_dir(run_dir, MODULE_NAME)

    try:
        criteria_idxs = gac.source_dataset_indices(shared_space.points, "miviludes_criteria")
        criteria_points = [shared_space.points[i] for i in criteria_idxs]
        criteria_vectors = shared_space.vectors[criteria_idxs]
        criteria_points_by_key = {p["key"]: p for p in criteria_points}
        full_pool = gac.corpus_vectors_and_points(shared_space, "full")
        per_source = gac.per_source_centroids_and_dispersion(shared_space)

        # Nearest EMERGENT ENTITIES per criterion -- deliberately a separate
        # pool/output from full_pool above, never merged into it: full_pool
        # is documented throughout this module as "exactly the three
        # expression corpora, never reference/emergent points," an
        # invariant several other deliverables (language-sensitivity
        # comparison, controlled comparison) depend on. qualitative_retrieval
        # is generic over any {label: (points, vectors)} pool, so this reuses
        # it unchanged rather than writing a second retrieval function.
        entity_idxs = gac.source_dataset_indices(shared_space.points, "emergent_entities")
        entity_pool = {
            "emergent_entities": ([shared_space.points[i] for i in entity_idxs], shared_space.vectors[entity_idxs]),
        }
        logger.info("french_primary_shared_space: nearest emergent entities per criterion (top-%d)...", NEAREST_K_ENTITIES)
        qual_rows_entities = qualitative_retrieval(criteria_points, criteria_vectors, entity_pool, "french_primary_shared_space")
        qual_rows_entities = [r for r in qual_rows_entities if r["rank"] <= NEAREST_K_ENTITIES]
        gac.write_csv(out_dir / "qualitative_retrieval_entities.csv", qual_rows_entities)
        reference_pools = {
            ref_dataset: (
                [shared_space.points[i] for i in gac.source_dataset_indices(shared_space.points, ref_dataset)],
                shared_space.vectors[gac.source_dataset_indices(shared_space.points, ref_dataset)],
            )
            for ref_dataset in gac.REFERENCE_VOCAB_DATASETS
        }

        # === Representation 1: french_primary_shared_space (authoritative) ===
        logger.info("french_primary_shared_space: qualitative retrieval (full mode)...")
        qual_rows_fr = qualitative_retrieval(criteria_points, criteria_vectors, full_pool, "french_primary_shared_space")
        gac.write_csv(out_dir / "qualitative_retrieval_french_primary_shared_space.csv", qual_rows_fr)

        logger.info("french_primary_shared_space: controlled comparison (equal_n_expression, B=%d)...", args.bootstrap_reps)
        french_controlled = controlled_comparison_shared_space(
            shared_space, criteria_points, criteria_vectors, args.seed, args.bootstrap_reps, "french_primary_shared_space",
        )
        gac.write_csv(out_dir / "controlled_comparison_french_primary_shared_space.csv", french_controlled)

        logger.info("french_primary_shared_space: equal_weight centroid distances...")
        eqw_rows_fr = equal_weight_centroid_distances(per_source, criteria_points, criteria_vectors, "french_primary_shared_space")
        gac.write_csv(out_dir / "equal_weight_centroid_distances_french_primary_shared_space.csv", eqw_rows_fr)

        logger.info("french_primary_shared_space: reference-vocabulary comparison (raw + equal-size)...")
        ref_raw_fr, ref_eqsize_fr = reference_comparison_for_representation(
            criteria_points, criteria_vectors, reference_pools, "french_primary_shared_space", args.seed, args.bootstrap_reps,
        )
        gac.write_csv(out_dir / "nearest_reference_terms_raw_french_primary_shared_space.csv", ref_raw_fr)
        gac.write_csv(out_dir / "reference_comparison_equal_size_french_primary_shared_space.csv", ref_eqsize_fr)

        # --- New: equal-n, bootstrapped criterion-neighbour composition ---
        # (French-primary representation only -- see criterion_neighbour_composition's
        # own docstring for why). A candidate figure (not a final main-text/
        # appendix placement decision) is produced by generate_figures.py
        # from this CSV, k=10 only.
        logger.info(
            "french_primary_shared_space: equal-n criterion-neighbour composition "
            "(k=%s, B=%d)...", COMPOSITION_K_VALUES, args.bootstrap_reps,
        )
        composition_rows = criterion_neighbour_composition(
            shared_space, criteria_points, criteria_vectors,
            k_values=COMPOSITION_K_VALUES, seed=args.seed, reps=args.bootstrap_reps,
        )
        gac.write_csv(out_dir / "criterion_equal_n_neighbour_composition.csv", composition_rows)

        # === Representation 2 (new, REQUIRED): english_sensitivity_shared_space ===
        # Projects the stored English criterion embeddings into the SAME
        # 394-D space via the persisted transform -- see module docstring.
        # Any failure in this block fails the whole module (no local
        # try/except suppressing it) -- a partial run that skips this
        # representation is not an acceptable outcome of this run.
        pca_transform_path = embedding_space_path.parent / "pca_transform.joblib"
        pca_transform_metadata_path = embedding_space_path.parent / "pca_transform_metadata.json"
        logger.info("Loading persisted transform (%s) for english_sensitivity_shared_space...", pca_transform_path)
        if not pca_transform_metadata_path.exists():
            raise SystemExit(f"Missing {pca_transform_metadata_path} -- cannot verify transform currency.")
        transform_metadata = json.loads(pca_transform_metadata_path.read_text(encoding="utf-8"))
        if transform_metadata.get("embedding_space_sha256") != shared_space.input_sha256:
            raise SystemExit(
                f"pca_transform_metadata.json's embedding_space_sha256 does not match the "
                f"current embedding_space.jsonl ({shared_space.input_sha256}) -- the shared space "
                "was rebuilt since the persisted transform was fit. Re-run build_shared_space.py "
                "before running this module."
            )
        # joblib.load executes arbitrary code on untrusted input, but this file is
        # produced locally by build_shared_space.py earlier in this same pipeline
        # (never downloaded or user-supplied), same trust boundary already accepted
        # by build_interview_prototype_layer.py for the identical artifact.
        transform = joblib.load(pca_transform_path)
        scaler, pca = transform["scaler"], transform["pca"]
        criteria_en_records = load_criteria_english_records(CRITERIA_EMBEDDED_PATH)
        assert_english_criteria_compatible(criteria_en_records, scaler)
        criteria_points_en, criteria_vectors_en = project_criteria_to_shared_space(
            criteria_en_records, criteria_points_by_key, scaler, pca, k=shared_space.vectors.shape[1],
        )
        logger.info("english_sensitivity_shared_space: %d criteria projected via persisted transform, compatibility OK.", len(criteria_points_en))

        logger.info("english_sensitivity_shared_space: qualitative retrieval (full mode)...")
        qual_rows_en = qualitative_retrieval(criteria_points_en, criteria_vectors_en, full_pool, "english_sensitivity_shared_space")
        gac.write_csv(out_dir / "qualitative_retrieval_english_sensitivity_shared_space.csv", qual_rows_en)

        logger.info("english_sensitivity_shared_space: controlled comparison (equal_n_expression, B=%d)...", args.bootstrap_reps)
        english_shared_space_controlled = controlled_comparison_shared_space(
            shared_space, criteria_points_en, criteria_vectors_en, args.seed, args.bootstrap_reps, "english_sensitivity_shared_space",
        )
        gac.write_csv(out_dir / "controlled_comparison_english_sensitivity_shared_space.csv", english_shared_space_controlled)

        logger.info("english_sensitivity_shared_space: equal_weight centroid distances...")
        eqw_rows_en = equal_weight_centroid_distances(per_source, criteria_points_en, criteria_vectors_en, "english_sensitivity_shared_space")
        gac.write_csv(out_dir / "equal_weight_centroid_distances_english_sensitivity_shared_space.csv", eqw_rows_en)

        logger.info("english_sensitivity_shared_space: reference-vocabulary comparison (raw + equal-size)...")
        ref_raw_en, ref_eqsize_en = reference_comparison_for_representation(
            criteria_points_en, criteria_vectors_en, reference_pools, "english_sensitivity_shared_space", args.seed, args.bootstrap_reps,
        )
        gac.write_csv(out_dir / "nearest_reference_terms_raw_english_sensitivity_shared_space.csv", ref_raw_en)
        gac.write_csv(out_dir / "reference_comparison_equal_size_english_sensitivity_shared_space.csv", ref_eqsize_en)

        # === Representation 3 (optional secondary diagnostic): english_raw_embedding_cosine ===
        logger.info("english_raw_embedding_cosine: building raw bge-m3 pool...")
        criteria_en_ids = [r["id"] for r in criteria_en_records]
        criteria_en_raw_vectors = np.array([r["embedding_vector_en"] for r in criteria_en_records], dtype=np.float64)

        sensitivity_pool = build_raw_sensitivity_pool(shared_space)

        logger.info("english_raw_embedding_cosine: qualitative retrieval...")
        sens_qual_rows = qualitative_retrieval_sensitivity(criteria_en_ids, criteria_en_raw_vectors, sensitivity_pool)
        gac.write_csv(out_dir / "qualitative_retrieval_english_raw_embedding_cosine.csv", sens_qual_rows)

        logger.info("english_raw_embedding_cosine: controlled comparison (B=%d)...", args.bootstrap_reps)
        english_raw_controlled = controlled_comparison_sensitivity(sensitivity_pool, criteria_en_ids, criteria_en_raw_vectors, args.seed, args.bootstrap_reps)
        gac.write_csv(out_dir / "controlled_comparison_english_raw_embedding_cosine.csv", english_raw_controlled)

        # === Cross-representation ordering comparison (never merges distances) ===
        logger.info("Comparing nearest-corpus ordering across representations...")
        ordering_rows = compare_nearest_corpus_ordering(
            "french_primary_shared_space", french_controlled, "mean_nearest10_distance_mean", min,
            "english_sensitivity_shared_space", english_shared_space_controlled, "mean_nearest10_distance_mean", min,
        ) + compare_nearest_corpus_ordering(
            "french_primary_shared_space", french_controlled, "mean_nearest10_distance_mean", min,
            "english_raw_embedding_cosine", english_raw_controlled, "mean_nearest10_cosine_mean", max,
        )
        gac.write_csv(out_dir / "language_sensitivity_ordering_comparison.csv", ordering_rows)
        n_sensitive = sum(1 for r in ordering_rows if r["language_sensitive"])
        logger.info("%d/%d representation-pair x criterion rows show a different nearest-corpus.", n_sensitive, len(ordering_rows))

        gac.write_module_config(
            out_dir,
            input_sha256=shared_space.input_sha256,
            seed=args.seed,
            bootstrap_reps=args.bootstrap_reps,
            nearest_k=NEAREST_K,
            n_criteria=len(criteria_points),
            n_language_sensitive_rows=n_sensitive,
            reference_vocab_datasets=gac.REFERENCE_VOCAB_DATASETS,
            criterion_neighbour_composition={
                "k_values": COMPOSITION_K_VALUES,
                "equal_n_candidate_count": composition_rows[0]["equal_n_candidate_count"] if composition_rows else None,
                "bootstrap_reps": args.bootstrap_reps,
                "seed": args.seed,
                "source_order": gac.EXPRESSION_CORPORA,
                "criterion_representation": "french_primary_shared_space",
            },
            language_representations=(
                "french_primary_shared_space", "english_sensitivity_shared_space", "english_raw_embedding_cosine",
            ),
            french_primary_metric="euclidean",
            english_sensitivity_shared_space_metric="euclidean",
            english_raw_embedding_cosine_metric="cosine",
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
