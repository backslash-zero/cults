"""Cluster structure of the three expression corpora (literature, MIVILUDES,
interviews) in the shared 394-D space: k-nearest-neighbour source
composition (k=10, k=20), silhouette score by source_dataset, and every
2-D UMAP fit this toolkit uses.

Pool is exactly the three expression corpora and nothing else --
miviludes_criteria (a fixed reference list, not a corpus expression) and
all reference/emergent points are excluded. Every k-NN query excludes its
own point from its own results (k+1 requested, self dropped), so
composition is always computed from exactly k *other* points.

Naming discipline: "full" means every applicable point, no subsampling,
ever. k-NN composition is tractable as genuine `full` (an exact
nearest-neighbour query over all 36,574 points, not a full pairwise-distance
matrix). Silhouette's exact form needs an O(n^2) distance matrix that's
impractical at that scale, so its full-population stand-in is a
reproducible fixed-size sample explicitly named `full_sampled_pointwise`
-- never called "full". `reduced_literature` (2,500+732+204 points) is
small enough for exact k-NN *and* exact silhouette. `equal_n_expression`
(bootstrapped, B repetitions, mean/std/95% interval) is the primary
controlled comparison for both statistics; `full`/`full_sampled_pointwise`/
`reduced_literature` are descriptive/sensitivity views.

This module also owns every 2-D UMAP fit in the toolkit -- the only place
`umap.UMAP(...).fit_transform` is ever called. Input populations, saved as
separate coordinate files with a manifest each (input population, seed,
input checksum, UMAP parameters, umap-learn version, output checksum):
"overview" (all 44,325 points), "expression_sampled" (the same
full_sampled_pointwise subset silhouette uses), "equal_n_diagnostic" (one
single seeded equal_n_expression draw, never averaged across repetitions),
and -- only if build_interview_prototype_layer.py has already been run --
"with_interview_prototypes" (overview plus the interview initial-exemplar
prototype layer, fit fresh rather than reusing "overview"'s coordinates,
since several prototypes were dropped by build_shared_space.py's pooling
filter and have no position in the ordinary pooled space to look up at
all). generate_figures.py only ever reads these files.

Usage (from thesis/corpus/):
    python -m thesis_corpus.analyze_cluster_structure
    python -m thesis_corpus.analyze_cluster_structure --umap-param-grid
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import umap
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from thesis_corpus import geometric_analysis_common as gac

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("thesis_corpus.analyze_cluster_structure")

MODULE_NAME = "analyze_cluster_structure"
K_VALUES = (10, 20)
UMAP_PARAM_GRID = [(10, 0.1), (20, 0.2), (50, 0.3)]


def _concat_pool(pool: dict[str, tuple[list[dict], np.ndarray]]) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Flattens a {corpus: (points, vectors)} pool into one
    (points, vectors, labels) triple, `labels` being each point's
    source_dataset -- the shape every k-NN/silhouette/UMAP call here
    actually operates on."""
    all_points, all_vectors, all_labels = [], [], []
    for corpus, (points, vectors) in pool.items():
        all_points.extend(points)
        all_vectors.append(vectors)
        all_labels.extend([corpus] * len(points))
    return all_points, np.concatenate(all_vectors, axis=0), np.array(all_labels)


def full_sampled_pointwise_pool(
    shared_space: gac.SharedSpace, sample_size: int, seed: int,
) -> dict[str, tuple[list[dict], np.ndarray]]:
    """A reproducible uniform random sample of `sample_size` points drawn
    from the union of the three expression corpora -- NOT stratified by
    corpus, so it preserves the population's natural (imbalanced)
    composition, standing in for "full" wherever an O(n^2) exact
    computation (silhouette) is impractical at the true full scale."""
    pool = gac.expression_pool_indices(shared_space.points)
    all_indices = [i for idxs in pool.values() for i in idxs]
    sampled = gac.simple_random_sample(all_indices, sample_size, seed)
    result: dict[str, tuple[list[dict], np.ndarray]] = {c: ([], []) for c in gac.EXPRESSION_CORPORA}
    buckets: dict[str, list[int]] = {c: [] for c in gac.EXPRESSION_CORPORA}
    for i in sampled:
        buckets[shared_space.points[i]["source_dataset"]].append(i)
    for c, idxs in buckets.items():
        result[c] = ([shared_space.points[i] for i in idxs], shared_space.vectors[idxs])
    return result


def knn_composition(
    pool: dict[str, tuple[list[dict], np.ndarray]], k: int, n_jobs: int,
) -> list[dict]:
    """For each query corpus, the mean fraction of its points' k nearest
    *other* points (self excluded) that come from each corpus -- a
    query-corpus x neighbour-corpus composition table."""
    points, vectors, labels = _concat_pool(pool)
    nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=n_jobs).fit(vectors)
    _, neighbor_idx = nn.kneighbors(vectors)

    counts: dict[str, dict[str, int]] = {c: {c2: 0 for c2 in gac.EXPRESSION_CORPORA} for c in gac.EXPRESSION_CORPORA}
    totals: dict[str, int] = {c: 0 for c in gac.EXPRESSION_CORPORA}
    for i, row in enumerate(neighbor_idx):
        query_corpus = labels[i]
        neighbours = [j for j in row if j != i][:k]
        for j in neighbours:
            counts[query_corpus][labels[j]] += 1
        totals[query_corpus] += len(neighbours)

    rows = []
    for query_corpus in gac.EXPRESSION_CORPORA:
        for neighbor_corpus in gac.EXPRESSION_CORPORA:
            total = totals[query_corpus] or 1
            rows.append({
                "k": k,
                "query_corpus": query_corpus,
                "neighbor_corpus": neighbor_corpus,
                "fraction": counts[query_corpus][neighbor_corpus] / total,
            })
    return rows


def silhouette_for_pool(pool: dict[str, tuple[list[dict], np.ndarray]]) -> float:
    _points, vectors, labels = _concat_pool(pool)
    return float(silhouette_score(vectors, labels, metric="euclidean"))


def run_equal_n_bootstrap(shared_space: gac.SharedSpace, seed: int, reps: int, n_jobs: int) -> dict:
    knn_reps: dict[int, dict[tuple[str, str], list[float]]] = {
        k: {(a, b): [] for a in gac.EXPRESSION_CORPORA for b in gac.EXPRESSION_CORPORA} for k in K_VALUES
    }
    silhouette_reps: list[float] = []

    for rep in range(reps):
        draw = gac.corpus_vectors_and_points(shared_space, "equal_n_expression", seed=seed + rep)
        for k in K_VALUES:
            for row in knn_composition(draw, k, n_jobs):
                knn_reps[k][(row["query_corpus"], row["neighbor_corpus"])].append(row["fraction"])
        silhouette_reps.append(silhouette_for_pool(draw))

    knn_summary = []
    for k in K_VALUES:
        for query_corpus in gac.EXPRESSION_CORPORA:
            for neighbor_corpus in gac.EXPRESSION_CORPORA:
                summary = gac.bootstrap_summary(knn_reps[k][(query_corpus, neighbor_corpus)])
                knn_summary.append({"k": k, "query_corpus": query_corpus, "neighbor_corpus": neighbor_corpus, **summary})

    return {"knn_summary": knn_summary, "silhouette_summary": gac.bootstrap_summary(silhouette_reps)}


write_csv = gac.write_csv


def fit_and_save_umap(
    out_dir, population_name: str, points: list[dict], vectors: np.ndarray,
    n_neighbors: int, min_dist: float, seed: int,
) -> None:
    input_sha256 = gac.sha256_array(vectors)
    logger.info(
        "UMAP fit '%s' (n=%d, n_neighbors=%d, min_dist=%.2f)...",
        population_name, len(points), n_neighbors, min_dist,
    )
    coords = umap.UMAP(
        n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
        random_state=seed, metric="euclidean",
    ).fit_transform(vectors)

    stem = f"umap_{population_name}_n{n_neighbors}_d{min_dist}"
    coords_path = out_dir / f"{stem}.jsonl"
    with open(coords_path, "w", encoding="utf-8") as f:
        for p, coord in zip(points, coords):
            row = {
                "key": p["key"], "source_dataset": p["source_dataset"],
                "point_role": p.get("point_role"), "label": p.get("label"),
                "umap_2d": coord.tolist(),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    output_sha256 = gac.sha256_file(coords_path)

    manifest = {
        "population": population_name,
        "n_points": len(points),
        "input_sha256": input_sha256,
        "seed": seed,
        "umap_params": {"n_neighbors": n_neighbors, "min_dist": min_dist, "n_components": 2, "metric": "euclidean"},
        "umap_learn_version": umap.__version__,
        "output_coords_path": coords_path.name,
        "output_sha256": output_sha256,
    }
    (out_dir / f"{stem}.manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=gac.DEFAULT_SEED)
    parser.add_argument("--bootstrap-reps", type=int, default=gac.DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--point-sample-size", type=int, default=gac.DEFAULT_POINT_SAMPLE_SIZE)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--umap-neighbors", type=int, default=20)
    parser.add_argument("--umap-min-dist", type=float, default=0.2)
    parser.add_argument("--umap-param-grid", action="store_true",
                         help="Additionally fit (10,0.1), (20,0.2), (50,0.3) for each population.")
    args = parser.parse_args()

    logger.info("Loading %s ...", gac.EMBEDDING_SPACE_PATH)
    shared_space = gac.load_shared_space()
    logger.info("Loaded %d points, %d-d vectors", len(shared_space.points), shared_space.vectors.shape[1])

    run_id, run_dir = gac.get_or_create_run_dir(args.run_id)
    gac.init_run_manifest(run_dir, shared_space, defaults={
        "seed": args.seed, "bootstrap_reps": args.bootstrap_reps, "point_sample_size": args.point_sample_size,
    })
    gac.update_run_manifest(run_dir, MODULE_NAME, "running")
    out_dir = gac.module_run_dir(run_dir, MODULE_NAME)

    try:
        pools = {
            "full": gac.corpus_vectors_and_points(shared_space, "full"),
            "full_sampled_pointwise": full_sampled_pointwise_pool(shared_space, args.point_sample_size, args.seed),
            "reduced_literature": gac.corpus_vectors_and_points(shared_space, "reduced_literature"),
        }

        logger.info("k-NN composition (k=10,20) -- full / full_sampled_pointwise / reduced_literature...")
        knn_rows = []
        for mode, pool in pools.items():
            for k in K_VALUES:
                for row in knn_composition(pool, k, args.n_jobs):
                    knn_rows.append({"mode": mode, **row})
        write_csv(out_dir / "knn_composition_descriptive.csv", knn_rows)

        logger.info("Silhouette by source_dataset -- full_sampled_pointwise / reduced_literature...")
        silhouette_rows = [
            {"mode": mode, "silhouette_score": silhouette_for_pool(pools[mode])}
            for mode in ("full_sampled_pointwise", "reduced_literature")
        ]
        write_csv(out_dir / "silhouette_descriptive.csv", silhouette_rows)

        logger.info("equal_n_expression bootstrap (B=%d) -- PRIMARY controlled comparison...", args.bootstrap_reps)
        bootstrap = run_equal_n_bootstrap(shared_space, args.seed, args.bootstrap_reps, args.n_jobs)
        write_csv(out_dir / "knn_composition_equal_n_expression.csv", bootstrap["knn_summary"])
        (out_dir / "silhouette_equal_n_expression.json").write_text(
            json.dumps(bootstrap["silhouette_summary"], indent=2), encoding="utf-8",
        )

        logger.info("2-D UMAP fits (this module owns every UMAP computation in the toolkit)...")
        overview_points = shared_space.points
        overview_vectors = shared_space.vectors
        expr_sampled_points, expr_sampled_vectors, _ = _concat_pool(pools["full_sampled_pointwise"])
        equal_n_draw = gac.corpus_vectors_and_points(shared_space, "equal_n_expression", seed=args.seed)
        equal_n_points, equal_n_vectors, _ = _concat_pool(equal_n_draw)

        param_combinations = [(args.umap_neighbors, args.umap_min_dist)]
        if args.umap_param_grid:
            param_combinations = list(dict.fromkeys(param_combinations + UMAP_PARAM_GRID))

        # Optional 4th population: overview + the interview initial-exemplar
        # prototype layer (build_interview_prototype_layer.py), if it's been
        # built. Fit fresh rather than reusing "overview"'s coordinates --
        # several prototypes (e.g. "AI cult", "Illuminati") were dropped by
        # build_shared_space.py's pooling filter and have no position in the
        # ordinary pooled space at all, so there's nothing to look up there.
        prototypes_points_vectors = None
        if gac.INTERVIEW_PROTOTYPES_PATH.exists():
            proto_points, proto_vectors = [], []
            with open(gac.INTERVIEW_PROTOTYPES_PATH, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    proto_points.append({
                        "key": f"prototype:{item['document_id']}",
                        "source_dataset": item["source_dataset"],
                        "point_role": item["point_role"],
                        "label": item["source_expression_label"],
                    })
                    proto_vectors.append(item["shared_space_vector"])
            with_prototypes_points = overview_points + proto_points
            with_prototypes_vectors = np.concatenate([overview_vectors, np.array(proto_vectors, dtype=np.float64)], axis=0)
            prototypes_points_vectors = (with_prototypes_points, with_prototypes_vectors)
        else:
            logger.warning(
                "%s not found -- skipping the with_interview_prototypes UMAP population "
                "(run build_interview_prototype_layer.py first if you want it).",
                gac.INTERVIEW_PROTOTYPES_PATH,
            )

        for n_neighbors, min_dist in param_combinations:
            fit_and_save_umap(out_dir, "overview", overview_points, overview_vectors, n_neighbors, min_dist, args.seed)
            fit_and_save_umap(out_dir, "expression_sampled", expr_sampled_points, expr_sampled_vectors, n_neighbors, min_dist, args.seed)
            fit_and_save_umap(out_dir, "equal_n_diagnostic", equal_n_points, equal_n_vectors, n_neighbors, min_dist, args.seed)
            if prototypes_points_vectors is not None:
                fit_and_save_umap(out_dir, "with_interview_prototypes", *prototypes_points_vectors, n_neighbors, min_dist, args.seed)

        gac.write_module_config(
            out_dir,
            input_sha256=shared_space.input_sha256,
            k_values=K_VALUES,
            seed=args.seed,
            bootstrap_reps=args.bootstrap_reps,
            point_sample_size=args.point_sample_size,
            n_jobs=args.n_jobs,
            umap_param_combinations=param_combinations,
            metric="euclidean",
            git_commit=gac.git_commit_hash(),
        )
        gac.update_run_manifest(run_dir, MODULE_NAME, "completed", output_dir=out_dir)
    except Exception as e:
        gac.update_run_manifest(run_dir, MODULE_NAME, "failed", error=str(e))
        raise

    print(f"\nDone. Run {run_id} -> {out_dir}")


if __name__ == "__main__":
    main()
