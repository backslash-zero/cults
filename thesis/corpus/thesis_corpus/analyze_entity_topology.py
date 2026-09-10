"""Density, clustering, and centrality of the emergent-entities layer, in
the REAL (393-D, or whatever the current shared space's fitted
dimensionality is) space -- never a 2-D projection (this toolkit's
standing rule: pictures are a reading aid, real distances come from the
full space; see generate_voronoi_projections.py for the 2-D gap/Voronoi
work this module's output feeds).

Three questions, all about the 753 (or however many survive the current
run's entity-quality thresholds -- see build_shared_space.py's
MANUALLY_EXCLUDED_POOLED_KEYS/looks_like_named_entity/
load_cited_author_surnames) emergent-entity points specifically:

  1. Do they form dense clusters, or sit scattered? sklearn.cluster.HDBSCAN
     -- density-based, so it finds clusters of varying density without a
     pre-specified count, and explicitly leaves sparse points as noise
     (label -1) rather than forcing every entity into some cluster.
  2. How locally dense is each individual entity's own neighbourhood?
     Distance to its k-th nearest neighbour among ALL entity points
     (smaller = denser).
  3. How central or peripheral is each entity/cluster? Euclidean distance
     from the entity (or a cluster's own member-centroid) to the
     combined-expression grand centroid (equal-corpus-weighted, same
     "centre of gravity" framing analyze_typicality.py already uses for
     expressions -- applied here to entities for methodological
     consistency, not a new ad hoc metric).

GOVERNING PRINCIPLE (same posture as analyze_typicality.py's, stated here
for the same reason): cluster membership, local density, and centrality
are purely geometric facts about this embedding space under this model.
A "dense cluster" of entities is not itself evidence those entities are
conceptually similar in any deeper sense than "their extracted-context
embeddings landed near each other"; an HDBSCAN "noise" label does not mean
an entity is unimportant, only that it did not sit inside a
density-defined cluster at the chosen min_cluster_size.

Usage (from thesis/corpus/):
    python -m thesis_corpus.analyze_entity_topology --shared-space-dir processed/shared_space_v2
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.neighbors import NearestNeighbors

from thesis_corpus import geometric_analysis_common as gac

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("thesis_corpus.analyze_entity_topology")

MODULE_NAME = "analyze_entity_topology"
DEFAULT_MIN_CLUSTER_SIZE = 3
DEFAULT_LOCAL_DENSITY_K = 5
NEAREST_K_PER_CLUSTER = 5
TOP_N_CLUSTERS_FOR_RETRIEVAL = 10  # largest BY SIZE, not centrality -- a separate ordering, see main()

GOVERNING_PRINCIPLE = (
    "Cluster membership, local density, and centrality are purely geometric facts "
    "about this embedding space under this model. A \"dense cluster\" of entities is "
    "not itself evidence those entities are conceptually similar in any deeper sense "
    "than \"their extracted-context embeddings landed near each other\"; an HDBSCAN "
    "\"noise\" label does not mean an entity is unimportant, only that it did not sit "
    "inside a density-defined cluster at the chosen min_cluster_size."
)


# ---------------------------------------------------------------------------
# Pure computation -- no file I/O, unit-testable against small synthetic vectors.
# ---------------------------------------------------------------------------

def compute_clusters(vectors: np.ndarray, min_cluster_size: int) -> np.ndarray:
    """HDBSCAN labels over `vectors` -- -1 is noise (not "no cluster
    exists," but "this point did not sit inside a density-defined cluster
    at this min_cluster_size"). Deterministic (HDBSCAN has no random_state
    dependency the way UMAP/k-means do)."""
    if len(vectors) < min_cluster_size:
        raise ValueError(f"compute_clusters: {len(vectors)} points < min_cluster_size={min_cluster_size}")
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean", copy=True)
    return clusterer.fit_predict(vectors)


def local_density_scores(vectors: np.ndarray, k: int) -> np.ndarray:
    """Distance from each point to its own k-th nearest OTHER point (self
    excluded) -- smaller = denser local neighbourhood. Same
    NearestNeighbors machinery analyze_cluster_structure.py's
    knn_composition already uses for k-NN, applied here to raw distance
    rather than source-composition."""
    n = len(vectors)
    if n < k + 1:
        raise ValueError(f"local_density_scores: {n} points < k+1={k + 1} (need at least k other points)")
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(vectors)
    distances, _ = nn.kneighbors(vectors)
    return distances[:, -1]  # column 0 is self (distance 0); last column is the k-th OTHER neighbour


def grand_centroid(per_source: dict[str, dict]) -> np.ndarray:
    """Equal-corpus-weighted grand centroid across the 3 expression
    corpora, computed from an already-loaded per_source_centroids_and_dispersion
    result (in-memory) rather than re-reading embedding_space.jsonl from
    disk the way balanced_analysis.weighted_centroid does -- same
    definition, cheaper when the shared space is already loaded."""
    return np.mean([per_source[c]["centroid"] for c in gac.EXPRESSION_CORPORA], axis=0)


def entity_rows(
    points: list[dict], vectors: np.ndarray, labels: np.ndarray, density: np.ndarray, centroid: np.ndarray,
) -> list[dict]:
    """One row per entity: cluster membership, local density, and
    centrality (distance to the grand centroid) -- everything needed to
    reproduce or spot-check the cluster/gap figures downstream without
    recomputing from vectors."""
    distances_to_centroid = gac.euclidean_distances(centroid, vectors)
    rows = []
    for i, point in enumerate(points):
        rows.append({
            "key": point["key"],
            "entity": point["label"],
            "cluster_id": int(labels[i]),
            "is_noise": bool(labels[i] == -1),
            "local_density_distance": float(density[i]),
            "distance_to_grand_centroid": float(distances_to_centroid[i]),
            "mention_distribution": point.get("mention_distribution"),
        })
    return rows


def cluster_summary_rows(points: list[dict], vectors: np.ndarray, labels: np.ndarray, centroid: np.ndarray) -> list[dict]:
    """One row per cluster (excluding noise, which is summarized
    separately by the caller): size, member centroid, distance of that
    member centroid to the grand centroid (centrality), and a capped,
    readable list of member entity labels."""
    rows = []
    for cluster_id in sorted(set(labels) - {-1}):
        member_idxs = [i for i, label in enumerate(labels) if label == cluster_id]
        stats = gac.centroid_and_dispersion_for_indices(vectors, member_idxs)
        centrality = float(gac.euclidean_distances(centroid, stats["centroid"][np.newaxis, :])[0])
        member_labels = [points[i]["label"] for i in member_idxs]
        rows.append({
            "cluster_id": int(cluster_id),
            "size": stats["n"],
            "internal_dispersion_mean": stats["dispersion"],
            "distance_to_grand_centroid": centrality,
            "member_entities": "; ".join(sorted(member_labels)),
        })
    rows.sort(key=lambda r: r["distance_to_grand_centroid"])
    return rows


def cluster_centroid_nearest_rows(
    points: list[dict], vectors: np.ndarray, labels: np.ndarray,
    candidate_points: list[dict], candidate_vectors: np.ndarray,
    candidate_kind: str, top_n_clusters: int, k: int,
) -> list[dict]:
    """For the `top_n_clusters` LARGEST clusters (by member count -- a
    different ordering from cluster_summary_rows' centrality sort, kept
    deliberately separate since "biggest" and "most central" answer
    different questions and needn't agree), the k nearest points in
    `candidate_points`/`candidate_vectors` to that cluster's own member
    centroid. `candidate_kind` is a free-text label ("expression" or
    "entity") carried onto every row so the two calls' output (over
    different candidate pools) can be told apart after gac.write_csv
    concatenates them, or kept in separate files by the caller."""
    cluster_sizes: dict[int, int] = {}
    for label in labels:
        if label != -1:
            cluster_sizes[int(label)] = cluster_sizes.get(int(label), 0) + 1
    top_clusters = sorted(cluster_sizes, key=lambda c: -cluster_sizes[c])[:top_n_clusters]

    rows = []
    for cluster_id in top_clusters:
        member_idxs = [i for i, label in enumerate(labels) if label == cluster_id]
        cluster_centroid = vectors[member_idxs].mean(axis=0)
        nearest = gac.nearest_points(cluster_centroid, candidate_points, candidate_vectors, k)
        for entry in nearest:
            rows.append({
                "cluster_id": cluster_id, "cluster_size": cluster_sizes[cluster_id],
                "candidate_kind": candidate_kind, "rank": entry["rank"],
                "neighbor_key": entry["key"], "neighbor_label": entry["label"],
                "euclidean_distance": entry["euclidean_distance"], "cosine_similarity": entry["cosine_similarity"],
            })
    return rows


# ---------------------------------------------------------------------------
# Orchestration -- main() is the only part that touches disk.
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--shared-space-dir", type=Path, default=None,
                         help="Use a different pooled space (e.g. processed/shared_space_v2/) instead of v1's "
                              "processed/shared_space/; also switches the run-output root to a sibling "
                              "processed/analysis_v2/ directory so v1 and v2 runs are never mixed.")
    parser.add_argument("--min-cluster-size", type=int, default=DEFAULT_MIN_CLUSTER_SIZE)
    parser.add_argument("--local-density-k", type=int, default=DEFAULT_LOCAL_DENSITY_K)
    args = parser.parse_args()

    embedding_space_path, interview_prototypes_path, analysis_root = gac.resolve_space_paths(args.shared_space_dir)
    logger.info("Loading %s ...", embedding_space_path)
    shared_space = gac.load_shared_space(embedding_space_path)

    run_id, run_dir = gac.get_or_create_run_dir(args.run_id, analysis_root)
    gac.init_run_manifest(run_dir, shared_space, defaults={
        "min_cluster_size": args.min_cluster_size, "local_density_k": args.local_density_k,
    })
    gac.update_run_manifest(run_dir, MODULE_NAME, "running")
    out_dir = gac.module_run_dir(run_dir, MODULE_NAME)

    try:
        entity_idxs = gac.source_dataset_indices(shared_space.points, "emergent_entities")
        entity_points = [shared_space.points[i] for i in entity_idxs]
        entity_vectors = shared_space.vectors[entity_idxs]
        logger.info("%d emergent entities loaded.", len(entity_points))

        per_source = gac.per_source_centroids_and_dispersion(shared_space)
        centroid = grand_centroid(per_source)

        logger.info("HDBSCAN clustering (min_cluster_size=%d)...", args.min_cluster_size)
        labels = compute_clusters(entity_vectors, args.min_cluster_size)
        n_clusters = len(set(labels) - {-1})
        n_noise = int((labels == -1).sum())
        logger.info("%d clusters found; %d/%d entities are noise (min_cluster_size=%d).",
                    n_clusters, n_noise, len(entity_points), args.min_cluster_size)

        logger.info("Local density (k=%d)...", args.local_density_k)
        density = local_density_scores(entity_vectors, args.local_density_k)

        rows = entity_rows(entity_points, entity_vectors, labels, density, centroid)
        gac.write_csv(out_dir / "entity_clusters.csv", rows)

        summary_rows = cluster_summary_rows(entity_points, entity_vectors, labels, centroid)
        gac.write_csv(out_dir / "cluster_summary.csv", summary_rows)

        logger.info("Nearest expressions/entities per cluster centroid (top-%d largest clusters)...",
                    TOP_N_CLUSTERS_FOR_RETRIEVAL)
        expression_pool = gac.corpus_vectors_and_points(shared_space, "full")
        all_expression_points: list[dict] = []
        all_expression_vectors_list: list[np.ndarray] = []
        for _corpus, (pts, vecs) in expression_pool.items():
            all_expression_points.extend(pts)
            all_expression_vectors_list.append(vecs)
        all_expression_vectors = np.concatenate(all_expression_vectors_list, axis=0)

        cluster_nearest_expressions = cluster_centroid_nearest_rows(
            entity_points, entity_vectors, labels, all_expression_points, all_expression_vectors,
            "expression", TOP_N_CLUSTERS_FOR_RETRIEVAL, NEAREST_K_PER_CLUSTER,
        )
        gac.write_csv(out_dir / "cluster_centroid_nearest_expressions.csv", cluster_nearest_expressions)

        cluster_nearest_entities = cluster_centroid_nearest_rows(
            entity_points, entity_vectors, labels, entity_points, entity_vectors,
            "entity", TOP_N_CLUSTERS_FOR_RETRIEVAL, NEAREST_K_PER_CLUSTER + 1,  # +1 since a cluster's own centroid-nearest entity is often a member of itself
        )
        gac.write_csv(out_dir / "cluster_centroid_nearest_entities.csv", cluster_nearest_entities)

        (out_dir / "GOVERNING_PRINCIPLE.txt").write_text(GOVERNING_PRINCIPLE + "\n", encoding="utf-8")

        gac.write_module_config(
            out_dir,
            input_sha256=shared_space.input_sha256,
            n_entities=len(entity_points),
            min_cluster_size=args.min_cluster_size,
            local_density_k=args.local_density_k,
            n_clusters=n_clusters,
            n_noise=n_noise,
            governing_principle=GOVERNING_PRINCIPLE,
            git_commit=gac.git_commit_hash(),
        )
        gac.update_run_manifest(run_dir, MODULE_NAME, "completed", output_dir=out_dir)
    except Exception as e:
        gac.update_run_manifest(run_dir, MODULE_NAME, "failed", error=str(e))
        raise

    print(f"\nDone. Run {run_id} -> {out_dir} ({n_clusters} clusters, {n_noise} noise entities)")


if __name__ == "__main__":
    main()
