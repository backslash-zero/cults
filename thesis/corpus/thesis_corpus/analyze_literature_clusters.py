"""HDBSCAN clustering of the literature corpus's own raw bge-m3
expressions, plus their relationship to the emergent-entities layer --
"how do these expressions group into concepts, what's each cluster's own
centroid, and does that concept have a named entity anywhere near it."
First run of the clustering step logged as "Planned next" in
Analysis/Results_Draft.md.

GOVERNING PRINCIPLE (same posture as analyze_entity_topology.py's, restated
here for the same reason): cluster membership and centrality are purely
geometric facts about this embedding space under this model. A "dense
cluster" of expressions is not itself evidence those expressions are
conceptually identical beyond "their embeddings landed near each other";
an HDBSCAN "noise" label does not mean an expression is unimportant, only
that it did not sit inside a density-defined cluster at the chosen
min_cluster_size.

RAW, NOT shared_space_v3 -- same reasoning as analyze_corpus_centroids.py
(reused here directly via load_raw_points/group_indices): raw bge-m3
vectors are unit-norm and directly cosine-comparable with zero
transformation.

UMAP PRE-REDUCTION BEFORE CLUSTERING, PLAIN RAW VECTORS FOR EVERYTHING ELSE.
Checked empirically, not assumed: HDBSCAN directly on the raw 1024-d
vectors is unusable here -- swept min_cluster_size in {5,8,10,15}, noise
ratio was 84.5-92.6% every time (density differences between "topic
clusters" flatten out at this dimensionality, a well-known HDBSCAN/
high-dimensionality interaction, the reason tools like BERTopic reduce
before clustering rather than after). A 12-d UMAP pre-reduction
(metric="cosine", min_dist=0.0 -- tight packing is what you want feeding
a density-based clusterer, unlike the visualization use of UMAP
elsewhere in this toolkit where min_dist=0.1 avoids over-compressing the
picture) brings noise down to ~40-46% at a comparable min_cluster_size,
confirmed by the same sweep.

WHY, mechanistically, not just empirically: HDBSCAN's whole method depends on
detecting DENSITY CONTRAST -- regions measurably denser than their
surroundings. In high-dimensional space, pairwise distances tend to
concentrate (the gap between "nearest-neighbour distance" and "typical
distance" shrinks as dimensionality grows) -- bge-m3 embeddings sit on a
real semantic manifold, not a random cloud, so this doesn't hit the
theoretical worst case, but 1024 raw dimensions still carry a lot of
variance that isn't semantically discriminative for clustering (sentence
length, phrasing style, tokenization quirks), and that noise dilutes the
topical signal enough to flatten local density differences. HDBSCAN's
core distance (distance to a point's k-th nearest neighbour) ends up
large and near-uniform across most points as a result, so almost no
candidate grouping is PERSISTENT (stable across a range of density
thresholds) enough to survive its cluster-extraction step -- confirmed
directly by the sweep making this WORSE, not better, as min_cluster_size
rose from 5 to 15 (84.5% -> 92.6% noise): almost nothing in the raw space
was dense enough to support even a modest persistent cluster. UMAP's
reduction explicitly optimizes to preserve LOCAL neighbourhood topology,
which restores exactly the density contrast HDBSCAN needs -- not a
workaround specific to this project, but the reason tools like BERTopic
always reduce before clustering. The semantic structure was there in the
raw vectors all along (post-reduction clusters came out very clean, e.g.
Heaven's Gate at 0.985 cosine similarity to its own cluster) -- it just
wasn't DETECTABLE by a density method operating directly in the noisy
ambient space.

So: `compute_clusters` (cluster ASSIGNMENT
only) runs on the UMAP-reduced coordinates; every centroid, distance, and
nearest-neighbour computation downstream of that still runs on the RAW
1024-d vectors -- UMAP here is a tool for finding which points belong
together, never the space anything is actually measured in (same
standing rule as the rest of this toolkit: pictures/reductions are a
reading aid, real distances come from the full space).

Three pieces, in the order Analysis/Results_Draft.md's "Planned next" lists
them:

  1. compute_clusters (imported from analyze_entity_topology.py, not
     reimplemented -- identical algorithm/rationale, different population).
  2. Binding expression per cluster: the `k` real literature expressions
     nearest EACH CLUSTER'S OWN centroid, restricted to that cluster's own
     members (not the whole corpus) -- reuses gac.nearest_points, no new
     dependency, no paraphrase risk (it's an actual sentence, not a
     generated summary).
  3. Entities vs. clusters, three angles per the plan:
       (a) cluster -> nearest entity (and its distance) -- candidate pool
           is `entities_literature` (entities actually mentioned BY the
           literature corpus, not the cross-corpus entities_all) --
           "which clusters have close entities vs. not."
       (b) entity -> nearest cluster -- the inverse assignment, treating
           cluster centroids as a small candidate pool.
       (c) gap detection: clusters whose nearest-entity distance is above
           a percentile threshold across all clusters, flagged as "a
           concept with no nearby named entity." The inverse framing of
           generate_voronoi_projections.py's existing entity-only gap
           grid, computed in real distance space, not a 2-D grid.

Usage (from thesis/corpus/):
    python -m thesis_corpus.analyze_literature_clusters
    python -m thesis_corpus.analyze_literature_clusters --min-cluster-size 20
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import umap

from thesis_corpus import analyze_corpus_centroids as acc
from thesis_corpus import analyze_entity_topology as aet
from thesis_corpus import build_shared_space as bss
from thesis_corpus import geometric_analysis_common as gac

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("thesis_corpus.analyze_literature_clusters")

DEFAULT_MIN_CLUSTER_SIZE = 20
BINDING_EXPRESSIONS_K = 8  # full k written to cluster_binding_expressions.csv; cluster_summary.csv keeps its own first 3
NEAREST_ENTITIES_K = 10
VIZ_SEED = 42
VIZ_MIN_DIST = 0.1  # visualization fit -- looser packing than the 0.0 used for the clustering-only reduction
GAP_PERCENTILE = 75  # clusters at/above this percentile of nearest-entity distance are flagged as gaps
UMAP_N_COMPONENTS = 12
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.0  # tight packing for a density-based clusterer, not a visualization
UMAP_SEED = 42


def reduce_for_clustering(vectors: np.ndarray, n_components: int) -> np.ndarray:
    """UMAP reduction used ONLY to make HDBSCAN's density estimate
    meaningful (see module docstring for the empirical noise-ratio
    numbers this is based on) -- the returned coordinates are never used
    for any centroid/distance/nearest-neighbour computation, only as
    compute_clusters' input for assigning labels."""
    reducer = umap.UMAP(n_components=n_components, n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST,
                         metric="cosine", random_state=UMAP_SEED)
    return reducer.fit_transform(vectors)


def cluster_binding_expressions(points: list[dict], vectors: np.ndarray, labels: np.ndarray, k: int) -> dict[int, list[dict]]:
    """For each cluster (excluding noise), the k real expressions nearest
    THAT CLUSTER'S OWN centroid, candidates restricted to that cluster's
    own members -- this is what "binding expression" means here: an
    actual member sentence that best represents its own cluster, not a
    generated label and not borrowed from a different cluster."""
    result: dict[int, list[dict]] = {}
    for cluster_id in sorted(set(labels) - {-1}):
        member_idxs = [i for i, label in enumerate(labels) if label == cluster_id]
        member_points = [points[i] for i in member_idxs]
        member_vectors = vectors[member_idxs]
        centroid = member_vectors.mean(axis=0)
        result[cluster_id] = gac.nearest_points(centroid, member_points, member_vectors, k=min(k, len(member_idxs)))
    return result


def cluster_nearest_entities(vectors: np.ndarray, labels: np.ndarray,
                              entity_points: list[dict], entity_vectors: np.ndarray, k: int) -> dict[int, list[dict]]:
    """For each cluster's own centroid, the k nearest points in the
    entity candidate pool -- "which named entities sit closest to this
    concept," independent of whether any entity is actually a cluster
    member (entities and expressions are different point_roles, never
    mixed in one cluster)."""
    result: dict[int, list[dict]] = {}
    for cluster_id in sorted(set(labels) - {-1}):
        member_idxs = [i for i, label in enumerate(labels) if label == cluster_id]
        centroid = vectors[member_idxs].mean(axis=0)
        result[cluster_id] = gac.nearest_points(centroid, entity_points, entity_vectors, k=k)
    return result


def entity_nearest_cluster(vectors: np.ndarray, labels: np.ndarray,
                            entity_points: list[dict], entity_vectors: np.ndarray) -> list[dict]:
    """Inverse of cluster_nearest_entities: for each entity, which single
    cluster centroid is nearest (rank-1 only -- this is an assignment,
    not a ranked list). Cluster centroids are packaged as a small
    synthetic candidate pool (`key`=str(cluster_id)) so the same
    gac.nearest_points machinery applies unchanged."""
    cluster_ids = sorted(set(labels) - {-1})
    cluster_centroid_points = [{"key": str(cid), "label": f"cluster {cid}"} for cid in cluster_ids]
    cluster_centroid_vectors = np.array([vectors[[i for i, l in enumerate(labels) if l == cid]].mean(axis=0) for cid in cluster_ids])

    rows = []
    for i, entity_point in enumerate(entity_points):
        nearest = gac.nearest_points(entity_vectors[i], cluster_centroid_points, cluster_centroid_vectors, k=1)[0]
        rows.append({
            "entity_key": entity_point["key"], "entity_label": entity_point["label"],
            "nearest_cluster_id": nearest["key"], "euclidean_distance": nearest["euclidean_distance"],
            "cosine_similarity": nearest["cosine_similarity"],
        })
    return rows


def plot_cluster_overview(out_path: Path, lit_vectors: np.ndarray, labels: np.ndarray,
                           gap_cluster_ids: set[int]) -> Path:
    """One 2-D map of the whole literature corpus: every point coloured
    noise/ordinary-cluster/gap-cluster, every real cluster's own centroid
    marked. A FRESH 2-D UMAP fit (metric="cosine", min_dist=0.1 -- the
    looser, visualization-appropriate packing used elsewhere in this
    toolkit, NOT the 0.0 the 12-d clustering-only reduction uses) --
    entirely separate from the 12-d reduction compute_clusters ran on, and
    never used for anything but this picture: cluster membership and every
    distance in cluster_summary.csv/etc. come from the real 1024-d space,
    unaffected by whatever this plot's own 2-D fit happens to look like."""
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=VIZ_MIN_DIST, metric="cosine", random_state=VIZ_SEED)
    coords_2d = reducer.fit_transform(lit_vectors)

    cluster_ids = sorted(set(labels) - {-1})
    noise_mask = labels == -1
    gap_mask = np.array([lbl in gap_cluster_ids for lbl in labels])
    ordinary_mask = (~noise_mask) & (~gap_mask)

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.scatter(coords_2d[noise_mask, 0], coords_2d[noise_mask, 1], s=4, c="#e0e0e0", linewidths=0, zorder=1,
               label=f"noise ({int(noise_mask.sum())})")
    ax.scatter(coords_2d[ordinary_mask, 0], coords_2d[ordinary_mask, 1], s=6, c="#0072B2", linewidths=0, zorder=2,
               label=f"ordinary cluster ({int(ordinary_mask.sum())})")
    ax.scatter(coords_2d[gap_mask, 0], coords_2d[gap_mask, 1], s=6, c="#D55E00", linewidths=0, zorder=2,
               label=f"gap cluster ({int(gap_mask.sum())})")

    for cid in cluster_ids:
        member_2d = coords_2d[labels == cid]
        centroid_2d = member_2d.mean(axis=0)
        is_gap = cid in gap_cluster_ids
        ax.scatter(*centroid_2d, s=90, c="#D55E00" if is_gap else "#023858", marker="*",
                   edgecolors="black", linewidths=0.5, zorder=3)
        ax.annotate(str(cid), centroid_2d, fontsize=6, color="black", xytext=(3, 3), textcoords="offset points")

    ax.set_title("Literature clusters overview (59 clusters, UMAP-2D for display only,\n"
                  "membership + all distances computed in raw 1024-D)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(loc="best", fontsize=8, markerscale=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lit-mivi-run-tag", default="20260910")
    parser.add_argument("--interviews-archive-dir", type=Path,
                         default=bss.PROCESSED_DIR / "interviews_full" / "interviews" / "run_20260912")
    parser.add_argument("--min-cluster-size", type=int, default=DEFAULT_MIN_CLUSTER_SIZE)
    parser.add_argument("--binding-expressions-k", type=int, default=BINDING_EXPRESSIONS_K)
    parser.add_argument("--nearest-entities-k", type=int, default=NEAREST_ENTITIES_K)
    parser.add_argument("--gap-percentile", type=float, default=GAP_PERCENTILE)
    parser.add_argument("--umap-n-components", type=int, default=UMAP_N_COMPONENTS,
                         help="Dimensionality HDBSCAN actually clusters on (via UMAP pre-reduction) -- "
                              "0 disables pre-reduction and clusters on raw 1024-d vectors directly "
                              "(confirmed empirically to leave 84-93%% of literature as noise; kept as an "
                              "option for comparison, not the default).")
    parser.add_argument("--out-dir", type=Path, default=bss.PROCESSED_DIR / "analysis_raw" / "literature_clusters")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw_points = acc.load_raw_points(args.lit_mivi_run_tag, args.interviews_archive_dir)
    vectors = np.array([p["vector"] for p in raw_points], dtype=np.float64)
    points = [{k: v for k, v in p.items() if k != "vector"} for p in raw_points]
    logger.info("Loaded %d raw points", len(points))

    lit_idx = acc.group_indices(points, "literature")
    lit_points = [points[i] for i in lit_idx]
    lit_vectors = vectors[lit_idx]
    logger.info("literature: n=%d", len(lit_points))

    if args.umap_n_components > 0:
        logger.info("Reducing to %d-d (UMAP, cosine) for cluster ASSIGNMENT only -- "
                     "all subsequent distances/centroids stay in raw 1024-d.", args.umap_n_components)
        clustering_vectors = reduce_for_clustering(lit_vectors, args.umap_n_components)
    else:
        clustering_vectors = lit_vectors
    labels = aet.compute_clusters(clustering_vectors, args.min_cluster_size)
    cluster_ids = sorted(set(labels) - {-1})
    n_noise = int((labels == -1).sum())
    logger.info("HDBSCAN (min_cluster_size=%d): %d clusters, %d/%d expressions are noise (%.1f%%)",
                args.min_cluster_size, len(cluster_ids), n_noise, len(lit_points), 100 * n_noise / len(lit_points))
    sizes = [int((labels == cid).sum()) for cid in cluster_ids]
    if sizes:
        logger.info("cluster sizes: min=%d median=%d max=%d", min(sizes), int(np.median(sizes)), max(sizes))

    grand_stats = gac.centroid_and_dispersion_for_indices(lit_vectors, list(range(len(lit_points))))
    grand_centroid = grand_stats["centroid"]

    entity_idx = acc.group_indices(points, "entities_literature")
    entity_points = [points[i] for i in entity_idx]
    entity_vectors = vectors[entity_idx]
    logger.info("entities_literature: n=%d (candidate pool for cluster->entity)", len(entity_points))

    binding = cluster_binding_expressions(lit_points, lit_vectors, labels, args.binding_expressions_k)
    nearest_entities = cluster_nearest_entities(lit_vectors, labels, entity_points, entity_vectors, args.nearest_entities_k)

    # --- cluster_summary.csv ---
    summary_rows = []
    for cid in cluster_ids:
        member_idxs = [i for i, l in enumerate(labels) if l == cid]
        stats = gac.centroid_and_dispersion_for_indices(lit_vectors, member_idxs)
        centrality = float(gac.euclidean_distances(grand_centroid, stats["centroid"][np.newaxis, :])[0])
        nearest_entity = nearest_entities[cid][0]
        summary_rows.append({
            "cluster_id": cid,
            "size": stats["n"],
            "internal_dispersion_mean": round(stats["dispersion"], 4),
            "distance_to_grand_centroid": round(centrality, 4),
            "binding_expression_1": binding[cid][0]["label"] if binding[cid] else "",
            "binding_expression_2": binding[cid][1]["label"] if len(binding[cid]) > 1 else "",
            "binding_expression_3": binding[cid][2]["label"] if len(binding[cid]) > 2 else "",
            "nearest_entity": nearest_entity["label"],
            "nearest_entity_euclidean_distance": round(nearest_entity["euclidean_distance"], 4),
            "nearest_entity_cosine_similarity": round(nearest_entity["cosine_similarity"], 4),
        })

    gap_threshold = float(np.percentile([r["nearest_entity_euclidean_distance"] for r in summary_rows], args.gap_percentile))
    for r in summary_rows:
        r["entity_gap"] = r["nearest_entity_euclidean_distance"] >= gap_threshold
    n_gaps = sum(r["entity_gap"] for r in summary_rows)
    logger.info("Entity-gap threshold (p%.0f of nearest-entity distance): %.4f -- %d/%d clusters flagged as gaps",
                args.gap_percentile, gap_threshold, n_gaps, len(summary_rows))

    summary_rows.sort(key=lambda r: -r["size"])
    summary_path = args.out_dir / "cluster_summary.csv"
    gac.write_csv(summary_path, summary_rows)

    # --- cluster_nearest_entities.csv (full k, not just rank-1) ---
    entity_rows = []
    for cid in cluster_ids:
        for n in nearest_entities[cid]:
            entity_rows.append({"cluster_id": cid, **n})
    entities_path = args.out_dir / "cluster_nearest_entities.csv"
    gac.write_csv(entities_path, entity_rows)

    # --- cluster_binding_expressions.csv (full k, not just the 3 in cluster_summary.csv) ---
    binding_rows = []
    for cid in cluster_ids:
        for n in binding[cid]:
            binding_rows.append({"cluster_id": cid, **n})
    binding_path = args.out_dir / "cluster_binding_expressions.csv"
    gac.write_csv(binding_path, binding_rows)

    # --- entity_nearest_cluster.csv ---
    entity_assignment_rows = entity_nearest_cluster(lit_vectors, labels, entity_points, entity_vectors)
    assignment_path = args.out_dir / "entity_nearest_cluster.csv"
    gac.write_csv(assignment_path, entity_assignment_rows)

    # --- overview plot ---
    gap_cluster_ids = {r["cluster_id"] for r in summary_rows if r["entity_gap"]}
    plot_path = args.out_dir / "cluster_overview.png"
    plot_cluster_overview(plot_path, lit_vectors, labels, gap_cluster_ids)

    print(f"Done. {len(cluster_ids)} clusters, {n_gaps} flagged as entity gaps (p{args.gap_percentile:.0f} threshold).")
    print(f"{summary_path}\n{binding_path}\n{entities_path}\n{assignment_path}\n{plot_path}")


if __name__ == "__main__":
    main()
