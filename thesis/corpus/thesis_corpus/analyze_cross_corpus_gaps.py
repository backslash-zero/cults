"""Heavier, geometric-proximity counterpart to
analyze_entity_topology.py's lightweight cluster_corpus_coverage_rows:
instead of checking whether the SAME normalized entity is mentioned by
two corpora, this clusters each corpus's own entities INDEPENDENTLY,
then asks how far each corpus-A cluster sits from the NEAREST entity
corpus B mentions at all -- catching coverage that the lightweight check
cannot, e.g. literature discussing "Peoples Temple" and an interview
discussing "Jonestown": different normalized strings, so the lightweight
exact-entity check sees no overlap, but this proximity check can still
find them geometrically close.

Directly operationalizes the researcher's own framing: "take the
literature space -- which of its regions are not occupied by the
interviews, and the reverse" -- for all three ordered corpus pairs
(6 directional comparisons total: literature->{miviludes,interviews},
miviludes->{literature,interviews}, interviews->{literature,miviludes}).

GOVERNING PRINCIPLE (same posture as analyze_entity_topology.py's):
cluster membership and geometric proximity are purely geometric facts
about this embedding space under this model. This module deliberately
reports RAW distances only -- it does not invent a "this counts as a
real gap" distance threshold; that judgement is left to the researcher,
read alongside this space's other established distance scales (e.g.
typicality's own distance ranges) rather than decided silently here.

Usage (from thesis/corpus/):
    python -m thesis_corpus.analyze_cross_corpus_gaps --shared-space-dir processed/shared_space_v2
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from thesis_corpus import geometric_analysis_common as gac
from thesis_corpus.analyze_entity_topology import cluster_summary_rows, compute_clusters, grand_centroid

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("thesis_corpus.analyze_cross_corpus_gaps")

MODULE_NAME = "analyze_cross_corpus_gaps"
DEFAULT_MIN_CLUSTER_SIZE = 3
DEFAULT_NEAREST_K = 3

GOVERNING_PRINCIPLE = (
    "Cluster membership and geometric proximity are purely geometric facts about this "
    "embedding space under this model. This module deliberately reports RAW distances "
    "only -- it does not invent a \"this counts as a real gap\" distance threshold; that "
    "judgement is left to the researcher, read alongside this space's other established "
    "distance scales rather than decided silently here."
)


# ---------------------------------------------------------------------------
# Pure computation -- no file I/O, unit-testable.
# ---------------------------------------------------------------------------

def corpus_restricted_entities(
    points: list[dict], vectors: np.ndarray, corpus: str,
) -> tuple[list[dict], np.ndarray]:
    """Entities with a nonzero mention_distribution[corpus] -- i.e. that
    corpus's own "occupied" entities, the population its independent
    clustering is fit on."""
    idxs = [i for i, p in enumerate(points) if (p.get("mention_distribution") or {}).get(corpus, 0) > 0]
    return [points[i] for i in idxs], vectors[idxs]


def cluster_proximity_rows(
    corpus_a: str, corpus_a_points: list[dict], corpus_a_vectors: np.ndarray, corpus_a_labels: np.ndarray,
    corpus_b: str, corpus_b_points: list[dict], corpus_b_vectors: np.ndarray,
    k: int,
) -> list[dict]:
    """For each corpus_a cluster (its own member centroid), the k nearest
    entities corpus_b mentions at all -- regardless of whether any of
    them share an exact normalized string with a corpus_a cluster member.
    One row per (cluster, rank)."""
    rows = []
    for cluster_id in sorted(set(corpus_a_labels) - {-1}):
        member_idxs = [i for i, label in enumerate(corpus_a_labels) if label == cluster_id]
        cluster_centroid = corpus_a_vectors[member_idxs].mean(axis=0)
        member_labels = sorted(corpus_a_points[i]["label"] for i in member_idxs)
        nearest = gac.nearest_points(cluster_centroid, corpus_b_points, corpus_b_vectors, k)
        for entry in nearest:
            rows.append({
                "corpus_a": corpus_a, "corpus_a_cluster_id": int(cluster_id),
                "corpus_a_cluster_size": len(member_idxs),
                "corpus_a_cluster_members": "; ".join(member_labels),
                "corpus_b": corpus_b, "rank": entry["rank"],
                "nearest_corpus_b_entity": entry["label"],
                "distance": entry["euclidean_distance"],
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
    parser.add_argument("--nearest-k", type=int, default=DEFAULT_NEAREST_K)
    args = parser.parse_args()

    embedding_space_path, interview_prototypes_path, analysis_root = gac.resolve_space_paths(args.shared_space_dir)
    logger.info("Loading %s ...", embedding_space_path)
    shared_space = gac.load_shared_space(embedding_space_path)

    run_id, run_dir = gac.get_or_create_run_dir(args.run_id, analysis_root)
    gac.init_run_manifest(run_dir, shared_space, defaults={"min_cluster_size": args.min_cluster_size, "nearest_k": args.nearest_k})
    gac.update_run_manifest(run_dir, MODULE_NAME, "running")
    out_dir = gac.module_run_dir(run_dir, MODULE_NAME)

    try:
        entity_idxs = gac.source_dataset_indices(shared_space.points, "emergent_entities")
        entity_points = [shared_space.points[i] for i in entity_idxs]
        entity_vectors = shared_space.vectors[entity_idxs]

        per_source = gac.per_source_centroids_and_dispersion(shared_space)
        centroid = grand_centroid(per_source)

        per_corpus: dict[str, tuple[list[dict], np.ndarray, np.ndarray]] = {}
        all_summary_rows = []
        for corpus in gac.EXPRESSION_CORPORA:
            corpus_points, corpus_vectors = corpus_restricted_entities(entity_points, entity_vectors, corpus)
            logger.info("%s: %d entities mentioned at all; clustering (min_cluster_size=%d)...",
                        corpus, len(corpus_points), args.min_cluster_size)
            try:
                labels = compute_clusters(corpus_vectors, args.min_cluster_size)
            except ValueError as e:
                logger.warning("%s: skipping (%s) -- too few entities to cluster at this min_cluster_size.", corpus, e)
                continue
            n_clusters = len(set(labels) - {-1})
            n_noise = int((labels == -1).sum())
            logger.info("%s: %d clusters, %d/%d noise.", corpus, n_clusters, n_noise, len(corpus_points))
            per_corpus[corpus] = (corpus_points, corpus_vectors, labels)

            summary = cluster_summary_rows(corpus_points, corpus_vectors, labels, centroid)
            for row in summary:
                row["corpus"] = corpus
            all_summary_rows.extend(summary)

        gac.write_csv(out_dir / "per_corpus_cluster_summary.csv", all_summary_rows)

        proximity_rows = []
        for corpus_a in per_corpus:
            a_points, a_vectors, a_labels = per_corpus[corpus_a]
            for corpus_b in per_corpus:
                if corpus_a == corpus_b:
                    continue
                b_points, b_vectors, _b_labels = per_corpus[corpus_b]
                logger.info("%s clusters -> nearest %s entities...", corpus_a, corpus_b)
                proximity_rows.extend(
                    cluster_proximity_rows(corpus_a, a_points, a_vectors, a_labels, corpus_b, b_points, b_vectors, args.nearest_k)
                )
        gac.write_csv(out_dir / "cross_corpus_cluster_proximity.csv", proximity_rows)

        (out_dir / "GOVERNING_PRINCIPLE.txt").write_text(GOVERNING_PRINCIPLE + "\n", encoding="utf-8")

        gac.write_module_config(
            out_dir,
            input_sha256=shared_space.input_sha256,
            min_cluster_size=args.min_cluster_size,
            nearest_k=args.nearest_k,
            corpora_clustered=list(per_corpus.keys()),
            n_clusters_per_corpus={c: len(set(l) - {-1}) for c, (_p, _v, l) in per_corpus.items()},
            governing_principle=GOVERNING_PRINCIPLE,
            git_commit=gac.git_commit_hash(),
        )
        gac.update_run_manifest(run_dir, MODULE_NAME, "completed", output_dir=out_dir)
    except Exception as e:
        gac.update_run_manifest(run_dir, MODULE_NAME, "failed", error=str(e))
        raise

    print(f"\nDone. Run {run_id} -> {out_dir}")


if __name__ == "__main__":
    main()
