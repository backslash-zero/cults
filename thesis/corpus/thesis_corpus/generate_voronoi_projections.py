"""Three Voronoi tessellations over ONE shared 2-D UMAP basis of the
emergent-entities layer, plus a density/gap grid on that same basis --
"try a few different seed strategies and see what happens," per the
researcher's own framing, rather than picking one in advance.

All three tessellations fit UMAP on the SAME 753 entity vectors, with the
SAME n_neighbors/min_dist/seed, so the entity layout is expected to be
numerically identical across all three (umap-learn is deterministic given
a fixed random_state) -- only the OVERLAY (the seed points a Voronoi
diagram is built from) differs per strategy, via
geometric_analysis_common.fit_and_save_umap's existing member/overlay
split (seeds are transformed into the already-fitted manifold via
reducer.transform(), never part of the fit itself, same mechanism every
other focused-projection population already uses for centroid overlays).

  - `voronoi_criteria_seeded`: the 17 MIVILUDES criteria as seeds -- which
    region of entity-space each official criterion "owns."
  - `voronoi_corpus_centroid_seeded`: the 3 corpus centroids as seeds --
    coarse partition by source.
  - `voronoi_cluster_centroid_seeded`: analyze_entity_topology.py's own
    HDBSCAN cluster centroids as seeds -- a data-driven partition
    reflecting the entities' own density structure, rather than an
    externally-imposed one. Requires analyze_entity_topology.py to have
    already run for this run-id (reads its cluster_summary.csv /
    entity_clusters.csv -- fails loudly if missing, same "read a sibling
    module's saved output, never recompute silently" convention
    generate_focused_projections.py already uses for criterion-neighbours
    output).

The actual Voronoi cell geometry (scipy.spatial.Voronoi) is computed at
RENDER time, in generate_figures.py, from the seed coordinates this module
writes -- this module only fits/writes 2-D coordinates, same "analyze/
generate compute and write, a separate rendering pass draws PNGs"
convention as generate_focused_projections.py + generate_figures.py.

GAP GRID: bins the (shared) entity 2-D layout into a grid over its own
bounding box, counts entities per cell, and flags near-EMPTY cells INSIDE
the convex hull of the actual entity points as candidate "gaps" -- a cell
outside the hull is just unoccupied space at the edge of the 2-D
projection, not a meaningful coverage gap, so it is explicitly excluded
rather than counted. Like every 2-D picture in this toolkit, this is a
READING AID over the entities' 2-D UMAP layout specifically -- it is not
independent evidence about the full high-dimensional space (that's
analyze_entity_topology.py's job, in the real space).

Usage (from thesis/corpus/):
    python -m thesis_corpus.generate_voronoi_projections --run-id <existing-run-id>
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, Delaunay

from thesis_corpus import geometric_analysis_common as gac

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("thesis_corpus.generate_voronoi_projections")

MODULE_NAME = "generate_voronoi_projections"
DEFAULT_UMAP_NEIGHBORS = 20
DEFAULT_UMAP_MIN_DIST = 0.2
DEFAULT_GRID_BINS = 15
GAP_COUNT_THRESHOLD = 0  # a cell with <= this many entities, inside the hull, counts as a gap


# ---------------------------------------------------------------------------
# Seed builders -- each returns (points, vectors) for one strategy's overlay.
# ---------------------------------------------------------------------------

def criteria_seeds(shared_space: gac.SharedSpace) -> tuple[list[dict], np.ndarray]:
    idxs = gac.source_dataset_indices(shared_space.points, "miviludes_criteria")
    return [shared_space.points[i] for i in idxs], shared_space.vectors[idxs]


def corpus_centroid_seeds(per_source: dict[str, dict]) -> tuple[list[dict], np.ndarray]:
    points, vectors = [], []
    for corpus in gac.EXPRESSION_CORPORA:
        points.append({
            "key": f"centroid:{corpus}", "source_dataset": corpus,
            "point_role": "centroid", "label": f"{corpus} centroid",
        })
        vectors.append(per_source[corpus]["centroid"])
    return points, np.array(vectors, dtype=np.float64)


def cluster_centroid_seeds(entity_vectors: np.ndarray, cluster_rows: list[dict]) -> tuple[list[dict], np.ndarray]:
    """`cluster_rows` is analyze_entity_topology.py's own entity_clusters.csv
    rows (one per entity, carrying its cluster_id) -- NOT recomputed here,
    per this module's own "never silently recompute a sibling module's
    already-saved output" rule."""
    by_cluster: dict[int, list[int]] = {}
    for i, row in enumerate(cluster_rows):
        cluster_id = int(row["cluster_id"])
        if cluster_id == -1:
            continue
        by_cluster.setdefault(cluster_id, []).append(i)
    points, vectors = [], []
    for cluster_id in sorted(by_cluster):
        idxs = by_cluster[cluster_id]
        centroid = entity_vectors[idxs].mean(axis=0)
        points.append({
            "key": f"cluster:{cluster_id}", "source_dataset": "emergent_entities",
            "point_role": "cluster_centroid", "label": f"cluster {cluster_id} ({len(idxs)} entities)",
        })
        vectors.append(centroid)
    return points, np.array(vectors, dtype=np.float64)


# ---------------------------------------------------------------------------
# Gap grid -- pure over an already-fitted 2-D coordinate array.
# ---------------------------------------------------------------------------

def gap_grid_rows(coords_2d: np.ndarray, bins: int) -> list[dict]:
    """One row per grid cell over coords_2d's own bounding box: how many
    entity points fall in it, whether the cell's centre is inside the
    convex hull of the entity points, and whether it counts as a gap
    (inside the hull, count <= GAP_COUNT_THRESHOLD)."""
    if len(coords_2d) < 4:
        raise ValueError(f"gap_grid_rows: need >=4 points for a convex hull, got {len(coords_2d)}")
    x_min, x_max = coords_2d[:, 0].min(), coords_2d[:, 0].max()
    y_min, y_max = coords_2d[:, 1].min(), coords_2d[:, 1].max()
    x_edges = np.linspace(x_min, x_max, bins + 1)
    y_edges = np.linspace(y_min, y_max, bins + 1)
    counts, _, _ = np.histogram2d(coords_2d[:, 0], coords_2d[:, 1], bins=[x_edges, y_edges])

    hull = ConvexHull(coords_2d)
    hull_delaunay = Delaunay(coords_2d[hull.vertices])

    rows = []
    for i in range(bins):
        for j in range(bins):
            cell_center = np.array([(x_edges[i] + x_edges[i + 1]) / 2, (y_edges[j] + y_edges[j + 1]) / 2])
            inside_hull = bool(hull_delaunay.find_simplex(cell_center) >= 0)
            count = int(counts[i, j])
            rows.append({
                "cell_x": i, "cell_y": j,
                "x_center": float(cell_center[0]), "y_center": float(cell_center[1]),
                "x_min": float(x_edges[i]), "x_max": float(x_edges[i + 1]),
                "y_min": float(y_edges[j]), "y_max": float(y_edges[j + 1]),
                "count": count, "inside_convex_hull": inside_hull,
                "is_gap": bool(inside_hull and count <= GAP_COUNT_THRESHOLD),
            })
    return rows


def coords_path_from_manifest_path(manifest_path: Path) -> Path:
    """fit_and_save_umap names the manifest "<stem>.manifest.json" and the
    coordinates "<stem>.jsonl" -- NOT the same stem with a swapped suffix
    (manifest_path.with_suffix(".jsonl") would wrongly produce
    "<stem>.manifest.jsonl", which fit_and_save_umap never writes)."""
    stem = manifest_path.name.removesuffix(".manifest.json")
    return manifest_path.parent / f"{stem}.jsonl"


def _read_member_coords_2d(coords_path: Path) -> np.ndarray:
    """Reads back the "member" rows (excludes any overlay/seed rows) from
    a fit_and_save_umap-written .jsonl coordinate file, in file order."""
    coords = []
    with open(coords_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["point_kind"] == "member":
                coords.append(row["umap_2d"])
    return np.array(coords, dtype=np.float64)


def _read_csv_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", type=str, required=True,
                         help="An existing run-id with analyze_entity_topology.py already run (cluster seeds "
                              "are read from its saved entity_clusters.csv, never recomputed here).")
    parser.add_argument("--shared-space-dir", type=Path, default=None,
                         help="Use a different pooled space (e.g. processed/shared_space_v2/) instead of v1's "
                              "processed/shared_space/; also switches the run-output root to a sibling "
                              "processed/analysis_v2/ directory so v1 and v2 runs are never mixed.")
    parser.add_argument("--seed", type=int, default=gac.DEFAULT_SEED)
    parser.add_argument("--umap-neighbors", type=int, default=DEFAULT_UMAP_NEIGHBORS)
    parser.add_argument("--umap-min-dist", type=float, default=DEFAULT_UMAP_MIN_DIST)
    parser.add_argument("--grid-bins", type=int, default=DEFAULT_GRID_BINS)
    args = parser.parse_args()

    embedding_space_path, interview_prototypes_path, analysis_root = gac.resolve_space_paths(args.shared_space_dir)
    logger.info("Loading %s ...", embedding_space_path)
    shared_space = gac.load_shared_space(embedding_space_path)

    run_dir = gac.existing_run_dir(args.run_id, analysis_root)
    gac.init_run_manifest(run_dir, shared_space, defaults={"seed": args.seed})
    gac.update_run_manifest(run_dir, MODULE_NAME, "running")
    out_dir = gac.module_run_dir(run_dir, MODULE_NAME)

    try:
        entity_idxs = gac.source_dataset_indices(shared_space.points, "emergent_entities")
        entity_points = [shared_space.points[i] for i in entity_idxs]
        entity_vectors = shared_space.vectors[entity_idxs]
        logger.info("%d emergent entities as the shared UMAP basis for all 3 strategies + the gap grid.", len(entity_points))

        topology_dir = run_dir / "analyze_entity_topology"
        clusters_csv = topology_dir / "entity_clusters.csv"
        if not clusters_csv.exists():
            raise SystemExit(
                f"Missing: {clusters_csv} -- run analyze_entity_topology.py for run-id {args.run_id!r} first "
                "(cluster-centroid seeds are read from its output, never recomputed here)."
            )
        cluster_rows = _read_csv_rows(clusters_csv)

        per_source = gac.per_source_centroids_and_dispersion(shared_space)

        strategies = {
            "voronoi_criteria_seeded": criteria_seeds(shared_space),
            "voronoi_corpus_centroid_seeded": corpus_centroid_seeds(per_source),
            "voronoi_cluster_centroid_seeded": cluster_centroid_seeds(entity_vectors, cluster_rows),
        }

        manifest_paths: dict[str, str] = {}
        first_coords_path: Path | None = None
        for name, (seed_points, seed_vectors) in strategies.items():
            logger.info("%s: fitting UMAP (%d seeds as overlay)...", name, len(seed_points))
            manifest_path = gac.fit_and_save_umap(
                out_dir, name, entity_points, entity_vectors,
                args.umap_neighbors, args.umap_min_dist, args.seed,
                overlay_points=seed_points, overlay_vectors=seed_vectors,
            )
            manifest_paths[name] = str(manifest_path.name)
            if first_coords_path is None:
                first_coords_path = coords_path_from_manifest_path(manifest_path)

        logger.info("Gap grid (%dx%d bins) on the shared entity layout...", args.grid_bins, args.grid_bins)
        member_coords = _read_member_coords_2d(first_coords_path)
        grid_rows = gap_grid_rows(member_coords, args.grid_bins)
        n_gaps = sum(1 for r in grid_rows if r["is_gap"])
        n_inside = sum(1 for r in grid_rows if r["inside_convex_hull"])
        logger.info("%d/%d in-hull grid cells are gaps (count <= %d).", n_gaps, n_inside, GAP_COUNT_THRESHOLD)
        gac.write_csv(out_dir / "entity_gap_grid.csv", grid_rows)

        gac.write_module_config(
            out_dir,
            input_sha256=shared_space.input_sha256,
            seed=args.seed,
            umap_neighbors=args.umap_neighbors,
            umap_min_dist=args.umap_min_dist,
            grid_bins=args.grid_bins,
            n_entities=len(entity_points),
            strategies={name: len(pts) for name, (pts, _v) in strategies.items()},
            n_gap_cells=n_gaps,
            n_in_hull_cells=n_inside,
            manifest_paths=manifest_paths,
            git_commit=gac.git_commit_hash(),
        )
        gac.update_run_manifest(run_dir, MODULE_NAME, "completed", output_dir=out_dir)
    except Exception as e:
        gac.update_run_manifest(run_dir, MODULE_NAME, "failed", error=str(e))
        raise

    print(f"\nDone. Run {args.run_id} -> {out_dir} (3 Voronoi strategies + gap grid)")


if __name__ == "__main__":
    main()
