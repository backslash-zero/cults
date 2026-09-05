"""Renders figures from a completed analysis run's saved output --
never recomputes any statistic and never calls UMAP (analyze_cluster_structure.py
owns every UMAP fit in this toolkit; this module only reads its saved
coordinate files, verifying each one's checksum against its own manifest
before plotting, rather than assuming it's untouched).

Matplotlib only (no new plotting dependency), Okabe-Ito colour-vision-
deficiency-safe categorical palette, every figure written as both 300-DPI
PNG and SVG.

Requires an existing --run-id produced by the analysis modules -- never
falls back to "latest" and never scans other runs. Figures that need a
module this run didn't produce are skipped with a clear message, not
silently omitted.

Usage (from thesis/corpus/):
    python -m thesis_corpus.generate_figures --run-id 20260101-120000
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from thesis_corpus import geometric_analysis_common as gac

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("thesis_corpus.generate_figures")

MODULE_NAME = "generate_figures"
FIGURE_DPI = 300


def save_figure(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# 2-D UMAP scatter (reads analyze_cluster_structure.py's saved coordinates only)
# ---------------------------------------------------------------------------

def plot_umap_scatter(coords_path: Path, manifest_path: Path, out_dir: Path, color_by: str) -> None:
    if not coords_path.exists() or not manifest_path.exists():
        logger.warning("Skipping UMAP figure -- missing %s or %s", coords_path.name, manifest_path.name)
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_checksum = gac.sha256_file(coords_path)
    if actual_checksum != manifest["output_sha256"]:
        raise SystemExit(
            f"Coordinate file {coords_path} does not match its own manifest's recorded "
            f"checksum ({manifest['output_sha256'][:12]}... vs actual {actual_checksum[:12]}...) "
            "-- refusing to plot data that may have been altered since it was computed."
        )

    points = []
    with open(coords_path, encoding="utf-8") as f:
        for line in f:
            points.append(json.loads(line))

    color_map = gac.SOURCE_DATASET_COLORS if color_by == "source_dataset" else gac.POINT_ROLE_COLORS
    fig, ax = plt.subplots(figsize=(9, 8))
    groups: dict[str, list[list[float]]] = {}
    for p in points:
        key = p.get(color_by)
        groups.setdefault(key, []).append(p["umap_2d"])

    for key, coords in groups.items():
        coords_arr = np.array(coords)
        ax.scatter(coords_arr[:, 0], coords_arr[:, 1], s=4, alpha=0.5,
                   color=color_map.get(key, "#999999"), label=key, linewidths=0)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"{manifest['population']} (n={manifest['n_points']}, "
                 f"n_neighbors={manifest['umap_params']['n_neighbors']}, "
                 f"min_dist={manifest['umap_params']['min_dist']})")
    ax.legend(markerscale=4, fontsize=8, loc="best")
    save_figure(fig, out_dir, f"umap_{manifest['population']}_by_{color_by}_"
                              f"n{manifest['umap_params']['n_neighbors']}_"
                              f"d{manifest['umap_params']['min_dist']}")


def plot_initial_exemplar_highlight(overview_coords_path: Path, overview_manifest_path: Path,
                                     exemplar_summary_path: Path, out_dir: Path) -> None:
    if not overview_coords_path.exists() or not exemplar_summary_path.exists():
        logger.warning("Skipping initial-exemplar highlight -- missing overview UMAP or exemplar_summary.csv")
        return

    manifest = json.loads(overview_manifest_path.read_text(encoding="utf-8"))
    actual_checksum = gac.sha256_file(overview_coords_path)
    if actual_checksum != manifest["output_sha256"]:
        raise SystemExit(f"{overview_coords_path} checksum mismatch against its manifest -- refusing to plot.")

    coord_by_key = {}
    with open(overview_coords_path, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            coord_by_key[p["key"]] = p["umap_2d"]

    exemplars = [r for r in read_csv_rows(exemplar_summary_path) if r.get("status") == "resolved"]
    if not exemplars:
        logger.warning("No resolved exemplars in %s -- skipping highlight figure.", exemplar_summary_path)
        return

    all_coords = np.array(list(coord_by_key.values()))
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(all_coords[:, 0], all_coords[:, 1], s=2, alpha=0.15, color="#999999", linewidths=0)

    okabe_ito_hexes = list(gac.OKABE_ITO.values())
    type_order = sorted({e["exemplar_type"] for e in exemplars})
    for i, exemplar_type in enumerate(type_order):
        hue = okabe_ito_hexes[i % len(okabe_ito_hexes)]
        subset = [e for e in exemplars if e["exemplar_type"] == exemplar_type]
        coords = np.array([coord_by_key[e["source_expression_key"].rsplit(":", 1)[0]] for e in subset if e["source_expression_key"].rsplit(":", 1)[0] in coord_by_key])
        if len(coords) == 0:
            continue
        ax.scatter(coords[:, 0], coords[:, 1], s=40, color=hue, label=exemplar_type, edgecolors="black", linewidths=0.5)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Initial exemplars by exemplar_type (overview UMAP background)")
    ax.legend(fontsize=7, loc="best")
    save_figure(fig, out_dir, "initial_exemplars_by_type")


# ---------------------------------------------------------------------------
# Heatmaps and bar charts (analyze_global_structure / analyze_cluster_structure /
# analyze_criterion_neighbours / analyze_emergent_entities outputs)
# ---------------------------------------------------------------------------

def plot_centroid_heatmap(pairwise_path: Path, out_dir: Path) -> None:
    rows = read_csv_rows(pairwise_path)
    if not rows:
        logger.warning("Skipping centroid heatmap -- missing %s", pairwise_path)
        return
    sources = sorted({r["source_a"] for r in rows} | {r["source_b"] for r in rows})
    matrix = np.zeros((len(sources), len(sources)))
    index = {s: i for i, s in enumerate(sources)}
    for r in rows:
        matrix[index[r["source_a"]], index[r["source_b"]]] = float(r["euclidean_distance"])

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(sources)), sources, rotation=45, ha="right")
    ax.set_yticks(range(len(sources)), sources)
    ax.set_title("Pairwise source-centroid Euclidean distance")
    fig.colorbar(im, ax=ax, label="Euclidean distance")
    save_figure(fig, out_dir, "pairwise_source_centroid_heatmap")


def plot_knn_composition_bars(knn_path: Path, out_dir: Path, stem: str, k_filter: str) -> None:
    rows = [r for r in read_csv_rows(knn_path) if r.get("k") == k_filter]
    if not rows:
        logger.warning("Skipping k-NN composition bars (%s) -- missing/empty %s", stem, knn_path)
        return
    query_corpora = sorted({r["query_corpus"] for r in rows})
    neighbor_corpora = sorted({r["neighbor_corpus"] for r in rows})
    value_field = "fraction" if "fraction" in rows[0] else "mean"

    fig, ax = plt.subplots(figsize=(7, 5))
    bottom = np.zeros(len(query_corpora))
    x = np.arange(len(query_corpora))
    for i, neighbor_corpus in enumerate(neighbor_corpora):
        values = np.array([
            float(next((r[value_field] for r in rows if r["query_corpus"] == q and r["neighbor_corpus"] == neighbor_corpus), 0))
            for q in query_corpora
        ])
        ax.bar(x, values, bottom=bottom, label=neighbor_corpus,
               color=gac.SOURCE_DATASET_COLORS.get(neighbor_corpus, "#999999"))
        bottom += values

    ax.set_xticks(x, query_corpora)
    ax.set_ylabel("Fraction of k nearest neighbours")
    ax.set_title(f"k-NN neighbour composition (k={k_filter})")
    ax.legend(fontsize=8)
    save_figure(fig, out_dir, stem)


def plot_criterion_corpus_heatmap(controlled_path: Path, out_dir: Path) -> None:
    rows = read_csv_rows(controlled_path)
    if not rows:
        logger.warning("Skipping criterion x corpus heatmap -- missing %s", controlled_path)
        return
    value_field = "mean_nearest10_distance_mean" if "mean_nearest10_distance_mean" in rows[0] else "mean_nearest10_cosine_mean"
    criteria = sorted({r["criterion_key"] for r in rows})
    corpora = sorted({r["corpus"] for r in rows})
    matrix = np.zeros((len(criteria), len(corpora)))
    crit_index = {c: i for i, c in enumerate(criteria)}
    corp_index = {c: i for i, c in enumerate(corpora)}
    for r in rows:
        matrix[crit_index[r["criterion_key"]], corp_index[r["corpus"]]] = float(r[value_field])

    fig, ax = plt.subplots(figsize=(6, 10))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(corpora)), corpora)
    ax.set_yticks(range(len(criteria)), criteria, fontsize=6)
    ax.set_title(f"Criterion x corpus controlled comparison ({value_field})")
    fig.colorbar(im, ax=ax)
    save_figure(fig, out_dir, "criterion_corpus_heatmap")


def plot_entity_provenance_bars(provenance_path: Path, out_dir: Path) -> None:
    rows = read_csv_rows(provenance_path)
    if not rows:
        logger.warning("Skipping entity-provenance bars -- missing %s", provenance_path)
        return
    categories = [r["provenance_category"] for r in rows]
    counts = [int(r["n_entities"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(categories, counts, color=gac.OKABE_ITO["blue"])
    ax.set_ylabel("Number of entities")
    ax.set_title("Emergent-entity provenance categories")
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=30, ha="right")
    save_figure(fig, out_dir, "entity_provenance_categories")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", type=str, required=True)
    args = parser.parse_args()

    run_dir = gac.existing_run_dir(args.run_id)
    manifest_path = run_dir / "RUN_MANIFEST.json"
    if not manifest_path.exists():
        raise SystemExit(f"No RUN_MANIFEST.json in {run_dir} -- this doesn't look like a toolkit run.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    completed = {name for name, info in manifest["modules"].items() if info.get("status") == "completed"}
    logger.info("Run %s -- completed modules: %s", args.run_id, sorted(completed))

    out_dir = gac.module_run_dir(run_dir, MODULE_NAME)

    if "analyze_global_structure" in completed:
        gs_dir = run_dir / "analyze_global_structure"
        plot_centroid_heatmap(gs_dir / "pairwise_source_centroid_matrix.csv", out_dir)
    else:
        logger.warning("analyze_global_structure not completed in this run -- skipping its figures.")

    if "analyze_cluster_structure" in completed:
        cs_dir = run_dir / "analyze_cluster_structure"
        plot_knn_composition_bars(cs_dir / "knn_composition_descriptive.csv", out_dir, "knn_composition_full_k10", "10")
        plot_knn_composition_bars(cs_dir / "knn_composition_equal_n_expression.csv", out_dir, "knn_composition_equal_n_k10", "10")
        for coords_path in sorted(cs_dir.glob("umap_*.jsonl")):
            manifest_path = coords_path.parent / f"{coords_path.stem}.manifest.json"
            for color_by in ("source_dataset", "point_role"):
                plot_umap_scatter(coords_path, manifest_path, out_dir, color_by)
    else:
        logger.warning("analyze_cluster_structure not completed in this run -- skipping its figures.")

    if "analyze_criterion_neighbours" in completed:
        cn_dir = run_dir / "analyze_criterion_neighbours"
        plot_criterion_corpus_heatmap(cn_dir / "controlled_comparison_french_primary.csv", out_dir)
    else:
        logger.warning("analyze_criterion_neighbours not completed in this run -- skipping its figure.")

    if "analyze_emergent_entities" in completed:
        ee_dir = run_dir / "analyze_emergent_entities"
        plot_entity_provenance_bars(ee_dir / "provenance_category_counts.csv", out_dir)
    else:
        logger.warning("analyze_emergent_entities not completed in this run -- skipping its figure.")

    if "analyze_initial_exemplars" in completed and "analyze_cluster_structure" in completed:
        cs_dir = run_dir / "analyze_cluster_structure"
        ie_dir = run_dir / "analyze_initial_exemplars"
        overview = sorted(cs_dir.glob("umap_overview_*.jsonl"))
        if overview:
            plot_initial_exemplar_highlight(
                overview[0], overview[0].parent / f"{overview[0].stem}.manifest.json",
                ie_dir / "exemplar_summary.csv", out_dir,
            )

    print(f"\nDone. Figures -> {out_dir}")


if __name__ == "__main__":
    main()
