"""Renders figures from a completed analysis run's saved output --
never recomputes any statistic and never calls UMAP (analyze_cluster_structure.py
and generate_focused_projections.py own every UMAP fit/PCA-2D slice in
this toolkit; this module only reads their saved coordinate files,
verifying each one's checksum against its own manifest before plotting,
rather than assuming it's untouched).

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

def _load_checked_coords(coords_path: Path, manifest_path: Path) -> tuple[list[dict] | None, dict | None]:
    if not coords_path.exists() or not manifest_path.exists():
        return None, None
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
    return points, manifest


def plot_2d_scatter(
    coords_path: Path, manifest_path: Path, out_dir: Path, color_by: str,
    coord_field: str = "umap_2d", stem_prefix: str = "umap",
) -> None:
    """Renders one 2-D scatter from a saved coordinate file + manifest --
    used for both analyze_cluster_structure.py's whole-space UMAP
    populations and generate_focused_projections.py's ~23 small curated
    populations (UMAP and PCA-2D alike; `coord_field`/`stem_prefix` select
    which). A point_kind=="centroid_overlay" row (present only in focused
    populations, never in analyze_cluster_structure.py's own output) is
    drawn as a larger, black-edged marker on top of the member points --
    same visual technique plot_initial_exemplar_highlight already uses for
    prototypes over a greyed-out background."""
    points, manifest = _load_checked_coords(coords_path, manifest_path)
    if points is None:
        logger.warning("Skipping figure -- missing %s or %s", coords_path.name, manifest_path.name)
        return

    color_map = gac.SOURCE_DATASET_COLORS if color_by == "source_dataset" else gac.POINT_ROLE_COLORS
    fig, ax = plt.subplots(figsize=(9, 8))

    members = [p for p in points if p.get("point_kind", "member") == "member"]
    overlays = [p for p in points if p.get("point_kind") == "centroid_overlay"]

    groups: dict[str, list[list[float]]] = {}
    for p in members:
        key = p.get(color_by)
        groups.setdefault(key, []).append(p[coord_field])
    point_alpha = 0.5 if len(members) > 500 else 0.85
    for key, coords in groups.items():
        coords_arr = np.array(coords)
        ax.scatter(coords_arr[:, 0], coords_arr[:, 1], s=4 if len(members) > 500 else 24, alpha=point_alpha,
                   color=color_map.get(key, "#999999"), label=key, linewidths=0)

    if overlays:
        overlay_coords = np.array([p[coord_field] for p in overlays])
        overlay_colors = [color_map.get(p.get(color_by), "#000000") for p in overlays]
        ax.scatter(overlay_coords[:, 0], overlay_coords[:, 1], s=140, color=overlay_colors,
                   edgecolors="black", linewidths=1.3, label="centroid", zorder=5)

    axis_label = "UMAP" if coord_field == "umap_2d" else "PC"
    ax.set_xlabel(f"{axis_label} 1")
    ax.set_ylabel(f"{axis_label} 2")
    n_rendered = manifest.get("n_points_rendered", manifest.get("n_points"))
    if "umap_params" in manifest:
        params_str = (f", n_neighbors={manifest['umap_params']['n_neighbors']}, "
                      f"min_dist={manifest['umap_params']['min_dist']}")
        stem_suffix = f"n{manifest['umap_params']['n_neighbors']}_d{manifest['umap_params']['min_dist']}"
    else:
        params_str = " (PCA-2D slice)"
        stem_suffix = "pca"
    ax.set_title(f"{manifest['population']} (n={n_rendered}{params_str})")
    ax.legend(markerscale=3, fontsize=7, loc="best")
    save_figure(fig, out_dir, f"{stem_prefix}_{manifest['population']}_by_{color_by}_{stem_suffix}")


def plot_umap_scatter(coords_path: Path, manifest_path: Path, out_dir: Path, color_by: str) -> None:
    """Backward-compatible name for analyze_cluster_structure.py's own 4
    whole-space UMAP populations -- delegates to plot_2d_scatter."""
    plot_2d_scatter(coords_path, manifest_path, out_dir, color_by, coord_field="umap_2d", stem_prefix="umap")


def plot_initial_exemplar_highlight(prototype_coords_path: Path, prototype_manifest_path: Path,
                                     exemplar_summary_path: Path, out_dir: Path) -> None:
    """Reads the "with_interview_prototypes" UMAP population directly
    (analyze_cluster_structure.py) -- NOT "overview": several prototypes
    (e.g. "AI cult", "Illuminati") were dropped by build_shared_space.py's
    pooling filter and have no position in the ordinary pooled space to
    look up at all, so this needs a fit that actually includes them."""
    if not prototype_coords_path.exists() or not exemplar_summary_path.exists():
        logger.warning(
            "Skipping initial-exemplar highlight -- missing %s or exemplar_summary.csv "
            "(run build_interview_prototype_layer.py, then analyze_cluster_structure.py, "
            "for this figure).",
            prototype_coords_path.name,
        )
        return

    manifest = json.loads(prototype_manifest_path.read_text(encoding="utf-8"))
    actual_checksum = gac.sha256_file(prototype_coords_path)
    if actual_checksum != manifest["output_sha256"]:
        raise SystemExit(f"{prototype_coords_path} checksum mismatch against its manifest -- refusing to plot.")

    background_coords = []
    proto_coord_by_document_id = {}
    with open(prototype_coords_path, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            if p.get("source_dataset") == "interview_prototypes":
                document_id = p["key"].removeprefix("prototype:")
                proto_coord_by_document_id[document_id] = p["umap_2d"]
            else:
                background_coords.append(p["umap_2d"])

    exemplar_type_by_document_id = {
        r["document_id"]: r["exemplar_type"]
        for r in read_csv_rows(exemplar_summary_path) if r.get("status") == "resolved"
    }
    if not proto_coord_by_document_id:
        logger.warning("No interview_prototypes points in %s -- skipping highlight figure.", prototype_coords_path)
        return

    all_coords = np.array(background_coords)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(all_coords[:, 0], all_coords[:, 1], s=2, alpha=0.15, color="#999999", linewidths=0)

    okabe_ito_hexes = list(gac.OKABE_ITO.values())
    type_order = sorted(set(exemplar_type_by_document_id.values()))
    for i, exemplar_type in enumerate(type_order):
        hue = okabe_ito_hexes[i % len(okabe_ito_hexes)]
        coords = np.array([
            proto_coord_by_document_id[doc_id] for doc_id, et in exemplar_type_by_document_id.items()
            if et == exemplar_type and doc_id in proto_coord_by_document_id
        ])
        if len(coords) == 0:
            continue
        ax.scatter(coords[:, 0], coords[:, 1], s=40, color=hue, label=exemplar_type, edgecolors="black", linewidths=0.5)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Initial exemplars by exemplar_type (with_interview_prototypes UMAP background)")
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

    gac.update_run_manifest(run_dir, MODULE_NAME, "running")
    out_dir = gac.module_run_dir(run_dir, MODULE_NAME)

    try:
        _render_all(run_dir, completed, out_dir)
    except Exception as e:
        gac.update_run_manifest(run_dir, MODULE_NAME, "failed", error=str(e))
        raise
    gac.update_run_manifest(run_dir, MODULE_NAME, "completed", output_dir=out_dir)

    print(f"\nDone. Figures -> {out_dir}")


def _render_all(run_dir: Path, completed: set[str], out_dir: Path) -> None:
    """Every figure this module renders -- pulled out of main() so it can
    be wrapped in the same try/except-then-update_run_manifest pattern
    every other toolkit module uses (this module previously never
    registered itself in RUN_MANIFEST.json at all -- a pre-existing gap,
    fixed here rather than carried forward)."""
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
        plot_criterion_corpus_heatmap(cn_dir / "controlled_comparison_french_primary_shared_space.csv", out_dir)
    else:
        logger.warning("analyze_criterion_neighbours not completed in this run -- skipping its figure.")

    if "analyze_emergent_entities" in completed:
        ee_dir = run_dir / "analyze_emergent_entities"
        plot_entity_provenance_bars(ee_dir / "provenance_category_counts.csv", out_dir)
    else:
        logger.warning("analyze_emergent_entities not completed in this run -- skipping its figure.")

    if "generate_focused_projections" in completed:
        fp_dir = run_dir / "generate_focused_projections"
        umap_files = sorted(fp_dir.glob("umap_*.jsonl"))
        pca_files = sorted(fp_dir.glob("pca_*.jsonl"))
        logger.info("generate_focused_projections: rendering %d UMAP + %d PCA-2D coordinate files...",
                    len(umap_files), len(pca_files))
        for coords_path in umap_files:
            manifest_path = coords_path.parent / f"{coords_path.stem}.manifest.json"
            for color_by in ("source_dataset", "point_role"):
                plot_2d_scatter(coords_path, manifest_path, out_dir, color_by, coord_field="umap_2d", stem_prefix="umap")
        for coords_path in pca_files:
            manifest_path = coords_path.parent / f"{coords_path.stem}.manifest.json"
            for color_by in ("source_dataset", "point_role"):
                plot_2d_scatter(coords_path, manifest_path, out_dir, color_by, coord_field="pca_2d", stem_prefix="pca")
    else:
        logger.warning("generate_focused_projections not completed in this run -- skipping its figures.")

    if "analyze_initial_exemplars" in completed and "analyze_cluster_structure" in completed:
        cs_dir = run_dir / "analyze_cluster_structure"
        ie_dir = run_dir / "analyze_initial_exemplars"
        with_prototypes = sorted(cs_dir.glob("umap_with_interview_prototypes_*.jsonl"))
        if with_prototypes:
            plot_initial_exemplar_highlight(
                with_prototypes[0], with_prototypes[0].parent / f"{with_prototypes[0].stem}.manifest.json",
                ie_dir / "exemplar_summary.csv", out_dir,
            )


if __name__ == "__main__":
    main()
