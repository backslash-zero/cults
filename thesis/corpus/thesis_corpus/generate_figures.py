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
    """Writes both a 300-DPI PNG and an SVG.

    FIGURE-VALIDATION / REGRESSION POLICY (formalized here, applies to
    every figure this module produces, not just one deliverable):
    deterministic data outputs -- CSV, JSON, and the raster PNG rendered
    here -- require byte-identical comparison against a baseline, full
    stop, same as every other output in this toolkit. Matplotlib's SVG
    backend is the one documented exception: it embeds non-semantic,
    run-varying metadata (a <dc:date> timestamp, and randomly-generated
    clip-path/marker element IDs) into every SVG it writes, even when the
    plotted content is byte-for-byte identical -- this is a property of
    the SVG backend itself, not of anything computed in this module. So
    for an SVG specifically: byte-level identity is NOT required; instead,
    validate the corresponding PNG byte-identically (it has no such
    metadata and IS required to match exactly), and treat an SVG diff as
    acceptable ONLY when every difference is confined to a <dc:date> line
    and clip-path/marker id="..."/xlink:href="#..." tokens -- confirm by
    checking that the embedded base64 raster payload (the long
    "iVBORw0KGgo..." PNG data URI matplotlib embeds inside the SVG for an
    imshow-based heatmap) is identical, which is the actual data-bearing
    content. Any SVG difference OUTSIDE that specific pattern is a real
    regression, not covered by this exception, and must be treated as a
    normal validation failure -- never silently waved through, and never
    "fixed" by hand-editing a previously-saved SVG or by weakening PNG/CSV
    validation elsewhere."""
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


def plot_criterion_composition_heatmap(composition_path: Path, out_dir: Path, run_id: str) -> None:
    """Candidate figure for the eventual main text -- no main-text-vs-
    appendix placement decision is made here or anywhere in the toolkit;
    that's the researcher's call after inspecting this figure and its
    source CSV (criterion_equal_n_neighbour_composition.csv). k=10 only
    (the CSV itself also has k=20; this figure picks one for readability).
    Filename, title, and a JSON sidecar all identify the same parameters
    so this figure is never separated from its own provenance.

    VALIDATION NOTE (distinct from save_figure's SVG-metadata exception):
    this PNG's own rendered title text includes `run_id`, by design (the
    same provenance requirement that put it in the JSON sidecar). Between
    two runs against an otherwise-identical shared space, this PNG is
    therefore expected to differ in a small, title-sized pixel region
    even when every analytical value (k, equal_n_candidate_count,
    bootstrap_reps, seed, source_order) is unchanged -- confirm via the
    two runs' own `.config.json` sidecars (every field but `run_id`
    itself must match) rather than treating any PNG diff here as a
    regression by default. This is unlike every other PNG this toolkit
    produces, which carry no run-identifying text and are expected to be
    genuinely byte-identical across runs against the same data."""
    rows = read_csv_rows(composition_path)
    rows = [r for r in rows if r.get("k") == "10"]
    if not rows:
        logger.warning("Skipping criterion-composition heatmap -- missing/empty %s", composition_path)
        return
    criteria = sorted({r["criterion_key"] for r in rows})
    sources = [s for s in gac.EXPRESSION_CORPORA if s in {r["source"] for r in rows}]
    matrix = np.zeros((len(criteria), len(sources)))
    crit_index = {c: i for i, c in enumerate(criteria)}
    src_index = {s: i for i, s in enumerate(sources)}
    for r in rows:
        matrix[crit_index[r["criterion_key"]], src_index[r["source"]]] = float(r["mean_fraction"])

    equal_n_candidate_count = rows[0]["equal_n_candidate_count"]
    bootstrap_reps = rows[0]["bootstrap_reps"]
    seed = rows[0]["seed"]

    fig, ax = plt.subplots(figsize=(6, 10))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(sources)), sources)
    ax.set_yticks(range(len(criteria)), criteria, fontsize=6)
    ax.set_title(
        "Criterion neighbour composition -- CANDIDATE FIGURE\n"
        "(equal-n bootstrapped, k=10, french_primary_shared_space,\n"
        f"n_candidates={equal_n_candidate_count}, B={bootstrap_reps}, seed={seed}, run={run_id})",
        fontsize=8,
    )
    fig.colorbar(im, ax=ax, label="Mean neighbour fraction")
    stem = "criterion_neighbour_composition_heatmap_k10"
    save_figure(fig, out_dir, stem)
    (out_dir / f"{stem}.config.json").write_text(json.dumps({
        "criterion_representation": "french_primary_shared_space",
        "k": 10,
        "equal_n_candidate_count": equal_n_candidate_count,
        "bootstrap_reps": bootstrap_reps,
        "seed": seed,
        "source_order": sources,
        "run_id": run_id,
        "placement": "candidate -- main-text vs. appendix not decided by the toolkit",
    }, indent=2), encoding="utf-8")


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


# ---------------------------------------------------------------------------
# Entity topology / Voronoi / publication-date figures
# ---------------------------------------------------------------------------

def plot_voronoi_diagram(coords_path: Path, manifest_path: Path, out_dir: Path) -> None:
    """Entities as a grey scatter + Voronoi cells built from the overlay
    (seed) points -- reads the SAME coordinate files
    generate_voronoi_projections.py writes via gac.fit_and_save_umap
    (member=entities, overlay=one strategy's seeds), same checksum-verified
    load as plot_2d_scatter. scipy.spatial.Voronoi needs >=4 seeds for a
    2-D diagram; fewer (e.g. the 3-corpus-centroid strategy) gets seed
    markers only, no cell boundaries, rather than an error."""
    from scipy.spatial import Voronoi, voronoi_plot_2d

    points, manifest = _load_checked_coords(coords_path, manifest_path)
    if points is None:
        logger.warning("Skipping Voronoi figure -- missing %s or %s", coords_path.name, manifest_path.name)
        return
    members = [p for p in points if p.get("point_kind", "member") == "member"]
    seeds = [p for p in points if p.get("point_kind") == "centroid_overlay"]
    if not seeds:
        logger.warning("Skipping Voronoi figure for %s -- no seed/overlay points found.", manifest["population"])
        return

    member_coords = np.array([p["umap_2d"] for p in members])
    seed_coords = np.array([p["umap_2d"] for p in seeds])

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(member_coords[:, 0], member_coords[:, 1], s=8, alpha=0.5,
               color="#888888", linewidths=0, label="entity", zorder=1)

    if len(seed_coords) >= 4:
        vor = Voronoi(seed_coords)
        voronoi_plot_2d(vor, ax=ax, show_points=False, show_vertices=False,
                         line_colors="black", line_width=1.2, line_alpha=0.8)
    else:
        logger.info("%s: only %d seeds (<4) -- seed markers only, no Voronoi cells (scipy needs >=4).",
                     manifest["population"], len(seed_coords))

    ax.scatter(seed_coords[:, 0], seed_coords[:, 1], s=160, color="#d62728",
               edgecolors="black", linewidths=1.3, label="seed", zorder=5, marker="*")
    for p, coord in zip(seeds, seed_coords):
        ax.annotate(str(p.get("label", ""))[:30], coord, fontsize=6, alpha=0.8, zorder=6)

    # A seed reached via reducer.transform() into a manifold it wasn't
    # fitted on can land far outside the entity cloud (a known UMAP
    # out-of-sample-transform behaviour) -- clip the view to the entities'
    # own bounding box (+10% padding) so the figure doesn't shrink every
    # entity into one corner to make room for a stray seed.
    pad_x = (member_coords[:, 0].max() - member_coords[:, 0].min()) * 0.1
    pad_y = (member_coords[:, 1].max() - member_coords[:, 1].min()) * 0.1
    ax.set_xlim(member_coords[:, 0].min() - pad_x, member_coords[:, 0].max() + pad_x)
    ax.set_ylim(member_coords[:, 1].min() - pad_y, member_coords[:, 1].max() + pad_y)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"{manifest['population']} (n_entities={len(members)}, n_seeds={len(seeds)})")
    ax.legend(fontsize=8, loc="best")
    # manifest['population'] is already "voronoi_<strategy>_seeded" (the
    # population_name generate_voronoi_projections.py passed to
    # fit_and_save_umap) -- don't double-prefix it.
    save_figure(fig, out_dir, manifest["population"])


def plot_entity_clusters(coords_path: Path, manifest_path: Path, clusters_csv_path: Path, out_dir: Path) -> None:
    """Colours the entity 2-D layout (any of generate_voronoi_projections.py's
    3 coordinate files' member points -- numerically identical across all
    three, see that module's own docstring) by
    analyze_entity_topology.py's HDBSCAN cluster_id -- a qualitative
    colormap (cluster count varies run to run), not the fixed
    source_dataset/point_role palette plot_2d_scatter uses."""
    points, manifest = _load_checked_coords(coords_path, manifest_path)
    if points is None:
        logger.warning("Skipping entity-cluster figure -- missing %s or %s", coords_path.name, manifest_path.name)
        return
    cluster_rows = read_csv_rows(clusters_csv_path)
    if not cluster_rows:
        logger.warning("Skipping entity-cluster figure -- missing/empty %s", clusters_csv_path)
        return
    cluster_by_key = {r["key"]: int(r["cluster_id"]) for r in cluster_rows}

    members = [p for p in points if p.get("point_kind", "member") == "member"]
    coords = np.array([p["umap_2d"] for p in members])
    cluster_ids = np.array([cluster_by_key.get(p["key"], -1) for p in members])

    fig, ax = plt.subplots(figsize=(9, 8))
    noise_mask = cluster_ids == -1
    ax.scatter(coords[noise_mask, 0], coords[noise_mask, 1], s=6, alpha=0.35,
               color="#bbbbbb", linewidths=0, label="noise (no cluster)")

    non_noise_ids = sorted(set(cluster_ids[~noise_mask].tolist()))
    cmap = plt.get_cmap("tab20")
    for i, cluster_id in enumerate(non_noise_ids):
        mask = cluster_ids == cluster_id
        ax.scatter(coords[mask, 0], coords[mask, 1], s=18, alpha=0.85, color=cmap(i % 20), linewidths=0)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"Emergent entities coloured by HDBSCAN cluster "
                 f"({len(non_noise_ids)} clusters, {int(noise_mask.sum())} noise)")
    ax.legend(fontsize=8, loc="best")
    save_figure(fig, out_dir, "entity_clusters_umap")


def plot_gap_heatmap(gap_grid_path: Path, out_dir: Path) -> None:
    rows = read_csv_rows(gap_grid_path)
    if not rows:
        logger.warning("Skipping gap heatmap -- missing/empty %s", gap_grid_path)
        return
    bins_x = max(int(r["cell_x"]) for r in rows) + 1
    bins_y = max(int(r["cell_y"]) for r in rows) + 1
    grid = np.full((bins_y, bins_x), np.nan)
    gap_cells: list[tuple[int, int]] = []
    for r in rows:
        i, j = int(r["cell_x"]), int(r["cell_y"])
        if r["inside_convex_hull"] == "True":
            grid[j, i] = int(r["count"])
        if r["is_gap"] == "True":
            gap_cells.append((i, j))

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(grid, origin="lower", cmap="YlOrRd", aspect="auto")
    fig.colorbar(im, ax=ax, label="entities per cell")
    if gap_cells:
        gap_xs, gap_ys = zip(*gap_cells)
        ax.scatter(gap_xs, gap_ys, marker="x", color="blue", s=40, label="gap (inside hull, empty)")
    ax.set_xlabel("grid cell (x)")
    ax.set_ylabel("grid cell (y)")
    ax.set_title("Entity density grid -- candidate coverage gaps marked")
    ax.legend(fontsize=8)
    save_figure(fig, out_dir, "entity_gap_heatmap")


def plot_year_vs_distance(year_distance_csv: Path, out_dir: Path) -> None:
    rows = read_csv_rows(year_distance_csv)
    if not rows:
        logger.warning("Skipping year-vs-distance figure -- missing/empty %s", year_distance_csv)
        return
    years = np.array([int(r["year"]) for r in rows])
    distances = np.array([float(r["distance_to_literature_centroid"]) for r in rows])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(years, distances, s=10, alpha=0.25,
               color=gac.SOURCE_DATASET_COLORS.get("literature", "#1f77b4"), linewidths=0)
    unique_years = sorted(set(years.tolist()))
    year_means = [distances[years == y].mean() for y in unique_years]
    ax.plot(unique_years, year_means, color="black", linewidth=1.5, label="per-year mean")

    ax.set_xlabel("Publication year")
    ax.set_ylabel("Distance to literature's own centroid")
    ax.set_title("Literature: publication year vs. distance to own centroid")
    ax.legend(fontsize=8)
    save_figure(fig, out_dir, "publication_year_vs_distance")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--shared-space-dir", type=Path, default=None,
                         help="Must match whatever --shared-space-dir the run being read was itself produced "
                              "with -- switches the run-output root searched to a sibling processed/analysis_v2/ "
                              "directory instead of v1's processed/analysis/.")
    args = parser.parse_args()

    _, _, analysis_root = gac.resolve_space_paths(args.shared_space_dir)
    run_dir = gac.existing_run_dir(args.run_id, analysis_root)
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
        plot_criterion_composition_heatmap(cn_dir / "criterion_equal_n_neighbour_composition.csv", out_dir, run_dir.name)
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

    if "generate_voronoi_projections" in completed:
        vp_dir = run_dir / "generate_voronoi_projections"
        for coords_path in sorted(vp_dir.glob("umap_voronoi_*.jsonl")):
            manifest_path = coords_path.parent / f"{coords_path.stem}.manifest.json"
            plot_voronoi_diagram(coords_path, manifest_path, out_dir)
        plot_gap_heatmap(vp_dir / "entity_gap_grid.csv", out_dir)
        if "analyze_entity_topology" in completed:
            criteria_seeded = sorted(vp_dir.glob("umap_voronoi_criteria_seeded_*.jsonl"))
            if criteria_seeded:
                coords_path = criteria_seeded[0]
                manifest_path = coords_path.parent / f"{coords_path.stem}.manifest.json"
                plot_entity_clusters(coords_path, manifest_path,
                                      run_dir / "analyze_entity_topology" / "entity_clusters.csv", out_dir)
    else:
        logger.warning("generate_voronoi_projections not completed in this run -- skipping its figures.")

    if "analyze_publication_date" in completed:
        pd_dir = run_dir / "analyze_publication_date"
        plot_year_vs_distance(pd_dir / "year_distance_by_point.csv", out_dir)
    else:
        logger.warning("analyze_publication_date not completed in this run -- skipping its figure.")

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
