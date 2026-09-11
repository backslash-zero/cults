"""Per-corpus centroid + k-nearest-neighbour pass over RAW bge-m3 embeddings
(no StandardScaler/PCA): for each of a small set of corpora, the group's own
centroid, its k nearest actual member points, and a UMAP-2D scatter plot
with the centroid and its neighbours annotated.

WHY RAW, NOT shared_space_v3: this module originally worked on
shared_space_v3's standardized+PCA'd coordinates (like the rest of this
toolkit). Checked empirically against this corpus's own data: standardizing
alone (no PCA at all) already produces about as much nearest-neighbour
ranking jitter as the full standardize+PCA-truncate pipeline (Spearman
rho~0.996 vs raw on a 94-point test group; PCA's rotation step itself is
exactly distance-preserving, verified to 1e-14 -- all the distortion comes
from standardization + truncation, not rotation). And the ORIGINAL reason
for pooling+PCA (comparability across independently-fit per-corpus spaces,
v1's reduce_embeddings.py problem) doesn't apply to raw vectors at all: they
all come from the same fixed bge-m3 model with no per-corpus fitting, so
they're already directly cosine-comparable with zero transformation. bge-m3
is itself trained with a cosine-similarity contrastive objective, so raw-
space cosine is the geometry the model was actually optimized to produce.
So: skip standardize+PCA entirely for this module's analytics, work
straight off the embedding_vector fields build_shared_space.py's own loader
functions already read (reused here unmodified, stopping right before its
StandardScaler/PCA step) -- see load_raw_points below.

Raw bge-m3 vectors are unit-norm (verified: every embedding_vector has
norm 1.0), which makes this simpler, not just "more native": for two
unit vectors, ||a-b||^2 = 2 - 2*cos(a,b), so ranking candidates by
Euclidean distance to a FIXED query (a centroid or another point) is
exactly monotonically equivalent to ranking by cosine similarity to
that same query, even though the centroid itself isn't unit-norm (its
own norm is a constant across all candidates, so it doesn't affect the
ranking -- see the module docstring's own derivation, kept short here).
So geometric_analysis_common's existing Euclidean-based
centroid_and_dispersion_for_indices/nearest_points already ARE
cosine-similarity rankings on this data, unchanged, no new metric code
needed.

VISUALIZATION: also switched from a fixed global PCA-2D slice to a
per-group UMAP-2D fit (metric="cosine", matching the vectors' own native
geometry -- geometric_analysis_common.fit_and_save_umap hardcodes
metric="euclidean" for its PCA-space callers, so a small local UMAP
fit is used here instead of that shared helper). Why: a linear PCA
slice's 2 axes are chosen to explain global variance, not to keep true
nearest neighbours visually adjacent -- confirmed on real output
(centroid_rapport.png's rank-1 neighbour sat in the plot's far corner).
UMAP's fitting objective is specifically to preserve LOCAL neighbourhood
structure, so annotated nearest-neighbours should now actually cluster
near the centroid star.

Usage (from thesis/corpus/):
    python -m thesis_corpus.analyze_corpus_centroids
    python -m thesis_corpus.analyze_corpus_centroids --lit-mivi-run-tag 20260910 \
        --interviews-archive-dir processed/interviews_full/interviews/run_20260912 --k 10
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

from thesis_corpus import build_shared_space as bss
from thesis_corpus import geometric_analysis_common as gac

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("thesis_corpus.analyze_corpus_centroids")

DEFAULT_K = 10
MIVILUDES_REPORT_DOCUMENT_ID = "mission-interministe-rielle-de-vigilance-et-de-lutte-contre-les-de-rives-sectair"
DICTIONARY_SOURCE_DATASETS = ("concept_backbone", "structural_concepts", "conceptnet_concepts")
ENTITY_MENTION_CORPORA = ("literature", "miviludes", "interviews")
UMAP_SEED = 42
UMAP_MIN_DIST = 0.1


def load_raw_points(lit_mivi_run_tag: str, interviews_archive_dir: Path, min_expression_words: int = 0) -> list[dict]:
    """Replicates build_shared_space.main()'s v2 pooling exactly (same
    loader functions, same translation/exclusion/filter behaviour) but
    STOPS before StandardScaler/PCA -- every returned point's "vector" is
    the raw embedding_vector (or, for MIVILUDES expressions, the raw
    English-translation embedding_vector_en -- build_shared_space.py's own
    deliberate cross-lingual-comparability choice, orthogonal to the
    standardize/PCA question this module exists to avoid, so left as-is)."""
    v2_root = bss.PROCESSED_DIR / "v2"
    corpus_archives = {
        "literature": v2_root / "literature" / f"run_{lit_mivi_run_tag}" / "criterion_expressions.jsonl",
        "miviludes": v2_root / "miviludes" / f"run_{lit_mivi_run_tag}" / "criterion_expressions.jsonl",
        "interviews": interviews_archive_dir / "criterion_expressions.jsonl",
    }
    for corpus, path in corpus_archives.items():
        if not path.exists():
            raise SystemExit(f"Missing corpus archive for {corpus!r}: {path}")

    miviludes_translations_path = v2_root / "miviludes" / f"run_{lit_mivi_run_tag}" / "expression_translations_embedded.jsonl"
    miviludes_translations = bss.load_miviludes_translations(miviludes_translations_path)

    points: list[dict] = []
    for corpus_name, path in corpus_archives.items():
        if corpus_name == "miviludes":
            new_points, _ = bss.load_miviludes_points_v2(path, miviludes_translations, min_expression_words)
        else:
            new_points, _ = bss.load_corpus_points_v2(corpus_name, path, min_expression_words)
        points.extend(new_points)

    points.extend(bss.load_miviludes_criteria_points(bss.MIVILUDES_CRITERIA_PATH))
    points.extend(bss.load_concept_backbone_points(bss.CONCEPT_BACKBONE_PATH))
    points.extend(bss.load_structural_concepts_points(bss.STRUCTURAL_CONCEPTS_PATH))
    points.extend(bss.load_conceptnet_concepts_points(bss.CONCEPTNET_CONCEPTS_PATH))

    domain_term_paths = {
        corpus: (path.parent / "chunk_terms.jsonl", path.parent / "domain_term_vectors.jsonl")
        for corpus, path in corpus_archives.items()
    }
    min_mentions_by_corpus = {corpus: bss.ENTITY_ANCHOR_MIN_MENTIONS for corpus in corpus_archives}
    cited_author_surnames = bss.load_cited_author_surnames()
    points.extend(bss.load_emergent_entities(corpus_archives, min_mentions_by_corpus, domain_term_paths, cited_author_surnames))

    return points


def group_indices(points: list[dict], group: str) -> list[int]:
    if group == "rapport":
        return [i for i, p in enumerate(points)
                if p["source_dataset"] == "miviludes" and gac.key_document_id(p["key"]) == MIVILUDES_REPORT_DOCUMENT_ID]
    if group == "sectarian_drift_list":
        return gac.source_dataset_indices(points, "miviludes_criteria")
    if group == "dictionary":
        return [i for i, p in enumerate(points) if p["source_dataset"] in DICTIONARY_SOURCE_DATASETS]
    if group == "entities_all":
        return gac.source_dataset_indices(points, "emergent_entities")
    if group.startswith("entities_") and group[len("entities_"):] in ENTITY_MENTION_CORPORA:
        corpus = group[len("entities_"):]
        return [i for i, p in enumerate(points)
                if p["source_dataset"] == "emergent_entities" and (p.get("mention_distribution") or {}).get(corpus, 0) > 0]
    if "__" in group:
        corpus, status = group.split("__", 1)
        return [i for i, p in enumerate(points) if p["source_dataset"] == corpus and p.get("epistemic_status") == status]
    return gac.source_dataset_indices(points, group)


GROUPS = [
    "rapport", "sectarian_drift_list", "literature", "interviews", "dictionary",
    "entities_all", "entities_literature", "entities_miviludes", "entities_interviews",
]

EPISTEMIC_STATUS_CORPORA = ("literature", "miviludes", "interviews")
# Below this many points, a centroid is too noisy to be worth its own plot
# (still reported as a "too few" skip in the log, not silently dropped) --
# matches analyze_typicality.py's own "thin subgroups always computed, n
# always visible" spirit but a plot specifically needs enough points to
# fit UMAP on (>=3) and to mean something visually.
MIN_EPISTEMIC_SUBGROUP_N = 3


def discover_epistemic_status_groups(points: list[dict]) -> list[str]:
    """"<corpus>__<status>" for every (corpus, status) combination actually
    present in the data, for corpus in EPISTEMIC_STATUS_CORPORA -- DYNAMICALLY
    DISCOVERED, never a hardcoded set of statuses (literature/miviludes/
    interviews each have different statuses actually populated; see
    build_shared_space.epistemic_status_subgroup_indices for the same
    discovery discipline in the shared_space_v3-based toolkit). Insertion
    order = first-encountered order in `points` (file order), so this is
    deterministic run to run without being alphabetical. Skips a
    (corpus, status) pair with fewer than MIN_EPISTEMIC_SUBGROUP_N points."""
    counts: dict[str, int] = {}
    for p in points:
        if p["source_dataset"] not in EPISTEMIC_STATUS_CORPORA:
            continue
        status = p.get("epistemic_status")
        if status is None:
            continue
        group = f"{p['source_dataset']}__{status}"
        counts[group] = counts.get(group, 0) + 1
    skipped = {g: n for g, n in counts.items() if n < MIN_EPISTEMIC_SUBGROUP_N}
    if skipped:
        logger.info("Skipping epistemic-status subgroups with < %d points: %s", MIN_EPISTEMIC_SUBGROUP_N, skipped)
    return [g for g, n in counts.items() if n >= MIN_EPISTEMIC_SUBGROUP_N]


def truncate_label(label: str, max_len: int = 40) -> str:
    label = " ".join(label.split())
    return label if len(label) <= max_len else label[: max_len - 1].rstrip() + "…"


LOCAL_CONTEXT_N = 80


def umap_2d_with_centroid(local_vectors: np.ndarray, centroid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fits UMAP on local_vectors AND the centroid TOGETHER, as one joint
    fit -- not member-only-fit-then-.transform()-the-centroid-afterward
    (the previous version's approach). Two reasons the previous plots
    still didn't look right even after switching PCA-slice -> UMAP:
    (1) a centroid is the MEAN of many unit vectors, so it is itself
    NOT unit-norm and can sit in a genuinely empty region between
    sub-clusters -- a real geometric fact (mean of a multimodal cloud
    isn't near its mode), not a rendering bug; (2) UMAP's out-of-sample
    .transform() for a single new point is a local graph-interpolation
    approximation, not a re-optimized fit, and adds its OWN placement
    error on top of (1). Including the centroid in the actual fit removes
    (2) entirely -- it participates in the same neighbour-graph
    optimization as every real point, rather than being placed after the
    fact. (1) can still leave the centroid genuinely apart from a tight
    top-k cluster if the group truly is multimodal; that's now a real
    finding the plot can surface, not a rendering artefact to fix."""
    n = len(local_vectors)
    n_neighbors = min(15, n)  # +1 for the centroid row already in the fit
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=UMAP_MIN_DIST,
                         random_state=UMAP_SEED, metric="cosine")
    combined = np.vstack([local_vectors, centroid.reshape(1, -1)])
    coords = reducer.fit_transform(combined)
    return coords[:-1], coords[-1]


def plot_group(out_dir: Path, group: str, points: list[dict], vectors: np.ndarray,
                indices: list[int], centroid: np.ndarray, neighbors: list[dict],
                local_context_n: int = LOCAL_CONTEXT_N) -> Path:
    """Zoomed-in local-neighbourhood plot: fits UMAP on the `local_context_n`
    points nearest the centroid (the WHOLE group if it's smaller than that),
    not the full group -- so the plot shows the neighbourhood the top-k
    actually live in, rather than burying them inside a much larger
    population's global layout (literature/dictionary/entities_all are
    thousands of points; a 2D fit spanning all of them makes any local
    tightness among just the top 10 illegible regardless of method)."""
    # Positional (argsort), never key-based: group_points' own "key"
    # (document_id:chunk_index) is NOT guaranteed unique -- v2 extraction
    # often pulls several expressions from one chunk (checked directly on
    # "rapport": 94 expressions span only 70 unique keys). An earlier
    # version matched local-context points back to their neighbour by key,
    # which silently collapsed/misassigned points whenever a key repeated
    # (94 -> 62 local points instead of the requested 80, a real bug, not
    # a smaller-than-expected group). Using the sort position itself
    # sidesteps key uniqueness entirely.
    group_points = [points[i] for i in indices]
    group_vectors = vectors[indices]
    dist = gac.euclidean_distances(centroid, group_vectors)
    order = np.argsort(dist)
    local_context_n = min(local_context_n, len(order))
    local_order = order[:local_context_n]
    local_vectors = group_vectors[local_order]
    k = len(neighbors)

    member_2d, centroid_2d = umap_2d_with_centroid(local_vectors, centroid)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(member_2d[:, 0], member_2d[:, 1], s=10, c="#cfcfcf", linewidths=0, zorder=1,
               label=f"{len(local_order)} nearest-to-centroid (local context, of {len(indices)} total)")

    # local_order is itself ascending-distance-sorted, so its first k
    # positions ARE the k nearest by construction -- no need to match
    # back against `neighbors` by key at all.
    for j in range(min(k, len(local_order))):
        label = group_points[local_order[j]]["label"]
        ax.scatter(*member_2d[j], s=45, c="#0072B2", zorder=2)
        ax.annotate(f"{j + 1}. {truncate_label(label)}", member_2d[j], fontsize=7, color="#023858",
                    xytext=(4, 4), textcoords="offset points")
    ax.scatter([], [], s=45, c="#0072B2", label=f"{k} nearest to centroid")  # legend entry only

    ax.scatter(*centroid_2d, s=160, c="#D55E00", marker="*", zorder=3, label="centroid (fit jointly, not transformed post-hoc)")
    ax.set_title(f"{group}: centroid + {k} nearest neighbours\n(UMAP-2D, cosine, fit on the {len(local_order)} nearest points + centroid)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    out_path = out_dir / f"centroid_{group}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


CRITERION_ENTITY_K = 5
CRITERION_LITERATURE_K = 10
CRITERION_INTERVIEWS_K = 10


def criterion_neighbors(points: list[dict], vectors: np.ndarray, entity_k: int, literature_k: int,
                         interviews_k: int) -> list[dict]:
    """Per-criterion (not per-group-centroid) nearest neighbours: each of
    the 17 sectarian_drift_list points is its OWN query vector -- not a
    centroid of a group -- ranked against three separate candidate pools
    (entities_all, literature, interviews), each pool's own points only."""
    criterion_idx = group_indices(points, "sectarian_drift_list")
    entity_idx = group_indices(points, "entities_all")
    literature_idx = group_indices(points, "literature")
    interviews_idx = group_indices(points, "interviews")
    entity_points, entity_vectors = [points[i] for i in entity_idx], vectors[entity_idx]
    literature_points, literature_vectors = [points[i] for i in literature_idx], vectors[literature_idx]
    interviews_points, interviews_vectors = [points[i] for i in interviews_idx], vectors[interviews_idx]

    rows = []
    for i in criterion_idx:
        query = vectors[i]
        criterion_key, criterion_label = points[i]["key"], points[i]["label"]
        for pool_name, pool_points, pool_vectors, k in (
            ("entities", entity_points, entity_vectors, entity_k),
            ("literature", literature_points, literature_vectors, literature_k),
            ("interviews", interviews_points, interviews_vectors, interviews_k),
        ):
            for n in gac.nearest_points(query, pool_points, pool_vectors, k=k):
                rows.append({
                    "criterion_key": criterion_key, "criterion_label": criterion_label,
                    "pool": pool_name, **n,
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lit-mivi-run-tag", default="20260910")
    parser.add_argument("--interviews-archive-dir", type=Path,
                         default=bss.PROCESSED_DIR / "interviews_full" / "interviews" / "run_20260912")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--criterion-entity-k", type=int, default=CRITERION_ENTITY_K)
    parser.add_argument("--criterion-literature-k", type=int, default=CRITERION_LITERATURE_K)
    parser.add_argument("--criterion-interviews-k", type=int, default=CRITERION_INTERVIEWS_K)
    parser.add_argument("--local-context-n", type=int, default=LOCAL_CONTEXT_N,
                         help="Plot only the N points nearest the centroid (the whole group if smaller) -- "
                              "a zoomed-in local neighbourhood, not the full group's global layout.")
    parser.add_argument("--out-dir", type=Path, default=bss.PROCESSED_DIR / "analysis_raw" / "corpus_centroids")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw_points = load_raw_points(args.lit_mivi_run_tag, args.interviews_archive_dir)
    vectors = np.array([p["vector"] for p in raw_points], dtype=np.float64)
    points = [{k: v for k, v in p.items() if k != "vector"} for p in raw_points]
    logger.info("Loaded %d raw points (no standardize/PCA)", len(points))
    norms = np.linalg.norm(vectors, axis=1)
    logger.info("vector norm sanity check: min=%.4f max=%.4f (expect ~1.0, raw bge-m3 is unit-norm)", norms.min(), norms.max())

    epistemic_groups = discover_epistemic_status_groups(points)
    logger.info("Epistemic-status subgroups (n>=%d): %s", MIN_EPISTEMIC_SUBGROUP_N, epistemic_groups)

    csv_rows = []
    for group in GROUPS + epistemic_groups:
        indices = group_indices(points, group)
        if not indices:
            raise SystemExit(f"Group {group!r} has zero points -- check group_indices().")
        stats = gac.centroid_and_dispersion_for_indices(vectors, indices)
        centroid = stats["centroid"]
        group_points = [points[i] for i in indices]
        group_vectors = vectors[indices]
        neighbors = gac.nearest_points(centroid, group_points, group_vectors, k=args.k)

        logger.info("%-22s n=%-6d nearest: %s", group, len(indices), truncate_label(neighbors[0]["label"]))
        for n in neighbors:
            csv_rows.append({"group": group, "group_n": len(indices), **n})

        plot_group(args.out_dir, group, points, vectors, indices, centroid, neighbors, args.local_context_n)

    csv_path = args.out_dir / "centroid_neighbors.csv"
    gac.write_csv(csv_path, csv_rows)

    criterion_rows = criterion_neighbors(points, vectors, args.criterion_entity_k, args.criterion_literature_k,
                                          args.criterion_interviews_k)
    criterion_csv_path = args.out_dir / "criterion_nearest_neighbors.csv"
    gac.write_csv(criterion_csv_path, criterion_rows)
    logger.info("Per-criterion neighbours: %d criteria x (%d entities + %d literature + %d interviews) = %d rows",
                len(group_indices(points, "sectarian_drift_list")), args.criterion_entity_k,
                args.criterion_literature_k, args.criterion_interviews_k, len(criterion_rows))

    print(f"Done. {csv_path}, {criterion_csv_path}, and one centroid_<group>.png per group, under {args.out_dir}")


if __name__ == "__main__":
    main()
