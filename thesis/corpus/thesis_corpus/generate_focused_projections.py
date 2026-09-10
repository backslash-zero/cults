"""Focused, small, readable 2-D projections (UMAP + PCA) for the shared
embedding space -- a companion to analyze_cluster_structure.py's few
whole-space-scale UMAP populations, not a replacement for them. Those
populations (overview, expression_sampled, equal_n_diagnostic,
with_interview_prototypes) are useful as an overview, but at 44,520/3,436
points they are too dense to read any specific comparison off of. This
module instead builds ~23 small, curated point-sets, one figure's worth of
content each:

  - `expression_global`: the three expression corpora (reduced-literature)
    with their centroids overlaid.
  - `reference_comparison_{literature,miviludes,interviews}` (x3): one
    expression corpus against all three reference vocabularies in full.
  - `prototype_focus`: the 25 reviewed interview prototypes + the 17
    MIVILUDES criteria + the nearest terms (to the prototype-layer
    centroid) from each reference vocabulary, with corpus centroids
    overlaid.
  - `criterion_neighbourhood_<criterion_key>` (x17): one MIVILUDES
     criterion + its nearest expression-corpus neighbours + its nearest
     reference-vocabulary terms.
  - `emergent_entity_focus`: the top entities by mention count + the 17
    criteria + each entity's own already-computed nearest
    structural-concept/conceptnet-concept term.

Read-only with respect to the pipeline and every other analysis module:
never modifies build_shared_space.py/visualize_3d.py/embedding_space.jsonl,
never touches another module's own processed/analysis/<run-id>/
subdirectory. Reuses analyze_criterion_neighbours.py's and
analyze_emergent_entities.py's saved output when present in the SAME
run-id (to avoid recomputing nearest-neighbour tables this module also
needs), falling back to a local computation against embedding_space.jsonl
otherwise -- either way, purely additive reads, never a write into another
module's directory.

Every population gets both a UMAP-2D fit and a PCA-2D projection (the
first 2 columns of the already-PCA'd 394-D vectors -- a slice, never a
fresh per-subset PCA refit, same convention visualize_3d.py documents for
its own 3-D PCA output -- so every PCA-2D population shares one fixed
global basis and stays comparable to every other). Both are produced via
geometric_analysis_common.fit_and_save_umap/save_pca_projection -- this
module never fits UMAP or slices PCA itself.

Terminology used in every manifest: `n_points_fit` (what UMAP/PCA actually
fits on) vs. `n_points_overlay` (centroids, transformed into the fitted
layout *after* fitting -- never part of the manifold fit) vs.
`n_points_rendered` (= fit + overlay, what the figure actually shows). A
`shortages` dict records, per population, any source pool that had fewer
usable candidates than requested when the population was constructed --
entirely separate from (and never conflated with) fit_and_save_umap's own
n_neighbors clamp, which only changes how UMAP's algorithm behaves
internally and never removes a point.

Usage (from thesis/corpus/):
    python -m thesis_corpus.generate_focused_projections --run-id <existing-run-id>
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from thesis_corpus import geometric_analysis_common as gac
from thesis_corpus.analyze_initial_exemplars import load_interview_prototypes

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("thesis_corpus.generate_focused_projections")

MODULE_NAME = "generate_focused_projections"
NEAREST_K_EXPRESSION = 10
NEAREST_K_REFERENCE_PROTOTYPE = 10
NEAREST_K_REFERENCE_CRITERION = 5
DEFAULT_TOP_ENTITIES = 50
UMAP_PARAM_GRID = [(10, 0.1), (20, 0.2), (50, 0.3)]


@dataclass
class FocusedPopulation:
    name: str
    member_points: list[dict]
    member_vectors: np.ndarray
    overlay_points: list[dict] = field(default_factory=list)
    overlay_vectors: np.ndarray | None = None
    shortages: dict = field(default_factory=dict)


def _tagged_point(point: dict, population_role: str) -> dict:
    """A shallow copy of a shared_space point (or a synthetic centroid
    point) with population_role set -- never mutates the caller's own
    point dict, since several populations reuse the same underlying
    points (e.g. the same criterion appears in prototype_focus AND its
    own criterion_neighbourhood_* population)."""
    tagged = dict(point)
    tagged["population_role"] = population_role
    return tagged


def _centroid_point(source_dataset: str, centroid: np.ndarray) -> dict:
    return {
        "key": f"centroid:{source_dataset}", "source_dataset": source_dataset,
        "point_role": "centroid", "label": f"{source_dataset} centroid",
        "population_role": f"centroid:{source_dataset}",
    }


def _nearest_indices(query: np.ndarray, candidate_vectors: np.ndarray, k: int) -> np.ndarray:
    distances = gac.euclidean_distances(query, candidate_vectors)
    return np.argsort(distances)[:k]


# ---------------------------------------------------------------------------
# Population builders
# ---------------------------------------------------------------------------

def build_expression_global(shared_space: gac.SharedSpace, per_source: dict) -> FocusedPopulation:
    by_corpus = gac.corpus_vectors_and_points(shared_space, "reduced_literature")
    member_points, member_vectors = [], []
    for corpus, (points, vectors) in by_corpus.items():
        member_points.extend(_tagged_point(p, corpus) for p in points)
        member_vectors.append(vectors)
    overlay_points = [_centroid_point(c, per_source[c]["centroid"]) for c in gac.EXPRESSION_CORPORA]
    overlay_vectors = np.array([per_source[c]["centroid"] for c in gac.EXPRESSION_CORPORA], dtype=np.float64)
    return FocusedPopulation(
        name="expression_global",
        member_points=member_points, member_vectors=np.concatenate(member_vectors, axis=0),
        overlay_points=overlay_points, overlay_vectors=overlay_vectors,
    )


def build_reference_comparison(
    shared_space: gac.SharedSpace, reference_pools: dict[str, tuple[list[dict], np.ndarray]], corpus: str,
) -> FocusedPopulation:
    if corpus == "literature":
        corpus_points, corpus_vectors = gac.corpus_vectors_and_points(shared_space, "reduced_literature")[corpus]
    else:
        idxs = gac.source_dataset_indices(shared_space.points, corpus)
        corpus_points = [shared_space.points[i] for i in idxs]
        corpus_vectors = shared_space.vectors[idxs]

    member_points = [_tagged_point(p, "own_corpus") for p in corpus_points]
    member_vectors = [corpus_vectors]
    for ref_name, (ref_points, ref_vectors) in reference_pools.items():
        member_points.extend(_tagged_point(p, f"reference:{ref_name}") for p in ref_points)
        member_vectors.append(ref_vectors)
    return FocusedPopulation(
        name=f"reference_comparison_{corpus}",
        member_points=member_points, member_vectors=np.concatenate(member_vectors, axis=0),
    )


def _normalize_prototype_point(item: dict) -> dict:
    """interview_prototypes.jsonl rows use document_id/source_expression_label,
    not the key/label fields every other point in this toolkit carries --
    same normalization analyze_cluster_structure.py's with_interview_prototypes
    population already applies, reused here for consistency (same
    "prototype:<document_id>" key format)."""
    return {
        "key": f"prototype:{item['document_id']}",
        "source_dataset": item["source_dataset"],
        "point_role": item["point_role"],
        "label": item["source_expression_label"],
    }


def build_prototype_focus(
    prototype_points: list[dict], prototype_vectors: np.ndarray,
    criteria_points: list[dict], criteria_vectors: np.ndarray,
    reference_pools: dict[str, tuple[list[dict], np.ndarray]], per_source: dict,
) -> FocusedPopulation:
    member_points = [_tagged_point(_normalize_prototype_point(p), "prototype") for p in prototype_points]
    member_vectors = [prototype_vectors]
    member_points.extend(_tagged_point(p, "criterion") for p in criteria_points)
    member_vectors.append(criteria_vectors)

    prototype_centroid = prototype_vectors.mean(axis=0)
    shortages: dict = {}
    for ref_name, (ref_points, ref_vectors) in reference_pools.items():
        n_available = len(ref_points)
        k = min(NEAREST_K_REFERENCE_PROTOTYPE, n_available)
        if n_available < NEAREST_K_REFERENCE_PROTOTYPE:
            shortages[f"reference:{ref_name}"] = {"requested": NEAREST_K_REFERENCE_PROTOTYPE, "available": n_available}
        nearest_idx = _nearest_indices(prototype_centroid, ref_vectors, k)
        member_points.extend(_tagged_point(ref_points[i], f"nearest_reference:{ref_name}") for i in nearest_idx)
        member_vectors.append(ref_vectors[nearest_idx])

    overlay_points = [_centroid_point(c, per_source[c]["centroid"]) for c in gac.EXPRESSION_CORPORA]
    overlay_vectors = np.array([per_source[c]["centroid"] for c in gac.EXPRESSION_CORPORA], dtype=np.float64)
    return FocusedPopulation(
        name="prototype_focus",
        member_points=member_points, member_vectors=np.concatenate(member_vectors, axis=0),
        overlay_points=overlay_points, overlay_vectors=overlay_vectors,
        shortages=shortages,
    )


def _read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_criterion_neighbourhood(
    points_by_key: dict[str, tuple[dict, np.ndarray]],
    criterion_point: dict, criterion_vector: np.ndarray,
    expression_pool: dict[str, tuple[list[dict], np.ndarray]],
    reference_pools: dict[str, tuple[list[dict], np.ndarray]],
    run_dir: Path,
    entity_points: list[dict] | None = None,
    entity_vectors: np.ndarray | None = None,
) -> FocusedPopulation:
    criterion_key = criterion_point["key"]
    member_points = [_tagged_point(criterion_point, "criterion_query")]
    member_vectors = [criterion_vector[np.newaxis, :]]
    shortages: dict = {}

    # Reuse analyze_criterion_neighbours.py's saved French-primary output
    # when present in this run (avoids recomputing an identical
    # nearest-neighbour search); otherwise compute locally against the
    # current shared space. `points_by_key` is built once by the caller
    # (main()) and shared across all 17 criteria -- rebuilding a
    # 44,520-entry dict per criterion here would be wasteful.
    qual_csv = run_dir / "analyze_criterion_neighbours" / "qualitative_retrieval_french_primary_shared_space.csv"
    qual_rows = [r for r in _read_csv_rows(qual_csv) if r.get("criterion_key") == criterion_key]

    for corpus, (corpus_points, corpus_vectors) in expression_pool.items():
        rows_for_corpus = sorted(
            (r for r in qual_rows if r.get("corpus") == corpus),
            key=lambda r: int(r["rank"]),
        )[:NEAREST_K_EXPRESSION]
        if rows_for_corpus:
            n_available = len(rows_for_corpus)
            for r in rows_for_corpus:
                p, v = points_by_key[r["neighbor_key"]]
                member_points.append(_tagged_point(p, f"nearest_expression:{corpus}"))
                member_vectors.append(v[np.newaxis, :])
        else:
            n_available = min(NEAREST_K_EXPRESSION, len(corpus_points))
            nearest_idx = _nearest_indices(criterion_vector, corpus_vectors, n_available)
            for i in nearest_idx:
                member_points.append(_tagged_point(corpus_points[i], f"nearest_expression:{corpus}"))
                member_vectors.append(corpus_vectors[i][np.newaxis, :])
        if n_available < NEAREST_K_EXPRESSION:
            shortages[f"nearest_expression:{corpus}"] = {"requested": NEAREST_K_EXPRESSION, "available": n_available}

    ref_raw_csv = run_dir / "analyze_criterion_neighbours" / "nearest_reference_terms_raw_french_primary_shared_space.csv"
    ref_rows = [r for r in _read_csv_rows(ref_raw_csv) if r.get("criterion_key") == criterion_key]
    for ref_name, (ref_points, ref_vectors) in reference_pools.items():
        rows_for_ref = sorted(
            (r for r in ref_rows if r.get("reference_dataset") == ref_name),
            key=lambda r: int(r["rank"]),
        )[:NEAREST_K_REFERENCE_CRITERION]
        if rows_for_ref:
            n_available = len(rows_for_ref)
            ref_points_by_key = {p["key"]: (p, v) for p, v in zip(ref_points, ref_vectors)}
            for r in rows_for_ref:
                p, v = ref_points_by_key[r["neighbor_key"]]
                member_points.append(_tagged_point(p, f"nearest_reference:{ref_name}"))
                member_vectors.append(v[np.newaxis, :])
        else:
            n_available = min(NEAREST_K_REFERENCE_CRITERION, len(ref_points))
            nearest_idx = _nearest_indices(criterion_vector, ref_vectors, n_available)
            for i in nearest_idx:
                member_points.append(_tagged_point(ref_points[i], f"nearest_reference:{ref_name}"))
                member_vectors.append(ref_vectors[i][np.newaxis, :])
        if n_available < NEAREST_K_REFERENCE_CRITERION:
            shortages[f"nearest_reference:{ref_name}"] = {"requested": NEAREST_K_REFERENCE_CRITERION, "available": n_available}

    # Nearest EMERGENT ENTITIES -- reuses analyze_criterion_neighbours.py's
    # saved qualitative_retrieval_entities.csv (added alongside the
    # expression retrieval above) when present; falls back to a local
    # nearest-neighbour search the same way the expression/reference
    # blocks above do, so this mini-map still renders even if that CSV
    # is missing for some older run.
    if entity_points is not None and entity_vectors is not None and len(entity_points):
        entity_csv = run_dir / "analyze_criterion_neighbours" / "qualitative_retrieval_entities.csv"
        entity_rows = [r for r in _read_csv_rows(entity_csv) if r.get("criterion_key") == criterion_key]
        entity_points_by_key = {p["key"]: (p, v) for p, v in zip(entity_points, entity_vectors)}
        rows_for_entities = sorted(entity_rows, key=lambda r: int(r["rank"]))[:NEAREST_K_REFERENCE_CRITERION] if entity_rows else []
        if rows_for_entities:
            n_available = len(rows_for_entities)
            for r in rows_for_entities:
                p, v = entity_points_by_key[r["neighbor_key"]]
                member_points.append(_tagged_point(p, "nearest_entity"))
                member_vectors.append(v[np.newaxis, :])
        else:
            n_available = min(NEAREST_K_REFERENCE_CRITERION, len(entity_points))
            nearest_idx = _nearest_indices(criterion_vector, entity_vectors, n_available)
            for i in nearest_idx:
                member_points.append(_tagged_point(entity_points[i], "nearest_entity"))
                member_vectors.append(entity_vectors[i][np.newaxis, :])
        if n_available < NEAREST_K_REFERENCE_CRITERION:
            shortages["nearest_entity"] = {"requested": NEAREST_K_REFERENCE_CRITERION, "available": n_available}

    return FocusedPopulation(
        name=f"criterion_neighbourhood_{criterion_key}",
        member_points=member_points, member_vectors=np.concatenate(member_vectors, axis=0),
        shortages=shortages,
    )


def build_emergent_entity_focus(
    shared_space: gac.SharedSpace, run_dir: Path, criteria_points: list[dict], criteria_vectors: np.ndarray,
    top_n: int,
) -> FocusedPopulation | None:
    entities_csv = run_dir / "analyze_emergent_entities" / "emergent_entities_full.csv"
    rows = _read_csv_rows(entities_csv)
    if not rows:
        logger.warning(
            "%s not found -- skipping emergent_entity_focus (run analyze_emergent_entities first if you want it).",
            entities_csv,
        )
        return None

    entity_idxs = gac.source_dataset_indices(shared_space.points, "emergent_entities")
    entity_vector_by_label = {shared_space.points[i]["label"]: shared_space.vectors[i] for i in entity_idxs}
    entity_point_by_label = {shared_space.points[i]["label"]: shared_space.points[i] for i in entity_idxs}

    top_rows = sorted(rows, key=lambda r: int(r["rank"]))[:top_n]
    member_points, member_vectors = [], []
    for r in top_rows:
        label = r["entity"]
        if label not in entity_vector_by_label:
            continue
        member_points.append(_tagged_point(entity_point_by_label[label], "entity"))
        member_vectors.append(entity_vector_by_label[label])

    member_points.extend(_tagged_point(p, "criterion") for p in criteria_points)
    member_vectors.extend(list(criteria_vectors))

    # Each entity's own already-computed nearest structural_concept /
    # conceptnet_concepts term, deduped by key -- never recomputed here.
    seen_ref_keys: set[tuple[str, str]] = set()
    for ref_name in ("structural_concepts", "conceptnet_concepts"):
        key_col, label_col = f"nearest_{ref_name}_key", f"nearest_{ref_name}_label"
        if key_col not in (top_rows[0].keys() if top_rows else []):
            continue
        idxs = gac.source_dataset_indices(shared_space.points, ref_name)
        ref_point_by_key = {shared_space.points[i]["key"]: (shared_space.points[i], shared_space.vectors[i]) for i in idxs}
        for r in top_rows:
            ref_key = r.get(key_col)
            if not ref_key or (ref_name, ref_key) in seen_ref_keys or ref_key not in ref_point_by_key:
                continue
            seen_ref_keys.add((ref_name, ref_key))
            p, v = ref_point_by_key[ref_key]
            member_points.append(_tagged_point(p, f"nearest_reference:{ref_name}"))
            member_vectors.append(v)

    return FocusedPopulation(
        name="emergent_entity_focus",
        member_points=member_points, member_vectors=np.array(member_vectors, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Emit (UMAP + PCA) for one population, across the requested param grid
# ---------------------------------------------------------------------------

def emit_population(
    out_dir: Path, pop: FocusedPopulation, param_combinations: list[tuple[int, float]], seed: int,
) -> dict:
    for n_neighbors, min_dist in param_combinations:
        gac.fit_and_save_umap(
            out_dir, pop.name, pop.member_points, pop.member_vectors,
            n_neighbors, min_dist, seed,
            overlay_points=pop.overlay_points, overlay_vectors=pop.overlay_vectors,
            shortages=pop.shortages,
        )
    gac.save_pca_projection(
        out_dir, pop.name, pop.member_points, pop.member_vectors,
        overlay_points=pop.overlay_points, overlay_vectors=pop.overlay_vectors,
        shortages=pop.shortages,
    )
    return {
        "n_points_fit": len(pop.member_points),
        "n_points_overlay": len(pop.overlay_points),
        "n_points_rendered": len(pop.member_points) + len(pop.overlay_points),
        "shortages": pop.shortages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", type=str, required=True,
                         help="An existing run-id (this module reads other modules' output from it when present).")
    parser.add_argument("--shared-space-dir", type=Path, default=None,
                         help="Use a different pooled space (e.g. processed/shared_space_v2/) instead of v1's "
                              "processed/shared_space/; also switches the run-output root to a sibling "
                              "processed/analysis_v2/ directory so v1 and v2 runs are never mixed.")
    parser.add_argument("--seed", type=int, default=gac.DEFAULT_SEED)
    parser.add_argument("--umap-neighbors", type=int, default=20)
    parser.add_argument("--umap-min-dist", type=float, default=0.2)
    parser.add_argument("--umap-param-grid", action="store_true",
                         help="Additionally fit (10,0.1), (20,0.2), (50,0.3) for each population.")
    parser.add_argument("--top-entities", type=int, default=DEFAULT_TOP_ENTITIES)
    args = parser.parse_args()

    embedding_space_path, interview_prototypes_path, analysis_root = gac.resolve_space_paths(args.shared_space_dir)
    logger.info("Loading %s ...", embedding_space_path)
    shared_space = gac.load_shared_space(embedding_space_path)
    logger.info("Loaded %d points, %d-d vectors", len(shared_space.points), shared_space.vectors.shape[1])

    run_dir = gac.existing_run_dir(args.run_id, analysis_root)
    gac.init_run_manifest(run_dir, shared_space, defaults={"seed": args.seed})
    gac.update_run_manifest(run_dir, MODULE_NAME, "running")
    out_dir = gac.module_run_dir(run_dir, MODULE_NAME)

    param_combinations = [(args.umap_neighbors, args.umap_min_dist)]
    if args.umap_param_grid:
        param_combinations = list(dict.fromkeys(param_combinations + UMAP_PARAM_GRID))

    try:
        per_source = gac.per_source_centroids_and_dispersion(shared_space)
        reference_pools = {
            ref_dataset: (
                [shared_space.points[i] for i in gac.source_dataset_indices(shared_space.points, ref_dataset)],
                shared_space.vectors[gac.source_dataset_indices(shared_space.points, ref_dataset)],
            )
            for ref_dataset in gac.REFERENCE_VOCAB_DATASETS
        }
        criteria_idxs = gac.source_dataset_indices(shared_space.points, "miviludes_criteria")
        criteria_points = [shared_space.points[i] for i in criteria_idxs]
        criteria_vectors = shared_space.vectors[criteria_idxs]
        expression_pool = gac.corpus_vectors_and_points(shared_space, "full")
        entity_idxs = gac.source_dataset_indices(shared_space.points, "emergent_entities")
        entity_points_all = [shared_space.points[i] for i in entity_idxs]
        entity_vectors_all = shared_space.vectors[entity_idxs]

        populations_index: dict[str, dict] = {}

        logger.info("Population 1/5: expression_global ...")
        pop = build_expression_global(shared_space, per_source)
        populations_index[pop.name] = emit_population(out_dir, pop, param_combinations, args.seed)

        logger.info("Population 2/5: reference_comparison_{literature,miviludes,interviews} ...")
        for corpus in gac.EXPRESSION_CORPORA:
            pop = build_reference_comparison(shared_space, reference_pools, corpus)
            populations_index[pop.name] = emit_population(out_dir, pop, param_combinations, args.seed)

        logger.info("Population 3/5: prototype_focus ...")
        if interview_prototypes_path.exists():
            prototype_points, prototype_vectors = load_interview_prototypes(interview_prototypes_path)
            pop = build_prototype_focus(prototype_points, prototype_vectors, criteria_points, criteria_vectors, reference_pools, per_source)
            populations_index[pop.name] = emit_population(out_dir, pop, param_combinations, args.seed)
        else:
            logger.warning(
                "%s not found -- skipping prototype_focus (run build_interview_prototype_layer.py first if you want it).",
                interview_prototypes_path,
            )

        logger.info("Population 4/5: criterion_neighbourhood_<criterion_key> (x%d) ...", len(criteria_points))
        points_by_key = {p["key"]: (p, v) for p, v in zip(shared_space.points, shared_space.vectors)}
        for c_point, c_vector in zip(criteria_points, criteria_vectors):
            pop = build_criterion_neighbourhood(
                points_by_key, c_point, c_vector, expression_pool, reference_pools, run_dir,
                entity_points=entity_points_all, entity_vectors=entity_vectors_all,
            )
            populations_index[pop.name] = emit_population(out_dir, pop, param_combinations, args.seed)

        logger.info("Population 5/5: emergent_entity_focus (top %d) ...", args.top_entities)
        pop = build_emergent_entity_focus(shared_space, run_dir, criteria_points, criteria_vectors, args.top_entities)
        if pop is not None:
            populations_index[pop.name] = emit_population(out_dir, pop, param_combinations, args.seed)

        (out_dir / "populations_index.json").write_text(
            json.dumps(populations_index, indent=2), encoding="utf-8",
        )

        gac.write_module_config(
            out_dir,
            input_sha256=shared_space.input_sha256,
            seed=args.seed,
            umap_param_combinations=param_combinations,
            top_entities=args.top_entities,
            n_populations=len(populations_index),
            reference_vocab_datasets=gac.REFERENCE_VOCAB_DATASETS,
            git_commit=gac.git_commit_hash(),
        )
        gac.update_run_manifest(run_dir, MODULE_NAME, "completed", output_dir=out_dir)
    except Exception as e:
        gac.update_run_manifest(run_dir, MODULE_NAME, "failed", error=str(e))
        raise

    print(f"\nDone. Run {args.run_id} -> {out_dir} ({len(populations_index)} populations)")


if __name__ == "__main__":
    main()
