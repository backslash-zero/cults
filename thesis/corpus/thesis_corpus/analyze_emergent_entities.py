"""Provenance and geometric proximity of emergent entities (named
entities/groups/concepts the corpora themselves mention, mentioned >=3
times overall -- see Methods.tex, "Emergent Entity Anchors").

Reuses export_emergent_entities.rank_emergent_entities() for the ranking
itself rather than re-deriving it. Two things this module adds on top:

  1. Provenance category per entity, from mention_distribution: which of
     literature/MIVILUDES/interviews mention it, collapsed into
     literature_only / miviludes_only / interviews_only / shared_2_corpora /
     shared_all_3. Reports counts per category -- a term appearing in only
     one epistemology's vocabulary vs. one that recurs across all three is
     a structurally different kind of thing.
  2. Nearest MIVILUDES criterion and nearest term from each of the 3
     reference vocabularies (structural_concepts, concept_backbone,
     conceptnet_concepts) for every entity, in the shared 394-D space
     (Euclidean primary, cosine sensitivity column) -- plus (new) an
     equal-size-controlled reference comparison across all ranked
     entities at once, since the raw per-entity nearest-term columns are
     biased toward whichever reference set is largest by pool size alone
     (see geometric_analysis_common.equal_size_reference_comparison).

Explicitly does NOT classify entities as cult/mainstream or apply any
other normative label -- only geometric proximity and provenance category,
both purely descriptive.

  3. Shared discourse anchors (new): qualitative nearest-expression
     retrieval for every entity with provenance_category=="shared_all_3"
     (mentioned by all three corpora -- derived dynamically from the
     ranking above every run, never a hardcoded count or list). Two
     retrieval scopes, never merged:
       - PRIMARY: 5 nearest expressions within each of
         literature/MIVILUDES/interviews SEPARATELY, using each source's
         full available pool (not equal-n-sampled) -- the most direct
         answer to "what language surrounds this shared anchor within
         each source." This is what any later write-up should rely on.
       - EXPLORATORY/APPENDIX-ONLY: one deterministic equal-n draw (fixed
         seed, not bootstrapped -- averaging literal retrieved text across
         draws doesn't mean anything), giving an equal-sized combined
         candidate pool across the 3 corpora, top-10 within that single
         draw. Never relied on in a main-text narrative.
     "Shared-across-corpora terms are discourse anchors defined by
     co-mention provenance. They may be category labels, social roles,
     groups, movements, or other concepts; shared occurrence is not
     evidence that a term is a prototypical cult or that all sources
     evaluate it similarly." This module does not select which anchors
     belong in any thesis narrative -- that's a human judgment call made
     by reading shared_anchor_nearest_expressions.csv, not something
     computed here. `anchor_role` is left blank in the output for the
     same reason (manual coding, not automated).

Usage (from thesis/corpus/):
    python -m thesis_corpus.analyze_emergent_entities
"""
from __future__ import annotations

import argparse
import logging

import numpy as np

from thesis_corpus import export_emergent_entities as eee
from thesis_corpus import geometric_analysis_common as gac

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("thesis_corpus.analyze_emergent_entities")

MODULE_NAME = "analyze_emergent_entities"
ANCHOR_WITHIN_SOURCE_K = 5
ANCHOR_OVERALL_EQUAL_N_K = 10
ANCHOR_SEMANTIC_WARNING = (
    "Shared-across-corpora terms are discourse anchors defined by co-mention "
    "provenance. They may be category labels, social roles, groups, movements, "
    "or other concepts; shared occurrence is not evidence that a term is a "
    "prototypical cult or that all sources evaluate it similarly."
)


def provenance_category(mentions: dict[str, int]) -> str:
    present = sorted(c for c in gac.EXPRESSION_CORPORA if mentions.get(c, 0) > 0)
    if len(present) == 3:
        return "shared_all_3"
    if len(present) == 2:
        return "shared_2_corpora"
    if len(present) == 1:
        return f"{present[0]}_only"
    return "no_expression_corpus_mentions"


def shared_discourse_anchors(entity_rows: list[dict]) -> list[dict]:
    """provenance_category=='shared_all_3' rows from entity_rows -- derived
    fresh every run from the current ranking, never a hardcoded count or
    label list (the count is currently 9, but nothing here assumes that
    stays true after a future corpus/extraction change). Fails loudly on
    an empty result rather than silently producing a 0-anchor deliverable."""
    anchors = [r for r in entity_rows if r["provenance_category"] == "shared_all_3"]
    if not anchors:
        raise SystemExit(
            "shared_discourse_anchors: 0 entities with provenance_category=='shared_all_3' -- "
            "refusing to proceed with an empty shared-anchor set. Investigate before rerunning."
        )
    return anchors


def _nearest_with_global_indices(
    query: np.ndarray, candidate_points: list[dict], candidate_vectors: np.ndarray,
    candidate_global_indices: list[int], k: int,
) -> list[dict]:
    """Like gac.nearest_points, but each result row also carries
    `global_index` -- the candidate's position in `candidate_global_indices`
    (typically indices into shared_space.points), not just its local
    position within `candidate_points`/`candidate_vectors`. Needed because
    nearest_points' own `key` field is not guaranteed globally unique (see
    gac.derive_pooled_expression_keys), so resolving a specific retrieved
    row's context_window later (via gac.resolve_context_windows, keyed by
    global point index) can't be done from `key` alone. Only the
    exclude_self=False path is needed here (an emergent-entity anchor's
    own point is never itself a candidate in any of these pools), so this
    is exactly gac.nearest_points' own argsort-and-slice, replicated
    locally to also capture indices -- gac.nearest_points itself is left
    untouched, no risk to its already-validated callers."""
    rows = gac.nearest_points(query, candidate_points, candidate_vectors, k)
    euclidean = gac.euclidean_distances(query, candidate_vectors)
    order = list(np.argsort(euclidean)[:k])
    global_indices = [candidate_global_indices[i] for i in order]
    assert len(rows) == len(global_indices), (
        "_nearest_with_global_indices: nearest_points row count doesn't match the "
        "replicated argsort's index count -- ranking mismatch, investigate rather than zip."
    )
    for row, gi in zip(rows, global_indices):
        row["global_index"] = gi
    return rows


def anchor_nearest_expressions(
    shared_space: gac.SharedSpace, anchors: list[dict],
    entity_points: dict[str, tuple[int, dict]],
    context_resolutions: dict[str, gac.ContextWindowResolution],
    seed: int,
) -> list[dict]:
    """Builds shared_anchor_nearest_expressions.csv's rows for every
    anchor, both retrieval scopes (see module docstring). Hard-fails (per
    the same fail-loud rule as analyze_global_structure.py's Deliverable
    4) if a SELECTED row has no resolved context_window.
    """
    # Primary scope: each expression corpus's FULL available pool (not
    # equal-n-sampled) -- searched once, reused for every anchor.
    full_pool: dict[str, tuple[list[dict], np.ndarray, list[int]]] = {}
    for corpus in gac.EXPRESSION_CORPORA:
        idxs = gac.source_dataset_indices(shared_space.points, corpus)
        full_pool[corpus] = ([shared_space.points[i] for i in idxs], shared_space.vectors[idxs], idxs)

    # Exploratory scope: ONE deterministic equal-n draw, shared across
    # every anchor's overall_equal_n retrieval (not redrawn per anchor) --
    # a single, fixed-seed combined candidate pool.
    expr_pool = gac.expression_pool_indices(shared_space.points)
    equal_n_draw = gac.equal_n_expression_draw(shared_space.points, expr_pool, seed)
    equal_n_points, equal_n_global_indices = [], []
    equal_n_vector_parts = []
    for corpus in gac.EXPRESSION_CORPORA:
        idxs = equal_n_draw[corpus]
        equal_n_points.extend(shared_space.points[i] for i in idxs)
        equal_n_global_indices.extend(idxs)
        equal_n_vector_parts.append(shared_space.vectors[idxs])
    equal_n_vectors = np.concatenate(equal_n_vector_parts, axis=0)
    equal_n_candidate_size = len(equal_n_global_indices)

    rows = []
    for anchor in anchors:
        anchor_text = anchor["entity"]
        anchor_index, _anchor_point = entity_points[anchor_text]
        anchor_vector = shared_space.vectors[anchor_index]

        scoped_results: list[tuple[str, list[dict]]] = []
        for corpus in gac.EXPRESSION_CORPORA:
            cand_points, cand_vectors, cand_global_indices = full_pool[corpus]
            scoped_results.append((
                corpus,
                _nearest_with_global_indices(anchor_vector, cand_points, cand_vectors, cand_global_indices, ANCHOR_WITHIN_SOURCE_K),
            ))
        scoped_results.append((
            "overall_equal_n",
            _nearest_with_global_indices(anchor_vector, equal_n_points, equal_n_vectors, equal_n_global_indices, ANCHOR_OVERALL_EQUAL_N_K),
        ))

        for retrieval_scope, retrieved in scoped_results:
            for r in retrieved:
                global_idx = r["global_index"]
                point = shared_space.points[global_idx]
                source_dataset = point["source_dataset"]
                resolution = context_resolutions[source_dataset]
                pooled_key = point["key"]
                document_id = gac.key_document_id(pooled_key)
                chunk_index = gac.key_chunk_index(pooled_key)
                occurrence_key = resolution.occurrence_key_by_pooled_index.get(global_idx)
                context_window = resolution.context_window_by_pooled_index.get(global_idx)

                if context_window is None or context_window == "":
                    raise SystemExit(
                        f"shared_anchor_nearest_expressions: missing context_window for a "
                        f"SELECTED row -- anchor={anchor_text!r}, retrieval_scope={retrieval_scope!r}, "
                        f"source_dataset={source_dataset!r}, pooled_key={pooled_key!r}, "
                        f"occurrence_key={occurrence_key!r}. Hard-fail per the fail-loud "
                        "context-window resolution rule -- investigate this source's occurrence-aware join."
                    )

                rows.append({
                    "anchor_text": anchor_text, "anchor_role": "",
                    "retrieval_scope": retrieval_scope, "source_dataset": source_dataset,
                    "rank": r["rank"], "document_id": document_id, "chunk_index": chunk_index,
                    "pooled_key": pooled_key, "occurrence_key": occurrence_key,
                    "embedding_text": point["label"], "context_window": context_window,
                    "attribution": point.get("attribution"), "claim_mode": point.get("claim_mode"),
                    "epistemic_status": point.get("epistemic_status"),
                    "euclidean_distance": r["euclidean_distance"], "cosine_similarity": r["cosine_similarity"],
                })
    return rows, equal_n_candidate_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=gac.DEFAULT_SEED)
    parser.add_argument("--bootstrap-reps", type=int, default=gac.DEFAULT_BOOTSTRAP_REPS)
    args = parser.parse_args()

    logger.info("Loading %s ...", gac.EMBEDDING_SPACE_PATH)
    shared_space = gac.load_shared_space()

    run_id, run_dir = gac.get_or_create_run_dir(args.run_id)
    gac.init_run_manifest(run_dir, shared_space, defaults={})
    gac.update_run_manifest(run_dir, MODULE_NAME, "running")
    out_dir = gac.module_run_dir(run_dir, MODULE_NAME)

    try:
        logger.info("Ranking emergent entities (export_emergent_entities.rank_emergent_entities)...")
        ranked = eee.rank_emergent_entities(gac.EMBEDDING_SPACE_PATH)

        entity_idxs = gac.source_dataset_indices(shared_space.points, "emergent_entities")
        entity_points = {shared_space.points[i]["label"]: (i, shared_space.points[i]) for i in entity_idxs}

        criteria_idxs = gac.source_dataset_indices(shared_space.points, "miviludes_criteria")
        criteria_points = [shared_space.points[i] for i in criteria_idxs]
        criteria_vectors = shared_space.vectors[criteria_idxs]

        structural_idxs = gac.source_dataset_indices(shared_space.points, "structural_concepts")
        structural_points = [shared_space.points[i] for i in structural_idxs]
        structural_vectors = shared_space.vectors[structural_idxs]

        # New (Part 2/4): concept_backbone and conceptnet_concepts nearest-term
        # columns -- structural_concepts already had one before this session's
        # ConceptNet addition; the other two reference vocabularies had none.
        extra_reference_pools: dict[str, tuple[list[dict], np.ndarray]] = {}
        for ref_name in ("concept_backbone", "conceptnet_concepts"):
            idxs = gac.source_dataset_indices(shared_space.points, ref_name)
            extra_reference_pools[ref_name] = (
                [shared_space.points[i] for i in idxs], shared_space.vectors[idxs],
            )

        provenance_counts: dict[str, int] = {}
        entity_rows = []
        for rank, entry in enumerate(ranked, 1):
            label = entry["label"]
            if label not in entity_points:
                continue
            index, point = entity_points[label]
            vector = shared_space.vectors[index]

            category = provenance_category({
                "literature": entry["literature"], "miviludes": entry["miviludes"], "interviews": entry["interviews"],
            })
            provenance_counts[category] = provenance_counts.get(category, 0) + 1

            crit_euclidean = gac.euclidean_distances(vector, criteria_vectors)
            crit_cosine = gac.cosine_similarities(vector, criteria_vectors)
            nearest_crit_i = int(np.argmin(crit_euclidean))

            struct_euclidean = gac.euclidean_distances(vector, structural_vectors)
            struct_cosine = gac.cosine_similarities(vector, structural_vectors)
            nearest_struct_i = int(np.argmin(struct_euclidean))

            row = {
                "rank": rank, "entity": label, "total_mentions": entry["total"],
                "literature_mentions": entry["literature"], "miviludes_mentions": entry["miviludes"],
                "interviews_mentions": entry["interviews"], "provenance_category": category,
                "nearest_criterion_key": criteria_points[nearest_crit_i]["key"],
                "nearest_criterion_label": criteria_points[nearest_crit_i].get("label_en") or criteria_points[nearest_crit_i]["label"],
                "nearest_criterion_euclidean_distance": float(crit_euclidean[nearest_crit_i]),
                "nearest_criterion_cosine_similarity": float(crit_cosine[nearest_crit_i]),
                "nearest_structural_concept_key": structural_points[nearest_struct_i]["key"],
                "nearest_structural_concept_label": structural_points[nearest_struct_i]["label"],
                "nearest_structural_concept_euclidean_distance": float(struct_euclidean[nearest_struct_i]),
                "nearest_structural_concept_cosine_similarity": float(struct_cosine[nearest_struct_i]),
            }
            for ref_name, (ref_points, ref_vectors) in extra_reference_pools.items():
                ref_euclidean = gac.euclidean_distances(vector, ref_vectors)
                ref_cosine = gac.cosine_similarities(vector, ref_vectors)
                nearest_ref_i = int(np.argmin(ref_euclidean))
                row[f"nearest_{ref_name}_key"] = ref_points[nearest_ref_i]["key"]
                row[f"nearest_{ref_name}_label"] = ref_points[nearest_ref_i]["label"]
                row[f"nearest_{ref_name}_euclidean_distance"] = float(ref_euclidean[nearest_ref_i])
                row[f"nearest_{ref_name}_cosine_similarity"] = float(ref_cosine[nearest_ref_i])
            entity_rows.append(row)

        gac.write_csv(out_dir / "emergent_entities_full.csv", entity_rows)

        # --- New: shared discourse anchors (qualitative retrieval) ---
        anchors = shared_discourse_anchors(entity_rows)
        logger.info(
            "Shared discourse anchors: %d entities with provenance_category=='shared_all_3': %s",
            len(anchors), [a["entity"] for a in anchors],
        )
        context_resolutions = {corpus: gac.resolve_context_windows(shared_space, corpus) for corpus in gac.EXPRESSION_CORPORA}
        for corpus, resolution in context_resolutions.items():
            logger.info("resolve_context_windows(%s) stats: %s", corpus, resolution.stats)

        anchor_rows, anchor_equal_n_candidate_size = anchor_nearest_expressions(
            shared_space, anchors, entity_points, context_resolutions, seed=args.seed,
        )
        gac.write_csv(out_dir / "shared_anchor_nearest_expressions.csv", anchor_rows)
        (out_dir / "shared_anchor_nearest_expressions.README.txt").write_text(
            ANCHOR_SEMANTIC_WARNING + "\n\n"
            f"anchor_count: {len(anchors)}\n"
            f"anchors: {[a['entity'] for a in anchors]}\n"
            "retrieval_scopes:\n"
            f"  literature/miviludes/interviews: each source's FULL available expression pool "
            f"(not equal-n-sampled), top-{ANCHOR_WITHIN_SOURCE_K} per anchor -- PRIMARY output, "
            "what any later write-up should rely on.\n"
            f"  overall_equal_n: ONE deterministic equal-n draw (seed={args.seed}, "
            f"candidate_pool_size={anchor_equal_n_candidate_size}), top-{ANCHOR_OVERALL_EQUAL_N_K} "
            "per anchor -- EXPLORATORY/APPENDIX-ONLY, not relied on in the main narrative.\n"
            "rank ordering: Euclidean distance in the shared reduced space. cosine_similarity is "
            "a secondary descriptive field only, never used to select or order neighbours.\n"
            "anchor_role is left blank -- manual coding, not automated.\n",
            encoding="utf-8",
        )

        # --- Equal-size-controlled reference comparison (Part 4) ---
        # The per-entity nearest-term columns above are raw retrieval,
        # biased toward whichever reference set is largest (3,000/1,500/195
        # candidates) by pool size alone. Batches all ranked entities into
        # one gac.equal_size_reference_comparison call (one shared
        # per-repetition resample across every entity, not resampled per
        # entity -- vectorized and cheap even at ~3,251 entities).
        logger.info(
            "Equal-size reference comparison (n_reference=min of the 3 reference-set "
            "sizes, B=%d reps, batched over %d entities)...", args.bootstrap_reps, len(entity_rows),
        )
        reference_vectors_by_dataset = {"structural_concepts": structural_vectors, **{
            ref_name: vecs for ref_name, (_pts, vecs) in extra_reference_pools.items()
        }}
        entity_queries = {row["entity"]: shared_space.vectors[entity_points[row["entity"]][0]] for row in entity_rows}
        equal_size_result = gac.equal_size_reference_comparison(
            entity_queries, reference_vectors_by_dataset, k=10, seed=args.seed, reps=args.bootstrap_reps,
        )
        entity_rank_by_label = {row["entity"]: row["rank"] for row in entity_rows}
        reference_comparison_equal_size_rows = []
        for entity_label, per_reference in equal_size_result.items():
            for reference_dataset, stats in per_reference.items():
                for distance_statistic in ("mean", "median"):
                    summary = stats[f"nearest_k_distance_{distance_statistic}"]
                    reference_comparison_equal_size_rows.append({
                        "query_key": entity_label, "query_label": entity_label, "query_type": "entity",
                        "entity": entity_label, "rank": entity_rank_by_label[entity_label],
                        "reference_dataset": reference_dataset,
                        "n_reference": stats["n_reference"], "k": stats["k"],
                        "repetitions": args.bootstrap_reps, "seed": args.seed,
                        "distance_statistic": distance_statistic, **summary,
                    })
        gac.write_csv(out_dir / "reference_comparison_equal_size.csv", reference_comparison_equal_size_rows)

        provenance_rows = [{"provenance_category": k, "n_entities": v} for k, v in sorted(provenance_counts.items())]
        gac.write_csv(out_dir / "provenance_category_counts.csv", provenance_rows)

        (out_dir / "SCOPE_NOTICE.txt").write_text(
            "This analysis reports geometric proximity (nearest MIVILUDES criterion / "
            "nearest structural concept) and mention provenance (which corpora mention "
            "each entity) only. It does not classify, and must not be read as classifying, "
            "any entity as a 'cult' or 'mainstream' -- proximity to a criterion is not "
            "membership in a category.\n",
            encoding="utf-8",
        )

        gac.write_module_config(
            out_dir,
            input_sha256=shared_space.input_sha256,
            n_entities=len(entity_rows),
            provenance_counts=provenance_counts,
            metric="euclidean_primary_cosine_sensitivity",
            seed=args.seed,
            bootstrap_reps=args.bootstrap_reps,
            reference_vocab_datasets=gac.REFERENCE_VOCAB_DATASETS,
            shared_discourse_anchors={
                "anchor_count": len(anchors),
                "anchors": [a["entity"] for a in anchors],
                "retrieval_scopes": {
                    "literature": {"candidate_pool": "full_available_pool", "k": ANCHOR_WITHIN_SOURCE_K},
                    "miviludes": {"candidate_pool": "full_available_pool", "k": ANCHOR_WITHIN_SOURCE_K},
                    "interviews": {"candidate_pool": "full_available_pool", "k": ANCHOR_WITHIN_SOURCE_K},
                    "overall_equal_n": {
                        "candidate_pool": "single_deterministic_equal_n_draw",
                        "candidate_pool_size": anchor_equal_n_candidate_size,
                        "seed": args.seed, "k": ANCHOR_OVERALL_EQUAL_N_K,
                        "status": "exploratory_appendix_only",
                    },
                },
                "rank_ordering_metric": "euclidean",
                "cosine_similarity_role": "secondary_descriptive_only_not_used_for_selection",
            },
            context_window_resolution_stats={c: r.stats for c, r in context_resolutions.items()},
            git_commit=gac.git_commit_hash(),
        )
        gac.update_run_manifest(run_dir, MODULE_NAME, "completed", output_dir=out_dir)
    except Exception as e:
        gac.update_run_manifest(run_dir, MODULE_NAME, "failed", error=str(e))
        raise

    print(f"\nDone. Run {run_id} -> {out_dir}")


if __name__ == "__main__":
    main()
