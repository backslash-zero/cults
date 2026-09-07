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


def provenance_category(mentions: dict[str, int]) -> str:
    present = sorted(c for c in gac.EXPRESSION_CORPORA if mentions.get(c, 0) > 0)
    if len(present) == 3:
        return "shared_all_3"
    if len(present) == 2:
        return "shared_2_corpora"
    if len(present) == 1:
        return f"{present[0]}_only"
    return "no_expression_corpus_mentions"


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
            git_commit=gac.git_commit_hash(),
        )
        gac.update_run_manifest(run_dir, MODULE_NAME, "completed", output_dir=out_dir)
    except Exception as e:
        gac.update_run_manifest(run_dir, MODULE_NAME, "failed", error=str(e))
        raise

    print(f"\nDone. Run {run_id} -> {out_dir}")


if __name__ == "__main__":
    main()
