"""Prototype-theory typicality/borderline analysis over the shared
embedding space: for each expression corpus (literature/MIVILUDES/
interviews) as a whole, AND for each corpus's own epistemic-status
subgroups (e.g. `negated` -- "non-cult" statements -- as its own group),
which member expressions sit most/least typically near that group's own
centroid, and which points sit borderline between two groups.

GOVERNING PRINCIPLE (stated here, in every run's config.json, and in the
companion typicality_semantic_note.README.txt -- verbatim, never
paraphrased down): distance-to-centroid operationalizes "typicality" as a
purely geometric fact about this embedding space under this model; it is
not a claim about cognitive prototypicality, real-world category
membership, or whether an entity is "really" a cult. A "borderline" point
is one whose position is near-equidistant between two centroids in this
space; it does not by itself show the underlying claim is genuinely
ambiguous, disputed, or intermediate in meaning.

Four groupings, all dynamically discovered (statuses actually present per
corpus, never a hardcoded set of 5), thin subgroups always computed and
never dropped (n is always visible on the row):

  1. Per epistemology, whole corpus: typical (top-k nearest to own
     centroid) / atypical (top-k farthest).
  2. Per epistemology x epistemic-status subgroup: same, against that
     subgroup's own centroid.
  3. Borderline, cross-epistemology: the 3 unordered corpus pairs, pool =
     that pair's own points only.
  4. Borderline, cross-status-within-corpus: every unordered pair of one
     corpus's own present statuses, pool = that pair's own points only.

No bootstrap/equal-n correction anywhere in this module -- a deliberate
decision: typicality/borderline ranking is about a point's position
relative to *its own* group's centroid, not an aggregate statistic
literature's size would distort the way k-NN composition or dispersion
comparisons elsewhere in this toolkit need correcting for.

Usage (from thesis/corpus/):
    python -m thesis_corpus.analyze_typicality
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

from thesis_corpus import geometric_analysis_common as gac

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("thesis_corpus.analyze_typicality")

MODULE_NAME = "analyze_typicality"
DEFAULT_K = 10

GOVERNING_PRINCIPLE = (
    "Distance-to-centroid operationalizes \"typicality\" as a purely geometric fact "
    "about this embedding space under this model; it is not a claim about cognitive "
    "prototypicality, real-world category membership, or whether an entity is "
    "\"really\" a cult. A \"borderline\" point is one whose position is "
    "near-equidistant between two centroids in this space; it does not by itself "
    "show the underlying claim is genuinely ambiguous, disputed, or intermediate "
    "in meaning."
)


# ---------------------------------------------------------------------------
# Core ranking, pure over (points, vectors) -- no file I/O, fully unit-testable
# against small, synthetic, hand-built points/vectors.
# ---------------------------------------------------------------------------

def rank_group_by_centroid(
    points: list[dict], vectors: np.ndarray, group_indices: list[int],
) -> tuple[dict, list[dict], list[int]]:
    """Centroid + full ascending ranking of one group's own points against
    its own centroid.

    Returns (stats, ranked, global_indices_in_rank_order):
      - stats: centroid_and_dispersion_for_indices(vectors, group_indices).
      - ranked: gac.ranked_points(...) over exactly this group's own points
        (never a broader pool) -- ascending by distance, no k cutoff.
      - global_indices_in_rank_order[i]: the index into `points`/`vectors`
        that ranked[i] refers to. Recovered by replicating ranked_points'
        own argsort locally over the SAME candidate vectors -- the
        established pattern this toolkit already uses (see
        analyze_global_structure.py's source_centroid_nearest_expressions)
        to recover a global index per rank without parsing ranked_points'
        minimal per-row dict back apart. Asserted to line up 1:1 with
        `ranked` (same length, same order) since ranked_points with
        exclude_self=False never skips a candidate.
    """
    stats = gac.centroid_and_dispersion_for_indices(vectors, group_indices)
    centroid = stats["centroid"]
    group_points = [points[i] for i in group_indices]
    group_vectors = vectors[group_indices]

    ranked = gac.ranked_points(centroid, group_points, group_vectors)
    local_order = list(np.argsort(gac.euclidean_distances(centroid, group_vectors)))
    global_indices_in_rank_order = [group_indices[i] for i in local_order]

    if len(ranked) != len(global_indices_in_rank_order):
        raise AssertionError(
            f"rank_group_by_centroid: ranked_points returned {len(ranked)} rows but "
            f"the replicated argsort produced {len(global_indices_in_rank_order)} -- "
            "ranking mismatch, stop and investigate rather than silently zip mismatched rows."
        )
    return stats, ranked, global_indices_in_rank_order


def _base_row(points: list[dict], pooled_index: int, ranked_entry: dict, extra: dict) -> dict:
    point = points[pooled_index]
    pooled_key = point["key"]
    return {
        **extra,
        "pooled_key": pooled_key,
        "occurrence_key": None,  # filled in by attach_context_windows before writing
        "document_id": gac.key_document_id(pooled_key),
        "chunk_index": gac.key_chunk_index(pooled_key),
        "embedding_text": point["label"],
        "context_window": None,  # filled in by attach_context_windows before writing
        "attribution": point.get("attribution"),
        "claim_mode": point.get("claim_mode"),
        "epistemic_status": point.get("epistemic_status"),
        "euclidean_distance_to_centroid": ranked_entry["euclidean_distance"],
        "cosine_similarity_to_centroid": ranked_entry["cosine_similarity"],
    }


def typicality_rows_for_group(
    points: list[dict], vectors: np.ndarray, group_indices: list[int], k: int, extra: dict,
) -> tuple[list[dict], list[int]]:
    """Typical (top-k nearest to own centroid) + atypical (top-k farthest)
    rows for one group, sharing ranked_points' single ascending ranking
    for both directions (typical = ranked[:k], atypical =
    list(reversed(ranked))[:k]) -- never a separate "farthest" sort.

    `extra` is merged into every row (e.g. {"corpus": "literature"} or
    {"corpus": "literature", "epistemic_status_group": "qualified"}).

    Returns (rows, pooled_indices) -- pooled_indices[i] is the index into
    `points`/`vectors` rows[i] refers to, parallel arrays, consumed later
    by attach_context_windows.
    """
    stats, ranked, global_order = rank_group_by_centroid(points, vectors, group_indices)
    n_in_group = stats["n"]

    rows: list[dict] = []
    pooled_indices: list[int] = []

    typical_slice = list(zip(ranked[:k], global_order[:k]))
    for rank, (entry, pooled_index) in enumerate(typical_slice, start=1):
        row = _base_row(points, pooled_index, entry, extra)
        row.update({"direction": "typical", "rank": rank, "n_in_group": n_in_group})
        rows.append(row)
        pooled_indices.append(pooled_index)

    reversed_ranked = list(reversed(ranked))
    reversed_order = list(reversed(global_order))
    atypical_slice = list(zip(reversed_ranked[:k], reversed_order[:k]))
    for rank, (entry, pooled_index) in enumerate(atypical_slice, start=1):
        row = _base_row(points, pooled_index, entry, extra)
        row.update({"direction": "atypical", "rank": rank, "n_in_group": n_in_group})
        rows.append(row)
        pooled_indices.append(pooled_index)

    return rows, pooled_indices


def unordered_pairs(items: list[str]) -> list[tuple[str, str]]:
    return [(items[i], items[j]) for i in range(len(items)) for j in range(i + 1, len(items))]


def borderline_rows_for_pair(
    points: list[dict], vectors: np.ndarray,
    indices_a: list[int], indices_b: list[int],
    centroid_a: np.ndarray, centroid_b: np.ndarray,
    label_a: str, label_b: str, extra: dict,
) -> tuple[list[dict], list[int]]:
    """Borderline ranking over the union pool of exactly `indices_a` +
    `indices_b` (never a broader pool) -- wraps gac.borderline_ranking,
    replacing its generic "a"/"b" side markers with the actual group
    labels (corpus or epistemic-status names) in `nominally_closer_to`.

    Returns (rows, pooled_indices), parallel arrays as in
    typicality_rows_for_group.
    """
    pool_indices = list(indices_a) + list(indices_b)
    n_pool = len(pool_indices)
    if len(set(pool_indices)) != n_pool:
        raise AssertionError(
            f"borderline_rows_for_pair({label_a!r}, {label_b!r}): pool_indices contains "
            "duplicates -- the two groups being contrasted must be disjoint."
        )

    entries = gac.borderline_ranking(points, vectors, pool_indices, centroid_a, centroid_b)

    rows: list[dict] = []
    pooled_indices: list[int] = []
    for entry in entries:
        pooled_index = entry["pool_index"]
        point = points[pooled_index]
        pooled_key = point["key"]
        nominally_closer_to = label_a if entry["nominally_closer_to"] == "a" else label_b
        rows.append({
            **extra,
            "rank": entry["rank"], "n_pool": n_pool,
            "pooled_key": pooled_key, "occurrence_key": None,
            "document_id": gac.key_document_id(pooled_key), "chunk_index": gac.key_chunk_index(pooled_key),
            "embedding_text": point["label"], "context_window": None,
            "attribution": point.get("attribution"), "claim_mode": point.get("claim_mode"),
            "epistemic_status": point.get("epistemic_status"),
            "source_dataset_of_point": point.get("source_dataset"),
            "distance_to_a": entry["distance_to_a"], "distance_to_b": entry["distance_to_b"],
            "ambiguity_ratio": entry["ambiguity_ratio"],
            "nominally_closer_to": nominally_closer_to,
            "min_distance_to_either_centroid": entry["min_distance_to_either_centroid"],
        })
        pooled_indices.append(pooled_index)
    return rows, pooled_indices


# ---------------------------------------------------------------------------
# Context-window attachment -- the only I/O-adjacent step; pure with respect
# to its own inputs, so it's unit-testable with a fake `resolutions` dict
# (any object exposing the same two dict attributes gac.resolve_context_windows
# returns) -- no real archive files needed.
# ---------------------------------------------------------------------------

def attach_context_windows(points: list[dict], rows: list[dict], pooled_indices: list[int], resolutions: dict) -> None:
    """Mutates `rows` in place, filling `occurrence_key`/`context_window`
    from `resolutions` (keyed by source_dataset, values matching
    gac.ContextWindowResolution's shape). Hard-fails (ValueError) on any
    SELECTED row with no resolved context_window, per this toolkit's
    fail-loud rule (see gac.resolve_context_windows's docstring and
    analyze_global_structure.py's source_centroid_nearest_expressions for
    the established precedent) -- never a blank field.
    """
    if len(rows) != len(pooled_indices):
        raise AssertionError(f"attach_context_windows: {len(rows)} rows but {len(pooled_indices)} pooled_indices.")
    for row, pooled_index in zip(rows, pooled_indices):
        source_dataset = points[pooled_index]["source_dataset"]
        resolution = resolutions[source_dataset]
        occurrence_key = resolution.occurrence_key_by_pooled_index.get(pooled_index)
        context_window = resolution.context_window_by_pooled_index.get(pooled_index)
        if not context_window:
            raise ValueError(
                f"analyze_typicality: missing context_window for a SELECTED row -- "
                f"source_dataset={source_dataset!r}, pooled_key={row.get('pooled_key')!r}, "
                f"pooled_index={pooled_index}, occurrence_key={occurrence_key!r}. This is a "
                "hard-fail per the fail-loud context-window resolution rule; investigate the "
                "occurrence-aware join for this source before proceeding."
            )
        row["occurrence_key"] = occurrence_key
        row["context_window"] = context_window


# ---------------------------------------------------------------------------
# Validation -- entirely new files, no baseline diff to compare against.
# ---------------------------------------------------------------------------

def validate_typicality_ranks_and_monotonicity(rows: list[dict], group_fields: tuple[str, ...]) -> None:
    """Every rank 1..n (n = number of rows in that group/direction) present
    exactly once, no duplicates; distance strictly non-decreasing for
    `typical`, non-increasing for `atypical`."""
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row[f] for f in group_fields) + (row["direction"],)
        grouped.setdefault(key, []).append(row)

    for key, group_rows in grouped.items():
        ordered = sorted(group_rows, key=lambda r: r["rank"])
        ranks = [r["rank"] for r in ordered]
        if ranks != list(range(1, len(ranks) + 1)):
            raise AssertionError(f"validate_typicality_ranks_and_monotonicity: ranks {ranks} for group {key} are not exactly 1..{len(ranks)}.")
        distances = [r["euclidean_distance_to_centroid"] for r in ordered]
        direction = key[-1]
        if direction == "typical":
            bad = any(distances[i] > distances[i + 1] + 1e-9 for i in range(len(distances) - 1))
            if bad:
                raise AssertionError(f"validate_typicality_ranks_and_monotonicity: typical distances not non-decreasing for group {key}: {distances}")
        else:
            bad = any(distances[i] < distances[i + 1] - 1e-9 for i in range(len(distances) - 1))
            if bad:
                raise AssertionError(f"validate_typicality_ranks_and_monotonicity: atypical distances not non-increasing for group {key}: {distances}")


def validate_n_in_group_matches_direct_count(rows: list[dict], points: list[dict]) -> None:
    """`n_in_group` on every row, cross-checked against an independent
    direct population count (a fresh scan of `points`, not a reuse of
    whatever produced the row) -- hard-fail on mismatch."""
    for row in rows:
        corpus = row["corpus"]
        status_group = row.get("epistemic_status_group")
        if status_group is not None:
            direct_n = sum(1 for p in points if p.get("source_dataset") == corpus and p.get("epistemic_status") == status_group)
        else:
            direct_n = sum(1 for p in points if p.get("source_dataset") == corpus)
        if row["n_in_group"] != direct_n:
            raise AssertionError(
                f"validate_n_in_group_matches_direct_count: row claims n_in_group={row['n_in_group']} "
                f"for corpus={corpus!r} status_group={status_group!r}, but a direct count found {direct_n}."
            )


def validate_n_pool_matches_direct_count(rows: list[dict], points: list[dict]) -> None:
    """`n_pool` on every borderline row, cross-checked against an
    independent direct population count of the two contrasted groups --
    hard-fail on mismatch."""
    for row in rows:
        if "corpus_a" in row:
            n_a = sum(1 for p in points if p.get("source_dataset") == row["corpus_a"])
            n_b = sum(1 for p in points if p.get("source_dataset") == row["corpus_b"])
        else:
            corpus = row["corpus"]
            n_a = sum(1 for p in points if p.get("source_dataset") == corpus and p.get("epistemic_status") == row["status_a"])
            n_b = sum(1 for p in points if p.get("source_dataset") == corpus and p.get("epistemic_status") == row["status_b"])
        direct_n_pool = n_a + n_b
        if row["n_pool"] != direct_n_pool:
            raise AssertionError(
                f"validate_n_pool_matches_direct_count: row claims n_pool={row['n_pool']}, "
                f"but a direct count of its two groups found {direct_n_pool}."
            )


def validate_borderline_rows(rows: list[dict]) -> None:
    """`ambiguity_ratio >= 0`; `nominally_closer_to` consistent with which
    of distance_to_a/distance_to_b is smaller."""
    for row in rows:
        if row["ambiguity_ratio"] < 0:
            raise AssertionError(f"validate_borderline_rows: negative ambiguity_ratio in row {row}.")
        label_a = row.get("corpus_a", row.get("status_a"))
        label_b = row.get("corpus_b", row.get("status_b"))
        expected = label_a if row["distance_to_a"] <= row["distance_to_b"] else label_b
        if row["nominally_closer_to"] != expected:
            raise AssertionError(
                f"validate_borderline_rows: nominally_closer_to={row['nominally_closer_to']!r} "
                f"inconsistent with distance_to_a={row['distance_to_a']}, distance_to_b={row['distance_to_b']} "
                f"(expected {expected!r})."
            )


# ---------------------------------------------------------------------------
# Orchestration over the 4 groupings -- pure over (points, vectors, k), no
# file I/O; main() below is the only part that touches disk.
# ---------------------------------------------------------------------------

@dataclass
class TypicalityResult:
    typicality_by_corpus_rows: list[dict] = field(default_factory=list)
    typicality_by_corpus_and_status_rows: list[dict] = field(default_factory=list)
    borderline_cross_epistemology_rows: list[dict] = field(default_factory=list)
    borderline_cross_status_within_corpus_rows: list[dict] = field(default_factory=list)
    typicality_by_corpus_pooled_indices: list[int] = field(default_factory=list)
    typicality_by_corpus_and_status_pooled_indices: list[int] = field(default_factory=list)
    borderline_cross_epistemology_pooled_indices: list[int] = field(default_factory=list)
    borderline_cross_status_within_corpus_pooled_indices: list[int] = field(default_factory=list)
    groupings_summary: dict = field(default_factory=dict)


def compute_typicality(points: list[dict], vectors: np.ndarray, k: int) -> TypicalityResult:
    result = TypicalityResult()

    per_corpus_indices: dict[str, list[int]] = {
        corpus: gac.source_dataset_indices(points, corpus) for corpus in gac.EXPRESSION_CORPORA
    }
    per_corpus_stats: dict[str, dict] = {}
    per_corpus_subgroup_indices: dict[str, dict[str, list[int]]] = {}
    per_corpus_subgroup_stats: dict[str, dict[str, dict]] = {}

    groupings_summary: dict = {"per_corpus": {}, "cross_epistemology_pairs": [], "cross_status_pairs": []}

    # 1. Per epistemology, whole corpus.
    for corpus in gac.EXPRESSION_CORPORA:
        idxs = per_corpus_indices[corpus]
        rows, pooled_indices = typicality_rows_for_group(points, vectors, idxs, k, {"corpus": corpus})
        result.typicality_by_corpus_rows.extend(rows)
        result.typicality_by_corpus_pooled_indices.extend(pooled_indices)
        per_corpus_stats[corpus] = gac.centroid_and_dispersion_for_indices(vectors, idxs)

        subgroups = gac.epistemic_status_subgroup_indices(points, corpus)
        per_corpus_subgroup_indices[corpus] = subgroups
        per_corpus_subgroup_stats[corpus] = {
            status: gac.centroid_and_dispersion_for_indices(vectors, sub_idxs)
            for status, sub_idxs in subgroups.items()
        }
        groupings_summary["per_corpus"][corpus] = {
            "n": len(idxs),
            "statuses": {status: len(sub_idxs) for status, sub_idxs in subgroups.items()},
        }

        # 2. Per epistemology x epistemic-status subgroup.
        for status, sub_idxs in subgroups.items():
            sub_rows, sub_pooled_indices = typicality_rows_for_group(
                points, vectors, sub_idxs, k, {"corpus": corpus, "epistemic_status_group": status},
            )
            result.typicality_by_corpus_and_status_rows.extend(sub_rows)
            result.typicality_by_corpus_and_status_pooled_indices.extend(sub_pooled_indices)

    # 3. Borderline, cross-epistemology: the 3 unordered corpus pairs.
    for corpus_a, corpus_b in unordered_pairs(list(gac.EXPRESSION_CORPORA)):
        rows, pooled_indices = borderline_rows_for_pair(
            points, vectors,
            per_corpus_indices[corpus_a], per_corpus_indices[corpus_b],
            per_corpus_stats[corpus_a]["centroid"], per_corpus_stats[corpus_b]["centroid"],
            corpus_a, corpus_b, {"corpus_a": corpus_a, "corpus_b": corpus_b},
        )
        result.borderline_cross_epistemology_rows.extend(rows)
        result.borderline_cross_epistemology_pooled_indices.extend(pooled_indices)
        groupings_summary["cross_epistemology_pairs"].append({
            "corpus_a": corpus_a, "corpus_b": corpus_b, "n_pool": len(rows),
        })

    # 4. Borderline, cross-status-within-corpus: every unordered pair of one
    # corpus's own present statuses.
    for corpus in gac.EXPRESSION_CORPORA:
        statuses = list(per_corpus_subgroup_indices[corpus].keys())
        for status_a, status_b in unordered_pairs(statuses):
            rows, pooled_indices = borderline_rows_for_pair(
                points, vectors,
                per_corpus_subgroup_indices[corpus][status_a], per_corpus_subgroup_indices[corpus][status_b],
                per_corpus_subgroup_stats[corpus][status_a]["centroid"], per_corpus_subgroup_stats[corpus][status_b]["centroid"],
                status_a, status_b, {"corpus": corpus, "status_a": status_a, "status_b": status_b},
            )
            result.borderline_cross_status_within_corpus_rows.extend(rows)
            result.borderline_cross_status_within_corpus_pooled_indices.extend(pooled_indices)
            groupings_summary["cross_status_pairs"].append({
                "corpus": corpus, "status_a": status_a, "status_b": status_b, "n_pool": len(rows),
            })

    result.groupings_summary = groupings_summary
    return result


def run_all_validations(result: TypicalityResult, points: list[dict]) -> None:
    validate_typicality_ranks_and_monotonicity(result.typicality_by_corpus_rows, ("corpus",))
    validate_typicality_ranks_and_monotonicity(result.typicality_by_corpus_and_status_rows, ("corpus", "epistemic_status_group"))
    validate_n_in_group_matches_direct_count(result.typicality_by_corpus_rows, points)
    validate_n_in_group_matches_direct_count(result.typicality_by_corpus_and_status_rows, points)
    validate_n_pool_matches_direct_count(result.borderline_cross_epistemology_rows, points)
    validate_n_pool_matches_direct_count(result.borderline_cross_status_within_corpus_rows, points)
    validate_borderline_rows(result.borderline_cross_epistemology_rows)
    validate_borderline_rows(result.borderline_cross_status_within_corpus_rows)


def build_readme_text(groupings_summary: dict) -> str:
    lines = [GOVERNING_PRINCIPLE, "", "Groupings actually computed in this run:", ""]
    for corpus in gac.EXPRESSION_CORPORA:
        info = groupings_summary["per_corpus"][corpus]
        lines.append(f"- {corpus}: n={info['n']}")
        for status, n in info["statuses"].items():
            lines.append(f"    - epistemic_status={status}: n={n}")
    lines.append("")
    lines.append("Borderline, cross-epistemology pairs (pool = that pair's own points only):")
    for pair in groupings_summary["cross_epistemology_pairs"]:
        lines.append(f"- {pair['corpus_a']} vs {pair['corpus_b']}: n_pool={pair['n_pool']}")
    lines.append("")
    lines.append("Borderline, cross-status-within-corpus pairs (pool = that pair's own points only):")
    for pair in groupings_summary["cross_status_pairs"]:
        lines.append(f"- {pair['corpus']} ({pair['status_a']} vs {pair['status_b']}): n_pool={pair['n_pool']}")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--shared-space-dir", type=Path, default=None,
                         help="Use a different pooled space (e.g. processed/shared_space_v2/) instead of v1's "
                              "processed/shared_space/; also switches the run-output root to a sibling "
                              "processed/analysis_v2/ directory so v1 and v2 runs are never mixed.")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    args = parser.parse_args()

    embedding_space_path, interview_prototypes_path, analysis_root = gac.resolve_space_paths(args.shared_space_dir)
    logger.info("Loading %s ...", embedding_space_path)
    shared_space = gac.load_shared_space(embedding_space_path)

    run_id, run_dir = gac.get_or_create_run_dir(args.run_id, analysis_root)
    gac.init_run_manifest(run_dir, shared_space, defaults={"k": args.k})
    gac.update_run_manifest(run_dir, MODULE_NAME, "running")
    out_dir = gac.module_run_dir(run_dir, MODULE_NAME)

    try:
        logger.info("Computing typicality/borderline rankings (k=%d)...", args.k)
        result = compute_typicality(shared_space.points, shared_space.vectors, args.k)

        logger.info("Resolving context windows for literature/miviludes/interviews...")
        archive_paths = gac.resolve_archive_paths(shared_space)
        resolutions = {
            corpus: gac.resolve_context_windows(shared_space, corpus, archive_paths[corpus])
            for corpus in gac.EXPRESSION_CORPORA
        }

        attach_context_windows(shared_space.points, result.typicality_by_corpus_rows, result.typicality_by_corpus_pooled_indices, resolutions)
        attach_context_windows(shared_space.points, result.typicality_by_corpus_and_status_rows, result.typicality_by_corpus_and_status_pooled_indices, resolutions)
        attach_context_windows(shared_space.points, result.borderline_cross_epistemology_rows, result.borderline_cross_epistemology_pooled_indices, resolutions)
        attach_context_windows(shared_space.points, result.borderline_cross_status_within_corpus_rows, result.borderline_cross_status_within_corpus_pooled_indices, resolutions)

        logger.info("Running validations...")
        run_all_validations(result, shared_space.points)

        gac.write_csv(out_dir / "typicality_by_corpus.csv", result.typicality_by_corpus_rows)
        gac.write_csv(out_dir / "typicality_by_corpus_and_status.csv", result.typicality_by_corpus_and_status_rows)
        gac.write_csv(out_dir / "borderline_cross_epistemology.csv", result.borderline_cross_epistemology_rows)
        gac.write_csv(out_dir / "borderline_cross_status_within_corpus.csv", result.borderline_cross_status_within_corpus_rows)
        (out_dir / "typicality_semantic_note.README.txt").write_text(build_readme_text(result.groupings_summary), encoding="utf-8")

        gac.write_module_config(
            out_dir,
            input_sha256=shared_space.input_sha256,
            k=args.k,
            governing_principle=GOVERNING_PRINCIPLE,
            groupings_computed=result.groupings_summary,
            git_commit=gac.git_commit_hash(),
        )
        gac.update_run_manifest(run_dir, MODULE_NAME, "completed", output_dir=out_dir)
    except Exception as e:
        gac.update_run_manifest(run_dir, MODULE_NAME, "failed", error=str(e))
        raise

    print(f"\nDone. Run {run_id} -> {out_dir}")


if __name__ == "__main__":
    main()
