"""Shared library for the geometric-analysis toolkit
(analyze_global_structure.py, analyze_cluster_structure.py,
audit_free_listing_rank.py, propose_initial_exemplars.py,
build_interview_prototype_layer.py, analyze_initial_exemplars.py,
analyze_criterion_neighbours.py, analyze_emergent_entities.py,
analyze_typicality.py, generate_figures.py,
generate_geometric_draft_report.py) built on top of the shared embedding
space.

Read-only with respect to the pipeline that produced the data: never
imports from or modifies build_shared_space.py's code, and never writes
under processed/shared_space/ except for build_interview_prototype_layer.py's
own interview_prototypes.jsonl (a new, separate file). Reads two of
build_shared_space.py's own outputs there read-only: embedding_space.jsonl
itself, and (since this session's persistence addition)
pca_transform.joblib -- the fitted StandardScaler+PCA, needed to project
a reviewed interview exemplar into the *existing* shared space without
refitting it or otherwise touching embedding_space.jsonl. Does import
balanced_analysis.py's weighted_centroid()/per_corpus_centroids() directly
rather than reimplementing equal-corpus-weighting -- that module already
is this toolkit's imbalance-mitigation library, just predates the rest of
it.

Every module in this toolkit shares, from here:
  - loading embedding_space.jsonl into (points, 394-D vectors) -- never a
    2-D/3-D slice.
  - four analysis modes -- full / reduced_literature / equal_n_expression /
    equal_weight (see MODES and the functions below); never silently one.
  - versioned run directories under processed/analysis/<run-id>/<module>/,
    tied together by one RUN_MANIFEST.json per run.
  - the Okabe-Ito colour-vision-deficiency-safe categorical palette.
  - two virtual-key schemes, both "document_id:chunk_index:occurrence",
    needed because document_id:chunk_index alone is not unique per
    expression (a chunk can yield several expressions; interviews' 230 raw
    expressions span only 75 unique document_id:chunk_index pairs -- the
    same non-uniqueness build_shared_space.py's MIVILUDES French/English
    translation join hit and was fixed for, see that file's
    load_miviludes_translations docstring). `derive_interview_expression_keys`
    counts occurrence over the *pooled* embedding_space.jsonl -- an item
    build_shared_space.py's filters dropped has no key here at all.
    `derive_archive_expression_keys` counts occurrence over the raw,
    unfiltered archive instead -- stable regardless of pooling, used by
    the initial-exemplar workflow specifically because it needs to resolve
    short free-association answers the pooling filter routinely drops.

Library only -- no CLI of its own.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import umap

logger = logging.getLogger("thesis_corpus.geometric_analysis_common")

CORPUS_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = CORPUS_DIR / "processed"
SHARED_SPACE_DIR = PROCESSED_DIR / "shared_space"
ANALYSIS_DIR = PROCESSED_DIR / "analysis"

EMBEDDING_SPACE_PATH = SHARED_SPACE_DIR / "embedding_space.jsonl"
LITERATURE_BALANCED_SAMPLE_PATH = SHARED_SPACE_DIR / "literature_balanced_sample.jsonl"
PCA_TRANSFORM_PATH = SHARED_SPACE_DIR / "pca_transform.joblib"
PCA_TRANSFORM_METADATA_PATH = SHARED_SPACE_DIR / "pca_transform_metadata.json"
INTERVIEW_PROTOTYPES_PATH = SHARED_SPACE_DIR / "interview_prototypes.jsonl"
INTERVIEWS_ARCHIVE_PATH = PROCESSED_DIR / "interviews" / "criterion_expressions.jsonl"
LITERATURE_ARCHIVE_PATH = PROCESSED_DIR / "literature" / "criterion_expressions.jsonl"
MIVILUDES_ARCHIVE_PATH = PROCESSED_DIR / "miviludes" / "criterion_expressions.jsonl"
INITIAL_EXEMPLARS_CSV_PATH = CORPUS_DIR / "interviews" / "metadata" / "initial_exemplars.csv"

EXPRESSION_CORPORA = ("literature", "miviludes", "interviews")
ARCHIVE_PATH_BY_SOURCE = {
    "literature": LITERATURE_ARCHIVE_PATH,
    "miviludes": MIVILUDES_ARCHIVE_PATH,
    "interviews": INTERVIEWS_ARCHIVE_PATH,
}


def resolve_archive_paths(shared_space: "SharedSpace") -> dict[str, Path]:
    """Which raw per-corpus archive each expression corpus's context_window
    (resolve_context_windows below) should be resolved against, for
    whichever shared space was actually loaded -- never assumed to be v1's
    fixed ARCHIVE_PATH_BY_SOURCE.

    build_shared_space.py records the exact archive paths it pooled from
    into `archive_paths` in the sidecar pca_transform_metadata.json next to
    the space it built, from this session onward. Reads that field when
    present; falls back to ARCHIVE_PATH_BY_SOURCE (v1's fixed paths) for a
    metadata file that predates this field -- including v1's own, frozen
    and never rewritten, so this fallback is not optional scaffolding."""
    metadata_path = shared_space.input_path.parent / "pca_transform_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        archive_paths = metadata.get("archive_paths")
        if archive_paths:
            return {corpus: Path(path) for corpus, path in archive_paths.items()}
    return dict(ARCHIVE_PATH_BY_SOURCE)
ALL_SOURCE_DATASETS = (
    "literature", "miviludes", "interviews", "miviludes_criteria",
    "concept_backbone", "structural_concepts", "emergent_entities",
    "conceptnet_concepts",
)
# The three controlled-vocabulary reference term lists -- distinct from
# analyze_global_structure.py's REFERENCE_LIKE_DATASETS (a local, 4-item
# superset that also includes emergent_entities for centroid-only
# comparisons, where reference-pool-size bias doesn't apply since there's
# no k-NN retrieval involved). The one shared source of truth for "the
# three reference vocabularies whose different sizes need controlling
# for" (see equal_size_reference_comparison below) -- no module keeps its
# own copy of this list.
REFERENCE_VOCAB_DATASETS = ("concept_backbone", "structural_concepts", "conceptnet_concepts")

# epistemic_status is populated only on the three expression corpora
# (literature/miviludes/interviews) -- null everywhere else (see
# source_dataset_indices' epistemic_status_filter param below, which
# always lets a null value through regardless of filter). "all" (None)
# reproduces pre-filter behaviour exactly; "asserted_qualified" restricts
# to positively-claimed statements, excluding contested/negated/
# speculative ones -- added to check whether e.g. "X is NOT a cult"
# (negated) sitting geometrically close to "X is a cult" (asserted) is
# distorting proximity-based results.
EPISTEMIC_STATUS_FILTER_CHOICES: dict[str, tuple[str, ...] | None] = {
    "all": None,
    "asserted_qualified": ("asserted", "qualified"),
}

# "full" is reserved for a statistic actually computed over every
# applicable point -- see analyze_cluster_structure.py's
# full_sampled_pointwise for the sampled variant silhouette needs. Modes
# themselves stay four, always spelled out, never silently one.
MODES = ("full", "reduced_literature", "equal_n_expression", "equal_weight")

DEFAULT_SEED = 42
DEFAULT_BOOTSTRAP_REPS = 100
DEFAULT_POINT_SAMPLE_SIZE = 5000

# ---------------------------------------------------------------------------
# Okabe-Ito CVD-safe categorical palette
# ---------------------------------------------------------------------------

OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}
# Fixed hue order -- never cycled, never reassigned by filtered subset.
_OKABE_ITO_ORDER = [
    "blue", "vermillion", "bluish_green", "orange",
    "reddish_purple", "sky_blue", "yellow", "black",
]

SOURCE_DATASET_COLORS = dict(zip(ALL_SOURCE_DATASETS, [OKABE_ITO[k] for k in _OKABE_ITO_ORDER]))
POINT_ROLE_COLORS = {
    "expression": OKABE_ITO["blue"],
    "reference": OKABE_ITO["bluish_green"],
    "emergent": OKABE_ITO["vermillion"],
}


def source_dataset_color(source_dataset: str) -> str:
    return SOURCE_DATASET_COLORS[source_dataset]


def point_role_color(point_role: str) -> str:
    return POINT_ROLE_COLORS[point_role]


# ---------------------------------------------------------------------------
# Checksums
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    """Fieldnames are the ordered union across every row, not just
    rows[0] -- output rows across this toolkit aren't always perfectly
    homogeneous (e.g. a summary table mixing "resolved" and "unavailable"
    row shapes), and DictWriter raises on an unseen field otherwise."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Loading the shared space
# ---------------------------------------------------------------------------

@dataclass
class SharedSpace:
    points: list[dict]
    vectors: np.ndarray  # (n, 394), float64
    input_path: Path
    input_sha256: str


def resolve_space_paths(shared_space_dir: Path | None) -> tuple[Path, Path, Path]:
    """Which pooled space the toolkit should read, and where its run output
    should go. `shared_space_dir=None` (every module's default) keeps v1
    behaviour byte-for-byte unchanged: EMBEDDING_SPACE_PATH,
    INTERVIEW_PROTOTYPES_PATH, ANALYSIS_DIR.

    Passing e.g. `processed/shared_space_v2/` (the directory
    build_shared_space.py --run-tag writes) switches all three: that
    directory's own embedding_space.jsonl/interview_prototypes.jsonl, and a
    sibling `processed/analysis_v2/` output root -- derived from the space
    directory's own name (`shared_space` -> `analysis`, `shared_space_v2` ->
    `analysis_v2`) so v1 and v2 runs can never land in the same run-id
    directory by accident. `interview_prototypes.jsonl` is not required to
    exist at this path (checked by each caller the same way it already
    checks INTERVIEW_PROTOTYPES_PATH)."""
    if shared_space_dir is None:
        return EMBEDDING_SPACE_PATH, INTERVIEW_PROTOTYPES_PATH, ANALYSIS_DIR
    shared_space_dir = Path(shared_space_dir)
    suffix = shared_space_dir.name[len("shared_space"):] if shared_space_dir.name.startswith("shared_space") else f"_{shared_space_dir.name}"
    analysis_root = PROCESSED_DIR / f"analysis{suffix}"
    return (
        shared_space_dir / "embedding_space.jsonl",
        shared_space_dir / "interview_prototypes.jsonl",
        analysis_root,
    )


def load_shared_space(path: Path = EMBEDDING_SPACE_PATH) -> SharedSpace:
    """Streams embedding_space.jsonl. Keeps every metadata field and the
    full 394-D vector -- geometric analysis in this toolkit never operates
    on a 2-D/3-D slice (that's visualize_3d.py's job, for a different
    purpose)."""
    points: list[dict] = []
    vectors: list[list[float]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            points.append({k: v for k, v in item.items() if k != "shared_space_vector"})
            vectors.append(item["shared_space_vector"])
    return SharedSpace(
        points=points,
        vectors=np.array(vectors, dtype=np.float64),
        input_path=path,
        input_sha256=sha256_file(path),
    )


def load_reduced_literature_points(
    path: Path = LITERATURE_BALANCED_SAMPLE_PATH,
) -> tuple[list[dict], np.ndarray]:
    """literature_balanced_sample.jsonl -- same row shape as
    embedding_space.jsonl, a separate 2,500-row file, not addressable by
    index into the main SharedSpace."""
    points: list[dict] = []
    vectors: list[list[float]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            points.append({k: v for k, v in item.items() if k != "shared_space_vector"})
            vectors.append(item["shared_space_vector"])
    return points, np.array(vectors, dtype=np.float64)


# ---------------------------------------------------------------------------
# Virtual interview-expression key (document_id:chunk_index:occurrence)
# ---------------------------------------------------------------------------

def key_document_id(key: str) -> str:
    """"document_id:chunk_index" -> "document_id". rsplit on the last ':'
    since document_id itself may contain no colons in practice, but rsplit
    is the safe direction regardless (same convention as
    balanced_analysis._document_id)."""
    return key.rsplit(":", 1)[0]


def key_chunk_index(key: str) -> int:
    return int(key.rsplit(":", 1)[1])


def derive_pooled_expression_keys(points: list[dict], source_dataset: str) -> dict[str, int]:
    """Maps each `source_dataset == source_dataset` point's position in
    `points` to a virtual key "document_id:chunk_index:occurrence", where
    `occurrence` counts repeats of (document_id, chunk_index) among that
    source's rows in `points`' own order. `points` MUST be in
    embedding_space.jsonl's own file order (i.e. straight from
    load_shared_space) for this to be reproducible run to run.

    Source-agnostic generalization of what was originally
    derive_interview_expression_keys (interviews-only) -- needed once
    literature/miviludes also needed pooled-point-to-raw-archive
    resolution (see resolve_context_windows below), not just interviews.

    This key space is specific to the *pooled* file -- an expression
    dropped by build_shared_space.py's dedup/short-fragment filter has no
    key here at all, since occurrence is only counted over survivors. For
    something that needs to resolve an item regardless of whether it
    survived pooling (e.g. the initial-exemplar workflow, which cares
    about short free-association answers exactly the filter tends to
    drop), use derive_archive_expression_keys on the raw archive instead --
    see build_interview_prototype_layer.py.

    Returns {virtual_key: index_into_points}.
    """
    occurrence_by_chunk: dict[tuple[str, int], int] = defaultdict(int)
    keys: dict[str, int] = {}
    for i, p in enumerate(points):
        if p.get("source_dataset") != source_dataset:
            continue
        document_id = key_document_id(p["key"])
        chunk_index = key_chunk_index(p["key"])
        chunk = (document_id, chunk_index)
        occurrence = occurrence_by_chunk[chunk]
        occurrence_by_chunk[chunk] += 1
        keys[f"{document_id}:{chunk_index}:{occurrence}"] = i
    return keys


def derive_interview_expression_keys(points: list[dict]) -> dict[str, int]:
    """Backward-compatible interviews-only wrapper around
    derive_pooled_expression_keys -- kept in case anything still refers to
    this name; no known caller as of this generalization."""
    return derive_pooled_expression_keys(points, "interviews")


def load_raw_archive(path: Path) -> list[dict]:
    """A corpus's raw criterion_expressions.jsonl, as a flat list in its own
    on-disk order -- every extracted expression, including ones a
    pooling-time filter would later drop. Never mutated, never filtered
    here; that's the caller's job if they want it."""
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def derive_archive_expression_keys(archive_items: list[dict]) -> dict[str, int]:
    """Same virtual-key construction as derive_interview_expression_keys
    ("document_id:chunk_index:occurrence"), but over the full, unfiltered
    raw archive in its own on-disk order, rather than the pooled/filtered
    embedding_space.jsonl -- stable regardless of any pooling-time filter,
    since it never depends on which items later survive. `archive_items`
    should come straight from load_raw_archive (same order every time).

    Returns {virtual_key: index_into_archive_items}.
    """
    occurrence_by_chunk: dict[tuple[str, int], int] = defaultdict(int)
    keys: dict[str, int] = {}
    for i, item in enumerate(archive_items):
        chunk = (item["document_id"], item["chunk_index"])
        occurrence = occurrence_by_chunk[chunk]
        occurrence_by_chunk[chunk] += 1
        keys[f"{item['document_id']}:{item['chunk_index']}:{occurrence}"] = i
    return keys


@dataclass
class ContextWindowResolution:
    """Result of resolve_context_windows for one source_dataset. Built once
    per source per module run and reused across every retrieval call for
    that source (never rebuilt per individual row) -- callers should call
    resolve_context_windows(shared_space, source) once, then look up
    `.context_window_by_pooled_index`/`.occurrence_key_by_pooled_index` as
    many times as needed."""
    source_dataset: str
    context_window_by_pooled_index: dict[int, str]
    occurrence_key_by_pooled_index: dict[int, str]
    archive_items: list[dict]
    archive_index_by_pooled_index: dict[int, int]
    stats: dict


def resolve_context_windows(
    shared_space: "SharedSpace", source_dataset: str, archive_path: Path | None = None,
) -> ContextWindowResolution:
    """Strict, occurrence-aware join from pooled `embedding_space.jsonl`
    points back to their raw `criterion_expressions.jsonl` archive record,
    for the one field the pooled file never carries: `context_window`.

    `archive_path` (default None): which raw archive to join against.
    Callers should pass `resolve_archive_paths(shared_space)[source_dataset]`
    rather than rely on the None default whenever `shared_space` might not
    be v1's -- the default only exists for v1 callers that predate this
    parameter, and silently joining a non-v1 shared space against v1's
    fixed ARCHIVE_PATH_BY_SOURCE would resolve every occurrence key against
    the wrong archive without necessarily raising (a real, previously-hit
    bug: it can even "match" by coincidence before failing much later, or
    not at all).

    Built once over ALL of `source_dataset`'s points in
    `shared_space.points` (full file order, not a filtered subset --
    occurrence-counting must match the raw archive's own deterministic
    ordering), composing derive_pooled_expression_keys (pooled side) with
    derive_archive_expression_keys (raw-archive side) via the shared
    occurrence-aware virtual key "document_id:chunk_index:occurrence".

    NOTE (real, expected, not a bug): for `source_dataset="miviludes"`
    specifically, the pooled point's `label` (what was actually embedded,
    and therefore what any distance/nearest-neighbour computation used) is
    the ENGLISH machine translation, while the resolved `context_window`
    from the raw archive is the original FRENCH surrounding text -- MIVILUDES
    is pooled by its English translation with French kept only as
    `label_fr` (see ANALYSIS_OVERVIEW.md/Methods.tex). `embedding_text` and
    `context_window` are therefore in different languages for miviludes
    rows, by design of the translation pipeline, not a join defect --
    every caller of this function should say so explicitly wherever it
    presents both fields together.

    Does not raise on an unmatched pooled point by itself (the corpus-wide
    hit rate is a diagnostic, logged via `.stats`) -- it is the CALLER's
    job to hard-fail when a point that's actually selected for a
    deliverable has no resolved context_window; see analyze_global_structure.py's
    Deliverable 4 and the shared-anchor retrieval for that enforcement.
    """
    archive_path = archive_path or ARCHIVE_PATH_BY_SOURCE[source_dataset]
    archive_items = load_raw_archive(archive_path)
    archive_keys = derive_archive_expression_keys(archive_items)
    pooled_keys = derive_pooled_expression_keys(shared_space.points, source_dataset)

    context_window_by_pooled_index: dict[int, str] = {}
    occurrence_key_by_pooled_index: dict[int, str] = {}
    archive_index_by_pooled_index: dict[int, int] = {}
    used_archive_indices: set[int] = set()
    unmatched = 0
    for virtual_key, pooled_index in pooled_keys.items():
        occurrence_key_by_pooled_index[pooled_index] = virtual_key
        archive_index = archive_keys.get(virtual_key)
        if archive_index is None:
            unmatched += 1
            continue
        archive_index_by_pooled_index[pooled_index] = archive_index
        used_archive_indices.add(archive_index)
        context_window_by_pooled_index[pooled_index] = archive_items[archive_index].get("context_window", "")

    # "duplicate-key count observed before occurrence-level disambiguation":
    # how many (document_id, chunk_index) chunks needed occurrence
    # suffixing at all (i.e. yielded >1 pooled expression), over the pooled
    # side -- a chunk that yields exactly 1 expression never collides.
    chunk_counts: dict[tuple[str, int], int] = defaultdict(int)
    for p in shared_space.points:
        if p.get("source_dataset") == source_dataset:
            chunk_counts[(key_document_id(p["key"]), key_chunk_index(p["key"]))] += 1
    duplicate_key_chunks = sum(1 for count in chunk_counts.values() if count > 1)

    stats = {
        "source_dataset": source_dataset,
        "total_pooled_expressions": len(pooled_keys),
        "total_raw_archive_expressions": len(archive_items),
        "matched_pooled_expressions": len(context_window_by_pooled_index),
        "unmatched_pooled_expressions": unmatched,
        "unused_archive_expressions": len(archive_items) - len(used_archive_indices),
        "duplicate_key_chunks_before_disambiguation": duplicate_key_chunks,
    }
    logger.info(
        "resolve_context_windows(%s): %d pooled, %d archive, %d matched, %d unmatched, "
        "%d archive items unused, %d chunks needed occurrence disambiguation.",
        source_dataset, stats["total_pooled_expressions"], stats["total_raw_archive_expressions"],
        stats["matched_pooled_expressions"], stats["unmatched_pooled_expressions"],
        stats["unused_archive_expressions"], stats["duplicate_key_chunks_before_disambiguation"],
    )
    return ContextWindowResolution(
        source_dataset=source_dataset,
        context_window_by_pooled_index=context_window_by_pooled_index,
        occurrence_key_by_pooled_index=occurrence_key_by_pooled_index,
        archive_items=archive_items,
        archive_index_by_pooled_index=archive_index_by_pooled_index,
        stats=stats,
    )


# ---------------------------------------------------------------------------
# Per-source centroids (mode-independent -- see analyze_global_structure.py)
# ---------------------------------------------------------------------------

def source_dataset_indices(
    points: list[dict], source_dataset: str,
    epistemic_status_filter: tuple[str, ...] | None = None,
) -> list[int]:
    """`epistemic_status_filter`, when given, drops any point whose
    `epistemic_status` is populated and not in the given set. A point
    whose `epistemic_status` is null (every source_dataset except
    literature/miviludes/interviews) always passes regardless of the
    filter -- this is what makes it safe to pass this filter into
    per_source_centroids_and_dispersion below without emptying out the
    reference/emergent/criteria centroids, which have no notion of
    epistemic status at all."""
    idxs = [i for i, p in enumerate(points) if p["source_dataset"] == source_dataset]
    if epistemic_status_filter is None:
        return idxs
    return [i for i in idxs if points[i].get("epistemic_status") in (None, *epistemic_status_filter)]


def epistemic_status_subgroup_indices(points: list[dict], corpus: str) -> dict[str, list[int]]:
    """For one expression corpus (`corpus` in EXPRESSION_CORPORA), groups
    that corpus's own point indices by their actual `epistemic_status`
    value -- DYNAMICALLY DISCOVERED from the data actually present for
    this corpus, never a hardcoded set of 5. MIVILUDES currently has only
    asserted/contested/speculative; interviews only
    asserted/contested/negated/speculative -- both are returned exactly
    as-is, never padded out to literature's full 5.

    Insertion order = first-encountered order in `points` (i.e. the
    embedding_space.jsonl file order), so this is deterministic run to
    run without being alphabetical.

    Fails loudly (ValueError) if the corpus ends up with zero subgroups
    (zero points for this corpus, or every one of its points has a null
    epistemic_status) -- shouldn't happen for any of the three expression
    corpora, whose epistemic_status is always populated.

    Returns {epistemic_status: [indices into points]}.
    """
    subgroups: dict[str, list[int]] = {}
    for i in source_dataset_indices(points, corpus):
        status = points[i].get("epistemic_status")
        if status is None:
            continue
        subgroups.setdefault(status, []).append(i)
    if not subgroups:
        raise ValueError(
            f"epistemic_status_subgroup_indices({corpus!r}): no subgroups found -- "
            "either this corpus has zero points, or every point's epistemic_status is null."
        )
    return subgroups


def centroid_and_dispersion_for_indices(vectors: np.ndarray, indices: list[int]) -> dict:
    """Centroid + mean/median/p10/p90 dispersion (Euclidean distance to
    that centroid) over one arbitrary index list into `vectors` -- the
    per-source loop body originally inlined in
    per_source_centroids_and_dispersion below, factored out here so any
    OTHER grouping (an epistemic-status subgroup, a borderline pool, ...)
    reuses the exact same computation rather than a parallel copy.
    `indices` need not be a whole source_dataset -- it's an arbitrary
    index list into the SAME `vectors` array the caller already has.

    Returns {"centroid": np.ndarray(d,), "dispersion": float,
    "dispersion_median": float, "dispersion_p10": float,
    "dispersion_p90": float, "n": int}.
    """
    vecs = vectors[indices]
    centroid = vecs.mean(axis=0)
    distances = np.linalg.norm(vecs - centroid, axis=1)
    return {
        "centroid": centroid,
        "dispersion": float(distances.mean()),
        "dispersion_median": float(np.median(distances)),
        "dispersion_p10": float(np.percentile(distances, 10)),
        "dispersion_p90": float(np.percentile(distances, 90)),
        "n": len(indices),
    }


def per_source_centroids_and_dispersion(
    shared_space: SharedSpace,
    epistemic_status_filter: tuple[str, ...] | None = None,
) -> dict[str, dict]:
    """Every source_dataset's own centroid + dispersion (mean Euclidean
    distance to own centroid), computed independently, unconditional on
    any mode -- a corpus's own centroid isn't an imbalance-sensitive
    statistic, only combined/cross-corpus comparisons are.

    `epistemic_status_filter` (default None = every point, current
    behaviour unchanged) restricts the three expression corpora's own
    points before computing their centroid/dispersion -- see
    source_dataset_indices. Reference/emergent/criteria sources are
    unaffected by any filter value, since their epistemic_status is
    always null.

    `dispersion` (the mean distance to centroid) is kept as-is for backward
    compatibility with every existing caller. `dispersion_median`,
    `dispersion_p10`, `dispersion_p90` are an additive extension computed
    from the same per-point distance-to-centroid array -- a distribution's
    mean can legitimately fall outside its own [p10, p90] range under
    skew, so callers must not assume `dispersion` is bounded by those two.

    Returns {source_dataset: {"centroid": np.ndarray(394,), "dispersion": float,
    "dispersion_median": float, "dispersion_p10": float, "dispersion_p90": float,
    "n": int}}.
    """
    result = {}
    for source in ALL_SOURCE_DATASETS:
        idxs = source_dataset_indices(shared_space.points, source, epistemic_status_filter)
        result[source] = centroid_and_dispersion_for_indices(shared_space.vectors, idxs)
    return result


# ---------------------------------------------------------------------------
# Reference-set size control (concept_backbone/structural_concepts/
# conceptnet_concepts are 3,000/1,500/195 points -- a raw nearest-term
# comparison across them is biased toward whichever set is largest, purely
# by candidate-pool size, before any semantic content is considered)
# ---------------------------------------------------------------------------

def equal_size_reference_draw(
    reference_pools: dict[str, np.ndarray],
    seed: int,
    n_reference: int | None = None,
) -> dict[str, np.ndarray]:
    """One repetition of the equal-size reference-set draw -- the
    reference-vocabulary analogue of equal_n_expression_draw. n_reference
    is computed at runtime as min(len(pool) for pool in reference_pools)
    when not given explicitly -- never hardcoded, so this stays correct if
    a reference set's size changes later (currently min(3000, 1500, 195) =
    195). Each reference set's vectors are sampled down to n_reference via
    simple_random_sample (reference terms have no document_id to stratify
    by, unlike the expression corpora); whichever set already realizes
    n_reference is used in full, unsampled."""
    if n_reference is None:
        n_reference = min(len(vecs) for vecs in reference_pools.values())
    draw: dict[str, np.ndarray] = {}
    for name, vecs in reference_pools.items():
        idxs = simple_random_sample(list(range(len(vecs))), n_reference, seed)
        draw[name] = vecs[idxs]
    return draw


def equal_size_reference_comparison(
    queries: dict[str, np.ndarray],
    reference_pools: dict[str, np.ndarray],
    k: int = 10,
    seed: int = DEFAULT_SEED,
    reps: int = DEFAULT_BOOTSTRAP_REPS,
) -> dict[str, dict[str, dict[str, dict]]]:
    """Equal-size-controlled "how close is query X to reference-vocabulary
    Y" -- the reference-vocabulary analogue of analyze_cluster_structure's
    equal_n_expression bootstrap loop. All queries in `queries` are scored
    against the *same* per-repetition sampled subset (one resample per
    repetition, not per query), amortizing sampling cost across however
    many queries a caller passes (17 criteria, 25 prototypes, thousands of
    ranked entities).

    Returns {query_key: {reference_dataset: {
        "nearest_k_distance_mean": bootstrap_summary([...]),
        "nearest_k_distance_median": bootstrap_summary([...]),
    }}} -- bootstrap_summary reused verbatim for the mean/std/95%-CI
    aggregation, never reimplemented.
    """
    n_reference = min(len(vecs) for vecs in reference_pools.values())
    mean_reps: dict[str, dict[str, list[float]]] = {
        qk: {rd: [] for rd in reference_pools} for qk in queries
    }
    median_reps: dict[str, dict[str, list[float]]] = {
        qk: {rd: [] for rd in reference_pools} for qk in queries
    }

    for rep in range(reps):
        draw = equal_size_reference_draw(reference_pools, seed=seed + rep, n_reference=n_reference)
        for query_key, query_vector in queries.items():
            for reference_dataset, sampled_vectors in draw.items():
                distances = euclidean_distances(query_vector, sampled_vectors)
                nearest_k = np.sort(distances)[:k]
                mean_reps[query_key][reference_dataset].append(float(nearest_k.mean()))
                median_reps[query_key][reference_dataset].append(float(np.median(nearest_k)))

    result: dict[str, dict[str, dict[str, dict]]] = {}
    for query_key in queries:
        result[query_key] = {}
        for reference_dataset in reference_pools:
            result[query_key][reference_dataset] = {
                "n_reference": n_reference,
                "k": k,
                "nearest_k_distance_mean": bootstrap_summary(mean_reps[query_key][reference_dataset]),
                "nearest_k_distance_median": bootstrap_summary(median_reps[query_key][reference_dataset]),
            }
    return result


# ---------------------------------------------------------------------------
# Expression-corpus pool + the four modes
# ---------------------------------------------------------------------------

def expression_pool_indices(
    points: list[dict],
    epistemic_status_filter: tuple[str, ...] | None = None,
) -> dict[str, list[int]]:
    """The three expression corpora only -- literature/miviludes/interviews.
    Deliberately excludes miviludes_criteria (a fixed reference list, not a
    corpus expression) and all reference/emergent points. This is the pool
    analyze_cluster_structure.py's k-NN/silhouette and
    analyze_criterion_neighbours.py's candidate search both require.

    `epistemic_status_filter` (default None = unchanged behaviour) is
    passed straight through to source_dataset_indices for each corpus."""
    return {c: source_dataset_indices(points, c, epistemic_status_filter) for c in EXPRESSION_CORPORA}


def stratified_sample_by_document(points: list[dict], indices: list[int], n: int, seed: int) -> list[int]:
    """Proportional-by-document stratified sample of n indices from
    `indices` (all pointing to the same source_dataset), grouped by
    key_document_id -- same allocation algorithm as
    balanced_analysis.write_balanced_literature_sample, generalized here
    for repeated in-memory draws (equal_n_expression's bootstrap) rather
    than a single file write."""
    if n >= len(indices):
        return list(indices)

    by_document: dict[str, list[int]] = defaultdict(list)
    for i in indices:
        by_document[key_document_id(points[i]["key"])].append(i)

    total = len(indices)
    documents = sorted(by_document.items(), key=lambda item: -len(item[1]))
    allocations = {doc_id: int(len(idxs) / total * n) for doc_id, idxs in documents}
    shortfall = n - sum(allocations.values())
    for doc_id, _idxs in documents[:shortfall]:
        allocations[doc_id] += 1

    rng = random.Random(seed)
    sampled: list[int] = []
    for doc_id, idxs in documents:
        k = min(allocations[doc_id], len(idxs))
        sampled.extend(rng.sample(idxs, k))
    return sampled


def simple_random_sample(indices: list[int], n: int, seed: int) -> list[int]:
    if n >= len(indices):
        return list(indices)
    return random.Random(seed).sample(indices, n)


def equal_n_expression_draw(
    points: list[dict], pool: dict[str, list[int]], seed: int,
) -> dict[str, list[int]]:
    """One repetition of the equal_n_expression mode. n = size of the
    smallest expression corpus at runtime -- never hardcoded, so this stays
    correct if corpus sizes shift on a future rebuild. Literature is
    sampled stratified-by-document down to n; MIVILUDES is sampled simple
    random down to n; whichever corpus actually realizes n (interviews,
    currently) is used in full, unsampled -- generic over which corpus
    happens to be smallest, not hardcoded to "interviews"."""
    n = min(len(idxs) for idxs in pool.values())
    draw: dict[str, list[int]] = {}
    for corpus, idxs in pool.items():
        if len(idxs) == n:
            draw[corpus] = list(idxs)
        elif corpus == "literature":
            draw[corpus] = stratified_sample_by_document(points, idxs, n, seed)
        else:
            draw[corpus] = simple_random_sample(idxs, n, seed)
    return draw


def equal_n_dispersion(
    shared_space: SharedSpace, seed: int, reps: int,
    epistemic_status_filter: tuple[str, ...] | None = None,
) -> dict[str, dict]:
    """`equal_n_dispersion` -- semantic spread of each of the three
    expression corpora under equal-sized sampling, for a fair cross-corpus
    comparison of internal breadth (literature otherwise dwarfs the other
    two). Conceptually and terminologically distinct from
    per_source_centroids_and_dispersion's `full_source_dispersion` (no
    resampling there) -- never blend the two into one number.

    For each of `reps` repetitions: draw an equal-sized sample via
    equal_n_expression_draw, compute each corpus's dispersion (mean
    Euclidean distance to *that rep's own* equal-sized-sample centroid,
    not the full-source centroid), then bootstrap_summary across reps.
    Reuses equal_n_expression_draw/bootstrap_summary exactly -- no new
    sampling logic.

    `epistemic_status_filter` (default None) is applied once, up front, to
    the expression pool before any draw -- so the equal-n draw size for a
    filtered slice is the smallest *filtered* corpus size, never assumed
    equal to the unfiltered draw size (the same pool-size auto-recompute
    equal_n_expression_draw already provides).

    Returns {corpus: {**bootstrap_summary(...), "n_per_draw": int}} for
    corpus in EXPRESSION_CORPORA. `n_per_draw` is the same every rep (the
    filtered pool doesn't change across reps), so it's reported once per
    corpus rather than per rep.
    """
    pool = expression_pool_indices(shared_space.points, epistemic_status_filter)
    per_corpus_reps: dict[str, list[float]] = {c: [] for c in EXPRESSION_CORPORA}
    n_per_draw: int | None = None
    for rep in range(reps):
        draw = equal_n_expression_draw(shared_space.points, pool, seed + rep)
        for corpus in EXPRESSION_CORPORA:
            idxs = draw[corpus]
            if n_per_draw is None:
                n_per_draw = len(idxs)
            vecs = shared_space.vectors[idxs]
            centroid = vecs.mean(axis=0)
            dispersion = float(np.linalg.norm(vecs - centroid, axis=1).mean())
            per_corpus_reps[corpus].append(dispersion)
    return {
        corpus: {**bootstrap_summary(values), "n_per_draw": n_per_draw}
        for corpus, values in per_corpus_reps.items()
    }


def normalized_separation(
    per_source: dict, sources: tuple[str, ...] = EXPRESSION_CORPORA,
) -> list[dict]:
    """`normalized_separation_ratio` -- centroid separation relative to
    within-source spread: R_{A,B} = d(centroid_A, centroid_B) / mean(D_A, D_B),
    where `d` and both `D` values (mean dispersion) come from the SAME
    `per_source` dict passed in (i.e. the same full-source slice/epistemic
    status) -- never mix a full-source centroid distance with an
    equal-n-sampled dispersion, or vice versa. This is the interpretive
    baseline for "is a centroid distance of, say, 7.18 large or small
    relative to these two sources' normal semantic breadth" -- not a
    significance test.

    Only the full-source version is built here (this is a first pass); an
    equal-n version of this ratio would need centroid distance AND
    dispersion recomputed from the same equal-n draw within each bootstrap
    rep, a distinct, more expensive deliverable, not built in this pass.

    For each unordered pair of `sources`. Returns a list of dicts:
    {"source_a", "source_b", "centroid_distance", "source_a_mean_dispersion",
    "source_b_mean_dispersion", "source_a_median_dispersion",
    "source_b_median_dispersion", "normalized_separation_ratio"}.
    """
    rows = []
    sources = list(sources)
    for i, source_a in enumerate(sources):
        for source_b in sources[i + 1:]:
            a, b = per_source[source_a], per_source[source_b]
            centroid_distance = float(np.linalg.norm(a["centroid"] - b["centroid"]))
            mean_dispersion = (a["dispersion"] + b["dispersion"]) / 2
            rows.append({
                "source_a": source_a, "source_b": source_b,
                "centroid_distance": centroid_distance,
                "source_a_mean_dispersion": a["dispersion"], "source_b_mean_dispersion": b["dispersion"],
                "source_a_median_dispersion": a["dispersion_median"], "source_b_median_dispersion": b["dispersion_median"],
                "normalized_separation_ratio": centroid_distance / mean_dispersion,
            })
    return rows


def corpus_vectors_and_points(
    shared_space: SharedSpace, mode: str, seed: int = DEFAULT_SEED,
    epistemic_status_filter: tuple[str, ...] | None = None,
) -> dict[str, tuple[list[dict], np.ndarray]]:
    """Point-level resolution of a mode for the three expression corpora.
    Returns {corpus_name: (points, vectors)}.

      - "full": every point, unsampled (the true full population).
      - "reduced_literature": literature_balanced_sample.jsonl's 2,500
        points + all MIVILUDES + all interviews.
      - "equal_n_expression": one fresh draw at n = smallest corpus size
        (equal_n_expression_draw) -- pass a different `seed` per repetition
        to bootstrap.

    "equal_weight" has no point-level form -- see combined_expression_reference
    or balanced_analysis.weighted_centroid()/per_corpus_centroids() instead.

    `epistemic_status_filter` (default None = unchanged behaviour) applies
    to every mode, including "reduced_literature" -- that mode's literature
    subset comes from a separate pre-built file
    (literature_balanced_sample.jsonl), not from expression_pool_indices,
    so it needs its own explicit filtering step here rather than inheriting
    the filter for free; without this, a filtered call would silently leave
    "reduced_literature"'s literature slice unfiltered while its
    miviludes/interviews slices were filtered.
    """
    if mode not in ("full", "reduced_literature", "equal_n_expression"):
        raise ValueError(f"No point-level resolution for mode={mode!r}")

    pool = expression_pool_indices(shared_space.points, epistemic_status_filter)

    if mode == "full":
        return {
            c: ([shared_space.points[i] for i in idxs], shared_space.vectors[idxs])
            for c, idxs in pool.items()
        }

    if mode == "reduced_literature":
        # Sibling of whichever embedding_space.jsonl was actually loaded, not
        # a hardcoded v1 path -- so a v2 (or later) shared space uses its own
        # literature_balanced_sample.jsonl (see balanced_analysis.py
        # --input/--output), never v1's, and the two are never mixed.
        balanced_sample_path = shared_space.input_path.parent / "literature_balanced_sample.jsonl"
        if not balanced_sample_path.exists():
            raise SystemExit(
                f"Missing: {balanced_sample_path} -- 'reduced_literature' mode needs a "
                "literature-balanced sample built for this shared space. Run: python -m "
                f"thesis_corpus.balanced_analysis --input {shared_space.input_path} "
                f"--output {balanced_sample_path} --sample-size <N> (v1 used 2500 out of "
                "35,621 literature points, ~7%; scale proportionally to this space's own "
                "literature point count rather than reusing 2500 unchanged)."
            )
        lit_points, lit_vectors = load_reduced_literature_points(balanced_sample_path)
        if epistemic_status_filter is not None:
            keep = [
                i for i, p in enumerate(lit_points)
                if p.get("epistemic_status") in (None, *epistemic_status_filter)
            ]
            lit_points = [lit_points[i] for i in keep]
            lit_vectors = lit_vectors[keep]
        result: dict[str, tuple[list[dict], np.ndarray]] = {"literature": (lit_points, lit_vectors)}
        for c in ("miviludes", "interviews"):
            idxs = pool[c]
            result[c] = ([shared_space.points[i] for i in idxs], shared_space.vectors[idxs])
        return result

    draw = equal_n_expression_draw(shared_space.points, pool, seed)
    return {
        c: ([shared_space.points[i] for i in idxs], shared_space.vectors[idxs])
        for c, idxs in draw.items()
    }


def combined_expression_reference(shared_space: SharedSpace, mode: str) -> np.ndarray:
    """The single "expression corpora as a whole" reference point, used
    only where a statistic genuinely needs one (never for point-level
    statistics like k-NN/silhouette -- those use corpus_vectors_and_points
    directly).

      - "full": the plain pooled mean over every literature+MIVILUDES+interview point.
      - "reduced_literature": the plain pooled mean using the
        reduced-literature subset in place of full literature.
      - "equal_weight": literature/MIVILUDES/interview centroids computed
        separately, then averaged with equal 1/3 weight
        (balanced_analysis.weighted_centroid(), reused directly).
    """
    from thesis_corpus.balanced_analysis import _raw_unweighted_mean, weighted_centroid

    if mode == "equal_weight":
        return weighted_centroid(shared_space.input_path)
    if mode == "full":
        return _raw_unweighted_mean(shared_space.input_path, EXPRESSION_CORPORA)
    if mode == "reduced_literature":
        by_corpus = corpus_vectors_and_points(shared_space, "reduced_literature")
        all_vectors = np.concatenate([v for _, v in by_corpus.values()], axis=0)
        return all_vectors.mean(axis=0)
    raise ValueError(f"No combined-expression reference for mode={mode!r}")


def bootstrap_summary(values: list[float]) -> dict:
    """mean/std/95% empirical interval (2.5th/97.5th percentile) across a
    list of per-repetition statistic values -- the standard summary every
    equal_n_expression-mode statistic in this toolkit reports. Never a
    single-draw point estimate."""
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n_reps": int(len(arr)),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "ci95_low": float(np.percentile(arr, 2.5)),
        "ci95_high": float(np.percentile(arr, 97.5)),
    }


# ---------------------------------------------------------------------------
# Distance helpers -- Euclidean primary, cosine sensitivity
# ---------------------------------------------------------------------------

def euclidean_distances(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    return np.linalg.norm(candidates - query, axis=1)


def cosine_similarities(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    query_norm = query / np.linalg.norm(query)
    candidate_norms = candidates / np.linalg.norm(candidates, axis=1, keepdims=True)
    return candidate_norms @ query_norm


# for detecting a query point's own vector re-appearing in the candidate
# pool it's being compared against (e.g. an interview prototype that also
# independently survived the ordinary pooling filter) -- both are derived
# from the same raw vector through the same persisted transform, so they'd
# be exact duplicates, not a genuine nearest neighbour.
SAME_VECTOR_EPSILON = 1e-9


def ranked_points(
    query: np.ndarray, candidate_points: list[dict], candidate_vectors: np.ndarray,
    exclude_self: bool = False,
) -> list[dict]:
    """Every candidate ranked ascending by Euclidean distance to an
    arbitrary query vector (typically a centroid) -- no `k` cutoff. The
    shared base for both nearest_points below (nearest_points(...) ==
    ranked_points(...)[:k]) and for a "farthest" retrieval, which is just
    the reverse of the same ranking (list(reversed(ranked_points(...)))[:k])
    -- there is no separate "farthest_points" implementation, since
    reversing this one ascending ranking is exactly correct and avoids a
    second, easy-to-desync sort.

    `exclude_self`, when set, skips any candidate numerically identical to
    `query` (see SAME_VECTOR_EPSILON above) rather than letting a query's
    own point count as its own nearest (or farthest) neighbour.

    Returns a list of len(candidate_points) (minus any excluded self-match)
    dicts, ranked nearest-first:
    {"rank", "key", "label", "euclidean_distance", "cosine_similarity"}.
    """
    euclidean = euclidean_distances(query, candidate_vectors)
    cosine = cosine_similarities(query, candidate_vectors)
    order = np.argsort(euclidean)
    results: list[dict] = []
    for i in order:
        if exclude_self and euclidean[i] < SAME_VECTOR_EPSILON:
            continue
        results.append({
            "rank": len(results) + 1,
            "key": candidate_points[i]["key"],
            "label": candidate_points[i]["label"],
            "euclidean_distance": float(euclidean[i]),
            "cosine_similarity": float(cosine[i]),
        })
    return results


def nearest_points(
    query: np.ndarray, candidate_points: list[dict], candidate_vectors: np.ndarray,
    k: int, exclude_self: bool = False,
) -> list[dict]:
    """The k nearest actual points to an arbitrary query vector (typically
    a centroid) -- the one shared implementation for a pattern that used
    to be duplicated (analyze_global_structure.nearest_terms,
    analyze_initial_exemplars.nearest). A thin wrapper around
    ranked_points, kept as its own function since "give me the k nearest"
    is still by far the most common call shape in this toolkit.

    Returns a list of up to k dicts, ranked nearest-first:
    {"rank", "key", "label", "euclidean_distance", "cosine_similarity"}.
    """
    return ranked_points(query, candidate_points, candidate_vectors, exclude_self=exclude_self)[:k]


def borderline_ranking(
    points: list[dict], vectors: np.ndarray, pool_indices: list[int],
    centroid_a: np.ndarray, centroid_b: np.ndarray,
) -> list[dict]:
    """Ranks every point in `pool_indices` -- the UNION POOL of exactly the
    two groups being contrasted, never a broader or unrelated pool; it is
    the caller's job to pass exactly that pair's own points -- by how
    close it sits to equidistant between `centroid_a` and `centroid_b`.

    `ambiguity_ratio = abs(distance_to_a - distance_to_b) / mean(distance_to_a, distance_to_b)`:
    0 = exactly equidistant (maximally borderline), larger = more
    decisively closer to one side. Ranked ASCENDING by ambiguity_ratio, so
    the most borderline point is first.

    `min_distance_to_either_centroid` is a DIAGNOSTIC-ONLY column, never
    used to filter here -- a point that's borderline by ratio while far
    from both groups' typical range stays visible in the output via this
    column, rather than being silently dropped.

    Returns one dict per pool index, ranked ascending by ambiguity_ratio:
    {"rank", "pool_index" (the index into `points`/`vectors` this row
    refers to -- callers use this to pull any further metadata), "key",
    "label", "distance_to_a", "distance_to_b", "ambiguity_ratio",
    "nominally_closer_to" ("a" or "b"), "min_distance_to_either_centroid"}.
    """
    pool_vectors = vectors[pool_indices]
    distance_to_a = euclidean_distances(centroid_a, pool_vectors)
    distance_to_b = euclidean_distances(centroid_b, pool_vectors)
    mean_distance = (distance_to_a + distance_to_b) / 2
    ambiguity_ratio = np.abs(distance_to_a - distance_to_b) / mean_distance
    min_distance = np.minimum(distance_to_a, distance_to_b)

    order = np.argsort(ambiguity_ratio)
    rows: list[dict] = []
    for rank, local_pos in enumerate(order):
        pool_index = pool_indices[local_pos]
        rows.append({
            "rank": rank + 1,
            "pool_index": pool_index,
            "key": points[pool_index]["key"],
            "label": points[pool_index]["label"],
            "distance_to_a": float(distance_to_a[local_pos]),
            "distance_to_b": float(distance_to_b[local_pos]),
            "ambiguity_ratio": float(ambiguity_ratio[local_pos]),
            "nominally_closer_to": "a" if distance_to_a[local_pos] <= distance_to_b[local_pos] else "b",
            "min_distance_to_either_centroid": float(min_distance[local_pos]),
        })
    return rows


# ---------------------------------------------------------------------------
# 2-D UMAP / PCA projections. fit_and_save_umap was originally
# analyze_cluster_structure.py's own private helper (the only place
# umap.UMAP(...).fit_transform was called in the toolkit) -- moved here
# once generate_focused_projections.py needed the identical fitting/
# manifest logic for ~23 additional small populations, so both modules
# share one implementation rather than maintaining two copies.
#
# Compatibility with analyze_cluster_structure.py's pre-existing 4
# populations (overview/expression_sampled/equal_n_diagnostic/
# with_interview_prototypes) is preserved exactly: every field that
# function's manifest already had keeps its name and value type
# (`n_points`, `input_sha256`, `seed`, `umap_params.n_neighbors`, etc.);
# everything below is additive (`n_points_fit`/`n_points_overlay`/
# `n_points_rendered`, `composition`, `shortages`,
# `umap_params.n_neighbors_requested`/`n_neighbors_used`). Calling this
# with no overlay (as analyze_cluster_structure.py's own 4 populations
# do) reproduces its original output byte-for-byte apart from these
# additive keys.
# ---------------------------------------------------------------------------

def _population_role(point: dict) -> str:
    """Coordinate-row grouping key for composition/rendering -- prefers an
    explicit population_role a caller already tagged the point with
    (generate_focused_projections.py's curated populations always do
    this), falling back to source_dataset for populations that don't
    (analyze_cluster_structure.py's own 4, which have no population_role
    concept)."""
    return point.get("population_role") or point.get("source_dataset", "unknown")


def fit_and_save_umap(
    out_dir: Path, population_name: str,
    member_points: list[dict], member_vectors: np.ndarray,
    n_neighbors: int, min_dist: float, seed: int,
    overlay_points: list[dict] | None = None,
    overlay_vectors: np.ndarray | None = None,
    shortages: dict | None = None,
) -> Path:
    """Fits UMAP on member_vectors only. If overlay_vectors is given (e.g.
    a corpus centroid -- not a real corpus point, never part of any fit),
    it's embedded into the SAME fitted model via reducer.transform() after
    fitting, so an overlay can never perturb the member embedding.
    n_neighbors is clamped to min(n_neighbors, n_points_fit - 1) when a
    population is smaller than requested (several focused populations,
    e.g. a single criterion's 46-point neighbourhood, are far smaller than
    analyze_cluster_structure.py's whole-space-scale populations); this
    clamp only changes how many neighbours UMAP's algorithm uses
    internally -- it never removes a member/overlay point or changes
    n_points_fit/n_points_overlay/n_points_rendered. Fails loudly if fewer
    than 3 points are supplied to fit on (umap-learn isn't meaningful
    below that). `shortages` (default {}) records, verbatim from the
    caller, any source pool that had fewer candidates than requested when
    the population was constructed -- entirely separate from the
    n_neighbors clamp; never inferred here.

    Returns the manifest path."""
    n_points_fit = len(member_points)
    if n_points_fit < 3:
        raise ValueError(f"'{population_name}': need >=3 points to fit UMAP, got {n_points_fit}")
    n_neighbors_requested = n_neighbors
    n_neighbors_used = min(n_neighbors_requested, n_points_fit - 1)

    input_sha256 = sha256_array(member_vectors)
    logger.info(
        "UMAP fit '%s' (n_points_fit=%d, n_neighbors=%d->%d, min_dist=%.2f)...",
        population_name, n_points_fit, n_neighbors_requested, n_neighbors_used, min_dist,
    )
    reducer = umap.UMAP(
        n_components=2, n_neighbors=n_neighbors_used, min_dist=min_dist,
        random_state=seed, metric="euclidean",
    )
    member_coords = reducer.fit_transform(member_vectors)

    overlay_points = overlay_points or []
    if overlay_vectors is not None and len(overlay_vectors):
        overlay_coords = reducer.transform(overlay_vectors)
    else:
        overlay_coords = np.zeros((0, 2))
    n_points_overlay = len(overlay_points)
    n_points_rendered = n_points_fit + n_points_overlay

    stem = f"umap_{population_name}_n{n_neighbors_used}_d{min_dist}"
    coords_path = out_dir / f"{stem}.jsonl"
    composition: dict[str, int] = defaultdict(int)
    with open(coords_path, "w", encoding="utf-8") as f:
        for p, coord in zip(member_points, member_coords):
            role = _population_role(p)
            composition[role] += 1
            row = {
                "key": p["key"], "source_dataset": p["source_dataset"],
                "point_role": p.get("point_role"), "label": p.get("label"),
                "population_role": p.get("population_role"),
                "point_kind": "member",
                "umap_2d": coord.tolist(),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        for p, coord in zip(overlay_points, overlay_coords):
            role = _population_role(p)
            composition[role] += 1
            row = {
                "key": p["key"], "source_dataset": p.get("source_dataset"),
                "point_role": p.get("point_role"), "label": p.get("label"),
                "population_role": p.get("population_role"),
                "point_kind": "centroid_overlay",
                "umap_2d": coord.tolist(),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    output_sha256 = sha256_file(coords_path)

    manifest = {
        "population": population_name,
        "n_points": n_points_rendered,  # kept for backward compatibility with the pre-existing 4 populations
        "n_points_fit": n_points_fit,
        "n_points_overlay": n_points_overlay,
        "n_points_rendered": n_points_rendered,
        "composition": dict(composition),
        "shortages": shortages or {},
        "input_sha256": input_sha256,
        "seed": seed,
        "umap_params": {
            "n_neighbors": n_neighbors_used,  # kept for backward compatibility; equals n_neighbors_used
            "n_neighbors_requested": n_neighbors_requested,
            "n_neighbors_used": n_neighbors_used,
            "min_dist": min_dist, "n_components": 2, "metric": "euclidean",
        },
        "umap_learn_version": umap.__version__,
        "output_coords_path": coords_path.name,
        "output_sha256": output_sha256,
    }
    manifest_path = out_dir / f"{stem}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def pca_2d_slice(vectors: np.ndarray) -> np.ndarray:
    """The first 2 columns of the already-PCA'd 394-D shared_space_vector
    -- not a fresh PCA refit. Same convention visualize_3d.py documents
    for its own 3-D PCA output: refitting a fresh PCA on a subset would
    return the same leading axes up to sign for no benefit, while giving
    every population an incomparable rotation relative to every other --
    a fixed global basis (this slice) keeps all PCA-2D populations
    directly comparable to each other."""
    return vectors[:, :2]


def save_pca_projection(
    out_dir: Path, population_name: str,
    member_points: list[dict], member_vectors: np.ndarray,
    overlay_points: list[dict] | None = None,
    overlay_vectors: np.ndarray | None = None,
    shortages: dict | None = None,
) -> Path:
    """PCA-2D counterpart to fit_and_save_umap -- same coordinate-row/
    manifest shape (member/overlay, point_kind, population_role,
    n_points_fit/n_points_overlay/n_points_rendered, composition,
    shortages), but deterministic (no seed, no fitted model, no package
    version) since it's a pure slice, not a fit. An overlay's PCA-2D
    coordinate is just its own vector's first 2 columns -- no transform()
    step needed, unlike UMAP."""
    n_points_fit = len(member_points)
    member_coords = pca_2d_slice(member_vectors)
    overlay_points = overlay_points or []
    overlay_coords = pca_2d_slice(overlay_vectors) if overlay_vectors is not None and len(overlay_vectors) else np.zeros((0, 2))
    n_points_overlay = len(overlay_points)
    n_points_rendered = n_points_fit + n_points_overlay

    stem = f"pca_{population_name}"
    coords_path = out_dir / f"{stem}.jsonl"
    composition: dict[str, int] = defaultdict(int)
    with open(coords_path, "w", encoding="utf-8") as f:
        for p, coord in zip(member_points, member_coords):
            composition[_population_role(p)] += 1
            f.write(json.dumps({
                "key": p["key"], "source_dataset": p["source_dataset"],
                "point_role": p.get("point_role"), "label": p.get("label"),
                "population_role": p.get("population_role"),
                "point_kind": "member", "pca_2d": coord.tolist(),
            }, ensure_ascii=False) + "\n")
        for p, coord in zip(overlay_points, overlay_coords):
            composition[_population_role(p)] += 1
            f.write(json.dumps({
                "key": p["key"], "source_dataset": p.get("source_dataset"),
                "point_role": p.get("point_role"), "label": p.get("label"),
                "population_role": p.get("population_role"),
                "point_kind": "centroid_overlay", "pca_2d": coord.tolist(),
            }, ensure_ascii=False) + "\n")
    output_sha256 = sha256_file(coords_path)

    manifest = {
        "population": population_name,
        "method": "pca_2d_slice",
        "n_points_fit": n_points_fit,
        "n_points_overlay": n_points_overlay,
        "n_points_rendered": n_points_rendered,
        "composition": dict(composition),
        "shortages": shortages or {},
        "source_columns": [0, 1],
        "input_sha256": sha256_array(member_vectors),
        "output_coords_path": coords_path.name,
        "output_sha256": output_sha256,
        "note": (
            "first 2 columns of the already-PCA'd 394-D shared_space_vector; "
            "not a fresh PCA refit -- same convention as visualize_3d.py's 3-D PCA slice."
        ),
    }
    manifest_path = out_dir / f"{stem}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


# ---------------------------------------------------------------------------
# Run directories, config.json, RUN_MANIFEST.json
# ---------------------------------------------------------------------------

def git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=CORPUS_DIR,
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def get_or_create_run_dir(run_id: str | None, analysis_root: Path = ANALYSIS_DIR) -> tuple[str, Path]:
    """Creates processed/analysis/<run-id>/ if it doesn't exist yet
    (default run_id: UTC timestamp `YYYYMMDD-HHMMSS`), or returns the
    existing directory for a reused run_id -- multiple analysis modules
    are expected to share one run-id, tied together by RUN_MANIFEST.json.
    Reusing a run-id is not itself an overwrite -- module_run_dir below is
    what actually refuses to clobber a module's own prior output."""
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = analysis_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


def existing_run_dir(run_id: str, analysis_root: Path = ANALYSIS_DIR) -> Path:
    """For generate_figures.py / generate_geometric_draft_report.py, which
    require an already-populated run and must never fall back to
    "latest" or create one."""
    run_dir = analysis_root / run_id
    if not run_dir.exists():
        raise SystemExit(
            f"No such run: {run_dir} -- this script reads a run the analysis "
            "modules already produced; pass --run-id of an existing run."
        )
    return run_dir


def module_run_dir(run_dir: Path, module_name: str) -> Path:
    """This module's own subdirectory within a run. Refuses to reuse a
    run-id where *this same module* already produced output -- runs are
    never overwritten -- but a different module adding its own
    subdirectory to the same run-id is exactly the intended workflow."""
    out = run_dir / module_name
    if out.exists() and any(out.iterdir()):
        raise SystemExit(
            f"{module_name} already has output in run {run_dir.name} ({out}) "
            "-- runs are never overwritten. Use a new --run-id to redo this module."
        )
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_module_config(module_dir: Path, **kwargs) -> None:
    (module_dir / "config.json").write_text(
        json.dumps(kwargs, indent=2, default=str), encoding="utf-8",
    )


def init_run_manifest(run_dir: Path, shared_space: SharedSpace, defaults: dict) -> Path:
    """Writes RUN_MANIFEST.json if it doesn't exist yet for this run (the
    first module run under this run-id creates it); a no-op otherwise.
    Every module calls this before doing anything else."""
    manifest_path = run_dir / "RUN_MANIFEST.json"
    if manifest_path.exists():
        return manifest_path
    manifest = {
        "run_id": run_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(shared_space.input_path),
        "input_sha256": shared_space.input_sha256,
        "git_commit": git_commit_hash(),
        "defaults": defaults,
        "modules": {},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def update_run_manifest(
    run_dir: Path, module_name: str, status: str,
    output_dir: Path | None = None, error: str | None = None,
) -> None:
    """status: "running" / "completed" / "failed"."""
    manifest_path = run_dir / "RUN_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = {"status": status, "updated_at": datetime.now(timezone.utc).isoformat()}
    if output_dir is not None:
        entry["output_dir"] = str(output_dir.relative_to(run_dir))
    if error is not None:
        entry["error"] = error
    manifest["modules"][module_name] = entry
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
