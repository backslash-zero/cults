"""Builds a separate, curated point-set of interview initial exemplars --
one manually-reviewed opening response per interview, projected into the
*existing* shared 394-D space via its persisted StandardScaler+PCA
transform, without touching embedding_space.jsonl, its point counts, or
any existing 3-D projection.

This exists because `build_shared_space.py`'s pooling-time length filter
(drops expressions under 5 words) removes exactly the kind of short,
complete, spontaneous answer the interview protocol's opening prompt is
designed to elicit ("AI cult", "Illuminati", "Tomato cult!") -- real
content, not noise, for this purpose specifically (see that filter's own
docstring for why it's still the right call for literature/MIVILUDES's
longer-form prose). Rather than relaxing that filter -- which would
reshuffle the whole pooled space, its PCA fit, and every downstream
count -- this script builds a second, independent layer: the same
394-D coordinate system, populated only with what a human reviewer
actually selected as each interview's initial exemplar, using the raw,
unfiltered embedding already computed for it (never re-extracted, never
re-embedded).

Pipeline:
  1. Read thesis/corpus/interviews/metadata/initial_exemplars.csv (must be
     fully reviewed -- every row "reviewed" or "unavailable", same gate as
     analyze_initial_exemplars.py).
  2. Resolve each "reviewed" row's `source_expression_key` against the raw,
     unfiltered interview archive (geometric_analysis_common.derive_archive_expression_keys)
     -- stable regardless of whether build_shared_space.py's filter would
     have dropped that item, since it never depends on pooling survival.
     Exact-match guard on `source_expression_label` vs. the archive's own
     text, same as analyze_initial_exemplars.py's join-drift check.
  3. Load build_shared_space.py's persisted `scaler`/`pca`
     (`processed/shared_space/pca_transform.joblib` -- run
     `python -m thesis_corpus.build_shared_space` first if missing) and
     project each resolved item's existing raw embedding_vector through
     the identical transform every pooled point already went through:
     `pca.transform(scaler.transform(vector))[:, :k]`.
  4. Write `processed/shared_space/interview_prototypes.jsonl`:
     `source_dataset="interview_prototypes"`, `point_role="prototype"` --
     a fourth point_role, distinct from expression/reference/emergent,
     since these points are neither a claim pooled from the ordinary
     corpus pipeline, a topic-neutral yardstick, nor a mentioned entity.

Never modifies embedding_space.jsonl, its 3-D projections, or
initial_exemplars.csv. Never calls Ollama or any embedding model -- every
vector already exists in the raw archive.

Usage (from thesis/corpus/):
    python -m thesis_corpus.build_interview_prototype_layer
"""
from __future__ import annotations

import argparse
import json
import logging
import re

import joblib
import numpy as np

from thesis_corpus import geometric_analysis_common as gac

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("thesis_corpus.build_interview_prototype_layer")

_WHITESPACE_RE = re.compile(r"\s+")
VALID_INITIAL_RESPONSE_FORMS = ("named_exemplar", "descriptive_characterisation", "mixed", "unclear")


def _normalize(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.strip())


def load_initial_exemplars(path) -> list[dict]:
    import csv
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validate_review_status(rows: list[dict], path) -> None:
    pending = [r["document_id"] for r in rows if r["review_status"] not in ("reviewed", "unavailable")]
    if pending:
        raise SystemExit(
            f"{len(pending)} row(s) in {path} are not yet reviewed "
            f"(review_status must be 'reviewed' or 'unavailable'): {pending}. "
            "This script refuses to build the prototype layer from an unreviewed candidate."
        )


def validate_initial_response_form(rows: list[dict], path) -> None:
    """Required for every "reviewed" row -- not merely defaulted. An
    empty/missing value fails loudly here rather than being silently
    treated as the intentional analytic category "unclear"."""
    bad = [
        r["document_id"] for r in rows
        if r["review_status"] == "reviewed"
        and r.get("initial_response_form") not in VALID_INITIAL_RESPONSE_FORMS
    ]
    if bad:
        raise SystemExit(
            f"{len(bad)} reviewed row(s) in {path} have a missing or invalid "
            f"initial_response_form (must be exactly one of {VALID_INITIAL_RESPONSE_FORMS}): "
            f"{bad}. An empty value is not the same as the analytic category 'unclear' -- "
            "set it explicitly."
        )


def resolve_row(row: dict, archive_items: list[dict], archive_keys: dict[str, int]) -> dict:
    key = row["source_expression_key"]
    if key not in archive_keys:
        raise SystemExit(
            f"[{row['document_id']}] virtual key {key!r} does not resolve against the raw "
            f"interview archive ({gac.INTERVIEWS_ARCHIVE_PATH}) -- the archive itself must have "
            "changed since this CSV was reviewed (re-extraction?), since this key space doesn't "
            "depend on any pooling filter. Needs manual re-verification."
        )
    index = archive_keys[key]
    archive_text = archive_items[index]["embedding_text"]
    if _normalize(archive_text) != _normalize(row["source_expression_label"]):
        raise SystemExit(
            f"[{row['document_id']}] virtual key {key!r} resolved to different text than "
            f"reviewed -- possible drift. CSV: {row['source_expression_label']!r}, "
            f"archive: {archive_text!r}. Needs manual re-verification, not a fuzzy match."
        )
    return archive_items[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=type(gac.INITIAL_EXEMPLARS_CSV_PATH), default=gac.INITIAL_EXEMPLARS_CSV_PATH,
                         help="initial_exemplars.csv -- the manually-reviewed exemplar selections. Always the "
                              "same file regardless of --transform/--output below: the reviewed quotes and the "
                              "raw archive they're resolved against (below) don't change between a v1 and a v2 "
                              "shared space, only which fitted PCA transform they get projected through.")
    parser.add_argument("--output", type=type(gac.INTERVIEW_PROTOTYPES_PATH), default=gac.INTERVIEW_PROTOTYPES_PATH)
    parser.add_argument("--transform", type=type(gac.PCA_TRANSFORM_PATH), default=gac.PCA_TRANSFORM_PATH,
                         help="Persisted StandardScaler+PCA to project through (default: v1's, at "
                              "processed/shared_space/pca_transform.joblib). Pass "
                              "processed/shared_space_v2/pca_transform.joblib to place these same "
                              "manually-reviewed exemplars into the v2 shared space instead -- this reuses the "
                              "existing human-reviewed selections and their already-computed embeddings "
                              "unchanged (see resolve_row below, still resolved against the frozen v1 raw "
                              "interview archive either way); only the coordinate system they land in changes.")
    parser.add_argument("--transform-metadata", type=type(gac.PCA_TRANSFORM_METADATA_PATH),
                         default=gac.PCA_TRANSFORM_METADATA_PATH,
                         help="Metadata (k, etc.) for --transform above -- must be the sidecar produced by the "
                              "same build_shared_space.py run that wrote --transform, never mixed with a "
                              "different run's transform.")
    args = parser.parse_args()

    if not args.transform.exists():
        raise SystemExit(
            f"No persisted transform at {args.transform} -- run "
            "`python -m thesis_corpus.build_shared_space` first (it now persists the fitted "
            "StandardScaler+PCA alongside embedding_space.jsonl)."
        )

    logger.info("Loading %s ...", args.input)
    rows = load_initial_exemplars(args.input)
    validate_review_status(rows, args.input)
    validate_initial_response_form(rows, args.input)
    reviewed = [r for r in rows if r["review_status"] == "reviewed"]
    logger.info("%d rows: %d reviewed, %d unavailable", len(rows), len(reviewed), len(rows) - len(reviewed))

    logger.info("Loading raw interview archive %s ...", gac.INTERVIEWS_ARCHIVE_PATH)
    archive_items = gac.load_raw_archive(gac.INTERVIEWS_ARCHIVE_PATH)
    archive_keys = gac.derive_archive_expression_keys(archive_items)

    logger.info("Loading persisted transform %s ...", args.transform)
    # joblib.load is pickle-based, but this file is generated locally by
    # build_shared_space.py in this same pipeline, never from an external or
    # untrusted source -- standard practice for persisting a fitted sklearn
    # transform.
    transform = joblib.load(args.transform)
    scaler, pca = transform["scaler"], transform["pca"]
    metadata = json.loads(args.transform_metadata.read_text(encoding="utf-8"))
    k = metadata["k"]

    resolved_archive_items = []
    for row in reviewed:
        resolved_archive_items.append((row, resolve_row(row, archive_items, archive_keys)))

    raw_vectors = np.array([item["embedding_vector"] for _row, item in resolved_archive_items], dtype=np.float64)
    projected = pca.transform(scaler.transform(raw_vectors))[:, :k]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for (row, _item), coord in zip(resolved_archive_items, projected):
            out = {
                "source_dataset": "interview_prototypes",
                "point_role": "prototype",
                "document_id": row["document_id"],
                "source_expression_key": row["source_expression_key"],
                "transcript_initial_exemplar_text": row["transcript_initial_exemplar_text"],
                "source_expression_label": row["source_expression_label"],
                "exemplar_type": row["exemplar_type"],
                "initial_response_form": row["initial_response_form"],
                "follow_up_examples": row.get("follow_up_examples", ""),
                "shared_space_vector": coord.tolist(),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"\nDone. {len(resolved_archive_items)} interview prototype points -> {args.output}")
    print(f"({len(rows) - len(reviewed)} unavailable rows excluded, not padded or substituted.)")


if __name__ == "__main__":
    main()
