"""Proposes a candidate "initial exemplar" (first participant claim in
response to the interview's opening prompt) for each of the 26 interviews,
for a human to review and correct.

**This is a proposal heuristic only, never ground truth.** Candidate =
first item in the raw archive's own transcript/chunk order with
`attribution == "participant"` and `claim_mode != "question_or_reflection"`
-- confirmed against ollama_client.py's actual claim_mode enum and real
data (e.g. "b1-aug05-1650"'s pooled response_rank==1 item is exactly the
case this excludes: the interviewer's own question).

Candidates are found by scanning `processed/interviews/criterion_expressions.jsonl`
(the raw, unfiltered archive -- never mutated), because the pooled
`embedding_space.jsonl` has already dropped some short/duplicate
expressions (build_shared_space.py's dedup/length filter), and that filter
disproportionately removes exactly the short, pithy first-association
exemplars this heuristic is looking for (found in practice: audit_free_listing_rank.py
shows only 11/26 interviews still have a response_rank==1 pooled point at
all). If the true first candidate didn't survive pooling, this script
falls back to the next candidate that did, and says so explicitly in
`notes` -- it never silently substitutes without a trace.

Writes thesis/corpus/interviews/metadata/initial_exemplars.csv with
`review_status="pending"` for every row. Every row must be manually
checked against the original transcript
(interviews/cleaned/<id>/transcript.txt) -- filling in
`transcript_initial_exemplar_text` from the real wording, correcting the
candidate if it's wrong, setting `exemplar_type`, and setting
`initial_response_form` (some opening answers are a feature-based
characterisation rather than one named exemplar -- see
INITIAL_RESPONSE_FORMS -- with any named examples that only emerge later,
via a follow-up probe, recorded separately in `follow_up_examples` rather
than folded into the opening answer) -- before `review_status` can become
"reviewed". analyze_initial_exemplars.py
enforces that gate; this script does not.

Usage (from thesis/corpus/):
    python -m thesis_corpus.propose_initial_exemplars
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

from thesis_corpus import geometric_analysis_common as gac

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("thesis_corpus.propose_initial_exemplars")

INTERVIEWS_DIR = gac.CORPUS_DIR / "interviews"
DATABASE_PATH = INTERVIEWS_DIR / "metadata" / "database.json"
RAW_ARCHIVE_PATH = gac.PROCESSED_DIR / "interviews" / "criterion_expressions.jsonl"
OUTPUT_PATH = INTERVIEWS_DIR / "metadata" / "initial_exemplars.csv"

CSV_FIELDS = [
    "document_id", "participant_id", "transcript_initial_exemplar_text",
    "source_expression_key", "source_expression_label", "source_chunk_index",
    "exemplar_type", "initial_response_form", "follow_up_examples",
    "review_status", "notes",
]
EXEMPLAR_TYPES = (
    "classic_nrm_or_religious_group", "named_group_or_movement",
    "organisation_or_institution", "political_or_ideological",
    "digital_or_technology", "interpersonal_or_family",
    "metaphorical_or_other", "unclear",
)
# Some participants' opening answer isn't a named exemplar at all -- it's a
# feature-based characterisation (guru, group, doctrine, rules, marginal
# religion, ...), with named examples only emerging later via a follow-up
# probe (found reviewing "b3-aug18-1645"). initial_response_form records
# which kind of answer this was, independently of exemplar_type (which
# describes *what* is named, not *whether* something was named at all);
# follow_up_examples keeps later-mentioned named groups on record without
# folding them into the opening answer. Neither is auto-classified here --
# both default to the reviewer's starting point, same as exemplar_type.
INITIAL_RESPONSE_FORMS = ("named_exemplar", "descriptive_characterisation", "mixed", "unclear")


def load_raw_archive_by_document(path: Path) -> dict[str, list[dict]]:
    by_document: dict[str, list[dict]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            by_document.setdefault(item["document_id"], []).append(item)
    return by_document


def build_pooled_lookup(shared_space: gac.SharedSpace) -> tuple[dict[tuple, int], dict[int, str]]:
    """(document_id, chunk_index, label) -> index into shared_space.points,
    and index -> virtual_key, both restricted to interviews points -- so a
    raw archive item can be checked for pooling survival and, if it
    survived, resolved to its virtual key."""
    virtual_keys = gac.derive_interview_expression_keys(shared_space.points)
    index_to_virtual_key = {i: k for k, i in virtual_keys.items()}

    pooled_lookup: dict[tuple, int] = {}
    for i, p in enumerate(shared_space.points):
        if p.get("source_dataset") != "interviews":
            continue
        document_id = gac.key_document_id(p["key"])
        chunk_index = gac.key_chunk_index(p["key"])
        pooled_lookup[(document_id, chunk_index, p["label"])] = i

    return pooled_lookup, index_to_virtual_key


def is_candidate(item: dict) -> bool:
    return item.get("attribution") == "participant" and item.get("claim_mode") != "question_or_reflection"


def propose_for_document(
    document_id: str, participant_id: str, archive_items: list[dict],
    pooled_lookup: dict[tuple, int], index_to_virtual_key: dict[int, str],
) -> dict:
    candidates = [item for item in archive_items if is_candidate(item)]
    if not candidates:
        return {
            "document_id": document_id, "participant_id": participant_id,
            "transcript_initial_exemplar_text": "", "source_expression_key": "",
            "source_expression_label": "", "source_chunk_index": "",
            "exemplar_type": "unclear", "initial_response_form": "unclear", "follow_up_examples": "",
            "review_status": "unavailable",
            "notes": "No participant claim (excluding questions/reflections) found anywhere in the raw archive for this document.",
        }

    true_first = candidates[0]
    true_first_key = (true_first["document_id"], true_first["chunk_index"], true_first["embedding_text"])
    notes = ""

    chosen = true_first
    if true_first_key not in pooled_lookup:
        survivor = next(
            (c for c in candidates if (c["document_id"], c["chunk_index"], c["embedding_text"]) in pooled_lookup),
            None,
        )
        if survivor is None:
            return {
                "document_id": document_id, "participant_id": participant_id,
                "transcript_initial_exemplar_text": "", "source_expression_key": "",
                "source_expression_label": "", "source_chunk_index": "",
                "exemplar_type": "unclear", "initial_response_form": "unclear", "follow_up_examples": "",
            "review_status": "unavailable",
                "notes": (
                    f"Candidate first claim ({true_first['embedding_text']!r}, chunk "
                    f"{true_first['chunk_index']}) was filtered out of the shared space "
                    "(likely the short-fragment/duplicate filter), and no later candidate "
                    "for this document survived pooling either -- needs manual resolution."
                ),
            }
        chosen = survivor
        notes = (
            f"True first candidate ({true_first['embedding_text']!r}, chunk "
            f"{true_first['chunk_index']}) was filtered out of the shared space (likely "
            "the short-fragment/duplicate filter) -- this row uses the next surviving "
            "candidate instead. Reviewer: confirm this is still a reasonable stand-in, "
            "or mark review_status=unavailable."
        )

    point_index = pooled_lookup[(chosen["document_id"], chosen["chunk_index"], chosen["embedding_text"])]
    virtual_key = index_to_virtual_key[point_index]

    return {
        "document_id": document_id, "participant_id": participant_id,
        "transcript_initial_exemplar_text": "",
        "source_expression_key": virtual_key,
        "source_expression_label": chosen["embedding_text"],
        "source_chunk_index": chosen["chunk_index"],
        "exemplar_type": "unclear", "initial_response_form": "unclear", "follow_up_examples": "",
        "review_status": "pending",
        "notes": notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    logger.info("Loading %s ...", DATABASE_PATH)
    entries = json.loads(DATABASE_PATH.read_text(encoding="utf-8"))["interviews"]

    logger.info("Loading raw archive %s ...", RAW_ARCHIVE_PATH)
    archive_by_document = load_raw_archive_by_document(RAW_ARCHIVE_PATH)

    logger.info("Loading shared space to resolve pooling survival ...")
    shared_space = gac.load_shared_space()
    pooled_lookup, index_to_virtual_key = build_pooled_lookup(shared_space)

    rows = []
    for entry in entries:
        document_id = entry["id"]
        participant_id = entry.get("participant_id", document_id)
        archive_items = archive_by_document.get(document_id, [])
        rows.append(propose_for_document(document_id, participant_id, archive_items, pooled_lookup, index_to_virtual_key))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    n_pending = sum(1 for r in rows if r["review_status"] == "pending")
    n_unavailable = sum(1 for r in rows if r["review_status"] == "unavailable")
    n_with_fallback_note = sum(1 for r in rows if r["notes"])
    print(f"\nWrote {len(rows)} rows -> {args.output}")
    print(f"pending: {n_pending}, unavailable: {n_unavailable}, flagged (fallback/no-candidate) notes: {n_with_fallback_note}")
    print("Every row needs manual review against interviews/cleaned/<id>/transcript.txt before analyze_initial_exemplars.py will accept it.")


if __name__ == "__main__":
    main()
