"""Proposes a candidate "initial exemplar" (first participant claim in
response to the interview's opening prompt) for each of the 26 interviews,
for a human to review and correct.

**This is a proposal heuristic only, never ground truth.** Candidate =
first item in the raw archive's own transcript/chunk order with
`attribution == "participant"` and `claim_mode != "question_or_reflection"`
-- confirmed against ollama_client.py's actual claim_mode enum and real
data (e.g. "b1-aug05-1650"'s pooled response_rank==1 item is exactly the
case this excludes: the interviewer's own question).

Candidates are addressed by a virtual key
(geometric_analysis_common.derive_archive_expression_keys) resolved
directly against the raw, unfiltered interview archive
(processed/interviews/criterion_expressions.jsonl) -- deliberately NOT
checked against the pooled embedding_space.jsonl or its dedup/short-fragment
filter, since build_interview_prototype_layer.py projects each reviewed
exemplar's existing raw embedding through the shared space's own persisted
transform independently of whether that expression happened to survive
ordinary pooling. A short, genuine free-association answer like "AI cult"
or "Illuminati" is exactly the kind of thing that filter drops (it targets
a different, unrelated extraction-noise pattern) but is perfectly usable
here regardless.

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
"reviewed". build_interview_prototype_layer.py enforces that gate; this
script does not.

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
OUTPUT_PATH = gac.INITIAL_EXEMPLARS_CSV_PATH

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


def is_candidate(item: dict) -> bool:
    return item.get("attribution") == "participant" and item.get("claim_mode") != "question_or_reflection"


def propose_for_document(
    document_id: str, participant_id: str, indexed_archive_items: list[tuple[int, dict]],
    index_to_key: dict[int, str],
) -> dict:
    candidates = [(i, item) for i, item in indexed_archive_items if is_candidate(item)]
    if not candidates:
        return {
            "document_id": document_id, "participant_id": participant_id,
            "transcript_initial_exemplar_text": "", "source_expression_key": "",
            "source_expression_label": "", "source_chunk_index": "",
            "exemplar_type": "unclear", "initial_response_form": "unclear", "follow_up_examples": "",
            "review_status": "unavailable",
            "notes": "No participant claim (excluding questions/reflections) found anywhere in the raw archive for this document.",
        }

    index, chosen = candidates[0]
    return {
        "document_id": document_id, "participant_id": participant_id,
        "transcript_initial_exemplar_text": "",
        "source_expression_key": index_to_key[index],
        "source_expression_label": chosen["embedding_text"],
        "source_chunk_index": chosen["chunk_index"],
        "exemplar_type": "unclear", "initial_response_form": "unclear", "follow_up_examples": "",
        "review_status": "pending",
        "notes": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    logger.info("Loading %s ...", DATABASE_PATH)
    entries = json.loads(DATABASE_PATH.read_text(encoding="utf-8"))["interviews"]

    logger.info("Loading raw archive %s ...", gac.INTERVIEWS_ARCHIVE_PATH)
    archive_items = gac.load_raw_archive(gac.INTERVIEWS_ARCHIVE_PATH)
    archive_keys = gac.derive_archive_expression_keys(archive_items)
    index_to_key = {i: k for k, i in archive_keys.items()}

    by_document: dict[str, list[tuple[int, dict]]] = {}
    for i, item in enumerate(archive_items):
        by_document.setdefault(item["document_id"], []).append((i, item))

    rows = []
    for entry in entries:
        document_id = entry["id"]
        participant_id = entry.get("participant_id", document_id)
        rows.append(propose_for_document(document_id, participant_id, by_document.get(document_id, []), index_to_key))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    n_pending = sum(1 for r in rows if r["review_status"] == "pending")
    n_unavailable = sum(1 for r in rows if r["review_status"] == "unavailable")
    print(f"\nWrote {len(rows)} rows -> {args.output}")
    print(f"pending: {n_pending}, unavailable: {n_unavailable}")
    print("Every row needs manual review against interviews/cleaned/<id>/transcript.txt before "
          "build_interview_prototype_layer.py will accept it.")


if __name__ == "__main__":
    main()
