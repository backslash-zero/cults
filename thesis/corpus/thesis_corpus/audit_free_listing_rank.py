"""Audits whether `response_rank` (embedding_space.jsonl, interview points
only) can support any claim about free-listing/cognitive-salience order.

Background: `response_rank` is computed in build_shared_space.py as a
plain 1..N counter over every extracted item for an interview document, in
raw chunk/extraction order -- it does not separate "answer to the opening
prompt" from later probing, and does not exclude the interviewer's own
question text. The interview protocol itself (thesis/04_Appendix/3_Appendix.tex,
sec:interview_protocol) is also not a multi-item free-listing task: it's
one prompt eliciting a single example, a second prompt asking the
participant to justify that same example, and open-ended probes -- there
was only ever one exemplar to rank, not a ranked list.

This audit checks the narrowest, most favorable claim that could still be
true: for each interview, is response_rank == 1 a participant's own claim
(not the interviewer's question) from the first chunk? Verified this holds
for some documents and not others (e.g. "b1-aug05-1650": rank 1 is the
interviewer's own question text, attribution="unspecified"). The verdict
is a documentation/provenance artifact, not a gate -- no downstream module
in this toolkit depends on it passing, since rank-based prototype analysis
is retired outright (see analyze_initial_exemplars.py) rather than
conditionally unblocked by this check.

Usage (from thesis/corpus/):
    python -m thesis_corpus.audit_free_listing_rank
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from thesis_corpus import geometric_analysis_common as gac

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("thesis_corpus.audit_free_listing_rank")

MODULE_NAME = "audit_free_listing_rank"


def audit(points: list[dict]) -> dict:
    interview_points = [p for p in points if p.get("source_dataset") == "interviews"]

    by_document: dict[str, list[dict]] = {}
    for p in interview_points:
        by_document.setdefault(gac.key_document_id(p["key"]), []).append(p)

    per_document = []
    for document_id, doc_points in sorted(by_document.items()):
        rank1 = [p for p in doc_points if p.get("response_rank") == 1]
        if not rank1:
            per_document.append({
                "document_id": document_id, "rank1_exists": False,
                "rank1_is_participant_claim": False, "reason": "no response_rank==1 item pooled (possibly filtered out)",
            })
            continue
        p = rank1[0]
        chunk_index = gac.key_chunk_index(p["key"])
        is_participant = p.get("attribution") == "participant"
        is_chunk_zero = chunk_index == 0
        passes = is_participant and is_chunk_zero
        per_document.append({
            "document_id": document_id,
            "rank1_exists": True,
            "rank1_attribution": p.get("attribution"),
            "rank1_claim_mode": p.get("claim_mode"),
            "rank1_chunk_index": chunk_index,
            "rank1_label": p.get("label"),
            "rank1_is_participant_claim": passes,
        })

    n_documents = len(per_document)
    n_passing = sum(1 for d in per_document if d.get("rank1_is_participant_claim"))
    pass_rate = n_passing / n_documents if n_documents else 0.0

    return {
        "salience_proxy_valid": False,
        "reason": (
            "response_rank is a whole-transcript extraction-order counter, not a "
            "free-listing rank: (1) the interview protocol elicits one exemplar "
            "per participant, not a ranked list (sec:interview_protocol); (2) even "
            "the narrowest claim checked here -- rank==1 is the participant's own "
            "first claim, not the interviewer's question -- only holds for "
            f"{n_passing}/{n_documents} documents ({pass_rate:.1%})."
        ),
        "narrow_check": "response_rank == 1 is attribution=='participant' from chunk_index == 0",
        "n_documents": n_documents,
        "n_passing_narrow_check": n_passing,
        "narrow_check_pass_rate": pass_rate,
        "per_document": per_document,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--shared-space-dir", type=Path, default=None,
                         help="Use a different pooled space (e.g. processed/shared_space_v2/) instead of v1's "
                              "processed/shared_space/; also switches the run-output root to a sibling "
                              "processed/analysis_v2/ directory so v1 and v2 runs are never mixed.")
    args = parser.parse_args()

    embedding_space_path, interview_prototypes_path, analysis_root = gac.resolve_space_paths(args.shared_space_dir)
    logger.info("Loading %s ...", embedding_space_path)
    shared_space = gac.load_shared_space(embedding_space_path)

    run_id, run_dir = gac.get_or_create_run_dir(args.run_id, analysis_root)
    gac.init_run_manifest(run_dir, shared_space, defaults={})
    gac.update_run_manifest(run_dir, MODULE_NAME, "running")
    out_dir = gac.module_run_dir(run_dir, MODULE_NAME)

    try:
        verdict = audit(shared_space.points)
        (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8")

        gac.write_module_config(
            out_dir,
            input_sha256=shared_space.input_sha256,
            n_interview_documents=verdict["n_documents"],
            git_commit=gac.git_commit_hash(),
        )
        gac.update_run_manifest(run_dir, MODULE_NAME, "completed", output_dir=out_dir)
    except Exception as e:
        gac.update_run_manifest(run_dir, MODULE_NAME, "failed", error=str(e))
        raise

    print(f"\nsalience_proxy_valid: {verdict['salience_proxy_valid']}")
    print(f"Narrow check (rank==1 is participant's own first claim): "
          f"{verdict['n_passing_narrow_check']}/{verdict['n_documents']} documents "
          f"({verdict['narrow_check_pass_rate']:.1%})")
    print(f"\nDone. Run {run_id} -> {out_dir}")


if __name__ == "__main__":
    main()
