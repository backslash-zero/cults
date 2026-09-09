"""Reproducible random quality-review sample from a v2 extraction run.

Draws N rows uniformly at random (fixed seed) from a run's final archive
(processed/v2/<corpus>/run_<tag>/expressions_v2.jsonl -- judge-accepted
expressions) for manual spot-review. No automated judgement of quality is
added: the judge model's own verdict is surfaced as reference judge_*
columns (labelled as such), and every manual-review column is left blank.

Optionally stratifies by document (--stratify-by-document, floor-then-
proportional like audit_expression_fidelity_sample.py's status
stratification) so one large document cannot dominate the sample -- off by
default, since a plain random draw is what a first spot-check usually
means, and stratification is more useful once a run covers every document.

The run need not be finished: the output records exactly how many documents
were present in the source file at sampling time versus the corpus's total,
so an incomplete-run sample is never silently mistaken for a full-corpus one.

Usage (from thesis/corpus/):
    python -m thesis_corpus.sample_v2_expressions --corpus literature --run-tag 20260910 --n 100
    python -m thesis_corpus.sample_v2_expressions --corpus literature --run-tag 20260910 --n 100 --stratify-by-document
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from thesis_corpus.pilot_v2_literature import (
    PROCESSED_ROOT, git_commit_hash, judge_columns, read_jsonl, sha256_file, write_json,
)

REVIEW_PREFILLED_COLUMNS = [
    "sample_id", "document_id", "chunk_index", "candidate_rank", "expression_kind",
    "verbatim_expression", "embedding_text", "text_transform", "attribution", "attribution_source",
    "model_attribution", "claim_mode", "epistemic_status", "entity_anchors", "context_window", "screen_flags",
    "judge_model", "judge_faithful", "judge_self_contained", "judge_cult_relevant", "judge_textually_intelligible",
    "judge_atomic", "judge_attribution_correct", "judge_claim_mode_correct", "judge_epistemic_status_correct",
    "judge_recommended_epistemic_status", "judge_better_span", "judge_extraction_issue", "judge_note",
]
MANUAL_REVIEW_COLUMNS = [
    "faithful", "self_contained", "cult_relevant", "textually_intelligible", "atomic",
    "attribution_correct", "claim_mode_correct", "epistemic_status_correct",
    "recommended_epistemic_status", "better_span_exists_in_chunk", "extraction_issue", "reviewer_notes",
]


def floor_then_proportional(counts: dict[str, int], n: int) -> dict[str, int]:
    """Same floor-then-proportional design as
    audit_expression_fidelity_sample.stratified_allocation_by_status: every
    group present gets at least 1 (smallest groups first, capped at n and at
    the group's own size), then the remainder is allocated proportionally
    with largest-remainder rounding."""
    groups = sorted(counts, key=lambda g: counts[g])
    allocation = {g: 0 for g in groups}
    remaining_n = n
    for g in groups:
        if remaining_n <= 0:
            break
        take = min(1, counts[g])
        allocation[g] += take
        remaining_n -= take
    if remaining_n > 0:
        pool = {g: counts[g] - allocation[g] for g in groups}
        total_pool = sum(pool.values())
        if total_pool > 0:
            raw = {g: remaining_n * pool[g] / total_pool for g in groups}
            floor_alloc = {g: int(raw[g]) for g in groups}
            allocation = {g: allocation[g] + floor_alloc[g] for g in groups}
            leftover = remaining_n - sum(floor_alloc.values())
            remainders = sorted(groups, key=lambda g: (raw[g] - floor_alloc[g]), reverse=True)
            for g in remainders:
                if leftover <= 0:
                    break
                if allocation[g] < counts[g]:
                    allocation[g] += 1
                    leftover -= 1
    return allocation


def sample_rows(rows: list[dict], n: int, seed: int, stratify_by_document: bool) -> list[dict]:
    n = min(n, len(rows))
    if not stratify_by_document:
        rng = random.Random(seed)
        chosen_indices = sorted(rng.sample(range(len(rows)), n))
        return [rows[i] for i in chosen_indices]

    by_doc: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_doc[row["document_id"]].append(i)
    allocation = floor_then_proportional({d: len(idx) for d, idx in by_doc.items()}, n)
    chosen: list[int] = []
    for doc_index, document_id in enumerate(sorted(by_doc)):
        quota = allocation[document_id]
        if quota <= 0:
            continue
        rng = random.Random(seed * 1000 + doc_index)
        chosen.extend(rng.sample(by_doc[document_id], quota))
    return [rows[i] for i in sorted(chosen)]


def build_review_rows(sampled: list[dict]) -> list[dict]:
    rows = []
    for rec in sorted(sampled, key=lambda r: (r["document_id"], r["chunk_index"], r["candidate_rank"])):
        anchors = rec.get("entity_anchors") or []
        rows.append({
            "sample_id": None, "document_id": rec["document_id"], "chunk_index": rec["chunk_index"],
            "candidate_rank": rec["candidate_rank"], "expression_kind": rec["expression_kind"],
            "verbatim_expression": rec["verbatim_expression"], "embedding_text": rec["embedding_text"],
            "text_transform": rec["text_transform"], "attribution": rec["attribution"],
            "attribution_source": rec.get("attribution_source", "model"), "model_attribution": rec.get("model_attribution", rec["attribution"]),
            "claim_mode": rec["claim_mode"], "epistemic_status": rec["epistemic_status"],
            "entity_anchors": "; ".join(anchors), "context_window": rec["context_window"],
            "screen_flags": ";".join(rec.get("screen_flags", [])),
            **judge_columns(rec),
            **{c: "" for c in MANUAL_REVIEW_COLUMNS},
        })
    for i, row in enumerate(rows, start=1):
        row["sample_id"] = i
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stratify-by-document", action="store_true")
    parser.add_argument("--out-root", type=Path, default=PROCESSED_ROOT / "v2")
    parser.add_argument("--date-tag", default=datetime.now(timezone.utc).strftime("%Y%m%d"))
    args = parser.parse_args()

    run_dir = args.out_root / args.corpus / f"run_{args.run_tag}"
    source_path = run_dir / "expressions_v2.jsonl"
    if not source_path.exists():
        raise SystemExit(f"{source_path} missing")
    rows = read_jsonl(source_path)
    if not rows:
        raise SystemExit(f"{source_path} has no rows yet")

    docs_in_sample_source = sorted({r["document_id"] for r in rows})
    all_documents_dir = PROCESSED_ROOT / args.corpus / "documents"
    total_documents = len([d for d in all_documents_dir.iterdir() if d.is_dir()]) if all_documents_dir.exists() else None
    per_doc_counts = Counter(r["document_id"] for r in rows)

    sampled = sample_rows(rows, args.n, args.seed, args.stratify_by_document)
    review_rows = build_review_rows(sampled)

    review_dir = run_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    method = "stratified_by_document" if args.stratify_by_document else "uniform_random"
    stem = f"quality_sample_{args.n}_{method}_{args.date_tag}"
    csv_path = review_dir / f"{stem}.csv"
    config_path = review_dir / f"{stem}.config.json"
    if csv_path.exists():
        raise SystemExit(f"Refusing to overwrite {csv_path} (pick another --date-tag)")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_PREFILLED_COLUMNS + MANUAL_REVIEW_COLUMNS)
        w.writeheader()
        w.writerows(review_rows)

    write_json(config_path, {
        "script": "thesis_corpus.sample_v2_expressions", "corpus": args.corpus, "run_tag": args.run_tag,
        "method": method, "n_requested": args.n, "n_sampled": len(review_rows), "seed": args.seed,
        "source_path": str(source_path), "source_sha256": sha256_file(source_path), "source_rows": len(rows),
        "documents_in_source": docs_in_sample_source, "documents_in_source_count": len(docs_in_sample_source),
        "documents_total_in_corpus": total_documents,
        "run_complete": (total_documents is not None and len(docs_in_sample_source) == total_documents),
        "per_document_row_counts_in_source": dict(per_doc_counts),
        "git_commit": git_commit_hash(), "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": "No automated judgement beyond the run's own judge_* columns (already model-judged, "
                "not human review); manual review columns are blank.",
    })
    print(f"Sampled {len(review_rows)}/{len(rows)} rows ({method}) from {len(docs_in_sample_source)}/"
          f"{total_documents if total_documents is not None else '?'} documents -> {csv_path}")
    if total_documents is not None and len(docs_in_sample_source) < total_documents:
        print(f"NOTE: run not yet complete -- only {len(docs_in_sample_source)}/{total_documents} "
              f"documents are represented in the source file this sample was drawn from.")


if __name__ == "__main__":
    main()
