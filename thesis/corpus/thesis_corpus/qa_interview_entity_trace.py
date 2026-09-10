"""QA sanity-check aid (not an analytical deliverable): for every interview
document, trace its own kept expressions and every domain_term it
contributed, resolved forward to whichever final pooled emergent entity
(if any) it became in the shared space -- exactly the "does this make
sense" check requested against the raw per-interview extraction, rather
than only the pooled, cross-corpus aggregates the rest of the toolkit
reports.

Deliberately kept OUTSIDE the run-id'd analysis toolkit
(geometric_analysis_common's run-manifest machinery): this reads the raw
v2 interview archive directly and cross-references it against whatever
shared-space build is on disk, rather than being itself a step in a
reproducible analysis run.

Two resolution paths, mirroring build_shared_space.py's own pooling
exactly (reusing its functions rather than re-deriving the filtering
logic):
  - Expressions: pooled into an "expression" point unless manually
    excluded (MANUALLY_EXCLUDED_POOLED_KEYS) -- min_expression_words is 0
    for v2 runs, so that filter never applies here.
  - Domain terms: resolved through the same three gates
    load_domain_term_entity_mentions + load_emergent_entities apply --
    looks_like_named_entity(), embedded at all (present in
    domain_term_vectors.jsonl), not a cited-author surname, and clearing
    some corpus's min-mentions threshold -- reporting exactly which gate
    a filtered-out term failed, not just "filtered".

Note: interviews' own criterion_expressions.jsonl currently carries no
entity_anchor_vectors at all (checked directly -- 0 of 64 expressions
have any), so every interview-sourced entity mention in the shared space
comes via domain_terms, not entity_anchors; this module reflects that
rather than assuming both sources apply.

Usage (from thesis/corpus/):
    python -m thesis_corpus.qa_interview_entity_trace \
        --interview-run-tag 20260910 --shared-space-dir processed/shared_space_v2
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from thesis_corpus import build_shared_space as bss
from thesis_corpus import geometric_analysis_common as gac

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "processed"
DEFAULT_QA_OUTPUT_DIR = PROCESSED_DIR / "qa" / "interview_entity_trace"


# ---------------------------------------------------------------------------
# Pure computation -- no file I/O, unit-testable.
# ---------------------------------------------------------------------------

def resolve_expression(document_id: str, chunk_index: int) -> str:
    """"pooled" or "manually_excluded" -- the only two outcomes possible
    for a v2 interview expression at min_expression_words=0."""
    key = f"{document_id}:{chunk_index}"
    return "manually_excluded" if key in bss.MANUALLY_EXCLUDED_POOLED_KEYS else "pooled"


def resolve_domain_term(
    raw_term: str, embedded_terms: set[str], cited_author_surnames: set[str],
    emergent_entities_by_key: dict[str, dict],
) -> dict:
    """Walks the same gates load_domain_term_entity_mentions +
    load_emergent_entities apply to a raw domain_term, reporting exactly
    which gate stopped it (or its final resolved entity if it survived
    all of them)."""
    if not bss.looks_like_named_entity(raw_term):
        return {"status": "filtered_not_named_entity_shaped", "resolved_key": None,
                "resolved_label": None, "mention_distribution": None}
    if raw_term not in embedded_terms:
        return {"status": "filtered_not_embedded", "resolved_key": None,
                "resolved_label": None, "mention_distribution": None}
    key = bss.normalize_anchor(raw_term)
    if not key or key in cited_author_surnames:
        return {"status": "filtered_cited_author_surname", "resolved_key": key,
                "resolved_label": None, "mention_distribution": None}
    point = emergent_entities_by_key.get(key)
    if point is None:
        return {"status": "filtered_below_mention_threshold", "resolved_key": key,
                "resolved_label": None, "mention_distribution": None}
    return {"status": "pooled", "resolved_key": key, "resolved_label": point["label"],
             "mention_distribution": point.get("mention_distribution")}


def build_trace_rows(
    expressions_by_doc: dict[str, list[dict]],
    domain_terms_by_doc_chunk: dict[tuple[str, int], list[str]],
    embedded_terms: set[str],
    cited_author_surnames: set[str],
    emergent_entities_by_key: dict[str, dict],
) -> list[dict]:
    """One row per (expression) or (chunk, domain_term) -- interleaved by
    document so a human reading the CSV in order sees each interview's own
    material together."""
    rows = []
    for document_id in sorted(expressions_by_doc):
        for item in expressions_by_doc[document_id]:
            chunk_index = item["chunk_index"]
            status = resolve_expression(document_id, chunk_index)
            rows.append({
                "document_id": document_id, "chunk_index": chunk_index,
                "item_type": "expression", "text": item["verbatim_expression"],
                "attribution": item.get("attribution"), "claim_mode": item.get("claim_mode"),
                "epistemic_status": item.get("epistemic_status"),
                "resolution_status": status, "resolved_entity": None, "mention_distribution": None,
            })
            for raw_term in domain_terms_by_doc_chunk.get((document_id, chunk_index), []):
                resolution = resolve_domain_term(raw_term, embedded_terms, cited_author_surnames, emergent_entities_by_key)
                rows.append({
                    "document_id": document_id, "chunk_index": chunk_index,
                    "item_type": "domain_term", "text": raw_term,
                    "attribution": None, "claim_mode": None, "epistemic_status": None,
                    "resolution_status": resolution["status"],
                    "resolved_entity": resolution["resolved_label"],
                    "mention_distribution": resolution["mention_distribution"],
                })
    return rows


# ---------------------------------------------------------------------------
# I/O -- loading the raw v2 interview archive and the shared space.
# ---------------------------------------------------------------------------

def load_interview_expressions_by_doc(criterion_expressions_path: Path) -> dict[str, list[dict]]:
    by_doc: dict[str, list[dict]] = defaultdict(list)
    with open(criterion_expressions_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            by_doc[item["document_id"]].append(item)
    for items in by_doc.values():
        items.sort(key=lambda it: it["chunk_index"])
    return dict(by_doc)


def load_domain_terms_by_doc_chunk(chunk_terms_path: Path) -> dict[tuple[str, int], list[str]]:
    by_chunk: dict[tuple[str, int], list[str]] = {}
    with open(chunk_terms_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            by_chunk[(row["document_id"], row["chunk_index"])] = row.get("domain_terms", [])
    return by_chunk


def load_embedded_domain_term_set(term_vectors_path: Path) -> set[str]:
    """Mirrors load_domain_term_entity_mentions' own vector-loading loop
    (build_shared_space.py) -- the raw terms that both look named-entity-
    shaped and were actually embedded."""
    if not term_vectors_path.exists():
        return set()
    terms = set()
    with open(term_vectors_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if bss.looks_like_named_entity(row["term"]):
                terms.add(row["term"])
    return terms


def load_emergent_entities_by_key(shared_space: gac.SharedSpace) -> dict[str, dict]:
    idxs = gac.source_dataset_indices(shared_space.points, "emergent_entities")
    return {shared_space.points[i]["key"]: shared_space.points[i] for i in idxs}


# ---------------------------------------------------------------------------
# Output -- CSV (queryable) + a compact per-interview PDF (plain layout).
# ---------------------------------------------------------------------------

def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["document_id", "chunk_index", "item_type", "text", "attribution",
                  "claim_mode", "epistemic_status", "resolution_status",
                  "resolved_entity", "mention_distribution"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["mention_distribution"] = json.dumps(out["mention_distribution"], ensure_ascii=False) if out["mention_distribution"] else ""
            writer.writerow(out)


def write_pdf(rows: list[dict], path: Path) -> None:
    """One page per interview, plain layout -- verbatim expressions and
    their resolved entities, no interpretation. A QA reading aid, not a
    thesis figure."""
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_doc[row["document_id"]].append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        for document_id in sorted(by_doc):
            doc_rows = by_doc[document_id]
            fig = plt.figure(figsize=(8.5, 11))
            fig.suptitle(f"Interview {document_id} -- extraction trace", fontsize=12, y=0.98)
            ax = fig.add_axes([0.04, 0.02, 0.92, 0.92])
            ax.axis("off")

            lines = []
            current_chunk = None
            for row in doc_rows:
                if row["chunk_index"] != current_chunk:
                    current_chunk = row["chunk_index"]
                    lines.append("")
                    lines.append(f"chunk {current_chunk}")
                if row["item_type"] == "expression":
                    text = (row["text"] or "")[:110]
                    lines.append(f"  [expr, {row['resolution_status']}] ({row['attribution']}/"
                                 f"{row['claim_mode']}/{row['epistemic_status']}) {text}")
                else:
                    resolved = row["resolved_entity"] or "--"
                    lines.append(f"    term '{row['text']}' -> {row['resolution_status']} -> {resolved}")

            wrapped = "\n".join(lines)
            ax.text(0, 1, wrapped, fontsize=6.5, family="monospace", va="top", ha="left", wrap=False)
            pdf.savefig(fig)
            plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--interview-run-tag", type=str, required=True,
                         help="v2 interview run tag, e.g. 20260910 -- reads "
                              "processed/v2/interviews/run_<tag>/{criterion_expressions,chunk_terms,"
                              "domain_term_vectors}.jsonl.")
    parser.add_argument("--shared-space-dir", type=Path, default=PROCESSED_DIR / "shared_space_v2")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_QA_OUTPUT_DIR)
    args = parser.parse_args()

    run_dir = PROCESSED_DIR / "v2" / "interviews" / f"run_{args.interview_run_tag}"
    criterion_expressions_path = run_dir / "criterion_expressions.jsonl"
    chunk_terms_path = run_dir / "chunk_terms.jsonl"
    term_vectors_path = run_dir / "domain_term_vectors.jsonl"
    for p in (criterion_expressions_path, chunk_terms_path):
        if not p.exists():
            raise SystemExit(f"Missing: {p}")

    print(f"Loading interview expressions from {criterion_expressions_path} ...")
    expressions_by_doc = load_interview_expressions_by_doc(criterion_expressions_path)
    print(f"Loading domain terms from {chunk_terms_path} ...")
    domain_terms_by_doc_chunk = load_domain_terms_by_doc_chunk(chunk_terms_path)
    embedded_terms = load_embedded_domain_term_set(term_vectors_path)

    cited_author_surnames = bss.load_cited_author_surnames()

    embedding_space_path, _interview_prototypes_path, _analysis_root = gac.resolve_space_paths(args.shared_space_dir)
    print(f"Loading shared space from {embedding_space_path} ...")
    shared_space = gac.load_shared_space(embedding_space_path)
    emergent_entities_by_key = load_emergent_entities_by_key(shared_space)

    rows = build_trace_rows(
        expressions_by_doc, domain_terms_by_doc_chunk, embedded_terms,
        cited_author_surnames, emergent_entities_by_key,
    )

    n_docs = len(expressions_by_doc)
    n_expressions = sum(1 for r in rows if r["item_type"] == "expression")
    n_terms = sum(1 for r in rows if r["item_type"] == "domain_term")
    n_terms_pooled = sum(1 for r in rows if r["item_type"] == "domain_term" and r["resolution_status"] == "pooled")
    print(f"{n_docs} interviews, {n_expressions} expressions, {n_terms} domain_term mentions "
          f"({n_terms_pooled} resolved to a pooled entity).")

    csv_path = args.output_dir / "interview_entity_trace.csv"
    pdf_path = args.output_dir / "interview_entity_trace.pdf"
    write_csv(rows, csv_path)
    write_pdf(rows, pdf_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
