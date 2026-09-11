"""Builds one archive record per interview segment -- guaranteed, regardless
of whether the model returned a valid label for it.

This replaces screen_interviews_full.py's candidate-screening design.
There, the model chose which spans to return, and screen_candidate() then
decided which of those spans to keep -- so a piece of text could be
missing from the archive either because the model never emitted anything
for it, or because it emitted something that got rejected (not_verbatim,
dangling_boundary, etc.). Neither failure mode is possible here: the
segments to embed come entirely from interview_chunking.py (deterministic,
independent of the model), and every one of them becomes a record. The
model's job is only to LABEL each segment (interview_extraction_schema.py);
build_segment_records() below attaches a label when a valid one exists for
that segment_index, and a "missing"/"invalid" placeholder when it doesn't
-- but the segment is embedded either way.

Reused, unmodified: screen_v2.prepare_chunk/fold_newlines/screen_domain_terms;
text_integrity.has_hard_corruption/soft_flags (informational only here --
never a reason to drop a segment, since text_integrity issues in an
already-reviewed, hand-typed transcript are vanishingly rare and, even if
present, the segment must still get a point in the space).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from thesis_corpus import interview_extraction_schema as schema
from thesis_corpus import text_integrity as ti
from thesis_corpus.screen_v2 import ChunkContext, fold_newlines, prepare_chunk, screen_domain_terms

__all__ = [
    "ChunkContext", "prepare_chunk", "fold_newlines", "screen_domain_terms",
    "segment_spans", "build_segment_records",
]

_BLANK_LINE_RE = re.compile(r"\n\s*\n")


def segment_spans(nfc_text: str, turn_roles: list[str]) -> list[tuple[int, int, str]]:
    """Pairs each blank-line-separated paragraph of `nfc_text` with its role
    from `turn_roles`, positionally and in order -- interview_chunking.py
    joins segments with exactly this separator, in this same order. This is
    the authoritative list of what gets embedded: every entry returned here
    becomes exactly one archive record in build_segment_records(), whether
    or not the model labeled it."""
    spans, pos = [], 0
    for part in _BLANK_LINE_RE.split(nfc_text):
        idx = nfc_text.find(part, pos) if part else pos
        if idx < 0:
            idx = pos
        spans.append((idx, idx + len(part)))
        pos = idx + len(part)
    return [(s, e, role) for (s, e), role in zip(spans, turn_roles)]


@dataclass
class LabelIssue:
    segment_index: int
    code: str
    detail: str = ""
    raw_label: object = None


def _validate_raw_labels(raw_labels: list) -> tuple[dict[int, schema.SegmentLabel], list[LabelIssue]]:
    by_index: dict[int, schema.SegmentLabel] = {}
    issues: list[LabelIssue] = []
    for raw in raw_labels:
        label, code, detail = schema.validate_label(raw)
        if label is None:
            idx = raw.get("segment_index") if isinstance(raw, dict) else None
            issues.append(LabelIssue(idx if isinstance(idx, int) else -1, code, detail, raw))
            continue
        if label.segment_index in by_index:
            issues.append(LabelIssue(label.segment_index, "duplicate_segment_index",
                                     f"segment {label.segment_index} labeled more than once; first kept", raw))
            continue
        by_index[label.segment_index] = label
    return by_index, issues


def build_segment_records(
    ctx: ChunkContext, turn_roles: list[str], raw_labels: list,
) -> tuple[list[dict], list[LabelIssue]]:
    """Returns (records, issues). `records` has exactly one entry per
    segment in `segment_spans(ctx.nfc_text, turn_roles)`, in order -- this
    is the coverage guarantee: nothing here can shrink that list. `issues`
    reports label-quality problems (a malformed label, a segment_index with
    no matching segment, a duplicate) for QA -- diagnostic only, never a
    reason to drop a segment."""
    by_index, issues = _validate_raw_labels(raw_labels)
    spans = segment_spans(ctx.nfc_text, turn_roles)
    valid_indices = set(range(len(spans)))
    for idx in list(by_index):
        if idx not in valid_indices:
            issues.append(LabelIssue(idx, "segment_index_out_of_range",
                                     f"no segment at index {idx} (transcript has {len(spans)})"))
            del by_index[idx]

    records = []
    for i, (start, end, role) in enumerate(spans):
        text = ctx.nfc_text[start:end]
        embedding_text, transform = fold_newlines(text)
        label = by_index.get(i)
        if label is None:
            issues.append(LabelIssue(i, "segment_unlabeled", "no valid label for this segment_index"))
        record = {
            "document_id": ctx.document_id,
            "chunk_index": ctx.chunk_index,
            "page_range": ctx.page_range,
            "segment_index": i,
            "candidate_rank": i + 1,
            "verbatim_expression": text,
            "embedding_text": embedding_text,
            "text_transform": transform,
            "span_start": start,
            "span_end": end,
            "nfc_chunk_sha256": ctx.nfc_sha256,
            "raw_chunk_sha256": ctx.raw_sha256,
            "nfc_changed_codepoints": ctx.nfc_changed_codepoints,
            "context_window": ctx.nfc_text,
            "attribution": role,
            "expression_kind": label.expression_kind if label else None,
            "claim_mode": label.claim_mode if label else None,
            "epistemic_status": label.epistemic_status if label else None,
            "entity_anchors": list(label.entity_anchors) if label else [],
            "cult_relevant": label.cult_relevant if label else None,
            "relevance_note": label.relevance_note if label else "",
            "label_status": "model" if label else "missing",
            "screen_flags": ti.soft_flags(text) + ([ti.has_hard_corruption(text)] if ti.has_hard_corruption(text) else []),
            "chunk_integrity_score": ctx.chunk_integrity_score,
            "document_integrity_flag": ";".join(ctx.document_integrity_flags),
        }
        records.append(record)
    return records, issues
