"""Deterministic screening for the exhaustive interview extraction pipeline
(no model calls).

Every model-emitted candidate ends up in exactly one of two places: the
retained expression list, or the rejected list with a single rejection code
(the first failing rule, in the order below). Unlike screen_v2.py's S1-S15,
this rule set keeps only genuine structural/correctness checks -- nothing
here rejects on cult-relevance, on being short, or on being the
interviewer's own speech (that becomes the `attribution` label instead of
a reject code).

Rule order (first-failing-rule wins):
  not_verbatim                   span_cuts_word
  span_outside_known_turn        span_crosses_speaker_turn
  dangling_boundary               too_long
  integrity_*                     duplicate_or_overlapping_span

Attribution is resolved via `turn_spans()`, which pairs the blank-line
-separated paragraphs of a unit's (already label-free -- see
interview_chunking.py) text with the `turn_roles` list interview_chunking.py
built alongside it, positionally. There is no `Interviewer:`/`Interviewee:`
label left anywhere in the text for a candidate span to contain, so unlike
screen_v2.py there is no `span_includes_speaker_label` rule here, and no
`span_in_transcript_header`/`span_in_transcript_notes` codes either -- the
header and transcriber-notes text were already dropped at chunking time,
never reaching a unit's text at all.

Reused, unmodified: screen_v2.prepare_chunk/resolve_span/fold_newlines/
screen_domain_terms; text_integrity.has_hard_corruption/
ligature_substitution_present/soft_flags. The two small boundary checks
below (span_cuts_word, dangling_boundary) are not exposed as standalone
functions in screen_v2.py, so they are re-implemented here (a handful of
lines) rather than reaching into that module's private internals.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from thesis_corpus import interview_extraction_schema as schema
from thesis_corpus import text_integrity as ti
from thesis_corpus.extraction_v2_schema import DANGLING_END_WORDS, DANGLING_START_WORDS, MAX_WORDS
from thesis_corpus.screen_v2 import ChunkContext, fold_newlines, prepare_chunk, resolve_span, screen_domain_terms

__all__ = [
    "ChunkContext", "prepare_chunk", "resolve_span", "fold_newlines", "screen_domain_terms",
    "turn_spans", "ScreenOutcome", "screen_candidate", "screen_unit",
]

_BLANK_LINE_RE = re.compile(r"\n\s*\n")
_WORD_STRIP_RE = re.compile(r"^[^\w]+|[^\w]+$")
_TRAILING_PUNCT_RE = re.compile(r"[.!?\"”»’')\]]+$")
_DANGLING_PUNCT_END_RE = re.compile(r"[,;:(\[]$")
_OPENING_QUOTE_END_RE = re.compile(r"[\"“«‘]$")
_LONE_CAPITAL_END_RE = re.compile(r"(?:^|\s)[A-Z]\.?$")


def _bare(token: str) -> str:
    return _WORD_STRIP_RE.sub("", token)


def turn_spans(nfc_text: str, turn_roles: list[str]) -> list[tuple[int, int, str]]:
    """Pairs each blank-line-separated paragraph of `nfc_text` with its role
    from `turn_roles`, positionally and in order -- interview_chunking.py
    joins kept turns with exactly this separator, in this same order, so a
    paragraph count mismatch would indicate a chunking bug, not a normal
    case (defensively, any unmatched trailing paragraphs are dropped, and
    any unmatched trailing roles are simply unused -- neither should ever
    happen with unit text this module itself produced)."""
    spans, pos = [], 0
    for part in _BLANK_LINE_RE.split(nfc_text):
        idx = nfc_text.find(part, pos) if part else pos
        if idx < 0:
            idx = pos
        spans.append((idx, idx + len(part)))
        pos = idx + len(part)
    return [(s, e, role) for (s, e), role in zip(spans, turn_roles)]


@dataclass
class ScreenOutcome:
    rejection_code: str | None
    detail: str = ""
    flags: list[str] = field(default_factory=list)
    span: tuple[int, int] | None = None
    text: str | None = None
    attribution: str | None = None


def screen_candidate(cand: schema.CandidateFull, ctx: ChunkContext, turn_roles: list[str]) -> ScreenOutcome:
    flags: list[str] = []

    span = resolve_span(ctx.nfc_text, cand.verbatim_expression)
    if span is None:
        return ScreenOutcome("not_verbatim", cand.verbatim_expression[:120])
    start, end = span
    text = ctx.nfc_text[start:end]

    if (start > 0 and ctx.nfc_text[start - 1].isalnum()) or (end < len(ctx.nfc_text) and ctx.nfc_text[end].isalnum()):
        return ScreenOutcome("span_cuts_word", "alphanumeric character adjacent to span boundary", span=span, text=text)

    # Structural speaker-turn resolution -- the transcript decides who spoke;
    # the model is never asked and never overrides this. No header/notes
    # text can ever appear here: both were dropped when the unit's text was
    # built (interview_chunking.py), before this module ever sees it.
    overlapping = [(s, e, role) for s, e, role in turn_spans(ctx.nfc_text, turn_roles) if start < e and end > s]
    roles = {role for _, _, role in overlapping}
    if len(overlapping) > 1:
        return ScreenOutcome("span_crosses_speaker_turn", f"spans {sorted(roles)}", span=span, text=text)
    if not overlapping:
        return ScreenOutcome("span_outside_known_turn", "", span=span, text=text)
    attribution = overlapping[0][2]

    tokens = text.split()
    first = _bare(tokens[0]).lower() if tokens else ""
    last_raw = tokens[-1] if tokens else ""
    last = _bare(_TRAILING_PUNCT_RE.sub("", last_raw)).lower()
    if first in DANGLING_START_WORDS:
        return ScreenOutcome("dangling_boundary", f"starts with {first!r}", span=span, text=text)
    if last in DANGLING_END_WORDS:
        return ScreenOutcome("dangling_boundary", f"ends with {last!r}", span=span, text=text)
    if _DANGLING_PUNCT_END_RE.search(text) or _OPENING_QUOTE_END_RE.search(text):
        return ScreenOutcome("dangling_boundary", f"ends with {text[-1]!r}", span=span, text=text)
    if _LONE_CAPITAL_END_RE.search(text):
        return ScreenOutcome("dangling_boundary", "ends with a lone capital letter", span=span, text=text)

    words = len(tokens)
    if words > MAX_WORDS:
        return ScreenOutcome("too_long", f"{words} words", span=span, text=text)

    hard = ti.has_hard_corruption(text)
    if hard:
        return ScreenOutcome(hard, "", span=span, text=text)
    for region_start, region_end in ctx.corrupted_regions:
        if start < region_end and end > region_start:
            return ScreenOutcome("integrity_cipher_region", f"overlaps corrupted region {region_start}-{region_end}",
                                 span=span, text=text)
    if "ligature_substitution" in ctx.document_integrity_flags and ti.ligature_substitution_present(text):
        return ScreenOutcome("integrity_ligature_substitution", "", span=span, text=text)
    flags.extend(ti.soft_flags(text))

    return ScreenOutcome(None, "", flags=flags, span=span, text=text, attribution=attribution)


def screen_unit(raw_candidates: list, ctx: ChunkContext, seen_document_texts: set[str],
                turn_roles: list[str]) -> tuple[list[dict], list[dict]]:
    """Returns (retained_records, rejected_records). `seen_document_texts`
    is mutated: retained texts are added so a later unit of the same
    document (only possible via interview_chunking's turn-aligned split
    fallback) cannot retain an identical expression twice. No per-chunk cap
    (unlike screen_v2.screen_chunk's S15): exhaustive segmentation has no
    fixed limit on how many expressions a unit may contribute."""
    retained: list[dict] = []
    rejected: list[dict] = []
    kept: list[tuple[int, schema.CandidateFull, ScreenOutcome]] = []

    for rank, raw in enumerate(raw_candidates, start=1):
        cand, code, detail = schema.validate_candidate(raw)
        if cand is None:
            rejected.append(_rejected_record(ctx, rank, raw, code, detail))
            continue
        outcome = screen_candidate(cand, ctx, turn_roles)
        if outcome.rejection_code:
            rejected.append(_rejected_record(ctx, rank, raw, outcome.rejection_code, outcome.detail,
                                             span=outcome.span, resolved_text=outcome.text))
            continue

        start, end = outcome.span
        overlap = next(((r, o) for r, _c, o in kept if start < o.span[1] and end > o.span[0]), None)
        if overlap is not None:
            rejected.append(_rejected_record(ctx, rank, raw, "duplicate_or_overlapping_span",
                                             f"overlaps rank {overlap[0]}", span=outcome.span, resolved_text=outcome.text))
            continue
        if outcome.text in seen_document_texts:
            rejected.append(_rejected_record(ctx, rank, raw, "duplicate_or_overlapping_span",
                                             "identical text already retained in this document",
                                             span=outcome.span, resolved_text=outcome.text))
            continue
        kept.append((rank, cand, outcome))
        seen_document_texts.add(outcome.text)

    for rank, cand, outcome in kept:
        retained.append(_retained_record(ctx, rank, cand, outcome))
    return retained, rejected


def _rejected_record(ctx: ChunkContext, rank: int, raw, code: str, detail: str,
                     span: tuple[int, int] | None = None, resolved_text: str | None = None) -> dict:
    return {
        "document_id": ctx.document_id,
        "chunk_index": ctx.chunk_index,
        "candidate_rank": rank,
        "rejection_code": code,
        "rule_detail": detail,
        "span_start": span[0] if span else None,
        "span_end": span[1] if span else None,
        "resolved_text": resolved_text,
        "raw_candidate": raw,
    }


def _retained_record(ctx: ChunkContext, rank: int, cand: schema.CandidateFull, outcome: ScreenOutcome) -> dict:
    start, end = outcome.span
    embedding_text, transform = fold_newlines(outcome.text)
    return {
        "document_id": ctx.document_id,
        "chunk_index": ctx.chunk_index,
        "page_range": ctx.page_range,
        "candidate_rank": rank,
        "verbatim_expression": outcome.text,
        "embedding_text": embedding_text,
        "text_transform": transform,
        "span_start": start,
        "span_end": end,
        "nfc_chunk_sha256": ctx.nfc_sha256,
        "raw_chunk_sha256": ctx.raw_sha256,
        "nfc_changed_codepoints": ctx.nfc_changed_codepoints,
        "context_window": ctx.nfc_text,
        "expression_kind": cand.expression_kind,
        "attribution": outcome.attribution,
        "claim_mode": cand.claim_mode,
        "epistemic_status": cand.epistemic_status,
        "entity_anchors": list(cand.entity_anchors),
        "cult_relevant": cand.cult_relevant,
        "relevance_note": cand.relevance_note,
        "model_verbatim_expression": cand.verbatim_expression,
        "screen_flags": list(outcome.flags),
        "chunk_integrity_score": ctx.chunk_integrity_score,
        "document_integrity_flag": ";".join(ctx.document_integrity_flags),
    }
