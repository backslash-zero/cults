"""Deterministic screening for extraction v2 (no model calls).

Every model-emitted candidate ends up in exactly one of two places: the
retained expression list, or the rejected list with a single rejection
code (the first failing rule, in the order below). Non-fatal observations
go into `screen_flags` on retained rows and never reject anything.

Chunk pre-screen (before the model call, source-configured):
  C1 chunk_too_short                  C2 chunk_is_bibliography_or_index
Candidate screen (after the model call):
  S1 missing_required_field           S2 non_boolean_validation_field
  S3 validation_flag_false            S4 not_verbatim
  S5 span_cuts_word                   S6 dangling_boundary
  S7 too_long / short_fragment_no_referent
  S8 integrity_* (high-confidence corruption only)
  S9 heading_or_scaffold              S10 meta_discourse
  S11 standalone_personal_name        S12 interviewer_utterance
  S13 citation_dominated              S14 duplicate_or_overlapping_span
  S15 exceeds_per_chunk_cap

Text model: `raw_text` is the chunker's output; `nfc_text` (NFC of raw)
is what the model saw and the only matching target; a verbatim expression
is an exact contiguous substring of `nfc_text` at [span_start, span_end).
No repair, no transliteration, no rewriting anywhere in this module. The
single approved transform is fold_newlines() (a PDF line break inside a span
becomes one space in embedding_text, recorded as text_transform =
"newline_to_space"); verbatim_expression always keeps the source bytes.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from thesis_corpus import extraction_v2_schema as schema
from thesis_corpus import text_integrity as ti

# ---------------------------------------------------------------------------
# Chunk context
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD_STRIP_RE = re.compile(r"^[^\w]+|[^\w]+$")
_HEADING_KEYWORD_RE = re.compile(
    r"^(?:" + "|".join(re.escape(k) for k in schema.HEADING_KEYWORDS) + r")(?![\w-])", re.IGNORECASE,
)
_PAGE_FURNITURE_RE = re.compile(r"^(?:pp?\.\s*[\divx]+|\d+|doi:\S*|ISBN[\s:]|https?://\S+)$", re.IGNORECASE)
_META_DISCOURSE_RE = re.compile(
    r"^(?:this|the present|in this|the following) (?:chapter|article|book|section|essay|paper|volume)\b"
    r"|will be the subject of this (?:chapter|article|section)"
    r"|^(?:we|i) (?:will|shall) (?:now )?(?:turn|return|argue|show|examine) ",
    re.IGNORECASE,
)
_CITATION_RE = re.compile(r"\([^()]*?\b(?:1[5-9]|20)\d{2}[a-z]?\b[^()]*\)")
# A reference entry: "Surname, Initials/First (1989) Title..." -- surname
# followed by a comma, and a year within the first 80 characters -- or a
# line carrying a publisher/journal marker together with a year. Ordinary
# prose that merely starts with a proper noun and mentions a year
# ("State Shinto shaped the suppression ... in 1895.") must NOT match.
_REFERENCE_LINE_RE = re.compile(
    r"^[A-Z][\w'’\-]+,\s[^\n]{0,70}?\(?\b(?:1[5-9]|20)\d{2}[a-z]?\b\)?"
)
_PUBLISHER_MARKER_RE = re.compile(
    r"University Press|\bPress\b|\bpp?\.\s*\d|\bvol\.|\b[Ee]ds?\.\s|\bJournal\b|\bReview\b|\bRoutledge\b|\bBlackwell\b"
)
_YEAR_RE = re.compile(r"\b(?:1[5-9]|20)\d{2}[a-z]?\b")
_LONE_CAPITAL_END_RE = re.compile(r"(?:^|\s)[A-Z]\.?$")
_OPENING_QUOTE_END_RE = re.compile(r"[\"“«‘]$")
_DANGLING_PUNCT_END_RE = re.compile(r"[,;:(\[]$")
_TRAILING_PUNCT_RE = re.compile(r"[.!?\"”»’')\]]+$")
# A personal-name token: Capitalised-then-lowercase (Dawson, Hervieu-Léger)
# or an initial (L., J). All-caps tokens (CIA, ISKCON) are acronyms, not names.
_NAME_TOKEN_RE = re.compile(r"^(?:[A-ZÀ-Ý][a-zà-ÿ'’\-]+|[A-Z]\.?)$")
_LEXICON = frozenset(schema.DOMAIN_LEXICON)


@dataclass
class ChunkContext:
    document_id: str
    chunk_index: int
    page_range: list[int]
    source: str
    raw_text: str
    nfc_text: str
    nfc_changed_codepoints: int
    raw_sha256: str
    nfc_sha256: str
    document_integrity_flags: list[str] = field(default_factory=list)
    corrupted_regions: list[tuple[int, int]] = field(default_factory=list)
    chunk_integrity_score: float = 0.0
    line_spans: list[tuple[int, int]] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return len(self.nfc_text.split())


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _line_spans(text: str) -> list[tuple[int, int]]:
    spans, pos = [], 0
    for line in text.split("\n"):
        spans.append((pos, pos + len(line)))
        pos += len(line) + 1
    return spans


def chunk_integrity_score(nfc_text: str) -> float:
    """Flag-class hits per 100 words (isolated capitals, spaced accents,
    replacement/PUA/control/mojibake characters, mixed-script tokens)."""
    counts = ti.count_character_classes(nfc_text)
    hits = (counts["isolated_capital"] + counts["spaced_accent"] + counts["replacement_char"]
            + counts["private_use"] + counts["control_char"] + counts["mojibake_seq"]
            + counts["mixed_script_tokens"])
    return round(100.0 * hits / max(len(nfc_text.split()), 1), 3)


def prepare_chunk(
    document_id: str, chunk_index: int, page_range: list[int], raw_text: str, source: str,
    document_integrity_flags: list[str] | None = None,
    corrupted_line_texts: set[str] | frozenset[str] = frozenset(),
) -> ChunkContext:
    """Builds the context: NFC text, hashes, corrupted regions resolved by
    matching audit-flagged line texts (already NFC + collapsed, exactly as
    the audit stored them) against this chunk's lines."""
    nfc_text, changed = ti.nfc_normalize(raw_text)
    flags = list(document_integrity_flags or [])
    line_spans = _line_spans(nfc_text)
    regions: list[tuple[int, int]] = []
    if corrupted_line_texts and ({"cmap_cipher", "private_use_cipher"} & set(flags)):
        for (start, end), line in zip(line_spans, nfc_text.split("\n")):
            if line.strip() and line.strip() in corrupted_line_texts:
                regions.append((start, end))
    return ChunkContext(
        document_id=document_id, chunk_index=chunk_index, page_range=list(page_range), source=source,
        raw_text=raw_text, nfc_text=nfc_text, nfc_changed_codepoints=changed,
        raw_sha256=_sha256(raw_text), nfc_sha256=_sha256(nfc_text),
        document_integrity_flags=flags, corrupted_regions=regions,
        chunk_integrity_score=chunk_integrity_score(nfc_text), line_spans=line_spans,
    )


# ---------------------------------------------------------------------------
# Chunk pre-screen
# ---------------------------------------------------------------------------

def pre_screen_chunk(ctx: ChunkContext, config: dict) -> tuple[str | None, str]:
    """(skip_code, detail) -- None when the chunk goes to the model."""
    min_words = config.get("min_chunk_words")
    if min_words is not None and ctx.word_count < min_words:
        return "chunk_too_short", f"{ctx.word_count} words < {min_words}"
    share_threshold = config.get("bibliography_line_share")
    if share_threshold is not None:
        lines = [line.strip() for line in ctx.nfc_text.split("\n") if line.strip()]
        min_lines = config.get("bibliography_min_lines") or 3
        if len(lines) >= min_lines:
            ref_like = sum(1 for line in lines if is_reference_line(line))
            share = ref_like / len(lines)
            starts_with_list_heading = bool(lines) and lines[0].lower().split()[0] in (
                "references", "bibliography", "notes", "index", "further")
            # A list heading alone is not enough: an encyclopedia entry can
            # open with a two-line "Further reading" and continue with body
            # text. Require most lines (or, after a list heading, a clear
            # majority of the rest) to be reference entries.
            if share >= share_threshold or (starts_with_list_heading and share >= share_threshold * 0.75):
                return "chunk_is_bibliography_or_index", f"{ref_like}/{len(lines)} reference-like lines"
    return None, ""


def is_reference_line(line: str) -> bool:
    if _REFERENCE_LINE_RE.match(line):
        return True
    return bool(_PUBLISHER_MARKER_RE.search(line) and _YEAR_RE.search(line) and len(line) <= 300)


# ---------------------------------------------------------------------------
# Span resolution (S4) -- the v1 resolve_quote logic, on NFC text
# ---------------------------------------------------------------------------

def resolve_span(nfc_text: str, candidate: str) -> tuple[int, int] | None:
    candidate = candidate.strip()
    if not candidate:
        return None
    index = nfc_text.find(candidate)
    if index >= 0:
        return index, index + len(candidate)
    tokens = candidate.split()
    pattern = r"\s+".join(re.escape(token) for token in tokens)
    match = re.search(pattern, nfc_text, re.DOTALL)
    if match:
        return match.start(), match.end()
    return None


# ---------------------------------------------------------------------------
# Helpers for the candidate rules
# ---------------------------------------------------------------------------

def _tokens(text: str) -> list[str]:
    return text.split()


def _bare(token: str) -> str:
    return _WORD_STRIP_RE.sub("", token)


def lexicon_hit(text: str) -> bool:
    for token in re.split(r"[\s\-/]+", text):
        if _bare(token).lower() in _LEXICON:
            return True
    return False


def _is_capitalized(token: str) -> bool:
    bare = _bare(token)
    return len(bare) >= 2 and bare[0].isupper() and bare != "I" and not bare.isdigit()


def has_named_entity(span_text: str, ctx: ChunkContext, start: int) -> bool:
    """A capitalized token that is not merely sentence-initial."""
    tokens = _tokens(span_text)
    if any(_is_capitalized(t) for t in tokens[1:]):
        return True
    if tokens and _is_capitalized(tokens[0]):
        before = ctx.nfc_text[:start].rstrip()
        return bool(before) and before[-1] not in ".!?\n"
    return False


def _sentence_windows(ctx: ChunkContext) -> list[tuple[int, int]]:
    spans, pos = [], 0
    for part in _SENTENCE_SPLIT_RE.split(ctx.nfc_text):
        index = ctx.nfc_text.find(part, pos) if part else pos
        if index < 0:
            index = pos
        spans.append((index, index + len(part)))
        pos = index + len(part)
    return spans


def context_connected(ctx: ChunkContext, start: int, end: int,
                      window: int = schema.SHORT_SPAN_CONTEXT_SENTENCES) -> bool:
    """True when a domain-lexicon term occurs within ±`window` sentences of
    the span (the span itself excluded)."""
    sentences = _sentence_windows(ctx)
    hit_index = next((i for i, (s, e) in enumerate(sentences) if s <= start < e), None)
    if hit_index is None:
        return False
    for i in range(max(0, hit_index - window), min(len(sentences), hit_index + window + 1)):
        s, e = sentences[i]
        text = ctx.nfc_text[s:e]
        if i == hit_index:
            text = ctx.nfc_text[s:start] + " " + ctx.nfc_text[end:e]
        if lexicon_hit(text):
            return True
    return False


def is_title_case(text: str) -> bool:
    tokens = [_bare(t) for t in _tokens(text)]
    tokens = [t for t in tokens if t]
    if not tokens or len(tokens) > schema.TITLE_CASE_MAX_WORDS:
        return False
    content = [t for t in tokens if t.lower() not in schema.TITLE_CASE_STOPWORDS]
    if not content:
        return False
    return all(t[0].isupper() for t in content) and tokens[0][0].isupper()


def is_subtitled_title(text: str) -> bool:
    if ":" not in text or text.count(":") != 1:
        return False
    head, tail = (part.strip() for part in text.split(":", 1))
    if not head or not tail or re.search(r"[.!?;]$", text):
        return False
    return is_title_case(head) and is_title_case(tail)


def is_all_caps(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return len(letters) >= 2 and all(c.isupper() for c in letters)


def on_own_line(ctx: ChunkContext, start: int, end: int) -> bool:
    for line_start, line_end in ctx.line_spans:
        if line_start <= start and end <= line_end:
            return ctx.nfc_text[line_start:line_end].strip() == ctx.nfc_text[start:end].strip()
    return False


def run_in_heading(ctx: ChunkContext, text: str, end: int) -> bool:
    """'The Course of Growth. While the growth of cults...' -- a title-cased
    fragment ending in a period, followed on the same line by a new
    capitalised sentence."""
    if not text.endswith("."):
        return False
    after = ctx.nfc_text[end:end + 3].lstrip()
    return bool(after) and after[0].isupper()


def _unbalanced(text: str) -> bool:
    if text.count('"') % 2:
        return True
    if text.count("“") != text.count("”") or text.count("«") != text.count("»"):
        return True
    return text.count("(") != text.count(")")


# ---------------------------------------------------------------------------
# Candidate screen
# ---------------------------------------------------------------------------

@dataclass
class ScreenOutcome:
    rejection_code: str | None
    detail: str = ""
    flags: list[str] = field(default_factory=list)
    span: tuple[int, int] | None = None
    text: str | None = None


def screen_candidate(cand: schema.CandidateV2, ctx: ChunkContext) -> ScreenOutcome:
    flags: list[str] = []

    # S3
    for name in schema.VALIDATION_FLAG_FIELDS:
        if getattr(cand, name) is not True:
            return ScreenOutcome("validation_flag_false", name)

    # S4
    span = resolve_span(ctx.nfc_text, cand.verbatim_expression)
    if span is None:
        return ScreenOutcome("not_verbatim", cand.verbatim_expression[:120])
    start, end = span
    text = ctx.nfc_text[start:end]

    # S5
    if (start > 0 and ctx.nfc_text[start - 1].isalnum()) or (end < len(ctx.nfc_text) and ctx.nfc_text[end].isalnum()):
        return ScreenOutcome("span_cuts_word", "alphanumeric character adjacent to span boundary", span=span, text=text)

    # S6
    tokens = _tokens(text)
    first = _bare(tokens[0]).lower() if tokens else ""
    last_raw = tokens[-1] if tokens else ""
    last = _bare(_TRAILING_PUNCT_RE.sub("", last_raw)).lower()
    if first in schema.DANGLING_START_WORDS:
        return ScreenOutcome("dangling_boundary", f"starts with {first!r}", span=span, text=text)
    if last in schema.DANGLING_END_WORDS:
        return ScreenOutcome("dangling_boundary", f"ends with {last!r}", span=span, text=text)
    if _DANGLING_PUNCT_END_RE.search(text) or _OPENING_QUOTE_END_RE.search(text):
        return ScreenOutcome("dangling_boundary", f"ends with {text[-1]!r}", span=span, text=text)
    if _LONE_CAPITAL_END_RE.search(text):
        return ScreenOutcome("dangling_boundary", "ends with a lone capital letter", span=span, text=text)

    # S7
    words = len(tokens)
    if words > schema.MAX_WORDS:
        return ScreenOutcome("too_long", f"{words} words", span=span, text=text)
    if words > schema.LONG_WORDS:
        flags.append("long")
    hit = lexicon_hit(text)
    if words <= schema.SHORT_SPAN_MAX_WORDS:
        referent = hit or has_named_entity(text, ctx, start)
        if not referent:
            association_path = (
                cand.expression_kind in ("association_or_framing", "example_or_named_group")
                and cand.self_contained and cand.cult_relevant
                and context_connected(ctx, start, end)
            )
            if not association_path:
                return ScreenOutcome("short_fragment_no_referent", f"{words} words, no cult-related referent",
                                     span=span, text=text)
            flags.append("short_association")

    # S8 -- high-confidence corruption only
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
    if _unbalanced(text):
        flags.append("unbalanced_quotes_or_brackets")

    # S9
    stripped = text.strip()
    tagged_named = cand.expression_kind == "example_or_named_group"
    own_line = on_own_line(ctx, start, end)
    # A heading keyword ("Chapter", "Part II", "References") rejects when
    # the span is a line of its own or carries no domain term; "an
    # introduction to cults" inside a sentence is not a heading.
    if _HEADING_KEYWORD_RE.match(stripped) and (own_line or not hit):
        return ScreenOutcome("heading_or_scaffold", "heading keyword", span=span, text=text)
    if _PAGE_FURNITURE_RE.match(stripped):
        return ScreenOutcome("heading_or_scaffold", "page furniture", span=span, text=text)
    # A work title with a subtitle -- "The Cadre Ideal: Origins and Development of a
    # Political Cult" -- is Title Case on both sides of a colon. Named groups never
    # take this shape, so it rejects regardless of tag or domain term.
    if is_subtitled_title(stripped):
        return ScreenOutcome("heading_or_scaffold", "title with subtitle", span=span, text=text)
    # S11 -- checked before the title-case heading branch so a bare
    # "Lorne L. Dawson" gets the name code, not the heading code.
    if (2 <= len(tokens) <= schema.NAME_MAX_TOKENS and not hit and not tagged_named
            and all(_NAME_TOKEN_RE.match(_TRAILING_PUNCT_RE.sub("", t)) for t in tokens)):
        return ScreenOutcome("standalone_personal_name", "", span=span, text=text)

    if is_all_caps(stripped):
        heading_like = own_line and words <= schema.TITLE_CASE_MAX_WORDS and not tagged_named and not hit
        if heading_like:
            return ScreenOutcome("heading_or_scaffold", "all-caps line on its own", span=span, text=text)
        if not tagged_named and not hit:
            flags.append("possible_heading_or_acronym")
    elif is_title_case(stripped):
        no_terminal = not re.search(r"[.!?]$", stripped)
        heading_shaped = own_line or run_in_heading(ctx, stripped, end) or no_terminal
        if heading_shaped and not hit and not tagged_named:
            return ScreenOutcome("heading_or_scaffold", "title-cased fragment", span=span, text=text)
        if heading_shaped:
            flags.append("possible_heading_or_acronym")

    # S10
    if _META_DISCOURSE_RE.search(stripped):
        return ScreenOutcome("meta_discourse", "", span=span, text=text)

    # S12
    if cand.attribution == "interviewer":
        return ScreenOutcome("interviewer_utterance", "", span=span, text=text)

    # S13
    citation_chars = sum(len(m.group(0)) for m in _CITATION_RE.finditer(text))
    if len(text) and citation_chars / len(text) > schema.CITATION_DOMINANCE_SHARE:
        return ScreenOutcome("citation_dominated", f"{citation_chars}/{len(text)} chars", span=span, text=text)

    return ScreenOutcome(None, "", flags=flags, span=span, text=text)


# ---------------------------------------------------------------------------
# Chunk-level screening (S1/S2 validation, S14, S15) and record building
# ---------------------------------------------------------------------------

def screen_chunk(
    raw_candidates: list, ctx: ChunkContext, seen_document_texts: set[str],
    max_per_chunk: int = schema.MAX_PER_CHUNK,
) -> tuple[list[dict], list[dict]]:
    """Returns (retained_records, rejected_records). `seen_document_texts`
    is mutated: retained texts are added so a later chunk of the same
    document cannot retain an identical expression."""
    retained: list[dict] = []
    rejected: list[dict] = []
    provisional: list[tuple[int, schema.CandidateV2, ScreenOutcome]] = []

    for rank, raw in enumerate(raw_candidates, start=1):
        cand, code, detail = schema.validate_candidate(raw)
        if cand is None:
            rejected.append(_rejected_record(ctx, rank, raw, code, detail))
            continue
        outcome = screen_candidate(cand, ctx)
        if outcome.rejection_code:
            rejected.append(_rejected_record(ctx, rank, raw, outcome.rejection_code, outcome.detail,
                                             span=outcome.span, resolved_text=outcome.text))
            continue
        provisional.append((rank, cand, outcome))

    # S14: overlapping spans within the chunk (keep the higher-ranked, i.e.
    # earlier) and exact duplicates within the document.
    kept: list[tuple[int, schema.CandidateV2, ScreenOutcome]] = []
    for rank, cand, outcome in provisional:
        start, end = outcome.span
        overlap = next(((r, o) for r, _c, o in kept if start < o.span[1] and end > o.span[0]), None)
        if overlap is not None:
            rejected.append(_rejected_record(ctx, rank, cand.model_dump(), "duplicate_or_overlapping_span",
                                             f"overlaps rank {overlap[0]}", span=outcome.span, resolved_text=outcome.text))
            continue
        if outcome.text in seen_document_texts:
            rejected.append(_rejected_record(ctx, rank, cand.model_dump(), "duplicate_or_overlapping_span",
                                             "identical text already retained in this document",
                                             span=outcome.span, resolved_text=outcome.text))
            continue
        kept.append((rank, cand, outcome))
        seen_document_texts.add(outcome.text)

    # S15: cap, preserving the model's own strongest-first order.
    for position, (rank, cand, outcome) in enumerate(kept):
        if position >= max_per_chunk:
            seen_document_texts.discard(outcome.text)
            rejected.append(_rejected_record(ctx, rank, cand.model_dump(), "exceeds_per_chunk_cap",
                                             f"position {position + 1} > {max_per_chunk}",
                                             span=outcome.span, resolved_text=outcome.text))
            continue
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


_NEWLINE_RUN_RE = re.compile(r"[ \t]*\n+[ \t]*")


def fold_newlines(text: str) -> tuple[str, str]:
    """The one approved text transform (2026-09-09): a PDF line break inside a
    span becomes a single space for embedding. Returns (embedding_text,
    text_transform) with text_transform "none" when nothing changed. The
    verbatim span is never altered; a hyphen before the break is kept as is
    ("re-\nligious" -> "re- ligious") and stays flagged contains_hyphen_linebreak."""
    folded = _NEWLINE_RUN_RE.sub(" ", text)
    return (folded, "newline_to_space") if folded != text else (text, "none")


def _retained_record(ctx: ChunkContext, rank: int, cand: schema.CandidateV2, outcome: ScreenOutcome) -> dict:
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
        "attribution": cand.attribution,
        "claim_mode": cand.claim_mode,
        "epistemic_status": cand.epistemic_status,
        "entity_anchors": list(cand.entity_anchors),
        "self_contained": True,
        "cult_relevant": True,
        "textually_intelligible": True,
        "single_coherent_expression": True,
        "relevance_note": cand.relevance_note,
        "model_verbatim_expression": cand.verbatim_expression,
        "screen_flags": list(outcome.flags),
        "chunk_integrity_score": ctx.chunk_integrity_score,
        "document_integrity_flag": ";".join(ctx.document_integrity_flags),
    }


# ---------------------------------------------------------------------------
# Domain terms (diagnostic inventory only)
# ---------------------------------------------------------------------------

def screen_domain_terms(terms: list, ctx: ChunkContext) -> tuple[list[str], list[dict]]:
    kept: list[str] = []
    dropped: list[dict] = []
    seen: set[str] = set()
    lowered = ctx.nfc_text.lower()
    for term in terms:
        if not isinstance(term, str) or not term.strip():
            dropped.append({"term": term, "reason": "empty_or_non_string"})
            continue
        normalized, _ = ti.nfc_normalize(term.strip())
        if normalized.lower() not in lowered:
            dropped.append({"term": normalized, "reason": "term_not_in_chunk"})
            continue
        if normalized.lower() in seen:
            continue
        seen.add(normalized.lower())
        kept.append(normalized)
    return kept, dropped


# ---------------------------------------------------------------------------
# Independent re-check used by the pilot's validation step
# ---------------------------------------------------------------------------

def recheck_retained_record(record: dict, ctx: ChunkContext) -> list[str]:
    """Hard-zero conditions on a retained row, evaluated from scratch."""
    problems = []
    expected_embedding, expected_transform = fold_newlines(record["verbatim_expression"])
    if record["embedding_text"] != expected_embedding:
        problems.append("embedding_text is not the verbatim span (with line breaks folded)")
    if record["text_transform"] != expected_transform:
        problems.append(f"text_transform {record['text_transform']!r} != expected {expected_transform!r}")
    if ctx.nfc_text[record["span_start"]:record["span_end"]] != record["verbatim_expression"]:
        problems.append("span does not resolve to verbatim_expression")
    if record["nfc_chunk_sha256"] != ctx.nfc_sha256:
        problems.append("nfc_chunk_sha256 mismatch")
    if record["attribution"] == "interviewer":
        problems.append("interviewer attribution retained")
    if ti.has_hard_corruption(record["verbatim_expression"]):
        problems.append("hard corruption in retained text")
    for region_start, region_end in ctx.corrupted_regions:
        if record["span_start"] < region_end and record["span_end"] > region_start:
            problems.append("retained span inside corrupted region")
    if not all(record.get(k) is True for k in schema.VALIDATION_FLAG_FIELDS):
        problems.append("validation flag not true")
    return problems
