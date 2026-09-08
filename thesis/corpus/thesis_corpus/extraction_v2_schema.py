"""Extraction v2: prompt, structured-output schema, and screening constants.

v1 (ollama_client.SYSTEM_PROMPT / AnnotationItem) let the model *derive* an
embedding_text from a verbatim source_quote; on the literature archive
that produced a model rewrite in 32% of rows and a sub-span in another
51%, and every truncation / corruption found by the 2026-09-08 fidelity
audit sat in that rewrite path. v2 asks the model for one thing only -- an
exact contiguous span -- and sets embedding_text = verbatim_expression in
code. Selection is deliberately conservative (prefer false negatives).

Nothing in v1 is modified: this module is additive and versioned, and every
v2 output records EXTRACTION_VERSION and PROMPT_SHA256.

The user-message schema description is generated from the Pydantic model
(schema_description()), so the prompt and the validator cannot drift.
"""
from __future__ import annotations

import hashlib
import typing
from typing import Literal

from pydantic import BaseModel, ConfigDict, StrictBool, ValidationError

EXTRACTION_VERSION = "2.0.0-pilot"

# ---------------------------------------------------------------------------
# Enumerations. claim_mode / epistemic_status are byte-identical to v1 so the
# geometric toolkit's filters keep working; attribution gains "interviewer"
# (v1 folded it into "unspecified").
# ---------------------------------------------------------------------------
EXPRESSION_KINDS = (
    "claim", "definition", "criterion", "example_or_named_group",
    "association_or_framing", "question_or_reflection",
)
ATTRIBUTIONS = ("author", "cited_author", "participant", "interviewer", "institution", "journalist", "unspecified")
CLAIM_MODES = ("direct_statement", "attributed_statement", "quotation", "definition", "question_or_reflection", "other")
EPISTEMIC_STATUSES = ("asserted", "qualified", "contested", "negated", "speculative")
VALIDATION_FLAG_FIELDS = ("self_contained", "cult_relevant", "textually_intelligible", "single_coherent_expression")

# ---------------------------------------------------------------------------
# Screening constants (all recorded in every arm's config.json)
# ---------------------------------------------------------------------------
MAX_PER_CHUNK = 4
MAX_WORDS = 50            # > MAX_WORDS -> reject too_long
LONG_WORDS = 35           # > LONG_WORDS -> non-fatal flag "long"
SHORT_SPAN_MAX_WORDS = 6  # <= this -> the short-span referent rule (S7) applies
SHORT_SPAN_CONTEXT_SENTENCES = 2
TITLE_CASE_MAX_WORDS = 8
NAME_MAX_TOKENS = 4
CITATION_DOMINANCE_SHARE = 0.40

# Source-configured chunk pre-screen. Literature chunks are 300-700-word
# prose so a hard floor is safe there; interview participant turns must
# never be dropped on length (min_chunk_words=None disables C1).
PRE_SCREEN_BY_SOURCE: dict[str, dict] = {
    "literature": {"min_chunk_words": 40, "bibliography_line_share": 0.6, "bibliography_min_lines": 3},
    "miviludes": {"min_chunk_words": None, "bibliography_line_share": 0.6, "bibliography_min_lines": 3},  # to be set at migration
    "interviews": {"min_chunk_words": None, "bibliography_line_share": None, "bibliography_min_lines": None},
}

# Used ONLY as the referent test for spans of <= SHORT_SPAN_MAX_WORDS words
# and for the S9/S11 "is this a name/heading or a domain term" distinction --
# never as a relevance filter for longer spans.
DOMAIN_LEXICON = (
    "cult", "cults", "cultic", "cultism", "cultist", "cultists",
    "sect", "sects", "sectarian", "secte", "sectes", "sectaire", "sectaires",
    "nrm", "nrms", "guru", "gurus", "gourou", "gourous",
    "brainwashing", "brainwashed", "emprise", "dérive", "dérives",
    "charisma", "charismatic", "charismatique", "charismatiques",
    "leader", "leaders", "adept", "adepts", "adepte", "adeptes",
)

DANGLING_START_WORDS = ("and", "or", "but", "which", "that", "whereas", "et", "ou", "mais", "dont", "que")
DANGLING_END_WORDS = (
    "and", "or", "but", "that", "which", "of", "to", "with", "the", "a", "an", "is", "are", "was",
    "be", "can", "will", "de", "la", "le", "les", "et", "ou", "que", "qui", "à", "en", "du", "des",
    "pour", "par",
)
HEADING_KEYWORDS = (
    "chapter", "part", "section", "introduction", "conclusion", "notes", "references",
    "bibliography", "index", "table", "figure", "appendix", "further reading",
)
TITLE_CASE_STOPWORDS = {"of", "the", "and", "in", "a", "an", "for", "to", "on", "de", "la", "le", "les", "et", "du", "des", "à"}

# Ollama request options
OLLAMA_OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 8192}
OLLAMA_THINK = False

# ---------------------------------------------------------------------------
# Prompt (verbatim from the approved plan)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_V2 = """\
You are assisting a thesis that studies how "cults", "sects", sectarian drift and new
religious movements are described, defined, disputed, exemplified and framed in scholarly,
institutional and interview text.

Your task is NOT to decide whether anything in the text is really a cult. Your task is to
select, from the supplied text, the few expressions that are genuinely worth keeping as
standalone data points, and to copy them exactly.

SELECTION STANDARD — be conservative. Prefer returning nothing over returning something
doubtful. Read the whole text first, then keep at most 4 expressions, usually 0 to 2. A
kept expression must satisfy ALL of the following:
1. Self-contained: a reader understands what it claims, describes or associates without the
   previous sentence, the question that prompted it, or the surrounding paragraph.
2. Cult-relevant: it materially describes, defines, disputes, exemplifies, frames or
   associates cults, sects, sectarian drift, new religious movements, group authority or
   control, manipulation, spiritual abuse, a relevant named group or leader, or an
   everyday use of "cult".
3. Textually intelligible: no garbled characters, split or misspelled words, stray symbols,
   detached accents, or broken syntax. Never repair such text; skip it.
4. One coherent expression: one claim, definition, criterion, association or named
   example — not a chain of claims, a paragraph-long quotation, or several ideas.
5. Complete: a whole clause or sentence, or a complete short noun phrase. Never cut a
   clause in the middle.

Do not output a broad sentence merely because it contains one cult-related term. If the
relevant term cannot form a complete, self-contained, cult-relevant source expression by
itself, omit it from expressions. It may appear only in domain_terms.

DO NOT KEEP: chapter or section titles and headings; author names, bibliographic entries,
page furniture; acknowledgements, hedges or replies that carry no content on their own
("Well, perhaps to some extent."); sentences announcing what a chapter or article will do;
administrative, procedural or statistical statements about an institution's activity;
descriptions of services, delegations or budgets; general history or theology that is not
about the cult/sect question; questions asked by an interviewer.

WHEN THE TEXT OFFERS SEVERAL FORMULATIONS of the same idea, keep the most precise one and
drop the vaguer or broader one. If a long sentence contains one relevant clause, keep only
that clause (it must still be a contiguous span).

LENGTH: normally 4 to 35 words. Never more than 50 words. A span of 1 to 3 words is
allowed only when it names a specific group, movement, leader, concept or a lay
association that is meaningful on its own (e.g. a named group, "destructive sects").

COPY RULE: verbatim_expression must be an exact, contiguous, character-for-character
substring of the supplied text, in the text's own language. Do not translate, paraphrase,
normalise spelling, fix typos, add or remove words, or merge separate spans.

ORDER: list kept expressions from strongest to weakest (most precise and most clearly
cult-relevant first).

For every kept expression also report: expression_kind; attribution (who is speaking:
author, cited_author, participant, interviewer, institution, journalist, unspecified);
claim_mode; epistemic_status (asserted = stated as true; qualified = stated with explicit
limits or conditions; contested = presented as disputed or debated; negated = denied or
rejected; speculative = offered as possibility or uncertainty); entity_anchors (named
groups, leaders, movements mentioned in the span, verbatim); the four validation flags,
which must all be true for anything you keep; and a relevance_note of at most 15 words
saying why it is worth keeping.

Separately, list domain_terms: cult-related concepts or named groups that appear in the
text (verbatim, as they appear), even if no expression around them is worth keeping.

Return only a JSON object matching the schema. No commentary."""

PROMPT_SHA256 = hashlib.sha256(SYSTEM_PROMPT_V2.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Pydantic models. CandidateV2 is strict (booleans must be JSON booleans,
# enums exact, no extra keys) and is validated PER CANDIDATE so that one
# malformed candidate becomes an S1/S2 rejection rather than a chunk-level
# model failure. ChunkEnvelopeV2 is the permissive top-level shape.
# ---------------------------------------------------------------------------
class CandidateV2(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    verbatim_expression: str
    expression_kind: Literal[EXPRESSION_KINDS]  # type: ignore[valid-type]
    attribution: Literal[ATTRIBUTIONS]  # type: ignore[valid-type]
    claim_mode: Literal[CLAIM_MODES]  # type: ignore[valid-type]
    epistemic_status: Literal[EPISTEMIC_STATUSES]  # type: ignore[valid-type]
    entity_anchors: list[str] = []
    self_contained: StrictBool
    cult_relevant: StrictBool
    textually_intelligible: StrictBool
    single_coherent_expression: StrictBool
    relevance_note: str = ""


class ChunkAnnotationV2(BaseModel):
    """Full strict shape -- used for the JSON schema sent to Ollama."""
    model_config = ConfigDict(extra="forbid")

    chunk_relevance: Literal["relevant", "not_relevant"]
    expressions: list[CandidateV2] = []
    domain_terms: list[str] = []


class ChunkEnvelopeV2(BaseModel):
    """Permissive top-level shape used when *reading* a response: candidates
    are validated one by one afterwards (see validate_candidate)."""
    model_config = ConfigDict(extra="ignore")

    chunk_relevance: Literal["relevant", "not_relevant"]
    expressions: list[dict] = []
    domain_terms: list = []


def response_json_schema() -> dict:
    return ChunkAnnotationV2.model_json_schema()


def validate_candidate(raw: object) -> tuple[CandidateV2 | None, str | None, str]:
    """Returns (candidate, rejection_code, detail). Codes: S1
    missing_required_field, S2 non_boolean_validation_field, or
    invalid_candidate_shape for anything else Pydantic rejects."""
    if not isinstance(raw, dict):
        return None, "invalid_candidate_shape", f"candidate is {type(raw).__name__}, not an object"
    try:
        return CandidateV2.model_validate(raw), None, ""
    except ValidationError as e:
        errors = e.errors()
        for err in errors:
            loc = ".".join(str(p) for p in err.get("loc", ()))
            if err.get("type") == "missing":
                return None, "missing_required_field", loc
        for err in errors:
            loc = ".".join(str(p) for p in err.get("loc", ()))
            if loc in VALIDATION_FLAG_FIELDS:
                return None, "non_boolean_validation_field", f"{loc}: {err.get('msg')}"
        first = errors[0]
        return None, "invalid_candidate_shape", f"{'.'.join(str(p) for p in first.get('loc', ()))}: {first.get('msg')}"


def _literal_values(annotation) -> tuple | None:
    if typing.get_origin(annotation) is Literal:
        return typing.get_args(annotation)
    return None


def schema_description() -> str:
    """Human-readable schema for the user message, generated from the models."""
    lines = ["Return a JSON object with this exact shape:", "{",
             '  "chunk_relevance": "relevant" | "not_relevant",',
             '  "expressions": [', "    {"]
    for name, field in CandidateV2.model_fields.items():
        values = _literal_values(field.annotation)
        if values:
            rendered = " | ".join(f'"{v}"' for v in values)
        elif name == "verbatim_expression":
            rendered = '"Exact contiguous substring of the supplied text."'
        elif name == "entity_anchors":
            rendered = '["named group or leader mentioned in the span", ...]'
        elif name == "relevance_note":
            rendered = '"At most 15 words on why this is worth keeping."'
        elif name in VALIDATION_FLAG_FIELDS:
            rendered = "true | false   (JSON boolean, not a string)"
        else:
            rendered = "..."
        lines.append(f'      "{name}": {rendered},')
    lines[-1] = lines[-1].rstrip(",")
    lines += ["    }", "  ],",
              '  "domain_terms": ["cult-related concept or named group appearing verbatim in the text", ...]',
              "}",
              "expressions may be empty. Order expressions from strongest to weakest. "
              "Every enumerated field must use exactly one of the listed literal values.",
              'If nothing is worth keeping, return {"chunk_relevance": "not_relevant", "expressions": [], "domain_terms": [...]}.']
    return "\n".join(lines)


def user_message(document_id: str, page_range: list[int], chunk_index: int, nfc_chunk_text: str) -> str:
    return (
        f"document_id: {document_id}\n"
        f"page_range: {page_range}\n"
        f"chunk_index: {chunk_index}\n\n"
        f"{schema_description()}\n\n"
        f"Text:\n{nfc_chunk_text}"
    )


def screening_constants() -> dict:
    """Everything a run's config.json records so a reader can reproduce the screen."""
    return {
        "extraction_version": EXTRACTION_VERSION,
        "prompt_sha256": PROMPT_SHA256,
        "max_per_chunk": MAX_PER_CHUNK,
        "max_words": MAX_WORDS,
        "long_words_flag": LONG_WORDS,
        "short_span_max_words": SHORT_SPAN_MAX_WORDS,
        "short_span_context_sentences": SHORT_SPAN_CONTEXT_SENTENCES,
        "title_case_max_words": TITLE_CASE_MAX_WORDS,
        "name_max_tokens": NAME_MAX_TOKENS,
        "citation_dominance_share": CITATION_DOMINANCE_SHARE,
        "pre_screen_by_source": PRE_SCREEN_BY_SOURCE,
        "domain_lexicon": list(DOMAIN_LEXICON),
        "dangling_start_words": list(DANGLING_START_WORDS),
        "dangling_end_words": list(DANGLING_END_WORDS),
        "heading_keywords": list(HEADING_KEYWORDS),
        "ollama_options": OLLAMA_OPTIONS,
        "ollama_think": OLLAMA_THINK,
    }
