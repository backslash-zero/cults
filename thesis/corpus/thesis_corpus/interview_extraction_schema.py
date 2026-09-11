"""Exhaustive interview extraction: prompt and structured-output schema.

Unlike extraction_v2_schema.py (shared with literature/MIVILUDES), this
schema asks the model to segment a transcript EXHAUSTIVELY -- every
expression from every speaker, filler included -- rather than selecting
only "the few... genuinely worth keeping" cult-relevant spans. Relevance
becomes a per-expression LABEL (`cult_relevant`), never a reason to omit
something. Speaker attribution is NOT asked of the model at all: for
interviews it is fully deterministic from the transcript's own turn
structure -- interview_chunking.py parses the `Interviewer:`/`Interviewee:`
labels, strips them (and the header/transcriber-notes text) before this
schema's prompt ever sees the transcript, and hands `screen_interviews_full.py`
a parallel `turn_roles` list it pairs back up with the label-free text
positionally (`turn_spans()`) -- so asking the model to guess attribution, or
even to see the labels at all, would only add error and output tokens for no
benefit.

`expression_kind`/`claim_mode`/`epistemic_status` are imported unchanged
from extraction_v2_schema so any future reuse of the geometric-analysis
toolkit's filters (which key off those exact enum values) keeps working on
this archive too.

This module is additive and versioned; extraction_v2_schema.py is not
modified, and this schema is not used for literature/MIVILUDES.
"""
from __future__ import annotations

import hashlib
import typing
from typing import Literal

from pydantic import BaseModel, ConfigDict, StrictBool, ValidationError

from thesis_corpus.extraction_v2_schema import CLAIM_MODES, EPISTEMIC_STATUSES, EXPRESSION_KINDS

EXTRACTION_VERSION = "interviews-full-1.0.0"

# Ollama request options. num_ctx is sized generously above the worst-case
# observed transcript (~1536 words / ~2,600 tokens) plus prompt/schema
# (~1,200-1,700 tokens) plus a large structured response spanning dozens of
# lean candidates -- if the longest transcript still truncates at this size,
# the mitigation is interview_chunking.py's turn-aligned split, not a larger
# context window here.
OLLAMA_OPTIONS_FULL = {"temperature": 0, "seed": 42, "num_ctx": 16384}
OLLAMA_THINK = False

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_FULL = """\
You are assisting a thesis that studies how "cults", "sects", sectarian drift and new
religious movements are described, defined, disputed, exemplified and framed in
interview text. The supplied text is one short interview transcript, already split into
short segments (roughly one sentence each), each separated by a blank line, in the
order they were said -- several segments in a row can belong to the same person's
reply. There is no speaker label, header, date, demographic information, or
transcriber commentary anywhere in the text below: only what was actually said.

Your task is EXHAUSTIVE segmentation, not selection. Read every segment -- both the
interviewer's and the interviewee's -- and, if it still contains more than one distinct
expression merged together, break it into its constituent expressions: every distinct
clause, short exchange or bare reply, including brief acknowledgements, interjections
and filler ("Okay.", "Um.", "Yeah.", "[laughs]."). Do not skip anything for being short,
off-topic, or unrelated to cults: the goal is complete coverage of everything said, not
curation. A segment that is already a single short reply ("Illuminati.", "Yes.")
becomes exactly one expression, unchanged. There is no limit on how many expressions
you return.

For each expression, report whether it is cult-relevant: materially describing,
defining, disputing, exemplifying, framing or associating cults, sects, sectarian
drift, new religious movements, group authority or control, manipulation, spiritual
abuse, a relevant named group or leader, or an everyday use of "cult". This is a LABEL,
not a filter -- expressions that are NOT cult-relevant must still be returned, with
cult_relevant set to false.

SEGMENTATION: a blank line always marks a segment boundary -- never merge text from two
different blank-line-separated segments into one expression, even if the topic
continues across them or they belong to the same reply. Within a segment, split at
clause boundaries wherever it still contains more than one distinct expression, claim,
or reply merged together. Never split a clause in the middle.

COPY RULE: verbatim_expression must be an exact, contiguous, character-for-character
substring of the supplied text, in the text's own language, taken from inside a single
segment only. Do not translate, paraphrase, normalise spelling, fix typos, add or
remove words, or merge separate spans.

For every expression also report: expression_kind; claim_mode; epistemic_status
(asserted = stated as true; qualified = stated with explicit limits or conditions;
contested = presented as disputed or debated; negated = denied or rejected;
speculative = offered as possibility or uncertainty); entity_anchors (named groups,
leaders, movements mentioned in the span, verbatim); cult_relevant; and a
relevance_note of at most 15 words on why it is, or is not, cult-relevant.

Separately, list domain_terms: cult-related concepts or named groups that appear
anywhere in the text (verbatim, as they appear), whether or not an expression around
them was already captured.

ORDER: list expressions in the order they occur in the transcript (turn order).

Return only a JSON object matching the schema. No commentary."""

PROMPT_SHA256 = hashlib.sha256(SYSTEM_PROMPT_FULL.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class CandidateFull(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    verbatim_expression: str
    expression_kind: Literal[EXPRESSION_KINDS]  # type: ignore[valid-type]
    claim_mode: Literal[CLAIM_MODES]  # type: ignore[valid-type]
    epistemic_status: Literal[EPISTEMIC_STATUSES]  # type: ignore[valid-type]
    entity_anchors: list[str] = []
    cult_relevant: StrictBool
    relevance_note: str = ""


class ChunkAnnotationFull(BaseModel):
    """Full strict shape -- used for the JSON schema sent to Ollama."""
    model_config = ConfigDict(extra="forbid")

    expressions: list[CandidateFull] = []
    domain_terms: list[str] = []


class ChunkEnvelopeFull(BaseModel):
    """Permissive top-level shape used when *reading* a response: candidates
    are validated one by one afterwards (see validate_candidate)."""
    model_config = ConfigDict(extra="ignore")

    expressions: list[dict] = []
    domain_terms: list = []


def response_json_schema() -> dict:
    return ChunkAnnotationFull.model_json_schema()


def validate_candidate(raw: object) -> tuple[CandidateFull | None, str | None, str]:
    """Returns (candidate, rejection_code, detail). Codes: I1
    missing_required_field, or invalid_candidate_shape for anything else
    Pydantic rejects (including a non-boolean cult_relevant)."""
    if not isinstance(raw, dict):
        return None, "invalid_candidate_shape", f"candidate is {type(raw).__name__}, not an object"
    try:
        return CandidateFull.model_validate(raw), None, ""
    except ValidationError as e:
        errors = e.errors()
        for err in errors:
            if err.get("type") == "missing":
                return None, "missing_required_field", ".".join(str(p) for p in err.get("loc", ()))
        first = errors[0]
        return None, "invalid_candidate_shape", f"{'.'.join(str(p) for p in first.get('loc', ()))}: {first.get('msg')}"


def _literal_values(annotation) -> tuple | None:
    if typing.get_origin(annotation) is Literal:
        return typing.get_args(annotation)
    return None


def schema_description() -> str:
    """Human-readable schema for the user message, generated from the models."""
    lines = ["Return a JSON object with this exact shape:", "{", '  "expressions": [', "    {"]
    for name, field in CandidateFull.model_fields.items():
        values = _literal_values(field.annotation)
        if values:
            rendered = " | ".join(f'"{v}"' for v in values)
        elif name == "verbatim_expression":
            rendered = '"Exact contiguous substring of the supplied text, from inside one segment."'
        elif name == "entity_anchors":
            rendered = '["named group or leader mentioned in the span", ...]'
        elif name == "relevance_note":
            rendered = '"At most 15 words on why this is, or is not, cult-relevant."'
        elif name == "cult_relevant":
            rendered = "true | false   (JSON boolean, not a string)"
        else:
            rendered = "..."
        lines.append(f'      "{name}": {rendered},')
    lines[-1] = lines[-1].rstrip(",")
    lines += ["    }", "  ],",
              '  "domain_terms": ["cult-related concept or named group appearing verbatim in the text", ...]',
              "}",
              "expressions may be empty only if the transcript's own text is empty. Segment exhaustively and "
              "list expressions in transcript order. Every enumerated field must use exactly one of the listed "
              "literal values."]
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
    """Everything a run's config.json records so a reader can reproduce the
    screen (see screen_interviews_full.py for the actual rule set)."""
    return {
        "extraction_version": EXTRACTION_VERSION,
        "prompt_sha256": PROMPT_SHA256,
        "ollama_options": OLLAMA_OPTIONS_FULL,
        "ollama_think": OLLAMA_THINK,
    }
