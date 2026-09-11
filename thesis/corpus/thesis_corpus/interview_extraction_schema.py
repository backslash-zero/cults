"""Exhaustive interview LABELING: prompt and structured-output schema.

v1 of this module (interviews-full-1.0.0) asked the model to find and copy
expressions out of a transcript, the same shape as extraction_v2_schema.py
-- then screen_interviews_full.py verified each returned span was an exact
verbatim substring. That's a real coverage gap for a use case that needs
EVERY piece of the transcript embedded (e.g. displaying interview
embeddings synced to subtitles): the model could simply fail to emit a
candidate for some sentence, and nothing downstream can fix a candidate
that was never produced. A verbatim-matching mismatch (`not_verbatim`) is
also actively harmful for that use case specifically -- embedding
paraphrased text under a subtitle that shows the real transcript breaks the
correspondence the whole thing is for.

v2 of this module (interviews-full-2.0.0) inverts the design: chunking
(interview_chunking.py) already deterministically produces the exact
segments that should end up embedded -- there is nothing left for the model
to "find." So the model's only job now is to LABEL each already-known
segment (cult_relevant, entity_anchors, expression_kind, claim_mode,
epistemic_status), by echoing back the segment_index it's labeling. No
`verbatim_expression` field exists in this schema at all: there is no text
for the model to copy, and therefore no verbatim-matching machinery, no
possibility of a `not_verbatim`/`span_cuts_word` rejection, and no
"the model just didn't emit anything for this bit of text" gap -- coverage
is guaranteed by construction in interview_segment_labels.py, which builds
one record per segment regardless of whether a valid label for it exists.

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

from pydantic import BaseModel, ConfigDict, StrictBool, StrictInt, ValidationError

from thesis_corpus.extraction_v2_schema import CLAIM_MODES, EPISTEMIC_STATUSES, EXPRESSION_KINDS

EXTRACTION_VERSION = "interviews-full-2.0.0"

OLLAMA_OPTIONS_FULL = {"temperature": 0, "seed": 42, "num_ctx": 16384}
OLLAMA_THINK = False

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_FULL = """\
You are assisting a thesis that studies how "cults", "sects", sectarian drift and new
religious movements are described, defined, disputed, exemplified and framed in
interview text. You are given a numbered list of short segments from one interview
transcript (already split by speaker turn and by sentence, in the order they were
said). There is no speaker label, header, date, demographic information, or
transcriber commentary anywhere: only what was actually said, one segment per number.

Your task is to LABEL, not to select or copy. Every single segment listed -- both the
interviewer's and the interviewee's, including brief acknowledgements, interjections
and filler ("Okay.", "Um.", "Yeah.", "[laughs].") -- must get exactly one label entry.
Do not skip any segment, do not merge two segments into one entry, and do not split one
segment into more than one entry: the segments are already the exact unit to label, one
label per segment_index, matching the numbers shown.

For each segment report: segment_index (the number shown next to it, unchanged);
expression_kind; claim_mode; epistemic_status (asserted = stated as true; qualified =
stated with explicit limits or conditions; contested = presented as disputed or
debated; negated = denied or rejected; speculative = offered as possibility or
uncertainty); entity_anchors (named groups, leaders, movements mentioned in the
segment, verbatim); cult_relevant: whether the segment materially describes, defines,
disputes, exemplifies, frames or associates cults, sects, sectarian drift, new
religious movements, group authority or control, manipulation, spiritual abuse, a
relevant named group or leader, or an everyday use of "cult". This is a LABEL, not a
filter -- every segment gets one, including cult_relevant=false ones; a
relevance_note of at most 15 words on why it is, or is not, cult-relevant.

Separately, list domain_terms: cult-related concepts or named groups that appear
anywhere in the transcript (verbatim, as they appear), whether or not any individual
segment was tagged cult_relevant.

Return only a JSON object matching the schema. No commentary."""

PROMPT_SHA256 = hashlib.sha256(SYSTEM_PROMPT_FULL.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class SegmentLabel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    segment_index: StrictInt
    expression_kind: Literal[EXPRESSION_KINDS]  # type: ignore[valid-type]
    claim_mode: Literal[CLAIM_MODES]  # type: ignore[valid-type]
    epistemic_status: Literal[EPISTEMIC_STATUSES]  # type: ignore[valid-type]
    entity_anchors: list[str] = []
    cult_relevant: StrictBool
    relevance_note: str = ""


class TranscriptLabelsFull(BaseModel):
    """Full strict shape -- used for the JSON schema sent to Ollama."""
    model_config = ConfigDict(extra="forbid")

    segment_labels: list[SegmentLabel] = []
    domain_terms: list[str] = []


class TranscriptLabelsEnvelope(BaseModel):
    """Permissive top-level shape used when *reading* a response: each
    segment_labels entry is validated one by one afterwards (see
    validate_label)."""
    model_config = ConfigDict(extra="ignore")

    segment_labels: list[dict] = []
    domain_terms: list = []


def response_json_schema() -> dict:
    return TranscriptLabelsFull.model_json_schema()


def validate_label(raw: object) -> tuple[SegmentLabel | None, str | None, str]:
    """Returns (label, rejection_code, detail). Codes: L1
    missing_required_field, or invalid_label_shape for anything else
    Pydantic rejects (including a non-integer segment_index or non-boolean
    cult_relevant). A label failing validation does not remove its segment
    from the archive -- interview_segment_labels.py still embeds that
    segment, just without real label fields (label_status="invalid")."""
    if not isinstance(raw, dict):
        return None, "invalid_label_shape", f"label is {type(raw).__name__}, not an object"
    try:
        return SegmentLabel.model_validate(raw), None, ""
    except ValidationError as e:
        errors = e.errors()
        for err in errors:
            if err.get("type") == "missing":
                return None, "missing_required_field", ".".join(str(p) for p in err.get("loc", ()))
        first = errors[0]
        return None, "invalid_label_shape", f"{'.'.join(str(p) for p in first.get('loc', ()))}: {first.get('msg')}"


def _literal_values(annotation) -> tuple | None:
    if typing.get_origin(annotation) is Literal:
        return typing.get_args(annotation)
    return None


def schema_description() -> str:
    """Human-readable schema for the user message, generated from the models."""
    lines = ["Return a JSON object with this exact shape:", "{", '  "segment_labels": [', "    {"]
    for name, field in SegmentLabel.model_fields.items():
        values = _literal_values(field.annotation)
        if values:
            rendered = " | ".join(f'"{v}"' for v in values)
        elif name == "segment_index":
            rendered = "0   (the number shown next to the segment, an integer)"
        elif name == "entity_anchors":
            rendered = '["named group or leader mentioned in the segment", ...]'
        elif name == "relevance_note":
            rendered = '"At most 15 words on why this is, or is not, cult-relevant."'
        elif name == "cult_relevant":
            rendered = "true | false   (JSON boolean, not a string)"
        else:
            rendered = "..."
        lines.append(f'      "{name}": {rendered},')
    lines[-1] = lines[-1].rstrip(",")
    lines += ["    }", "  ],",
              '  "domain_terms": ["cult-related concept or named group appearing verbatim in the transcript", ...]',
              "}",
              "segment_labels must have exactly one entry per numbered segment below, in any order, each with "
              "the matching segment_index. Every enumerated field must use exactly one of the listed literal "
              "values."]
    return "\n".join(lines)


def user_message(document_id: str, page_range: list[int], chunk_index: int, segments: list[str]) -> str:
    numbered = "\n".join(f"[{i}] {segment}" for i, segment in enumerate(segments))
    return (
        f"document_id: {document_id}\n"
        f"page_range: {page_range}\n"
        f"chunk_index: {chunk_index}\n\n"
        f"{schema_description()}\n\n"
        f"Segments ({len(segments)} total, 0-indexed):\n{numbered}"
    )


def screening_constants() -> dict:
    """Everything a run's config.json records so a reader can reproduce the
    run (see interview_segment_labels.py for how labels attach to the
    guaranteed per-segment records)."""
    return {
        "extraction_version": EXTRACTION_VERSION,
        "prompt_sha256": PROMPT_SHA256,
        "ollama_options": OLLAMA_OPTIONS_FULL,
        "ollama_think": OLLAMA_THINK,
    }
