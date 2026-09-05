"""Thin client for a locally-running Ollama instance.

Two calls are used: chat-based structured annotation (qwen3:4b by default)
and batched embedding (bge-m3 by default). Everything here assumes Ollama is
reachable at OLLAMA_HOST -- no remote host is ever contacted.
"""
from __future__ import annotations

import json
import re
from typing import Literal

import httpx
from pydantic import BaseModel, ValidationError

SYSTEM_PROMPT = """\
You are assisting a thesis investigating how criteria for "cults" and \
sectarian formations are articulated in scholarly and journalistic writing.

Your task is NOT to decide whether anything described in the text is \
objectively a cult. Instead, identify expressions through which the supplied \
text describes, attributes, disputes, rejects, defines, or problematises \
criteria relevant to cult-like or sectarian formations.

Relevant material may be implicit: it does not need to contain the words \
"cult", "sect", "NRM", or "sectarian drift" to be relevant.

Do not invent content that is not in the supplied text. Every source_quote \
must be an exact, verbatim substring of the supplied text.

Return only valid JSON conforming to the schema you have been given. No \
commentary, no markdown fences, no text before or after the JSON object."""

_JSON_REMINDER = "\n\nReturn only valid JSON, no commentary, no markdown fences."

_SCHEMA_DESCRIPTION = """\
Return a JSON object with this exact shape:
{
  "chunk_relevance": "relevant" | "not_relevant" | "uncertain",
  "items": [
    {
      "source_quote": "Exact substring from the supplied text.",
      "embedding_text": "Short self-contained expression derived from the source_quote (often identical).",
      "entity_anchors": ["Scientology", "charismatic leader"],
      "claim_mode": "direct_statement" | "attributed_statement" | "quotation" | "definition" | "question_or_reflection" | "other",
      "epistemic_status": "asserted" | "qualified" | "contested" | "negated" | "speculative",
      "attribution": "author" | "cited_author" | "participant" | "institution" | "journalist" | "unspecified"
    }
  ]
}
attribution must be exactly one of those six literal values -- never any other
word. In an interview transcript, the person being interviewed is "participant"
(not "interviewee"); the person asking questions is "unspecified" unless they
are themselves citing someone else (not "interviewer").
If nothing is relevant, return {"chunk_relevance": "not_relevant", "items": []}."""

# The model (especially a small local one) sometimes echoes a transcript's own
# literal speaker labels here instead of following the enum -- normalize known
# synonyms before validation rather than relying on the prompt alone.
_ATTRIBUTION_SYNONYMS = {
    "interviewee": "participant",
    "respondent": "participant",
    "subject": "participant",
    "interviewer": "unspecified",
    "researcher": "unspecified",
}


def _normalize_attribution(parsed: dict) -> dict:
    for item in parsed.get("items", []):
        if not isinstance(item, dict):
            continue
        value = item.get("attribution")
        if isinstance(value, str):
            item["attribution"] = _ATTRIBUTION_SYNONYMS.get(value.strip().lower(), value)
    return parsed


class AnnotationItem(BaseModel):
    source_quote: str
    embedding_text: str
    entity_anchors: list[str] = []
    claim_mode: Literal[
        "direct_statement", "attributed_statement", "quotation",
        "definition", "question_or_reflection", "other",
    ]
    epistemic_status: Literal["asserted", "qualified", "contested", "negated", "speculative"]
    attribution: Literal["author", "cited_author", "participant", "institution", "journalist", "unspecified"]


class ChunkAnnotation(BaseModel):
    chunk_relevance: Literal["relevant", "not_relevant", "uncertain"]
    items: list[AnnotationItem] = []


class OllamaUnavailableError(Exception):
    pass


class AnnotationError(Exception):
    pass


class EmbeddingError(Exception):
    pass


def check_available(host: str, timeout: float = 5.0) -> None:
    try:
        resp = httpx.get(f"{host}/api/tags", timeout=timeout)
        resp.raise_for_status()
    except Exception as e:
        raise OllamaUnavailableError(
            f"Could not reach Ollama at {host} ({e}). "
            "Make sure `ollama serve` is running on this machine."
        ) from e


def _strip_code_fence(text: str) -> str:
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text.strip(), re.DOTALL)
    return match.group(1) if match else text


def _user_message(document_id: str, page_range: list[int], chunk_index: int, text: str) -> str:
    return (
        f"document_id: {document_id}\n"
        f"page_range: {page_range}\n"
        f"chunk_index: {chunk_index}\n\n"
        f"{_SCHEMA_DESCRIPTION}\n\n"
        f"Text:\n{text}"
    )


def _call_chat(host: str, model: str, user_content: str, timeout: float, think: bool) -> str:
    resp = httpx.post(
        f"{host}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "format": "json",
            "stream": False,
            "think": think,
            "options": {"temperature": 0, "num_ctx": 4096},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def annotate_chunk(
    host: str,
    model: str,
    document_id: str,
    page_range: list[int],
    chunk_index: int,
    text: str,
    timeout: float = 120.0,
    think: bool = False,
) -> ChunkAnnotation:
    user_content = _user_message(document_id, page_range, chunk_index, text)

    last_error: Exception | None = None
    for attempt_content in (user_content, user_content + _JSON_REMINDER):
        try:
            raw = _call_chat(host, model, attempt_content, timeout, think)
            parsed = _normalize_attribution(json.loads(_strip_code_fence(raw)))
            return ChunkAnnotation.model_validate(parsed)
        except httpx.HTTPError as e:
            raise AnnotationError(f"Ollama chat request failed: {e}") from e
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = e
            continue

    raise AnnotationError(f"Model did not return valid/schema-conforming JSON: {last_error}")


class TranslationError(Exception):
    pass


def translate_text(
    host: str,
    model: str,
    text: str,
    target_language: str = "English",
    source_language: str | None = None,
    timeout: float = 60.0,
    retries: int = 2,
) -> str:
    """Bare chat-completion translation -- no JSON schema, no Pydantic
    validation, unlike annotate_chunk (which is tied to the annotation
    schema above). Used by translate_miviludes_expressions.py to translate
    short, already-extracted expressions.

    qwen3's "thinking" mode can leak an extended reasoning trace into the
    response content even with "think": False in the request (observed in
    practice: every one of a 914-item run came back as a multi-paragraph
    trace ending in a literal "</think>" marker followed by the real
    answer, rather than a bare translation) -- unlike annotate_chunk, which
    is shielded from this by "format": "json" constraining the output
    shape. Defensively strip anything up to and including the last
    "</think>" if present, so a bad chat-template default can't silently
    embed a reasoning trace instead of a translation."""
    system_prompt = (
        f"You are a translator. Translate the user's text into {target_language}. "
        "Return only the translation itself -- no commentary, no quotation "
        "marks, no explanation."
    )
    if source_language:
        system_prompt += f" The source text is in {source_language}."

    last_error: Exception | None = None
    for _attempt in range(retries + 1):
        try:
            resp = httpx.post(
                f"{host}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text},
                    ],
                    "stream": False,
                    "think": False,
                    "options": {"temperature": 0},
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            translated = resp.json()["message"]["content"].strip()
            if "</think>" in translated:
                translated = translated.rsplit("</think>", 1)[-1].strip()
            if not translated:
                raise TranslationError("Model returned an empty translation")
            return translated
        except httpx.HTTPError as e:
            last_error = TranslationError(f"Ollama chat request failed: {e}")
        except (KeyError, TypeError) as e:
            last_error = TranslationError(f"Unexpected chat response shape: {e}")
        except TranslationError as e:
            last_error = e

    raise last_error


EMBED_BATCH_SIZE = 64


def _embed_batch(host: str, model: str, texts: list[str], timeout: float | None, retries: int) -> list[list[float]]:
    """Embeds a single batch (already <= EMBED_BATCH_SIZE) in one request."""
    if timeout is None:
        timeout = max(180.0, 2.0 * len(texts))

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = httpx.post(
                f"{host}/api/embed",
                json={"model": model, "input": texts},
                timeout=timeout,
            )
            if resp.status_code >= 400:
                raise EmbeddingError(
                    f"Ollama embed request failed: HTTP {resp.status_code}: {resp.text[:500]}"
                )
            embeddings = resp.json()["embeddings"]
            if len(embeddings) != len(texts):
                raise EmbeddingError(f"Expected {len(texts)} embeddings, got {len(embeddings)}")
            return embeddings
        except httpx.HTTPError as e:
            last_error = EmbeddingError(f"Ollama embed request failed: {e}")
        except (KeyError, TypeError) as e:
            last_error = EmbeddingError(f"Unexpected embed response shape: {e}")
        except EmbeddingError as e:
            last_error = e

    raise last_error


def embed_texts(
    host: str, model: str, texts: list[str], timeout: float | None = None, retries: int = 2,
    batch_size: int = EMBED_BATCH_SIZE,
) -> list[list[float]]:
    """Embeds texts in fixed-size sub-batches rather than one request per
    call site, however many texts that site has. A single document's worth
    of accepted items can run into the thousands (a 3,602-item document was
    observed to fail its embed call outright when sent as one request,
    despite a generous timeout -- consistent with Ollama rejecting or
    choking on an oversized single request, not merely being slow). Batching
    keeps every request a fixed, known-good size and limits the blast radius
    of a failure to one batch's timeout/retries rather than the whole
    document's.
    """
    if not texts:
        return []
    if len(texts) <= batch_size:
        return _embed_batch(host, model, texts, timeout, retries)

    results: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        results.extend(_embed_batch(host, model, texts[i:i + batch_size], timeout, retries))
    return results
