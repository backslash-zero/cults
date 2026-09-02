"""Splits a document's pages into coherent, page-tracked chunks for the LLM
annotation stage.

Chunks are built greedily from paragraph boundaries within a single document
(never mixing content across documents), aiming for 300-700 words or 2-5
paragraphs per chunk, whichever threshold is reached first. Page provenance
is preserved as a [min, max] page_range per chunk.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from thesis_corpus.writer import collapse_whitespace

MIN_WORDS = 300
MAX_WORDS = 700
MAX_PARAGRAPHS = 5

_BLANK_LINE_RE = re.compile(r"\n\s*\n")


@dataclass
class Chunk:
    document_id: str
    chunk_index: int
    page_range: list[int]
    text: str


def _paragraphs_by_page(pages: list[dict]) -> list[tuple[int, str]]:
    """Flatten a document's pages into an ordered (page_number, paragraph) sequence."""
    result = []
    for page in sorted(pages, key=lambda p: p["page_number"]):
        normalized = collapse_whitespace(page["text"]) if page["text"] else ""
        for paragraph in _BLANK_LINE_RE.split(normalized):
            paragraph = paragraph.strip()
            if paragraph:
                result.append((page["page_number"], paragraph))
    return result


def build_chunks(document_id: str, pages: list[dict]) -> list[Chunk]:
    paragraphs = _paragraphs_by_page(pages)

    chunks: list[Chunk] = []
    buf_paragraphs: list[str] = []
    buf_pages: list[int] = []
    buf_words = 0

    def flush() -> None:
        nonlocal buf_paragraphs, buf_pages, buf_words
        if not buf_paragraphs:
            return
        chunks.append(Chunk(
            document_id=document_id,
            chunk_index=len(chunks),
            page_range=[min(buf_pages), max(buf_pages)],
            text="\n\n".join(buf_paragraphs),
        ))
        buf_paragraphs, buf_pages, buf_words = [], [], 0

    for page_no, paragraph in paragraphs:
        word_count = len(paragraph.split())

        # Closing early rather than crossing MAX_WORDS, unless the buffer is
        # still empty (a single paragraph longer than MAX_WORDS stays whole).
        if buf_paragraphs and buf_words + word_count > MAX_WORDS:
            flush()

        buf_paragraphs.append(paragraph)
        buf_pages.append(page_no)
        buf_words += word_count

        if len(buf_paragraphs) >= 2 and (buf_words >= MIN_WORDS or len(buf_paragraphs) >= MAX_PARAGRAPHS):
            flush()

    flush()  # final, possibly-short chunk
    return chunks
