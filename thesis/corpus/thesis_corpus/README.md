# thesis_corpus

## Stage 1: PDF-to-clean-text pipeline (first version)

Finds PDFs recursively under `thesis/corpus/litterature with pdfs/Generated
Corpus/files/`, checks whether each can be opened, extracts text (native
first, falling back to OCR when the native pass looks implausible),
preserves page provenance, and logs what succeeded and failed. It never
modifies, moves, renames, or overwrites the source PDFs.

Deliberately out of scope at this stage: chapter selection, Zotero
integration, entity extraction, LLM processing, embeddings, semantic search,
reference removal, or any other text analysis.

## Setup

```
brew install tesseract-lang   # adds fra/deu language data (eng already ships
                               # with the base `tesseract` formula)
cd thesis/corpus
pip install -r thesis_corpus/requirements-clean_text.txt
```

Docling pulls in a moderately large model/dependency footprint on first run
(CPU-only layout-model download — not a tiny install, but not GPU-only
either). This stage's dependencies (`requirements-clean_text.txt`) are kept
separate from Stage 2's (`requirements-extract_and_embed.txt`) — the two
stages typically run on different machines and neither needs the other's
dependencies.

## Usage

```
cd thesis/corpus
python -m thesis_corpus.clean_text --limit 5   # test on the first 5 PDFs
python -m thesis_corpus.clean_text             # full batch
```

## OCR languages

Configured in one place: `OCR_LANGUAGES` in `extraction.py`, currently
`["eng", "fra", "deu"]`. Tesseract needs the matching `tessdata` language
files installed for these to actually work — check with `tesseract
--list-langs`.

## Outputs

```
thesis/corpus/processed/
  documents/<document_id>/
    extracted.md      # Docling markdown export, page markers preserved
    extracted.txt      # plain text, whitespace collapsed, page markers kept
    pages.jsonl         # one record per PDF page
    metadata.json        # per-document processing outcome
  corpus_manifest.csv    # one row per discovered PDF
  pipeline.log            # append-only log of the whole run
```

`document_id` is derived deterministically from the PDF's own filename
(normalized), disambiguated with the source Zotero-storage-key folder name
on collision (real filename collisions exist in this corpus).

## Status values

`processed`, `processed_with_ocr`, `unreadable` (couldn't even be opened —
corrupted or password-protected), `failed` (opened, but extraction raised).

## Stage 2: extraction + embedding (qwen3:4b + bge-m3 via Ollama)

Reads Stage 1's `processed/documents/<document_id>/pages.jsonl`, chunks each
document into ~300-700 word / 2-5 paragraph pieces (page provenance kept as
`page_range`), sends every chunk to a locally-running Ollama for structured
annotation of cult/sect-criterion expressions, embeds every accepted item
and every unique entity anchor, and writes one consolidated
`criterion_expressions.jsonl`. Single pass — no separate chapter-selection,
entity-extraction, or embedding pass.

This stage does **not** touch the source PDFs, and does not need Docling or
PyMuPDF — only Stage 1's already-extracted `pages.jsonl` files.

### Prerequisite: Ollama, running locally

This must run on the same machine as Ollama (default
`http://127.0.0.1:11434`, override with `--ollama-host`) — no remote host is
ever contacted.

```
ollama pull qwen3:4b
ollama pull bge-m3
ollama serve            # if not already running
pip install -r thesis_corpus/requirements-extract_and_embed.txt
```

Only `httpx`, `tqdm`, and `pydantic` — this stage does not need Docling or
PyMuPDF (Stage 1's dependencies, in `requirements-clean_text.txt`), so don't
install that file here; it pulls in a much larger dependency tree
(transformers/torch and friends) for nothing.

### Usage

```
cd thesis/corpus
python -m thesis_corpus.extract_and_embed --limit 5      # smoke test, first 5 not-yet-done documents
python -m thesis_corpus.extract_and_embed                # full batch
python -m thesis_corpus.extract_and_embed --document-id <id>   # reprocess one document
python -m thesis_corpus.extract_and_embed --force         # reprocess everything, overwrite
```

### Restart / `--document-id` semantics

- **Default (no flags)**: appends to `criterion_expressions.jsonl`, skipping
  any `document_id` already present in it — safe to re-run after a crash or
  to extend the corpus later.
- **`--force`**: reprocesses every document and overwrites
  `criterion_expressions.jsonl` from scratch.
- **`--document-id <id>`**: processes only that one document and writes to a
  **separate** file, `criterion_expressions_<id>.jsonl` — it never modifies
  the main file. This is deliberate: merging a single document's corrected
  rows back into a large JSONL by rewriting lines is exactly the kind of
  operation that's easy to get subtly wrong, so it's left as a manual step.
  After inspecting `criterion_expressions_<id>.jsonl`, decide by hand whether
  to append it into the main file, use it to replace that document's
  existing rows there, or discard it.

### Outputs

```
thesis/corpus/processed/
  criterion_expressions.jsonl   # one JSON object per accepted item (main output)
  criterion_expressions_<id>.jsonl  # only from --document-id runs; see above
  extraction_summary.json        # documents/chunks/items/embeddings/errors counts
  logs/extraction.log             # every error and warning from this stage
```

If an embedding call fails for a document, that document writes zero items
for the run (its annotation results are discarded rather than stored without
vectors) and is retried in full on the next default run, since it's absent
from `criterion_expressions.jsonl`.

### Running this on Windows

Stage 2 was written and unit-tested (chunking, JSON/substring validation)
without Ollama available, since Ollama runs on your Windows GPU machine, not
here. To actually run it there:

- **Copy over**: the whole `thesis/corpus/thesis_corpus/` package, and
  `thesis/corpus/processed/documents/` + `corpus_manifest.csv` (Stage 1's
  output — the only input Stage 2 reads).
- **Don't need**: `thesis/corpus/litterature with pdfs/` (source PDFs —
  Stage 2 never reads them), Docling, or PyMuPDF.
- **Watch out**: `processed/documents/**/{extracted.md,extracted.txt,pages.jsonl}`
  are gitignored, so a `git pull` on Windows will **not** bring them across —
  copy them by hand (external drive, sync tool, etc.) first.
