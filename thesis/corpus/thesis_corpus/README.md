# thesis_corpus — PDF-to-clean-text pipeline (first version)

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
pip install -r thesis_corpus/requirements.txt
```

Docling pulls in a moderately large model/dependency footprint on first run
(CPU-only layout-model download — not a tiny install, but not GPU-only
either).

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
