# thesis_corpus

A three-stage pipeline (clean text → LLM-annotate + embed → reduce for
analysis), designed to run over more than one corpus. Every corpus's output
lives as a sibling under one root:

```
thesis/corpus/processed/
  literature/    <- the scholarly literature corpus (this README's default examples)
    documents/, corpus_manifest.csv, pending_annotations.jsonl,
    criterion_expressions.jsonl, annotated_documents.txt, logs/, ...
  miviludes/     <- same structure, once processed (see "Running on a different corpus")
  interviews/    <- same structure, once it has a Stage 1 (see caveat there)
  registry.csv   <- cross-corpus index, one row per document; see Stage 4
```

Each corpus subdirectory has the identical internal shape — Stage 1/2/3 just
get pointed at a different `--output-dir` (and, for Stage 1, a different
`--source-dir`). The literature corpus is simply the first one, not special;
its defaults are what every command below shows unless noted.

## Stage 1: PDF-to-clean-text pipeline (first version)

Finds PDFs recursively under `thesis/corpus/litterature with pdfs/Generated
Corpus/files/`, checks whether each can be opened, extracts text (native
first, falling back to OCR when the native pass looks implausible),
preserves page provenance, and logs what succeeded and failed. It never
modifies, moves, renames, or overwrites the source PDFs.

Deliberately out of scope at this stage: chapter selection, Zotero
integration, entity extraction, LLM processing, embeddings, semantic search,
reference removal, or any other text analysis.

### Setup

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

### Usage

```
cd thesis/corpus
python -m thesis_corpus.clean_text --limit 5   # test on the first 5 PDFs
python -m thesis_corpus.clean_text             # full batch
```

### OCR languages

Configured in one place: `OCR_LANGUAGES` in `extraction.py`, currently
`["eng", "fra", "deu"]`. Tesseract needs the matching `tessdata` language
files installed for these to actually work — check with `tesseract
--list-langs`.

### Outputs

```
thesis/corpus/processed/literature/
  documents/<document_id>/
    extracted.md      # Docling markdown export, page markers preserved
    extracted.txt      # plain text, whitespace collapsed, page markers kept
    pages.jsonl         # one record per PDF page
    metadata.json        # per-document processing outcome
  corpus_manifest.csv    # one row per discovered PDF
  logs/pipeline.log       # append-only log of the whole run
```

`document_id` is derived deterministically from the PDF's own filename
(normalized), disambiguated with the source Zotero-storage-key folder name
on collision (real filename collisions exist in this corpus).

### Status values

`processed`, `processed_with_ocr`, `unreadable` (couldn't even be opened —
corrupted or password-protected), `failed` (opened, but extraction raised).

## Stage 2: extraction + embedding (qwen3:4b + bge-m3 via Ollama)

Reads Stage 1's `processed/literature/documents/<document_id>/pages.jsonl`, chunks each
document into ~300-700 word / 2-5 paragraph pieces (page provenance kept as
`page_range`), sends every chunk to a locally-running Ollama for structured
annotation of cult/sect-criterion expressions, embeds every accepted item
and every unique entity anchor, and writes one consolidated
`criterion_expressions.jsonl`. One pipeline — no separate chapter-selection
or entity-extraction curation pass.

Internally it's split into two **checkpointed** phases so an embedding
failure never costs a re-run of the (slow, expensive) LLM annotation:

1. **annotate**: chunks + annotates with the LLM, writing each accepted item
   immediately to `processed/literature/pending_annotations.jsonl` — no
   embedding calls happen in this phase at all. A document is marked done in
   `processed/literature/annotated_documents.txt` once every one of its
   chunks has been annotated (even if it produced zero items).
2. **embed**: reads `pending_annotations.jsonl`, batch-embeds each
   not-yet-embedded document's items and entity anchors, and appends the
   final records to `criterion_expressions.jsonl`.

`--stage all` (the default) runs both, one full pass then the other. Since
annotation is durable on disk before any embedding call is attempted, a
failed or timed-out embed batch only costs re-running the embed phase for
that one document — the LLM is never called again for it.

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
python -m thesis_corpus.extract_and_embed --limit 5      # smoke test, first 5 not-yet-done documents (both stages)
python -m thesis_corpus.extract_and_embed                # full batch, both stages
python -m thesis_corpus.extract_and_embed --stage annotate   # annotation only (safe to leave running unattended)
python -m thesis_corpus.extract_and_embed --stage embed      # embedding only, from pending_annotations.jsonl
python -m thesis_corpus.extract_and_embed --document-id <id>   # one document, both stages -> separate file
python -m thesis_corpus.extract_and_embed --force         # reprocess everything, overwrite
```

Running `--stage annotate` and `--stage embed` as separate invocations (back
to back, or with the embed one re-run later) is equivalent to `--stage all`
but lets you, e.g., let the slow annotation phase run overnight unattended
and only deal with embedding afterwards, or re-run just the embed phase on
its own after a failure without touching the LLM again.

### Restart / `--document-id` semantics

- **Default (no flags)**: each phase appends to its own file
  (`pending_annotations.jsonl` for annotate, `criterion_expressions.jsonl`
  for embed), independently skipping any document already marked done for
  that phase — safe to re-run after a crash, an Ollama restart, or to extend
  the corpus later. A document that produced zero relevant chunks is still
  marked "annotated" (so it isn't retried forever) even though it never
  appears in `pending_annotations.jsonl`.
- **`--force`**: reprocesses everything for whichever phase(s) are running,
  overwriting that phase's checkpoint file(s) from scratch. `--force --stage
  embed` re-embeds every item in `pending_annotations.jsonl` without
  touching the LLM; `--force --stage all` redoes everything.
- **`--document-id <id>`**: processes only that one document, both phases,
  end to end, writing to a **separate** file, `criterion_expressions_<id>.jsonl`
  — it never touches the main `pending_annotations.jsonl` or
  `criterion_expressions.jsonl`. This is deliberate: merging a single
  document's corrected rows back into a large JSONL by rewriting lines is
  exactly the kind of operation that's easy to get subtly wrong, so it's
  left as a manual step. After inspecting `criterion_expressions_<id>.jsonl`,
  decide by hand whether to append it into the main file, use it to replace
  that document's existing rows there, or discard it.

### Outputs

```
thesis/corpus/processed/literature/
  pending_annotations.jsonl      # raw annotated items, no embeddings yet (annotate phase checkpoint)
  annotated_documents.txt        # one document_id per line, marks annotation done
  criterion_expressions.jsonl    # one JSON object per accepted item, with embeddings (main output)
  criterion_expressions_<id>.jsonl  # only from --document-id runs; see above
  extract_and_embed_summary.json  # documents/chunks/items/embeddings/errors counts
  logs/extract_and_embed.log      # every error and warning from this stage
```

If an embedding batch fails for a document, that document writes zero items
to `criterion_expressions.jsonl` this run and is retried on the next `embed`
run — but its annotation is already safe in `pending_annotations.jsonl`, so
retrying never re-calls the LLM. `pending_annotations.jsonl` and
`criterion_expressions*.jsonl` both contain full verbatim excerpts
(`context_window`/`source_quote`) from the copyrighted literature corpus, so
they're gitignored the same way `pages.jsonl`/`extracted.txt` are.

### Running this on Windows

Stage 2 was written and unit-tested (chunking, JSON/substring validation)
without Ollama available, since Ollama runs on your Windows GPU machine, not
here. To actually run it there:

- **Copy over**: the whole `thesis/corpus/thesis_corpus/` package, and
  `thesis/corpus/processed/literature/documents/` + `corpus_manifest.csv`
  (Stage 1's output — the only input Stage 2 reads).
- **Don't need**: `thesis/corpus/litterature with pdfs/` (source PDFs —
  Stage 2 never reads them), Docling, or PyMuPDF.
- **Watch out**: `processed/*/documents/**/{extracted.md,extracted.txt,pages.jsonl}`
  are gitignored, so a `git pull` on Windows will **not** bring them across —
  copy them by hand (external drive, sync tool, etc.) first.

## Stage 3: reduced/downsampled JSONL for analysis (`reduce_embeddings`)

`criterion_expressions.jsonl` (Stage 2's output) is a durable archive with
every field, including two large ones (`context_window`, the whole chunk
text; `entity_anchor_vectors`, one 1024-float vector per anchor) — useful
for provenance, too heavy for interactive analysis/visualization (a 7-
document test file is already 251MB; the full corpus is projected at
1.5-2GB). This stage produces a smaller sibling file with only the fields
needed for embedding-based analysis, optionally downsampled by clustering
so it stays a manageable size while preserving diversity rather than just
truncating. It needs no Ollama and no network access — only `numpy` and
`scikit-learn` — so it can run on any machine that has the archive file,
including this Mac.

```
pip install -r thesis_corpus/requirements-reduce_embeddings.txt
python -m thesis_corpus.reduce_embeddings                 # full corpus, downsampled
python -m thesis_corpus.reduce_embeddings --no-downsample  # field reduction only, every item kept
python -m thesis_corpus.reduce_embeddings --input <path> --output <path> --source-type miviludes
```

Downsampling: PCA to `--pca-dim` (default 100), `MiniBatchKMeans` into
`--n-clusters` (default 1000), then up to `--samples-per-cluster` (default
5) items randomly sampled per cluster — deterministic given `--seed`
(default 42). The archive's original full-precision `embedding_vector` is
always what's written out; a `float32` copy is used only internally for the
clustering math. Output rows are sorted back into original file order, not
grouped by cluster, so row positions stay traceable to the archive.

Never touches the source archive — always reads `--input`, writes
`--output`. `criterion_expressions_reduced.jsonl` is gitignored by the same
`criterion_expressions*.jsonl` pattern as the full archive (it still
carries `embedding_text`, effectively a paraphrase/quote of the source, and
full embedding vectors).

Reusable for other corpora via `--input`/`--output`/`--source-type` — no
code changes needed, just different paths and a different `--source-type`
value (see "Running on a different corpus" below).

## Running on a different corpus

Stages 1-2 accept a configurable root instead of the literature-corpus
defaults shown above. Both default to today's literature paths when the
flags are omitted, so nothing above changes unless you pass them:

```bash
# Stage 1: clean text for a non-default corpus
python -m thesis_corpus.clean_text \
  --source-dir "thesis/corpus/MIVILUDES/Latest Report/Latest Report" \
  --output-dir "thesis/corpus/processed/miviludes"

# Stage 2: extract + embed for that corpus
python -m thesis_corpus.extract_and_embed \
  --output-dir "thesis/corpus/processed/miviludes"

# Stage 3: reduce + downsample
python -m thesis_corpus.reduce_embeddings \
  --input thesis/corpus/processed/miviludes/criterion_expressions.jsonl \
  --output thesis/corpus/processed/miviludes/criterion_expressions_reduced.jsonl \
  --source-type miviludes
```

`--output-dir` on both Stage 1 and Stage 2 relocates that corpus's entire
output tree (`documents/`, checkpoint files, `logs/`) at once — every
sub-path is always derived from it the same way, so there's nothing else to
configure per-corpus.

**Interviews specifically**: unlike literature/MIVILUDES, interview
transcripts aren't PDFs (they live under `thesis/corpus/interviews/cleaned/
<id>/transcript.txt` already as plain text) — Stage 1 (built around Docling)
doesn't apply. `thesis_corpus.prepare_interviews` is the Stage 1 equivalent:
it treats each interview as a single page (no natural page structure like a
PDF has) and uses each interview's **original-language** transcript, not
`translation_en.txt` even where one exists — mixing original and translated
text would be an inconsistent basis for comparison, and `bge-m3`'s
multilingual embeddings (already proven on MIVILUDES' French) are exactly
why each interview can stay in its own language. There's no
`corpus_manifest.csv` for this one (no meaningful native/OCR/failed
distinction for already-transcribed text — `build_registry` reports
`stage1_status: n/a` for these documents instead):

```bash
python -m thesis_corpus.prepare_interviews \
  --output-dir "thesis/corpus/processed/interviews"

python -m thesis_corpus.extract_and_embed \
  --output-dir "thesis/corpus/processed/interviews"

python -m thesis_corpus.reduce_embeddings \
  --input thesis/corpus/processed/interviews/criterion_expressions.jsonl \
  --output thesis/corpus/processed/interviews/criterion_expressions_reduced.jsonl \
  --source-type interviews
```

## Stage 4: cross-corpus registry (`build_registry`)

With more than one corpus under `processed/`, `registry.csv` is a single
place to see every document's status across all of them — one row per
document: `corpus`, `document_id`, `stage1_status`, `stage2_status`,
`item_count`, `output_dir`. It's **generated**, not hand-maintained: it
scans each `processed/<corpus>/` folder's own existing manifests
(`corpus_manifest.csv` for Stage 1, `annotated_documents.txt` +
`criterion_expressions.jsonl` for Stage 2) and derives everything from
them, so it can never drift out of sync with the files it summarizes — just
re-run it after any pipeline run:

```
python -m thesis_corpus.build_registry
```

Writes `thesis/corpus/processed/registry.csv`. A corpus with no
`corpus_manifest.csv` (e.g. a future non-PDF corpus fed some other way)
simply shows `stage1_status: n/a` for its documents rather than erroring.
