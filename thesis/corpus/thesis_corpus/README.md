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
Corpus/files/` (the misspelling is the actual folder name on disk and
`clean_text.py`'s default `--source-dir` — a known, long-standing naming
quirk, left as-is since renaming it now touches a working default path for
zero functional benefit), checks whether each can be opened, extracts text
(native first, falling back to OCR when the native pass looks implausible),
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

## Concept backbone (`build_concept_backbone_en`)

A generic, low-bias list of abstract English concepts, meant to serve as
neutral anchor points that structure the shared embedding space across
every corpus above -- deliberately **not** derived from any cult/sect-
related seed list.

**Resource**: [Open English WordNet](https://en-word.net/) `oewn:2024`, via
the [`wn`](https://pypi.org/project/wn/) library. ConceptNet was the
originally planned resource, but its live API is currently down (persistent
502) and its bulk dump does not give clean cross-lingual/abstractness
signal for this without heuristics; OMW's WordNet gives both directly.

**Filtering/ranking** (fully structural -- no topical seeds anywhere):

1. Candidate pool: every noun synset in `oewn:2024` (84,956 total).
2. Abstractness: kept only if its hypernym chain leads to the WordNet root
   synset `abstraction.n.06` ("abstraction, abstract entity"), as opposed
   to `physical_entity.n.01` or other roots -- WordNet's own top-level
   entity/abstraction split.
3. Named-entity exclusion: dropped if the synset is an `instance_hyponym`
   of something (WordNet's mechanism for individuals, e.g. "Paris"), or its
   primary lemma is capitalized.
4. Taxonomy exclusion: dropped if the synset's lexicographer file is
   `noun.animal` or `noun.plant` -- biological classification terms (e.g.
   "bird genus", "asterid dicot genus") pass the abstraction test
   structurally (a genus is a hyponym of `group`) but have very high
   hyponym counts purely from enumerating species, which dominated
   degree-based ranking before this exclusion was added.
5. Centrality: node degree -- the total count of all WordNet relations
   (hypernym, hyponym, meronym, holonym, attribute, similar_to, ...)
   touching the synset.
6. Top `--target-size` synsets by degree, descending.

Of 84,956 noun synsets: 41,351 passed the abstractness filter, 9,151 of
those were excluded as named entities, 875 more as biological taxonomy,
leaving 31,325 candidates. The default run keeps the top 3,000 by degree --
now general concepts like "law", "chemistry", "biology", "mathematics",
"quality", "time period".

### Setup

```
cd thesis/corpus
pip install -r thesis_corpus/requirements-build_concept_backbone_en.txt
python -c "import wn; wn.download('oewn:2024')"   # one-time, ~13MB
```

### Usage

```
python -m thesis_corpus.build_concept_backbone_en                    # default target size 3000
python -m thesis_corpus.build_concept_backbone_en --target-size 5000
```

### Output

`thesis/corpus/dictionaries/concept_backbone_omw_en.csv`, columns:
`concept_id` (WordNet interlingual index ID), `concept_en` (primary
lemma), `gloss_en` (WordNet definition -- included because bare polysemous
words like "state" or "matter" are poor embedding anchors without sense
disambiguation), `source`, `centrality_score` (node degree).

### Embedding the backbone

`thesis_corpus.embed_concept_backbone` embeds each concept as
`"<concept_en>: <gloss_en>"` (not the bare word alone, for the same
sense-disambiguation reason `gloss_en` is kept in the CSV) with `bge-m3`,
same `ollama_client.embed_texts` path and Ollama prerequisite as
[`embed_miviludes_criteria.py`](thesis_corpus/embed_miviludes_criteria.py):

```
python -m thesis_corpus.embed_concept_backbone
```

Writes `thesis/corpus/dictionaries/concept_backbone_embedded.jsonl` (one
record per concept: the original CSV fields plus `embedding_text`,
`embedding_model`, `embedding_vector`). Gitignored, same reasoning as every
other large embedding output in this repo.

**This list's role is now operationalized, not just a static reference
sitting next to the corpora**: `thesis_corpus.build_shared_space` (below)
uses these 3,000 vectors as active participants in fitting the one shared
space every corpus and the MIVILUDES criteria get projected into — the
concept backbone isn't compared *against* that space after the fact, it
helps *define* it.

## Shared cross-corpus space (`build_shared_space`)

Every embedding step above (the three `extract_and_embed` corpora,
`embed_miviludes_criteria`, `embed_concept_backbone`) produces vectors in
the same raw 1024-d `bge-m3` space, but `reduce_embeddings.py`'s
downsampling (Stage 3) fits an **independent** PCA per corpus — literature
gets its own 100-d space, MIVILUDES its own, interviews its own. Those are
three unrelated coordinate systems: a point's position in one says nothing
about a point's position in another. **This is the space any cross-corpus
geometric comparison or visualization should actually be built on instead**
— not any single corpus's `reduce_embeddings.py` output.

`build_shared_space.py` instead pools every vector from every dataset,
standardizes, and fits **one** PCA on the pooled matrix, so every item from
every dataset ends up in the same shared coordinate system:

- Each corpus item (literature/miviludes/interviews) contributes **one**
  point (`embedding_vector`).
- Each MIVILUDES criterion contributes **two** points, not one — its
  `embedding_vector_fr` and `embedding_vector_en` separately (both are
  legitimate embedded representations of the same content; their distance
  from each other in the shared space is itself a sanity check on the
  multilingual embedding).
- Each concept-backbone entry contributes **one** point (`embedding_vector`).
- Expected total, asserted at runtime: 39,236 + 914 + 231 + 34 + 3,000 =
  **43,415** points.

**Standardization**: `StandardScaler` (zero mean, unit variance per
dimension) runs before PCA. Every vector already comes from the same
embedding model, but the five datasets differ a lot in register (academic
prose, government French, casual interview speech, bare word+gloss
dictionary entries) and could plausibly carry different per-dimension
distributions — standardizing is a defensive measure against any one
dataset dominating the fit purely due to scale, not a claim that such an
imbalance is known to exist.

**Dimensionality is not a fixed constant**: PCA is first fit at full rank
to get the complete explained-variance curve — written to
`processed/shared_space_variance_curve.{csv,json,png}` so the choice is
inspectable rather than asserted — and the smallest `k` reaching 95%
cumulative variance is picked from that curve. On the actual pooled data:
`k=390` for 95.0% (curve is fairly gradual, not a sharp knee — e.g. 61% at
k=100, 90% at k=300 — so this is a real but not dramatic compression;
worth knowing when interpreting distances in the shared space).

```
python -m thesis_corpus.build_shared_space
```

No Ollama needed — pure `numpy`/`scikit-learn`/`matplotlib` on data that's
already local (same `requirements-reduce_embeddings.txt`, now with
`matplotlib` added). Never modifies any of the five source files, only
writes new ones under `processed/`.

### Output

```
thesis/corpus/processed/
  shared_embedding_space.jsonl        # one row per pooled point: source_dataset, key, label, shared_space_vector
  shared_space_variance_curve.csv     # n_components, cumulative_variance -- the full curve
  shared_space_variance_curve.json    # {curve, chosen_k, variance_at_k, threshold}
  shared_space_variance_curve.png     # plot of the above
```

`shared_embedding_space.jsonl` is gitignored (large — 358MB at 43,415
points × 390 dims — and, like every other embedding output, derived from
copyrighted/participant text). The variance-curve files are small and
tracked.

After writing the output, the script prints two diagnostic (not pass/fail)
sanity checks: mean vector norm by `source_dataset` (flags any one dataset
being pushed to the periphery or center relative to the others), and each
MIVILUDES criterion's FR vs EN point norms side by side (expected to be
close, confirming the multilingual embedding + shared PCA are behaving
sensibly).
