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

## Structural concepts, a second reference subset (`extract_structural_concepts`)

The WordNet concept backbone above is deliberately **topic-neutral** — its
whole value as a yardstick depends on not being derived from the corpora it
interprets. That neutrality has a cost: measured in the shared space, the
concept backbone's centroid sits notably farther from the corpus-expression
centroid than emergent entities do (16.7 vs. 12.5 shared-space units, against
a ~33-unit expression-to-expression baseline) — a real, if moderate, "distant
neutral island" effect, register mismatch between formal WordNet glosses and
corpus prose being the likely cause.

**`structural_concepts`** is a second `point_role="reference"` subset,
built the opposite way on purpose: extracted *from* the corpora's own
expression text, so it's geometrically closer to the data by construction —
useful for labeling where corpus clusters actually sit, at the cost of no
longer being topic-neutral (it exists because the corpora use this
vocabulary). Use whichever fits the question: `concept_backbone` when
independence from the corpus matters, `structural_concepts` when
interpretive proximity matters more.

**Why not mine `entity_anchors`** (the same field `emergent_entities`
pools): checked first, and it doesn't work. `entity_anchors` captures
concrete named things ("Scientology", "charismatic leader"), not abstract
social-structure vocabulary — even summed across the *entire* corpus with no
threshold, words like "control" (4 mentions), "authority" (12),
"manipulation" (3), "harm" (0) barely register as tagged anchors, at any
frequency threshold. `extract_structural_concepts.py` instead tokenizes the
actual expression prose (`embedding_text`) across all three corpora and
keeps only tokens that:

1. Are valid **Open English WordNet** (`oewn:2024`) lemmas — the same
   lexicon `build_concept_backbone_en.py` uses, no new dependency. This
   discards proper nouns, typos, and non-English tokens (MIVILUDES is
   mostly French) essentially for free, since none of those are WordNet
   entries, and supplies a ready-made English gloss for each survivor.
2. Fall in an **in-domain lexicographer file** (`adj.all`, `adj.pert`,
   `noun.person`, `noun.cognition`, `noun.act`, `noun.communication`,
   `noun.group`, `noun.attribute`, `noun.state`, `noun.relation`,
   `noun.possession`, `noun.motive`, `noun.feeling`, `noun.phenomenon`,
   `noun.process`, `noun.Tops`) — restricting to social/relational/
   psychological/group-dynamics domains and excluding concrete/physical
   ones (places, objects, body parts, substances, dates, quantities) that
   pass plain WordNet-membership but aren't structural concepts in the
   relevant sense. Verbs are excluded entirely: the target vocabulary names
   a concept, attribute, or role, not an action.
3. Aren't a **named-entity instance** (WordNet's `instance_hyponym`
   mechanism, the same check `build_concept_backbone_en.py` uses) — catches
   further proper-noun leakage a plain WordNet-membership check wouldn't,
   though not all of it (see below).

A word is counted once per *expression* containing it (not once per raw
occurrence), so one repetitive sentence can't inflate a count. A small
number of high-frequency words whose top-ranked WordNet sense is clearly
wrong for how the corpora use them (e.g. "religious" picking the noun
"monk" sense instead of the adjective) are hand-corrected via a
`GLOSS_OVERRIDES` dict; a handful of proper-noun leaks and wrong-sense
matches the domain filter didn't catch were spot-checked and excluded via
`EXCLUDE_WORDS`.

Extending the target size from an initial 600 to the current 1,500 surfaced
a systematic gap rather than one-off noise: OEWN's `instance_hyponym`
marking is incomplete for minor historical figures, so common words
coinciding with their WordNet entries leaked through as "structural
concepts" ("smith", "land", "king", "richardson", ...) — including two
on-topic-but-wrong leaks that would have undermined the whole point of this
dataset: "hubbard" (matches L. Ron Hubbard's WordNet bio entry) and
"iskcon" (a specific named sect, the Hare Krishnas) are exactly the kind of
named-person/named-group contamination this filter exists to keep out —
structural concepts describe roles and dynamics, not specific people or
groups (that's `emergent_entities`' job). A systematic regex sweep for
biographical-looking glosses (`United States \w+`, a birth year, a
`(YYYY-YYYY)` span) caught 53 of these at once; a few more (acronym
collisions like "LET" = Lashkar-e-Taiba, "SHAPE" = NATO's Supreme
Headquarters; demonyms; wrong senses) were spot-checked and excluded or
overridden by hand. Verified zero biographical-pattern matches remain in
the final 1,500 — but neither list is exhaustive, and a pool this size will
have residual noise beyond what any spot-check catches; this is documented
rather than claimed fully clean, in the same spirit as the two
known-but-deferred extraction issues already noted in Methods.tex.

**Target size**: not picked in advance. 7,053 candidates survive WordNet
and domain filtering out of 33,113 unique post-stopword tokens; mentions
per term fall off gradually (70 at rank 600, 40 at rank 1,000, 24 at rank
1,500), with proper-noun leakage and per-term mention counts both degrading
noticeably past that point — checked directly (`total_mentions.most_common()`
against the full qualifying pool) before settling on 1,500 as the practical
ceiling, rather than assuming 600 (an earlier, arbitrary middle-of-range
choice) was the natural stopping point.

On the current corpus: 33,113 unique post-stopword tokens tokenized, 1,500
kept after WordNet/domain filtering (the chosen target size), ranked by
expression-frequency. Top by mentions: "religious" (2,688), "movement"
(1,512), "church" (1,250), "cult" (1,229), "group" (1,129), "religion"
(1,066) — genuinely structural/relational vocabulary, unlike entity_anchors'
top mentions (named groups).

```
python -m thesis_corpus.extract_structural_concepts
```

Output: `dictionaries/structural_concepts_candidates.csv` — `concept_id`
(`sc_0001`, ...), `concept_en`, `gloss_en`, `total_mentions`,
`literature_mentions`, `miviludes_mentions`, `interviews_mentions`,
`n_corpora`, `is_generic` (always `"true"` here — the domain/lexfile filter
already does the curation the column name implies; kept for schema
continuity and as a place to hand-flip an outlier to `false` later without
a code change).

**Embedding**: no new script needed — `embed_concept_backbone.py` is
already fully generic over its `--input`/`--output` CSV (reads any
`concept_en`/`gloss_en` pair, embeds `"<concept_en>: <gloss_en>"`, preserves
every other input column into the output JSONL). Run on the Ollama-serving
machine (see "Running this on Windows" above):

```
python -m thesis_corpus.embed_concept_backbone \
  --input dictionaries/structural_concepts_candidates.csv \
  --output dictionaries/structural_concepts_embedded.jsonl
```

`build_shared_space.py` requires this file to exist (fails with a clear
message naming the exact command above if it doesn't) and pools it via
`load_structural_concepts_points()`, reconstructing `mention_distribution`
from the CSV's per-corpus mention columns — the same provenance-metadata
role it plays on `emergent_entities` points.

## Emergent entities as their own point-set (`build_shared_space`)

Every extracted expression (Stage 2, above) is tagged with `entity_anchors`
— the named entities/dimensions it mentions (e.g. "Scientology",
"charismatic leader") — and Stage 2 already embeds each one individually,
writing a per-item `entity_anchor_vectors` map (`{anchor_string: 1024-d
vector}`). Until now this sat unused in the archive. `build_shared_space.py`
now pools these into their own set of points in the shared space, called
**emergent entities**: named entities/groups/concepts mentioned *by* the
corpora themselves, as distinct from a corpus-expression point (a claim a
source makes, `point_role="expression"`) or the concept backbone (an
external, corpus-independent vocabulary, `point_role="reference"`) —
emergent entities get `point_role="emergent"`. See "Point roles" below for
the full three-way distinction. This gives named entities an actual
position relative to both the corpus expressions that mention them and the
topic-neutral concept backbone, rather than leaving them as a buried
per-item list field.

**Normalization**: each anchor string is lowercased, stripped, and has
internal whitespace runs collapsed to one space (`"Charismatic  Leader"` and
`"charismatic leader"` merge to the same entity) before pooling. Anchors are
pooled once **across all three corpora together** (not per-corpus), so the
same named entity mentioned in literature, MIVILUDES, and an interview
becomes a single point.

**Frequency threshold**: 19,596 unique normalized anchors exist across the
whole corpus, most mentioned only once or twice — mostly noise (typos,
overly specific one-off phrases). Only anchors mentioned **at least 3
times** across all corpora get their own point (`--entity-anchor-min-mentions`,
default 3), giving 3,251 emergent-entity points — deliberately the same
order of magnitude as the 3,000-entry concept backbone. Top anchors by
mention count: Scientology (3,867), charismatic leader (3,417), NRMs
(1,279), cults (688), Heaven's Gate (577), new religious movements (529),
new age (387), Unification Church (273), Jehovah's Witnesses (233),
brainwashing (204).

Each emergent-entity point carries the same minimal shape as every other
point (`source_dataset="emergent_entities"`, `point_role="emergent"`,
`key`/`label` = the normalized anchor text, `shared_space_vector`), plus a
`mention_distribution` field: a per-corpus mention count (e.g.
`{"literature": 3683, "miviludes": 130, "interviews": 54}` for
"scientology") — provenance metadata only, not used in the PCA fit, that
tells apart an entity mentioned near-exclusively in one corpus from one that
genuinely recurs across all three.

## Point roles

Every point in the shared space carries a `point_role`, cutting across
`source_dataset` to group the seven datasets into three kinds of thing:

| `point_role` | Datasets | What it is |
|---|---|---|
| `expression` | `literature`, `miviludes`, `interviews`, `miviludes_criteria` | A criterion expression extracted from a text, or the MIVILUDES's own criterion text — something a source actually said |
| `reference` | `concept_backbone`, `structural_concepts` | A backdrop vocabulary point, not itself a claim any source makes. Two subsets, kept distinct: `concept_backbone` is topic-neutral (WordNet, not derived from any corpus, an independent yardstick); `structural_concepts` is corpus-derived (extracted from the corpora's own expression text, geometrically closer to the data, but not topic-neutral) |
| `emergent` | `emergent_entities` | A named entity/group/concept mentioned *by* the corpora themselves — corpus-derived like an expression, but a recurring reference object rather than a claim |

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
- Each MIVILUDES criterion contributes **one** point — its French embedding
  (`embedding_vector_fr`, the official original) only. The English
  translation is kept purely as a display label (`label_en`) on that same
  point, not embedded separately; translation fidelity between the two is
  instead checked directly via raw-space (pre-PCA) cosine similarity between
  `embedding_vector_fr` and `embedding_vector_en`, printed for all 17
  criteria before pooling (see "Diagnostics" below) — this gets the same
  sanity check the old two-points-per-criterion approach gave, without
  spending a second near-duplicate point in the analytical space to get it.
- Each concept-backbone entry contributes **one** point (`embedding_vector`).
- Each structural-concept entry contributes **one** point (`embedding_vector`,
  plus `mention_distribution`) — see "Structural concepts, a second
  reference subset" above. Requires `structural_concepts_embedded.jsonl` to
  exist; the script fails with a clear message (naming the exact embed
  command) if it doesn't.
- Each emergent entity mentioned at least 3 times across all corpora
  contributes **one** point (see "Emergent entities as their own point-set"
  above).
- Total pooled points is logged at runtime, not asserted against a
  hardcoded constant (it will keep changing as the corpus grows, the
  emergent-entity threshold is adjusted, or the structural-concepts target
  size changes): **44,325** points on the current corpus (35,621 literature
  + 732 MIVILUDES + 204 interviews + 17 MIVILUDES criteria + 3,000 concept
  backbone + 1,500 structural concepts + 3,251 emergent entities) — the
  three expression-corpus counts are *after* the duplicate/short-fragment
  filter below (raw archive sizes are larger: 39,236 / 914 / 230).

**Standardization**: `StandardScaler` (zero mean, unit variance per
dimension) runs before PCA. Every vector already comes from the same
embedding model, but the six datasets differ a lot in register (academic
prose, government French, casual interview speech, bare word+gloss
dictionary entries, bare named entities) and could plausibly carry different
per-dimension distributions — standardizing is a defensive measure against
any one dataset dominating the fit purely due to scale, not a claim that
such an imbalance is known to exist.

**Dimensionality is not a fixed constant**: PCA is first fit at full rank
to get the complete explained-variance curve — written to
`processed/shared_space/variance_curve.{csv,json,png}` so the choice is
inspectable rather than asserted — and the smallest `k` reaching 95%
cumulative variance is picked from that curve. On the actual pooled data:
`k=394` for 95.0% (curve is fairly gradual, not a sharp knee — e.g. 61% at
k=100, 90% at k=300 — so this is a real but not dramatic compression;
worth knowing when interpreting distances in the shared space).

**Extraction-noise filter** (applied inside `load_corpus_points`, never on
the archive itself): two known-and-deferred extraction issues (Methods.tex,
"Vectorising Scholarly Work") are filtered at pooling time. Exact-duplicate
expressions within the same document (keeping the first occurrence) and
expressions under `MIN_EXPRESSION_WORDS` (5) words are dropped before the
point is created; `response_rank` is computed *before* filtering, over
every item in original archive order, so a dropped item doesn't shift the
rank of items after it. On the current corpus: 775 duplicates + 3,048 short
fragments removed (literature 707/2,908, MIVILUDES 67/115, interviews
1/25) — logged at build time as `{"duplicates": N, "short_fragments": M}`
per corpus.

**Diagnostics printed at build time** (not pass/fail checks): MIVILUDES
criteria FR/EN raw-embedding cosine similarity, reporting the full
distribution (mean, median, 10th percentile, min) rather than just
mean/min, with any pair below `COSINE_FLAG_THRESHOLD` (0.70, not 0.90 --
see below) named for manual inspection. On the current corpus: mean 0.873,
median 0.901, p10 0.806, min 0.500 (`crit-legal-disputes`) — only that one
pair falls under 0.70. **Why 0.70, not 0.90**: a flat <0.90 rule initially
flagged 8 of 17 pairs, including `crit-legal-disputes` (0.50) and
`crit-indoctrination-of-children` (0.71); hand inspection found both were
accurate translations (\"La déstabilisation...\"/\"Mental
destabilization...\" style short official-French phrases against slightly
more explanatory English glosses) -- short phrases just embed less stably
cross-lingually than length alone would suggest, and a flat 0.90 threshold
conflated that expected noise with genuine mistranslation. 0.70 is
calibrated to catch the latter without flagging the former. The shared
`_report_translation_fidelity` helper this uses is written to be reused by
an analogous, larger check over MIVILUDES's 732 expressions, planned once
they're translated (see "Known limitations" below) but not yet
implemented -- worth bearing in mind whenever comparing short
texts (this also applies to emergent entities, often 1-3 words) across
languages in this space.

```
python -m thesis_corpus.build_shared_space
```

No Ollama needed — pure `numpy`/`scikit-learn`/`matplotlib` on data that's
already local (same `requirements-reduce_embeddings.txt`, now with
`matplotlib` added). Never modifies any of the five source files, only
writes new ones under `processed/shared_space/`.

### Output

```
thesis/corpus/processed/
  shared_space/
    embedding_space.jsonl   # one row per pooled point: source_dataset, point_role, key, label,
                             # label_en, attribution, claim_mode, epistemic_status, response_rank,
                             # mention_distribution, shared_space_vector
    variance_curve.csv      # n_components, cumulative_variance -- the full curve
    variance_curve.json     # {curve, chosen_k, variance_at_k, threshold}
    variance_curve.png      # plot of the above
```

All cross-corpus output lives under `processed/shared_space/` — a sibling
of `processed/<corpus>/`, kept structurally distinct from any single
corpus's own pipeline output (which is what `processed/<corpus>/` holds).
This is also the natural home for anything built **on top of** this space
later — e.g. a further 2D/3D projection for visualization (TouchDesigner or
otherwise): that kind of step would read `embedding_space.jsonl` and write
its own file alongside it here, rather than a fixed-dimension shared space
and a later visualization-specific reduction living in different places.

`embedding_space.jsonl` is gitignored (large, and like every other embedding
output derived from copyrighted/participant text). The variance-curve files
are small and tracked. `key` (`document_id:chunk_index` for corpus items,
`id` for MIVILUDES criteria, `concept_id` for concept-backbone entries, the
normalized anchor text for emergent entities) is what any later reduction
should carry through unchanged, so a point can always be traced back to
this file, and from there back to the original corpus archive it came from.

`point_role` is the universal three-way split (see "Point roles" above):
`expression` (literature/miviludes/interviews/miviludes_criteria),
`reference` (concept_backbone), or `emergent` (emergent_entities).

Per-point fields beyond `source_dataset`/`point_role`/`key`/`label`/
`shared_space_vector`, all `null` where not applicable rather than given a
fabricated default:

- `label_en`: the English translation, MIVILUDES-criteria points only —
  display-only, not embedded (see above).
- `attribution`: each corpus item's annotation-stage attribution tag
  (`author`, `cited_author`, `participant`, `institution`, `journalist`,
  `unspecified` -- see `ollama_client.py`'s schema), straight through from
  the source archive. `null` for MIVILUDES criteria, concept-backbone, and
  emergent-entity points, which were never annotated this way. This is what
  lets an interview point be filtered by who said it -- in particular,
  separating the interviewee's own statements (`participant`) from the
  interviewer's questions (`unspecified`, unless the interviewer is
  themselves quoting someone), which otherwise end up pooled together
  indistinguishably since both speakers' turns can land in the same
  ~300-700 word chunk.
- `claim_mode` / `epistemic_status`: the same annotation-stage tags,
  corpus-item points only (`null` elsewhere) — how the expression makes its
  claim (a direct statement, a definition, a reflective question, ...) and
  its epistemic status (asserted, contested, negated, speculative, ...).
- `response_rank`: interviews only (`null` elsewhere). Interviews open with
  a free-listing prompt ("what comes to mind when you hear the word
  cult?"), and order of mention is a standard cognitive-salience proxy in
  prototype theory (first-mentioned = most prototypical). This is the
  1-indexed position of the item within its own interview document, in the
  order the archive already lists them — not filtered to the interviewee's
  turns only, so an interviewer's question can also carry a rank.
- `mention_distribution`: emergent-entity and structural-concept points only
  (`null` elsewhere, including `concept_backbone`, which has no
  corpus-mention notion). A per-corpus mention count (e.g.
  `{"literature": 3683, "miviludes": 130, "interviews": 54}` for
  "scientology") — provenance metadata, not used in the PCA fit, that tells
  apart a term mentioned almost exclusively in one corpus from one that
  recurs across all three.

After writing the output, the script prints diagnostic (not pass/fail)
sanity checks: MIVILUDES criteria FR/EN raw-embedding cosine similarity
(see above, printed before pooling since only the French vector survives
into the shared space), the top 20 most-mentioned emergent entities (with
their per-corpus mention distribution), and mean vector norm by
`source_dataset` (flags any one dataset being pushed to the periphery or
center relative to the others).

## 3-D visualization projections (`visualize_3d`)

`embedding_space.jsonl`'s 394 dimensions can't be plotted directly. This
script reads it once and writes three separate 3-D projections, one per
method, so the shared space can actually be visualized:

- **PCA**: `shared_space_vector`'s columns are already full-rank PCA output
  from `build_shared_space.py` — ordered by descending explained variance
  and mutually uncorrelated. The first 3 principal components are therefore
  exactly the first 3 columns of `shared_space_vector`; this is a slice, not
  a refit.
- **UMAP** (`n_neighbors=20`, `min_dist=0.2`, euclidean metric): a nonlinear
  projection that tends to preserve local neighborhood structure.
- **t-SNE** (`perplexity=30`, euclidean metric, PCA-initialized): a
  nonlinear projection tuned for cluster structure, at the cost of global
  distances being less meaningful than UMAP's or PCA's.

All three are fit directly on the 394-d shared-space vectors (already
standardized + PCA'd upstream in `build_shared_space.py`; no further scaling
applied here).

```
python -m thesis_corpus.visualize_3d
```

Needs `umap-learn` in addition to `numpy`/`scikit-learn`
(`requirements-visualize_3d.txt`). Never modifies `embedding_space.jsonl`,
only writes new files under `processed/shared_space/`.

### Output

```
thesis/corpus/processed/
  shared_space/
    visualization_pca_3d.jsonl    # source_dataset, point_role, key, label, label_en, label_fr,
                                    # attribution, claim_mode, epistemic_status, response_rank,
                                    # mention_distribution, pca_3d_vector
    visualization_umap_3d.jsonl   # ...same fields, umap_3d_vector
    visualization_tsne_3d.jsonl   # ...same fields, tsne_3d_vector
```

Each file has one row per point in `embedding_space.jsonl` (44,325), keeping
every field but the vector unchanged and carrying only its own 3-d vector.
Unlike `embedding_space.jsonl`, these are small (44,325 × 3 floats each) and
tracked in git, same as `variance_curve.*`.

## Corpus-imbalance mitigation (`balanced_analysis`)

Literature is ~97% of expression points (35,621 vs. MIVILUDES's 732,
interviews' 204) — see "Known limitations" below. `thesis_corpus.balanced_analysis`
provides two tools for any *quantitative* claim about the corpus as a whole
(visualizations and the PCA fit itself keep every literature point
unchanged; this is a post-hoc, analysis-time correction, not a re-fit):

- `weighted_centroid(embedding_space_path)` / `per_corpus_centroids(...)`:
  reusable functions computing a grand centroid where literature, MIVILUDES,
  and interviews each contribute equal total weight, regardless of point
  count. On the current corpus, the equal-weighted centroid sits 4.02
  shared-space units from the plain unweighted one — confirming the
  imbalance actually moves a raw statistic, not just a theoretical concern.
- A stratified-by-document literature subsample: `python -m
  thesis_corpus.balanced_analysis` writes
  `processed/shared_space/literature_balanced_sample.jsonl` (2,500 points
  by default, `--sample-size` to change it), grouping literature points by
  `document_id` and sampling proportionally to each document's share of the
  corpus, so no single chunk-heavy document dominates the sample. Same row
  shape as `embedding_space.jsonl` — a drop-in subset.

```
python -m thesis_corpus.balanced_analysis
python -m thesis_corpus.balanced_analysis --sample-size 3000 --seed 7
```

## Known limitations

- **Corpus imbalance**: see "Corpus-imbalance mitigation" above.
- **MIVILUDES = 2 documents**: the MIVILUDES corpus (914 raw / 732 pooled
  expressions) comes from exactly two source documents. Treat as one
  influential operational framework, not a representative sample of French
  state framing broadly.
- **Language asymmetry, mitigation pending**: MIVILUDES's expressions are
  currently embedded in their original French, while both reference
  point-sets (`concept_backbone`, `structural_concepts`) are English-only —
  the same asymmetry the MIVILUDES-criteria FR/EN cosine spread (0.50-0.95)
  already demonstrates adds real noise. `thesis_corpus.translate_miviludes_expressions`
  (new, code-complete) will translate MIVILUDES's 732 expressions to
  English and re-embed them, so `build_shared_space.py` can use the English
  embedding as the point's primary vector (French becomes a `label_fr`
  display field, mirroring `miviludes_criteria`'s `label`/`label_en`
  pattern, just inverted) — needs Ollama, run on the Windows/Ollama machine
  (same handoff pattern as `structural_concepts`'s embedding step), not yet
  run:

  ```
  # On the Ollama-serving machine, from thesis/corpus/:
  python -m thesis_corpus.translate_miviludes_expressions
  # Then copy processed/miviludes/expression_translations_embedded.jsonl back,
  # and rerun build_shared_space.py + visualize_3d.py here.
  ```

  Interview transcripts are deliberately left in their original language
  regardless (see "Running on a different corpus" above) — only 19% of
  interviews are French, a much smaller asymmetry than MIVILUDES's
  near-total French text.
- **Interview sample**: convenience-sampled through the researcher's own
  network (documented in Methods.tex, "Vectorising Prototypes": one
  response excluded for researcher-influence bias, another named the
  researcher's own academic programme a "cult"). Treat as exploratory
  prototype data, not a representative sample of lay usage.
