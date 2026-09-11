# Available data — inventory

Snapshot as of 2026-09-11. Three primary corpora, one supporting-resource layer, two
downstream pooled embedding spaces. Paths below are relative to `thesis/corpus/`.

---

## 1. MIVILUDES

French state anti-cult-mission material. Small, dense, used as the "official criteria"
reference layer rather than a bulk text corpus.

- **Documents (2)**
  - `MIVILUDES/Latest Report/` — *Rapport d'activité de la Miviludes 2022-2024* (PDF, French).
  - `MIVILUDES/Sectarian Drifts/Raw Transcript.txt` — "Comment identifier une dérive sectaire ?"
    (the criteria-definition page, French).
- **Structured criteria**: `metadata/miviludes_criteria.json` — **17 criteria**, each with
  `criterion_fr` / `criterion_en`, an `order`, and source text. Already bilingual, embedded
  separately at `metadata/miviludes_criteria_embedded.jsonl`.
- **Extraction (v2, precision-focused)**: `processed/v2/miviludes/run_20260910/` — 126 chunks
  → 108 kept expressions.
- **Subsets you can slice on**: by the 17 named criteria (`crit-mental-destabilization`, etc.);
  by document (report narrative vs. criteria definitions); by extracted expression's matched
  criterion (via `entity_anchors`/`domain_terms` overlap with the criteria list).

---

## 2. Scholar literature

Academic corpus: articles, books, book chapters on NRMs/cults.

- **68 documents** (`metadata/literature.json`): 33 articles, 23 books, 12 book chapters.
  Years 2001–2024. Author, DOI/ISBN, abstract present for most; `tags` field exists but is
  currently empty for all entries (not yet used).
- **Extraction (v2, precision-focused)**: `processed/v2/literature/run_20260910/` — 57 of 68
  docs processed, 6,260 chunks → **5,741 kept expressions** (screen + judge reconciled,
  ~95% judge acceptance rate).
- **Subsets you can slice on**: by `type` (article/book/book_chapter); by publication year or
  decade; by author; by abstract keyword (no tags yet, so this is a text search, not a field
  filter); by extracted `expression_kind`/`claim_mode`/`epistemic_status` once you're working
  from `expressions_v2.jsonl` rather than the raw metadata.

---

## 3. Interviews

Primary fieldwork corpus. Two parallel extraction runs exist for this corpus — read the note
below, they're not interchangeable.

- **28 interviews** (`interviews/metadata/database.json`):
  - By batch: Batch 1 = 11, Batch 2 = 2, Batch 3 = 7, Batch 4 = 2, Instagram Batch = 6.
  - By language: English = 22, French = 6 (the 6 French ones have an English display
    translation, embeddings stay French).
  - By method: in person = 22, Instagram (text) = 6.
  - By location: Germany = 16, France = 5 (+2 more French-labeled variants), Slovakia,
    United States, Spain = 1 each, 1 unknown/redacted.
  - By interviewee gender: male = 14, female = 11, other/woman/redacted = 1 each.
  - By interviewee age: 26–97 (27 of 28 have a recorded age).
- **Two extraction runs — use the second one going forward**:
  - `processed/v2/interviews/run_20260910/` — **archived**, selective/precision-focused (same
    pipeline as literature/MIVILUDES). Only kept ~64 expressions total across 26 interviews;
    drops interviewer speech and short answers. Still the live basis for `shared_space_v2` /
    `Emergent_Entities_v2` / `Geometric_Analysis_Draft_v3` — not yet re-pooled.
  - `processed/interviews_full/interviews/run_20260912/` — **current, exhaustive/recall-first**.
    Every sentence from every speaker (interviewer + participant, including filler) is embedded,
    guaranteed: **669 segments, 667 labeled** (2 unlabeled = model gap, still embedded). Each
    segment carries `attribution` (interviewer/participant), `cult_relevant` (label, never a
    filter), `entity_anchors`, and chunk-level `domain_terms`. Has its own
    `emergent_entities_ranked.csv` and English display translations for French segments
    (`segment_translations_en.jsonl`).
- **Subsets you can slice on**: by batch, language, method, location, or demographics (above);
  by speaker (`attribution` interviewer vs. participant, new run only); by `cult_relevant` label
  (new run only — old run pre-filtered this away); by named entity mentioned
  (`entity_anchors`/`domain_terms`, either run).

---

## Supporting resources (not corpora to analyze, but inputs the pipeline uses)

- `dictionaries/` — ConceptNet + WordNet-derived concept backbones and a hand-curated
  "structural concepts" list, all embedded. Used to screen/validate `domain_terms` candidates
  during extraction, not itself a text corpus.
- `custom_terms/`, `metadata/custom_terms*` — small hand-added term list, same role.

## Downstream pooled spaces (combine 1–3 into one embedding geometry)

- `processed/shared_space/` — v1 (older, selective pipeline for all three corpora).
- `processed/shared_space_v2/` — v2 (current for literature/MIVILUDES; interviews half is
  still the *archived* selective run, not yet re-pooled with `interviews_full`).
- No pooled space yet includes the new exhaustive interview archive — building one (or doing
  interview-only analysis straight off `run_20260912`) is the natural next step once you pick
  a subset to start with.
