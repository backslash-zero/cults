# Corpus

The unified research corpus for the thesis: interviews, literature, dictionary/
thesaurus entries, and custom terms. File-based JSON/YAML is the source of
truth (git-tracked, human-reviewable diffs); a sync script mirrors it into the
Sanity project (`studio-cults/`) so the frontend can eventually query the same
data. The generated LaTeX appendices in `thesis/04_Appendix/` are also built
from these same files — there is exactly one source of truth per item.

## Folder structure

```
corpus/
  literature/raw/            # collected articles/chapters/reports
  interviews/                # migrated from thesis/Interviews/Corpus/
    cleaned/<id>/transcript.txt, translation_en.txt, corrections.log
    metadata/interviews_source.yaml, database.json, database.csv
    analysis/analysis.md     # standalone, not part of the thesis text
  dictionaries/raw/          # collected dictionary/thesaurus source material, if saved
  custom_terms/              # no raw/ — terms are authored directly in the YAML
  metadata/
    literature_source.yaml, dictionaries_source.yaml, custom_terms_source.yaml
    literature.json/.csv, dictionaries.json/.csv, custom_terms.json/.csv
  scripts/
    build_corpus.py          # generates all JSON/CSV/LaTeX from the *_source.yaml files
    sync_sanity.mjs          # pushes the JSON databases into Sanity (studio-cults)
```

Raw interview batch material (`Transcripts-Raw`, `Transcripts Finished`,
`Transcripts Edited`) stays at `thesis/Interviews/<Batch Name>/` — only the
processed corpus moved here. See `thesis/Interviews/README.md` for that layer.

## Naming conventions

Every item gets a lowercase, hyphen-only ID that doubles as its LaTeX
cross-reference label (`\label{}`/`\ref{}`) and its Sanity `corpusId` field —
one ID, usable everywhere.

| Type | Scheme | Example |
|---|---|---|
| Interview | `<batch>-<slug>` | `b1-aug05-1650`, `ig-02` |
| Literature | `lit-<firstauthor><year>-<title-slug>` | `lit-durkheim1912-elementary-forms` |
| Dictionary/thesaurus entry | `dict-<source-slug>-<term-slug>` | `dict-thesoz-sekte` |
| Custom term | `term-<slug>` | `term-semantic-drift` |
| *(reserved, not built yet)* Chunk | `<parent-id>-chunk-<n>` | `lit-durkheim1912-...-chunk-3` |
| *(reserved, not built yet)* Named entity | `<parent-id>-ent-<n>` | `b1-aug05-1650-ent-1` |

The same table is reproduced in `thesis/04_Appendix/5_Appendix.tex` so it's
documented in the thesis itself, not just here.

## Pipeline

1. **Collect** raw material into the relevant `raw/` folder (literature,
   dictionaries) or author it directly (custom terms), or run the interview
   proofreading pipeline (unchanged, see `thesis/Interviews/README.md`).
2. **Add a metadata entry** to the type's `*_source.yaml` under
   `corpus/metadata/` — see the schema comments at the top of each file.
3. **Regenerate**: `python3 corpus/scripts/build_corpus.py` (requires PyYAML) —
   rebuilds every type's `database.json`/`.csv` and the LaTeX appendix content
   in `thesis/04_Appendix/`, from whatever's currently in the `*_source.yaml`
   files and (for interviews) `corpus/interviews/cleaned/`.
4. **Recompile the thesis** (`latexmk -pdf thesis.tex` from `thesis/`) to check
   the new appendix content typesets cleanly.
5. **Sync to Sanity** (optional, requires a `SANITY_API_TOKEN` — see below):
   `node corpus/scripts/sync_sanity.mjs`.

## Update procedure — when you tell me you've added new content

- **Literature**: I read the new raw file(s) in `corpus/literature/raw/`,
  propose a `literature_source.yaml` entry (bibliographic metadata) for your
  review before adding it — same "propose, don't silently commit" spirit as
  the interview corrections step.
- **Dictionary entries**: added with the term's definition quoted **verbatim**
  from the source and a precise citation (edition/page) — never paraphrased.
- **Custom terms**: added with your own definition, as given.
- **Interviews**: unchanged — see `thesis/Interviews/README.md` (proofread →
  approve → clean/translate → metadata → regenerate).
- **Every time, regardless of type**: re-run `build_corpus.py`, recompile the
  thesis, and re-run `sync_sanity.mjs` if a Sanity token is configured.

## Sanity sync

`studio-cults/` is a real, provisioned Sanity project (see its
`sanity.config.ts`) with four document types matching the four corpus types
(`schemaTypes/interview.ts`, `literatureItem.ts`, `dictionaryEntry.ts`,
`customTerm.ts`), each carrying a `corpusId` field. The sync script **never**
sets a document's `_id` directly (Sanity IDs stay Sanity-generated, per Sanity's
own guidance) — it looks up each record by `corpusId` and creates or updates
accordingly, so re-running it is always safe/idempotent.

To actually run it you need a Sanity API token with write access:
1. `cd studio-cults && npx sanity login` (if not already logged in), then
   create a token at manage.sanity.io → your project → API → Tokens (Editor
   permission is enough).
2. `export SANITY_API_TOKEN=<your token>` (or put it in a local `.env` —
   **do not commit it**).
3. `node corpus/scripts/sync_sanity.mjs` from the repo root.

No token exists in this repo yet — until one is configured, Sanity stays empty/
out of sync with the file-based corpus, which remains the authoritative source
either way.

## Frontend

Not wired up yet — `frontend/` (SvelteKit) has no Sanity client or data-fetching
code. Once the frontend design work starts, it should query Sanity the normal
way (a `@sanity/client` + GROQ), reading the same documents this sync script
populates — no separate data layer needed.
