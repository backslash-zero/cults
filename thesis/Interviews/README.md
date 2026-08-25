# Interviews — structured corpus

This folder holds the raw interview data for the thesis and the derived, structured
corpus built from it. There are four batches: Interview Batch 1 (11 interviews),
Interview Batch 2 (2 interviews), Interview Batch 3 (7 interviews, several in
French), and the Instagram Batch (6 interviews collected via Instagram DM — its
`Transcirpts-Finished` folder name has a typo but it's in scope, since it's what
the corpus's own `method` field distinguishes against "in person").

## Folder structure

```
Interviews/
  <Batch Name>/
    Transcripts-Raw/          # unedited ASR dumps (untouched by this pipeline)
    Transcripts Edited/       # scaffolding from the `edit-transcripts` skill;
                               # currently just a README — see note below
    Transcripts Finished/     # human-diarized, lightly-cleaned transcripts —
                               # the actual source of truth this pipeline reads from
  Corpus/                     # everything in this README's pipeline produces
    cleaned/<id>/
      transcript.txt          # proofread text (Step 1), corrections applied
      translation_en.txt      # English translation, French interviews only
      corrections.log         # every proofreading correction, per interview (Step 0)
    metadata/
      interviews_source.yaml  # hand-authored metadata (Step 2, script input)
      database.json           # canonical generated database (Step 3)
      database.csv            # flat metadata export (Step 3)
    scripts/
      build_corpus.py         # generates database.json/csv + the LaTeX appendix
    analysis/
      analysis.md             # standalone cross-interview analysis (Step 5)
```

`Transcripts Finished` is the actual current source of truth despite the sibling
`Transcripts Edited` folder's name — the diarization/cleanup work the
`edit-transcripts` skill was designed to write into `Transcripts Edited` was, at
some point, produced into `Transcripts Finished` instead, leaving `Transcripts
Edited` with just a stale README. This pipeline reads `Transcripts Finished`
directly, per instruction, and doesn't touch either raw folder.

The generated LaTeX appendix lives outside this folder, in
`thesis/04_Appendix/Interview_Corpus/` (`appendix.tex`, `database_table.tex`),
wired into the thesis via `thesis/04_Appendix/4_Appendix.tex`.

## Pipeline (reproducible on a future batch)

1. **Proofread (Step 0) — human checkpoint, always required.** Read every
   `Transcripts Finished/*.txt` file and identify only clear typing/transcription
   slips (misspellings, stray characters, missing punctuation) — never grammar,
   filler words, hesitations, repetition, phrasing, slurs, profanity, or existing
   transcription notes, all of which are preserved verbatim. Write one
   `Corpus/cleaned/<id>/corrections.log` per interview listing every proposed
   correction (original → corrected, with location), even if empty. **Do not
   proceed to Step 1 until the corrections have been explicitly approved by a
   human reviewer.** This is the one mandatory pause in an otherwise scriptable
   pipeline.
2. **Prepare (Step 1).** Once approved, write `Corpus/cleaned/<id>/transcript.txt`
   with only the approved corrections applied. For French-language interviews,
   also write `translation_en.txt` — an English translation done directly by the
   assistant compiling the corpus, not a certified translation; worth spot-checking
   against the original for nuance.
3. **Metadata (Step 2).** Hand-author `Corpus/metadata/interviews_source.yaml`:
   one entry per interview with `id`, `date_time` (ISO 8601; explicit "unknown"/a
   `date_time_precision` note where the source doesn't give a full timestamp),
   `method` ("in person" by default, "Instagram" where indicated), `language` +
   `translated`, `interviewer`, `participant_id`, `location`, and interviewee
   demographics as captured in the source (explicitly marked "redacted" or
   flagged where the source itself is incomplete — never invented). This step is
   necessarily manual: header formats vary too much across files (field order,
   missing sections, one missing date header, one fully redacted file) for a
   reliable automated parse. `n_questions`, `n_answers`, and `total_word_count` are
   deliberately **not** hand-authored here — they're computed by the build script
   directly from each cleaned transcript so they can never drift from the text.
4. **Build the database (Step 3) + LaTeX appendix (Step 4).** Run
   `python3 Corpus/scripts/build_corpus.py` (requires PyYAML). It reads
   `interviews_source.yaml` plus every `cleaned/<id>/*.txt`, computes the derived
   counts, and writes `database.json` (canonical: full text + translation +
   metadata + a link to that interview's `corrections.log`), `database.csv` (flat
   metadata only), `04_Appendix/Interview_Corpus/appendix.tex` (one `\subsection`
   per interview, `\label{int:<id>}`), and `database_table.tex` (a `longtable` of
   the metadata, each row `\ref{int:<id>}`-linked to its transcript). Re-running
   the script is idempotent — it regenerates all four outputs from the same
   inputs.
5. **Analysis (Step 5) — standalone, not part of the thesis.** `analysis.md` is a
   plain-Markdown cross-interview research note (themes, divergences, vocabulary
   patterns, distribution stats, data-quality gaps), grounded in specific
   interview IDs and quotes. It is **not** `\input` anywhere in the thesis and has
   no LaTeX counterpart — it's for the researcher to pick and choose from, not for
   direct inclusion in the paper.
6. **This README (Step 6).** Kept in sync whenever the pipeline or folder
   structure changes.

## Deliverables and where they live

| Deliverable | Location |
|---|---|
| Per-interview corrections log | `Corpus/cleaned/<id>/corrections.log` |
| Cleaned transcripts | `Corpus/cleaned/<id>/transcript.txt` |
| English translations (French interviews) | `Corpus/cleaned/<id>/translation_en.txt` |
| Metadata source (hand-authored) | `Corpus/metadata/interviews_source.yaml` |
| Canonical database | `Corpus/metadata/database.json` |
| Flat metadata export | `Corpus/metadata/database.csv` |
| Generation script | `Corpus/scripts/build_corpus.py` |
| Standalone cross-interview analysis | `Corpus/analysis/analysis.md` |
| LaTeX appendix (per-interview) | `thesis/04_Appendix/Interview_Corpus/appendix.tex` |
| LaTeX metadata table | `thesis/04_Appendix/Interview_Corpus/database_table.tex` |
| Appendix chapter wiring | `thesis/04_Appendix/4_Appendix.tex` (input from `0_Appendix.tex`) |

## Notes for whoever runs this next

- The Step 0 approval checkpoint is not automated — it exists so a human reads the
  proposed corrections before any transcript text is altered. Don't skip it, and
  don't apply corrections speculatively "to save a round trip."
- `01_Document_administration/a_Packages.tex` has `french` added to its `babel`
  option list (alongside the existing `english`, `ngerman`) specifically to support
  the French transcript text in the appendix; `ngerman` remains the default/last
  language in that list, unchanged.
- Corpus size is currently 26 interviews (11 + 2 + 7 + 6 across the four batches).
  If new batches are added later, repeat Steps 0–4 for the new files only, then
  re-run `build_corpus.py`, which regenerates the full database/appendix from
  whatever's present in `interviews_source.yaml` and `Corpus/cleaned/`.
