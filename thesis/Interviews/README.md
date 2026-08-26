# Interviews — raw batch material

This folder holds the **raw** interview data only. The processed corpus
(cleaned transcripts, translations, metadata database, generated LaTeX
appendix, standalone analysis) now lives at `corpus/interviews/` — see
`corpus/README.md` for the full pipeline, naming conventions, and update
procedure. This README just documents what's still here.

There are four batches: Interview Batch 1 (11 interviews), Interview Batch 2
(2 interviews), Interview Batch 3 (7 interviews, several in French), and the
Instagram Batch (6 interviews collected via Instagram DM — its
`Transcirpts-Finished` folder name has a typo but it's in scope, since it's
what the corpus's own `method` field distinguishes against "in person").

## Folder structure

```
Interviews/
  <Batch Name>/
    Transcripts-Raw/          # unedited ASR dumps (untouched by any pipeline)
    Transcripts Edited/       # scaffolding from the `edit-transcripts` skill;
                               # currently just a README — see note below
    Transcripts Finished/     # human-diarized, lightly-cleaned transcripts —
                               # the source of truth corpus/interviews/ is built from
```

`Transcripts Finished` is the actual current source of truth despite the sibling
`Transcripts Edited` folder's name — the diarization/cleanup work the
`edit-transcripts` skill was designed to write into `Transcripts Edited` was, at
some point, produced into `Transcripts Finished` instead, leaving `Transcripts
Edited` with just a stale README.

## Where everything else moved

- Processed corpus (cleaned/translated transcripts, metadata, database,
  analysis): `corpus/interviews/`
- Generation script: `corpus/scripts/build_corpus.py`
- Generated LaTeX appendix: `thesis/04_Appendix/Interview_Corpus/`
  (wired in via `thesis/04_Appendix/4_Appendix.tex`)
- Pipeline docs, naming conventions, update procedure: `corpus/README.md`
- `interviews_source.yaml`'s `source_file` fields point back into this folder
  using repo-root-relative paths (e.g. `thesis/Interviews/Interview Batch
  1/Transcripts Finished/Aug 5 at 16-50.txt`).

## Notes

- `01_Document_administration/a_Packages.tex` has `french` added to its `babel`
  option list (alongside the existing `english`, `ngerman`) specifically to
  support the French transcript text in the appendix; `ngerman` remains the
  default/last language in that list, unchanged.
- Corpus size is currently 26 interviews (11 + 2 + 7 + 6 across the four
  batches). If new batches are added later, add the raw files here first, then
  follow the Step 0–4 pipeline documented in `corpus/README.md`.
