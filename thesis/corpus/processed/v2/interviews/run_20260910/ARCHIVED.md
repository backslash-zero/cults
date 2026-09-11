# This run is superseded for interviews (not deleted, not moved)

Added 2026-09-11. This note is documentation only — nothing in this
directory has been changed, moved, or deleted.

## Status

This is the last run of the **selective** extraction pipeline
(`extract_v2.py --corpus interviews`) against the interview corpus: 26
interviews, 80 chunks, 64 kept expressions total. It remains the live,
unmodified basis for:

- `processed/shared_space_v2/embedding_space.jsonl` (the interview points
  pooled into the shared cross-corpus space)
- The `Emergent_Entities_v2` thesis appendix
  (`thesis/04_Appendix/Emergent_Entities_v2/`)
- The `Geometric_Analysis_Draft_v3` report
  (`thesis/04_Appendix/Geometric_Analysis_Draft_v3/`)

None of the above have been rebuilt or touched. Do not delete, move, or
rename anything in this directory — that would break the provenance those
outputs still point back to.

## Why it's superseded

Going forward, new interview extraction work uses a separate, exhaustive
pipeline (`extract_interviews_full.py`), not this one. This run's prompt
asks the model to find only "the few... genuinely worth keeping"
cult-relevant spans (at most 4/chunk) and discards all interviewer speech
outright — under 3 kept expressions per interview, and it specifically loses
short-but-real answers ("Illuminati.", "Tomato cult!"). The new pipeline
extracts every expression from every speaker, filler included, and records
cult-relevance as a label rather than a reason to drop something — see
`thesis_corpus/README.md`'s "Exhaustive interview extraction" section and
`thesis/corpus/ANALYSIS_OVERVIEW.md`'s "Interviews, going forward" note for
the full rationale (in short: the shared space is a geometric/clustering
structure that benefits from more points, and post-embedding proximity to
cult-relevant material is itself a usable relevance signal — one the
extractor no longer needs to guess at before anything is embedded).

The new pipeline's output lives at
`processed/interviews_full/interviews/run_<tag>/` — a separate, standalone,
single-corpus archive. It is not pooled into `shared_space_v2` and has not
replaced anything referenced above.
