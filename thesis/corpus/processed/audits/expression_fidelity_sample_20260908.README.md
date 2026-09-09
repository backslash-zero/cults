# Expression fidelity sample -- manual review sheet

Generated 2026-09-08T18:20:11.111350+00:00 (seed=42, script version 1.0.0, git commit 46d0d4102d1846d31fd0a42f292a6955d92251b6).

## What this is

A stratified, reproducible sample of 100 extracted expressions drawn
directly from the three raw source archives (literature (/Users/celestinmeunier/Documents/cults/thesis/corpus/processed/literature/criterion_expressions.jsonl), miviludes (/Users/celestinmeunier/Documents/cults/thesis/corpus/processed/miviludes/criterion_expressions.jsonl), interviews (/Users/celestinmeunier/Documents/cults/thesis/corpus/processed/interviews/criterion_expressions.jsonl)), for a manual
fidelity check: does each machine-extracted expression, read against its
own surrounding context, actually say what its metadata claims it says?

**This is a targeted quality-control sample, not a statistical estimate
of universal pipeline accuracy, and not a replacement for close reading
in the thesis.** 100 items, drawn once, cannot establish an error
rate for the ~40,000-expression archive as a whole -- they can surface
whether the extraction/tagging pipeline is producing plausible output on
a cross-section of real cases, and catch any systematic problem worth
knowing about before relying on the pipeline's output further.

Sample allocation (deliberately non-proportional to each source's real
size, so no single source dominates the review purely by candidate-pool
size): literature=34, miviludes=33, interviews=33.

## How to review each row

For every row in `expression_fidelity_sample_20260908.csv`, read `embedding_text` against its own
`context_window` (the surrounding source text the expression was
extracted from), then fill in the six manual-review columns:

- **`expression_faithful`** -- does `embedding_text` accurately and
  adequately represent the relevant claim in the surrounding
  `context_window`? (e.g. `yes` / `no` / `partial` -- use whatever
  convention you prefer, this script does not prescribe one)
- **`attribution_correct`** -- is the statement attributed to the
  correct speaker or source (`attribution` column)?
- **`claim_mode_correct`** -- is the claim-mode label (`claim_mode`)
  plausible when read in context?
- **`epistemic_status_correct`** -- is the epistemic label
  (`epistemic_status`) correct when read in context?
- **`recommended_epistemic_status`** -- only fill this in if the current
  status is clearly wrong; otherwise leave blank. Do not fill this in
  merely because you'd have phrased the label differently.
- **`reviewer_notes`** -- briefly state the reason for a "no", any
  ambiguity, or a proposed correction. Leave blank if there's nothing to
  add.

Every manual-review column starts blank in this script's own output --
nothing here has been prefilled, predicted, or automatically scored.

## Source-by-epistemic-status distribution

Full counts (eligible population and selected sample, per source and
per status) are recorded in `expression_fidelity_sample_20260908.config.json`, not repeated here.

## Scope note

This script draws the sample and leaves every judgement column blank.
It does not decide, suggest, or imply whether any row is faithful or
correctly labelled -- that determination is yours alone.
