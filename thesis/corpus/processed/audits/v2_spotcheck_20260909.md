# v2 spot-check: quality-review samples (2026-09-09)

Drawn with `sample_v2_expressions.py`, seed=42, from run-tag `20260910` (the same
run used to build `shared_space_v2`):

| corpus | n sampled | of | method | file |
|---|---|---|---|---|
| literature | 100 | 5,741 | stratified by document (53/57 docs represented — the other 4 documents produced 0 final expressions each, confirmed via `summary.json`'s `documents.done: 57`, not an incomplete run) | `processed/v2/literature/run_20260910/review/quality_sample_100_stratified_by_document_20260909.csv` |
| miviludes | 40 | 108 | uniform random (2/2 docs) | `processed/v2/miviludes/run_20260910/review/quality_sample_40_uniform_random_20260909.csv` |
| interviews | 30 | 64 | stratified by document (25/26 docs — the 26th interview produced 0 expressions, already known) | `processed/v2/interviews/run_20260910/review/quality_sample_30_stratified_by_document_20260909.csv` |

Manual-review columns are intentionally left blank in the CSVs themselves (per
the script's design, for a human reviewer). This note records an automated
check plus an independent read of a subset of rows, done as this task's
spot-check pass — not a substitute for the blank manual-review columns.

## Automated check

For all 170 sampled rows: `verbatim_expression` is a verbatim substring of its
own `context_window` (0 mismatches), and no row has `judge_faithful == False`
or a non-`none` `judge_extraction_issue`. This mostly re-confirms the judge's
own labels are internally consistent — expected, since the source file only
contains judge-accepted rows — rather than an independent quality signal.

## Independent read (~30 rows across the three corpora)

Literature and MIVILUDES rows read as expected: verbatim claims/definitions/
named groups from NRM scholarship and the French sectarian-drift report,
correctly attributed (`author`/`cited_author`/`participant`), sensible
`claim_mode`/`epistemic_status`. One literature row's expression looked
unrelated to its printed context at first glance (`"Heathenry as a generic
term..."` next to a health/wealth-teachings passage); the full 2,800-character
context window shows this is a dictionary source where one entry ends and the
next begins mid-chunk, and the expression is genuinely present verbatim — a
false alarm from truncating the context window for a quick read, not an
extraction error.

## One genuine finding: `min-expression-words 0` and "Sept?"

v2's shared-space build uses `--min-expression-words 0` (see
`build_shared_space.py`'s v2 CLI flags), deliberately not filtering by a
word-count floor because a floor can't tell a meaningful short item
(`"Tomato cult!"`, `"Illuminati"`, `"Gourou"`) from a meaningless fragment.
Checking every ≤2-word expression across all three corpora (1,132/5,741 in
literature, 7/108 in MIVILUDES, 17/64 in interviews) confirms this worked as
intended in all but one case: 16 of the 17 short interview items are genuine
named entities or associations (`"Scientology"`, `"the Moonies"`, `"les
sectes"`, `"New age"`, etc.).

The one exception, from interview `b3-aug23-1213`:

> Interviewer (French): *"Quand je te dis le mot secte à quoi tu penses?"*
> ("When I say the word 'secte' \[cult], what do you think of?")
> Interviewee (age 97): **"Sept?"**

"Sept?" ("seven?") is almost certainly the 97-year-old participant mishearing
*secte* as *sept*, not a substantive response about cults. The judge model
scored it `faithful/self_contained/cult_relevant/textually_intelligible/
atomic = True` and noted "a direct quote from the interviewee, relevant to
the topic of sects, and self-contained" — true as a transcript fact, but the
string itself carries no propositional content about cults, so its
embedding will sit wherever "Sept?" alone maps in the space, not wherever a
cult-relevant response would.

**Scale:** 1 of 10,646 points in the shared space (~0.01%; 1/64 = 1.6% of the
interviews corpus alone). Not large enough to affect any aggregate result in
this analysis, but worth flagging as a known, real limitation of the
`--min-expression-words 0` choice rather than papering over it. No change
made to the retained run or the point itself — left as an explicit note for
the write-up rather than a unilateral exclusion, consistent with how the
`--min-expression-words`/`--entity-anchor-min-mentions` threshold choices
earlier in this project were surfaced rather than decided silently.

## Verdict

No extraction-quality problems found beyond the single "Sept?" edge case
above. The v2 retained run (`processed/analysis_v2/20260909-224215/`) is fit
to write up.
