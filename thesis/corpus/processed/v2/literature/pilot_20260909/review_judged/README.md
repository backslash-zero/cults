# Extraction v2 literature pilot -- manual review packet (20260909)

**What this is.** A side-by-side manual review of the frozen v1 extraction and the conservative v2 extraction on the same 28 literature chunks. Nothing here has been judged automatically: every manual-review column is blank and is yours to fill.

**Files**
- `pilot_review.csv` -- 87 rows (v1: 56, v2_qwen3-4b: 31). Ordered random stratum first, then forced-regression chunks; within a stratum by document, chunk, arm, rank. Blind: no.
- `pilot_chunk_comparison.csv` -- one row per chunk: v1 count and texts, v2 retained/rejected counts, rejection codes, and two blank columns (`missed_expression`, `notes`) for anything clearly useful that v2 omitted.
- `pilot_rejected_candidates.csv` -- every candidate the deterministic screen rejected, with its rejection code, the model's fields, the chunk text, and blank `rejection_correct` / `should_have_been_retained` / `reviewer_notes` columns.
- `validation.json` -- the independent hard-zero re-check of every arm (span resolution, verbatim = embedding, no interviewer rows, no high-confidence corruption, candidate/chunk accounting).

**Judge columns.** When an arm was judged by the second model, the `judge_*` columns carry that model's answers (model-judged, not human review) and the v2 rows are the judge-accepted set; judge-rejected rows are in `pilot_rejected_candidates.csv` with code `judge_rejected`.

**Row budget.** v1 arm: seeded sample of at most 2 items per chunk (the full v1 list of every chunk is in the comparison file). v2 arm: none (every retained v2 row is included).

**How to read a row.** `verbatim_expression` is the exact source span (for v1 rows, the v1 `source_quote`). `embedding_text` is what was / would be embedded; for v2 it is identical by construction (`embedding_equals_verbatim`). `context_window` is the full chunk the span comes from. `screen_flags` are non-fatal observations from the deterministic screen (e.g. `long`, `spaced_accent`, `isolated_capital`, `possible_heading_or_acronym`, `short_association`) -- they are hints for you, not judgements.

**Manual columns** (leave blank if not applicable):
- `faithful`, `self_contained`, `cult_relevant`, `textually_intelligible`, `atomic`: yes / no.
- `attribution_correct`, `claim_mode_correct`, `epistemic_status_correct`: yes / no / unclear; `recommended_epistemic_status` if you would change it.
- `better_span_exists_in_chunk`: yes / no -- is there a clearly better expression in `context_window` than the one selected?
- `extraction_issue`: none | off_topic | contextless_fragment | truncated | overlong | multiple_claims | ocr_or_encoding_error | heading_or_name | interviewer_question | other (several allowed, separated by `;`).
- `reviewer_notes`: free text.

**What the pilot is for.** A decision gate: precision (are retained expressions usable?), yield (how many remain per chunk?), missed valuable material, and whether the known v1 failure cases in the forced stratum disappeared. No accuracy claim is made from it.

**Validation output**
```
[
  {
    "arm": "arm_qwen3-4b",
    "screen_retained": 32,
    "judge": "judge_qwen3-8b",
    "final_retained": 31,
    "rejected": 46,
    "responses": 28,
    "model_failures": 0,
    "model_failure_rate": 0.0,
    "failure_rate_exceeds_review_blocker": false,
    "hard_zero_problems": []
  }
]
```
