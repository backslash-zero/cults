# A cult prototype, constructed from the corpus

Data inventory: [`00_Available_Data.md`](00_Available_Data.md). Index: [`Results_Draft.md`](Results_Draft.md).

Operationalizes the thesis's own `\S Family resemblance and prototype theory` (Rosch/Mervis:
category members vary in typicality; the most prototypical members share the most attributes
with the rest of the category) directly on this corpus's data — not asserting that the 17
official criteria are equally central, but measuring it.

## Method

Two different centroids, deliberately not conflated:
- **The criteria list's own centroid** (`sectarian_drift_list` in `02_Corpus_Centroids.md`) —
  the mean of the 17 criteria's own vectors. Measures which criterion is most central *to the
  official list itself* (there, "mental destabilization" ranked first).
- **The corpus-wide "cult prototype"** (this file) — the mean of every cult-relevant
  *expression* pooled across all three corpora: all of literature + all of miviludes (both
  100% `cult_relevant=True` already — the selective v2 pipeline only ever kept cult-relevant
  candidates, verified directly) + interviews restricted to `cult_relevant=True` (618 of 669,
  excluding labelled filler and 2 unlabelled segments). Measures which criterion is most
  central to *how this corpus, in aggregate, actually talks about cults* — a genuinely
  different, more data-driven question.

**A real robustness check, not a formality**: the plain pooled centroid is ~93% literature by
point count (5,741 of 6,466) — checked directly, not assumed, since literature dominating by
sheer volume was exactly the mechanism `build_shared_space.py`'s own
`literature_balanced_sample.jsonl` already exists to correct for elsewhere in this toolkit. So
the criteria ranking is reported **twice**: once against the plain pooled centroid, once
against an **equal-weighted** centroid (each corpus's own sub-centroid averaged, regardless of
point count) — reported side by side below, so any conclusion that only holds under one
weighting is visible as such.

Code: `thesis_corpus/analyze_cult_prototype.py`.

## Result: which criteria are most central to the corpus's own usage

| Rank (pooled) | Criterion | cos (pooled) | Rank (equal-wt) | cos (equal-wt) |
|---|---|---|---|---|
| 1 | Radical behavior change | 0.687 | 1 | 0.690 |
| 2 | Authoritarian/opaque group | 0.646 | 2 | 0.662 |
| 3 | Infiltration of public authorities | 0.633 | 5 | 0.635 |
| 4 | Rupture with origin environment | 0.631 | 3 | 0.645 |
| 5 | Difficulty leaving the group | 0.624 | 4 | 0.637 |
| 6 | Alarming dietary change | 0.614 | 6 | 0.626 |
| 7 | Harsh living conditions | 0.602 | 9 | 0.606 |
| 8 | Indoctrination of children | 0.601 | 11 | 0.597 |
| 9 | Contesting public order | 0.598 | 10 | 0.606 |
| 10 | Rejection of outside world | 0.585 | 12 | 0.596 |
| 11 | Physical-integrity harm | 0.584 | 8 | 0.615 |
| 12 | Mental destabilization | 0.581 | 7 | 0.617 |
| 13 | Dubious/exclusive care | 0.577 | 13 | 0.592 |
| 14 | Deceptive recruitment | 0.567 | 15 | 0.564 |
| 15 | Violation of Republic principles | 0.545 | 16 | 0.540 |
| 16 | Financial demands/opacity | 0.542 | 14 | 0.577 |
| 17 | Legal disputes | 0.522 | 17 | 0.526 |

**Robust across both weightings** (rank barely moves): *radical behavior change* (1st in
both), *authoritarian/opaque group* (2nd in both), *legal disputes* (17th, dead last, in
both) — these three conclusions don't depend on how literature's volume is handled.

**Sensitive to weighting** — worth flagging rather than picking one number: *mental
destabilization* jumps from 12th (pooled) to 7th (equal-weighted) — MIVILUDES and interviews
treat it as considerably more central than literature's own huge volume implies once that
volume stops dominating. *Physical-integrity harm* shows the same pattern (11th → 8th).
*Financial demands/opacity* moves the other direction (16th → 14th, still low either way).

**A genuinely interesting divergence from the criteria list's own internal structure**:
`02_Corpus_Centroids.md` found *mental destabilization* to be the 17-item list's own most
central member (its own self-centroid nearest neighbour). Here, ranked against how the corpus
actually talks about cults, mental destabilization is mid-table (7th–12th depending on
weighting) while *radical behavior change* is consistently first. These are two different,
equally legitimate notions of "central" — the official list is organized around mental
destabilization as its conceptual anchor, but the corpus's actual usage is organized more
around behavioral change and authoritarian structure.

## Nearest real expressions to the (pooled) prototype centroid

The single most prototypical statements in the entire corpus, by this measure — a genuine mix
of interview and literature voices once interviews were correctly included in the pool (see
Method):

1. "Religion." *(interview, appears twice — two different French Batch-4 interviewees)*
2. "Un groupe qui suit une doctrine." *(interview)*
3. "Somehow they try to convince people to join the cult." *(interview)*
4. "Une secte." *(interview)*
5. "It is not just religious but a religion." *(literature — Noll 2002)*
6. "Ouais, religion." *(interview)*
7. "Une communauté." *(interview)*
8. "This is Christianity." *(literature — Hayes 2006)*
9. "It is a religious organization without a doubt." *(literature — Davis & Hankins 2002)*

Full data: [`data/criteria_centrality_ranking.csv`](data/criteria_centrality_ranking.csv),
[`data/criteria_centrality_ranking_equal_weighted.csv`](data/criteria_centrality_ranking_equal_weighted.csv),
[`data/nearest_expressions_to_prototype.csv`](data/nearest_expressions_to_prototype.csv).
