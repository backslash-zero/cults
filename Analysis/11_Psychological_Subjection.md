# Is psychological subjection what separates cults from ordinary organizations?

Data inventory: [`00_Available_Data.md`](00_Available_Data.md). Index: [`Results_Draft.md`](Results_Draft.md).
Prior steps this corrects: [`10_Secular_Groups_vs_Sectarian_Drifts.md`](10_Secular_Groups_vs_Sectarian_Drifts.md)
(Results 2 and 3) and [`Plain_Language_Summary.md`](Plain_Language_Summary.md) (finding 7).

MIVILUDES criterion #1 — *"La déstabilisation mentale et la sujétion psychologique ou physique…"* —
is the brainwashing / coercive-control / undue-influence claim. [`10`](10_Secular_Groups_vs_Sectarian_Drifts.md)
appeared to show it was the one criterion ordinary organizations don't match, and therefore the one
carrying the whole framework's discriminating power. **This document tests that claim and withdraws
it**, then reports what the data supports instead.

Code: `thesis_corpus/analyze_psychological_subjection.py`. The length control is a
unit-tested regression guard (`tests/test_analyze_psychological_subjection.py`), asserted against
the real on-disk data, so the retracted claim cannot silently return.

## The retraction

**The claim.** *Mental destabilization* ranked 17th of 17 in mean cosine similarity to 169 ordinary
non-religious organization names (0.289), and was the nearest criterion for only 2 of them.
Conclusion drawn: the basis for assessment generalizes everywhere except at its own conceptual core.

**Why it fails.** *Mental destabilization* is **the longest of the 17 criteria — 29 French words,
against a shortest of 3.** Across the 17, criterion text length predicts absolute similarity to
short organization names at **Spearman −0.613**. A long sentence is far from a two-word name
regardless of what it says. The "nearest criterion for only 2 of 169" figure is argmax over the same
confound: a 29-word criterion almost never wins a nearest-neighbour contest against a 3-word one.

**The length-robust measure.** The *gap* between ordinary organizations and the religious/NRM
baseline on the same criterion cancels length, because the penalty applies to both sides. Its
correlation with length is **−0.039**. On that measure:

| Measure | Spearman vs criterion length | Rank of *mental destabilization* |
|---|---|---|
| Absolute cosine (the retracted basis) | **−0.613** — confounded | **1 of 17** (lowest) |
| Gap vs religious baseline (length-robust) | −0.039 — clean | **6 of 17** (mid-pack) |

Mid-pack, unremarkable, and no basis for a claim about discriminating power.

**Blast radius, checked.** [`07_Cult_Prototype.md`](07_Cult_Prototype.md) also ranks the 17 criteria,
so it was tested for the same confound: Spearman between criterion length and cosine to the corpus
prototype is **−0.150** (pooled) and **+0.093** (equal-weighted). **`07` is clean and stands
unchanged.** The difference is instructive — `07` compares each criterion to a dense centroid of
thousands of corpus expressions, which no single criterion's length systematically penalises,
whereas `10` compares criteria to short names, where the length mismatch acts directly. `10`'s
Result 3 was downgraded from "independent replication" to "consistent with," since *radical
behavior change* is 5 words and its rank-1 position there is partly the same effect (it is 5th of
17 on the length-robust measure).

This is the third artifact of one family in this project, after the +0.061 "The" token and the
r=+0.384 word-count effect in [`08`](08_Secular_Groups_Hypothesis.md). The general rule: **whenever
the two sides of a cosine comparison differ systematically in length or surface form, measure that
difference before interpreting the similarity.**

## What the length-robust measure shows instead: enclosure, not coercion

The criteria where the religious baseline comes closest to parity — i.e. that come nearest to
marking cults out at all:

| Sectarian drift | Gap | French words |
|---|---|---|
| indoctrination of children | +0.005 | 3 |
| violation of Republic principles | +0.007 | 8 |
| difficulty leaving the group | +0.011 | 13 |
| rejection of the outside world | +0.014 | 20 |
| rupture with environment of origin | +0.016 | 7 |

Set aside *violation of Republic principles* — a French-administrative category with almost no
English-language purchase, consistently worst-covered in
[`03_Per_Criterion_Neighbors.md`](03_Per_Criterion_Neighbors.md) — and the remaining four are one
coherent family: **enclosure.** Indoctrinating the children, cutting the member off from family,
refusing the outside world, making exit impossible.

**Caveat that matters: every one of these gaps is far inside the ±0.061 artifact band.** This is an
ordering, not five measured effects. It is reported because the ordering is theoretically
interpretable and length-robust, not because the magnitudes are resolvable.

## Where the literature's coercion discourse actually sits — and it is secular

The clusters answer the user's question better than the criteria do, and the answer was already on
disk. Three of the 59 literature clusters ([`06_Literature_Clusters.md`](06_Literature_Clusters.md))
are about psychological subjection:

| Cluster | n | Binding expressions | Nearest entity | Gap? |
|---|---|---|---|---|
| **40** | 85 | the bare token *brainwashing*, from 8 different books | `brainwashing` (0.872) | no |
| **43** | 59 | *mental manipulation* ×4, *coercive persuasion* ×2, then two real sentences on pressure techniques altering judgment | `coercive persuasion` (0.789) | no |
| **48** | 39 | "the 'cults' apply real 'brainwashing'", *cultic brainwashing theory*, *cult/mind control ideology* | *conversion and 'brainwashing' in NRMs* (0.754) | **gap** |

Using the nearest-cluster assignments computed from the **true** cluster centroids (persisted before
the per-expression labels were discarded, so no reconstruction is involved):

**0 of 9 religious/NRM entities is nearest a coercion cluster.** Every one sits in its own
eponymous cluster instead:

| Entity | Nearest cluster | Cosine |
|---|---|---|
| heaven's gate | 4 | 0.985 |
| aum shinrikyo | 26 | 0.933 |
| peoples temple | 0 | 0.928 |
| solar temple | 7 | 0.925 |
| unification church | 27 | 0.915 |
| the family (formerly the children of god) | 30 | 0.910 |
| theosophical society | 34 | 0.889 |
| branch davidians | 1 | 0.883 |
| rajneesh movement | 20 | 0.832 |

**2 of 10 secular entities are** — and they are the two thought-reform cases:

| Entity | Nearest cluster | Cosine |
|---|---|---|
| **maoist thought reform** | **43** | 0.587 |
| **erhard seminar training** | **43** | 0.524 |
| communist party | 36 | 0.701 |
| landmark forum | 35 | 0.669 |
| nazism | 24 | 0.652 |
| chinese communist party | 36 | 0.651 |
| national socialism | 24 | 0.581 |
| lifespring | 35 | 0.579 |
| amway | 25 | 0.509 |
| werner erhard | 35 | 0.462 |

Cluster 43's other named residents are the coercion-and-therapy lexicon: mystical manipulation
(0.731), submissiveness (0.705), gestalt therapy (0.681), *Thought Reform and the Psychology of
Totalism* (0.642), Silva Mind Control (0.639), primal therapy (0.671), psychosynthesis (0.669).

**The reading.** The literature discusses religious cults as **named cases** — each generating its
own tight cluster of case-specific discussion — and discusses coercion as a **mechanism**, in
general psychological vocabulary whose concrete named exemplars are secular. Read alongside
[`05_Epistemic_Status_Centroids.md`](05_Epistemic_Status_Centroids.md)'s finding that brainwashing is
geometrically central *because it is contested rather than asserted*: the coercion claim is the
field's live argument, it is conducted in general terms rather than about particular religious
groups, and where it does attach to specific organizations those organizations are political
re-education programmes and commercial seminars.

**This supports the central Hypothesis more directly than the retracted claim did.** It is not that
the criteria fail to exclude ordinary organizations by accident of wording; it is that the field's
own analytical apparatus for coercion was built partly on secular cases in the first place.

## Graded view: group sets against coercion poles

The argmax view above answers "which cluster is nearest," but not "how close is each group set to
coercion vocabulary." For that, four poles, most defensible first. The primary two are the real
entity anchors `brainwashing` and `coercive persuasion` — actual points in the space, lowercase and
1–2 words (so name-form-matched to the filtered organization names, per `08`'s artifact), with known
cosine to the true cluster centroids (0.872 to 40, 0.789 to 43).

| Pole | Religious (n=9) | Ordinary orgs, name-matched (n=90) | Gap | AUC | Clears ±0.061? |
|---|---|---|---|---|---|
| `brainwashing` (entity anchor) | 0.413 | **0.479** | **+0.067** | 0.844 | **yes** |
| `coercive persuasion` (entity anchor) | 0.409 | **0.472** | **+0.063** | 0.773 | **yes** |
| `crit-mental-destabilization` (criterion text) | 0.272 | 0.289 | +0.018 | 0.626 | no |
| all-entity grand centroid (**register control**) | 0.608 | 0.654 | +0.046 | 0.740 | no |

**Ordinary organizations sit closer to the literature's coercion vocabulary than famous religious
cults do**, on both entity anchors, clearing the artifact band. Consistent with the cluster result.

**But the register control does most of the work, and this is the honest limit.** Ordinary
organizations are also closer to the *whole* entity vocabulary (+0.046). Subtracting that, the
coercion-*specific* excess is only **+0.021** (`brainwashing`) and **+0.017** (`coercive
persuasion`) — well inside the ±0.061 band. So the defensible statement is: *ordinary organizations
are not further from coercion vocabulary than religious cults are, and are somewhat closer, but most
of that proximity is a general register effect rather than coercion specifically.*

**Pole agreement gate** (Spearman over all 3,785 entities, the check on whether "subjection" is one
direction at all): the two entity anchors agree at **+0.792**, and with the cluster core proxies at
+0.67 to +0.89 — so it is broadly one direction. Two qualifications: the grand centroid correlates
**+0.815** with the `brainwashing` pole, confirming the entanglement with general register above;
and the official criterion text correlates only **+0.567 / +0.572** with the entity anchors. **The
MIVILUDES wording points somewhere measurably different from the literature's own coercion
vocabulary** — a finding in its own right, and a caution against treating the criterion text as
interchangeable with the concept.

### Robustness poles, and why they are only that

Cluster centroids cannot be recomputed: `analyze_literature_clusters.py` never persisted its
per-expression labels, and re-running UMAP+HDBSCAN would reassign the cluster IDs that `06` and
[`09`](09_Interview_Prototypes_vs_Clusters.md) depend on. The clustering was therefore deliberately
**not** re-run. Instead, "core proxies" were rebuilt by averaging each cluster's persisted binding
expressions — its members nearest the true centroid. Reported as robustness only, with fidelity
published rather than assumed:

| Cluster | Rows resolved | Distinct vectors | Members' cosine to true centroid |
|---|---|---|---|
| 40 | 8/8 | **2** | 0.921 |
| 43 | 8/8 | 6 | 0.764–0.839 |
| 48 | 8/8 | 7 | 0.869–0.882 |

**Cluster 40's proxy is effectively degenerate** — all eight binding expressions are the literal
token "brainwashing", so the "centroid" reduces to that one word's embedding. Two further limits:
binding expressions are the members *nearest* the centroid, so their mean is a dense **core** that
overstates similarity to anything lexically close while ignoring the periphery (cluster 40's internal
dispersion is 0.664 against a binding distance of 0.432); and 8 of 85 members is a 9% sample. The
proxies broadly agree with the entity anchors, which is why the anchors are trusted — not the
reverse.

**Recommended fix for future work:** persist the per-expression `labels` array in
`analyze_literature_clusters.py` so true cluster centroids are recoverable without re-running and
without renumbering. Not done here, because writing it would mean re-running.

## Result: does the space represent coercive control at all? (run, pre-registered)

**Headline, by the rule committed to in advance: null.** Mean B−C across the two coercion poles is
**−0.032 on names** and **+0.040 on descriptions**; neither clears the ±0.061 band, so
`interpret()` returns *"B ≈ C on both views — this space cannot represent coercive control from
these inputs."* That is the reportable conclusion of the pre-registered test.

**The structure underneath it is coherent, and matches the predicted pattern.** Reported as a
secondary observation, explicitly **not** the pre-registered result:

| Pole | Names B−C | Descriptions B−C | Descriptions AUC | Descriptions, register-corrected |
|---|---|---|---|---|
| `coercive persuasion` | −0.018 | **+0.062 (clears)** | **0.824** | **+0.094** |
| `brainwashing` | −0.046 | +0.018 | 0.640 | +0.050 |
| `crit-mental-destabilization` | +0.010 | +0.015 | 0.605 | +0.047 |
| all-entity centroid (register control) | −0.054 | −0.032 | 0.265 | — |

Three things are worth more than the null:

**1. The direction flips between views exactly as predicted.** Both coercion poles are *negative*
on names and *positive* on descriptions. Names carry no conduct information; descriptions do.

**2. The four-cell ordering on descriptions is the designed one.** On `coercive persuasion`:

| Cell | Mean cos |
|---|---|
| A famous cults (n=6) | **0.506** |
| B coercive (n=22) | **0.428** |
| D demanding (n=25) | **0.409** |
| C ordinary (n=23) | **0.366** |

Famous cults > coercive > demanding > ordinary, with **cell D sitting between B and C** — the
"demanding ≠ coercive" discrimination the design was built to test, working. On **names** there is
no ordering at all (A 0.423 ≈ C 0.423 ≈ D 0.422 > B 0.405): cell B is *last*. That is
[`10`](10_Secular_Groups_vs_Sectarian_Drifts.md)'s topical-overlap mechanism measured rather than
asserted.

**3. Why the pre-registered mean returned null anyway.** It averaged `coercive persuasion` with
`brainwashing`, and `brainwashing` was already flagged above as the weaker pole — a bare token
correlating **+0.815** with the all-entity centroid, i.e. mostly register. Averaging a strong pole
with a mostly-register one diluted the signal below the band. **This is stated as a lesson for the
follow-up's pre-registration, not as grounds for reinterpreting this run**: switching now to
"`coercive persuasion` only" would be exactly the post-hoc move the pre-registration exists to
prevent.

**Specificity: the descriptions signal is real but not about subjection.** B beats C on 13 of 17
criteria, so some general lift is present. The largest gaps are **deceptive recruitment (+0.051)**
and **difficulty leaving the group (+0.042)** — substantively apt for a cell of MLMs, fraternities
and sales bootcamps. But *violation of Republic principles* is third (+0.037), which is content
nonsense and confirms the general lift, and **mental destabilization is only 6th (+0.015)**. So
even where descriptions do carry signal, it concentrates on **recruitment and exit**, not
psychological subjection — consistent with Result 1's finding that the procedural criteria are
where ordinary organizations match most.

**Circularity control, and the hand review it required.** 24 of 100 curated names are already
corpus entities and were excluded. The documented 0.85-cosine residual rule flagged 6 rows for
human eyes and **found a real defect in the automatic tag**: it had stripped a leading "The" only
from the generated name, missing the cases where the *corpus* holds the decorated form —
`Family International` vs the corpus's *the family international* (0.870) and `Church Universal
Triumphant` vs *church universal and triumphant* (0.905). Both are the same organization and were
passing through as novel. Fixed with two-way normalization, verified not to merge genuinely
different organizations (`Revolutionary Communist Party` still does not collapse onto the corpus's
*communist workers party*). **The fix moved two entries out of cell A and left cells B and C
untouched, so the headline is unchanged** — which is what a working control should do. `Harvard`
was flagged and deliberately kept: it is in the corpus as a cited institution and publisher, not
as cult discourse, so the rule over-triggers there.

### What this licenses

The pre-registered null stands as the result. The secondary pattern — direction flip, correct
four-cell ordering, one pole clearing at AUC 0.824 — is strong enough to justify a **properly
powered, re-pre-registered follow-up**: more items per cell (22 vs 23 is thin), `coercive
persuasion`-type poles named in advance rather than averaged with a register-heavy one, and a
specificity requirement stated numerically. What it does **not** license is a claim that this space
measures coercive control; the honest position remains that **conduct descriptions carry signal
that names do not**, which was the useful thing to establish either way.

## Appendix: the original pre-registration

Preserved verbatim as written *before* the run, so the result reported above cannot be read post
hoc. The motivating question at the time: every comparison in this project compares *names* to
*vocabulary*, and `10`'s Mechanism section had established that this measures topical overlap
rather than conduct — the groups nearest "difficulty leaving the group" were a mountain-biking
group, an Arduino users' group and two sports clubs. So the open question was whether coercive
control is representable here **when the input actually has the quality**.

Four cells of 25 organizations, scored on the poles above as **bare names** and as **one-line
descriptions of the control dynamic**:

- **A** — groups already famous as cults. Circularity control; excluded from the headline.
- **B** — ordinary-*sounding* organizations credibly described as coercive, not commonly called
  cults (MLMs, fitness franchises, fraternities, sales bootcamps, talent academies, political sects).
- **C** — ordinary organizations with no coercion claim. Matched control.
- **D** — highly demanding but not coercive (military academies, monasteries, conservatories,
  surgical residencies). Separates "demanding" from "coercive".

Headline test is **B vs C**, on the coercion poles only, against the ±0.061 band. Pre-registered
readings:

| Outcome | Interpretation |
|---|---|
| B > C on descriptions but not names | The quality is legible in conduct language and invisible in names — the expected result, and it converts `10`'s limitation into a measurement |
| B > C on both | Names carry more than assumed |
| B ≈ C on both | The space cannot represent coercive control; this document's retraction stands unqualified |
| A > everything | Confirms only that famous cult names sit inside cult discourse |

Specificity check: B must **not** also beat C on the other 16 criteria. If it beats everything
uniformly, the result is register, not coercion — the same trap the retracted finding fell into.

Source list (hand-curated, with the construction rules and the rejected model attempt documented
in its own header): `thesis/corpus/dictionaries/coercive-control-groups/coercive-groups-curated.txt`.
The qwen3:8b generation it replaced returned cell B as 76% generic category labels ("Gym Chain",
"Corporate Training Program") against a cell C of 100% real proper nouns, which would have
guaranteed B > C for reasons unrelated to coercion.

Full data: [`data/criterion_length_control.csv`](data/criterion_length_control.csv),
[`data/poles_vs_group_sets.csv`](data/poles_vs_group_sets.csv),
[`data/pole_agreement.csv`](data/pole_agreement.csv),
[`data/pole_fidelity.csv`](data/pole_fidelity.csv),
[`data/entity_sets_nearest_cluster.csv`](data/entity_sets_nearest_cluster.csv),
[`data/cells_vs_poles.csv`](data/cells_vs_poles.csv),
[`data/cells_specificity_by_criterion.csv`](data/cells_specificity_by_criterion.csv),
[`data/corpus_membership_tags.csv`](data/corpus_membership_tags.csv).
