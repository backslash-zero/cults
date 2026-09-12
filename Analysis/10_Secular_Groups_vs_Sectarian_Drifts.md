# Non-religious organizations against the 17 MIVILUDES sectarian drifts, criterion by criterion

Data inventory: [`00_Available_Data.md`](00_Available_Data.md). Index: [`Results_Draft.md`](Results_Draft.md).
Aggregate version of this test: [`08_Secular_Groups_Hypothesis.md`](08_Secular_Groups_Hypothesis.md).

[`08`](08_Secular_Groups_Hypothesis.md) measured non-religious groups against the *pooled*
criteria centroid. But the thesis's central Hypothesis — *"the basis for assessment that certain
groups are cults would imply that other social structures are cults as well"* — is a claim about
the **basis**, i.e. the individual criteria. A pooled centroid can't answer it: it averages away
exactly the question of *which* drift does the implicating. This document takes the 169
non-religious organizations (C2, see `08`'s Method) against each of the 17 MIVILUDES
*dérives sectaires* separately, with the 9 religious/NRM baseline entities as control.

**This turns out to be the most informative and the most cautionary of the Hypothesis tests, in
that order.** Read the *Mechanism* section before quoting any number from here.

Code: `thesis_corpus/analyze_secular_groups_vs_criteria.py`.

## Method

Each of the 17 criteria is its own `miviludes_criteria` point (raw bge-m3, no PCA — same space as
every other analysis this session; criterion text is the official French wording). Each of the 169
organization names is its own bge-m3 vector. Reported per criterion: mean cosine similarity of
each group set to *that criterion's own vector*, never to a centroid.

**The name-form control is mandatory here, not optional.** C2's names and the corpus-native entity
anchors differ systematically in surface form — C2 is Title Case, averages 3.30 words, and 71/169
start with "The"; the corpus-native anchors are all lowercase, average 2.56 words, and almost never
start with "The" (they come from normalized anchor text via
`build_shared_space.load_emergent_entities`). That difference alone moves the metric more than any
finding below: see [`08`'s Mechanism section](08_Secular_Groups_Hypothesis.md#mechanism--what-this-metric-actually-measures)
for the "The" natural experiment (+0.061 for one semantically empty token, on the same
organization). So every comparison below uses **C2-matched** — the 90 of 169 names that are ≤3
words and lack a leading "The" — as the headline figure, with raw C2-all shown alongside for
transparency. Nothing is claimed on the basis of C2-all alone.

## Result 1 — no criterion discriminates religious from non-religious

**Read the `words` column before the similarity columns.** Criterion texts range from 3 to 29
French words, and length alone predicts the absolute-similarity column at **Spearman −0.613** — a
long criterion sentence is far from every short organization name regardless of content. The
**Diff** column is the length-robust one (Spearman with length just **−0.039**), because the length
penalty applies to the religious baseline and the ordinary organizations equally and cancels. Every
claim below rests on Diff, never on the absolute column. Sorted here by absolute similarity only to
make the confound visible.

| Sectarian drift (criterion key) | words | C2-matched (n=90) | Religious baseline (n=9) | Diff | C2-all (n=169) |
|---|---|---|---|---|---|
| radical behavior change | 5 | **0.404** | 0.373 | +0.031 | 0.417 |
| indoctrination of children | 3 | 0.377 | 0.372 | +0.005 | 0.387 |
| infiltration of public authorities | 6 | 0.369 | 0.347 | +0.022 | 0.390 |
| alarming dietary change | 6 | 0.351 | 0.320 | +0.031 | 0.367 |
| rupture with environment of origin | 7 | 0.350 | 0.335 | +0.015 | 0.359 |
| harsh living conditions | 8 | 0.339 | 0.306 | +0.033 | 0.361 |
| difficulty leaving the group | 13 | 0.337 | 0.327 | +0.011 | 0.338 |
| deceptive recruitment | 5 | 0.334 | 0.292 | **+0.042** | 0.344 |
| public order contestation | 17 | 0.324 | 0.304 | +0.020 | 0.350 |
| authoritarian, opaque group | 23 | 0.321 | 0.303 | +0.018 | 0.335 |
| violation of Republic principles | 8 | 0.320 | 0.314 | +0.007 | 0.346 |
| physical integrity harm | 28 | 0.314 | 0.285 | +0.029 | 0.335 |
| rejection of the outside world | 20 | 0.303 | 0.290 | +0.014 | 0.318 |
| exorbitant financial demands | 26 | 0.297 | 0.250 | **+0.048** | 0.314 |
| dubious exclusive care/medicine | 15 | 0.296 | 0.277 | +0.019 | 0.317 |
| legal disputes | 4 | 0.292 | 0.266 | +0.026 | 0.307 |
| mental destabilization | **29** | 0.289 | 0.272 | +0.018 | 0.314 |

**The diff column is positive for all 17.** On no single sectarian drift does the religious/NRM
baseline sit closer to the criterion than ordinary non-religious organizations do — and this holds
in the name-form-matched subset, so it isn't the surface-form artifact. That is a considerably
stronger statement than `08`'s aggregate result, which only established rough parity on the pooled
centroid. There is no subset of the 17 that picks out religion.

Reproduced with the length covariate by
`thesis_corpus/analyze_psychological_subjection.py`'s `criterion_length_control()`; full table in
[`data/criterion_length_control.csv`](data/criterion_length_control.csv).

The two widest margins are **exorbitant financial demands** (+0.048) and **deceptive recruitment**
(+0.042) — the two most *procedural/material* drifts on the list. That is what the Hypothesis
predicts if it is right: exploitative finance and misleading recruitment are generic
organizational pathologies with no religious content whatsoever. The two narrowest margins are
**indoctrination of children** (+0.005) and **violation of Republic principles** (+0.007) — the
former because C2 is full of youth/education organizations (a list-composition effect, not a
finding), the latter because it is a French-administrative category with barely any English-language
purchase at all, consistent with its bottom-ranked coverage in
[`03_Per_Criterion_Neighbors.md`](03_Per_Criterion_Neighbors.md).

## Result 2 — RETRACTED: "one criterion resists generalization"

**The first version of this document claimed** that *mental destabilization* ranks 17th of 17 for
non-religious organizations and is nearest-drift for only 2 of 169, and concluded that "the basis
for assessment generalizes almost everywhere except at its own conceptual core" — that psychological
subjection alone carries the framework's discriminating power. **That conclusion does not survive
the length control and is withdrawn.** Kept visible rather than deleted, same as `08`'s retraction,
because the failure mode is reusable knowledge.

**Why it fails.** *Mental destabilization* is the **longest of the 17 criteria** — 29 French words,
against a shortest of 3 — and criterion length predicts absolute similarity at **Spearman −0.613**.
Its rank-17 position is a property of its sentence, not of ordinary organizations. The
"nearest-drift for only 2 of 169" figure is argmax over the same confound: a 29-word criterion
almost never wins a nearest-neighbour contest against a 3-word one.

**What the length-robust measure says instead.** On Diff — the measure that cancels length —
*mental destabilization* sits at **+0.018, rank 6 of 17**: mid-pack, unremarkable. The criteria that
genuinely discriminate least are:

| Sectarian drift | Diff | words |
|---|---|---|
| indoctrination of children | +0.005 | 3 |
| violation of Republic principles | +0.007 | 8 |
| difficulty leaving the group | +0.011 | 13 |
| rejection of the outside world | +0.014 | 20 |
| rupture with environment of origin | +0.016 | 7 |

Set aside *violation of Republic principles* (a French-administrative category with no
English-language purchase — see [`03`](03_Per_Criterion_Neighbors.md)) and the remaining four are
one coherent family: **enclosure**. Indoctrinating the children, cutting the member off from family,
refusing the outside world, making exit impossible. **Where the religious baseline comes closest to
parity with ordinary organizations is the criteria about sealing a group off from the world** — not
the criteria about psychological harm, and not the procedural ones.

That is a genuine finding with the same shape as the retracted one but the opposite content, and it
is length-robust. It is also weaker in magnitude: all five gaps are far inside the ±0.061 artifact
band, so read it as an ordering, not as five measured effects.

## Result 2b — where the literature's own coercion discourse actually sits

The clusters answer this better than the criteria do. Three of the 59 literature clusters
([`06`](06_Literature_Clusters.md)) are about psychological subjection — **40** (n=85, binding
expressions are the bare token *brainwashing*), **43** (n=59, *mental manipulation* / *coercive
persuasion*), **48** (n=39, flagged an entity gap). Using the nearest-cluster assignments computed
from the **true** cluster centroids:

- **0 of 9 religious/NRM entities** are nearest a coercion cluster. Every one sits in its own
  eponymous cluster instead — Heaven's Gate→4 (0.985), Aum Shinrikyo→26 (0.933), Peoples
  Temple→0 (0.928), Solar Temple→7 (0.925), Unification Church→27 (0.915).
- **2 of 10 secular entities are**: *Maoist thought reform*→43 (0.587) and *Erhard Seminar
  Training*→43 (0.524). Cluster 43's other residents are the coercive-control and therapy lexicon —
  mystical manipulation (0.731), submissiveness (0.705), gestalt therapy (0.681), Silva Mind Control
  (0.639), *Thought Reform and the Psychology of Totalism* (0.642).

**The mechanism vocabulary attaches to secular coercion programmes; religious cults are discussed
as named cases rather than as mechanisms.** This supports the Hypothesis more directly than the
retracted claim did — and it needed no new data, only the argmax view that was already on disk.
Full treatment in [`11_Psychological_Subjection.md`](11_Psychological_Subjection.md).

## Result 3 — a partial replication, weaker than first stated

**Radical behavior change** is rank-1 in *absolute* mean similarity here, and the nearest drift for
**85 of 169 groups (50%)**. [`07_Cult_Prototype.md`](07_Cult_Prototype.md) independently found it
robustly rank-1 against the corpus-wide prototype in both weightings — and `07`'s ranking has since
been **checked for the same length confound and is clean** (Spearman between criterion length and
prototype cosine is −0.150 pooled, +0.093 equal-weighted). So `07` stands unchanged.

**But this document's half of the agreement is the confounded half.** *Radical behavior change* is
5 words, among the shortest of the 17, so its rank-1 absolute position and its 85/169 argmax
concentration are both partly length effects — a short criterion is close to short names and wins
nearest-neighbour contests easily. On the length-robust Diff measure it is **5th of 17** (+0.031),
behind exorbitant financial demands, deceptive recruitment, harsh living conditions and alarming
dietary change. Treat this as *consistent with* `07` rather than as an independent replication of it.

Why `07` escapes the confound and this document does not: `07` compares each criterion to a **pooled
centroid of thousands of corpus expressions**, a dense average that no single criterion's length
systematically penalises. This document compares each criterion to **short organization names**,
where the mismatch between a 29-word sentence and a 2-word name drives cosine down directly. The
lesson generalises — length confounds appear when the two sides of a comparison differ
systematically in length, which is also exactly why `08`'s name-form control was necessary.

| Nearest drift | Groups (of 169) |
|---|---|
| radical behavior change | 85 |
| indoctrination of children | 29 |
| infiltration of public authorities | 26 |
| alarming dietary change | 10 |
| difficulty leaving the group | 4 |
| violation of Republic principles / physical integrity harm / public order contestation | 3 each |
| rupture with environment of origin / mental destabilization / harsh living conditions | 2 each |
| *(6 criteria are nearest for no group at all)* | 0 |

## Mechanism — what is actually driving these numbers

Inspecting which organizations land on which drift makes the mechanism unmistakable, and it is
**topical keyword overlap between the organization's name and the criterion's wording** — not any
assessment of organizational conduct:

| Sectarian drift | Its nearest non-religious organizations | Shared vocabulary |
|---|---|---|
| difficulty leaving **the group** | Mountain Biking **Groups**, Arduino User **Group**, Triathlon **Clubs**, Soccer Fans **Clubs** (all 4) | "groupe" / group, club |
| indoctrination of **children** | Free the **Children** (0.570, highest of all), **UNICEF**, Buddy Bench, National Education Association | children, youth, education |
| alarming **dietary** change | The **Hunger** Project, Sustainable **Food** Trust, American **Diabetes** Association, **WHO**, **NIH** | food, hunger, nutrition, health |
| **mental destabilization** | National Institute of **Mental Health**, **Mental Health** America (only 2) | mental health |
| **infiltration of public authorities** | Center for American **Progress**, **Open Society** Foundations, Brennan Center for **Justice** | policy, public, society, justice |

The four organizations whose nearest drift is *"great difficulty, even impossibility, for a member
to leave said group"* are a mountain-biking group, an Arduino users' group, triathlon clubs and
soccer fan clubs — matched on the word **group/club**, not on exit costs. The highest single
criterion-to-organization score in the entire run is **Free the Children at 0.570 against
"l'embrigadement des enfants" (indoctrination of children)** — a children's charity, matched on the
word *children*. A sustainable-food charity is the best match for cult dietary control; the
national mental-health institute is the best match for psychological subjection.

**Consequence, stated plainly: cosine similarity between an organization's *name* and a criterion's
*text* cannot test whether that organization meets that criterion.** The 17 drifts are behavioural
predicates about what an organization *does*; a name carries almost no information about conduct.
What the geometry recovers is subject matter. Every number in Results 1–3 must be read as a
statement about *vocabulary*, not about groups.

## So what does this license?

Carefully separated, because the strong version is methodological and the substantive version is
weaker than it looks:

**Robust (methodological).** Embedding-based operationalization of criterion satisfaction fails in
a specific, diagnosable way: it collapses to topical overlap, and it is sensitive to semantically
empty surface form (a leading "The" is worth +0.061, larger than any effect in Result 1's diff
column). Any future use of this space to score entities against criteria needs behavioural text
about each entity — a description, a corpus of coverage — not its name. This applies to
[`08`](08_Secular_Groups_Hypothesis.md)'s aggregate result equally, and is why that document now
retracts its prototype-centroid claim.

**Defensible, hedged (substantive).** The criteria's *wording*, taken at face value, is topically
generic enough that ordinary civil-society organizations match all 17 at least as well as known
NRMs do. That is not a demonstration that mountain-biking clubs are cults; it is a demonstration
that **the language in which the drifts are written does not, on its own, exclude benign
organizations** — the drifts are phrased in ordinary sociological vocabulary (group, children,
food, money, recruitment, leaving) that ordinary organizations inhabit. This is a real argument in
the Hypothesis's direction, but it is an argument about the criteria's *discriminating language*,
one register removed from the Hypothesis's own claim about assessment practice. MIVILUDES
assessors read case files, not names; the analogy should be stated with that limit attached.

**Genuinely surprising, and the best candidate for the thesis.** Not the criterion ranking — that
was the retracted claim — but *where the literature's coercion vocabulary lives* (Result 2b). The
brainwashing and mental-manipulation clusters have **no religious/NRM entity** as their nearest
neighbour; their named residents are secular thought-reform and seminar programmes. The literature
discusses religious cults as **cases** and secular coercion as **mechanism**. Combined with
[`05_Epistemic_Status_Centroids.md`](05_Epistemic_Status_Centroids.md)'s finding that brainwashing
is central *because it is contested rather than asserted*, the picture is: the coercion claim is the
field's live argument, it is conducted in general psychological vocabulary rather than about
particular religious groups, and its concrete named exemplars are secular. That is a claim about
how the concept is *built*, and it does not depend on any criterion's sentence length.

## Caveats

- **Criterion text length is a confound on every absolute number in this document** (Spearman
  −0.613), because a 3-to-29-word range is being compared against 1-to-3-word organization names.
  The Diff column cancels it (−0.039) and is the only measure any claim here rests on. This is the
  same family as `08`'s +0.061 "The" artifact: surface form moving the metric further than content
  does. It already cost this document one retracted result (Result 2) and one downgraded one
  (Result 3).
- **Cross-lingual comparison.** The criterion texts are official French; the organization names are
  almost entirely English. bge-m3 is multilingual and this session's other analyses rely on that,
  but cross-lingual cosine values are not directly comparable in magnitude to same-language ones —
  the absolute levels here (0.29–0.40) run lower than the criteria-*centroid* similarities in
  [`08`](08_Secular_Groups_Hypothesis.md) (0.44–0.50) partly for this reason.
- **The religious baseline is n=9**, hand-picked. Every diff in Result 1 is computed against it, and
  none is large in absolute terms (+0.005 to +0.048, against a name-form artifact of +0.061). Result
  1's claim is therefore "no criterion picks out religion," not "secular organizations score
  meaningfully higher."
- **C2's composition drives several rankings** — it over-represents education/youth, advocacy, and
  health organizations (see [`08`'s Caveats](08_Secular_Groups_Hypothesis.md#caveats)), which is
  directly responsible for indoctrination-of-children ranking 2nd and infiltration-of-public-authorities
  ranking 3rd. Those two rows are list-composition effects and should not be quoted as findings.
- **Six criteria are nearest for no group at all**, so the nearest-drift distribution is thin in the
  tail; the 85/169 concentration on radical behavior change also means that column is dominated by
  one criterion's general availability.

Full data: [`data/criteria_vs_secular_groups.csv`](data/criteria_vs_secular_groups.csv) (17 rows,
all four group sets), [`data/secular_groups_nearest_criterion.csv`](data/secular_groups_nearest_criterion.csv)
(169 rows, with the name-form flag per row),
[`data/name_form_artifact_check.csv`](data/name_form_artifact_check.csv) (the "The X" vs "X" pairs).
