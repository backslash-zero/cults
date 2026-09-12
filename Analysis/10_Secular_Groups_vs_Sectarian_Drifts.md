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

| Sectarian drift (criterion key) | C2-matched (n=90) | Religious baseline (n=9) | Diff | C2-all (n=169) |
|---|---|---|---|---|
| radical behavior change | **0.404** | 0.373 | +0.031 | 0.417 |
| indoctrination of children | 0.377 | 0.372 | +0.005 | 0.387 |
| infiltration of public authorities | 0.369 | 0.347 | +0.022 | 0.390 |
| alarming dietary change | 0.351 | 0.320 | +0.031 | 0.367 |
| rupture with environment of origin | 0.350 | 0.335 | +0.015 | 0.359 |
| harsh living conditions | 0.339 | 0.306 | +0.033 | 0.361 |
| difficulty leaving the group | 0.337 | 0.327 | +0.011 | 0.338 |
| deceptive recruitment | 0.334 | 0.292 | **+0.042** | 0.344 |
| public order contestation | 0.324 | 0.304 | +0.020 | 0.350 |
| authoritarian, opaque group | 0.321 | 0.303 | +0.018 | 0.335 |
| violation of Republic principles | 0.320 | 0.314 | +0.007 | 0.346 |
| physical integrity harm | 0.314 | 0.285 | +0.029 | 0.335 |
| rejection of the outside world | 0.303 | 0.290 | +0.014 | 0.318 |
| exorbitant financial demands | 0.297 | 0.250 | **+0.048** | 0.314 |
| dubious exclusive care/medicine | 0.296 | 0.277 | +0.019 | 0.317 |
| legal disputes | 0.292 | 0.266 | +0.026 | 0.307 |
| mental destabilization | 0.289 | 0.272 | +0.018 | 0.314 |

**The diff column is positive for all 17.** On no single sectarian drift does the religious/NRM
baseline sit closer to the criterion than ordinary non-religious organizations do — and this holds
in the name-form-matched subset, so it isn't the surface-form artifact. That is a considerably
stronger statement than `08`'s aggregate result, which only established rough parity on the pooled
centroid. There is no subset of the 17 that picks out religion.

The two widest margins are **exorbitant financial demands** (+0.048) and **deceptive recruitment**
(+0.042) — the two most *procedural/material* drifts on the list. That is what the Hypothesis
predicts if it is right: exploitative finance and misleading recruitment are generic
organizational pathologies with no religious content whatsoever. The two narrowest margins are
**indoctrination of children** (+0.005) and **violation of Republic principles** (+0.007) — the
former because C2 is full of youth/education organizations (a list-composition effect, not a
finding), the latter because it is a French-administrative category with barely any English-language
purchase at all, consistent with its bottom-ranked coverage in
[`03_Per_Criterion_Neighbors.md`](03_Per_Criterion_Neighbors.md).

## Result 2 — one criterion resists generalization: the concept's own anchor

**Mental destabilization ranks 17th of 17** for non-religious organizations (0.289), and is the
single nearest drift for only **2 of 169** groups. Set against the rest of this session:

- It is the **criteria list's own most central member** ([`02_Corpus_Centroids.md`](02_Corpus_Centroids.md)) — the
  conceptual anchor of the official grid.
- It is **mid-table (7th–12th)** against the corpus's emergent prototype ([`07_Cult_Prototype.md`](07_Cult_Prototype.md)).
- It is **dead last** here, against non-religious organizations.

So the criterion that most defines the concept internally is the one that generalizes least. The
Hypothesis, read through this table, gets a sharper answer than yes-or-no: **the basis for
assessment generalizes almost everywhere except at its own conceptual core.** Whatever is
distinctively "cult" about the MIVILUDES grid is concentrated in psychological subjection — and
that is precisely the part ordinary organizations do not resemble.

## Result 3 — a replication, arriving from outside the corpus

**Radical behavior change** is rank-1 in mean similarity here *and* the nearest drift for **85 of
169 groups (50%)**. [`07_Cult_Prototype.md`](07_Cult_Prototype.md) independently found it robustly
rank-1 against the corpus-wide prototype, in both weightings. Those two results come from
completely different data — 07 from pooled corpus expressions, this from externally-sourced
organization names that never touched the corpus — so the agreement is a genuine replication rather
than the same measurement twice. Radical behavior change is the most semantically *available*
criterion in this vocabulary, whatever is being described.

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

**Genuinely surprising, and the best candidate for the thesis.** The criterion that anchors the
concept (mental destabilization) is the one that most resists generalization, while the criteria
that generalize most readily are the procedural ones (money, recruitment). Combined with
[`05_Epistemic_Status_Centroids.md`](05_Epistemic_Status_Centroids.md)'s finding that brainwashing
is central *because it is contested rather than asserted*, a consistent picture emerges: the
cult concept's centre of gravity is psychological subjection, which is simultaneously (a) the most
definitionally load-bearing, (b) the most contested in the literature, and (c) the least
transferable to non-religious structures. The concept is anchored precisely where it is least
stable and least general.

## Caveats

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
