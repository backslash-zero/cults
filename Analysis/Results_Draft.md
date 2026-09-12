# Results — working draft (index)

Running log of the geometric/embedding analysis, built up step by step — the staging ground
for the thesis Results section. This file is the index: a short summary of each step with a
link to its own file. Individual step files hold the actual writeup, tables, and figures.
Nothing is pruned automatically — entries stay unless explicitly removed. Figures live in
`figures/`, referenced by relative path from whichever file embeds them; supporting data
tables live in `data/`.

Data inventory: [`00_Available_Data.md`](00_Available_Data.md) — what's available across
MIVILUDES, scholar literature, and interviews, with subset breakdowns.

Plain-language synthesis: [`Plain_Language_Summary.md`](Plain_Language_Summary.md) — everything
below in non-technical words, what it adds up to, and concrete next directions. Start there for
the overall argument; come back here for the numbers.

## Steps

1. [`01_Shared_Space_v3.md`](01_Shared_Space_v3.md) — the pooled, standardized+PCA'd
   cross-corpus embedding space (literature + MIVILUDES v2 + the new exhaustive interview
   archive). Point counts and available subsets. Kept for reference; superseded as the basis
   for centroid/nearest-neighbour work by raw-embedding analysis (see step 2).
2. [`02_Corpus_Centroids.md`](02_Corpus_Centroids.md) — per-corpus (and per-corpus-entity)
   centroids + nearest neighbours, computed on raw bge-m3 embeddings (no PCA) with UMAP-2D
   visualizations; plus a per-criterion (17 MIVILUDES criteria) summary table.
3. [`03_Per_Criterion_Neighbors.md`](03_Per_Criterion_Neighbors.md) — full listing: for each
   of the 17 criteria, its 5 nearest entities + 10 nearest literature expressions + 10 nearest
   interview segments.
4. [`04_Projection_Techniques_Comparison.md`](04_Projection_Techniques_Comparison.md) — why
   some centroid plots in step 2 still show the centroid apart from its nearest neighbours:
   tested 5 different 2-D projection techniques (PCA, MDS, UMAP ×2, t-SNE) on the same data:
   all 5 agree, so it's a real property of the group's geometry, not a projection artifact.
5. [`05_Epistemic_Status_Centroids.md`](05_Epistemic_Status_Centroids.md) — step 2's method,
   split one level further by each expression's `epistemic_status`
   (asserted/contested/speculative/negated). Confirms that "brainwashing"'s centrality in the
   literature corpus comes disproportionately from the tiny *contested* subgroup, not from
   broad assertion.
6. [`06_Literature_Clusters.md`](06_Literature_Clusters.md) — HDBSCAN clustering of the 5,741
   literature expressions (12-D UMAP pre-reduction for cluster assignment only, raw 1024-D for
   every actual distance) into 59 concept clusters, each with a real-sentence "binding
   expression" and its nearest named entity, plus an overview map (2-D UMAP, gap clusters
   highlighted). 15 clusters flagged as entity gaps, each detailed in full (8 binding
   expressions + 10 nearest entities) and sorted into 6 types — most are genuinely conceptual
   (abstract/theoretical/statistical/attitudinal discourse, or book titles, none tied to a
   named group), a third are narrower entity-extraction near-misses.

7. [`07_Cult_Prototype.md`](07_Cult_Prototype.md) — an emergent "cult prototype" pooled from
   every cult-relevant expression across all three corpora (not the 17 criteria's own
   centroid), with the 17 criteria ranked by centrality to it, reported both plain-pooled and
   equal-corpus-weighted. *Radical behavior change* is robustly rank-1 in both weightings;
   *legal disputes* is robustly last. *Mental destabilization* — the criteria list's own most
   central member — is only mid-table here, a real divergence between the two notions of
   "central."
8. [`08_Secular_Groups_Hypothesis.md`](08_Secular_Groups_Hypothesis.md) — the first direct test
   of the thesis's central Hypothesis, in two rounds. C1 (n=10, hand-picked): secular structures
   (Amway, Nazism, Maoist thought reform, LGAT seminar programs) score *as well as or better
   than* a religious/NRM baseline against the 17 official criteria, but *worse* against the
   corpus's own emergent usage prototype. C2 (n=169, a broad human-reviewed list) confirms the
   criteria-list half and **exposes a metric artifact that forced a retraction**: a leading
   "The" on the same organization is worth +0.061 cosine — more than the effect being
   measured — so C2's apparent overtaking of the baseline on the prototype centroid was name
   formatting, not substance. What survives: sports/tech/commercial organizations are the
   *farthest* of all 169 from the cult concept (NFL last), while anti-cult watchdogs and
   mental-health institutions sit near the top — the centroid tracks participation in cult
   *discourse*, not cult-likeness.
9. [`09_Interview_Prototypes_vs_Clusters.md`](09_Interview_Prototypes_vs_Clusters.md) — do
   people's spontaneous prototype exemplars (not just any interview segment) land in literature's
   well-covered regions or its gaps? 13 of 25 (52%) land in a gap cluster, roughly double the
   corpus-wide base rate (25.4%) — the theoretically-privileged first-association answer skews
   generic even more than ordinary interview speech does overall.
10. [`10_Secular_Groups_vs_Sectarian_Drifts.md`](10_Secular_Groups_vs_Sectarian_Drifts.md) — the
    same 169 organizations against each of the 17 MIVILUDES *dérives sectaires* separately, since
    the Hypothesis is a claim about the individual criteria, not a pooled centroid. **No criterion
    discriminates**: all 17 sit at least as close to ordinary non-religious organizations as to
    the religious baseline, name-form-controlled. *Mental destabilization* — the criteria list's
    own conceptual anchor — ranks last of 17, so the basis for assessment generalizes everywhere
    except at its core. But the mechanism is topical keyword overlap, not conduct: the four groups
    nearest *"difficulty leaving the group"* are a mountain-biking group, an Arduino users' group
    and two sports clubs, and the single highest criterion match in the run is **Free the Children
    against "indoctrination of children" (0.570)**. Reported as a methodological limit first and
    a substantive result second.

## Planned next (not yet implemented)

**Voronoi visualization** over the 59 literature clusters from step 6, seeded by their
centroids — same pattern `generate_voronoi_projections.py` already uses for entity clusters.
Held for after the numeric clustering results (step 6) are reviewed.

**Title-Case control re-embed** (steps 8, 10) — the one name-form confound that could not be
corrected on this machine. Every C2 name is Title Case; every corpus-native anchor is lowercase.
Re-embedding the 9 religious baseline entities in Title Case (`The Peoples Temple` etc.) on the
Windows/Ollama machine would quantify that residual the way the "The" pairs already quantified
the article effect. Until then, all cross-set magnitudes in 08/10 carry an unmeasured component.

**Behavioural-text scoring instead of names** (step 10) — the deeper fix. Scoring an organization
against a criterion needs text about what it *does*; names only support topical matching. Either
embed the LLM's own one-line descriptions (already present, currently stripped during parsing) or
pull real coverage per organization.

**A religiously-neutral subset of C2** (step 8) — the current 169-name C2 list
over-represents secularist/humanist-advocacy, education and health organizations (flagged in
`08`/`10`'s Caveats, and directly responsible for two of step 10's criterion rankings); a
narrower rerun restricted to non-ideological groups (MLMs, sports, hobby, tech, wellness) would
isolate the Hypothesis test from that confound.

---

## For publication

Tracks which pieces of this working draft are earmarked for the actual thesis Results section,
as distinct from exploratory/working steps kept here for process transparency. Marked here as
decided, not automatically from the steps list above.

| File | Status | Notes |
|---|---|---|
| [`02_Corpus_Centroids.md`](02_Corpus_Centroids.md) | **To publish** | Per-corpus (and per-corpus-entity) centroids + nearest neighbours, with figures. |
| [`03_Per_Criterion_Neighbors.md`](03_Per_Criterion_Neighbors.md) | **To publish** | Full per-criterion (17 MIVILUDES criteria, FR+EN) nearest-neighbour listing. |
| [`05_Epistemic_Status_Centroids.md`](05_Epistemic_Status_Centroids.md) | **To publish** | Split out of 02 (already marked to publish); the confirmed "brainwashing is central because it's contested, not asserted" finding. |
| [`06_Literature_Clusters.md`](06_Literature_Clusters.md) | not yet decided | 59 literature concept clusters + entity-gap analysis; strong candidate once the Voronoi visualization (planned next) is added. |
| [`07_Cult_Prototype.md`](07_Cult_Prototype.md) | **To publish** | Directly answers "whether criteria form recognisable clusters or prototypes" — a core Research Goal item. |
| [`08_Secular_Groups_Hypothesis.md`](08_Secular_Groups_Hypothesis.md) | **To publish** | First direct test of the central Hypothesis; C1+C2 both in. Contains a retracted claim (kept visible) and the name-form artifact that forced it — publish the criteria-list result, not the prototype one. |
| [`10_Secular_Groups_vs_Sectarian_Drifts.md`](10_Secular_Groups_vs_Sectarian_Drifts.md) | **To publish** | Criterion-level Hypothesis test: no drift discriminates religious from non-religious, and mental destabilization (the concept's anchor) generalizes least. Publish with its Mechanism section — the topical-overlap limit is inseparable from the result. |
| [`09_Interview_Prototypes_vs_Clusters.md`](09_Interview_Prototypes_vs_Clusters.md) | **To publish** | Sharpens the folk-vs-expert-concept finding using the thesis's own prototype-theory-motivated exemplar layer. |
| [`01_Shared_Space_v3.md`](01_Shared_Space_v3.md) | not yet decided | Reference/methodology; superseded as the analysis basis by step 2. |
| [`04_Projection_Techniques_Comparison.md`](04_Projection_Techniques_Comparison.md) | not yet decided | Methodological justification (why the step-2 plots are trustworthy); may belong in a methods/appendix section rather than Results. |

---

## Key findings so far

*The next two bullets answer two of the thesis's own stated research questions ("Whether
sources converge on recurring criteria," "Whether different sources organise them
differently") directly — restated here succinctly, with the why, rather than left implicit in
individual step files.*

- **Sources converge strongly on some criteria, barely at all on others** — not a uniform
  agreement or a uniform disagreement (`03_Per_Criterion_Neighbors.md`'s coverage-similarity
  ranking). Convergence is highest for *radical behavior change*, *authoritarian/opaque group*,
  and *mental destabilization* — criteria with a long psychological/sociological research
  tradition behind them. It's lowest for *legal disputes* and *violation of Republic
  principles* — dead last across literature, entities, *and* interviews alike. **Why**: the
  English-language scholarly literature this corpus draws on simply has no real vocabulary for
  French administrative/constitutional categories — convergence tracks whether a shared
  descriptive tradition exists for a criterion, not whether the criterion is any less real.

- **Different sources organise the same underlying concept around entirely different
  registers**, not just different emphases (`02_Corpus_Centroids.md`/`05_Epistemic_Status_Centroids.md`).
  MIVILUDES centers on its own institutional self-reference; scholarly literature centers on
  definitional/category debate (arguing over what to call things, not describing specific
  groups); interviews center on generic, low-content speech and, where substantive, on
  established world religions rather than the historical NRM case studies literature favors.
  **Why**: each source is structurally answering a different question — an administrative
  agency logging complaints, scholars debating category boundaries, laypeople free-associating
  — so the same word organises around whatever each source's own institutional purpose is,
  not around one shared underlying structure.

- **Each corpus's centroid ("center of gravity") reflects a different register**, not
  interchangeable content (step 2): `rapport` (MIVILUDES report) centers on its own
  institutional vocabulary, repeating its core term ("sectarian deviations"); the structured
  17-criterion list centers on **mental destabilization** — the anchor criterion the other 16
  elaborate on; `literature` centers on plain definitional statements ("Is this a religion, a
  cult, an NRM?"); `interviews` centers on generic, content-free speech ("This comes to my
  mind") — a useful sanity check that the exhaustive pipeline really is embedding everything,
  not secretly pre-filtered to cult-relevant content.

- **The corpora diverge sharply in which named entities sit closest to their centroid.**
  Literature's entity-centroid sits in identity/psychology vocabulary (self, the self, faith,
  society); MIVILUDES's sits on institutional/technological concern (internet, CNRS,
  constitution, RGPD). Interviews' entity-centroid, distinctively, sits on **established world
  religions** (Christianity, Judaism, Islam, Mormonism) rather than the specific historical
  NRMs (Heaven's Gate, Jonestown, Branch Davidians, Solar Temple) that dominate literature's
  own entity list. Read as a discourse pattern, not a claim about any of those religions: lay
  association with the prompt "cult" pulls toward major established religions far more than
  toward the niche case studies the academic literature centers on — a folk-concept vs.
  expert-concept gap worth developing further.

- **The corpus's conceptual vocabulary isn't evenly rich across the 17 official criteria —
  quantified, not just eyeballed** (step 3, full table + detail in
  [`03_Per_Criterion_Neighbors.md`](03_Per_Criterion_Neighbors.md#coverage-gap-analysis-which-criteria-are-well--vs-poorly-covered-and-where)).
  Mean cosine similarity of each criterion's nearest neighbours, per pool (entities/literature/
  interviews), gives a direct "how close is even the *closest* material this corpus has"
  number. Best covered: **radical behavior change**, **authoritarian/opaque group**,
  **difficulty leaving the group**, **mental destabilization** (all ≥0.65 in literature) — the
  criteria with the longest psychological/sociological research tradition (thought reform,
  charismatic authority, self-transformation). Worst covered, and consistently so across *all
  three* pools (not a literature-specific artifact): **legal disputes** is dead last in every
  single pool (literature 0.493, entities 0.447, interviews 0.418); **violation of Republic
  principles** is second-worst overall and worst-in-interviews specifically (0.451) — both are
  administrative/legal/French-constitutional criteria with no close counterpart in
  English-language NRM scholarship or ordinary interview speech. A real coverage gap between
  French regulatory categories and this corpus's own vocabulary, not a flaw in the embedding.

- **"Brainwashing" is geometrically central because it's contested, not because it's
  accepted.** The embedding space only reflects **linguistic/discursive centrality** — how
  much a term co-occurs with related vocabulary, how often it's discussed at all — never
  whether the corpus affirms, disputes, or rejects it. A heavily-debated, heavily-refuted term
  can sit just as "central" as a widely-accepted one, since raw geometric position doesn't
  distinguish affirmation from critique. **Checked, not just argued** (step 5, full writeup in
  [`05_Epistemic_Status_Centroids.md`](05_Epistemic_Status_Centroids.md)): splitting each
  corpus's centroid by `epistemic_status` shows literature's own *contested* subgroup (n=14,
  0.2% of the corpus) has "brainwashing" as its own rank-1 **and** rank-2 nearest neighbour,
  surrounded by expressions explicitly about it being a live scholarly debate — while the
  *asserted* subgroup (99.4% of the corpus) centers on plain definitional statements instead.
  Worth keeping in mind for every finding in this draft that reads a centroid or
  nearest-neighbour list as "what the corpus is about": it's "what the corpus talks about,"
  which includes what it argues against.

- **The literature's abstract/theoretical discourse has no nearby named entity — its
  concrete, group-specific discourse does** (step 6, full writeup + all 59 clusters in
  [`06_Literature_Clusters.md`](06_Literature_Clusters.md)). Clustering the 5,741 literature
  expressions gives 59 concepts; most are tight, single-group clusters whose nearest entity is
  almost a perfect match (Heaven's Gate 0.985 cosine similarity to its own cluster; Aum
  Shinrikyo 0.933; The Church of Scientology 0.959). The 15 flagged as **entity gaps**
  (25%, no named entity close by) are overwhelmingly abstract, definitional, or attitudinal:
  "NRMs are religions," "It is not just religious but a religion," "We don't approach/go
  towards the groups, we do not have the time," "There are around 800,000 followers in
  total..." — category-level or evaluative claims rather than discussion of any specific
  group. Consistent with (and now more granular than) the earlier finding that literature's
  own overall centroid sits on definitional statements: this shows concretely *why* — the
  corpus's theoretical vocabulary and its named-group vocabulary occupy different regions of
  the space, and definitional discourse is specifically the part with no concrete referent
  nearby. Two flagged clusters (Rissho Kōseikai/Soka Gakkai; the Family Federation for World
  Peace and Unification) are better read as entity-extraction near-misses than genuine
  conceptual gaps — worth a manual check before treating them as substantive.

- **The literature's single biggest recurring activity is arguing about terminology itself**
  (step 6). Set the named-group clusters aside, and the two largest remaining clusters are
  literally about the terms "new religious movement" (n=143) and "cult" (n=117) — genuine prose
  discourse, not index entries, spread across 6+ different source books rather than dominated
  by any one. Consistent with (and sharpening) the earlier finding that literature's overall
  centroid sits on definitional statements: the single largest thing this corpus does, by
  volume, is debate what to call the thing it's studying.

- **A sanity check on the whole embedding pipeline, in plain terms**: when the analysis groups
  literature sentences by topic and then separately asks "which named group is this closest
  to," it gets almost exactly the right answer, unprompted — the cluster of sentences about
  Heaven's Gate lands right next to the "Heaven's Gate" entity (98.5% similarity, about as
  close as two independently-computed things can get); same for Aum Shinrikyo (93.3%) and the
  Church of Scientology (95.9%). Nobody told the system which sentences were about which group —
  it figured that out purely from the text itself. That's a good sign the underlying model is
  actually understanding the content, not just producing plausible-looking noise — which
  matters because most of the *other* findings in this draft (which entities are central, which
  corpora diverge, which criteria are poorly covered) all rest on that same model being
  trustworthy. See [`06_Literature_Clusters.md`](06_Literature_Clusters.md) for the full
  writeup.

- **The corpus's own emergent "cult prototype" prioritizes different criteria than the
  official list's own internal structure does** (step 7). Pooling every cult-relevant
  expression across all three corpora and ranking the 17 criteria by centrality to *that*
  (not to each other) puts **radical behavior change** robustly first, regardless of whether
  literature's much larger volume is allowed to dominate the pool or each corpus is weighted
  equally. **Legal disputes** is robustly last either way — consistent with its coverage-gap
  finding above. **Mental destabilization**, the criteria list's own most central member
  (step 2), is only mid-table here (7th–12th depending on weighting): the official list is
  organized around destabilization as its conceptual anchor, but the corpus's actual usage
  organizes more around behavioral change and authoritarian structure. Two different, equally
  legitimate notions of "central" that genuinely diverge.

- **The central Hypothesis is supported at the level of the criteria, and every one of the 17
  criteria individually fails to discriminate** (steps 8 and 10). Secular structures score *as
  well as or better than* a religious/NRM baseline against the 17 official criteria in two
  independently-built sets — C1's hand-picked 10 (Amway, Nazism, Maoist thought reform, LGAT
  programs; "Maoist thought reform" alone outscores all 9 religious comparison entities) and
  C2's 169-name list (89% beat the religious baseline's mean). Per criterion (step 10), the gap
  is positive for **all 17**: no sectarian drift sits closer to religious groups than to ordinary
  non-religious organizations. The widest margins are the procedural drifts — exorbitant
  financial demands (+0.048) and deceptive recruitment (+0.042) — exactly what the Hypothesis
  predicts, since exploitative finance and misleading recruitment are generic organizational
  pathologies with no religious content.

- **The one criterion that resists generalization is the concept's own anchor** (steps 2, 7, 10).
  *Mental destabilization* is the criteria list's most central member internally (step 2),
  mid-table against the corpus prototype (step 7), and **dead last of 17** against non-religious
  organizations (step 10) — nearest drift for only 2 of 169. So the basis for assessment
  generalizes almost everywhere *except* at its core. Read alongside step 5 (brainwashing is
  central *because contested, not asserted*), a consistent picture: the cult concept is anchored
  precisely where it is least stable and least transferable.

- **A metric artifact forced a retraction, and is a finding in its own right** (steps 8, 10).
  Adding the semantically empty word "The" to an organization's name is worth **+0.061 cosine**
  on the same organization (four accidental duplicate pairs in the C2 list isolate this), and
  word count correlates with the criteria score at r=+0.384 — both larger than the substantive
  effects being measured. C2's apparent overtaking of the religious baseline on the prototype
  centroid was name formatting and has been retracted. More broadly, cosine similarity between an
  organization's *name* and a criterion's *text* measures topical overlap, not conduct: the four
  organizations nearest *"difficulty leaving the group"* are a mountain-biking group, an Arduino
  users' group and two sports clubs (matched on "group/club"), and the highest single criterion
  match in the whole run is **Free the Children against "indoctrination of children" (0.570)**.
  Scoring entities against criteria in this space needs behavioural text per entity, not names.

- **The organizations popular discourse most readily calls "basically a cult" are the farthest
  from this corpus's cult concept** (step 8). The National Football League is last of all 169
  (0.281 criteria, 0.383 prototype — a wide margin below the next), followed by TechCrunch, the
  International Olympic Committee and OpenStreetMap; all are short, artifact-free names.
  Meanwhile anti-cult watchdogs (Southern Poverty Law Center, Anti-Defamation League) and
  mental-health institutions (NIMH, APA) sit near the top. The centroid tracks **participation in
  cult discourse, not cult-likeness** — a body that writes about manipulation and deviance lands
  beside the groups accused of practising it. Sports fandom and intense corporate culture, the
  stock analogies, are about intensity of affiliation; this concept is about belief, ideology and
  psychological harm.

- **The gap between lay prototype and expert vocabulary is even sharper than the general
  folk-vs-expert finding suggested** (step 9). Restricting to exactly the 25 theoretically-motivated
  prototype exemplars (each interview's own first free-association answer, not just any
  segment) shows 52% landing in one of literature's entity-gap clusters — roughly double the
  corpus-wide base rate of 25.4%. The specific answer people give first, not casual interview
  speech in general, is what skews most toward the abstract/attitudinal register literature has
  the least specific vocabulary for.

---

## Notes

### Why a centroid can look far from its own "nearest" points, geometrically

Two distinct, purely geometric reasons, not a plotting failure (see
[`04_Projection_Techniques_Comparison.md`](04_Projection_Techniques_Comparison.md) for the full
investigation, including a 5-technique side-by-side that confirms this): (1) the nearest points
to a centroid aren't necessarily much closer than a *typical* point — some groups don't have
one dense "typical" mode, so "nearest" is only modestly closer than average, not a tight
standout cluster; (2) even when the nearest points genuinely are close to the centroid, they
can each sit close to it from a *different direction*, without being close to each other — a
"shell" around the centroid rather than a "ball." No 2-D picture can make either case look like
a tight huddle without misrepresenting the real geometry.

### Which 2-D technique to use going forward

Tested PCA, MDS, UMAP (two parameter sets), and t-SNE on the same data — all five agreed on
the qualitative picture for every group tested, so the choice doesn't change *what* the data
shows, only how defensible the picture is. Recommendation: **MDS for static thesis figures**
(its optimization target — 2-D distance should match real distance — is literally the claim a
centroid/nearest-neighbour plot makes, so it's the one technique whose picture means what it
looks like it means; PCA only preserves distance along its top 2 variance axes, and UMAP/t-SNE
preserve local neighbour *topology*, not metric distance, which is why both carry standard
warnings against reading cluster size/spacing literally). **UMAP if/when an interactive
browsing view is built** (e.g. the planned subtitle-synced interview-embedding display) — there
the goal is live exploration of thousands of points, not one static distance claim, and UMAP
scales better and gives a more intuitively "clustered" feel for browsing. The current centroid
plots in [`02_Corpus_Centroids.md`](02_Corpus_Centroids.md) still use UMAP (not yet switched to
MDS).

---

*Add new steps above this line as `0N_Title.md`, with a one-line summary here and the full
writeup in its own file.*
