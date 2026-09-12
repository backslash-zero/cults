# Testing the central Hypothesis: do non-religious structures score as "cult-like"?

Data inventory: [`00_Available_Data.md`](00_Available_Data.md). Index: [`Results_Draft.md`](Results_Draft.md).

The thesis's own central Hypothesis: *"The basis for assessment that certain groups are cults
would actually imply that other social structures are cults as well."* Tested directly here for
the first time — the corpus itself is entirely about cults/NRMs already, so this required
pulling a deliberate comparison set from outside that frame.

## Method

**C1 — corpus-native (done now).** No automated religious/secular classifier exists anywhere in
this codebase (`looks_like_named_entity` is purely orthographic; `provenance_category`
classifies by *which corpus* mentions a term, not semantic type) — building one wasn't the point
of this check. Instead: keyword-assisted candidate generation over `entities_all`'s 3,785
labels (grep for corporate/MLM/self-help-seminar/political-movement patterns), then manual
review and tagging — same rigor as the cluster-membership checks in `06_Literature_Clusters.md`.
Confirmed genuine secular structural mentions, hand-reviewed (full reasoning in
`thesis_corpus/find_secular_group_mentions.py`'s `SECULAR_STRUCTURE_ENTITIES`):
**Amway** (MLM), **Landmark Forum** / **Erhard Seminar Training ("est")** / **Lifespring** /
**Werner Erhard** (Large Group Awareness Training — secular self-improvement seminars, widely
discussed in cult-studies literature as structural analogues), and **Nazism / National
Socialism / Maoist thought reform / Communist Party / Chinese Communist Party** (secular
political movements whose indoctrination mechanics this literature explicitly compares to
religious "thought reform"). Scientology's entity forms were deliberately excluded — it's
conventionally classified as a new religious movement in NRM scholarship, not a clean secular
comparison case.

**Baseline**: 9 unambiguously religious/NRM entities already characterized as tight,
well-anchored literature clusters in `06_Literature_Clusters.md` (Heaven's Gate, Aum Shinrikyo,
Branch Davidians, Peoples Temple, Unification Church, Solar Temple, the Family/Children of God,
Rajneesh Movement, Theosophical Society).

Both sets measured against the same three centroids as `07_Cult_Prototype.md`: the 17-criteria
list's own centroid, and the pooled/equal-weighted corpus-wide prototype.

**C2 — LLM-generated comparison list (done).** Live, on-the-fly generation
(`generate_and_embed_secular_groups.py`, asking a locally-run Ollama model for N names directly)
was abandoned after three straight failures across a slow manual Windows-run/copy-back cycle —
a request timeout, an off-topic response (a markdown tutorial about scraping Reddit instead of
a name list), then a generic chatbot non-answer ("Are you asking about a specific topic?...").
Instead, the user ran the same kind of prompt themselves, reviewed the raw output, and pasted it
into `dictionaries/non-religious-groups/*.txt` — two free-text LLM transcripts (one English, one
French) listing well-known non-religious groups across humanist/secular-advocacy, environmental,
social-justice, health, education, arts, political, and miscellaneous-hobby categories.
`thesis_corpus/parse_manual_secular_groups.py` deterministically extracts bare names from that
free text (distinguishing a real entry from a category-header line by the presence/absence of a
name–description separator, confirmed against every line in both source files) and dedupes
case-insensitively across both files: **169 unique names**, zero overlap with C1's hand-picked
10. `thesis_corpus/embed_manual_secular_groups.py` then embeds each via bge-m3 (same embedding
call `embed_v2.py`/`embed_domain_terms.py` already use) — the one step that still needed the
Windows/Ollama machine, since Ollama remains unreachable on this Mac.

Code: `thesis_corpus/find_secular_group_mentions.py` (comparison), `thesis_corpus/parse_manual_secular_groups.py`
+ `thesis_corpus/embed_manual_secular_groups.py` (C2 list construction). Source lists:
`dictionaries/non-religious-groups/`.

## Result — C1 (n=10, hand-picked): a nuanced answer, not a flat yes/no

| Centroid | Secular mean cos | Religious-baseline mean cos | Difference |
|---|---|---|---|
| 17-criteria list | **0.454** (n=10) | **0.440** (n=9) | secular *slightly higher* |
| Pooled prototype | 0.517 (n=10) | 0.576 (n=9) | religious higher |
| Equal-weighted prototype | 0.494 (n=10) | 0.541 (n=9) | religious higher |

**Against the official structural criteria, secular and religious examples score essentially
the same — direct support for the Hypothesis.** Secular structures aren't merely "close enough"
to the 17 criteria; they score *marginally higher on average* than the religious baseline set.
Individually: **"Maoist thought reform" (0.544) scores higher against the criteria list than
every single one of the 9 religious baseline entities** (the highest religious score is
Theosophical Society at 0.522). Nazism (0.490) and National Socialism (0.484) also both beat
several religious entries (Unification Church 0.468, Aum Shinrikyo 0.463, Rajneesh Movement
0.436, Branch Davidians 0.411, Peoples Temple 0.406, Heaven's Gate 0.396, Solar Temple 0.347).
The *structural* criteria, applied consistently, genuinely do not discriminate between religious
and secular groups.

**Against the corpus's own emergent "cult prototype," religious examples score consistently
higher (in this small, hand-picked set).** This is the opposite pattern from the criteria-list
result. C2 initially appeared to overturn it; that apparent overturning did not survive the
name-form control (see *Mechanism*), so this C1 result stands unchallenged rather than refuted —
but also unreplicated.

## Result — C2 (n=169, broad LLM-sourced list)

| Centroid | Secular (C1, n=10) | Religious baseline (n=9) | **Generated secular (C2, n=169)** |
|---|---|---|---|
| 17-criteria list | 0.454 | 0.440 | **0.496** (median 0.499, range 0.281–0.605) |
| Pooled prototype | 0.517 | 0.576 | **0.587** (median 0.591, range 0.383–0.693) |
| Equal-weighted prototype | 0.494 | 0.541 | **0.562** (median 0.563, range 0.355–0.686) |

Taken at face value these numbers say the religious baseline no longer leads on *any* centroid.
**One of those two results survives scrutiny and the other does not** — see the next section
before using either.

**The criteria-list result holds.** 151/169 (89%) of the generated names score above the religious
baseline's own *mean*, and 55/169 (33%) score above its *maximum* (0.522). It survives the
name-form control below (+0.033 rather than +0.056), and it points the same direction C1 found
independently on short lowercase names (+0.014). Two differently-constructed secular sets agree.

**The prototype-centroid "reversal" does not hold, and is retracted.** It is an artifact of how
C2's names are spelled, not a property of the groups — see below.

## Mechanism — what this metric actually measures

Before interpreting anything above: C2's names and the corpus-native entity anchors differ
systematically in *surface form*, and bge-m3 is measurably sensitive to that difference.

| Group set | n | Mean words | Starts with "The" | All-lowercase |
|---|---|---|---|---|
| C2 generated | 169 | 3.30 | 71/169 | 1/169 |
| religious baseline | 9 | 2.56 | 1/9 | 9/9 |
| C1 corpus-native secular | 10 | 2.00 | 0/10 | 10/10 |

Corpus-native anchors are normalized lowercase text (`build_shared_space.load_emergent_entities`);
C2's are Title Case organization names as an LLM writes them. **C2's source lists happen to contain
four organizations under both a bare and a "The"-prefixed name, which isolates the effect of one
semantically empty token on the same organization:**

| Organization | bare name | with "The" | delta |
|---|---|---|---|
| International Gay and Lesbian Human Rights Commission | 0.456 | 0.536 | **+0.081** |
| American Civil Liberties Union | 0.423 | 0.496 | **+0.073** |
| Women's March | 0.482 | 0.527 | +0.045 |
| International Humanist and Ethical Union | 0.531 | 0.575 | +0.044 |
| | | **mean** | **+0.061** |

**Adding the word "The" is worth +0.061 cosine — larger than the entire +0.056 C2-vs-baseline gap
it would otherwise be credited to.** Word count correlates with the criteria score at r = +0.384,
and 71/169 C2 names carry the "The" bonus that essentially no baseline entity has.

Restricting C2 to the baseline's own name profile (≤3 words, no leading "The"; 90 of 169 qualify):

| Comparison | Criteria list | Pooled prototype |
|---|---|---|
| religious baseline (n=9) | 0.440 | 0.576 |
| C2 all (n=169) | 0.496 (**+0.056**) | 0.587 (**+0.011**) |
| C2 name-form-matched (n=90) | 0.473 (**+0.033**) | 0.571 (**−0.005**) |

The criteria-list advantage shrinks but survives. **The prototype advantage inverts to −0.005,
i.e. vanishes.** The "reversal" reported in the first version of this document was name
formatting. A residual confound remains uncorrected: every C2 name is Title Case and every
baseline anchor is lowercase, which cannot be tested without re-embedding the baseline in Title
Case (needs the Windows/Ollama machine).

**A third member of the same family, found later.** The artifact above is about the *group names*.
The same sensitivity applies to the *other* side of any comparison — the text a group is compared
**to**. In [`10_Secular_Groups_vs_Sectarian_Drifts.md`](10_Secular_Groups_vs_Sectarian_Drifts.md),
where groups are scored against the 17 individual criteria rather than their pooled centroid,
criterion text length (3 to 29 French words) predicts absolute similarity at **Spearman −0.613**,
and that artifact was large enough to manufacture an entire false finding before it was caught —
see `10`'s retracted Result 2. This document's own numbers are not exposed to it, because the
criteria *centroid* is one fixed vector for every group set, so no per-criterion length varies. The
general rule, worth carrying into any future use of this space: **whenever the two sides of a
cosine comparison differ systematically in length or surface form, measure that difference before
interpreting the similarity.**

**What does survive name-form control**, checked inside the matched 90:

- **The thematic gradient is real.** Ideological/psychological vocabulary in the name (humanist,
  secular, rights, mental, mindful, equality, justice) scores 0.496 vs 0.465 for
  concrete/brand-name vocabulary — a +0.031 gap within the matched subset alone.
- **Sports, tech and commercial organizations are the farthest of all 169 from the cult concept.**
  National Football League is last (0.281 criteria, 0.383 prototype — a wide margin below
  TechCrunch at 0.321), followed by TechCrunch, Doctors Without Borders, the International Olympic
  Committee and OpenStreetMap. All are short, "The"-less names, so this is not the artifact.
  **This is the most counterintuitive finding here:** the organizations popular discourse most
  readily calls "basically a cult" — sports fandom, intense tech companies, fitness brands — are
  exactly what this corpus's cult concept does *not* resemble. The concept as constituted here is
  about belief, ideology and psychological harm, not about intensity of affiliation or tribal
  loyalty.
- **Anti-cult and mental-health institutions score near the top**: Southern Poverty Law Center and
  the Anti-Defamation League — organizations whose actual function is to *monitor* hate groups —
  plus the National Institute of Mental Health and the American Psychological Association. The
  centroid measures **participation in cult discourse, not cult-likeness**: a body that writes
  about manipulation, deviance and belief sits geometrically beside the groups accused of
  practising it. This mirrors [`05_Epistemic_Status_Centroids.md`](05_Epistemic_Status_Centroids.md)'s
  finding that brainwashing is central *because it is contested rather than asserted* — the same
  structure, arrived at from the opposite direction.

**Put together across C1 and C2**: the basis for classifying groups as cults (the structural
criteria) does extend to non-religious structures when applied consistently — supported by two
independently-built secular sets, and by all 17 criteria individually in
[`10_Secular_Groups_vs_Sectarian_Drifts.md`](10_Secular_Groups_vs_Sectarian_Drifts.md). Whether the
corpus's *actual discourse* prototype behaves the same way is **unresolved**: C1's small set said
religious groups score higher, C2's apparent contradiction was an artifact, and no name-form-clean
comparison currently separates them. That question needs the criterion-level analysis in
[`10`](10_Secular_Groups_vs_Sectarian_Drifts.md) and, ultimately, behavioural text per
organization rather than bare names.

## Full comparison (C1, n=19)

| Group | Entity | Criteria-list cos | Pooled prototype cos | Equal-wt prototype cos |
|---|---|---|---|---|
| religious_baseline | theosophical society | 0.522 | 0.658 | 0.608 |
| religious_baseline | the family (formerly the children of god) | 0.514 | 0.650 | 0.612 |
| religious_baseline | unification church | 0.468 | 0.639 | 0.593 |
| religious_baseline | aum shinrikyo | 0.463 | 0.572 | 0.554 |
| religious_baseline | rajneesh movement | 0.436 | 0.476 | 0.451 |
| religious_baseline | branch davidians | 0.411 | 0.524 | 0.508 |
| religious_baseline | peoples temple | 0.406 | 0.571 | 0.532 |
| religious_baseline | heaven's gate | 0.396 | 0.553 | 0.511 |
| religious_baseline | solar temple | 0.347 | 0.541 | 0.502 |
| secular_structure | maoist thought reform | 0.544 | 0.579 | 0.559 |
| secular_structure | nazism | 0.490 | 0.562 | 0.532 |
| secular_structure | national socialism | 0.484 | 0.538 | 0.514 |
| secular_structure | landmark forum | 0.483 | 0.567 | 0.553 |
| secular_structure | erhard seminar training | 0.468 | 0.498 | 0.487 |
| secular_structure | communist party | 0.466 | 0.536 | 0.502 |
| secular_structure | lifespring | 0.427 | 0.522 | 0.509 |
| secular_structure | chinese communist party | 0.413 | 0.486 | 0.446 |
| secular_structure | amway | 0.388 | 0.462 | 0.443 |
| secular_structure | werner erhard | 0.378 | 0.417 | 0.393 |

Within the secular set, a real internal gradient is visible too: political-ideology examples
(Maoist thought reform, Nazism) score highest against the criteria; the more overtly
commercial ones (Amway, Werner Erhard personally) score lowest — worth noting if this set is
expanded later.

## C2 full ranking (n=169): top and bottom 10 per centroid

| Rank | Top 10 — criteria-list cos | cos | Bottom 10 — criteria-list cos | cos |
|---|---|---|---|---|
| 1 | Volunteering Matters | 0.605 | National Football League | 0.281 |
| 2 | The Anti-Defamation League | 0.591 | TechCrunch | 0.321 |
| 3 | The Disability Rights Education & Defense Fund | 0.582 | Doctors Without Borders | 0.387 |
| 4 | The National Institute of Mental Health | 0.579 | OpenStreetMap | 0.405 |
| 5 | The Southern Poverty Law Center | 0.577 | International Olympic Committee | 0.411 |
| 6 | The International Humanist and Ethical Union | 0.575 | Black Lives Matter | 0.412 |
| 7 | The American Psychological Association | 0.575 | Khan Academy | 0.414 |
| 8 | Mindfulness.org | 0.574 | TimeBank | 0.417 |
| 9 | The Progressive Alliance | 0.573 | Freecycle | 0.420 |
| 10 | The Center for American Progress | 0.567 | NAACP | 0.422 |

| Rank | Top 10 — pooled prototype cos | cos | Bottom 10 — pooled prototype cos | cos |
|---|---|---|---|---|
| 1 | Volunteering Matters | 0.693 | National Football League | 0.383 |
| 2 | Mindfulness.org | 0.678 | TechCrunch | 0.417 |
| 3 | The Secular Student Alliance | 0.675 | Doctors Without Borders | 0.471 |
| 4 | The Anti-Defamation League | 0.670 | International Olympic Committee | 0.491 |
| 5 | The Skeptics Society | 0.668 | Library of Congress | 0.503 |
| 6 | The Progressive Alliance | 0.667 | Black Lives Matter | 0.506 |
| 7 | The Secular Education Network | 0.667 | American Civil Liberties Union | 0.511 |
| 8 | The International Secular Humanist Association | 0.657 | Khan Academy | 0.512 |
| 9 | The Secular Outlook | 0.657 | OpenStreetMap | 0.514 |
| 10 | The Freethought Association | 0.657 | Sustainable Development Goals | 0.515 |

Full data (all 169 generated names + C1's 19): [`data/secular_vs_religious_distances.csv`](data/secular_vs_religious_distances.csv).

## Caveats

- **Surface form moves this metric more than the effects being measured** (see *Mechanism*): a
  leading "The" is worth +0.061 on the same organization, and word count correlates at r=+0.384.
  Any comparison between group sets whose names are spelled differently — which is every
  C2-vs-corpus-native comparison here — is confounded to that degree. The matched-subset figures
  are the ones to quote; the Title-Case-vs-lowercase difference remains uncorrected entirely.
- **One claim in the first version of this document was retracted on that basis** (C2 overtaking
  the religious baseline on the prototype centroid). Kept visible rather than quietly deleted,
  since the retraction is itself the methodological finding.
- **C1's n=10/n=9 is small and hand-picked, not sampled** — C2's much larger, LLM-sourced list
  addresses that specific limitation, but introduces a different one (below).
- **C2 over-represents secularist/humanist advocacy organizations** relative to the "ordinary
  non-religious structure" the Hypothesis has in mind (MLMs, fandoms, workplace culture, sports
  fanbases) — a consequence of how the source lists were generated (an LLM asked broadly for
  "non-religious groups" leaned heavily on organizations defined by secularism/humanism/rights
  advocacy specifically). These are the top scorers on every centroid, so they're doing a
  disproportionate amount of work in the headline C2 numbers above. A stricter test would rerun
  the comparison on a version of C2 filtered down to religiously-neutral, non-ideological groups
  only.
- "Werner Erhard" (C1) is a person, not a group/structure — kept in the LGAT cluster for
  completeness (he's essentially synonymous with the movement he founded) but worth flagging as
  a category mismatch with the rest of the secular set if this list is used for anything
  requiring strict group/person consistency.
- C2's names are bare organization names with no descriptive context (the LLM's own one-line
  descriptions, e.g. "Promotes secular humanism," were stripped during parsing and never
  embedded). All three sets are anchor-text-only embeddings
  (`build_shared_space.load_emergent_entities`'s `entity_anchor_vectors`, the convention
  throughout this session's raw-space analyses) — but "same kind of vector" is *not* the same as
  "comparable on equal terms," which is exactly what the name-form finding above establishes.
  None of the three captures anything about what these organizations actually do;
  [`10_Secular_Groups_vs_Sectarian_Drifts.md`](10_Secular_Groups_vs_Sectarian_Drifts.md)'s
  criterion-level breakdown shows how far that limitation reaches.
