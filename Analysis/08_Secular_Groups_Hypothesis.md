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
result — and, as C2 below shows, it doesn't survive a larger, more broadly-sampled secular set.

## Result — C2 (n=169, broad LLM-sourced list): the pattern reverses again

| Centroid | Secular (C1, n=10) | Religious baseline (n=9) | **Generated secular (C2, n=169)** |
|---|---|---|---|
| 17-criteria list | 0.454 | 0.440 | **0.496** (median 0.499, range 0.281–0.605) |
| Pooled prototype | 0.517 | 0.576 | **0.587** (median 0.591, range 0.383–0.693) |
| Equal-weighted prototype | 0.494 | 0.541 | **0.562** (median 0.563, range 0.355–0.686) |

**With a broad, 169-name secular set, the religious baseline no longer scores highest on *any*
centroid — including the prototype centroid where it previously did.** Against the criteria
list, 151/169 (89%) of the generated secular names score above the religious baseline's own
*mean*, and 55/169 (33%) score above the religious baseline's *maximum* (0.522). Against the
pooled prototype, the religious baseline's earlier lead shrinks from a 0.059 gap (C1) to being
overtaken outright: 105/169 (62%) of the generated names exceed the religious baseline's mean.
This is the opposite conclusion from the C1-only prototype result — reversed once the secular
comparison set gets bigger and more heterogeneous, not confirmed by it.

**Within the 169, the ranking is not noise — there's a real semantic gradient, and it points at
a specific confound.** The top scorers on both centroids are heavily concentrated in
**humanist/secular-advocacy and rights/ideology organizations**: Anti-Defamation League, Southern
Poverty Law Center, American Psychological Association, Secular Student Alliance, Freethought
Association, International Humanist and Ethical Union, Progressive Alliance, Center for American
Progress. The bottom scorers are **commercial/recreational/entertainment organizations** with
comparably generic institutional names: National Football League (lowest on both centroids),
TechCrunch, Doctors Without Borders, International Olympic Committee, OpenStreetMap. Since both
ends of the ranking share the same "The X for/of Y" institutional naming register, this isn't
simply a lexical-register artifact (bare organization names embedding close to the corpus's own
formal register regardless of content) — there's a genuine thematic split between
ideology/belief/rights-oriented organizations and purely commercial/recreational ones.

**But that gradient is also the result's biggest methodological caveat.** A large fraction of
the C2 list is explicitly *secularist* advocacy organizations (Freedom From Religion Foundation,
Secular Coalition for America, Secular Student Alliance, Freethought Association, Secular
Society, International Humanist and Ethical Union) — organizations whose entire identity is
defined by their stance *on* religion, not merely non-religious ones. That they embed close to a
corpus about religious/cult category boundaries is unsurprising on its own terms (shared
vocabulary: belief, doctrine, ideology, opposition to a dominant worldview) and arguably measures
something adjacent to the Hypothesis rather than the Hypothesis itself — this was flagged before
C2 was run (see Caveats below) and the run confirms it's a real effect on the ranking, not a
hypothetical one.

**Put together across C1 and C2, this is the Hypothesis's own shape, not a flat refutation or
confirmation in isolation**: the basis for classifying certain groups as cults (the structural
criteria) *does* extend to other social structures when applied consistently — both the small
hand-picked C1 set and the much larger C2 set support this on the criteria-list centroid. Whether
that also holds against the corpus's own *actual discourse* prototype is less settled than C1
alone suggested: C1's small set said no; C2's much larger set says the religious lead disappears
entirely, though a meaningful chunk of that reversal is plausibly driven by including
religion-opposing secularist organizations rather than a religiously-neutral secular sample. A
version of C2 restricted to secular groups with no explicit religious/ideological stance (MLMs,
sports, hobby, tech, wellness — the bottom half of the current ranking) would be a cleaner,
narrower test of the same question.

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
  embedded) — consistent with how C1's and the religious baseline's entity vectors are also
  anchor-text-only embeddings (`build_shared_space.load_emergent_entities`'s
  `entity_anchor_vectors`, the same convention throughout this session's raw-space analyses), so
  the three groups are compared on equal terms, but none of the three captures whatever additional
  signal a full descriptive sentence about each group might carry.
