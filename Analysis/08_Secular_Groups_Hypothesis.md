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

**C2 — LLM-generated comparison list (code written, not yet run).** Ollama is unreachable on
this Mac right now (confirmed: connection refused, binary not on PATH). Per the user's choice,
`thesis_corpus/generate_and_embed_secular_groups.py` is written and ready — it asks an LLM for
N well-known non-religious groups across varied domains (MLMs, workplace culture, fandoms,
sports fanbases, wellness brands, startups, fraternities, political movements) and embeds each
with bge-m3, the same embedding call `embed_v2.py`/`embed_domain_terms.py` already use. **To
run**: on the Windows/Ollama machine, from `thesis/corpus/`:
```
python -m thesis_corpus.generate_and_embed_secular_groups --n 30
```
then copy `processed/analysis_raw/secular_groups/generated_secular_groups.jsonl` back to this
Mac at the same relative path, and re-run `find_secular_group_mentions.py` — it automatically
folds the generated list in as a third group if the file is present, without needing any other
change.

Code: `thesis_corpus/find_secular_group_mentions.py` (comparison, run now),
`thesis_corpus/generate_and_embed_secular_groups.py` (C2, pending).

## Result — a real, nuanced answer, not a flat yes/no

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
higher.** This is the opposite pattern from the criteria-list result, and it's the more
sociologically expected one: however consistent the *official criteria* are, the corpus's
*actual discourse* about "cult" still skews toward religious framing — usage lags behind what
the criteria themselves would imply.

**Put together, this is the Hypothesis's own shape, not a refutation or confirmation of it in
isolation**: the basis for classifying certain groups as cults (the structural criteria) *would*
extend to other social structures if applied consistently (the criteria-list result) — but
whether they're actually *discussed* that way in practice is a separate, sociological question,
and the answer there is no, not yet (the prototype result). That gap between what the criteria
imply and what discourse actually does is itself a citable finding.

## Full comparison

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

Full data: [`data/secular_vs_religious_distances.csv`](data/secular_vs_religious_distances.csv).

## Caveats

- n=10/n=9 is small, and both lists are hand-picked, not sampled — a real limitation for any
  claim stronger than "these specific examples," not yet a properly powered comparison. C2's
  LLM-generated list (larger, differently-selected) is a direct check on whether this holds up
  outside this specific hand-curated set.
- "Werner Erhard" is a person, not a group/structure — kept in the LGAT cluster for
  completeness (he's essentially synonymous with the movement he founded) but worth flagging as
  a category mismatch with the rest of the secular set if this list is used for anything
  requiring strict group/person consistency.
