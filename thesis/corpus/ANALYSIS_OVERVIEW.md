# Analysis Overview

A reference snapshot of the shared cross-corpus embedding space, for planning
the geometrical analysis. Reflects the pipeline as of the `build_shared_space`
run that produced `processed/shared_space/embedding_space.jsonl` (44,325
points, k=394, after the duplicate/short-fragment filter below). Regenerate
the numbers below after any pipeline rerun — they are not guaranteed to stay
in sync automatically.

**Pending**: MIVILUDES's 732 expression points are still embedded in their
original French (language-asymmetry mitigation planned, not yet run — see
"Known Limitations" below); once translated, `label` for these points will
become the English translation and `label_fr` will hold the French original,
same inverted pattern as `miviludes_criteria`'s `label`/`label_en`.

## Project Goals & Hypothesis

This research investigates the criteria through which particular objects,
groups, practices, and social formations come to be considered "cults." It
asks which conceptual implications follow from those criteria, what
contemporary forms the object of "the cult" may take, and whether "cult" can
be treated as a coherent analytical category at all — mapping the
heterogeneous expressions through which cult-like formations are described,
rather than starting from a fixed definition.

**Hypothesis** (verbatim, `1_Introduction.tex`):
> The basis for assessment that certain groups are cults would actually imply
> that other social structures are cults as well. Can we uncover these
> structures and what are they.

Framed against: "The meaning of cult goes beyond NRM as forecasted by the
reports to the MIVILUDES, spirituality not as a main driver."

What the analysis is meant to establish:
- Whether sources converge on recurring criteria.
- Whether the criteria form recognisable clusters or prototypes.
- Whether different sources organise them differently.
- Whether a "cult" is represented as a religious group, a social relation, a
  form of authority, a risk category, a process of capture, or a broader
  family of resemblances.
- Whether the category fragments when compared across academic,
  institutional, and everyday discourse.

## Three Epistemologies

| Epistemology | Corpus | What it captures |
|---|---|---|
| Legal/administrative | MIVILUDES | The French state's operational "sectarian drift" framework — deliberately avoids labeling groups, focuses on harmful consequences |
| Scholarly | Literature | Academic NRM/cult-studies literature |
| Lay/prototype | Interviews | 26 semi-structured interviews, opening with a free-listing prompt; everyday use of "cult" |

## Datasets in the Shared Space

One shared 394-dimensional PCA space (95.0% cumulative variance), pooled from
seven sources, 44,325 points total. The three expression-corpus counts below
are *after* pooling-time filtering (exact-duplicate expressions within the
same document, and expressions under 5 words — both known LLM-extraction
artefacts documented in Methods.tex, never removed from the archives
themselves): 775 duplicates + 3,048 short fragments removed in total
(literature 707/2,908, MIVILUDES 67/115, interviews 1/25):

| `source_dataset` | Points | Key format | Label semantics |
|---|---|---|---|
| `literature` | 35,621 | `document_id:chunk_index` | An extracted expression's short embedding text |
| `miviludes` | 732 | `document_id:chunk_index` | Same, from the 2 MIVILUDES source documents (currently French — see "Pending" above) |
| `interviews` | 204 | `document_id:chunk_index` | Same, from 26 interview transcripts |
| `miviludes_criteria` | 17 | `crit-<slug>` | French criterion text (`label`); English translation as `label_en`, display-only — not a separate point |
| `concept_backbone` | 3,000 | WordNet ILI id (e.g. `i71809`) | The concept's primary English lemma |
| `structural_concepts` | 1,500 | `sc_<0001..1500>` | The term itself (e.g. "control", "authority") |
| `emergent_entities` | 3,251 | normalized anchor text | The anchor text itself (e.g. "scientology") |

## Point Roles

Every point also carries a `point_role`, cutting across `source_dataset` to
group the seven datasets into three kinds of thing:

| `point_role` | Datasets | What it is |
|---|---|---|
| `expression` | `literature`, `miviludes`, `interviews`, `miviludes_criteria` | A criterion expression extracted from a text, or the MIVILUDES's own criterion text — something a source actually said |
| `reference` | `concept_backbone`, `structural_concepts` | A backdrop vocabulary point, not itself a claim any source makes. Two subsets: `concept_backbone` is topic-neutral (WordNet, not derived from any corpus — an independent yardstick); `structural_concepts` is corpus-derived (extracted from the corpora's own expression text, geometrically closer to the data by construction, but not topic-neutral — see "Why two reference subsets" below) |
| `emergent` | `emergent_entities` | A named entity/group/concept mentioned *by* the corpora themselves — corpus-derived like an expression, but a recurring reference object rather than a claim |

### Why two reference subsets

The original `concept_backbone` is deliberately topic-neutral, which is
exactly what makes it a valid independent yardstick — but measured in the
shared space, it sits notably farther from the corpus-expression centroid
than emergent entities do (16.7 vs. 12.5 shared-space units, against a
~33-unit expression-to-expression baseline): a real, if moderate, "distant
neutral island" effect. Mining `entity_anchors` for a closer, generic
vocabulary (the same field `emergent_entities` uses) was tried and
abandoned first: even summed across the whole corpus with no threshold,
words like "control" (4 mentions), "authority" (12), "manipulation" (3),
"harm" (0) barely register as tagged anchors — that field captures concrete
named things, not abstract social-structure vocabulary, at any scale.
`structural_concepts` instead tokenizes the actual expression prose,
keeping only tokens that are valid Open English WordNet lemmas in an
in-domain lexicographer file (social/relational/cognitive/group-dynamics,
excluding concrete/physical domains; verbs excluded entirely) and not a
named-entity instance — this both filters out proper nouns/non-English
tokens and supplies a free English gloss per survivor. 7,053 candidates
survived out of 33,113 unique tokens found in the expression text; the top
1,500 by expression-frequency are kept, chosen empirically rather than
picked in advance — mentions per term fall off gradually (70 at rank 600,
40 at rank 1,000, 24 at rank 1,500), and both proper-noun leakage and
per-term mention counts degrade noticeably past that point, putting 1,500
at the practical ceiling before quality drops. Top by mentions: "religious"
(2,688), "movement" (1,512), "church" (1,250), "cult" (1,229), "group"
(1,129), "religion" (1,066) — genuinely structural/relational, unlike
`emergent_entities`' own top mentions (named groups). A handful of wrong
WordNet senses (e.g. "religious" defaulting to a noun sense meaning a monk)
were hand-corrected. Extending from an initial 600 to 1,500 surfaced a
systematic gap, not just one-off noise: Open English WordNet's
`instance_hyponym` marking is incomplete for minor historical figures, so
common words coinciding with their entries leaked through ("smith", "land",
"king", ...) — including two on-topic-but-wrong leaks that would have
undermined the dataset's own purpose: "hubbard" (L. Ron Hubbard) and
"iskcon" (a specific named sect, the Hare Krishnas) are exactly the kind of
named-person/named-group contamination this filter exists to keep out. A
systematic sweep for biographical-looking glosses ("United States ...", a
birth year, a year range) caught 53 of these at once; verified zero such
matches remain in the final 1,500. Neither correction is exhaustive —
residual noise should be expected at this scale, not assumed absent.

**Verified after embedding and rerunning** (at 600 concepts, again at the
final 1,500, and again after the duplicate/short-fragment filter below):
`structural_concepts`' nearest expression-point distance (mean 33.79,
sampled) sits close to the expression-to-expression baseline (33.68) —
essentially embedded within the expression cloud, not a separate cluster.
`concept_backbone` remains farther out (34.85), consistent with staying
topic-neutral rather than corpus-proximate. Centroid distance to the
expression centroid also improved: 16.2 (`structural_concepts`) vs. 16.7
(`concept_backbone`), both against `emergent_entities`' 13.2. The exact
figures have shifted slightly across each rerun (600→1,500 concepts, then
the pooling-time filter) but the qualitative finding — structural concepts
close to baseline, concept backbone consistently farther out — has held at
every stage.

## Categorical Facets Available for Analysis

All fields below live directly in `embedding_space.jsonl` and every
`visualization_{pca,umap,tsne}_3d.jsonl` file — no extra join needed for
these. `null`/absent where not applicable, never a fabricated default.

- **`source_dataset`** (7 values, table above) — the coarsest split.
- **`point_role`** (3 values, table above) — `expression`/`reference`/
  `emergent`; the coarser split when the question is about the *kind* of
  point rather than which specific dataset it came from (e.g. "compare
  expression points against reference points" without caring whether an
  expression came from literature or an interview).
- **`attribution`** — corpus-expression points only (`literature`,
  `miviludes`, `interviews`); `null` for the other three. Values: `author`,
  `cited_author`, `participant`, `institution`, `journalist`, `unspecified`.
  For interviews specifically: 185 `participant` / 19 `unspecified` (this is
  what separates the interviewee's own words from the interviewer's
  questions — both can land in the same extracted chunk otherwise).
- **`claim_mode`** — same coverage as `attribution`. Values:
  `direct_statement`, `attributed_statement`, `quotation`, `definition`,
  `question_or_reflection`, `other`.
- **`epistemic_status`** — same coverage. Values: `asserted`, `qualified`,
  `contested`, `negated`, `speculative` — the small non-`asserted` slivers
  are exactly the sites of live disagreement/ambivalence ("is this a cult or
  a religion?") worth isolating geometrically.
- **`response_rank`** — interviews only; `null` elsewhere. 1-indexed position
  of the item within its own interview transcript, in archive order (not
  filtered to the interviewee's own turns). Interviews open with a
  free-listing prompt ("what comes to mind when you hear the word cult?"),
  and order of mention is a standard cognitive-salience proxy in prototype
  theory — first-mentioned is treated as most prototypical. A useful
  *analysis-time derived* facet (not itself stored): `is_prototypical_core`
  = `response_rank <= 3`, to separate a prototypical core from peripheral
  mentions.
- **`emergent_entities`** (the point-set, not a per-point field) — 3,251
  named entities/dimensions mentioned ≥3 times across all corpora, each with
  its own position in the shared space (`point_role="emergent"`). Not
  joined onto corpus-expression points as a list field; compare by
  proximity instead. Top by mention count: Scientology (3,867), charismatic
  leader (3,417), NRMs (1,279), cults (688), Heaven's Gate (577), new
  religious movements (529), cult (426), new age (387), new religions
  (332), Unification Church (273), Jehovah's Witnesses (233), sect (205),
  brainwashing (204).
- **`mention_distribution`** — emergent-entity and structural-concept
  points only; `null` elsewhere (including
  `concept_backbone`, which has no corpus-mention notion at all). A
  per-corpus mention count, e.g.
  `{"literature": 3683, "miviludes": 130, "interviews": 54}` for
  "scientology" — provenance metadata, not used in the PCA fit (still one
  point per term either way). Lets a term mentioned near-exclusively in
  one epistemology (e.g. "NRMs", "Heaven's Gate", "Unification Church" — all
  literature-only) be told apart from one that recurs across all three (e.g.
  "scientology", "cults") — useful for checking whether the vocabulary that
  actually structures the space is epistemology-specific or genuinely
  shared.
- **`miviludes_criteria`** (17 points) — joinable to the official MIVILUDES
  17-criteria list via `key` (`crit-<slug>`). FR/EN translation fidelity
  (raw-embedding cosine, not a shared-space property): mean 0.87, median
  0.90, p10 0.81, min 0.50 (`crit-legal-disputes`) — only that one pair
  falls under the 0.70 manual-inspection threshold (recalibrated from an
  earlier flat <0.90 rule, which flagged this same pair as if it were a
  fresh concern; hand inspection confirms it's an accurate translation, not
  an error — short official phrases just embed less stably cross-lingually
  than length alone would suggest, worth remembering when comparing other
  short texts, like emergent entities, across languages).

## Joinable Document/Interview Metadata (not embedded, join via `document_id`/`id`)

- **Interviews** (`interviews/metadata/database.csv`): `batch`, `method`
  (in-person/Instagram DM), `language` (21 EN/5 FR), `age`, `gender` (13M/
  11F/1 other/1 redacted), `nationality`, `interviewer`, `date_time`.
- **Literature** (`metadata/literature.csv`): `type` (23 book/12
  book_chapter/33 article), `year` (2001–2024), `authors`, `tags`. Caveat:
  68 catalog rows vs. 57 processed documents — verify the join key lines up
  before relying on it.
- **MIVILUDES**: only 2 source documents (the 2022–2024 activity report;
  the "Comment identifier une dérive sectaire?" criteria page) —
  `document_id` itself is the only useful split.

## Known Limitations

Mirrors `Methods.tex`'s "Known Limitations" subsection — see there for full
prose; summarized here for quick reference while planning analysis:

- **Corpus imbalance**: literature is ~97% of expression points (35,621 vs.
  MIVILUDES's 732, interviews' 204). Measured, not assumed: the
  equal-corpus-weighted grand centroid sits 4.02 shared-space units from the
  plain unweighted one. Use `thesis_corpus.balanced_analysis` for any
  quantitative (not visualization) claim about the corpus as a whole —
  `weighted_centroid()` / `per_corpus_centroids()` as reusable functions, or
  the pre-written `processed/shared_space/literature_balanced_sample.jsonl`
  (2,500 points, stratified by document) as a drop-in literature subset.
- **MIVILUDES = 2 documents**: treat as one influential operational
  framework, not a representative sample of French state framing broadly.
- **Language asymmetry**: MIVILUDES is ~100% French; both reference
  point-sets are English-only. Mitigation (translate MIVILUDES's 732
  expressions to English, use as the primary embedding, French as a
  `label_fr` display field) is designed and code-complete
  (`translate_miviludes_expressions.py`) but not yet run — needs Ollama, on
  the Windows machine, same handoff pattern as `structural_concepts`'s
  embedding step. See "Pending" at the top.
- **Interview sample**: convenience-sampled through the researcher's own
  network (one response excluded for researcher-influence bias; another
  named the researcher's own academic programme a "cult"). Treat as
  exploratory prototype data, not a representative sample of lay usage.

## What's NOT Yet Done

- MIVILUDES's expression translation (see "Known Limitations") needs
  running on the Ollama-serving machine, then `build_shared_space.py` and
  `visualize_3d.py` need rerunning.
- No clustering or distance analysis has been run on the shared space.
- No criterion-centred nearest-neighbour inspection (e.g. which corpus
  expressions sit closest to each of the 17 MIVILUDES criteria).
- No rank-stratified analysis (prototypical core vs. peripheral mentions in
  interviews).
- Results.tex is still a placeholder beyond the qualitative interview-survey
  notes; no geometric finding has been written up yet.
