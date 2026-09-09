# Analysis Overview

A reference snapshot of the shared cross-corpus embedding space, for planning
the geometrical analysis. Reflects the pipeline as of the `build_shared_space`
run that produced `processed/shared_space/embedding_space.jsonl` (44,520
points, k=394, after the duplicate/short-fragment filter below). Regenerate
the numbers below after any pipeline rerun — they are not guaranteed to stay
in sync automatically.

**Done**: MIVILUDES's 732 expression points are now embedded via their
English translation (`label`), with the original French in `label_fr` —
same inverted pattern as `miviludes_criteria`'s `label`/`label_en`. See
"Known Limitations" below for the translation-fidelity numbers and known
residual noise.

## Extraction pipeline v2 — status

**Everything below in this file describes the first-generation (v1) archive
and the shared space built from it.** A 100-item manual fidelity review
(2026-09-08) found the v1 pipeline systematically over-inclusive — expressions
paraphrased, mistranslated, or truncated relative to their source, headings
and citations extracted as though they were claims, interviewer questions
occasionally mislabelled as participant statements. The extraction pipeline
was redesigned around a strict verbatim-span rule (the model returns only an
exact substring of the source chunk; a deterministic code layer, not a
further model call, screens it), a Stage-1 text-integrity audit (broken PDF
font/character-map corruption, found to enter before any model ever sees the
text), deterministic interview speaker-turn attribution, and — since full
manual review of a corpus this size was not feasible on this thesis's
timeline — a second, independently-prompted local model that verifies every
retained expression against the same criteria a human reviewer would apply
("model-judged", never treated as a substitute for human review). Full design
and rationale: `thesis/03_Content/3_Methods.tex`, "Revising the Extraction
Pipeline" (`sec:extraction_v2`). Implementation and usage:
`thesis_corpus/README.md`'s "Extraction v2" sections
(`text_integrity.py`/`audit_stage1_text_integrity.py`, `screen_v2.py`,
`judge_v2.py`, `extract_v2.py`, `embed_v2.py`, `sample_v2_expressions.py`).

**v1 is frozen** — its archives, this shared space, and every retained
analysis run listed below stay on disk unmodified, as historical/comparison
record, per the same never-modify-the-source discipline this file already
documents for the archives themselves.

**v2 status**: full-corpus extraction (`thesis_corpus.extract_v2`, checkpointed
per document, resumable) is running against all three corpora as of
2026-09-09; not yet complete, not yet embedded, and not yet pooled into a new
shared space. Nothing in this document should be read as describing v2's
corpus until that rebuild happens and this file is regenerated against it —
consistent with this file's own opening instruction to regenerate after any
pipeline rerun.

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
| Lay/everyday | Interviews | 26 semi-structured interviews, opening with a single first-association prompt (not a multi-item free list — see `response_rank`/initial-exemplars note below); everyday use of "cult" |

## Datasets in the Shared Space

One shared 394-dimensional PCA space (95.0% cumulative variance), pooled from
eight sources, 44,520 points total. The three expression-corpus counts below
are *after* pooling-time filtering (exact-duplicate expressions within the
same document, and expressions under 5 words — both known LLM-extraction
artefacts documented in Methods.tex, never removed from the archives
themselves): 775 duplicates + 3,048 short fragments removed in total
(literature 707/2,908, MIVILUDES 67/115, interviews 1/25):

| `source_dataset` | Points | Key format | Label semantics |
|---|---|---|---|
| `literature` | 35,621 | `document_id:chunk_index` | An extracted expression's short embedding text |
| `miviludes` | 732 | `document_id:chunk_index` | English translation (`label`); original French in `label_fr` — from the 2 MIVILUDES source documents |
| `interviews` | 204 | `document_id:chunk_index` | Same, from 26 interview transcripts |
| `miviludes_criteria` | 17 | `crit-<slug>` | French criterion text (`label`); English translation as `label_en`, display-only — not a separate point |
| `concept_backbone` | 3,000 | WordNet ILI id (e.g. `i71809`) | The concept's primary English lemma |
| `structural_concepts` | 1,500 | `sc_<0001..1500>` | The term itself (e.g. "control", "authority") |
| `conceptnet_concepts` | 195 | `cn_<0001..1345>` | The term itself (e.g. "spell", "mentor"); ConceptNet's associative neighborhood of the `structural_concepts` seeds, hand-pruned from 1,345 automated candidates down to 195 kept as domain-relevant (1,150 flagged as generic hub words on manual review — see `dictionaries/conceptnet_concepts_candidates.csv`'s `is_generic` column) |
| `emergent_entities` | 3,251 | normalized anchor text | The anchor text itself (e.g. "scientology") |

**`document_id:chunk_index` is not a unique key per point** for `literature`,
`miviludes`, or `interviews` — it identifies the *source chunk*, and one
chunk's LLM response can (and regularly does) yield several expressions
(literature: 39,236 raw expressions across only 5,146 unique
`document_id:chunk_index` pairs pre-filter; MIVILUDES: 914/118; interviews:
230/75). Fine for tracing a point back to its source chunk; not safe as a
dict/join key across two independently-produced files unless both are
known to iterate in identical order (this is exactly how the MIVILUDES
French/English join was briefly broken during this session's translation
work — fixed by keying on `document_id:chunk_index:occurrence`, see
`load_miviludes_translations()` in `build_shared_space.py`).

## Point Roles

Every point also carries a `point_role`, cutting across `source_dataset` to
group the eight datasets into three kinds of thing:

| `point_role` | Datasets | What it is |
|---|---|---|
| `expression` | `literature`, `miviludes`, `interviews`, `miviludes_criteria` | A criterion expression extracted from a text, or the MIVILUDES's own criterion text — something a source actually said |
| `reference` | `concept_backbone`, `structural_concepts`, `conceptnet_concepts` | A backdrop vocabulary point, not itself a claim any source makes. Three subsets: `concept_backbone` is topic-neutral (WordNet, not derived from any corpus — an independent yardstick); `structural_concepts` is corpus-derived (extracted from the corpora's own expression text, geometrically closer to the data by construction, but not topic-neutral); `conceptnet_concepts` generalizes `structural_concepts` via ConceptNet's associative graph, then hand-pruned — see "Why three reference subsets" below |
| `emergent` | `emergent_entities` | A named entity/group/concept mentioned *by* the corpora themselves — corpus-derived like an expression, but a recurring reference object rather than a claim |

A fourth role, `prototype`, exists too — but only in the separate
`interview_prototypes.jsonl` file described below, never pooled into
`embedding_space.jsonl` itself. The 44,520-point count and everything in
this table above is unaffected by it.

### Why three reference subsets

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
final 1,500, again after the duplicate/short-fragment filter, and again
after the MIVILUDES translation fix below): the qualitative finding —
structural concepts closer to the expression cloud, concept backbone
consistently farther out — has held at every pipeline stage. These exact
figures shift with *every* pooling-time change, not just ones that touch
structural concepts or the backbone directly — standardization is fit
jointly across the full pooled matrix before PCA, so swapping MIVILUDES's
732 points from French to English (or adding a new source_dataset)
shifted every dimension's mean/variance slightly, and with it every
point's coordinates, including literature's.

**Current figures** (from `analyze_global_structure.py`'s
`reference_to_combined_expression_centroid.csv`, run `20260908-175344`,
against the current 44,520-point space including `conceptnet_concepts`;
these exact figures are unchanged since `20260907-002832`, as neither of
the two intervening data corrections (documented below) touched any
embedding vector, only single metadata fields):
centroid distance to the combined (`full`-mode) expression centroid —
`concept_backbone` 16.70, `structural_concepts` 16.18,
`conceptnet_concepts` 14.92, `emergent_entities` 13.20. The pattern from
before this addition holds: structural concepts and the new ConceptNet
set both read closer to the expression cloud than the topic-neutral
concept backbone, with `conceptnet_concepts` closest of the three
reference vocabularies (unsurprising, given it was hand-pruned to
domain-relevant terms specifically). Note this centroid comparison is
pool-size-**un**controlled by construction (a centroid summarizes a whole
set regardless of size) — the earlier "mean nearest-expression-point
distance" figures reported in prior revisions of this document (31.61 /
33.08 / 29.06 baseline) were a separate, ad-hoc diagnostic computed
outside the toolkit and have not been recomputed against the current
space; they are removed here rather than left stale. The toolkit's own
equal-size-controlled nearest-term comparison
(`nearest_reference_terms_equal_size.csv`) is the current, reproducible
analogue — see "Why three reference subsets" and the interpretation note
in `thesis_corpus/README.md`.

### The third reference subset: `conceptnet_concepts`

Seeded from `structural_concepts`' 1,500 terms (not a repeat of them —
already-present terms are excluded), `extract_conceptnet_concepts.py`
pulls every English ConceptNet edge touching one of those seed words
(475,700 edges after excluding loose/morphological/oppositional relation
types — see that script's docstring), ranks the resulting ~1,300 candidate
terms by `n_seed_connections × specificity` (specificity = how much of a
term's *total* ConceptNet connectivity is accounted for by these seeds,
not just raw seed-edge weight — needed because ranking by raw weight alone
floods the result with hub words like "make"/"have"/"activity" that
loosely connect to nearly everything), then applies the same in-domain
Open English WordNet lexfile filter `structural_concepts` uses. That
automated pipeline still surfaced real hub-word noise near the top of its
1,345 candidates (e.g. "make", "location", "communicating", "intellect"),
so every row was then hand-reviewed: 195 kept as genuinely relevant to the
cult/sect/authority/manipulation/religion domain (e.g. "spell", "oracle",
"mentor", "grooming", "insulation" [isolation sense], "torah", "yoke"),
1,150 flagged `is_generic=false` and excluded (see
`dictionaries/conceptnet_concepts_candidates.csv`). Only the 195 kept rows
are ever embedded (`filter_conceptnet_concepts.py` extracts them before
the Ollama embedding pass) — the excluded 1,150 never leave the candidates
CSV. This is a smoother generalization of `structural_concepts` — thematically
anchored to the same domain via its seed words, but reached through
ConceptNet's associative graph rather than drawn directly from corpus
text, so it is not itself topic-neutral the way `concept_backbone` is.

## Interview Initial-Exemplar Prototype Layer

`processed/shared_space/interview_prototypes.jsonl` — a **separate,
manually-reviewed, projected layer**, not part of `embedding_space.jsonl`
and not counted in its 44,520 points or its PCA fit. It exists because
`build_shared_space.py`'s pooling-time length filter (drops expressions
under 5 words) removes exactly the kind of short, complete, spontaneous
answer the interview protocol's opening prompt is designed to elicit
("AI cult", "Illuminati", "Tomato cult!") — real content for this purpose,
not noise (see "Known Limitations" below for the filter's own, still-valid
rationale for literature/MIVILUDES). Rather than relaxing that filter —
which would reshuffle the whole pooled space, its PCA fit, and every
downstream count — this layer draws directly from the **unfiltered**
interview archive (`processed/interviews/criterion_expressions.jsonl`),
so it can include short genuine opening responses the pooled space cannot.

**How a point gets here**: a human reviews `interviews/metadata/initial_exemplars.csv`
row by row against the real transcript, then `thesis_corpus.build_interview_prototype_layer`
resolves the reviewed row against the raw archive (a virtual key stable
regardless of pooling — `thesis_corpus.geometric_analysis_common.derive_archive_expression_keys`),
takes that item's *already-computed* raw 1024-d embedding (no re-extraction,
no re-embedding), and projects it through `build_shared_space.py`'s
**persisted** fitted `StandardScaler`+PCA transform — the same
transform every pooled point already went through, so a prototype point
sits in the exact same 394-D coordinate system as everything else and is
directly comparable to it by Euclidean distance. Fails loudly (not
silently) on an unreviewed row, a missing `initial_response_form`, or a
row whose resolved archive text no longer matches what was reviewed.

**Current state**: 25 resolved prototype points, 1 interview unavailable
(`b3-aug23-1213` — no participant claim, excluding questions/reflections,
exists anywhere in that transcript; not a filtering casualty, a genuine
absence).

**Fields** (`source_dataset="interview_prototypes"`, `point_role="prototype"`):
`document_id`, `source_expression_key` (the archive-level virtual key),
`source_expression_label` (the pipeline's own extracted text for that
item — what's actually embedded), `transcript_initial_exemplar_text` (the
reviewer's own verbatim transcription of what the participant said —
deliberately a separate field, since an LLM extraction can trim or
paraphrase the verbatim wording), `exemplar_type` (*what* was named, if
anything), `initial_response_form` (*whether* anything was named at all —
`named_exemplar` / `descriptive_characterisation` / `mixed` / `unclear`),
`follow_up_examples` (named examples that only emerged after a follow-up
probe, kept distinct from the opening answer), `shared_space_vector`.

**What this is not**: a ranked free-list (`response_rank` is not a
free-listing rank — see that field's own note above) and not a
representative-population estimate (n=25, a convenience-sampled
interview corpus — exploratory prototype evidence about what a
spontaneous first association can look like geometrically, nothing
stronger).

## Categorical Facets Available for Analysis

All fields below live directly in `embedding_space.jsonl` and every
`visualization_{pca,umap,tsne}_3d.jsonl` file — no extra join needed for
these. `null`/absent where not applicable, never a fabricated default.

- **`source_dataset`** (8 values, table above) — the coarsest split.
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
- **`response_rank`** — interviews only; `null` elsewhere. A 1-indexed
  extraction-order counter (chunk order, then extraction order within a
  chunk) over every pooled item in a document, computed *before* the
  dedup/short-fragment filter. **Not a free-listing rank and not a
  cognitive-salience proxy** — checked directly
  (`thesis_corpus.audit_free_listing_rank`) and found unreliable even at
  its narrowest, most favorable reading: only 11/26 interviews have
  `response_rank == 1` land on the participant's own first claim rather
  than, e.g., the interviewer's own question text (`unspecified`
  attribution). The interview protocol itself isn't a multi-item
  free-listing task either (`sec:interview_protocol`,
  04\_Appendix/3\_Appendix.tex) — one prompt elicits a single example,
  followed by a second prompt and probes asking the participant to justify
  *that same example*, not to list more items. Retained here purely as
  extraction-order provenance. The actual interview-side geometric
  evidence comes from a separate, manually-reviewed initial-exemplar
  prototype layer instead — see "Interview Initial-Exemplar Prototype
  Layer" below.
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
  equal-corpus-weighted grand centroid sits 3.98 shared-space units from the
  plain unweighted one. Use `thesis_corpus.balanced_analysis` for any
  quantitative (not visualization) claim about the corpus as a whole —
  `weighted_centroid()` / `per_corpus_centroids()` as reusable functions, or
  the pre-written `processed/shared_space/literature_balanced_sample.jsonl`
  (2,500 points, stratified by document) as a drop-in literature subset.
- **MIVILUDES = 2 documents**: treat as one influential operational
  framework, not a representative sample of French state framing broadly.
- **Language asymmetry**: MIVILUDES is ~100% French; both reference
  point-sets are English-only. Fixed: all 914 MIVILUDES expressions
  (pre-filter) were machine-translated to English (`qwen3:4b`) and
  re-embedded (`bge-m3`); the 732 surviving points now pool the English
  translation as `label`/vector, with French in `label_fr`. FR/EN raw
  cosine similarity across all 914: mean 0.90, median 0.91, p10 0.82, min
  0.55; 12/914 (1.3%) fall under the 0.70 hand-inspection threshold, all
  confirmed accurate on inspection (short-phrase cross-lingual noise, not
  mistranslation). Separately, 5/914 (0.55%) translations are
  explanation/meta-commentary rather than a bare translation (caught by an
  anomalous English/French word-count ratio, e.g. 38.7x); 3 of those 5
  survived into the final 732 — left as documented residual noise rather
  than hand-corrected.
- **Interview sample**: convenience-sampled through the researcher's own
  network (one response excluded for researcher-influence bias; another
  named the researcher's own academic programme a "cult"). Treat as
  exploratory prototype data, not a representative sample of lay usage.

## What's NOT Yet Done

A geometric-analysis toolkit exists (`thesis_corpus.analyze_global_structure`,
`analyze_cluster_structure`, `audit_free_listing_rank`,
`propose_initial_exemplars`, `build_interview_prototype_layer`,
`analyze_initial_exemplars`, `analyze_criterion_neighbours`,
`analyze_emergent_entities`, `generate_focused_projections`,
`generate_figures`, `generate_geometric_draft_report` — see that package's
own docstrings, and `processed/analysis/<run-id>/` for output), covering
centroids/dispersion, k-NN/silhouette cluster structure, criterion-neighbour
distances (French-primary, English-shared-space-sensitivity, and an
optional raw-embedding-cosine diagnostic — three representations, never
merged), emergent-entity provenance, the interview initial-exemplar
prototype layer (see above), an equal-size-controlled comparison across
the three reference vocabularies (controls for their very different
sizes — 3,000/1,500/195 — not evidence they're otherwise interchangeable),
and both whole-space and ~23 small, focused 2-D UMAP/PCA projections. The
interview-side manual review is done: 25/26 interviews resolved into
`interview_prototypes.jsonl`, 1 unavailable (see above).

Since the `20260907-002832` run, one data correction and five toolkit
additions landed:

- **Data correction**: one interview expression
  (`b1-aug12-1231`, chunk 3 — an interviewer's paraphrased question,
  "how it's perceived as a whole") was misattributed `attribution=participant`
  instead of `unspecified`; corrected, and the shared space + interview
  prototype layer rebuilt. Verified to produce byte-identical
  `shared_space_vector`s for all 44,520 points — this correction changed
  only that one metadata field, nothing geometric.
- **Epistemic-status filtering**: `analyze_global_structure.py` and
  `analyze_cluster_structure.py` gained `--epistemic-status-filter
  {all,asserted_qualified}`, restricting the three expression corpora to
  `asserted`/`qualified` statements only (excluding
  `contested`/`negated`/`speculative`) — a robustness check on whether
  e.g. a negated "X is NOT a cult" sitting close to an asserted "X is a
  cult" was distorting proximity results. Status-less datasets
  (reference vocabularies, `miviludes_criteria`, `emergent_entities`) are
  provably unaffected by this filter (verified byte-identical between
  filter settings).
- **Dispersion and normalized separation**: `per_source_centroids_dispersion.csv`
  now reports median/p10/p90 alongside mean, for both epistemic slices;
  new `source_centroid_separation_normalized.csv` (centroid distance
  relative to within-source spread — an interpretive baseline for "is
  this distance large or small") and `equal_n_dispersion.csv`
  (bootstrapped, equal-sized-sample dispersion, kept file-separate from
  the ratio table).
- **Enriched nearest-centroid-expressions**: `source_centroid_nearest_expressions.csv`
  now carries full provenance (document/chunk/pooled/occurrence keys,
  attribution, claim mode, epistemic status, and the raw archive's
  `context_window`, resolved via a strict, occurrence-aware,
  fail-loud join) plus a diversity-audit companion file.
- **Criterion-neighbour composition**: new `criterion_equal_n_neighbour_composition.csv`
  in `analyze_criterion_neighbours.py` — equal-n, bootstrapped source
  composition of each of the 17 criteria's nearest neighbours (k=10,20),
  plus a candidate heatmap figure.
- **Shared discourse anchors**: new `shared_anchor_nearest_expressions.csv`
  in `analyze_emergent_entities.py` — for every entity with
  `provenance_category=="shared_all_3"` (dynamically derived, currently
  9), 5 nearest expressions within each corpus separately (primary) plus
  one exploratory equal-n combined retrieval (appendix-only).

**A second, manual-review-driven correction** followed: the shared-anchor
qualitative-retrieval output surfaced one interview expression
(`document_id=b2-aug13-1840`, `chunk_index=2`, `embedding_text="Scientology
is also a cult"`) tagged `epistemic_status=negated`; manual reading of its
full `context_window` (`"Or maybe Scientology is also a cult, maybe? Not
sure about it."`) found this mislabeled — the participant neither
affirms nor denies the claim, they express uncertainty about it, so
`speculative` is the correct tag under this project's epistemic-status
scheme, not `negated`. Corrected by hand; shared space and interview
prototype layer rebuilt again, again verified byte-identical
`shared_space_vector`s for all 44,520 points. Since both `negated` and
`speculative` are excluded by the `asserted_qualified` filter, this
correction changed no filtered-pool membership and left every filtered
numeric output byte-identical to before it.

**All 9 modules have now been run end-to-end** against the twice-corrected
44,520-point space under retained run `20260908-175344` (baseline,
unfiltered) — superseding `20260908-170432` (which itself superseded
`20260907-002832`), all kept as superseded historical record, not deleted
or reused. A second retained run, `20260908-180406`, holds
`analyze_global_structure`/`analyze_cluster_structure` only, under
`--epistemic-status-filter asserted_qualified`, superseding `20260908-171742`.
(A third, incomplete run-id, `20260908-153701`, was created mid-session
before the toolkit additions above were finished, never treated as
authoritative, and has been deleted rather than kept as historical
record, since it was never a complete or valid run in the first place.)
What's still outstanding:

- No run's output has been interpreted or written into Results.tex yet —
  the toolkit produces tables/figures/a neutral draft report, not a
  finished analysis.
- Rank-based prototype analysis is retired outright, not deferred:
  `response_rank` cannot support a salience/prototype claim (see that
  field's own note above) — this was a design finding, not unfinished work.
