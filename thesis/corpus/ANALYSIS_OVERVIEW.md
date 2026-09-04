# Analysis Overview

A reference snapshot of the shared cross-corpus embedding space, for planning
the geometrical analysis. Reflects the pipeline as of the `build_shared_space`
run that produced `processed/shared_space/embedding_space.jsonl` (46,648
points, k=396). Regenerate the numbers below after any pipeline rerun — they
are not guaranteed to stay in sync automatically.

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

One shared 396-dimensional PCA space (95.0% cumulative variance), pooled from
six sources, 46,648 points total:

| `source_dataset` | Points | Key format | Label semantics |
|---|---|---|---|
| `literature` | 39,236 | `document_id:chunk_index` | An extracted expression's short embedding text |
| `miviludes` | 914 | `document_id:chunk_index` | Same, from the 2 MIVILUDES source documents |
| `interviews` | 230 | `document_id:chunk_index` | Same, from 26 interview transcripts |
| `miviludes_criteria` | 17 | `crit-<slug>` | French criterion text (`label`); English translation as `label_en`, display-only — not a separate point |
| `concept_backbone` | 3,000 | WordNet ILI id (e.g. `i71809`) | The concept's primary English lemma |
| `entity_anchors` | 3,251 | normalized anchor text | The anchor text itself (e.g. "scientology") |

## Categorical Facets Available for Analysis

All fields below live directly in `embedding_space.jsonl` and every
`visualization_{pca,umap,tsne}_3d.jsonl` file — no extra join needed for
these. `null`/absent where not applicable, never a fabricated default.

- **`source_dataset`** (6 values, table above) — the coarsest split.
- **`attribution`** — corpus-expression points only (`literature`,
  `miviludes`, `interviews`); `null` for the other three. Values: `author`,
  `cited_author`, `participant`, `institution`, `journalist`, `unspecified`.
  For interviews specifically: 210 `participant` / 20 `unspecified` (this is
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
- **`entity_anchors`** (the point-set, not a per-point field) — 3,251 named
  entities/dimensions mentioned ≥3 times across all corpora, each with its
  own position in the shared space. Not joined onto corpus-expression
  points as a list field; compare by proximity instead. Top by mention
  count: Scientology (3,867), charismatic leader (3,417), NRMs (1,279),
  cults (688), Heaven's Gate (577), new religious movements (529), cult
  (426), new age (387), new religions (332), Unification Church (273),
  Jehovah's Witnesses (233), sect (205), brainwashing (204).
- **`mention_distribution`** — entity-anchor points only; `null` elsewhere.
  A per-corpus mention count, e.g.
  `{"literature": 3683, "miviludes": 130, "interviews": 54}` for
  "scientology" — provenance metadata, not used in the PCA fit (still one
  point per anchor either way). Lets an anchor mentioned near-exclusively in
  one epistemology (e.g. "NRMs", "Heaven's Gate", "Unification Church" — all
  literature-only) be told apart from one that recurs across all three (e.g.
  "scientology", "cults") — useful for checking whether the vocabulary that
  actually structures the space is epistemology-specific or genuinely
  shared.
- **`miviludes_criteria`** (17 points) — joinable to the official MIVILUDES
  17-criteria list via `key` (`crit-<slug>`). FR/EN translation fidelity
  (raw-embedding cosine, not a shared-space property): mean 0.87, min 0.50
  (`crit-legal-disputes`) — 8/17 pairs under 0.90; the two lowest were
  inspected by hand and are accurate translations, not errors (a
  short-official-phrase cross-lingual embedding effect — worth remembering
  when comparing other short texts, like entity anchors, across languages).

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

## What's NOT Yet Done

- No clustering or distance analysis has been run on the shared space.
- No criterion-centred nearest-neighbour inspection (e.g. which corpus
  expressions sit closest to each of the 17 MIVILUDES criteria).
- No rank-stratified analysis (prototypical core vs. peripheral mentions in
  interviews).
- Results.tex is still a placeholder beyond the qualitative interview-survey
  notes; no geometric finding has been written up yet.
