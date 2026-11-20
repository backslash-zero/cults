# Methodology: inventory for triage

**Scope revised 7 Nov (see the plan).** Settled by the plan: first-generation extraction and shared spaces get a brief mention at most, the dictionary stays as a baseline, interview entities are redone by hand, HDBSCAN and nearest neighbours stay, Voronoi and publication dates go, interviewer segments are excluded. The sections on secular groups, coercive control and the aspect matrix below no longer apply as planned work. Earlier tests get one short section in Results.

## Triage decisions (18 Nov)

- 1 Pipeline figure, redrawn for raw embeddings: keep, but make it last, as a summary of the finished chapter.
- 2 Stance that the space is model-mediated, not a map of meaning: keep.
- 3 Footnote that secte translates to cult: keep.
- 4 The two original expectations: keep.
- 5 Query list and title-only matching: keep.
- 6 Why NRM is not searched alone: keep.
- 7 Screening steps: keep.
- 8 Ranking by citations and query presence, stratified by discipline: keep.
- 9 Docling extraction and the manual check: keep.
- 10 Exclusion lists: keep.
- 11 The Que dit la loi page: dropped. It is not in the corpus (only a bib entry, never cited).
- 12 The 108 MIVILUDES expressions as a figure in the text: keep.
- MIVILUDES corpus, checked: the activity report gives 94 expressions, the Comment identifier page gives 14 plus the 17 hand-copied criteria.

## Entity and filler review (19 Nov)

- 45 entities, no merges. 15 enter the masked test (3+ segments in 2+ interviews), including the broad category Religion in general (55 segments, 16 interviews).
- Entities named in only one interview (Druze, Moonies, Children of God, MK Ultra, Tomato cult, AI cult) stay in the map but not in the masked test.
- Filler rule accepted: 48 participant segments with no content word removed, 430 kept. Three borderline ones stay removed (no answer given): How do we all do the same thing?, Oh, okay, yeah, not good, They sell it that way, okay.

## Decision on style and authorship (18 Nov)

- No corrections for style or authorship (no register pairs, no subtracting corpus averages, no leave-one-interview-out).
- Reason, Celestin's: a custom patch would bring in its own biases, and since this is also a study of language differences, differences in style count as meaningful. They may express similar details, but the analysis does not go that deep.
- To explain in two places: Introduction (Language as mediation tool, next to proxy presumption) and What proximity measures (moved to the Conceptual Background on 20 Nov).
- Controls that stay, because they are part of the existing plan: topic (calibration pairs, the word cult check), names (masked test, conduct descriptions), length (ranks, not cosines), the dictionary baseline, the layered map, and prevalence counted per interview.

Facts only, no old prose. Mark each bullet K (keep), C (cut) or ? (unsure), and add what is missing.
Budget: about 5 pages, 1,500 words (What proximity measures moved to the Conceptual Background on 20 Nov, with its 400 words). The old chapter was 14,243 words, so most of it has to go.

## Proposed order and budget (change freely)

| Section | Words |
|---|---|
| Design overview: three corpora, one space (one figure) | 150 |
| Positionality and expectations | 150 |
| Interviews | 350 |
| Scholarly corpus | 250 |
| MIVILUDES | 100 |
| Extraction and embedding | 300 |
| Analysis: layers, clusters, controls | 200 |

## What the old draft covered

### Pipeline figure
- Exists as TikZ: corpora, extraction, bge-m3, shared space, analysis.
- Out of date: it runs through the shared space, while the later analysis uses raw embeddings.

### Precedents and epistemic stance
- Bullet notes only.
- One stance worth deciding on: the space is model-mediated, not a map of meaning. Relational geometry is tested, and coordinates are not treated as features.
- Footnote: French "secte" translates to "cult", so "sectarian drift" means cult-like harm.

### Expectations (two lines)
- Lay views of cults are more tied to religion than scholarly views are.
- The MIVILUDES concept goes beyond the popular idea of a cult.
- Decide: are these still your expectations? Were they written before you saw results?

### Scholarly corpus (fully written, far too long)
- Source: OpenAlex API.
- Fixed keyword queries: cult, sect, new religious movements, "extreme beliefs", plus "sectarian drifts" as a bridge to MIVILUDES.
- Matching is on titles only, except "sectarian drifts" (title and abstract).
- "NRM" is not searched alone because of acronym collisions.
- Screening: dedup, type, field, subfield, topic, title patterns, publication dates.
- Ranking: citations plus presence across queries, stratified by discipline.
- Final: 68 documents (33 articles, 23 books, 12 chapters), 2001 to 2024. 57 have usable full text.
- Text extraction: Docling, then a manual check.
- The 11 missing documents are an access bias: paywalled articles and out-of-print books.
- Exclusion lists can move to the appendix.

### MIVILUDES (thin)
- Two documents: the 2022–2024 activity report and "Comment identifier une dérive sectaire ?".
- 17 structured criteria, French and English.
- 108 extracted expressions.
- The draft also cites a page on the law ("Que dit la loi"). Check whether it is used in the corpus; the data inventory lists only the two documents above.

### Interviews (bullets only, the most important gap)
- Google Recorder for transcription, then Claude for clean-up, then your manual pass against the audio.
- Metadata: age, gender, nationality, language, residence, exemplars in order of mention.
- Also run through Instagram stories.
- Still to explain: why this method, what the questions were, and how people were recruited.

### Extraction, embedding, shared space (long, mostly obsolete)
- First generation: qwen3 extraction, 39k expressions. Superseded.
- v2: verbatim spans, deterministic screen, LLM judge, text-integrity audit. 5,741 literature expressions, 108 MIVILUDES.
- Three reference vocabularies (WordNet backbone, structural concepts, ConceptNet). Decide whether any are needed now.
- Emergent entities v1 and v2.
- Shared spaces v1 to v3. Step 01 marks v3 as reference only, replaced by raw bge-m3 vectors.

### Analysis (written, partly relevant)
- Centroids and typicality, nearest neighbours, HDBSCAN, Voronoi, publication-date effects.
- Keep only what Results will actually use. Decide this after Tue 10.

### Known limitations (written)
- Corpus imbalance, thin MIVILUDES base, language asymmetry, convenience sample recruited through your own network, two corrected metadata records.

## Missing from the old draft

- The exhaustive interview run `run_20260912`. It is the input for the new analysis.
- The move to raw embeddings.
- Secular-group and coercive-control tests, and the retractions they caused.
- What proximity measures.
- The layered analysis.

## Numbers in the old draft that are now wrong

| Old figure | Current figure |
|---|---|
| 26 interviews | 28 |
| 39,236 literature expressions | 5,741 (v2) |
| Pooled counts 35,621 / 732 / 204 | Refer to spaces no longer used |

## New facts checked today (`run_20260912`)

- 669 segments in total.
- Participant and cult-relevant: 459. These would feed the aspect clustering.
- Interviewer and cult-relevant: 159. Proposal: exclude them from aspects.
- Across the 28 interviews, participant segments are uneven: one interview has 1, a few have 2 to 6, the largest have 43 and 47.
- 94% of participant segments are labelled relevant, so the label barely filters anything.

Consequences to decide on:
- A few talkative people could dominate the clusters, which is why prevalence counts interviews, not segments.
- 459 short sentences is a small base for clustering. Aspects may come out coarse.

## Bias candidates for the log (your call)

- Convenience sample from your own network, and interviews conducted by you.
- 22 English and 6 French, mostly living in Germany.
- 6 of the 28 were Instagram text answers, a different format from in-person talk.
- An LLM cleaned the transcripts and extracted the expressions.
- Most of the first draft was AI-generated, which may have shaped your framing.
- You chose the entity lists by hand, and you also interpret the clusters.
- Your long fascination with cults, and with spatial models, as described in your meditation.
- Results seen before this rewrite, so expectations are no longer blind.
- Access bias in the literature (57 of 68 full texts).
