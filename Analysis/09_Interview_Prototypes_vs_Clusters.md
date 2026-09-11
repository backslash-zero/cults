# Do people's prototype examples land in literature's well-covered regions, or its gaps?

Data inventory: [`00_Available_Data.md`](00_Available_Data.md). Index: [`Results_Draft.md`](Results_Draft.md).

A sharper version of the folk-vs-expert-concept finding already in the index's Key Findings:
that finding used every interview segment, including filler. This restricts to exactly the 25
*prototype exemplars* — the interview protocol's own first free-association answer, hand-reviewed
in `interviews/metadata/initial_exemplars.csv`, and the specific thing the thesis's
`\S Interviews` section frames as accessing lay prototype structure directly (Rosch/Mervis),
not a definition.

## Method

**Raw-vector resolution by fuzzy text match, not the old key-based join.**
`interview_prototypes.jsonl` (built by `build_interview_prototype_layer.py`) only ever stores a
PCA'd `shared_space_vector`, resolved against an older archive
(`processed/interviews/criterion_expressions.jsonl`), not this session's exhaustive
`processed/interviews_full/interviews/run_20260912/`. To stay self-consistent with every other
raw-space analysis this session, each prototype's raw vector is instead resolved by fuzzy text
match (`difflib.SequenceMatcher`) against that document's own segments in the new archive.

**A real bug caught by spot-checking before trusting the result** (per this session's standing
practice): the first version matched only against the CSV's composite
`transcript_initial_exemplar_text` field (a hand-composed, "..."-joined multi-part quote) —
this gave a false, low-confidence match for `b2-aug13-1832` (ratio 0.43) even though its
precise `source_expression_label` field is an *exact* match (ratio 1.00) to a real segment in
the new archive. Fixed by trying both text fields per candidate and keeping the best match; all
25 now resolve at ratio ≥0.78 (see `thesis_corpus/analyze_interview_prototypes.py`'s
`best_match_in_document` docstring for the full account). This changed the headline number from
15/25 to 13/25 landing in a gap cluster — the fix mattered, not just a formality.

For each resolved prototype: nearest of the 59 literature clusters (`06_Literature_Clusters.md`)
to its raw vector, and whether that cluster is one of the 15 flagged entity gaps.

Code: `thesis_corpus/analyze_interview_prototypes.py`.

## Result

**13 of 25 (52%) prototype exemplars land in a gap cluster — roughly double the corpus-wide
base rate of 15/59 (25.4%) clusters.** Not a subtle effect: if prototypes landed in gap and
non-gap clusters at the same rate literature clusters do overall, you'd expect ~6–7 of 25 in a
gap cluster, not 13.

**A striking concentration in one particular non-gap cluster**: 9 of the 25 prototypes' nearest
cluster is **cluster 58** ("The major criterion of the concept of cult is its oppositional
nature: A cult is a group that has beliefs and/or practices that are counter to those of the
dominant culture," n=54, cosine similarity 0.775 to its own nearest entity — *not* itself a
flagged gap). Lay prototype answers gravitate toward this one abstract, oppositional-definition
register more than toward any specific named-group cluster.

| Document | Prototype text | Match ratio | Nearest cluster | Gap? | Cosine sim |
|---|---|---|---|---|---|
| b1-aug05-1650 | it's a local community of people growing local tomatoes. | 0.89 | 49 | **gap** | 0.660 |
| b1-aug06-1955 | AI cult | 0.80 | 58 |  | 0.632 |
| b1-aug08-1420 | brainwashing, suppressive, abusive | 0.80 | 40 |  | 0.757 |
| b1-aug08-2128 | Something suspicious, something dangerous, something not normal | 0.80 | 50 | **gap** | 0.741 |
| b1-aug08-2137 | My first thing that comes is probably the Ku Klux Klan | 1.00 | 45 | **gap** | 0.639 |
| b1-aug08-2145 | where do you draw the line — different societies have very different... | 0.95 | 58 |  | 0.795 |
| b1-aug09-2048 | smaller church denominations with a charismatic leader | 1.00 | 58 |  | 0.739 |
| b1-aug09-2124 | it's not really a belief system | 0.80 | 56 | **gap** | 0.637 |
| b1-aug11-2011 | To my mind comes the image of a group of people that decide... | 0.91 | 58 |  | 0.713 |
| b1-aug12-1231 | some kind of new wave band — Blue Öyster Cult | 0.82 | 57 |  | 0.632 |
| b1-aug13-1342 | kind of privileged people — that they have the privilege to... | 1.00 | 53 | **gap** | 0.678 |
| b2-aug13-1832 | cult films are usually films that were not very economically successful | 0.80 | 58 |  | 0.718 |
| b2-aug13-1840 | cult — if we translate it properly in Italian, should be "la setta" | 0.95 | 58 |  | 0.697 |
| b3-aug16-1514 | Charles Manson — and CIA. | 1.00 | 21 | **gap** | 0.557 |
| b3-aug16-1517 | When you say the word cult, it comes to my mind — religious... | 1.00 | 58 |  | 0.754 |
| b3-aug18-1645 | Gourou. Groupe. Un groupe qui suit une doctrine. | 0.80 | 49 | **gap** | 0.642 |
| b3-aug21-1932 | Illuminati | 0.95 | 49 | **gap** | 0.757 |
| b3-aug22-2057 | Manipulation, groupe, idéologie. | 1.00 | 43 |  | 0.756 |
| b3-aug22-2101 | groupes fermés qui endoctrinent des gens en échange d'argent | 1.00 | 50 | **gap** | 0.669 |
| ig-02 | organize special camps, give scholarship | 0.80 | 58 |  | 0.666 |
| ig-03 | Weird people coming together to do weird things | 0.99 | 49 | **gap** | 0.766 |
| ig-04 | There's often someone dressed in white, pope, orthodox ppl | 1.00 | 55 | **gap** | 0.648 |
| ig-05 | enclosed group with one strange thing in the full center of attention | 0.96 | 50 | **gap** | 0.695 |
| ig-06 | attempting to deviate from academic norms through "breaking boundaries" | 1.00 | 53 | **gap** | 0.703 |
| ig-07 | groups of people with some commonly considered 'alternative' beliefs | 0.89 | 58 |  | 0.719 |

Full data: [`data/prototype_vs_clusters.csv`](data/prototype_vs_clusters.csv).

## Reading this

The pattern in *which* gap clusters prototypes land in matches
`06_Literature_Clusters.md`'s own typology directly: cluster 49 (heterogeneous
fragments, 4 prototypes), cluster 50 (leadership-behaviour description, 3 prototypes), cluster
53 (process theory, 2 prototypes), 45/56/21 (theoretical/statistical/negation registers, 1
each). Lay prototype answers — the theoretically privileged, spontaneous "what comes to mind
first" exemplar — skew toward exactly the abstract/attitudinal/definitional territory the
literature itself has the least specific vocabulary for, sharpening rather than complicating
the existing folk-vs-expert-concept finding: it isn't just that ordinary interview speech in
general drifts generic, it's that the *specific, theoretically-motivated* answer people give
first does too.
