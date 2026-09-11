# Shared space v3

Data inventory: [`00_Available_Data.md`](00_Available_Data.md). Index: [`Results_Draft.md`](Results_Draft.md).

Kept for reference; the corpus-centroids step ([`02_Corpus_Centroids.md`](02_Corpus_Centroids.md))
no longer reads this space (it works on raw embeddings instead — see that file's own writeup).
Still a valid, separately-usable pooled PCA space if a later step wants one (e.g. a whole-space
UMAP/PCA overview plot spanning every corpus at once, where *some* shared low-D basis is
unavoidable).

One pooled, PCA-aligned coordinate system (k=408, 95% variance): literature + MIVILUDES
(v2, `run_20260910`) plus interviews from the new **exhaustive** pipeline (`run_20260912`,
every segment embedded) in place of the old archived selective run. Everything below is
geometrically comparable. 15,014 points total. (`shared_space_v2` is untouched, still
interview-archived.)

| source_dataset | n | role |
|---|---|---|
| literature | 5,741 | expression |
| interviews | 668 | expression |
| miviludes | 108 | expression |
| miviludes_criteria | 17 | expression |
| concept_backbone | 3,000 | reference |
| structural_concepts | 1,500 | reference |
| conceptnet_concepts | 195 | reference |
| emergent_entities | 3,785 | emergent |

**Subsets available within it:**

| Source | Dimension | Breakdown |
|---|---|---|
| interviews | batch | B1: 11 · B3: 7 · Instagram: 6 · B2: 2 · B4: 2 |
| interviews | language | English: 22 · French: 6 |
| interviews | method | in person: 22 · Instagram (text): 6 |
| interviews | location | Germany: 16 · France: 5(+2) · Slovakia/US/Spain: 1 each · unknown: 1 |
| interviews | gender | male: 14 · female: 11 · other/woman/redacted: 1 each |
| interviews | age | 26–97 |
| interviews | speaker (`attribution`) | interviewer vs. participant, every point tagged |
| interviews | `cult_relevant` | labeled on every point, never filtered out |
| literature | type | article: 33 · book: 23 · chapter: 12 |
| literature | year | 2001–2024 |
| miviludes | criterion | 17 named criteria, each its own point |
| concept_backbone | — | topic-neutral WordNet vocabulary, independent yardstick |
| structural_concepts | — | corpus-derived generic vocabulary |
| conceptnet_concepts | — | ConceptNet associative expansion of the above, hand-pruned |
| emergent_entities | per-corpus mentions | named entities mentioned ≥3× in any one corpus |
