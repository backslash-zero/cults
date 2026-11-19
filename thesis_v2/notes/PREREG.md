# Pre-registration: calibration and layered analysis

Status: **APPROVED by Celestin on 19 Nov 2026, before the Ollama run. Frozen.** The frozen version is the git commit "Approve and freeze PREREG.md". The file is not edited again; later changes go in a dated "Deviations" section at the end.

## Honest limit

Results of earlier versions of this analysis (steps 02 to 11, the coercive-control test) have been seen. This pre-registration therefore limits analytic freedom, but it is not blind. This is recorded in the bias log.

## Part 1: Calibration (sensitivity profile)

**Input.** `thesis_v2/notes/calibration_set.csv` after curation (rows marked `n` dropped), as hashed in the run manifest. Model: bge-m3 via Ollama, raw 1024-d vectors, cosine similarity.

**What is reported.**
- Distribution of cosine per contrast type and domain (n, min, quartiles, max), and the profile figure.
- Triplets: for each base, whether each edited variant is closer to the base than the base's same-topic, different-claim partner.
- Nothing is turned into a threshold or a noise band. The result is an ordering between two anchors (near-identical at the top, unrelated at the bottom).

**Expectations (accepted by Celestin, 19 Nov).**
- E1. Near-identical > paraphrase > same topic different claim > unrelated, by median.
- E2. Negation pairs sit closer than paraphrases (the model registers negation weakly).
- E3. Opposite-stance pairs sit closer than same-topic different-claim pairs (topic outweighs stance).
- E4. Translation pairs sit near paraphrases.
- E5. Singular and plural pairs sit near paraphrases.
- E6. Matched name–description pairs sit above mismatched ones, with a small margin.
- E7. The choir control shows the same ordering as the cult set.

Whichever way these come out, all are reported. A failed expectation is a finding about the model, not a reason to change the set.

## Part 2: Masked-entity test

**Input.** Participant segments of `run_20260912`, fillers removed (`prepare_interview_segments.py`). Entities from `entity_aliases.csv` with at least 3 segments in at least 2 interviews (15 at the time of writing). Masking: each whole word matching the alias, plus the known first names in `MASK_EXTRA`, replaced by `[name]`. Checked with the `preview` stage before embedding.

**Measure.** Query = bare entity name. Pool = all kept participant segments (430). Precision at k (k = the entity's own segments) and mean rank, unmasked vs masked, against chance (k / pool).

**Reading.**
- Masked precision near chance: retrieval rested on the name. Entity results in Results are then read from the quotes, not from proximity.
- Masked precision well above chance: proximity carries something about the entity beyond its name.
- Reported per entity. With 3 to 11 segments per entity, this is descriptive, and no significance test is run.

## Part 3: Layered analysis (`analyze_layers.py`, not yet written)

**Map.** The interview layer alone sets the map: kept participant segments, raw bge-m3 vectors, HDBSCAN.
- Settings, fixed now: min_cluster_size in {5, 8, 12}, min_samples = None (default), metric = euclidean on L2-normalised vectors (equivalent to cosine ordering).
- A cluster is reported only if it persists across at least 2 of the 3 settings (overlap of members, Jaccard ≥ 0.5).
- Noise points are reported as a share, not reassigned.

**Prevalence.** Counted in interviews, not segments: a cluster is "shared" if it contains segments from at least 5 of the 28 interviews.

**The word "cult".** Each cluster's share of segments containing cult, sect or secte is compared with the overall share (146 of 478). A cluster is flagged if it exists only because of the word: re-cluster with those words masked and check whether it persists.

**Placing the other layers (never re-clustered with the interviews).**
- Sectarian drifts (17 criteria, 108 expressions), scholars (59 literature clusters, via their centroids), dictionary baseline.
- For each item: nearest interview cluster, and its rank among all clusters.
- Both directions are reported: drifts or scholar concepts that no interview cluster reaches, and interview clusters that no drift or scholar concept reaches. "Reaches" = the cluster is that item's nearest.
- Ranks and differences are reported, not raw cosines (length and text-form artifacts).
- Group-level claims (for example, "drifts sit closer to cluster X than scholars do") are tested with a permutation test of the layer labels (10,000 permutations, two-sided, α = 0.05).

**Dictionary as baseline.** If ordinary vocabulary gets as close to a cluster as the drifts do, the drift–cluster proximity is not read as meaningful.

**Not corrected for.** Style and authorship (decision of 18 Nov). Style differences between layers count as meaningful.

**Segmenting.** Groupings that persist across settings are read with their quotes and offered as a suggestion in the Discussion. No test.

## Deviations

(none yet)
