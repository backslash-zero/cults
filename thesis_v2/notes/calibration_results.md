# Calibration results, run 20261119 (facts for the proximity section)

Source: thesis/corpus/processed/calibration/run_20261119/report/. bge-m3 via Ollama, 249 pairs (all kept), 269 unique texts. Pre-registration: notes/PREREG.md (frozen, commit e157e3a).

## Medians, cult set (choir in brackets where it differs)
- near-identical 0.984 (top anchor)
- singular vs plural 0.977
- paraphrase 0.900
- register shift 0.883
- label swap 0.881 (choir rows 0.817)
- translation EN to FR 0.865 (choir 0.926)
- negation 0.837
- opposite stance 0.843
- same topic, different claim 0.626
- unrelated 0.517 (bottom anchor; cosines stay well above 0)
- name vs description 0.367 (choir 0.601); matched 0.404 vs mismatched 0.307
- single words (weak): cult/cults 0.93, cult/sect and cult/secte lower (0.59 to 0.76)

## Expectations (PREREG)
- E1 anchors in order: holds
- E2 negation closer than paraphrase: fails (0.837 vs 0.900); negation still far above a different claim
- E3 opposite stance closer than a different claim: holds (20 of 20 triplets)
- E4 translation near paraphrase: roughly holds; lower for cult sentences than choir sentences (hint only, 15 pairs)
- E5 singular and plural near paraphrase: holds, closer still (about near-identical)
- E6 matched name above mismatched: holds, small margin; name vs description sits below unrelated pairs
- E7 choir follows the same ordering: holds; main difference is name vs description

## Triplets
- Every edited variant (paraphrase, stance, negation, translation, number) is closer to its base than the base's same-topic different-claim partner: 98 of 98.

## Masked-entity test (pool 430 participant segments, query = bare name)
- Unmasked precision at k: 0.71 to 1.0 for Manson family, Raëlism, Scientology, Jehovah's Witnesses, Satanists, Bike polo club; chance 0.007 to 0.03.
- Masked precision: 0 for most entities; 0.29 Jehovah's Witnesses, 0.27 Religion in general, 0.25 Bike polo club, 0.14 Manson family, 0.13 Catholicism.
- Masked mean rank 50 to 170 of 430 for most (chance about 215): some content beyond the name remains. Exception: Ku Klux Klan 291.
- Reading (as pre-registered): retrieval rested mainly on the name; entity results are read from the quotes.

## To fix before final figures
- Legend overlaps the bottom row; Matplotlib vert= deprecation warning.
