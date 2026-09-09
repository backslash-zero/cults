# v2 pilot acceptance report (arm_qwen3-4b / judge_qwen3-8b)
Basis: model-judged (second-model verdicts), not human review. Generated 2026-09-09T00:42:18.604861+00:00.

## Checks (random stratum unless stated)

- PASS — faithful >= 0.98
- PASS — issue_none >= 0.85
- PASS — off_topic <= 0.05
- PASS — overlong+multiple <= 0.05
- PASS — zero heading/name/corrupt/truncated/contextless
- PASS — attribution_correct >= 0.90
- PASS — claim_mode_correct >= 0.90
- PASS — epistemic_status_correct >= 0.90
- PASS — model failure rate <= 0.03 (extractor)
- PASS — judge failure rate <= 0.03
- PASS — hard-zero conditions
- PASS — no v1 failure text reappears in final retained (forced chunks)

**All checks pass: True**

## Precision

| metric | random | forced |
|---|---|---|
| screen_retained | 21 | 11 |
| judged_ok | 21 | 11 |
| judge_accepted | 21 | 10 |
| judge_acceptance_rate | 1.0 | 0.9091 |
| faithful_rate | 1.0 | 1.0 |
| issue_none_rate | 1.0 | 1.0 |
| off_topic_rate | 0.0 | 0.0 |
| overlong_plus_multiple_rate | 0.0 | 0.0 |
| heading_name_corrupt_truncated_contextless | 0 | 0 |
| attribution_correct_rate | 1.0 | 1.0 |
| claim_mode_correct_rate | 1.0 | 1.0 |
| epistemic_status_correct_rate | 1.0 | 1.0 |
| better_span_pointed | 0 | 0 |

## Yield

- chunks called: 28; v1 items: 310 (11.0714/chunk)
- emitted 78 → screen retained 32 (rejected 46: {'validation_flag_false': 36, 'dangling_boundary': 5, 'short_fragment_no_referent': 3, 'not_verbatim': 2})
- judge rejected 1 ({'none': 1}), judge failed 0
- **final retained 31** = 1.1071/chunk = 0.1 of v1; chunks with zero final: 8

## Missed material and disagreements (listed for inspection)

- better-span pointers: 0
- label disagreements: 0
- judge rejections: 1
  - davis-and-hankins-2002-new-religious-movements-and-religious-liberty-in-america:105#1: "it is this same acknowledgement that Bush makes" — issue=none; false=cult_relevant; note=The expression associates Bush's stance with a prior idea about religious freedom, but does not directly relate to cults or religious movements.

## Forced-regression chunks

- lalich-2004-bounded-choicetrue-believers:213 — ligature substitution ('di Y cult') -> integrity_ligature_substitution
  - v1 failure text "di Y cult" reappeared: False
  - final retained (1): a very controlling environment existed in both as well.
  - screen codes: ['validation_flag_false']; judge rejected: []
- 2008-the-oxford-handbook-of-new-religiou:194 — detached accent ('Hervieu-Le ´ger', flag) + off-topic 'cult apologist' sentence
  - v1 failure text "would be called a 'cult apologist'" reappeared: False
  - final retained (2): Type II reports have also influenced Belgium || French anti-cult law of May 30, 2001
- dawson-2009-cults-and-new-religious-move:133 — model truncation of a complete quote
  - v1 failure text "when I really mean something that can be" reappeared: False
  - final retained (0): 
- dawson-2009-cults-and-new-religious-move:135 — model rewrite ("self 'actualization'")
  - v1 failure text "self 'actualization'" reappeared: False
  - final retained (1): No radical departers I have studied felt committed to a value system at the time
- davis-and-hankins-2002-new-religious-mov:105 — contextless acknowledgement ('Well, perhaps to some extent.')
  - v1 failure text "Well, perhaps to some extent." reappeared: False
  - final retained (0): 
  - screen codes: []; judge rejected: ['it is this same acknowledgement that Bush makes']
- melton-2014-encyclopedic-handbook-of-cul:10 — heading extracted as expression ('The Course of Growth.')
  - v1 failure text "The Course of Growth." reappeared: False
  - final retained (2): Over ninety percent of
those who join an alternative religion leave it within a  || Over one million people
took the basic course in Transcendental Meditation, but 
- tomkins-2010-the-clapham-sect-how-wilber:131 — relevance boundary (historical 'sect')
  - v1 failure text "she did not like the Clapham sect" reappeared: False
  - final retained (1): the Clapham sect
- clarke-2004-encyclopedia-of-new-religiou:585 — encyclopedia entry-style short fragments
  - v1 failure text "regarded as unduly authoritarian and liable to abuse" reappeared: False
  - final retained (2): British House Church Movement || International Church of Christ
  - screen codes: ['short_fragment_no_referent']; judge rejected: []
- lalich-2004-bounded-choicetrue-believers:239 — overlong autobiographical / multiple claims
  - v1 failure text "I was assigned leadership of the Party's publishing house" reappeared: False
  - final retained (1): The Cadre Ideal: Origins and Development of a Political Cult
- card-2019-archaeology-and-new-religious-:1 — broken-CMap cipher text -> deterministic rejection only
  - v1 failure text "Boliǀia" reappeared: False
  - final retained (0): 
