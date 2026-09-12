# Prompt for the 4-cell coercive-control list

Run this on the Ollama machine, then save the model's **raw output verbatim** into this directory
as a `.txt` file (add `prompt:` and `model:` header lines at the top, same convention as
`dictionaries/non-religious-groups/`). The parser
(`parse_manual_secular_groups.parse_manual_group_list_with_cells`) reads the `Block A`..`Block D`
headers as the experimental cell and keeps each entry's description.

**Do not edit the model's cell assignments before saving.** If it puts something in a cell you
disagree with, leave it — the analysis reports cross-cell conflicts and the circularity tag
separately, and silently fixing them destroys the control.

---

## The prompt

> Give four separate labelled blocks of 25 organizations each, 100 total.
>
> Format every entry as `Name – one-line description of its internal control dynamic`, one entry
> per line. Use plain organization names: no leading "The", three words or fewer where possible.
> No numbering, no markdown tables, no commentary before or after the blocks.
>
> **Block A — organizations widely known as cults.**
>
> **Block B — ordinary-sounding organizations credibly described as coercive or high-control, that
> are NOT commonly called cults.** Examples of the kind of thing meant: multi-level-marketing
> companies, fitness franchises, fraternities, door-to-door sales bootcamps, talent and sports
> academies, secular political sects. Exclude anything commonly labelled a cult.
>
> **Block C — ordinary organizations with no coercion claim at all.**
>
> **Block D — organizations that are highly demanding but not considered coercive.** Examples of
> the kind of thing meant: military academies, monasteries, elite conservatories, surgical
> residencies.

---

## Why these four cells

The comparison that matters is **B vs C**. Both are ordinary-sounding organizations from the same
generator with the same naming register; only B is described as coercive. If the embedding space can
represent coercive control, B should sit closer to the coercion vocabulary than C does.

- **A** is the circularity control. Famous cult names sit inside cult discourse by construction, so
  a high score for A proves nothing. Reported, excluded from the headline. Asking for A explicitly
  also satisfies the model's pull toward famous cases inside a labelled cell, instead of letting it
  contaminate B.
- **D** separates *demanding* from *coercive*. A monastery and a surgical residency are gruelling
  without being manipulative. If D scores like B, the space is tracking intensity, not coercion.

Full pre-registered interpretation table:
[`Analysis/11_Psychological_Subjection.md`](../../../../Analysis/11_Psychological_Subjection.md).

## Then, on the Ollama machine

```
python -m thesis_corpus.embed_manual_secular_groups --cells \
  --source-dir dictionaries/coercive-control-groups \
  --out-dir processed/analysis_raw/coercive_control \
  --out-name coercive_control_groups.jsonl
```

That writes two files — names and descriptions embedded separately. Copy **both** back to the Mac
at the same relative paths, then run:

```
python -m thesis_corpus.analyze_coercive_control_groups
```
