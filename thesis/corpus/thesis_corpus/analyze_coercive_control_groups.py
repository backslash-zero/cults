"""The designed test of whether this embedding space can represent coercive
control at all -- the follow-up Analysis/11_Psychological_Subjection.md
scaffolds and pre-registers.

WHY A DESIGNED TEST RATHER THAN ANOTHER LIST. Analysis/10 established that
comparing an organization's NAME to a criterion's TEXT measures topical
vocabulary overlap, not conduct: the four organizations nearest "great
difficulty for a member to leave said group" were a mountain-biking group, an
Arduino users' group, triathlon clubs and soccer fan clubs -- matched on the
word "group". And Analysis/11 retracted the claim that psychological
subjection discriminates, as a criterion-text-length artifact. So asking an
LLM for "100 coercive groups" and observing that they score highly would
prove nothing: famous cult names sit inside cult discourse by construction,
and any uniform lift is register rather than coercion.

THE DESIGN. Four cells of 25, generated together so they share a generator
and a naming register (see parse_manual_secular_groups.parse_manual_group_list_with_cells
for the source format):

  A  groups already famous as cults        -- circularity control, excluded from the headline
  B  ordinary-SOUNDING organizations credibly described as coercive,
     not commonly called cults             -- should score HIGH if the space works
  C  ordinary organizations, no coercion claim  -- matched control, should score LOW
  D  highly demanding but not coercive
     (military academies, monasteries, conservatories) -- separates "demanding" from "coercive"

Headline test is B vs C on the coercion poles only. Scored twice: on bare
NAMES and on the one-line DESCRIPTIONS of each group's control dynamic. That
contrast is the point -- names carry no conduct information and descriptions
do, so running both converts Analysis/10's limitation from a caveat into a
measurement.

PRE-REGISTERED INTERPRETATION, fixed before the data existed so the result
cannot be read post hoc:
  B > C on descriptions but not names -> the quality is legible in conduct
      language and invisible in names (the expected outcome)
  B > C on both                       -> names carry more than assumed
  B ~ C on both                       -> the space cannot represent coercive
      control; Analysis/11's retraction stands unqualified
  A > everything                      -> confirms only that famous cult names
      sit inside cult discourse
Plus a SPECIFICITY gate: B must not also beat C across the other 16 criteria.
If it beats everything uniformly, the result is register, not coercion --
the exact trap the retracted finding fell into.

Every comparison is reported against the +-0.061 artifact band from
Analysis/08 (the value of prepending "The" to an organization name), which is
the resolution floor of this whole method.

Usage (from thesis/corpus/, after the Windows machine has produced the JSONL):
    python -m thesis_corpus.analyze_coercive_control_groups
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from thesis_corpus import analyze_corpus_centroids as acc
from thesis_corpus import analyze_psychological_subjection as aps
from thesis_corpus import analyze_secular_groups_vs_criteria as asgc
from thesis_corpus import build_shared_space as bss
from thesis_corpus import find_secular_group_mentions as fsgm
from thesis_corpus import geometric_analysis_common as gac

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("thesis_corpus.analyze_coercive_control_groups")

HEADLINE_CELLS = ("B", "C")
COERCION_POLES = ("brainwashing", "coercive persuasion")


def load_cell_groups(path: Path) -> tuple[list[dict], np.ndarray]:
    """Reads embed_manual_secular_groups.py --cells output: one JSON object
    per line with label, cell, description and embedding_vector."""
    rows, vectors = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            rows.append({"label": item["label"], "cell": item.get("cell"),
                         "description": item.get("description", "")})
            vectors.append(item["embedding_vector"])
    return rows, np.array(vectors, dtype=np.float64)


def tag_corpus_membership(labels: list[str], points: list[dict], vectors: np.ndarray) -> list[dict]:
    """The circularity control. A generated name that is ALREADY one of the
    corpus's 3,785 emergent entities is already inside cult discourse, so its
    proximity to coercion vocabulary is not evidence of anything.

    Automatic tagging is necessary but NOT sufficient: it matches on the
    normalized entity key, so it misses aliases, and it misses groups that
    post-date the corpus entirely (NXIVM). `max_cos_to_entity_pool` is
    therefore reported alongside, and rows with a high pool cosine but no
    exact match are the ones to hand-review -- see
    residual_review_candidates()."""
    entity_idx = acc.group_indices(points, "entities_all")
    entity_vectors = vectors[entity_idx]
    entity_keys = {points[i]["key"].lower() for i in entity_idx}

    rows = []
    for label in labels:
        bare = label.lower()
        stripped = bare[4:] if bare.startswith("the ") else bare
        in_corpus = bare in entity_keys or stripped in entity_keys
        rows.append({
            "label": label,
            "in_corpus_entities": in_corpus,
            "matched_form": bare if bare in entity_keys else (stripped if stripped in entity_keys else ""),
        })
    return rows, entity_vectors


def add_pool_proximity(tag_rows: list[dict], group_vectors: np.ndarray,
                       entity_vectors: np.ndarray) -> list[dict]:
    """Attaches each group's maximum cosine to any corpus entity -- the
    alias/near-miss detector the exact-key match cannot provide."""
    for row, vec in zip(tag_rows, group_vectors):
        sims = gac.cosine_similarities(vec, entity_vectors)
        row["max_cos_to_entity_pool"] = round(float(np.max(sims)), 4)
    return tag_rows


def residual_review_candidates(tag_rows: list[dict], threshold: float = 0.85) -> list[dict]:
    """Rows the automatic tag calls novel but that sit very close to some
    corpus entity -- likely aliases or spelling variants, and the only rows
    that need human eyes. Documented as a rule so the hand-review is
    reproducible rather than ad hoc."""
    return [r for r in tag_rows
            if not r["in_corpus_entities"] and r["max_cos_to_entity_pool"] >= threshold]


def cell_vectors(rows: list[dict], vectors: np.ndarray, exclude_in_corpus: set[str] | None = None,
                 name_form_matched_only: bool = False) -> dict[str, np.ndarray]:
    """Group the embedded rows by experimental cell, optionally dropping
    names that are already corpus entities (the circularity control) and
    optionally restricting to the name-form-matched subset (<=3 words, no
    leading "The") that Analysis/08's artifact requires for any cross-set
    comparison."""
    exclude_in_corpus = exclude_in_corpus or set()
    by_cell: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        if row["label"] in exclude_in_corpus:
            continue
        if name_form_matched_only and not asgc.match_baseline_name_form([row["label"]]):
            continue
        by_cell.setdefault(str(row["cell"]), []).append(i)
    return {cell: vectors[idx] for cell, idx in sorted(by_cell.items())}


def cell_pole_table(pole_points: list[dict], pole_vectors: np.ndarray,
                    grouped: dict[str, np.ndarray], view: str) -> list[dict]:
    """One row per (pole, view): mean cosine per cell plus the headline
    B-vs-C discriminance, flagged against the artifact band."""
    rows = []
    for k, pole in enumerate(pole_points):
        row = {"view": view, "pole": pole["pole"], "pole_kind": pole["pole_kind"]}
        sims = {}
        for cell, vecs in grouped.items():
            if len(vecs) == 0:
                continue
            s = gac.cosine_similarities(pole_vectors[k], vecs)
            sims[cell] = s
            row[f"cell_{cell}_mean_cos"] = round(float(np.mean(s)), 4)
            row[f"cell_{cell}_n"] = len(vecs)
        b, c = HEADLINE_CELLS
        if b in sims and c in sims:
            d = aps.discriminance(sims[b], sims[c])
            row["B_minus_C"] = d["gap"]
            row["B_vs_C_auc"] = d["auc"]
            row["clears_artifact_band"] = d["clears_artifact_band"]
        rows.append(row)
    return rows


def specificity_rows(criteria_points: list[dict], criteria_vectors: np.ndarray,
                     grouped: dict[str, np.ndarray], view: str) -> list[dict]:
    """B minus C against every one of the 17 criteria. The coercion result is
    only about coercion if B's advantage is concentrated on the subjection
    criterion rather than spread evenly -- otherwise it is register."""
    b, c = HEADLINE_CELLS
    if b not in grouped or c not in grouped:
        return []
    rows = []
    for k, point in enumerate(criteria_points):
        sims_b = gac.cosine_similarities(criteria_vectors[k], grouped[b])
        sims_c = gac.cosine_similarities(criteria_vectors[k], grouped[c])
        d = aps.discriminance(sims_b, sims_c)
        rows.append({
            "view": view,
            "criterion_key": point["key"],
            "cell_B_mean_cos": d["mean_a"],
            "cell_C_mean_cos": d["mean_b"],
            "B_minus_C": d["gap"],
            "auc": d["auc"],
            "clears_artifact_band": d["clears_artifact_band"],
        })
    rows.sort(key=lambda r: -r["B_minus_C"])
    return rows


def interpret(name_gap: float | None, desc_gap: float | None,
              band: float = aps.ARTIFACT_BAND) -> str:
    """Applies the pre-registered reading from the module docstring. Kept as
    a function so the conclusion is mechanical rather than a judgement made
    after seeing the numbers."""
    if name_gap is None and desc_gap is None:
        return "NO DATA"
    name_sig = name_gap is not None and name_gap > band
    desc_sig = desc_gap is not None and desc_gap > band
    if desc_sig and not name_sig:
        return ("B>C on descriptions but not names -- coercive quality is legible in conduct "
                "language and invisible in names (the pre-registered expected outcome)")
    if desc_sig and name_sig:
        return "B>C on both views -- names carry more conduct information than assumed"
    if not desc_sig and not name_sig:
        return ("B~C on both views -- this space cannot represent coercive control from these "
                "inputs; Analysis/11's retraction stands unqualified")
    return ("B>C on names but not descriptions -- unexpected; suspect naming register rather "
            "than conduct, inspect cell composition before interpreting")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lit-mivi-run-tag", default="20260910")
    parser.add_argument("--interviews-archive-dir", type=Path,
                        default=bss.PROCESSED_DIR / "interviews_full" / "interviews" / "run_20260912")
    parser.add_argument("--names-path", type=Path,
                        default=bss.PROCESSED_DIR / "analysis_raw" / "coercive_control" / "coercive_control_groups.jsonl")
    parser.add_argument("--descriptions-path", type=Path, default=None,
                        help="Defaults to the --names-path filename with _descriptions before .jsonl")
    parser.add_argument("--out-dir", type=Path,
                        default=bss.PROCESSED_DIR / "analysis_raw" / "coercive_control")
    parser.add_argument("--exclude-corpus-entities", action="store_true", default=True,
                        help="Drop generated names that are already corpus entities (circularity control).")
    args = parser.parse_args()

    if not args.names_path.exists():
        raise SystemExit(
            f"{args.names_path} not found.\n"
            "This module needs the 4-cell list generated and embedded on the Ollama machine first:\n"
            "  1. run the prompt in dictionaries/coercive-control-groups/PROMPT.md\n"
            "  2. save the raw output into that directory as a .txt\n"
            "  3. python -m thesis_corpus.embed_manual_secular_groups --cells \\\n"
            "       --source-dir dictionaries/coercive-control-groups \\\n"
            "       --out-dir processed/analysis_raw/coercive_control \\\n"
            "       --out-name coercive_control_groups.jsonl\n"
            "  4. copy both JSONL files back to the Mac at the same relative paths"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    desc_path = args.descriptions_path or Path(
        str(args.names_path).replace(".jsonl", "_descriptions.jsonl"))

    raw_points = acc.load_raw_points(args.lit_mivi_run_tag, args.interviews_archive_dir)
    vectors = np.array([p["vector"] for p in raw_points], dtype=np.float64)
    points = [{k: v for k, v in p.items() if k != "vector"} for p in raw_points]

    name_rows, name_vectors = load_cell_groups(args.names_path)
    logger.info("Loaded %d cell entries from %s", len(name_rows), args.names_path)

    # ---- circularity control
    tag_rows, entity_vectors = tag_corpus_membership([r["label"] for r in name_rows], points, vectors)
    tag_rows = add_pool_proximity(tag_rows, name_vectors, entity_vectors)
    gac.write_csv(args.out_dir / "corpus_membership_tags.csv", tag_rows)
    already = {r["label"] for r in tag_rows if r["in_corpus_entities"]}
    review = residual_review_candidates(tag_rows)
    print(f"\n=== Circularity control ===")
    print(f"  {len(already)}/{len(tag_rows)} generated names are ALREADY corpus entities "
          f"{'(excluded below)' if args.exclude_corpus_entities else '(kept)'}")
    print(f"  {len(review)} rows need hand review (novel by exact match, but >=0.85 cosine to some entity):")
    for r in review[:15]:
        print(f"    {r['label'][:44]:46s} max cos to pool {r['max_cos_to_entity_pool']:.3f}")

    excluded = already if args.exclude_corpus_entities else set()
    pole_points, pole_vectors = aps.build_poles(points, vectors, proxies=None)
    criteria_idx = acc.group_indices(points, "sectarian_drift_list")
    criteria_points = [points[i] for i in criteria_idx]
    criteria_vectors = vectors[criteria_idx]

    all_pole_rows, all_spec_rows = [], []
    gaps: dict[str, float | None] = {"names": None, "descriptions": None}

    views = [("names", name_rows, name_vectors)]
    if desc_path.exists():
        desc_rows, desc_vectors = load_cell_groups(desc_path)
        views.append(("descriptions", desc_rows, desc_vectors))
        logger.info("Loaded %d descriptions from %s", len(desc_rows), desc_path)
    else:
        logger.warning("%s not found -- names view only; the headline contrast needs both.", desc_path)

    for view, rows, vecs in views:
        # Name-form matching applies to the NAME view only: descriptions are
        # sentences, so the <=3-word rule would delete all of them.
        grouped = cell_vectors(rows, vecs, excluded, name_form_matched_only=(view == "names"))
        pole_rows = cell_pole_table(pole_points, pole_vectors, grouped, view)
        all_pole_rows.extend(pole_rows)
        all_spec_rows.extend(specificity_rows(criteria_points, criteria_vectors, grouped, view))

        print(f"\n=== {view.upper()} view: cells vs each pole ===")
        for r in pole_rows:
            cells = "  ".join(f"{c}={r[f'cell_{c}_mean_cos']:.3f}(n={r[f'cell_{c}_n']})"
                              for c in "ABCD" if f"cell_{c}_mean_cos" in r)
            print(f"  {r['pole'][:30]:32s} {cells}")
            if "B_minus_C" in r:
                band = "CLEARS" if r["clears_artifact_band"] else f"inside +-{aps.ARTIFACT_BAND}"
                print(f"      B-C {r['B_minus_C']:+.4f}  AUC {r['B_vs_C_auc']:.3f}  ({band})")

        coercion = [r for r in pole_rows if r["pole"] in COERCION_POLES and "B_minus_C" in r]
        if coercion:
            gaps[view] = float(np.mean([r["B_minus_C"] for r in coercion]))

    gac.write_csv(args.out_dir / "cells_vs_poles.csv", all_pole_rows)
    gac.write_csv(args.out_dir / "cells_specificity_by_criterion.csv", all_spec_rows)

    print("\n=== Specificity gate: is B's advantage concentrated, or uniform? ===")
    for view, _, _ in views:
        sel = [r for r in all_spec_rows if r["view"] == view]
        if not sel:
            continue
        pos = sum(1 for r in sel if r["B_minus_C"] > 0)
        print(f"  {view}: B>C on {pos}/{len(sel)} criteria; "
              f"largest {sel[0]['criterion_key']} {sel[0]['B_minus_C']:+.4f}, "
              f"smallest {sel[-1]['criterion_key']} {sel[-1]['B_minus_C']:+.4f}")
        if pos >= len(sel) - 1:
            print("    WARNING: B beats C on nearly every criterion -- that is a register effect, "
                  "not a coercion effect. Do not report this as coercion.")

    print("\n=== PRE-REGISTERED READING ===")
    print(f"  mean B-C on coercion poles: names {gaps['names']}, descriptions {gaps['descriptions']}")
    print(f"  -> {interpret(gaps['names'], gaps['descriptions'])}")
    print(f"\nDone. Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
