"""Per-criterion comparison of the non-religious group set (C2, 169 names
from dictionaries/non-religious-groups/, see parse_manual_secular_groups.py)
against the 17 individual MIVILUDES sectarian-drift criteria -- not just
the criteria list's pooled centroid, which is all
find_secular_group_mentions.py measured.

Why this is a different question: the aggregate result
(Analysis/08_Secular_Groups_Hypothesis.md) says secular groups score about
as close to the criteria centroid as religious/NRM groups do. But the
Hypothesis ("the basis for assessment that certain groups are cults would
imply that other social structures are cults as well") is really a claim
about the *basis* -- the individual criteria. Some of the 17 plainly
generalize beyond religion (authoritarian opaque organization, deceptive
recruitment, difficulty leaving); others are tied to a specifically
religious or French-administrative frame (violation of Republic principles,
alarming dietary change). This module measures which is which, per
criterion, so the Hypothesis can be evaluated at the granularity it's
actually stated at.

NAME-FORM CONTROL IS BUILT IN, NOT OPTIONAL. The C2 names and the
corpus-native entity anchors differ systematically in surface form, and
bge-m3 is measurably sensitive to that difference: C2 names are Title Case
and average 3.30 words with 71/169 starting with "The", while every
corpus-native entity anchor is lowercase and averages 2.56 words with
almost no "The" (they come from normalized anchor text -- see
build_shared_space.load_emergent_entities). The same organization appearing
in C2 under both names ("American Civil Liberties Union" vs "The American
Civil Liberties Union", and three other such accidental pairs) scores
+0.061 higher on the criteria centroid with the semantically empty "The"
attached -- an artifact LARGER than the entire C2-vs-religious-baseline gap
it would otherwise be credited to. So every comparison here is reported
both raw and restricted to a name-form-matched subset (<=3 words, no
leading "The"), and only effects that survive matching are treated as real.
See name_form_report() / match_baseline_name_form().

Usage (from thesis/corpus/):
    python -m thesis_corpus.analyze_secular_groups_vs_criteria
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from thesis_corpus import analyze_corpus_centroids as acc
from thesis_corpus import build_shared_space as bss
from thesis_corpus import find_secular_group_mentions as fsgm
from thesis_corpus import geometric_analysis_common as gac

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("thesis_corpus.analyze_secular_groups_vs_criteria")

MAX_MATCHED_WORDS = 3


def load_generated_groups(path: Path) -> tuple[list[str], np.ndarray]:
    """Reads embed_manual_secular_groups.py's JSONL (label +
    embedding_vector per line) into parallel labels/vectors."""
    labels, vectors = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            labels.append(item["label"])
            vectors.append(item["embedding_vector"])
    return labels, np.array(vectors, dtype=np.float64)


def match_baseline_name_form(labels: list[str], max_words: int = MAX_MATCHED_WORDS) -> list[int]:
    """Indices of labels whose surface form resembles the corpus-native
    entity anchors these are compared against: short (<= max_words) and
    without the leading "The" that bge-m3 measurably rewards. See module
    docstring -- this is the artifact control, not a cosmetic filter."""
    return [i for i, lab in enumerate(labels)
            if len(lab.split()) <= max_words and not lab.lower().startswith("the ")]


def name_form_report(labels: list[str], sims: np.ndarray) -> list[dict]:
    """The "The X" vs "X" natural experiment: C2's source lists happen to
    contain four organizations under both names, which isolates the effect
    of one semantically empty token on cosine similarity. Returns one row
    per pair found. `sims` is a per-label similarity vector (any centroid)."""
    by_label = {lab: i for i, lab in enumerate(labels)}
    rows = []
    for lab, i in by_label.items():
        if lab.startswith("The ") and lab[4:] in by_label:
            j = by_label[lab[4:]]
            rows.append({
                "bare_name": lab[4:],
                "with_the": lab,
                "bare_similarity": round(float(sims[j]), 4),
                "with_the_similarity": round(float(sims[i]), 4),
                "delta": round(float(sims[i] - sims[j]), 4),
            })
    return rows


def per_criterion_comparison(
    criteria_points: list[dict], criteria_vectors: np.ndarray,
    group_vectors: dict[str, np.ndarray],
) -> list[dict]:
    """One row per criterion: mean cosine similarity of each named group
    set to THAT criterion's own vector (not the criteria centroid). Groups
    with no members are skipped rather than reported as zero."""
    rows = []
    for k, point in enumerate(criteria_points):
        row = {"criterion_key": point["key"], "criterion_label": point["label"]}
        crit = criteria_vectors[k]
        for name, vecs in group_vectors.items():
            if len(vecs) == 0:
                continue
            sims = gac.cosine_similarities(crit, vecs)
            row[f"{name}_mean_cos"] = round(float(np.mean(sims)), 4)
        rows.append(row)
    return rows


def nearest_criterion_per_group(
    labels: list[str], vectors: np.ndarray,
    criteria_points: list[dict], criteria_vectors: np.ndarray,
) -> list[dict]:
    """For each group, which single sectarian drift it sits closest to --
    the 'if this organization were assessed against the MIVILUDES grid,
    which drift would flag it first' view."""
    rows = []
    for i, lab in enumerate(labels):
        sims = np.array([float(gac.cosine_similarities(criteria_vectors[k], vectors[i][np.newaxis, :])[0])
                         for k in range(len(criteria_points))])
        best = int(np.argmax(sims))
        rows.append({
            "label": lab,
            "nearest_criterion_key": criteria_points[best]["key"],
            "nearest_criterion_cos": round(float(sims[best]), 4),
            "word_count": len(lab.split()),
            "name_form_matched": len(lab.split()) <= MAX_MATCHED_WORDS and not lab.lower().startswith("the "),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lit-mivi-run-tag", default="20260910")
    parser.add_argument("--interviews-archive-dir", type=Path,
                        default=bss.PROCESSED_DIR / "interviews_full" / "interviews" / "run_20260912")
    parser.add_argument("--out-dir", type=Path,
                        default=bss.PROCESSED_DIR / "analysis_raw" / "secular_groups")
    parser.add_argument("--generated-groups-path", type=Path, default=None)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    points = acc.load_raw_points(args.lit_mivi_run_tag, args.interviews_archive_dir)
    vectors = np.array([p["vector"] for p in points], dtype=np.float64)

    criteria_idx = acc.group_indices(points, "sectarian_drift_list")
    criteria_points = [points[i] for i in criteria_idx]
    criteria_vectors = vectors[criteria_idx]
    logger.info("Loaded %d MIVILUDES criteria points", len(criteria_points))

    gen_path = args.generated_groups_path or (args.out_dir / "generated_secular_groups.jsonl")
    gen_labels, gen_vectors = load_generated_groups(gen_path)
    logger.info("Loaded %d generated secular groups", len(gen_labels))

    matched_idx = match_baseline_name_form(gen_labels)
    logger.info("%d/%d generated groups match the baseline's name form", len(matched_idx), len(gen_labels))

    religious_idx = fsgm.find_entities_by_key(points, fsgm.RELIGIOUS_BASELINE_ENTITIES)
    secular_c1_idx = fsgm.find_entities_by_key(points, set(fsgm.SECULAR_STRUCTURE_ENTITIES))

    group_vectors = {
        "generated_all": gen_vectors,
        "generated_name_matched": gen_vectors[matched_idx],
        "religious_baseline": vectors[religious_idx],
        "secular_corpus_native": vectors[secular_c1_idx],
    }

    per_crit = per_criterion_comparison(criteria_points, criteria_vectors, group_vectors)
    for row in per_crit:
        row["matched_minus_religious"] = round(
            row["generated_name_matched_mean_cos"] - row["religious_baseline_mean_cos"], 4)
    per_crit.sort(key=lambda r: -r["generated_name_matched_mean_cos"])
    crit_path = args.out_dir / "criteria_vs_secular_groups.csv"
    gac.write_csv(crit_path, per_crit)

    print("\n=== Per-criterion: how close do non-religious groups sit to each sectarian drift? ===")
    print(f"{'criterion':42s} {'C2-matched':>11s} {'relig':>8s} {'diff':>8s}  {'C2-all':>8s}")
    for r in per_crit:
        print(f"{r['criterion_key']:42s} {r['generated_name_matched_mean_cos']:11.3f} "
              f"{r['religious_baseline_mean_cos']:8.3f} {r['matched_minus_religious']:+8.3f} "
              f"{r['generated_all_mean_cos']:8.3f}")

    nearest = nearest_criterion_per_group(gen_labels, gen_vectors, criteria_points, criteria_vectors)
    nearest.sort(key=lambda r: -r["nearest_criterion_cos"])
    nearest_path = args.out_dir / "secular_groups_nearest_criterion.csv"
    gac.write_csv(nearest_path, nearest)

    counts: dict[str, int] = {}
    for r in nearest:
        counts[r["nearest_criterion_key"]] = counts.get(r["nearest_criterion_key"], 0) + 1
    print("\n=== Which drift is each group's single nearest? (all 169) ===")
    for key, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {key:42s} {c:3d} groups")

    centroid = gac.centroid_and_dispersion_for_indices(vectors, criteria_idx)["centroid"]
    sims_to_centroid = gac.cosine_similarities(centroid, gen_vectors)
    nf = name_form_report(gen_labels, sims_to_centroid)
    nf_path = args.out_dir / "name_form_artifact_check.csv"
    gac.write_csv(nf_path, nf)
    print("\n=== Name-form artifact: same org, with vs without a leading \"The\" ===")
    for r in nf:
        print(f"  {r['bare_name']:52s} {r['bare_similarity']:.3f} -> {r['with_the_similarity']:.3f}  ({r['delta']:+.3f})")
    if nf:
        print(f"  mean delta from one semantically empty token: {np.mean([r['delta'] for r in nf]):+.4f}")

    print(f"\nDone.\n{crit_path}\n{nearest_path}\n{nf_path}")


if __name__ == "__main__":
    main()
