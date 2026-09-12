"""Tests whether "psychological subjection" (MIVILUDES criterion
crit-mental-destabilization -- the brainwashing / coercive-control claim) is
actually what separates cults from ordinary organizations in this embedding
space, and locates the literature's own coercive-control discourse relative
to the religious and secular entity sets.

WHY THIS MODULE EXISTS: to correct a claim, not to confirm one. An earlier
reading of analyze_secular_groups_vs_criteria.py's output held that
crit-mental-destabilization was the single criterion carrying all the
discriminating power of the MIVILUDES framework, because it ranked dead last
of 17 in absolute cosine similarity to 169 ordinary non-religious
organization names. That ranking is a TEXT-LENGTH ARTIFACT:
crit-mental-destabilization is the longest of the 17 criteria (29 French
words; the shortest is 3), and criterion length predicts absolute cosine at
Spearman -0.627. On the length-robust measure -- the gap between ordinary
organizations and the religious/NRM baseline, which is essentially
uncorrelated with length (-0.047) -- it ranks 6th of 17, not 1st. See
criterion_length_control(), which recomputes both and is asserted against
those figures in the tests.

Same family of failure as the "The" artifact already documented in
Analysis/08 (a leading "The" is worth +0.061 cosine on the same
organization): the metric is sensitive to surface form at a magnitude that
swamps the effects being measured. ARTIFACT_BAND below is the resolution
floor any cross-set claim has to clear.

WHAT REPLACES THE RETRACTED CLAIM: the literature's coercive-control
discourse forms three HDBSCAN clusters (Analysis/06) -- 40 (n=85, binding
expressions are the bare token "brainwashing"), 43 (n=59, "mental
manipulation"/"coercive persuasion"), 48 (n=39, flagged an entity gap). The
already-persisted entity_nearest_cluster.csv shows their catchment is
SECULAR: all 9 religious/NRM baseline entities sit in their own eponymous
clusters (Heaven's Gate 0.985, Aum Shinrikyo 0.933) and none is nearest to a
brainwashing cluster, while Maoist thought reform (0.587) and Erhard Seminar
Training (0.524) both land in cluster 43 alongside the coercive-control and
therapy lexicon. The mechanism vocabulary attaches to secular coercion
programmes; religious cults are discussed as named cases.

DELIBERATELY DOES NOT RE-RUN THE CLUSTERING. analyze_literature_clusters.py
never persists its per-expression `labels`, so true cluster centroids are not
recoverable; re-running UMAP+HDBSCAN would reassign cluster IDs and
invalidate the numbering Analysis/06 and Analysis/09 both depend on (09's
headline result is stated in terms of specific gap-cluster IDs). Instead:
the primary poles are the real entity anchors `brainwashing` and `coercive
persuasion`, which are points in the space already, are lowercase and 1-2
words (so name-form-matched to the filtered external names under the "The"
artifact), and have a known cosine to the true cluster centroids (0.872 to
40, 0.789 to 43). Cluster "core proxies" rebuilt from the persisted binding
expressions are reported only as a robustness check, and carry their own
degeneracy warning -- see load_cluster_core_proxies.

Usage (from thesis/corpus/):
    python -m thesis_corpus.analyze_psychological_subjection
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from thesis_corpus import analyze_corpus_centroids as acc
from thesis_corpus import analyze_secular_groups_vs_criteria as asgc
from thesis_corpus import build_shared_space as bss
from thesis_corpus import find_secular_group_mentions as fsgm
from thesis_corpus import geometric_analysis_common as gac

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("thesis_corpus.analyze_psychological_subjection")

# Real entity anchors, not reconstructions -- see module docstring.
POLE_ENTITY_KEYS = ("brainwashing", "coercive persuasion")
SUBJECTION_CRITERION_KEY = "crit-mental-destabilization"
SUBJECTION_CLUSTER_IDS = (40, 43, 48)

# From Analysis/08's "The X" vs "X" natural experiment: one semantically
# empty token moves cosine by this much on the SAME organization. Any
# cross-set difference smaller than this is not interpretable.
ARTIFACT_BAND = 0.061

DEFAULT_CLUSTERS_DIR = bss.PROCESSED_DIR / "analysis_raw" / "literature_clusters"
DEFAULT_CLUSTER_BINDING_CSV = DEFAULT_CLUSTERS_DIR / "cluster_binding_expressions.csv"
DEFAULT_ENTITY_NEAREST_CLUSTER_CSV = DEFAULT_CLUSTERS_DIR / "entity_nearest_cluster.csv"
# The un-embedded criteria JSON, which still carries the full criterion_fr
# text. bss.MIVILUDES_CRITERIA_PATH points at the *_embedded.jsonl instead.
CRITERIA_JSON_PATH = bss.CORPUS_DIR / "metadata" / "miviludes_criteria.json"


def load_criterion_word_counts(path: Path | None = None) -> dict[str, int]:
    """French word count per criterion id. Read from the criteria JSON, NOT
    from the loaded points' `label` field -- point labels are truncated to
    ~80 chars, which would silently compress the very covariate this module
    exists to measure."""
    path = path or CRITERIA_JSON_PATH
    with open(path, encoding="utf-8") as f:
        criteria = json.load(f)["miviludes_criteria"]
    return {c["id"]: len(c["criterion_fr"].split()) for c in criteria}


def _residuals_on_length(word_counts: list[float], values: list[float]) -> list[float]:
    """Ordinary least squares residual of `values` on `word_counts` -- what's
    left of a score once criterion text length is accounted for."""
    x = np.asarray(word_counts, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    return list(y - (slope * x + intercept))


def criterion_length_control(
    per_criterion_rows: list[dict],
    criterion_word_counts: dict[str, int],
    absolute_field: str = "generated_name_matched_mean_cos",
    diff_field: str = "matched_minus_religious",
    focus_key: str = SUBJECTION_CRITERION_KEY,
) -> tuple[list[dict], dict]:
    """THE CORRECTION. Adds the French word count to each per-criterion row,
    then reports how strongly length predicts each of the two measures and
    where `focus_key` ranks on each.

    `absolute_field` (mean cosine of ordinary organizations to the criterion)
    is heavily length-confounded: a long criterion sentence is far from every
    short organization name regardless of content. `diff_field` (that mean
    minus the religious baseline's own mean against the SAME criterion) is
    not, because the length penalty applies to both sides and cancels. Only
    the diff measure supports a claim about discrimination."""
    rows = []
    for r in per_criterion_rows:
        row = dict(r)
        row["criterion_fr_word_count"] = criterion_word_counts[r["criterion_key"]]
        rows.append(row)

    wc = [float(r["criterion_fr_word_count"]) for r in rows]
    absolute = [float(r[absolute_field]) for r in rows]
    diff = [float(r[diff_field]) for r in rows]

    for row, resid in zip(rows, _residuals_on_length(wc, diff)):
        row["diff_residual_on_length"] = round(resid, 4)

    def rank_of(field: str) -> int:
        ordered = sorted(rows, key=lambda r: float(r[field]))
        return next(i for i, r in enumerate(ordered) if r["criterion_key"] == focus_key) + 1

    stats = {
        "focus_criterion": focus_key,
        "focus_fr_word_count": criterion_word_counts[focus_key],
        "longest_fr_word_count": max(criterion_word_counts.values()),
        "shortest_fr_word_count": min(criterion_word_counts.values()),
        "spearman_length_vs_absolute": round(float(spearmanr(wc, absolute).statistic), 3),
        "spearman_length_vs_diff": round(float(spearmanr(wc, diff).statistic), 3),
        "focus_rank_absolute_of_17": rank_of(absolute_field),
        "focus_rank_diff_of_17": rank_of(diff_field),
        "n_criteria": len(rows),
    }
    return rows, stats


def load_cluster_core_proxies(
    csv_path: Path, cluster_ids, points: list[dict], vectors: np.ndarray,
) -> dict[int, dict]:
    """Rebuilds an approximate centroid for each named cluster from the
    binding expressions persisted in cluster_binding_expressions.csv, by
    resolving each row's `key` back to its raw vector and averaging.

    ROBUSTNESS CHECK ONLY -- not a substitute for the true centroid, and for
    cluster 40 it is degenerate. All 8 of cluster 40's persisted rows carry
    identical distance (0.43186) and cosine (0.92078) because all 8 labels
    are the literal token "brainwashing", so its "proxy centroid" collapses
    to the embedding of that single word (`n_distinct_vectors == 1`). Cluster
    43 is 4x "mental manipulation" + 2x "coercive persuasion" + 2 real
    sentences, so it is mode-weighted rather than degenerate. Only 48 has 8
    distinct texts.

    Two further limits, reported per row rather than hidden: the binding
    expressions are the members NEAREST the centroid, so their mean is a
    dense CORE and overstates similarity to anything lexically close while
    ignoring the cluster's periphery (cluster 40's internal dispersion is
    0.664 against a binding distance of 0.432); and 8 of 85 members is a 9%
    sample. `member_cos_to_true_centroid_*` is read straight from the CSV's
    own column, so the distance from proxy to truth is quantified, not
    guessed at."""
    by_key = {p["key"]: i for i, p in enumerate(points)}
    wanted = {int(c) for c in cluster_ids}
    grouped: dict[int, list[dict]] = {c: [] for c in wanted}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = int(row["cluster_id"])
            if cid in wanted:
                grouped[cid].append(row)

    proxies = {}
    for cid, rows in grouped.items():
        resolved, missing, member_cos = [], [], []
        for row in rows:
            member_cos.append(float(row["cosine_similarity"]))
            idx = by_key.get(row["key"])
            if idx is None:
                missing.append(row["key"])
            else:
                resolved.append(idx)
        if not resolved:
            logger.warning("Cluster %d: no binding expression keys resolved; skipping proxy.", cid)
            continue
        member_vectors = vectors[resolved]
        proxies[cid] = {
            "vector": member_vectors.mean(axis=0),
            "n_rows": len(rows),
            "n_resolved": len(resolved),
            "n_distinct_vectors": len({v.tobytes() for v in member_vectors}),
            "missing_keys": missing,
            "member_cos_to_true_centroid_min": round(min(member_cos), 4),
            "member_cos_to_true_centroid_mean": round(float(np.mean(member_cos)), 4),
        }
    return proxies


def proxy_fidelity_rows(proxies: dict[int, dict]) -> list[dict]:
    """Flat, reportable version of load_cluster_core_proxies' diagnostics --
    so the degeneracy is in the output CSV, not only in a docstring."""
    rows = []
    for cid in sorted(proxies):
        p = proxies[cid]
        rows.append({
            "cluster_id": cid,
            "n_binding_rows": p["n_rows"],
            "n_resolved": p["n_resolved"],
            "n_distinct_vectors": p["n_distinct_vectors"],
            "degenerate": p["n_distinct_vectors"] == 1,
            "n_missing_keys": len(p["missing_keys"]),
            "member_cos_to_true_centroid_min": p["member_cos_to_true_centroid_min"],
            "member_cos_to_true_centroid_mean": p["member_cos_to_true_centroid_mean"],
        })
    return rows


def build_poles(
    points: list[dict], vectors: np.ndarray, proxies: dict[int, dict] | None = None,
) -> tuple[list[dict], np.ndarray]:
    """Assembles the query vectors this module measures everything against,
    most defensible first:

      1-2. the `brainwashing` / `coercive persuasion` entity anchors -- real
           points, lowercase, 1-2 words, known cosine to the true centroids.
      3.   crit-mental-destabilization's own vector. Reported, but it is the
           longest criterion text in the list, so its ABSOLUTE similarities
           are length-depressed (see criterion_length_control) and only
           relative comparisons between group sets against this one pole
           mean anything.
      4.   all-entity grand centroid, as a REGISTER CONTROL. If a group set
           scores high on the subjection poles and equally high here, it is
           close to the whole entity vocabulary rather than to subjection
           specifically -- the distinction the retracted claim missed.
      5+.  cluster core proxies, flagged degenerate where they are.
    """
    pole_points, pole_vectors = [], []

    for key in POLE_ENTITY_KEYS:
        idx = fsgm.find_entities_by_key(points, {key})
        if not idx:
            logger.warning("Pole entity %r not found; skipping.", key)
            continue
        pole_points.append({"pole": key, "pole_kind": "entity_anchor"})
        pole_vectors.append(vectors[idx[0]])

    crit_idx = [i for i in acc.group_indices(points, "sectarian_drift_list")
                if points[i]["key"] == SUBJECTION_CRITERION_KEY]
    if crit_idx:
        pole_points.append({"pole": SUBJECTION_CRITERION_KEY, "pole_kind": "criterion_text"})
        pole_vectors.append(vectors[crit_idx[0]])

    entity_idx = acc.group_indices(points, "entities_all")
    pole_points.append({"pole": "all_entities_grand_centroid", "pole_kind": "register_control"})
    pole_vectors.append(gac.centroid_and_dispersion_for_indices(vectors, entity_idx)["centroid"])

    for cid in sorted(proxies or {}):
        p = proxies[cid]
        kind = "cluster_core_proxy_degenerate" if p["n_distinct_vectors"] == 1 else "cluster_core_proxy"
        pole_points.append({"pole": f"cluster_{cid}_core", "pole_kind": kind})
        pole_vectors.append(p["vector"])

    return pole_points, np.array(pole_vectors, dtype=np.float64)


def discriminance(sims_a: np.ndarray, sims_b: np.ndarray) -> dict:
    """Gap plus Mann-Whitney AUC between two similarity samples.

    AUC ("probability a randomly drawn member of A scores above one of B")
    is the headline rather than Cohen's d because the group sizes here are
    wildly unequal (9 religious entities vs 169 external names), which makes
    pooled-variance effect sizes unstable. AUC needs no variance estimate.
    No p-value is reported: with 17 correlated criteria and n=9 on one side
    nothing reaches significance, so the defensible claim is ordinal."""
    a = np.asarray(sims_a, dtype=np.float64)
    b = np.asarray(sims_b, dtype=np.float64)
    wins = (a[:, None] > b[None, :]).sum()
    ties = (a[:, None] == b[None, :]).sum()
    auc = (wins + 0.5 * ties) / (len(a) * len(b))
    gap = float(a.mean() - b.mean())
    return {
        "n_a": len(a), "n_b": len(b),
        "mean_a": round(float(a.mean()), 4), "mean_b": round(float(b.mean()), 4),
        "gap": round(gap, 4),
        "auc": round(float(auc), 4),
        "clears_artifact_band": abs(gap) > ARTIFACT_BAND,
    }


def pole_similarity_rows(
    pole_points: list[dict], pole_vectors: np.ndarray, group_vectors: dict[str, np.ndarray],
) -> list[dict]:
    """One row per pole: mean cosine of each named group set to that pole,
    plus the religious-vs-each-secular-set discriminance for it."""
    rows = []
    for k, pole in enumerate(pole_points):
        row = dict(pole)
        sims = {}
        for name, vecs in group_vectors.items():
            if len(vecs) == 0:
                continue
            s = gac.cosine_similarities(pole_vectors[k], vecs)
            sims[name] = s
            row[f"{name}_mean_cos"] = round(float(np.mean(s)), 4)
            row[f"{name}_n"] = len(vecs)
        if "religious_baseline" in sims:
            for other in ("secular_corpus_native", "generated_name_matched"):
                if other in sims:
                    d = discriminance(sims[other], sims["religious_baseline"])
                    row[f"{other}_minus_religious"] = d["gap"]
                    row[f"{other}_auc_vs_religious"] = d["auc"]
                    row[f"{other}_clears_band"] = d["clears_artifact_band"]
        rows.append(row)
    return rows


def pole_agreement(pole_points: list[dict], pole_vectors: np.ndarray,
                   entity_vectors: np.ndarray) -> list[dict]:
    """THE GATE. Spearman correlation between every pair of poles, over their
    rankings of all 3,785 entities. If the poles disagree, "psychological
    subjection" is not a single direction in this space and no single-pole
    number should be interpreted -- including the coercive-control experiment
    that depends on it."""
    rows = []
    sims = [gac.cosine_similarities(pole_vectors[k], entity_vectors) for k in range(len(pole_points))]
    for i in range(len(pole_points)):
        for j in range(i + 1, len(pole_points)):
            rows.append({
                "pole_a": pole_points[i]["pole"],
                "pole_b": pole_points[j]["pole"],
                "spearman_over_all_entities": round(float(spearmanr(sims[i], sims[j]).statistic), 3),
            })
    return rows


def summarize_entity_nearest_cluster(csv_path: Path, cluster_ids, entity_keys: dict[str, set]) -> list[dict]:
    """Reads the ALREADY-PERSISTED argmax view (computed from the true
    cluster centroids, before the labels were discarded) and reports, per
    named entity set, which cluster each member is nearest and whether that
    is one of the coercive-control clusters. No recomputation, so the true
    centroids are used rather than any proxy."""
    wanted_clusters = {str(c) for c in cluster_ids}
    lookup = {}
    for set_name, keys in entity_keys.items():
        for k in keys:
            lookup[k.lower()] = set_name

    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            set_name = lookup.get(row["entity_key"].lower())
            if set_name is None:
                continue
            rows.append({
                "group_set": set_name,
                "entity_key": row["entity_key"],
                "entity_label": row["entity_label"],
                "nearest_cluster_id": row["nearest_cluster_id"],
                "is_subjection_cluster": row["nearest_cluster_id"] in wanted_clusters,
                "cosine_similarity": round(float(row["cosine_similarity"]), 4),
            })
    rows.sort(key=lambda r: (r["group_set"], -r["cosine_similarity"]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lit-mivi-run-tag", default="20260910")
    parser.add_argument("--interviews-archive-dir", type=Path,
                        default=bss.PROCESSED_DIR / "interviews_full" / "interviews" / "run_20260912")
    parser.add_argument("--generated-groups-path", type=Path,
                        default=bss.PROCESSED_DIR / "analysis_raw" / "secular_groups" / "generated_secular_groups.jsonl")
    parser.add_argument("--cluster-binding-csv", type=Path, default=DEFAULT_CLUSTER_BINDING_CSV)
    parser.add_argument("--entity-nearest-cluster-csv", type=Path, default=DEFAULT_ENTITY_NEAREST_CLUSTER_CSV)
    parser.add_argument("--out-dir", type=Path,
                        default=bss.PROCESSED_DIR / "analysis_raw" / "psychological_subjection")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw_points = acc.load_raw_points(args.lit_mivi_run_tag, args.interviews_archive_dir)
    vectors = np.array([p["vector"] for p in raw_points], dtype=np.float64)
    points = [{k: v for k, v in p.items() if k != "vector"} for p in raw_points]

    # ---- 1. The correction: is the subjection criterion's rank a length artifact?
    criteria_idx = acc.group_indices(points, "sectarian_drift_list")
    criteria_points = [points[i] for i in criteria_idx]
    criteria_vectors = vectors[criteria_idx]

    gen_labels, gen_vectors = asgc.load_generated_groups(args.generated_groups_path)
    matched_idx = asgc.match_baseline_name_form(gen_labels)
    religious_idx = fsgm.find_entities_by_key(points, fsgm.RELIGIOUS_BASELINE_ENTITIES)
    secular_idx = fsgm.find_entities_by_key(points, set(fsgm.SECULAR_STRUCTURE_ENTITIES))

    group_vectors = {
        "generated_all": gen_vectors,
        "generated_name_matched": gen_vectors[matched_idx],
        "religious_baseline": vectors[religious_idx],
        "secular_corpus_native": vectors[secular_idx],
    }

    per_crit = asgc.per_criterion_comparison(criteria_points, criteria_vectors, group_vectors)
    for row in per_crit:
        row["matched_minus_religious"] = round(
            row["generated_name_matched_mean_cos"] - row["religious_baseline_mean_cos"], 4)

    length_rows, length_stats = criterion_length_control(per_crit, load_criterion_word_counts())
    gac.write_csv(args.out_dir / "criterion_length_control.csv", length_rows)

    print("\n=== THE CORRECTION: is the subjection criterion's rank a text-length artifact? ===")
    print(f"  {SUBJECTION_CRITERION_KEY} is {length_stats['focus_fr_word_count']} French words "
          f"(longest of 17 = {length_stats['longest_fr_word_count']}, shortest = {length_stats['shortest_fr_word_count']})")
    print(f"  Spearman(criterion length, ABSOLUTE cosine) = {length_stats['spearman_length_vs_absolute']:+.3f}  <- confounded")
    print(f"  Spearman(criterion length, DIFF vs baseline) = {length_stats['spearman_length_vs_diff']:+.3f}  <- not confounded")
    print(f"  rank on absolute cosine: {length_stats['focus_rank_absolute_of_17']}/17")
    print(f"  rank on diff measure   : {length_stats['focus_rank_diff_of_17']}/17   <- the defensible one")
    print("\n  five smallest diffs (the criteria that genuinely least discriminate):")
    for r in sorted(length_rows, key=lambda r: float(r["matched_minus_religious"]))[:5]:
        print(f"    {r['criterion_key']:42s} {float(r['matched_minus_religious']):+.4f} "
              f"({r['criterion_fr_word_count']} words)")

    # ---- 2. Poles, fidelity, agreement gate
    proxies = load_cluster_core_proxies(args.cluster_binding_csv, SUBJECTION_CLUSTER_IDS, points, vectors)
    fidelity = proxy_fidelity_rows(proxies)
    gac.write_csv(args.out_dir / "pole_fidelity.csv", fidelity)
    print("\n=== Cluster core-proxy fidelity (robustness poles only) ===")
    for r in fidelity:
        flag = "  DEGENERATE (reduces to one embedding)" if r["degenerate"] else ""
        print(f"  cluster {r['cluster_id']:>2}: {r['n_resolved']}/{r['n_binding_rows']} resolved, "
              f"{r['n_distinct_vectors']} distinct vectors, members' cos to true centroid "
              f"{r['member_cos_to_true_centroid_min']}-{r['member_cos_to_true_centroid_mean']}{flag}")

    pole_points, pole_vectors = build_poles(points, vectors, proxies)
    entity_idx = acc.group_indices(points, "entities_all")
    agreement = pole_agreement(pole_points, pole_vectors, vectors[entity_idx])
    gac.write_csv(args.out_dir / "pole_agreement.csv", agreement)
    print("\n=== Pole agreement gate (Spearman over all entities) ===")
    for r in agreement:
        print(f"  {r['pole_a'][:34]:36s} vs {r['pole_b'][:30]:32s} {r['spearman_over_all_entities']:+.3f}")

    pole_rows = pole_similarity_rows(pole_points, pole_vectors, group_vectors)
    gac.write_csv(args.out_dir / "poles_vs_group_sets.csv", pole_rows)
    print("\n=== Group sets vs each pole (mean cosine; AUC/gap vs the religious baseline) ===")
    for r in pole_rows:
        print(f"\n  pole: {r['pole']}  [{r['pole_kind']}]")
        for name in ("religious_baseline", "secular_corpus_native", "generated_name_matched", "generated_all"):
            if f"{name}_mean_cos" in r:
                print(f"    {name:24s} n={r[f'{name}_n']:3d}  mean cos {r[f'{name}_mean_cos']:.3f}")
        for other in ("secular_corpus_native", "generated_name_matched"):
            if f"{other}_minus_religious" in r:
                band = "clears" if r[f"{other}_clears_band"] else f"INSIDE +-{ARTIFACT_BAND} band"
                print(f"    {other} vs religious: gap {r[f'{other}_minus_religious']:+.4f}, "
                      f"AUC {r[f'{other}_auc_vs_religious']:.3f}  ({band})")

    # ---- 3. The already-persisted argmax view (true centroids, no proxy)
    nearest = summarize_entity_nearest_cluster(
        args.entity_nearest_cluster_csv, SUBJECTION_CLUSTER_IDS,
        {"religious_baseline": set(fsgm.RELIGIOUS_BASELINE_ENTITIES),
         "secular_corpus_native": set(fsgm.SECULAR_STRUCTURE_ENTITIES)},
    )
    gac.write_csv(args.out_dir / "entity_sets_nearest_cluster.csv", nearest)
    print("\n=== Which cluster is each entity set's members nearest? (true centroids, persisted) ===")
    for set_name in ("religious_baseline", "secular_corpus_native"):
        sel = [r for r in nearest if r["group_set"] == set_name]
        hits = sum(1 for r in sel if r["is_subjection_cluster"])
        print(f"\n  {set_name}: {hits}/{len(sel)} nearest a coercive-control cluster {SUBJECTION_CLUSTER_IDS}")
        for r in sel:
            mark = "  <-- SUBJECTION CLUSTER" if r["is_subjection_cluster"] else ""
            print(f"    {r['entity_label'][:44]:46s} -> cluster {r['nearest_cluster_id']:>3s} "
                  f"({r['cosine_similarity']:.3f}){mark}")

    print(f"\nDone. Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
