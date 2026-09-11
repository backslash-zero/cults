"""Corpus-native half of testing the thesis's central Hypothesis ("The
basis for assessment that certain groups are cults would actually imply
that other social structures are cults as well") -- pull entities already
present across the three corpora that name a structurally NON-religious
social group (an MLM, a secular self-improvement seminar program, a
political movement), and measure their distance to the 17-criteria
centroid and the corpus-wide "cult prototype" centroid
(analyze_cult_prototype.py), against a baseline of unambiguously
religious/NRM entities.

NO AUTOMATED CLASSIFIER EXISTS FOR THIS, AND ONE ISN'T BUILT HERE EITHER --
checked directly: looks_like_named_entity() is purely orthographic
(capitalization-based), and analyze_emergent_entities.py's
provenance_category classifies by WHICH CORPUS mentions a term, not by
semantic type. Building a real classifier would need either a working
Ollama connection (unavailable on this machine right now, see
generate_and_embed_secular_groups.py) or non-trivial new NLP work neither
justified by nor the point of this specific check. Instead: keyword-assisted
candidate generation over the existing entities_all pool (3,785 entities),
then MANUAL review and tagging -- same rigor as the cluster-membership
checks already done by hand in 06_Literature_Clusters.md. The reviewed
list is a small, explicit, documented constant (SECULAR_STRUCTURE_ENTITIES
below), same convention as build_shared_space.py's own
MANUALLY_EXCLUDED_POOLED_KEYS -- a hand-curated allowlist, not a
classifier, checked into code so the reasoning for each inclusion is
visible and revisable.

Candidate generation (for the record, not re-run by this module): keyword
grep over entities_all's labels for corporate/business/MLM/self-help/
political-movement patterns, reviewed by hand. Confirmed genuine secular
structural mentions: Amway (MLM), Landmark Forum / Erhard Seminar Training
("est") / Lifespring (Large Group Awareness Training programs -- explicitly
NOT religious, widely discussed in the cult-studies literature as
structural analogues), and several secular political movements (Nazism,
Maoism, Communism) whose "thought reform" mechanics this literature
explicitly compares to religious cult indoctrination. Explicitly EXCLUDED
despite matching keywords: Scientology (conventionally classified as a
new religious movement in NRM scholarship, not a clean secular
comparison case, even though it has a commercial/self-help-adjacent
public face).

Usage (from thesis/corpus/):
    python -m thesis_corpus.find_secular_group_mentions
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from thesis_corpus import analyze_corpus_centroids as acc
from thesis_corpus import analyze_cult_prototype as acp
from thesis_corpus import build_shared_space as bss
from thesis_corpus import geometric_analysis_common as gac

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("thesis_corpus.find_secular_group_mentions")

# Hand-reviewed, from a keyword-assisted scan of entities_all's 3,785
# labels (see module docstring) -- normalized-anchor keys, matched
# case-insensitively against each entity point's own `key`/`label`.
SECULAR_STRUCTURE_ENTITIES = {
    "amway": "MLM (multi-level marketing) -- commercial, not religious",
    "landmark forum": "Large Group Awareness Training (LGAT) seminar program -- secular self-improvement",
    "erhard seminar training": "LGAT ('est') -- secular self-improvement seminar",
    "lifespring": "LGAT -- secular self-improvement seminar",
    "werner erhard": "LGAT founder (est/Landmark) -- represents the secular seminar movement",
    "nazism": "secular political ideology/movement",
    "national socialism": "secular political ideology/movement",
    "maoist thought reform": "secular political movement's indoctrination apparatus, explicitly compared to cult 'thought reform' in this literature",
    "chinese communist party": "secular political party/movement",
    "communist party": "secular political party/movement",
}

# Baseline: unambiguously religious/NRM entities already characterized as
# tight, well-anchored literature clusters in 06_Literature_Clusters.md
# (Heaven's Gate 0.985 cosine similarity to its own cluster, Aum Shinrikyo
# 0.933, etc.) -- reused here as the comparison population, not re-derived.
RELIGIOUS_BASELINE_ENTITIES = {
    "heaven's gate", "aum shinrikyo", "branch davidians", "peoples temple",
    "unification church", "solar temple", "the family (formerly the children of god)",
    "rajneesh movement", "theosophical society",
}


def find_entities_by_key(points: list[dict], keys: set[str]) -> list[int]:
    """Indices into `points` for emergent_entities whose own `key`
    (normalized anchor) matches one of `keys`, case-insensitively."""
    wanted = {k.lower() for k in keys}
    return [i for i, p in enumerate(points)
            if p["source_dataset"] == "emergent_entities" and p["key"].lower() in wanted]


def distances_to_centroids(points: list[dict], vectors: np.ndarray, indices: list[int],
                            centroids: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for i in indices:
        row = {"key": points[i]["key"], "label": points[i]["label"]}
        for name, centroid in centroids.items():
            d = float(gac.euclidean_distances(centroid, vectors[i][np.newaxis, :])[0])
            sim = float(gac.cosine_similarities(centroid, vectors[i][np.newaxis, :])[0])
            row[f"{name}_euclidean_distance"] = round(d, 4)
            row[f"{name}_cosine_similarity"] = round(sim, 4)
        rows.append(row)
    return rows


def load_generated_group_distances(path: Path, centroids: dict[str, np.ndarray]) -> list[dict]:
    """Optional third group: generate_and_embed_secular_groups.py's output
    (an LLM-generated, then bge-m3-embedded, list of non-religious groups
    -- run on the Windows/Ollama machine, copied back to `path`). Returns
    [] if that file doesn't exist yet -- this script's C1 comparison is
    reported in full either way, never blocked on C2."""
    if not path.exists():
        logger.info("%s not found -- C2 (LLM-generated comparison list) not yet run; "
                     "reporting C1 (corpus-native) only.", path)
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            vector = np.array(item["embedding_vector"], dtype=np.float64)
            row = {"key": item["label"], "label": item["label"]}
            for name, centroid in centroids.items():
                d = float(gac.euclidean_distances(centroid, vector[np.newaxis, :])[0])
                sim = float(gac.cosine_similarities(centroid, vector[np.newaxis, :])[0])
                row[f"{name}_euclidean_distance"] = round(d, 4)
                row[f"{name}_cosine_similarity"] = round(sim, 4)
            row["group"] = "generated_secular"
            row["reasoning"] = "LLM-generated, bge-m3-embedded on the Windows/Ollama machine"
            rows.append(row)
    logger.info("Loaded %d LLM-generated secular groups from %s", len(rows), path)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lit-mivi-run-tag", default="20260910")
    parser.add_argument("--interviews-archive-dir", type=Path,
                         default=bss.PROCESSED_DIR / "interviews_full" / "interviews" / "run_20260912")
    parser.add_argument("--out-dir", type=Path, default=bss.PROCESSED_DIR / "analysis_raw" / "secular_groups")
    parser.add_argument("--generated-groups-path", type=Path, default=None,
                         help="Defaults to <out-dir>/generated_secular_groups.jsonl -- "
                              "generate_and_embed_secular_groups.py's output, copied back from the "
                              "Windows/Ollama machine. Folded in as a third group if present.")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw_points = acc.load_raw_points(args.lit_mivi_run_tag, args.interviews_archive_dir)
    vectors = np.array([p["vector"] for p in raw_points], dtype=np.float64)
    points = [{k: v for k, v in p.items() if k != "vector"} for p in raw_points]
    logger.info("Loaded %d raw points", len(points))

    criteria_idx = acc.group_indices(points, "sectarian_drift_list")
    criteria_centroid = gac.centroid_and_dispersion_for_indices(vectors, criteria_idx)["centroid"]

    pool_idx = acp.cult_prototype_pool_indices(points)
    pooled_prototype_centroid = gac.centroid_and_dispersion_for_indices(vectors, pool_idx)["centroid"]
    equal_prototype_centroid = acp.equal_weighted_prototype_centroid(points, vectors)

    centroids = {
        "criteria_list": criteria_centroid,
        "pooled_prototype": pooled_prototype_centroid,
        "equal_weighted_prototype": equal_prototype_centroid,
    }

    secular_idx = find_entities_by_key(points, set(SECULAR_STRUCTURE_ENTITIES))
    logger.info("Resolved %d/%d secular-structure entities", len(secular_idx), len(SECULAR_STRUCTURE_ENTITIES))
    missing = set(SECULAR_STRUCTURE_ENTITIES) - {points[i]["key"].lower() for i in secular_idx}
    if missing:
        logger.warning("Not found in entities_all (skipped): %s", missing)

    religious_idx = find_entities_by_key(points, RELIGIOUS_BASELINE_ENTITIES)
    logger.info("Resolved %d/%d religious-baseline entities", len(religious_idx), len(RELIGIOUS_BASELINE_ENTITIES))
    missing_r = RELIGIOUS_BASELINE_ENTITIES - {points[i]["key"].lower() for i in religious_idx}
    if missing_r:
        logger.warning("Not found in entities_all (skipped): %s", missing_r)

    secular_rows = distances_to_centroids(points, vectors, secular_idx, centroids)
    for r in secular_rows:
        r["group"] = "secular_structure"
        r["reasoning"] = SECULAR_STRUCTURE_ENTITIES.get(r["key"].lower(), "")
    religious_rows = distances_to_centroids(points, vectors, religious_idx, centroids)
    for r in religious_rows:
        r["group"] = "religious_baseline"
        r["reasoning"] = ""

    generated_path = args.generated_groups_path or (args.out_dir / "generated_secular_groups.jsonl")
    generated_rows = load_generated_group_distances(generated_path, centroids)

    all_rows = secular_rows + religious_rows + generated_rows
    out_path = args.out_dir / "secular_vs_religious_distances.csv"
    gac.write_csv(out_path, all_rows)

    for centroid_name in centroids:
        sec_vals = [r[f"{centroid_name}_cosine_similarity"] for r in secular_rows]
        rel_vals = [r[f"{centroid_name}_cosine_similarity"] for r in religious_rows]
        msg = "%-25s corpus-native secular mean cos=%.3f (n=%d) | religious-baseline mean cos=%.3f (n=%d)"
        margs = [centroid_name, np.mean(sec_vals), len(sec_vals), np.mean(rel_vals), len(rel_vals)]
        if generated_rows:
            gen_vals = [r[f"{centroid_name}_cosine_similarity"] for r in generated_rows]
            msg += " | LLM-generated secular mean cos=%.3f (n=%d)"
            margs += [np.mean(gen_vals), len(gen_vals)]
        logger.info(msg, *margs)

    print(f"Done. {out_path}")


if __name__ == "__main__":
    main()
