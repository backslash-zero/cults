"""An emergent "cult prototype" constructed from the corpus itself, and a
centrality/typicality ranking of the 17 official MIVILUDES criteria against
it -- operationalizing the thesis's own \\S "Family resemblance and prototype
theory" (Rosch/Mervis: category members vary in typicality; the most
prototypical members share the most attributes with the rest of the
category) directly on this corpus's data, rather than assuming the 17
official criteria are equally central by fiat.

WHY THIS IS A DIFFERENT CENTROID FROM sectarian_drift_list's OWN CENTROID
(analyze_corpus_centroids.py): that group's centroid is the mean of the 17
criteria's OWN vectors -- it measures which criterion is most central to
the *list of criteria itself*. This module instead pools every
cult-relevant EXPRESSION across all three corpora (what people/scholars/
the state actually SAY about cults, not the 17-item official list) and
asks which of the 17 criteria sits closest to *that* -- a genuinely
different, more data-driven question: not "which criterion is the anchor
of the other 16" but "which criterion is most central to how this corpus,
in aggregate, actually talks about cults."

POOLING RULE, checked against real data before deciding it (not assumed):
literature and miviludes v2 archives are 100% cult_relevant=True (the
selective v2 pipeline only ever kept cult-relevant candidates in the first
place, so filtering by it there is a no-op). Only interviews (the
exhaustive pipeline, which embeds literal filler/backchannel too) has real
variation: 618 True / 49 False / 2 unlabeled of 669. So the prototype pool
is ALL of literature + ALL of miviludes + interviews' cult_relevant=True
subset only -- this excludes labelled filler and the 2 unlabelled segments
from the interviews side specifically, without discarding anything from
the other two corpora (there is nothing there to discard).

RAW, NOT shared_space_v3 -- same reasoning/reuse as analyze_corpus_centroids.py
(imported directly, not reimplemented): raw bge-m3 vectors are unit-norm and
directly cosine-comparable with zero transformation.

Usage (from thesis/corpus/):
    python -m thesis_corpus.analyze_cult_prototype
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from thesis_corpus import analyze_corpus_centroids as acc
from thesis_corpus import build_shared_space as bss
from thesis_corpus import geometric_analysis_common as gac

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("thesis_corpus.analyze_cult_prototype")

NEAREST_EXPRESSIONS_K = 10


def cult_prototype_pool_indices(points: list[dict]) -> list[int]:
    """All literature + all miviludes (both 100% cult_relevant already,
    verified -- see module docstring) + interviews filtered to
    cult_relevant is True (excludes labelled-filler False and the 2
    unlabelled/None segments)."""
    indices = []
    for i, p in enumerate(points):
        if p["source_dataset"] in ("literature", "miviludes"):
            indices.append(i)
        elif p["source_dataset"] == "interviews" and p.get("cult_relevant") is True:
            indices.append(i)
    return indices


def equal_weighted_prototype_centroid(points: list[dict], vectors):
    """The plain pooled centroid (cult_prototype_pool_indices, unweighted)
    is ~98% literature by point count (5,741 of 5,849) -- checked directly
    on the first real run, not assumed -- so it's really "literature's own
    centroid with a thin garnish," not a balanced tri-source prototype.
    This is the known, already-addressed-elsewhere problem in this
    toolkit (see build_shared_space.py's literature_balanced_sample.jsonl,
    and analyze_typicality.py's own "equal-corpus-weighted... centre of
    gravity" framing) -- same fix, applied here: each corpus's own
    sub-centroid contributes equally, regardless of how many points it
    has, by averaging the three sub-centroids rather than pooling points
    directly."""
    sub_centroids = []
    for corpus in ("literature", "miviludes"):
        idx = acc.group_indices(points, corpus)
        sub_centroids.append(gac.centroid_and_dispersion_for_indices(vectors, idx)["centroid"])
    interview_idx = [i for i, p in enumerate(points) if p["source_dataset"] == "interviews" and p.get("cult_relevant") is True]
    sub_centroids.append(gac.centroid_and_dispersion_for_indices(vectors, interview_idx)["centroid"])
    return np.mean(sub_centroids, axis=0)


def rank_criteria_by_prototype_centrality(points: list[dict], vectors, prototype_centroid) -> list[dict]:
    """Full ranking (no k cutoff -- there are only 17) of the
    miviludes_criteria points by distance to the prototype centroid."""
    criteria_idx = acc.group_indices(points, "sectarian_drift_list")
    criteria_points = [points[i] for i in criteria_idx]
    criteria_vectors = vectors[criteria_idx]
    return gac.ranked_points(prototype_centroid, criteria_points, criteria_vectors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lit-mivi-run-tag", default="20260910")
    parser.add_argument("--interviews-archive-dir", type=Path,
                         default=bss.PROCESSED_DIR / "interviews_full" / "interviews" / "run_20260912")
    parser.add_argument("--nearest-expressions-k", type=int, default=NEAREST_EXPRESSIONS_K)
    parser.add_argument("--out-dir", type=Path, default=bss.PROCESSED_DIR / "analysis_raw" / "cult_prototype")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw_points = acc.load_raw_points(args.lit_mivi_run_tag, args.interviews_archive_dir)
    vectors = np.array([p["vector"] for p in raw_points], dtype=np.float64)
    points = [{k: v for k, v in p.items() if k != "vector"} for p in raw_points]
    logger.info("Loaded %d raw points", len(points))

    pool_idx = cult_prototype_pool_indices(points)
    logger.info("Prototype pool: %d points (literature + miviludes, all cult_relevant; "
                "interviews restricted to cult_relevant=True)", len(pool_idx))
    stats = gac.centroid_and_dispersion_for_indices(vectors, pool_idx)
    prototype_centroid = stats["centroid"]

    criteria_ranking = rank_criteria_by_prototype_centrality(points, vectors, prototype_centroid)
    criteria_path = args.out_dir / "criteria_centrality_ranking.csv"
    gac.write_csv(criteria_path, criteria_ranking)
    logger.info("Criteria ranked by centrality to the corpus-wide prototype (most -> least central):")
    for r in criteria_ranking:
        logger.info("  %2d. %-70s cos=%.3f", r["rank"], r["label"][:70], r["cosine_similarity"])

    pool_points = [points[i] for i in pool_idx]
    pool_vectors = vectors[pool_idx]
    nearest_expressions = gac.nearest_points(prototype_centroid, pool_points, pool_vectors, k=args.nearest_expressions_k)
    expressions_path = args.out_dir / "nearest_expressions_to_prototype.csv"
    gac.write_csv(expressions_path, nearest_expressions)
    logger.info("Nearest real expressions to the prototype centroid:")
    for n in nearest_expressions:
        logger.info("  %2d. %s", n["rank"], n["label"][:90])

    # Robustness check: the plain pool is ~98% literature by point count --
    # rerun the criteria ranking against an equal-corpus-weighted centroid
    # instead, to see whether the ranking is a genuine cross-source
    # prototype or an artefact of literature's sheer volume.
    equal_centroid = equal_weighted_prototype_centroid(points, vectors)
    equal_ranking = rank_criteria_by_prototype_centrality(points, vectors, equal_centroid)
    equal_path = args.out_dir / "criteria_centrality_ranking_equal_weighted.csv"
    gac.write_csv(equal_path, equal_ranking)
    logger.info("Criteria ranked by centrality to the EQUAL-WEIGHTED (literature/miviludes/interviews "
                "sub-centroids averaged, not pooled by point count) prototype:")
    for r in equal_ranking:
        logger.info("  %2d. %-70s cos=%.3f", r["rank"], r["label"][:70], r["cosine_similarity"])

    print(f"Done. {criteria_path}\n{expressions_path}\n{equal_path}")


if __name__ == "__main__":
    main()
