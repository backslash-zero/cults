"""Does literature's publication year relate to where a document's own
expressions sit in the shared space? LITERATURE ONLY -- checked directly
before writing this module: MIVILUDES has no usable per-document year
(only a .bib file's `date = {2024}` for one of its 2 source documents,
nothing structured); interviews have real collection dates, but all
within one narrow month, not a meaningful multi-year trend the way
literature's 2001-2024 span is.

Year comes from the literature document_id itself, not
metadata/literature.csv: that CSV's own `id` column is a Zotero citekey
with no relation to the pipeline's document_id slug (confirmed directly --
no join key exists between them), but every one of the 57 document_ids
embeds its year directly (e.g.
"clarke-2004-encyclopedia-of-new-religious-movements",
"2016-legal-cases-new-religious-movements-and-minority-faiths") --
verified 57/57 extractable, spot-checked against literature.csv for a
few titles.

Two questions:
  1. Correlation: does a literature point's distance to literature's OWN
     centroid (the same "how typical is this, geometrically" framing
     analyze_typicality.py already uses) correlate with its document's
     publication year? Spearman (rank-based, no linearity assumption).
  2. Period drift: split literature into year terciles (boundaries
     computed from the actual year distribution, never hardcoded), and
     compare each period's own centroid against the others' -- does
     which MIVILUDES criterion literature discourse sits closest to shift
     across the three periods?

GOVERNING PRINCIPLE: a correlation or period-centroid shift here is a
geometric fact about extracted-expression embeddings by publication year;
it is not, by itself, evidence of a causal trend in scholarly opinion, a
claim about which period's literature is "more correct," or a claim that
generalizes beyond this specific 57-document sample.

Usage (from thesis/corpus/):
    python -m thesis_corpus.analyze_publication_date --shared-space-dir processed/shared_space_v2
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from thesis_corpus import geometric_analysis_common as gac

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("thesis_corpus.analyze_publication_date")

MODULE_NAME = "analyze_publication_date"
YEAR_PATTERN = re.compile(r"(?:^|-)((?:19|20)\d{2})(?:-|$)")
NEAREST_K_CRITERION = 1

GOVERNING_PRINCIPLE = (
    "A correlation or period-centroid shift here is a geometric fact about extracted-"
    "expression embeddings by publication year; it is not, by itself, evidence of a "
    "causal trend in scholarly opinion, a claim about which period's literature is "
    "\"more correct,\" or a claim that generalizes beyond this specific 57-document sample."
)


# ---------------------------------------------------------------------------
# Pure computation -- no file I/O, unit-testable.
# ---------------------------------------------------------------------------

def extract_year(document_id: str) -> int | None:
    """The 4-digit year embedded in a literature document_id, or None if
    genuinely absent (verified 57/57 present for the current corpus, but
    this stays a soft None-on-miss rather than a hard-fail, so a future,
    differently-named document doesn't silently crash the whole module --
    callers report/skip missing years explicitly, never pretend they
    found one)."""
    m = YEAR_PATTERN.search(document_id)
    return int(m.group(1)) if m else None


def years_for_points(points: list[dict]) -> tuple[list[int], list[int]]:
    """Returns (resolved_indices, years) -- parallel lists; only points
    whose document_id yields an extractable year are included. Points
    without one are silently excluded here but their count is the
    caller's job to log (see main())."""
    resolved_indices, years = [], []
    for i, point in enumerate(points):
        document_id = gac.key_document_id(point["key"])
        year = extract_year(document_id)
        if year is not None:
            resolved_indices.append(i)
            years.append(year)
    return resolved_indices, years


def year_distance_correlation(years: list[int], distances: list[float]) -> dict:
    """Spearman rank correlation between publication year and distance to
    literature's own centroid -- rank-based (no assumption the
    relationship, if any, is linear)."""
    if len(years) < 3:
        raise ValueError(f"year_distance_correlation: need >=3 points, got {len(years)}")
    rho, p_value = spearmanr(years, distances)
    return {"spearman_rho": float(rho), "p_value": float(p_value), "n": len(years)}


def assign_terciles(years: list[int]) -> tuple[list[str], dict[str, tuple[int, int]]]:
    """Splits `years` into 3 roughly-equal-count periods using the actual
    distribution's own 33rd/66th percentiles as boundaries (never a fixed
    calendar split like "2001-2008/2009-2016/2017-2024") -- so the periods
    reflect this corpus's own publication-year distribution, not an
    arbitrary external calendar grid. Returns (period_label_per_point,
    {period_label: (min_year, max_year)})."""
    years_arr = np.array(years)
    low_cut = np.percentile(years_arr, 33.33)
    high_cut = np.percentile(years_arr, 66.67)

    labels = []
    for year in years:
        if year <= low_cut:
            labels.append("early")
        elif year <= high_cut:
            labels.append("mid")
        else:
            labels.append("late")

    bounds = {}
    for period in ("early", "mid", "late"):
        period_years = [y for y, lbl in zip(years, labels) if lbl == period]
        if period_years:
            bounds[period] = (min(period_years), max(period_years))
    return labels, bounds


def period_centroids(vectors: np.ndarray, period_labels: list[str]) -> dict[str, np.ndarray]:
    centroids = {}
    for period in sorted(set(period_labels)):
        idxs = [i for i, lbl in enumerate(period_labels) if lbl == period]
        centroids[period] = vectors[idxs].mean(axis=0)
    return centroids


def period_pairwise_rows(centroids: dict[str, np.ndarray], bounds: dict[str, tuple[int, int]]) -> list[dict]:
    periods = sorted(centroids)
    rows = []
    for i, period_a in enumerate(periods):
        for period_b in periods[i + 1:]:
            euclidean = float(gac.euclidean_distances(centroids[period_a], centroids[period_b][np.newaxis, :])[0])
            cosine = float(gac.cosine_similarities(centroids[period_a], centroids[period_b][np.newaxis, :])[0])
            rows.append({
                "period_a": period_a, "period_a_years": bounds.get(period_a),
                "period_b": period_b, "period_b_years": bounds.get(period_b),
                "euclidean_distance": euclidean, "cosine_similarity": cosine,
            })
    return rows


def period_nearest_criterion_rows(
    centroids: dict[str, np.ndarray], criteria_points: list[dict], criteria_vectors: np.ndarray,
    bounds: dict[str, tuple[int, int]],
) -> list[dict]:
    """For each period's own centroid, the single nearest MIVILUDES
    criterion -- does which criterion literature discourse sits closest
    to shift across the three periods?"""
    rows = []
    for period, centroid in centroids.items():
        nearest = gac.nearest_points(centroid, criteria_points, criteria_vectors, NEAREST_K_CRITERION)[0]
        rows.append({
            "period": period, "period_years": bounds.get(period),
            "nearest_criterion_key": nearest["key"], "nearest_criterion_label": nearest["label"],
            "euclidean_distance": nearest["euclidean_distance"],
        })
    return rows


# ---------------------------------------------------------------------------
# Orchestration -- main() is the only part that touches disk.
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--shared-space-dir", type=Path, default=None,
                         help="Use a different pooled space (e.g. processed/shared_space_v2/) instead of v1's "
                              "processed/shared_space/; also switches the run-output root to a sibling "
                              "processed/analysis_v2/ directory so v1 and v2 runs are never mixed.")
    args = parser.parse_args()

    embedding_space_path, interview_prototypes_path, analysis_root = gac.resolve_space_paths(args.shared_space_dir)
    logger.info("Loading %s ...", embedding_space_path)
    shared_space = gac.load_shared_space(embedding_space_path)

    run_id, run_dir = gac.get_or_create_run_dir(args.run_id, analysis_root)
    gac.init_run_manifest(run_dir, shared_space, defaults={})
    gac.update_run_manifest(run_dir, MODULE_NAME, "running")
    out_dir = gac.module_run_dir(run_dir, MODULE_NAME)

    try:
        lit_idxs = gac.source_dataset_indices(shared_space.points, "literature")
        lit_points = [shared_space.points[i] for i in lit_idxs]
        lit_vectors = shared_space.vectors[lit_idxs]

        resolved_local_indices, years = years_for_points(lit_points)
        n_missing = len(lit_points) - len(resolved_local_indices)
        logger.info("%d/%d literature points have an extractable publication year (%d missing).",
                    len(resolved_local_indices), len(lit_points), n_missing)
        if n_missing:
            missing_docs = sorted({gac.key_document_id(lit_points[i]["key"])
                                    for i in range(len(lit_points)) if i not in set(resolved_local_indices)})
            logger.warning("Documents with no extractable year (excluded from this module): %s", missing_docs)

        resolved_vectors = lit_vectors[resolved_local_indices]
        resolved_points = [lit_points[i] for i in resolved_local_indices]

        lit_centroid = lit_vectors.mean(axis=0)  # over ALL literature points, not just year-resolved ones
        distances_to_lit_centroid = gac.euclidean_distances(lit_centroid, resolved_vectors)

        logger.info("Spearman correlation: year vs. distance-to-literature-centroid...")
        correlation = year_distance_correlation(years, list(distances_to_lit_centroid))
        logger.info("Spearman rho=%.4f, p=%.4f, n=%d", correlation["spearman_rho"], correlation["p_value"], correlation["n"])

        per_point_rows = [
            {
                "key": point["key"], "document_id": gac.key_document_id(point["key"]),
                "year": year, "distance_to_literature_centroid": float(dist),
            }
            for point, year, dist in zip(resolved_points, years, distances_to_lit_centroid)
        ]
        gac.write_csv(out_dir / "year_distance_by_point.csv", per_point_rows)

        logger.info("Assigning year terciles (boundaries from this corpus's own distribution)...")
        period_labels, bounds = assign_terciles(years)
        period_counts = {p: period_labels.count(p) for p in ("early", "mid", "late")}
        logger.info("Period sizes: %s; year ranges: %s", period_counts, bounds)

        centroids = period_centroids(resolved_vectors, period_labels)
        pairwise_rows = period_pairwise_rows(centroids, bounds)
        gac.write_csv(out_dir / "period_pairwise_centroid_distances.csv", pairwise_rows)

        criteria_idxs = gac.source_dataset_indices(shared_space.points, "miviludes_criteria")
        criteria_points = [shared_space.points[i] for i in criteria_idxs]
        criteria_vectors = shared_space.vectors[criteria_idxs]
        nearest_criterion_rows = period_nearest_criterion_rows(centroids, criteria_points, criteria_vectors, bounds)
        gac.write_csv(out_dir / "period_nearest_criterion.csv", nearest_criterion_rows)

        (out_dir / "GOVERNING_PRINCIPLE.txt").write_text(GOVERNING_PRINCIPLE + "\n", encoding="utf-8")

        gac.write_module_config(
            out_dir,
            input_sha256=shared_space.input_sha256,
            n_literature_points=len(lit_points),
            n_year_resolved=len(resolved_points),
            n_year_missing=n_missing,
            correlation=correlation,
            period_counts=period_counts,
            period_year_bounds=bounds,
            governing_principle=GOVERNING_PRINCIPLE,
            git_commit=gac.git_commit_hash(),
        )
        gac.update_run_manifest(run_dir, MODULE_NAME, "completed", output_dir=out_dir)
    except Exception as e:
        gac.update_run_manifest(run_dir, MODULE_NAME, "failed", error=str(e))
        raise

    print(f"\nDone. Run {run_id} -> {out_dir}")


if __name__ == "__main__":
    main()
