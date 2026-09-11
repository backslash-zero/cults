"""Do people's spontaneous PROTOTYPE examples (the interview protocol's own
first free-association exemplar, hand-reviewed in
interviews/metadata/initial_exemplars.csv, per the thesis's own
`\\S Family resemblance and prototype theory` / `\\S Interviews` framing)
land near well-covered literature clusters, or in the entity-gap regions
found in 06_Literature_Clusters.md?

A sharper version of the folk-vs-expert-concept finding already in
Results_Draft.md: that finding used every interview segment (including
filler); this restricts to exactly the 25 theoretically-privileged
prototype exemplars the interview protocol was designed to elicit.

RAW-VECTOR RESOLUTION, NOT THE OLD KEY-BASED JOIN. `interview_prototypes.jsonl`
(built by build_interview_prototype_layer.py) only ever stores a PCA'd
shared_space_vector, resolved via document_id+source_expression_key against
processed/interviews/criterion_expressions.jsonl -- an older archive, not
this session's processed/interviews_full/interviews/run_20260912/. To stay
self-consistent with every other raw-space analysis this session, each
prototype's raw vector is instead resolved by FUZZY TEXT MATCH
(difflib.SequenceMatcher) of transcript_initial_exemplar_text against that
document's own verbatim_expression segments in the new exhaustive archive
-- checked directly, not assumed to just work: the two archives segment
differently (old = chunk/span extraction, new = one segment per sentence),
so an exact-string match found ZERO of 2 spot-checked rows; fuzzy
containment-aware matching is required. Every resolution's match score is
logged and written out so it can be spot-checked before trusting the
downstream distances (low-confidence matches are flagged, never silently
trusted).

Usage (from thesis/corpus/):
    python -m thesis_corpus.analyze_interview_prototypes
"""
from __future__ import annotations

import argparse
import csv
import difflib
import logging
from pathlib import Path

import numpy as np

from thesis_corpus import analyze_corpus_centroids as acc
from thesis_corpus import analyze_entity_topology as aet
from thesis_corpus import analyze_literature_clusters as alc
from thesis_corpus import build_shared_space as bss
from thesis_corpus import geometric_analysis_common as gac

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("thesis_corpus.analyze_interview_prototypes")

EXEMPLARS_CSV_PATH = bss.CORPUS_DIR / "interviews" / "metadata" / "initial_exemplars.csv"
MIN_MATCH_RATIO = 0.4  # below this, the resolution is flagged low-confidence, not discarded
CLUSTER_MIN_SIZE = 20  # must match analyze_literature_clusters.py's own default for labels to line up


def load_reviewed_exemplars(path: Path = EXEMPLARS_CSV_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r["review_status"] == "reviewed"]


def best_match_in_document(query_texts: list[str], candidates: list[dict]) -> tuple[dict | None, float]:
    """`candidates` is this document's own interview points (each carrying
    "label" = its verbatim_expression). Tries EACH of `query_texts` against
    every candidate and keeps the single best (point, ratio) pair --
    critical because the CSV's two text fields aren't interchangeable for
    matching: `source_expression_label` is a precise single-sentence
    verbatim span (matches a single new segment well), while
    `transcript_initial_exemplar_text` is a hand-composed, "..."-joined
    multi-part quote spanning several original sentences (matches poorly
    against any ONE new segment even when the label itself is an exact
    hit elsewhere). Caught on a real spot check, not assumed: querying
    with only the composite text gave b2-aug13-1832 a 0.43-ratio match to
    the WRONG segment, when its label field is an exact (ratio 1.0) match
    to segment 9 of that same document -- trying both fields and keeping
    the best fixes this.

    Returns (best-matching point, ratio) via difflib.SequenceMatcher,
    boosted to at least 0.8 on substring containment either direction."""
    best_point, best_ratio = None, 0.0
    for query_text in query_texts:
        if not query_text:
            continue
        q = query_text.lower().strip()
        for p in candidates:
            c = p["label"].lower().strip()
            ratio = difflib.SequenceMatcher(None, q, c).ratio()
            if q in c or c in q:
                ratio = max(ratio, 0.8)  # containment is strong evidence even if lengths differ a lot
            if ratio > best_ratio:
                best_point, best_ratio = p, ratio
    return best_point, best_ratio


def resolve_prototypes(exemplar_rows: list[dict], points: list[dict], vectors: np.ndarray) -> list[dict]:
    """For each reviewed exemplar row, resolve its raw vector against the
    new exhaustive interview archive within the SAME document_id only."""
    # points' own document_id isn't stored as a field (only in `key`) for
    # interviews via load_corpus_points_v2 -- key format is
    # "document_id:chunk_index"; gac.key_document_id does the rsplit correctly.
    interview_idx = acc.group_indices(points, "interviews")
    by_doc: dict[str, list[dict]] = {}
    for i in interview_idx:
        doc_id = gac.key_document_id(points[i]["key"])
        by_doc.setdefault(doc_id, []).append({"index": i, "label": points[i]["label"], "key": points[i]["key"]})

    resolved = []
    for row in exemplar_rows:
        doc_id = row["document_id"]
        candidates = by_doc.get(doc_id, [])
        # Precise label first, composite exemplar text second -- see
        # best_match_in_document's own docstring for why both are tried.
        query_texts = [row["source_expression_label"], row["transcript_initial_exemplar_text"]]
        match, ratio = best_match_in_document(query_texts, candidates) if candidates else (None, 0.0)
        resolved.append({
            "document_id": doc_id,
            "query_text": row["source_expression_label"] or row["transcript_initial_exemplar_text"],
            "matched_label": match["label"] if match else None,
            "matched_index": match["index"] if match else None,
            "match_ratio": round(ratio, 3),
            "low_confidence": ratio < MIN_MATCH_RATIO or match is None,
        })
    return resolved


def cluster_vectors_and_labels(points: list[dict], vectors: np.ndarray, min_cluster_size: int):
    lit_idx = acc.group_indices(points, "literature")
    lit_points = [points[i] for i in lit_idx]
    lit_vectors = vectors[lit_idx]
    reduced = alc.reduce_for_clustering(lit_vectors, alc.UMAP_N_COMPONENTS)
    labels = aet.compute_clusters(reduced, min_cluster_size)
    return lit_points, lit_vectors, labels


def gap_cluster_ids_from_summary(points: list[dict], vectors: np.ndarray, lit_points, lit_vectors, labels,
                                  entity_points, entity_vectors, gap_percentile: float) -> set[int]:
    """Re-derives which cluster ids are flagged gaps, same rule as
    analyze_literature_clusters.py's own main() (p75 of nearest-entity
    distance) -- kept in lockstep with that module rather than reading its
    CSV, so this stays correct even if that module's run parameters change."""
    cluster_ids = sorted(set(labels) - {-1})
    distances = {}
    for cid in cluster_ids:
        member_idxs = [i for i, l in enumerate(labels) if l == cid]
        centroid = lit_vectors[member_idxs].mean(axis=0)
        nearest = gac.nearest_points(centroid, entity_points, entity_vectors, k=1)[0]
        distances[cid] = nearest["euclidean_distance"]
    threshold = float(np.percentile(list(distances.values()), gap_percentile))
    return {cid for cid, d in distances.items() if d >= threshold}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lit-mivi-run-tag", default="20260910")
    parser.add_argument("--interviews-archive-dir", type=Path,
                         default=bss.PROCESSED_DIR / "interviews_full" / "interviews" / "run_20260912")
    parser.add_argument("--min-cluster-size", type=int, default=CLUSTER_MIN_SIZE)
    parser.add_argument("--gap-percentile", type=float, default=75.0)
    parser.add_argument("--out-dir", type=Path, default=bss.PROCESSED_DIR / "analysis_raw" / "interview_prototypes")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw_points = acc.load_raw_points(args.lit_mivi_run_tag, args.interviews_archive_dir)
    vectors = np.array([p["vector"] for p in raw_points], dtype=np.float64)
    points = [{k: v for k, v in p.items() if k != "vector"} for p in raw_points]
    logger.info("Loaded %d raw points", len(points))

    exemplar_rows = load_reviewed_exemplars()
    logger.info("%d reviewed prototype exemplars", len(exemplar_rows))
    resolved = resolve_prototypes(exemplar_rows, points, vectors)
    n_low_conf = sum(r["low_confidence"] for r in resolved)
    logger.info("Resolved %d/%d; %d flagged low-confidence (ratio<%.1f)",
                sum(r["matched_index"] is not None for r in resolved), len(resolved), n_low_conf, MIN_MATCH_RATIO)
    for r in resolved:
        flag = " [LOW-CONFIDENCE]" if r["low_confidence"] else ""
        logger.info("  %-16s ratio=%.2f%s  %r -> %r", r["document_id"], r["match_ratio"], flag,
                    r["query_text"][:50], (r["matched_label"] or "NO MATCH")[:50])

    lit_points, lit_vectors, labels = cluster_vectors_and_labels(points, vectors, args.min_cluster_size)
    entity_idx = acc.group_indices(points, "entities_literature")
    entity_points = [points[i] for i in entity_idx]
    entity_vectors = vectors[entity_idx]
    gap_ids = gap_cluster_ids_from_summary(points, vectors, lit_points, lit_vectors, labels,
                                            entity_points, entity_vectors, args.gap_percentile)
    logger.info("%d/%d literature clusters are gap clusters (p%.0f)", len(gap_ids), len(set(labels) - {-1}), args.gap_percentile)

    cluster_ids = sorted(set(labels) - {-1})
    cluster_centroid_points = [{"key": str(cid), "label": f"cluster {cid}" + (" [GAP]" if cid in gap_ids else "")} for cid in cluster_ids]
    cluster_centroid_vectors = np.array([lit_vectors[[i for i, l in enumerate(labels) if l == cid]].mean(axis=0) for cid in cluster_ids])

    rows = []
    n_gap, n_nongap, n_unresolved = 0, 0, 0
    for r in resolved:
        if r["matched_index"] is None:
            rows.append({**r, "nearest_cluster_id": None, "is_gap_cluster": None,
                         "cluster_euclidean_distance": None, "cluster_cosine_similarity": None})
            n_unresolved += 1
            continue
        query_vector = vectors[r["matched_index"]]
        nearest = gac.nearest_points(query_vector, cluster_centroid_points, cluster_centroid_vectors, k=1)[0]
        nearest_cid = int(nearest["key"])
        is_gap = nearest_cid in gap_ids
        n_gap += is_gap
        n_nongap += not is_gap
        rows.append({**r, "nearest_cluster_id": nearest_cid, "is_gap_cluster": is_gap,
                     "cluster_euclidean_distance": round(nearest["euclidean_distance"], 4),
                     "cluster_cosine_similarity": round(nearest["cosine_similarity"], 4)})

    out_path = args.out_dir / "prototype_vs_clusters.csv"
    gac.write_csv(out_path, rows)
    logger.info("Prototypes landing in a gap cluster: %d/%d resolved (%d unresolved) -- "
                "overall corpus gap rate is %d/%d clusters",
                n_gap, n_gap + n_nongap, n_unresolved, len(gap_ids), len(cluster_ids))
    print(f"Done. {out_path}")


if __name__ == "__main__":
    main()
