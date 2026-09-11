"""Emergent entities for the exhaustive interview archive, standalone --
single-corpus, no pooling, no PCA, and never touches shared_space_v2/ or
anything it feeds (Emergent_Entities_v2 appendix, Geometric_Analysis_Draft_v3).

Reuses build_shared_space.load_emergent_entities() (already standalone and
single-corpus-capable) directly against one run's
processed/interviews_full/interviews/run_<tag>/criterion_expressions.jsonl
(+ chunk_terms.jsonl / domain_term_vectors.jsonl for the decoupled
domain_terms mention channel -- see embed_domain_terms.py's docstring for
why that channel matters for entity recall), then reuses
export_emergent_entities.py's CSV/LaTeX-row writers to render the ranking.

Usage (from thesis/corpus/):
    python -m thesis_corpus.export_interview_emergent_entities --run-tag 20260911
    python -m thesis_corpus.export_interview_emergent_entities --run-tag 20260911 --min-mentions 2
"""
from __future__ import annotations

import argparse
from pathlib import Path

from thesis_corpus.build_shared_space import load_cited_author_surnames, load_emergent_entities
from thesis_corpus.export_emergent_entities import write_csv, write_latex_rows
from thesis_corpus.pilot_v2_literature import PROCESSED_ROOT

DEFAULT_OUT_ROOT = PROCESSED_ROOT / "interviews_full"


def points_to_rows(points: list[dict]) -> list[dict]:
    """Same shape export_emergent_entities.rank_emergent_entities() returns
    (label/total/literature/miviludes/interviews), sorted by total mentions
    descending -- so write_csv/write_latex_rows work unmodified. This is a
    single-corpus archive, so literature/miviludes are always 0 here; the
    columns are kept for schema consistency with the pooled ranking, not
    because those corpora contributed anything to THIS ranking."""
    rows = []
    for point in points:
        mentions = point.get("mention_distribution") or {}
        rows.append({
            "label": point["label"],
            "total": sum(mentions.values()),
            "literature": mentions.get("literature", 0),
            "miviludes": mentions.get("miviludes", 0),
            "interviews": mentions.get("interviews", 0),
        })
    rows.sort(key=lambda r: r["total"], reverse=True)
    return rows


def run(run_dir: Path, min_mentions: int, top_n: int) -> list[dict]:
    archive_path = run_dir / "criterion_expressions.jsonl"
    if not archive_path.exists():
        raise SystemExit(f"{archive_path} missing -- run extract_interviews_full then embed_v2 first")
    chunk_terms_path = run_dir / "chunk_terms.jsonl"
    term_vectors_path = run_dir / "domain_term_vectors.jsonl"
    domain_term_paths = {"interviews": (chunk_terms_path, term_vectors_path)} if term_vectors_path.exists() else None

    points = load_emergent_entities(
        archive_paths={"interviews": archive_path},
        min_mentions_by_corpus={"interviews": min_mentions},
        domain_term_paths=domain_term_paths,
        cited_author_surnames=load_cited_author_surnames(),
    )
    rows = points_to_rows(points)

    csv_path = run_dir / "emergent_entities_ranked.csv"
    latex_path = run_dir / "top_100_rows.tex"
    write_csv(rows, csv_path)
    write_latex_rows(rows, top_n, latex_path)
    print(f"{len(rows)} emergent entities ranked (min_mentions={min_mentions}).")
    print(f"Wrote full ranking -> {csv_path}")
    print(f"Wrote top {top_n} LaTeX rows -> {latex_path}")
    if rows:
        print("Top 10:")
        for r in rows[:10]:
            print(f"  {r['label']}: {r['total']} mentions")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--min-mentions", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=100)
    args = parser.parse_args()
    run_dir = args.out_root / "interviews" / f"run_{args.run_tag}"
    run(run_dir, args.min_mentions, args.top_n)


if __name__ == "__main__":
    main()
