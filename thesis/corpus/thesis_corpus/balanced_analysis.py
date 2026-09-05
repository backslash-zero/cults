"""Corpus-imbalance mitigation for the shared cross-corpus embedding space.

Literature (39,236 expression points, before this session's dedup/length
filter -- see build_shared_space.py) is roughly 97% of all expression
points, against MIVILUDES's 914 and interviews' 230. Any raw, unweighted
statistic computed over the pooled space (a grand centroid, "nearest
concept to the corpus as a whole") is really a statistic about literature,
not about the three epistemologies in balance.

This module doesn't touch embedding_space.jsonl or the PCA fit itself --
per the deliberate choice to keep all 39k literature points for
visualization and only correct for imbalance in *quantitative* analysis.
It provides two independent tools for that purpose:

  - `write_balanced_literature_sample()`: a CLI/function that writes a
    stratified-by-document literature subsample, same row shape as
    embedding_space.jsonl, so it's a drop-in subset for any analysis that
    wants literature at roughly MIVILUDES+interviews' order of magnitude.
  - `weighted_centroid()` / `per_corpus_centroids()`: reusable functions
    for computing a grand centroid where each of the three expression
    corpora contributes equal total weight, regardless of point count --
    no sampling involved, every point still counts, just not evenly by
    corpus size.

Usage as a library:
    from thesis_corpus.balanced_analysis import weighted_centroid
    centroid = weighted_centroid("processed/shared_space/embedding_space.jsonl")

Usage as a CLI (writes the balanced literature sample):
    python -m thesis_corpus.balanced_analysis
    python -m thesis_corpus.balanced_analysis --sample-size 3000 --seed 7
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

CORPUS_DIR = Path(__file__).resolve().parent.parent
SHARED_SPACE_DIR = CORPUS_DIR / "processed" / "shared_space"
EMBEDDING_SPACE_PATH = SHARED_SPACE_DIR / "embedding_space.jsonl"
BALANCED_SAMPLE_PATH = SHARED_SPACE_DIR / "literature_balanced_sample.jsonl"

DEFAULT_SAMPLE_SIZE = 2500
DEFAULT_SEED = 42
EXPRESSION_CORPORA = ("literature", "miviludes", "interviews")


def _document_id(key: str) -> str:
    """Expression-point keys are "document_id:chunk_index" -- see
    build_shared_space.py's load_corpus_points. Splits on the last ':'
    since document_id itself may contain no colons in practice, but
    rsplit is the safe direction regardless."""
    return key.rsplit(":", 1)[0]


def write_balanced_literature_sample(
    embedding_space_path: Path = EMBEDDING_SPACE_PATH,
    output_path: Path = BALANCED_SAMPLE_PATH,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> dict[str, int]:
    """Stratified random sample of `literature` expression points, grouped
    by document_id so no single (chunk-heavy) document is over-represented
    relative to its share of the full corpus. Writes `sample_size` rows,
    same shape as embedding_space.jsonl, to `output_path`.

    Returns {"total_literature_points": N, "n_documents": M, "sample_size": S}
    for logging/verification."""
    by_document: dict[str, list[dict]] = defaultdict(list)
    with open(embedding_space_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["source_dataset"] != "literature":
                continue
            by_document[_document_id(row["key"])].append(row)

    total_points = sum(len(rows) for rows in by_document.values())
    rng = random.Random(seed)

    # Proportional allocation: each document's share of the sample matches
    # its share of the full literature corpus, rounded down, with leftover
    # slots (from rounding) distributed round-robin to the largest
    # documents first so the total lands exactly on sample_size.
    documents = sorted(by_document.items(), key=lambda item: -len(item[1]))
    allocations = {
        doc_id: int(len(rows) / total_points * sample_size)
        for doc_id, rows in documents
    }
    allocated = sum(allocations.values())
    shortfall = sample_size - allocated
    for doc_id, _rows in documents[:shortfall]:
        allocations[doc_id] += 1

    sampled_rows = []
    for doc_id, rows in documents:
        n = min(allocations[doc_id], len(rows))
        sampled_rows.extend(rng.sample(rows, n))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in sampled_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "total_literature_points": total_points,
        "n_documents": len(by_document),
        "sample_size": len(sampled_rows),
    }


def per_corpus_centroids(
    embedding_space_path: Path = EMBEDDING_SPACE_PATH,
    corpus_names: tuple[str, ...] = EXPRESSION_CORPORA,
) -> dict[str, np.ndarray]:
    """Each named corpus's own mean shared_space_vector. Public on its own
    (not just an internal step of weighted_centroid) since per-corpus
    centroids are useful independently -- e.g. comparing how far apart the
    three epistemologies' centers already sit."""
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = defaultdict(int)
    with open(embedding_space_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            corpus = row["source_dataset"]
            if corpus not in corpus_names:
                continue
            vector = np.array(row["shared_space_vector"], dtype=np.float64)
            sums[corpus] = sums.get(corpus, np.zeros_like(vector)) + vector
            counts[corpus] += 1

    missing = set(corpus_names) - set(sums)
    if missing:
        raise SystemExit(f"No points found for corpus/corpora: {sorted(missing)}")

    return {corpus: sums[corpus] / counts[corpus] for corpus in corpus_names}


def weighted_centroid(
    embedding_space_path: Path = EMBEDDING_SPACE_PATH,
    corpus_names: tuple[str, ...] = EXPRESSION_CORPORA,
) -> np.ndarray:
    """The grand centroid across `corpus_names`, with each corpus
    contributing equal total weight (1/len(corpus_names)) regardless of
    point count -- literature's 39,236 points don't outweigh MIVILUDES's
    914 or interviews' 230 the way an unweighted mean over all points
    combined would. Compare against a plain unweighted mean (average every
    point in corpus_names directly) to see how much the imbalance actually
    moves a raw statistic."""
    centroids = per_corpus_centroids(embedding_space_path, corpus_names)
    return np.mean(list(centroids.values()), axis=0)


def _raw_unweighted_mean(embedding_space_path: Path, corpus_names: tuple[str, ...]) -> np.ndarray:
    """The plain mean over every point in corpus_names combined, with no
    per-corpus weighting -- one pass through the file, for comparison
    against weighted_centroid()."""
    total = None
    count = 0
    with open(embedding_space_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["source_dataset"] not in corpus_names:
                continue
            vector = np.array(row["shared_space_vector"], dtype=np.float64)
            total = vector if total is None else total + vector
            count += 1
    return total / count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=EMBEDDING_SPACE_PATH)
    parser.add_argument("--output", type=Path, default=BALANCED_SAMPLE_PATH)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    stats = write_balanced_literature_sample(args.input, args.output, args.sample_size, args.seed)
    print(f"Literature: {stats['total_literature_points']} points across {stats['n_documents']} documents.")
    print(f"Wrote {stats['sample_size']}-row stratified sample -> {args.output}")

    grand_raw = _raw_unweighted_mean(args.input, EXPRESSION_CORPORA)
    grand_weighted = weighted_centroid(args.input)
    shift = float(np.linalg.norm(grand_raw - grand_weighted))
    print(f"\nRaw (unweighted, point-count-dominated) vs. equal-corpus-weighted grand centroid "
          f"distance: {shift:.4f} (0 would mean the imbalance doesn't matter; nonzero confirms it does).")


if __name__ == "__main__":
    main()
