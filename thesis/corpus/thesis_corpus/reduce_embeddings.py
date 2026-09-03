"""Reduces a criterion_expressions.jsonl archive to a smaller working file.

Drops large/redundant fields (context_window, entity_anchor_vectors,
source_quote) and, by default, downsamples with PCA + MiniBatchKMeans
clustering so the result stays usable for interactive analysis and
visualization while preserving diversity, instead of just truncating.

The original archive is never modified -- this only ever reads --input and
writes --output.

Usage (from thesis/corpus/):
    python -m thesis_corpus.reduce_embeddings
    python -m thesis_corpus.reduce_embeddings --no-downsample
    python -m thesis_corpus.reduce_embeddings --input <path> --output <path> --source-type miviludes
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

CORPUS_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = CORPUS_DIR / "processed" / "literature" / "criterion_expressions.jsonl"
DEFAULT_OUTPUT = CORPUS_DIR / "processed" / "literature" / "criterion_expressions_reduced.jsonl"

EMBEDDING_DIM = 1024
KEEP_FIELDS = [
    "document_id", "chunk_index", "page_range", "embedding_text", "embedding_vector",
    "entity_anchors", "claim_mode", "epistemic_status", "attribution",
]

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("thesis_corpus.reduce_embeddings")


def _validate_and_reduce(raw: dict, line_no: int) -> dict | None:
    vector = raw.get("embedding_vector")
    if not isinstance(vector, list) or len(vector) != EMBEDDING_DIM or \
            not all(isinstance(v, (int, float)) for v in vector):
        logger.warning(
            "line %d: embedding_vector is not a list of %d numbers (doc=%s, chunk=%s) -- skipping",
            line_no, EMBEDDING_DIM, raw.get("document_id"), raw.get("chunk_index"),
        )
        return None

    missing = [f for f in KEEP_FIELDS if f not in raw]
    if missing:
        logger.warning(
            "line %d: missing required field(s) %s (doc=%s) -- skipping",
            line_no, missing, raw.get("document_id"),
        )
        return None

    return {field: raw[field] for field in KEEP_FIELDS}


def read_and_reduce(input_path: Path, source_type: str) -> tuple[list[dict], int]:
    """Returns (reduced_items, items_read). Streams the raw file line by
    line -- only the much smaller reduced records accumulate in memory,
    never the raw file's full text at once."""
    reduced_items = []
    items_read = 0
    with open(input_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            items_read += 1
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("line %d: malformed JSON -- skipping", line_no)
                continue
            reduced = _validate_and_reduce(raw, line_no)
            if reduced is not None:
                reduced["source_type"] = source_type
                reduced_items.append(reduced)
    return reduced_items, items_read


def downsample(
    items: list[dict], pca_dim: int, n_clusters: int, samples_per_cluster: int, seed: int,
) -> list[dict]:
    n_samples = len(items)
    vectors = np.array([it["embedding_vector"] for it in items], dtype=np.float32)

    effective_pca_dim = min(pca_dim, n_samples, vectors.shape[1])
    if effective_pca_dim < pca_dim:
        logger.warning(
            "pca-dim %d exceeds available samples/features (%d, %d) -- using %d",
            pca_dim, n_samples, vectors.shape[1], effective_pca_dim,
        )
    reduced_vectors = PCA(n_components=effective_pca_dim, random_state=seed).fit_transform(vectors)

    effective_n_clusters = min(n_clusters, n_samples)
    if effective_n_clusters < n_clusters:
        logger.warning("n-clusters %d exceeds sample count %d -- using %d", n_clusters, n_samples, effective_n_clusters)
    labels = MiniBatchKMeans(n_clusters=effective_n_clusters, random_state=seed, n_init="auto").fit_predict(reduced_vectors)
    logger.info("clusters formed: %d", effective_n_clusters)

    rng = np.random.default_rng(seed)
    clusters: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(idx)

    selected_indices = []
    for label, indices in clusters.items():
        k = min(samples_per_cluster, len(indices))
        chosen = rng.choice(indices, size=k, replace=False)
        selected_indices.extend(int(i) for i in chosen)

    selected_indices.sort()  # preserve original file order for traceability
    return [items[i] for i in selected_indices]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pca-dim", type=int, default=100)
    parser.add_argument("--n-clusters", type=int, default=1000)
    parser.add_argument("--samples-per-cluster", type=int, default=5)
    parser.add_argument("--no-downsample", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-type", default="academic_literature",
                         help="Value for the derived source_type field (e.g. 'miviludes', 'interviews').")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    if args.no_downsample:
        # Fully streaming: read one line, write one line, no accumulation.
        items_read = 0
        items_kept = 0
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.input, encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
            for line_no, line in enumerate(fin, 1):
                line = line.strip()
                if not line:
                    continue
                items_read += 1
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("line %d: malformed JSON -- skipping", line_no)
                    continue
                reduced = _validate_and_reduce(raw, line_no)
                if reduced is None:
                    continue
                reduced["source_type"] = args.source_type
                fout.write(json.dumps(reduced, ensure_ascii=False) + "\n")
                items_kept += 1

        logger.info("items_read: %d", items_read)
        logger.info("items_after_field_reduction: %d", items_kept)
        logger.info("downsampling: skipped (--no-downsample)")
        logger.info("items sampled: %d", items_kept)
        logger.info("output: %s", args.output)
        return

    items, items_read = read_and_reduce(args.input, args.source_type)
    logger.info("items_read: %d", items_read)
    logger.info("items_after_field_reduction: %d", len(items))

    if not items:
        raise SystemExit("No valid items after field reduction -- nothing to downsample.")

    sampled = downsample(items, args.pca_dim, args.n_clusters, args.samples_per_cluster, args.seed)
    logger.info("items sampled: %d", len(sampled))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for item in sampled:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info("output: %s", args.output)


if __name__ == "__main__":
    main()
