"""Project the shared cross-corpus embedding space down to 3-D for visualization.

`build_shared_space.py` pools every dataset into one shared 390-d coordinate
system (`processed/shared_space/embedding_space.jsonl`). This script reads
that file once and writes three separate 3-d projections of it, one per
method, so the shared space can actually be plotted:

  - PCA: `shared_space_vector` is already full-rank PCA output from
    build_shared_space.py, with columns ordered by descending explained
    variance and mutually uncorrelated. The first 3 principal components of
    this space are therefore exactly the first 3 columns of
    `shared_space_vector` -- refitting a fresh 3-component PCA on top would
    return the same axes (up to sign), so this is a slice, not a fit.
  - UMAP (n_neighbors=20, min_dist=0.2, euclidean): a nonlinear projection
    that tends to preserve local neighborhood structure.
  - t-SNE (perplexity=30, euclidean, PCA-initialized): a nonlinear
    projection tuned for revealing cluster structure at the cost of global
    distances being less meaningful.

All three are fit directly on the 390-d shared-space vectors (already
standardized + PCA'd upstream; no further scaling applied here).

Never modifies embedding_space.jsonl; only ever writes new files under
processed/shared_space/.

Usage (from thesis/corpus/):
    python -m thesis_corpus.visualize_3d
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import umap
from sklearn.manifold import TSNE

CORPUS_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = CORPUS_DIR / "processed"
SHARED_SPACE_DIR = PROCESSED_DIR / "shared_space"

INPUT_PATH = SHARED_SPACE_DIR / "embedding_space.jsonl"
# Written alongside embedding_space.jsonl -- same convention as
# variance_curve.{csv,json,png}: derived-from-the-shared-space output lives
# next to the space it was derived from.
PCA_OUTPUT_PATH = SHARED_SPACE_DIR / "visualization_pca_3d.jsonl"
UMAP_OUTPUT_PATH = SHARED_SPACE_DIR / "visualization_umap_3d.jsonl"
TSNE_OUTPUT_PATH = SHARED_SPACE_DIR / "visualization_tsne_3d.jsonl"

N_COMPONENTS = 3

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("thesis_corpus.visualize_3d")


def load_shared_space(path: Path) -> tuple[list[dict], np.ndarray]:
    """Streams the input JSONL line by line -- only the metadata (small) and
    the resulting vectors matrix (the one large object) end up in memory."""
    points: list[dict] = []
    vectors: list[list[float]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            points.append({
                "source_dataset": item["source_dataset"],
                "key": item["key"],
                "label": item["label"],
                "label_en": item.get("label_en"),
                "attribution": item.get("attribution"),
                "claim_mode": item.get("claim_mode"),
                "epistemic_status": item.get("epistemic_status"),
                "response_rank": item.get("response_rank"),
                "mention_distribution": item.get("mention_distribution"),
            })
            vectors.append(item["shared_space_vector"])
    return points, np.array(vectors, dtype=np.float64)


def write_projection(path: Path, points: list[dict], coords: np.ndarray, field_name: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for p, coord in zip(points, coords):
            out = dict(p)
            out[field_name] = coord.tolist()
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--umap-neighbors", type=int, default=20)
    parser.add_argument("--umap-min-dist", type=float, default=0.2)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Missing shared embedding space: {args.input}")

    logger.info("Loading %s ...", args.input)
    points, vectors = load_shared_space(args.input)
    logger.info("Loaded %d points, %d-d vectors", len(points), vectors.shape[1])

    SHARED_SPACE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("PCA (3-D): slicing the first %d columns of the already-PCA'd shared space...", N_COMPONENTS)
    t0 = time.perf_counter()
    pca_coords = vectors[:, :N_COMPONENTS]
    write_projection(PCA_OUTPUT_PATH, points, pca_coords, "pca_3d_vector")
    logger.info("PCA: %d points written in %.1fs -> %s", len(points), time.perf_counter() - t0, PCA_OUTPUT_PATH)

    logger.info(
        "UMAP (3-D): n_neighbors=%d, min_dist=%.2f, metric=euclidean ...",
        args.umap_neighbors, args.umap_min_dist,
    )
    t0 = time.perf_counter()
    umap_coords = umap.UMAP(
        n_components=N_COMPONENTS,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        random_state=args.seed,
        metric="euclidean",
    ).fit_transform(vectors)
    write_projection(UMAP_OUTPUT_PATH, points, umap_coords, "umap_3d_vector")
    logger.info("UMAP: %d points written in %.1fs -> %s", len(points), time.perf_counter() - t0, UMAP_OUTPUT_PATH)

    logger.info("t-SNE (3-D): perplexity=%.1f, metric=euclidean, init=pca ...", args.tsne_perplexity)
    t0 = time.perf_counter()
    tsne_coords = TSNE(
        n_components=N_COMPONENTS,
        perplexity=args.tsne_perplexity,
        random_state=args.seed,
        metric="euclidean",
        init="pca",
        learning_rate="auto",
    ).fit_transform(vectors)
    write_projection(TSNE_OUTPUT_PATH, points, tsne_coords, "tsne_3d_vector")
    logger.info("t-SNE: %d points written in %.1fs -> %s", len(points), time.perf_counter() - t0, TSNE_OUTPUT_PATH)

    print(f"\nDone. {len(points)} points projected to 3-D by each of PCA, UMAP, t-SNE.")
    print(f"Output: {PCA_OUTPUT_PATH}, {UMAP_OUTPUT_PATH}, {TSNE_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
