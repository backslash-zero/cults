"""Side-by-side comparison of 2-D projection techniques for the
centroid + nearest-neighbours plots in analyze_corpus_centroids.py, built
to answer a concrete complaint: even after switching from a global PCA-2D
slice to a UMAP-2D fit jointly including the centroid (see that module's
own docstring), some groups (e.g. "dictionary") still show the 10 nearest
points scattered around the centroid star rather than hugging it, while
others (e.g. "rapport") look tight. This module does not pick a winner --
it renders the SAME local neighbourhood (the centroid + the
`local_context_n` real points nearest it, raw bge-m3, cosine) through five
different projections side by side, so that can be judged by eye instead
of asserted.

Diagnostic worth reading before the plots: for a purely geometric reason,
"the 10 nearest points to a centroid" need not be near EACH OTHER -- they
can each individually sit close to the centroid while pointing in very
different directions from it (a "shell" around the centroid, not a
"ball"). No 2-D projection can make points that are genuinely spread
around a sphere look like a tight cluster without lying about the
geometry. Checked directly: mean pairwise cosine similarity among
"dictionary"'s own top-10 is 0.73, "rapport"'s is 0.64 -- if anything
dictionary's top-10 are MORE mutually similar than rapport's, so mutual
spread alone doesn't fully explain why dictionary looks worse; this is
exactly why an actual multi-technique comparison is more trustworthy here
than a single method's picture.

Five techniques, all fit JOINTLY on (local points + centroid) -- never a
real point's own fit with the centroid merely transformed in afterward
(see analyze_corpus_centroids.py's own docstring for why that specifically
was misleading):
  - pca_local: PCA(n_components=2) on this LOCAL neighbourhood only (not
    the whole group/corpus) -- a fresh local linear fit, different from
    the old global-PCA-slice approach this toolkit already moved away
    from, included here as the "cheap linear baseline" every other
    technique should be compared against.
  - mds: classical/metric multidimensional scaling on the precomputed
    cosine-distance matrix -- the one technique here whose OBJECTIVE is
    literally "make 2-D distances match the real distance matrix as
    closely as possible" (unlike PCA, which maximizes variance, or UMAP/
    t-SNE, which preserve local neighbour TOPOLOGY, not metric distance).
    The most theoretically apt technique for "does near-in-2-D mean
    near-in-reality", included as the principled comparison point.
  - umap_default: same params analyze_corpus_centroids.py currently uses
    (n_neighbors=min(15,n), min_dist=0.1, metric="cosine").
  - umap_tight: same but min_dist=0.0, n_neighbors halved -- tests
    whether more aggressive local-cluster-tightening params change the
    picture.
  - tsne: metric="cosine", perplexity capped below n_samples -- the other
    standard neighbour-preserving nonlinear technique, frequently reported
    to produce visually tighter clusters than UMAP at the cost of
    distorting relative cluster sizes/distances more.

Usage (from thesis/corpus/):
    python -m thesis_corpus.compare_projection_techniques
    python -m thesis_corpus.compare_projection_techniques --groups dictionary rapport entities_all
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE

from thesis_corpus import analyze_corpus_centroids as acc
from thesis_corpus import build_shared_space as bss
from thesis_corpus import geometric_analysis_common as gac

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("thesis_corpus.compare_projection_techniques")

DEFAULT_GROUPS = ["dictionary", "rapport", "entities_all"]
SEED = 42


def pca_local(combined: np.ndarray) -> np.ndarray:
    return PCA(n_components=2, random_state=SEED).fit_transform(combined)


def mds_cosine(combined: np.ndarray) -> np.ndarray:
    norm = combined / np.linalg.norm(combined, axis=1, keepdims=True)
    cosine_dist = 1 - norm @ norm.T
    np.fill_diagonal(cosine_dist, 0.0)
    cosine_dist = np.clip(cosine_dist, 0, None)  # guard tiny negative floating-point noise
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=SEED,
              normalized_stress=False, n_init=4)
    return mds.fit_transform(cosine_dist)


def umap_joint(combined: np.ndarray, n_neighbors: int, min_dist: float) -> np.ndarray:
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                         random_state=SEED, metric="cosine")
    return reducer.fit_transform(combined)


def tsne_cosine(combined: np.ndarray) -> np.ndarray:
    perplexity = min(30, len(combined) - 1)
    tsne = TSNE(n_components=2, metric="cosine", perplexity=perplexity, random_state=SEED, init="random")
    return tsne.fit_transform(combined)


def project_all(local_vectors: np.ndarray, centroid: np.ndarray) -> dict[str, np.ndarray]:
    """Returns {technique: coords}, coords.shape == (len(local_vectors)+1, 2),
    last row is always the centroid -- one consistent contract every
    technique above satisfies by construction (all fit jointly)."""
    combined = np.vstack([local_vectors, centroid.reshape(1, -1)])
    n = len(local_vectors)
    return {
        "pca_local": pca_local(combined),
        "mds": mds_cosine(combined),
        "umap_default": umap_joint(combined, n_neighbors=min(15, n), min_dist=0.1),
        "umap_tight": umap_joint(combined, n_neighbors=max(2, min(15, n) // 2), min_dist=0.0),
        "tsne": tsne_cosine(combined),
    }


def mean_pairwise_cosine(vectors: np.ndarray) -> float:
    norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    sims = norm @ norm.T
    iu = np.triu_indices(len(vectors), k=1)
    return float(sims[iu].mean())


def plot_comparison(out_dir: Path, group: str, points: list[dict], vectors: np.ndarray,
                     indices: list[int], centroid: np.ndarray, neighbors: list[dict],
                     local_context_n: int) -> Path:
    # Positional (argsort), never key-based -- see analyze_corpus_centroids.py's
    # plot_group docstring/comment for why: group_points' "key" is not
    # guaranteed unique (MIVILUDES especially pulls several expressions
    # from one chunk), so matching local-context points back to their rank
    # by key silently collapses/misassigns points whenever a key repeats.
    group_points = [points[i] for i in indices]
    group_vectors = vectors[indices]
    dist = gac.euclidean_distances(centroid, group_vectors)
    order = np.argsort(dist)
    local_context_n = min(local_context_n, len(order))
    local_order = order[:local_context_n]
    local_vectors = group_vectors[local_order]
    k = len(neighbors)

    neighbor_dist_cos = mean_pairwise_cosine(group_vectors[order[:k]])
    logger.info("%-14s mean pairwise cosine among top-%d neighbours: %.3f", group, k, neighbor_dist_cos)

    projections = project_all(local_vectors, centroid)

    fig, axes = plt.subplots(1, len(projections), figsize=(5.2 * len(projections), 5))
    for ax, (technique, coords) in zip(axes, projections.items()):
        member_2d, centroid_2d = coords[:-1], coords[-1]
        ax.scatter(member_2d[:, 0], member_2d[:, 1], s=10, c="#cfcfcf", linewidths=0, zorder=1)
        # local_order is itself ascending-distance-sorted, so its first k
        # positions in member_2d ARE the k nearest, by construction.
        ax.scatter(member_2d[:k, 0], member_2d[:k, 1], s=40, c="#0072B2", zorder=2)
        ax.scatter(*centroid_2d, s=140, c="#D55E00", marker="*", zorder=3)
        ax.set_title(technique, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"{group}: centroid + top-{k} across projection techniques "
                  f"(same {len(local_order)}-point local neighbourhood, raw bge-m3 cosine)", fontsize=12)
    fig.tight_layout()
    out_path = out_dir / f"compare_{group}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lit-mivi-run-tag", default="20260910")
    parser.add_argument("--interviews-archive-dir", type=Path,
                         default=bss.PROCESSED_DIR / "interviews_full" / "interviews" / "run_20260912")
    parser.add_argument("--groups", nargs="+", default=DEFAULT_GROUPS)
    parser.add_argument("--k", type=int, default=acc.DEFAULT_K)
    parser.add_argument("--local-context-n", type=int, default=acc.LOCAL_CONTEXT_N)
    parser.add_argument("--out-dir", type=Path, default=bss.PROCESSED_DIR / "analysis_raw" / "projection_comparison")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw_points = acc.load_raw_points(args.lit_mivi_run_tag, args.interviews_archive_dir)
    vectors = np.array([p["vector"] for p in raw_points], dtype=np.float64)
    points = [{k: v for k, v in p.items() if k != "vector"} for p in raw_points]
    logger.info("Loaded %d raw points", len(points))

    for group in args.groups:
        indices = acc.group_indices(points, group)
        stats = gac.centroid_and_dispersion_for_indices(vectors, indices)
        centroid = stats["centroid"]
        group_points = [points[i] for i in indices]
        group_vectors = vectors[indices]
        neighbors = gac.nearest_points(centroid, group_points, group_vectors, k=args.k)
        out_path = plot_comparison(args.out_dir, group, points, vectors, indices, centroid, neighbors, args.local_context_n)
        print(f"{group}: {out_path}")


if __name__ == "__main__":
    main()
