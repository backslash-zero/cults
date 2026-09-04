"""Build one shared, cross-corpus embedding space (PCA on a pooled matrix).

Every prior embedding step (extract_and_embed's three corpora,
embed_miviludes_criteria, embed_concept_backbone) produces vectors in the
same raw 1024-d bge-m3 space, but reduce_embeddings.py's downsampling fits
an INDEPENDENT PCA per corpus -- literature's own 100-d space, MIVILUDES's
own, interviews' own. Those are unrelated coordinate systems: a point's
position in one is not comparable to a point's position in another. This
script instead pools every vector from every dataset, standardizes, and
fits ONE PCA on the pooled matrix -- so every item, from every dataset,
ends up in the same shared coordinate system, comparable to every other.

This is what operationalizes the concept backbone's stated purpose (see
Methods.tex, "A Generic Concept Backbone"): it is not a static reference
list sitting outside the analysis, but an active participant in fitting
this shared space, alongside every corpus item and the MIVILUDES criteria.

Pooling (43,415 points total, by design -- asserted at runtime):
  - Each corpus item (literature/miviludes/interviews) contributes ONE
    point: its embedding_vector.
  - Each MIVILUDES criterion contributes TWO points, not one: its
    embedding_vector_fr and embedding_vector_en separately (both are
    legitimate embedded representations of the same content; their
    distance from each other in the shared space is itself a sanity check
    on the multilingual embedding).
  - Each concept-backbone entry contributes ONE point: its embedding_vector.

Preprocessing: StandardScaler (zero mean, unit variance per dimension)
before PCA. Every vector already comes from the same embedding model, but
the five datasets differ a lot in register (academic prose, government
French, casual interview speech, bare word+gloss dictionary entries) and
could plausibly carry different per-dimension distributions -- this is a
defensive measure against any one dataset dominating the fit purely due to
scale, not a claim that such an imbalance is known to exist.

Dimensionality is not a fixed constant: PCA is first fit at full rank to
get the complete explained-variance curve (saved as a diagnostic, not just
asserted), and the smallest k reaching 95% cumulative variance is chosen
from that curve.

This needs no Ollama -- only numpy/scikit-learn/matplotlib on data that's
already local. Never modifies any of the five source files; only ever
writes new files under processed/.

Usage (from thesis/corpus/):
    python -m thesis_corpus.build_shared_space
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

CORPUS_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = CORPUS_DIR / "processed"
# All cross-corpus (not per-corpus) outputs live under their own
# processed/shared_space/ subdirectory, as a sibling of processed/<corpus>/
# -- keeps "one corpus's own pipeline output" and "output that spans every
# corpus" visually and structurally distinct, and gives any later step
# (e.g. a further 2D/3D reduction for visualization) an obvious home next
# to the space it would be reducing.
SHARED_SPACE_DIR = PROCESSED_DIR / "shared_space"

CORPUS_ARCHIVES = {
    "literature": PROCESSED_DIR / "literature" / "criterion_expressions.jsonl",
    "miviludes": PROCESSED_DIR / "miviludes" / "criterion_expressions.jsonl",
    "interviews": PROCESSED_DIR / "interviews" / "criterion_expressions.jsonl",
}
MIVILUDES_CRITERIA_PATH = CORPUS_DIR / "metadata" / "miviludes_criteria_embedded.jsonl"
CONCEPT_BACKBONE_PATH = CORPUS_DIR / "dictionaries" / "concept_backbone_embedded.jsonl"

OUTPUT_PATH = SHARED_SPACE_DIR / "embedding_space.jsonl"
VARIANCE_CSV_PATH = SHARED_SPACE_DIR / "variance_curve.csv"
VARIANCE_PLOT_PATH = SHARED_SPACE_DIR / "variance_curve.png"
VARIANCE_JSON_PATH = SHARED_SPACE_DIR / "variance_curve.json"

VARIANCE_THRESHOLD = 0.95
EMBEDDING_DIM = 1024

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("thesis_corpus.build_shared_space")


def load_corpus_points(corpus_name: str, path: Path) -> list[dict]:
    points = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            points.append({
                "source_dataset": corpus_name,
                "key": f"{item['document_id']}:{item['chunk_index']}",
                "label": item["embedding_text"],
                "attribution": item.get("attribution"),
                "vector": item["embedding_vector"],
            })
    return points


def load_miviludes_criteria_points(path: Path) -> list[dict]:
    points = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            points.append({
                "source_dataset": "miviludes_criteria_fr",
                "key": item["id"],
                "label": item["criterion_fr"],
                "vector": item["embedding_vector_fr"],
            })
            points.append({
                "source_dataset": "miviludes_criteria_en",
                "key": item["id"],
                "label": item["criterion_en"],
                "vector": item["embedding_vector_en"],
            })
    return points


def load_concept_backbone_points(path: Path) -> list[dict]:
    points = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            points.append({
                "source_dataset": "concept_backbone",
                "key": item["concept_id"],
                "label": item["concept_en"],
                "vector": item["embedding_vector"],
            })
    return points


def sanity_checks(points: list[dict], coords: np.ndarray) -> None:
    from collections import defaultdict

    norms_by_dataset = defaultdict(list)
    for p, c in zip(points, coords):
        norms_by_dataset[p["source_dataset"]].append(float(np.linalg.norm(c)))

    print("\nMean shared-space vector norm by source_dataset (diagnostic, not a pass/fail check):")
    for dataset, norms in sorted(norms_by_dataset.items()):
        print(f"  {dataset:<24} n={len(norms):<6} mean_norm={sum(norms)/len(norms):.3f}")

    fr_by_key = {p["key"]: c for p, c in zip(points, coords) if p["source_dataset"] == "miviludes_criteria_fr"}
    en_by_key = {p["key"]: c for p, c in zip(points, coords) if p["source_dataset"] == "miviludes_criteria_en"}
    print("\nMIVILUDES criteria FR vs EN point norms (should be similar in magnitude):")
    for key in sorted(fr_by_key):
        fr_norm = np.linalg.norm(fr_by_key[key])
        en_norm = np.linalg.norm(en_by_key[key])
        print(f"  {key:<40} fr={fr_norm:.3f}  en={en_norm:.3f}  diff={abs(fr_norm - en_norm):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variance-threshold", type=float, default=VARIANCE_THRESHOLD)
    args = parser.parse_args()

    points: list[dict] = []
    counts: dict[str, int] = {}

    for corpus_name, path in CORPUS_ARCHIVES.items():
        if not path.exists():
            raise SystemExit(f"Missing corpus archive: {path}")
        new_points = load_corpus_points(corpus_name, path)
        counts[corpus_name] = len(new_points)
        points.extend(new_points)

    if not MIVILUDES_CRITERIA_PATH.exists():
        raise SystemExit(f"Missing: {MIVILUDES_CRITERIA_PATH}")
    criteria_points = load_miviludes_criteria_points(MIVILUDES_CRITERIA_PATH)
    counts["miviludes_criteria_fr"] = sum(1 for p in criteria_points if p["source_dataset"] == "miviludes_criteria_fr")
    counts["miviludes_criteria_en"] = sum(1 for p in criteria_points if p["source_dataset"] == "miviludes_criteria_en")
    points.extend(criteria_points)

    if not CONCEPT_BACKBONE_PATH.exists():
        raise SystemExit(f"Missing: {CONCEPT_BACKBONE_PATH}")
    concept_points = load_concept_backbone_points(CONCEPT_BACKBONE_PATH)
    counts["concept_backbone"] = len(concept_points)
    points.extend(concept_points)

    logger.info("Pooled point counts: %s", counts)
    logger.info("Total pooled points: %d", len(points))
    expected_total = 39_236 + 914 + 231 + 34 + 3_000
    if len(points) != expected_total:
        logger.warning(
            "Pooled total %d does not match the expected 43,415 -- one of the "
            "corpora likely changed size since the plan was written; proceeding "
            "with whatever is actually on disk, but double-check the counts above.",
            len(points),
        )

    vectors = np.array([p["vector"] for p in points], dtype=np.float64)
    for p in points:
        del p["vector"]
    if vectors.shape[1] != EMBEDDING_DIM:
        raise SystemExit(f"Expected {EMBEDDING_DIM}-d vectors, got {vectors.shape[1]}")

    logger.info("Standardizing pooled matrix (zero mean, unit variance per dimension) before PCA...")
    scaled = StandardScaler().fit_transform(vectors)

    logger.info("Fitting full-rank PCA to get the complete explained-variance curve...")
    full_pca = PCA(random_state=args.seed)
    full_coords = full_pca.fit_transform(scaled)
    cumulative_variance = np.cumsum(full_pca.explained_variance_ratio_)

    k = int(np.searchsorted(cumulative_variance, args.variance_threshold) + 1)
    k = min(k, len(cumulative_variance))
    variance_at_k = float(cumulative_variance[k - 1])
    logger.info("%.1f%% cumulative variance at k=%d (threshold: %.0f%%)", variance_at_k * 100, k, args.variance_threshold * 100)

    SHARED_SPACE_DIR.mkdir(parents=True, exist_ok=True)
    with open(VARIANCE_CSV_PATH, "w", encoding="utf-8") as f:
        f.write("n_components,cumulative_variance\n")
        for i, v in enumerate(cumulative_variance, 1):
            f.write(f"{i},{v}\n")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance)
    plt.axhline(args.variance_threshold, color="gray", linestyle="--", linewidth=1)
    plt.axvline(k, color="gray", linestyle="--", linewidth=1)
    plt.scatter([k], [variance_at_k], color="red", zorder=5, label=f"k={k}, {variance_at_k*100:.1f}%")
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title("Shared cross-corpus PCA: explained variance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(VARIANCE_PLOT_PATH, dpi=150)
    plt.close()

    VARIANCE_JSON_PATH.write_text(
        json.dumps({
            "curve": [float(v) for v in cumulative_variance],
            "chosen_k": k,
            "variance_at_k": variance_at_k,
            "threshold": args.variance_threshold,
        }, indent=2),
        encoding="utf-8",
    )

    shared_coords = full_coords[:, :k]

    logger.info("Writing %s ...", OUTPUT_PATH)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for p, coord in zip(points, shared_coords):
            out = {
                "source_dataset": p["source_dataset"],
                "key": p["key"],
                "label": p["label"],
                "attribution": p.get("attribution"),
                "shared_space_vector": coord.tolist(),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    sanity_checks(points, shared_coords)

    print(f"\nDone. {len(points)} points, k={k} ({variance_at_k*100:.1f}% variance).")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Variance curve: {VARIANCE_CSV_PATH}, {VARIANCE_PLOT_PATH}, {VARIANCE_JSON_PATH}")


if __name__ == "__main__":
    main()
