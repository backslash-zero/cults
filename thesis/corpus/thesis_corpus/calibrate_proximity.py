"""Calibration of bge-m3 proximity: a sensitivity profile, plus the masked-entity test.

Two stages, so only the first needs Ollama:

  embed    (Ollama host)  embeds every sentence of the calibration set, the bare
                          entity names, and the participant segments that mention
                          an entity with that entity's name masked.
  analyze  (any machine)  cosine per pair, distributions per contrast type and
                          domain, triplet checks, a profile figure, and the
                          masked-entity retrieval table.

The calibration set is read as an ordering of contrast types between two anchors
(near-identical at the top, unrelated at the bottom), never as a threshold.

Masked-entity test: for each entity named in at least MIN_SEGMENTS participant
segments from at least MIN_INTERVIEWS interviews, the bare entity name is the
query and all kept participant segments are the pool. We measure how well the
query retrieves the entity's own segments (precision at k = number of own
segments, and mean rank), first with the original segment vectors, then with
the entity's own segments re-embedded with the name (and known first names, MASK_EXTRA) replaced by MASK. A drop
shows how much retrieval rested on the name itself.

Outputs, in processed/calibration/run_<tag>/:
  vectors.jsonl, masked_segments.jsonl     gitignored (masked_segments holds interview text)
  manifest.json
  report/pair_similarities.csv             one row per calibration pair
  report/contrast_summary.csv              n, min, quartiles, max per contrast and domain
  report/triplets.csv                      edited variant vs same-topic different claim
  report/calibration_profile.png           the sensitivity profile
  report/masked_entity_test.csv            entity names and ranks only, no interview text

Usage (from thesis/corpus/):
  python -m thesis_corpus.calibrate_proximity preview --run-tag x          # Mac: check the masked texts
  python -m thesis_corpus.calibrate_proximity embed --run-tag 20261119     # Windows, Ollama running
  python -m thesis_corpus.calibrate_proximity analyze --run-tag 20261119   # either machine
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from thesis_corpus.prepare_interview_segments import (
    ALIASES, RUN_DIR, is_filler, load_aliases, load_segments, match_entities, segment_key,
)

CORPUS = Path(__file__).resolve().parents[1]
CALIBRATION_CSV = CORPUS.parents[1] / "thesis_v2" / "notes" / "calibration_set.csv"
OUT_ROOT = CORPUS / "processed" / "calibration"
EMBED_MODEL = "bge-m3"
DEFAULT_HOST = "http://127.0.0.1:11434"
MASK = "[name]"  # language-neutral, so French segments do not get an English phrase
# Words that identify an entity without matching its alias pattern (people's first names).
# Masked together with the alias. Check the masked texts with the `preview` stage before embedding.
MASK_EXTRA = {
    "Manson family": r"charles|linda",
    "Raëlism": r"claude\s+vorilhon|vorilhon",
    "Scientology": r"l\.?\s*ron\s+hubbard|hubbard|tom\s+cruise",
    "Ku Klux Klan": r"klan|kkk",
}
MIN_SEGMENTS = 3
MIN_INTERVIEWS = 2
TRIPLET_CONTRASTS = ("1_paraphrase", "2_opposite_stance", "3_negation", "4_translation", "8_singular_plural")


# ---------- inputs ----------

def load_calibration(path: Path) -> list[dict]:
    """Curated pairs: rows marked n/no in `keep` are dropped, blank counts as kept."""
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [r for r in rows if r.get("keep", "").strip().lower() not in ("n", "no")]


def mask_text(text: str, regex: re.Pattern, extra: str | None = None, mask: str = MASK) -> str:
    """Replace every whole word that contains an alias match (and any extra pattern) with the mask.

    Alias patterns are stems ("scientolog"), so the match is widened to the full word:
    "Scientologie" becomes "[name]", not "[name]ie". Adjacent masks are merged.
    """
    patterns = [regex] + ([re.compile(extra, re.IGNORECASE)] if extra else [])
    for pattern in patterns:
        text = re.sub(r"[\w'’-]*(?:" + pattern.pattern + r")[\w'’-]*", mask, text, flags=re.IGNORECASE)
    return re.sub(r"(?:" + re.escape(mask) + r"\s+)+" + re.escape(mask), mask, text)


def kept_participant_segments(run_dir: Path) -> list[dict]:
    """Same selection as prepare_interview_segments: participant, not filler."""
    return [s for s in load_segments(run_dir)
            if s["attribution"] == "participant" and not is_filler(s["verbatim_expression"])]


def eligible_entities(segments: list[dict], aliases: list[dict]) -> dict[str, list[dict]]:
    by_entity: dict[str, list[dict]] = defaultdict(list)
    for s in segments:
        for e in match_entities(s["verbatim_expression"], aliases):
            by_entity[e].append(s)
    return {e: segs for e, segs in by_entity.items()
            if len(segs) >= MIN_SEGMENTS and len({s["document_id"] for s in segs}) >= MIN_INTERVIEWS}


# ---------- embed stage ----------

def run_embed(args, embed=None, check=None) -> Path:
    from thesis_corpus.ollama_client import check_available, embed_texts
    from thesis_corpus.pilot_v2_literature import git_commit_hash, sha256_file, write_json
    embed = embed or embed_texts
    check = check or check_available

    out = OUT_ROOT / f"run_{args.run_tag}"
    out.mkdir(parents=True, exist_ok=True)
    if not args.run_dir.joinpath("criterion_expressions.jsonl").exists():
        raise SystemExit(f"Interview run not found: {args.run_dir}/criterion_expressions.jsonl. "
                         "Copy run_20260912 to this machine first, or pass --run-dir.")
    check(args.host)

    pairs = load_calibration(args.calibration)
    aliases = load_aliases(args.aliases)
    regex = {a["entity"]: a["regex"] for a in aliases}
    entities = eligible_entities(kept_participant_segments(args.run_dir), aliases)

    texts = sorted({t for r in pairs for t in (r["sentence_a"], r["sentence_b"])} | set(entities))
    vectors = embed(args.host, args.model, texts)
    with (out / "vectors.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for text, vec in zip(texts, vectors):
            f.write(json.dumps({"text": text, "vector": vec}, ensure_ascii=False) + "\n")

    masked_rows = []
    for entity, segs in entities.items():
        for s in segs:
            masked_rows.append({"entity": entity, "key": segment_key(s), "document_id": s["document_id"],
                                "masked_text": mask_text(s["verbatim_expression"], regex[entity], MASK_EXTRA.get(entity))})
    masked_vectors = embed(args.host, args.model, [r["masked_text"] for r in masked_rows])
    with (out / "masked_segments.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row, vec in zip(masked_rows, masked_vectors):
            f.write(json.dumps({**row, "vector": vec}, ensure_ascii=False) + "\n")

    write_json(out / "manifest.json", {
        "created_at": datetime.now(timezone.utc).isoformat(), "git_commit": git_commit_hash(),
        "model": args.model, "host": args.host, "mask": MASK,
        "calibration_csv_sha256": sha256_file(args.calibration), "aliases_sha256": sha256_file(args.aliases),
        "calibration_pairs": len(pairs), "unique_texts": len(texts),
        "entities": {e: len(s) for e, s in entities.items()}, "masked_segments": len(masked_rows),
    })
    print(f"embedded {len(texts)} texts and {len(masked_rows)} masked segments -> {out}")
    return out


def run_preview(args) -> Path:
    """Write the masked texts to a local review CSV (gitignored, quotes participants). No Ollama."""
    aliases = load_aliases(args.aliases)
    regex = {a["entity"]: a["regex"] for a in aliases}
    entities = eligible_entities(kept_participant_segments(args.run_dir), aliases)
    rows = [{"entity": e, "key": segment_key(s), "original": s["verbatim_expression"],
             "masked": mask_text(s["verbatim_expression"], regex[e], MASK_EXTRA.get(e))}
            for e, segs in sorted(entities.items()) for s in segs]
    path = OUT_ROOT / "masked_preview.csv"
    write_csv(path, rows)
    print(f"{len(rows)} masked segments for {len(entities)} entities -> {path}")
    return path


# ---------- analyze stage ----------

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def pair_similarities(pairs: list[dict], vec: dict[str, np.ndarray]) -> list[dict]:
    return [{**{k: r[k] for k in ("pair_id", "contrast", "domain", "register", "base_id")},
             "cosine": round(cosine(vec[r["sentence_a"]], vec[r["sentence_b"]]), 4)} for r in pairs]


def summarise(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        groups[(r["contrast"], r["domain"])].append(r["cosine"])
    out = []
    for (contrast, domain), vals in sorted(groups.items()):
        q = np.percentile(vals, [0, 25, 50, 75, 100])
        out.append({"contrast": contrast, "domain": domain, "n": len(vals),
                    **{k: round(float(v), 4) for k, v in zip(("min", "q1", "median", "q3", "max"), q)}})
    return out


def triplets(rows: list[dict]) -> list[dict]:
    """Is the edited variant closer to its base than the base's same-topic, different-claim partner?"""
    reference = {r["base_id"].split("|")[0]: r["cosine"] for r in rows if r["contrast"] == "7_same_topic_different_claim"}
    out = []
    for r in rows:
        if r["contrast"] in TRIPLET_CONTRASTS and r["base_id"] in reference:
            ref = reference[r["base_id"]]
            out.append({"base_id": r["base_id"], "domain": r["domain"], "register": r["register"],
                        "contrast": r["contrast"], "variant_cosine": r["cosine"], "same_topic_cosine": ref,
                        "variant_closer": r["cosine"] > ref, "margin": round(r["cosine"] - ref, 4)})
    return out


def retrieval(query: np.ndarray, pool: np.ndarray, own: np.ndarray) -> dict:
    """Precision at k (k = number of own segments) and mean rank (1 = closest) of the own segments."""
    sims = pool @ (query / np.linalg.norm(query)) / np.linalg.norm(pool, axis=1)
    order = np.argsort(-sims)
    ranks = np.empty(len(sims), dtype=int)
    ranks[order] = np.arange(1, len(sims) + 1)
    k = int(own.sum())
    return {"precision_at_k": round(float(own[order[:k]].sum()) / k, 3), "mean_rank": round(float(ranks[own].mean()), 1)}


def masked_entity_test(pool_keys: list[str], pool_vecs: np.ndarray, masked: list[dict],
                       name_vec: dict[str, np.ndarray]) -> list[dict]:
    index = {k: i for i, k in enumerate(pool_keys)}
    by_entity: dict[str, list[dict]] = defaultdict(list)
    for m in masked:
        by_entity[m["entity"]].append(m)
    out = []
    for entity, rows in sorted(by_entity.items()):
        own = np.zeros(len(pool_keys), dtype=bool)
        masked_pool = pool_vecs.copy()
        for m in rows:
            i = index[m["key"]]
            own[i] = True
            masked_pool[i] = m["vector"]
        before = retrieval(name_vec[entity], pool_vecs, own)
        after = retrieval(name_vec[entity], masked_pool, own)
        out.append({"entity": entity, "segments": int(own.sum()),
                    "interviews": len({m["document_id"] for m in rows}), "pool": len(pool_keys),
                    "chance_precision": round(own.sum() / len(pool_keys), 3),
                    "precision_unmasked": before["precision_at_k"], "precision_masked": after["precision_at_k"],
                    "mean_rank_unmasked": before["mean_rank"], "mean_rank_masked": after["mean_rank"]})
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        if not rows:
            f.write("")
            return
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def plot_profile(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    contrasts = sorted({r["contrast"] for r in rows}, key=lambda c: -np.median([x["cosine"] for x in rows if x["contrast"] == c]))
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(contrasts) + 1.5))
    for y, c in enumerate(contrasts):
        vals = [r["cosine"] for r in rows if r["contrast"] == c]
        ax.boxplot(vals, positions=[y], vert=False, widths=0.6, showfliers=False)
        for dom, colour in (("cult", "#27548a"), ("choir", "#b3322c")):
            pts = [r["cosine"] for r in rows if r["contrast"] == c and r["domain"] == dom]
            ax.scatter(pts, np.full(len(pts), y) + np.random.default_rng(y).uniform(-0.18, 0.18, len(pts)),
                       s=12, alpha=0.7, color=colour, label=dom if y == 0 else None)
    ax.set_yticks(range(len(contrasts)), [c.split("_", 1)[1].replace("_", " ") for c in contrasts])
    ax.invert_yaxis()
    ax.set_xlabel("cosine similarity (bge-m3)")
    ax.legend(loc="lower left", frameon=False)
    ax.set_title("Sensitivity profile: similarity by contrast type")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run_analyze(args) -> Path:
    out = OUT_ROOT / f"run_{args.run_tag}"
    report = out / "report"
    vec = {}
    with (out / "vectors.jsonl").open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            vec[d["text"]] = np.asarray(d["vector"], dtype=float)

    pairs = load_calibration(args.calibration)
    missing = [r["pair_id"] for r in pairs if r["sentence_a"] not in vec or r["sentence_b"] not in vec]
    if missing:
        raise SystemExit(f"{len(missing)} pairs have no vector (CSV edited after embedding?): {missing[:10]} ... re-run embed.")
    rows = pair_similarities(pairs, vec)
    write_csv(report / "pair_similarities.csv", rows)
    write_csv(report / "contrast_summary.csv", summarise(rows))
    write_csv(report / "triplets.csv", triplets(rows))
    plot_profile(rows, report / "calibration_profile.png")

    masked_path = out / "masked_segments.jsonl"
    if masked_path.exists() and args.run_dir.joinpath("criterion_expressions.jsonl").exists():
        segs = kept_participant_segments(args.run_dir)
        keys = [segment_key(s) for s in segs]
        pool = np.asarray([s["embedding_vector"] for s in segs], dtype=float)
        masked = [json.loads(line) for line in masked_path.open(encoding="utf-8")]
        for m in masked:
            m["vector"] = np.asarray(m["vector"], dtype=float)
        write_csv(report / "masked_entity_test.csv", masked_entity_test(keys, pool, masked, vec))
    else:
        print("masked-entity test skipped: masked_segments.jsonl or the interview run is missing here")

    for s in summarise(rows):
        print(f"{s['contrast']:32s} {s['domain']:6s} n={s['n']:3d} median={s['median']:.3f} [{s['min']:.3f}, {s['max']:.3f}]")
    print(f"report -> {report}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["preview", "embed", "analyze"])
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--calibration", type=Path, default=CALIBRATION_CSV)
    ap.add_argument("--run-dir", type=Path, default=RUN_DIR)
    ap.add_argument("--aliases", type=Path, default=ALIASES)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--model", default=EMBED_MODEL)
    args = ap.parse_args()
    {"preview": run_preview, "embed": run_embed, "analyze": run_analyze}[args.stage](args)


if __name__ == "__main__":
    main()
