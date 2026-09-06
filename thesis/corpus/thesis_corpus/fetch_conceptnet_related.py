"""Stage 1 of the ConceptNet-expanded reference set: downloads ConceptNet's
official assertions dump once (cached outside the repo, same convention as
`wn`'s own WordNet cache at ~/.wn_data) and streams through it a single
time, keeping only English-to-English edges that touch one of the 1,500
`structural_concepts` seed words -- see extract_conceptnet_concepts.py
(stage 2) for what happens to the result.

Why a targeted single-pass filter rather than a general ConceptNet client
library (e.g. conceptnet-lite): that package needs the same ~500MB dump
anyway, then imports the *entire* multilingual graph into a queryable
SQLite database via three extra dependencies (lmdb, pySmartDL, peewee) --
useful for arbitrary future lookups, overkill for answering "what's
related to these 1,500 known words" once. This script uses only the
standard library (gzip/csv/json) against a fixed, versioned ConceptNet
release (5.7.0), matching this codebase's existing preference for static,
offline, reproducible resources over live services or heavyweight
general-purpose stores.

The dump itself (~500MB compressed, ~2GB uncompressed, ~34M edges across
every language ConceptNet covers) is cached at
~/.conceptnet_data/conceptnet-assertions-5.7.0.csv.gz and re-used on every
subsequent run -- never re-downloaded once present.

Usage (from thesis/corpus/):
    python -m thesis_corpus.fetch_conceptnet_related
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("thesis_corpus.fetch_conceptnet_related")

CORPUS_DIR = Path(__file__).resolve().parent.parent
SEED_CSV_PATH = CORPUS_DIR / "dictionaries" / "structural_concepts_candidates.csv"
OUTPUT_PATH = CORPUS_DIR / "dictionaries" / "conceptnet_related_raw.jsonl"

CACHE_DIR = Path.home() / ".conceptnet_data"
DUMP_URL = "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz"
DUMP_PATH = CACHE_DIR / "conceptnet-assertions-5.7.0.csv.gz"


def download_dump(url: str, path: Path) -> None:
    if path.exists():
        logger.info("Using cached dump at %s (%.0f MB)", path, path.stat().st_size / 1e6)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".part")
    logger.info("Downloading %s -> %s ...", url, path)
    with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        last_pct = -1
        with open(tmp_path, "wb") as f:
            for chunk in response.iter_bytes(chunk_size=1 << 20):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = int(downloaded * 100 / total)
                    if pct != last_pct and pct % 5 == 0:
                        logger.info("  %d%% (%.0f / %.0f MB)", pct, downloaded / 1e6, total / 1e6)
                        last_pct = pct
    tmp_path.rename(path)
    logger.info("Download complete: %.0f MB", path.stat().st_size / 1e6)


def load_seed_terms(path: Path) -> set[str]:
    with open(path, encoding="utf-8") as f:
        return {row["concept_en"].strip().lower() for row in csv.DictReader(f)}


def concept_term(uri: str) -> str | None:
    """"/c/en/charismatic_leader/n/wn/..." -> "charismatic leader" (spaces,
    not underscores, to match how seed terms are written). Returns None for
    a non-English concept URI."""
    if not uri.startswith("/c/en/"):
        return None
    term = uri[len("/c/en/"):].split("/", 1)[0]
    return term.replace("_", " ")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed-csv", type=Path, default=SEED_CSV_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--dump-path", type=Path, default=DUMP_PATH)
    parser.add_argument("--dump-url", type=str, default=DUMP_URL)
    args = parser.parse_args()

    seed_terms = load_seed_terms(args.seed_csv)
    logger.info("%d seed terms loaded from %s", len(seed_terms), args.seed_csv)

    download_dump(args.dump_url, args.dump_path)

    logger.info("Streaming %s, filtering for English edges touching a seed term ...", args.dump_path)
    n_lines = 0
    n_matched = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.dump_path, "rt", encoding="utf-8") as f_in, \
         open(args.output, "w", encoding="utf-8") as f_out:
        reader = csv.reader(f_in, delimiter="\t")
        for row in reader:
            n_lines += 1
            if n_lines % 5_000_000 == 0:
                logger.info("  ... %dM lines scanned, %d matches so far", n_lines // 1_000_000, n_matched)
            if len(row) < 5:
                continue
            _uri, relation, start_uri, end_uri, metadata_json = row[:5]
            start_term = concept_term(start_uri)
            end_term = concept_term(end_uri)
            if start_term is None or end_term is None:
                continue  # not an English-to-English edge

            if start_term in seed_terms:
                seed, other, direction = start_term, end_term, "start"
            elif end_term in seed_terms:
                seed, other, direction = end_term, start_term, "end"
            else:
                continue
            if other == seed:
                continue

            try:
                weight = json.loads(metadata_json).get("weight", 1.0)
            except (json.JSONDecodeError, AttributeError):
                weight = 1.0

            f_out.write(json.dumps({
                "seed": seed, "related": other,
                "relation": relation.removeprefix("/r/"),
                "direction": direction, "weight": weight,
            }, ensure_ascii=False) + "\n")
            n_matched += 1

    logger.info("Done. %d lines scanned, %d matching edges -> %s", n_lines, n_matched, args.output)
    print(f"\nDone. {n_matched} ConceptNet edges touching a seed term -> {args.output}")


if __name__ == "__main__":
    main()
