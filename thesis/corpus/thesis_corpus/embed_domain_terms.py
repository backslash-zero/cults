"""Embed a v2 run's screened domain_terms that look like named entities, so
they can be pooled into the shared space's emergent_entities layer
alongside entity_anchors (see build_shared_space.py's load_emergent_entities).

Background: extract_v2's model call tags two different things per chunk --
entity_anchors (named groups/leaders/movements mentioned INSIDE a kept
expression only) and domain_terms (any cult-related concept or named group
appearing in the text, "even if no expression around it is worth
keeping" -- screened by screen_v2.screen_domain_terms, written to
chunk_terms.jsonl). Only entity_anchors currently feed the shared space.
Since v2's strict verbatim+judge screening keeps far fewer expressions than
v1's did, most named-entity mentions that happened not to sit inside a
surviving expression were being silently lost -- e.g. of the three v2
corpora, 0 entities end up mentioned by all three under entity_anchors
alone, even though a term like "Scientology" appears in every one of their
domain_terms lists (checked directly).

domain_terms is a much broader net than entity_anchors, though: it mixes
genuine named entities ("Scientology", "Heaven's Gate", "Jehovah's
Witnesses") with generic domain vocabulary ("cults", "brainwashing", "mind
control") that already has its own, better-curated home in the
structural_concepts/concept_backbone reference vocabularies -- pooling all
of it into emergent_entities would blur that distinction rather than add
useful coverage. A simple, checked-against-data filter separates the two:
domain_terms are stored verbatim, in the source text's own casing, and the
non-all-lowercase subset is overwhelmingly genuine named entities (checked
directly against a literature/miviludes/interviews sample: capitalized
terms are things like "ISKCON", "Branch Davidians", "UNADFI"; all-lowercase
terms are things like "secularization", "deprogramming", "mind control").
looks_like_named_entity() below is exactly that filter -- documented here,
not hidden downstream, since it is the one substantive judgment call this
script makes.

Only the filtered subset is embedded (bge-m3, same model/host as
everywhere else) -- no reason to pay for embedding calls on terms that will
be discarded before pooling. Checkpointed by raw term string (resumable,
same convention as embed_v2.py/translate_miviludes_expressions.py): a
term already present in the output file is never re-embedded unless
--force is passed.

Requires a live Ollama host -- run on the machine that has Ollama, then
transfer the output file back, same workflow as
translate_miviludes_expressions.py.

Usage (from thesis/corpus/, Ollama host):
    python -m thesis_corpus.embed_domain_terms --corpus literature --run-tag 20260910
    python -m thesis_corpus.embed_domain_terms --corpus miviludes --run-tag 20260910
    python -m thesis_corpus.embed_domain_terms --corpus interviews --run-tag 20260910
    python -m thesis_corpus.embed_domain_terms --corpus literature --run-tag 20260910 --limit 50   # smoke test
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from thesis_corpus.build_shared_space import looks_like_named_entity
from thesis_corpus.ollama_client import EmbeddingError, OllamaUnavailableError, check_available, embed_texts
from thesis_corpus.pilot_v2_literature import PROCESSED_ROOT, git_commit_hash, write_json

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_EMBED_MODEL = "bge-m3"
EMBED_CHUNK_SIZE = 200  # terms per write/checkpoint, independent of ollama_client's own internal sub-batching

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("thesis_corpus.embed_domain_terms")


def collect_candidate_terms(chunk_terms_path: Path) -> Counter[str]:
    """Raw (un-normalized) term -> mention count within this one corpus's
    run, restricted to looks_like_named_entity() terms. Normalization
    (case/whitespace folding across near-duplicate raw strings) happens
    downstream in build_shared_space.py, same as it already does for
    entity_anchors -- this script embeds by exact raw string, deliberately
    mirroring embed_v2.py's own entity_anchors convention."""
    counts: Counter[str] = Counter()
    with open(chunk_terms_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for term in row.get("domain_terms", []):
                if looks_like_named_entity(term):
                    counts[term] += 1
    return counts


def load_done_terms(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    done = set()
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(json.loads(line)["term"])
    return done


def run(
    chunk_terms_path: Path, output_path: Path, host: str, model: str,
    force: bool, limit: int | None,
) -> None:
    if not chunk_terms_path.exists():
        raise SystemExit(f"Missing: {chunk_terms_path}")

    all_counts = collect_candidate_terms(chunk_terms_path)
    logger.info(
        "%d chunk-level domain_term mentions read; %d unique terms look like named entities "
        "(not all-lowercase) and are candidates for embedding.",
        sum(all_counts.values()), len(all_counts),
    )

    done_terms = set() if force else load_done_terms(output_path)
    todo = sorted(t for t in all_counts if t not in done_terms)
    if limit is not None:
        todo = todo[:limit]

    print(f"{len(todo)} terms to embed -> {output_path} ({len(done_terms)} already done)")
    if not todo:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if force else "a"

    errors = 0
    embedded = 0
    with open(output_path, mode, encoding="utf-8") as out:
        for i in tqdm(range(0, len(todo), EMBED_CHUNK_SIZE), desc="embed", unit="chunk"):
            batch = todo[i:i + EMBED_CHUNK_SIZE]
            try:
                vectors = embed_texts(host, model, batch)
            except EmbeddingError as e:
                logger.error(
                    "Batch starting at term %r failed (already-embedded terms are safe on disk, "
                    "retry on next run to pick up the rest): %s", batch[0], e,
                )
                errors += 1
                continue
            for term, vector in zip(batch, vectors):
                out.write(json.dumps({"term": term, "vector": vector}, ensure_ascii=False) + "\n")
                embedded += 1
            out.flush()

    print(f"Done. {embedded} embedded this run, {errors} batch failure(s).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", choices=("literature", "miviludes", "interviews"), required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST)
    parser.add_argument("--limit", type=int, default=None, help="Embed only the first N not-yet-done terms (smoke test).")
    parser.add_argument("--force", action="store_true", help="Re-embed everything, overwriting the output file.")
    parser.add_argument("--out-root", type=Path, default=PROCESSED_ROOT / "v2")
    args = parser.parse_args()

    try:
        check_available(args.ollama_host)
    except OllamaUnavailableError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1)

    run_dir = args.out_root / args.corpus / f"run_{args.run_tag}"
    chunk_terms_path = run_dir / "chunk_terms.jsonl"
    output_path = run_dir / "domain_term_vectors.jsonl"

    run(chunk_terms_path, output_path, args.ollama_host, args.embed_model, args.force, args.limit)

    write_json(run_dir / "domain_term_vectors_config.json", {
        "script": "thesis_corpus.embed_domain_terms", "corpus": args.corpus, "run_tag": args.run_tag,
        "embed_model": args.embed_model, "filter": "not term.islower()",
        "git_commit": git_commit_hash(),
    })
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
