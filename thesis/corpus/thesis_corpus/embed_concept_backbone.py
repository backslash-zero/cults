"""Embed the concept-backbone list (bge-m3 via Ollama).

Reads thesis/corpus/dictionaries/concept_backbone_omw_en.csv (the
WordNet-derived list of neutral anchor concepts) and embeds each concept
with bge-m3, writing one JSON object per concept to
dictionaries/concept_backbone_embedded.jsonl.

Each concept is embedded as "<concept_en>: <gloss_en>", not the bare word
alone -- WordNet glosses exist specifically to disambiguate polysemous
words (e.g. "law" the legal-system sense vs. a scientific "law"), so
including the gloss gives a more precise anchor point in the embedding
space than the word in isolation would.

Like embed_miviludes_criteria.py, this list is already a set of discrete,
self-contained records -- no chunking or LLM annotation needed here,
straight to embedding.

Prerequisite: Ollama running locally with bge-m3 pulled (same as Stage 2 --
see thesis_corpus/README.md):
    ollama pull bge-m3
    ollama serve

Usage (from thesis/corpus/):
    python -m thesis_corpus.embed_concept_backbone
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from thesis_corpus.ollama_client import (
    EmbeddingError,
    OllamaUnavailableError,
    check_available,
    embed_texts,
)

CORPUS_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = CORPUS_DIR / "dictionaries" / "concept_backbone_omw_en.csv"
OUTPUT_PATH = CORPUS_DIR / "dictionaries" / "concept_backbone_embedded.jsonl"

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_EMBED_MODEL = "bge-m3"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=INPUT_PATH,
                         help="concept_backbone_omw_en.csv to read (default: dictionaries/concept_backbone_omw_en.csv).")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH,
                         help="Output .jsonl path (default: dictionaries/concept_backbone_embedded.jsonl).")
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    args = parser.parse_args()

    try:
        check_available(args.ollama_host)
    except OllamaUnavailableError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1)

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    with open(args.input, encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    texts = [f"{r['concept_en']}: {r['gloss_en']}" for r in records]

    print(f"Embedding {len(records)} concepts with {args.embed_model} at {args.ollama_host} ...")
    try:
        vectors = embed_texts(args.ollama_host, args.embed_model, texts)
    except EmbeddingError as e:
        print(f"ERROR: embedding failed: {e}", file=sys.stderr)
        raise SystemExit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for record, text, vector in zip(records, texts, vectors):
            out = dict(record)
            out["embedding_text"] = text
            out["embedding_model"] = args.embed_model
            out["embedding_vector"] = vector
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Done. {len(records)} concepts embedded -> {args.output}")


if __name__ == "__main__":
    main()
