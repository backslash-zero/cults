"""Embed the Miviludes sectarian-drift criteria list (bge-m3 via Ollama).

Reads thesis/corpus/metadata/miviludes_criteria.json (the curated list of
Miviludes "dérive sectaire" criteria, French + English) and embeds each
criterion's French and English text separately with bge-m3, writing one JSON
object per criterion to miviludes_criteria_embedded.jsonl.

Unlike thesis_corpus.extract_and_embed, this list is already a set of
discrete, self-contained criterion statements -- there is no unstructured
prose to chunk or LLM-annotate here, so this script skips straight to
embedding rather than running Stage 1/2's document pipeline.

Prerequisite: Ollama running locally with bge-m3 pulled (same as Stage 2 --
see thesis_corpus/README.md):
    ollama pull bge-m3
    ollama serve

Usage (from thesis/corpus/):
    python -m thesis_corpus.embed_miviludes_criteria
"""
from __future__ import annotations

import argparse
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
INPUT_PATH = CORPUS_DIR / "metadata" / "miviludes_criteria.json"
OUTPUT_PATH = CORPUS_DIR / "metadata" / "miviludes_criteria_embedded.jsonl"

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_EMBED_MODEL = "bge-m3"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=INPUT_PATH,
                         help="miviludes_criteria.json to read (default: metadata/miviludes_criteria.json).")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH,
                         help="Output .jsonl path (default: metadata/miviludes_criteria_embedded.jsonl).")
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

    data = json.loads(args.input.read_text(encoding="utf-8"))
    records = sorted(data["miviludes_criteria"], key=lambda r: r["order"])

    fr_texts = [r["criterion_fr"] for r in records]
    en_texts = [r["criterion_en"] for r in records]

    print(f"Embedding {len(records)} criteria x 2 languages with {args.embed_model} at {args.ollama_host} ...")
    try:
        fr_vectors = embed_texts(args.ollama_host, args.embed_model, fr_texts)
        en_vectors = embed_texts(args.ollama_host, args.embed_model, en_texts)
    except EmbeddingError as e:
        print(f"ERROR: embedding failed: {e}", file=sys.stderr)
        raise SystemExit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for record, fr_vector, en_vector in zip(records, fr_vectors, en_vectors):
            out = dict(record)
            out["embedding_model"] = args.embed_model
            out["embedding_vector_fr"] = fr_vector
            out["embedding_vector_en"] = en_vector
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Done. {len(records)} criteria embedded -> {args.output}")


if __name__ == "__main__":
    main()
