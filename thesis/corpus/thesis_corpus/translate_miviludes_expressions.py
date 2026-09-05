"""Translate MIVILUDES's extracted expressions to English, then re-embed.

MIVILUDES's expression points are currently embedded in their original
French, but both reference point-sets they get compared against
(concept_backbone, structural_concepts) are English-only -- a language
asymmetry with directly measured evidence it adds noise (the 17 official
criteria's own FR/EN translation-pair cosine similarities range from 0.50
to 0.95 for identical content; see build_shared_space.py's
check_miviludes_translation_fidelity). This script translates all 914
MIVILUDES expressions to English and re-embeds them with bge-m3, so
build_shared_space.py can use the English embedding as the point's primary
vector (French becomes a display-only label) -- mirroring the
miviludes_criteria FR/EN pattern already established, just inverted.

Two checkpointed stages, same rationale as extract_and_embed.py's
annotate/embed split: a slow/failed LLM translation call should never cost
re-translating already-done items, and a failed embed call should never
cost re-calling the LLM.
  - translate: one qwen3:4b chat call per expression
    (ollama_client.translate_text), writing each result immediately to
    expression_translations.jsonl -- each record already carries
    embedding_vector_fr (copied straight from the source archive), so the
    final output is self-contained for a translation-fidelity check
    without reopening the (large, gitignored) main archive.
  - embed: reads that checkpoint, embeds each English translation with
    bge-m3, and writes expression_translations_embedded.jsonl.

Never touches criterion_expressions.jsonl (the source archive) -- read
only, per this codebase's "never modify the source, only write new files"
convention.

Prerequisite: Ollama running locally with qwen3:4b and bge-m3 pulled, same
as Stage 2 (thesis_corpus/README.md).

Usage (from thesis/corpus/):
    python -m thesis_corpus.translate_miviludes_expressions               # full batch, both stages
    python -m thesis_corpus.translate_miviludes_expressions --limit 5      # smoke test
    python -m thesis_corpus.translate_miviludes_expressions --stage translate
    python -m thesis_corpus.translate_miviludes_expressions --stage embed
    python -m thesis_corpus.translate_miviludes_expressions --force        # reprocess everything
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from tqdm import tqdm

from thesis_corpus.ollama_client import (
    EmbeddingError,
    OllamaUnavailableError,
    TranslationError,
    check_available,
    embed_texts,
    translate_text,
)

CORPUS_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = CORPUS_DIR / "processed" / "miviludes"
SOURCE_PATH = OUTPUT_DIR / "criterion_expressions.jsonl"
TRANSLATIONS_PATH = OUTPUT_DIR / "expression_translations.jsonl"
EMBEDDED_PATH = OUTPUT_DIR / "expression_translations_embedded.jsonl"

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_LLM_MODEL = "qwen3:4b"
DEFAULT_EMBED_MODEL = "bge-m3"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("thesis_corpus.translate_miviludes_expressions")


def _key(item: dict) -> str:
    return f"{item['document_id']}:{item['chunk_index']}"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_done_keys(path: Path) -> set[str]:
    return {_key(item) for item in load_jsonl(path)}


def run_translate_stage(
    source_items: list[dict], host: str, model: str, force: bool, limit: int | None, translations_path: Path,
) -> None:
    done_keys = set() if force else load_done_keys(translations_path)
    todo = [item for item in source_items if _key(item) not in done_keys]
    if limit is not None:
        todo = todo[:limit]

    print(f"[translate] {len(todo)} expressions -> {translations_path}")
    mode = "w" if force else "a"
    translations_path.parent.mkdir(parents=True, exist_ok=True)

    errors = 0
    with open(translations_path, mode, encoding="utf-8") as out:
        for item in tqdm(todo, desc="translate", unit="expr"):
            try:
                text_en = translate_text(
                    host, model, item["embedding_text"],
                    target_language="English", source_language="French",
                )
            except TranslationError as e:
                logger.error("[%s] translation failed: %s", _key(item), e)
                errors += 1
                continue
            record = {
                "document_id": item["document_id"],
                "chunk_index": item["chunk_index"],
                "text_fr": item["embedding_text"],
                "text_en": text_en,
                "embedding_vector_fr": item["embedding_vector"],
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

    print(f"[translate] done. {errors} error(s).")


def run_embed_stage(
    host: str, model: str, force: bool, limit: int | None, translations_path: Path, embedded_path: Path,
) -> None:
    pending = load_jsonl(translations_path)
    done_keys = set() if force else load_done_keys(embedded_path)
    todo = [item for item in pending if _key(item) not in done_keys]
    if limit is not None:
        todo = todo[:limit]

    print(f"[embed] {len(todo)} translations -> {embedded_path}")
    if not todo:
        return

    embedded_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if force else "a"

    try:
        vectors = embed_texts(host, model, [item["text_en"] for item in todo])
    except EmbeddingError as e:
        logger.error(
            "[embed] batch embedding failed (translations are safe on disk in %s, retry on next run): %s",
            translations_path.name, e,
        )
        raise SystemExit(1)

    with open(embedded_path, mode, encoding="utf-8") as out:
        for item, vector in zip(todo, vectors):
            record = dict(item)
            record["embedding_vector_en"] = vector
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[embed] done. {len(todo)} expressions embedded.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["all", "translate", "embed"], default="all",
                         help="Run only the translation stage, only the embedding stage, or both (default).")
    parser.add_argument("--limit", type=int, default=None,
                         help="Process only the first N not-yet-done expressions (smoke test).")
    parser.add_argument("--force", action="store_true",
                         help="Reprocess everything for the running stage(s), overwriting their checkpoint files.")
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--source", type=Path, default=SOURCE_PATH,
                         help="MIVILUDES's criterion_expressions.jsonl (never modified, only read).")
    parser.add_argument("--translations-output", type=Path, default=TRANSLATIONS_PATH)
    parser.add_argument("--output", type=Path, default=EMBEDDED_PATH)
    args = parser.parse_args()

    try:
        check_available(args.ollama_host)
    except OllamaUnavailableError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1)

    if not args.source.exists():
        raise SystemExit(f"Missing source archive: {args.source}")

    if args.stage in ("all", "translate"):
        source_items = load_jsonl(args.source)
        run_translate_stage(source_items, args.ollama_host, args.llm_model, args.force, args.limit, args.translations_output)

    if args.stage in ("all", "embed"):
        run_embed_stage(args.ollama_host, args.embed_model, args.force, args.limit, args.translations_output, args.output)

    print(f"\nDone ({args.stage}).")
    print(f"Translations checkpoint: {args.translations_output}")
    print(f"Final output: {args.output}")


if __name__ == "__main__":
    main()
