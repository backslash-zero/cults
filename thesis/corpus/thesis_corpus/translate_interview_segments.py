"""Translate non-English interview segments to English, for display only.

extract_interviews_full.py's archive embeds every segment in its own
original language (bge-m3 is multilingual, so this is the right choice for
embedding fidelity -- the same reasoning prepare_interviews.py's own
docstring gives for feeding chunking the original-language transcript.txt
rather than translation_en.txt). But a planned use of this archive is
displaying interview embeddings synced to subtitles, using English as a
lingua franca across the corpus -- so the 6 French interviews' segments
need an English line to show, even though their embedding stays French.

This is purely a display-layer enrichment, run ON TOP of an existing,
already-embedded run: it reads that run's expressions_v2.jsonl (read only,
per this codebase's "never modify the source, only write new files"
convention -- same discipline as translate_miviludes_expressions.py, which
this script mirrors the structure of), translates each non-English
segment's verbatim_expression with ollama_client.translate_text (one
qwen3:4b chat call per segment, no JSON schema), and writes
segment_translations_en.jsonl alongside it. Nothing is re-embedded, nothing
in expressions_v2.jsonl/criterion_expressions.jsonl changes, and re-running
extract_interviews_full.py or embed_v2.py is not required. Checkpointed by
(document_id, segment_index), resumable.

Usage (from thesis/corpus/, Ollama host):
    python -m thesis_corpus.translate_interview_segments --run-tag 20260912
    python -m thesis_corpus.translate_interview_segments --run-tag 20260912 --limit 5   # smoke test
    python -m thesis_corpus.translate_interview_segments --run-tag 20260912 --force      # reprocess everything
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from tqdm import tqdm

from thesis_corpus.pilot_v2_literature import CORPUS_DIR, PROCESSED_ROOT

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3:4b"
DATABASE_PATH = CORPUS_DIR / "interviews" / "metadata" / "database.json"
DEFAULT_OUT_ROOT = PROCESSED_ROOT / "interviews_full"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("thesis_corpus.translate_interview_segments")


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


def load_document_languages(database_path: Path = DATABASE_PATH) -> dict[str, str]:
    entries = json.loads(database_path.read_text(encoding="utf-8"))["interviews"]
    return {e["id"]: e["language"] for e in entries}


def _key(item: dict) -> str:
    return f"{item['document_id']}:{item['segment_index']}"


def run(
    source_path: Path, output_path: Path, host: str, model: str, force: bool, limit: int | None,
    languages: dict[str, str] | None = None, translate=None,
) -> Path:
    from thesis_corpus.ollama_client import TranslationError, translate_text
    translate = translate or translate_text
    if not source_path.exists():
        raise SystemExit(f"Missing source archive: {source_path} -- run extract_interviews_full then embed_v2 first")
    languages = languages if languages is not None else load_document_languages()

    source_items = load_jsonl(source_path)
    to_translate = [item for item in source_items if languages.get(item["document_id"], "English") != "English"]
    done_keys = set() if force else {_key(item) for item in load_jsonl(output_path)}
    todo = [item for item in to_translate if _key(item) not in done_keys]
    if limit is not None:
        todo = todo[:limit]

    print(f"{len(to_translate)} non-English segment(s) total, {len(todo)} to translate -> {output_path}")
    if not todo:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if force else "a"

    errors = 0
    with open(output_path, mode, encoding="utf-8") as out:
        for item in tqdm(todo, desc="translate", unit="segment"):
            source_language = languages.get(item["document_id"])
            try:
                translation = translate(host, model, item["verbatim_expression"],
                                        target_language="English", source_language=source_language)
            except TranslationError as e:
                logger.error("[%s] translation failed: %s", _key(item), e)
                errors += 1
                continue
            record = {
                "document_id": item["document_id"],
                "segment_index": item["segment_index"],
                "chunk_index": item["chunk_index"],
                "source_language": source_language,
                "text_source": item["verbatim_expression"],
                "translation_en": translation,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

    print(f"Done. {errors} error(s).")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--source", type=Path, default=None,
                        help="Defaults to <out-root>/interviews/run_<run-tag>/expressions_v2.jsonl")
    parser.add_argument("--output", type=Path, default=None,
                        help="Defaults to <out-root>/interviews/run_<run-tag>/segment_translations_en.jsonl")
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None, help="translate only the first N not-yet-done segments (smoke test)")
    parser.add_argument("--force", action="store_true", help="reprocess everything, overwriting the output file")
    args = parser.parse_args()

    from thesis_corpus.ollama_client import OllamaUnavailableError, check_available
    try:
        check_available(args.ollama_host)
    except OllamaUnavailableError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1)

    run_dir = args.out_root / "interviews" / f"run_{args.run_tag}"
    source_path = args.source or (run_dir / "expressions_v2.jsonl")
    output_path = args.output or (run_dir / "segment_translations_en.jsonl")
    run(source_path, output_path, args.ollama_host, args.model, args.force, args.limit)


if __name__ == "__main__":
    main()
