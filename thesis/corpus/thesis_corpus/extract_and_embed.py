"""Single-pass local extraction + embedding stage.

Reads the clean-text corpus (thesis_corpus.clean_text's output), chunks each
document, sends every chunk to a locally-running Ollama (qwen3:4b by
default) for structured annotation of cult/sect-criterion expressions,
embeds every accepted item's embedding_text and every unique entity anchor
with bge-m3 (also via Ollama), and writes one consolidated
criterion_expressions.jsonl plus an extraction_summary.json.

Runs entirely against a local Ollama at http://127.0.0.1:11434 by default --
no cloud APIs, no remote host.

Usage (from thesis/corpus/):
    python -m thesis_corpus.extract_and_embed             # full batch
    python -m thesis_corpus.extract_and_embed --limit 5    # first 5 not-yet-done documents
    python -m thesis_corpus.extract_and_embed --document-id <id>  # one document -> separate file
    python -m thesis_corpus.extract_and_embed --force       # reprocess everything, overwrite

Deliberately out of scope: chapter selection and entity extraction as
separate passes -- this is a single pass over the corpus.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from thesis_corpus.chunking import build_chunks
from thesis_corpus.ollama_client import (
    AnnotationError,
    EmbeddingError,
    OllamaUnavailableError,
    annotate_chunk,
    check_available,
    embed_texts,
)

CORPUS_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = CORPUS_DIR / "processed"
DOCUMENTS_DIR = OUTPUT_DIR / "documents"
OUTPUT_PATH = OUTPUT_DIR / "criterion_expressions.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "extraction_summary.json"
LOG_DIR = OUTPUT_DIR / "logs"
LOG_PATH = LOG_DIR / "extraction.log"

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_LLM_MODEL = "qwen3:4b"
DEFAULT_EMBED_MODEL = "bge-m3"

SUMMARY_PRINT_INTERVAL = 100

logger = logging.getLogger("thesis_corpus.extract_and_embed")


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")],
    )


def discover_documents() -> list[Path]:
    if not DOCUMENTS_DIR.exists():
        raise SystemExit(f"Documents directory not found: {DOCUMENTS_DIR}")
    dirs = []
    for d in sorted(DOCUMENTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        if (d / "pages.jsonl").exists():
            dirs.append(d)
        else:
            logger.warning("Skipping %s: no pages.jsonl", d.name)
    return dirs


def read_pages(doc_dir: Path) -> list[dict]:
    pages = []
    with open(doc_dir / "pages.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pages.append(json.loads(line))
    return pages


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["document_id"])
            except (json.JSONDecodeError, KeyError):
                logger.warning("Skipping malformed line while loading done ids: %r", line[:200])
    return done


def resolve_quote(chunk_text: str, quote: str) -> str | None:
    """Confirms `quote` is a substring of `chunk_text`.

    Tries an exact match first; falls back to whitespace-insensitive
    matching (the model can reproduce text with slightly different
    incidental spacing) and, if that succeeds, returns the actual substring
    found in chunk_text rather than the model's version -- this is what
    guarantees the "source_quote is always a substring of the chunk text"
    invariant downstream. Returns None if no match is found at all.
    """
    if quote in chunk_text:
        return quote
    tokens = quote.split()
    if not tokens:
        return None
    pattern = r"\s+".join(re.escape(t) for t in tokens)
    match = re.search(pattern, chunk_text, re.DOTALL)
    return match.group(0) if match else None


def _print_progress_summary(counters: dict, start_time: datetime) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tqdm.write(
        f"[{timestamp}] Processed {counters['chunks_processed']:,} chunks: "
        f"{counters['items_extracted']:,} items extracted, "
        f"{counters['embeddings_generated']:,} embeddings generated, "
        f"{counters['errors']:,} errors"
    )


def process_document(
    doc_dir: Path,
    host: str,
    llm_model: str,
    embed_model: str,
    anchor_cache: dict[str, list[float]],
    counters: dict,
    think: bool = False,
) -> list[dict]:
    document_id = doc_dir.name
    pages = read_pages(doc_dir)
    chunks = build_chunks(document_id, pages)
    tqdm.write(f"[{document_id}] {len(chunks)} chunks")

    raw_items: list[dict] = []
    chunk_bar = tqdm(chunks, desc=document_id[:40], leave=False, unit="chunk")
    for chunk in chunk_bar:
        counters["chunks_processed"] += 1

        try:
            annotation = annotate_chunk(
                host, llm_model, document_id, chunk.page_range, chunk.chunk_index, chunk.text,
                think=think,
            )
        except AnnotationError as e:
            logger.error("[%s] chunk %d annotation failed: %s", document_id, chunk.chunk_index, e)
            counters["errors"] += 1
            if counters["chunks_processed"] % SUMMARY_PRINT_INTERVAL == 0:
                _print_progress_summary(counters, counters["start_time"])
            continue

        counters["relevance_" + annotation.chunk_relevance] = (
            counters.get("relevance_" + annotation.chunk_relevance, 0) + 1
        )

        if annotation.chunk_relevance == "relevant" and annotation.items:
            for item in annotation.items:
                resolved_quote = resolve_quote(chunk.text, item.source_quote)
                if resolved_quote is None:
                    logger.warning(
                        "[%s] chunk %d: source_quote not found in chunk text, dropping item: %r",
                        document_id, chunk.chunk_index, item.source_quote[:200],
                    )
                    continue
                raw_items.append({
                    "document_id": document_id,
                    "chunk_index": chunk.chunk_index,
                    "page_range": chunk.page_range,
                    "source_quote": resolved_quote,
                    "embedding_text": item.embedding_text,
                    "entity_anchors": item.entity_anchors,
                    "claim_mode": item.claim_mode,
                    "epistemic_status": item.epistemic_status,
                    "attribution": item.attribution,
                    "context_window": chunk.text,
                })
            chunk_bar.set_postfix(items=len(raw_items))

        if counters["chunks_processed"] % SUMMARY_PRINT_INTERVAL == 0:
            _print_progress_summary(counters, counters["start_time"])

    if not raw_items:
        return []

    try:
        embeddings = embed_texts(host, embed_model, [it["embedding_text"] for it in raw_items])
    except EmbeddingError as e:
        logger.error(
            "[%s] embedding batch failed, 0 items written for this document "
            "(it will be retried in full on the next run): %s", document_id, e,
        )
        counters["errors"] += 1
        return []

    unique_anchors = sorted({a for it in raw_items for a in it["entity_anchors"]} - anchor_cache.keys())
    if unique_anchors:
        try:
            anchor_vectors = embed_texts(host, embed_model, unique_anchors)
            anchor_cache.update(zip(unique_anchors, anchor_vectors))
        except EmbeddingError as e:
            logger.error("[%s] entity anchor embedding failed: %s", document_id, e)
            counters["errors"] += 1
            # Items themselves already have real embeddings; anchor vectors
            # for anchors not already cached are simply omitted below.

    final_items = []
    for item, vector in zip(raw_items, embeddings):
        item["embedding_vector"] = vector
        item["entity_anchor_vectors"] = {
            a: anchor_cache[a] for a in item["entity_anchors"] if a in anchor_cache
        }
        final_items.append(item)

    counters["items_extracted"] += len(final_items)
    counters["embeddings_generated"] += len(embeddings) + len(unique_anchors)
    return final_items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None,
                         help="Process only the first N not-yet-done documents (for testing).")
    parser.add_argument("--document-id", default=None,
                         help="Process only this document, writing to a separate "
                              "criterion_expressions_<document_id>.jsonl instead of the main file.")
    parser.add_argument("--force", action="store_true",
                         help="Reprocess every document, overwriting criterion_expressions.jsonl.")
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--think", action="store_true",
                         help="Let the model use its chain-of-thought mode (much slower; off by default).")
    args = parser.parse_args()

    setup_logging()

    try:
        check_available(args.ollama_host)
    except OllamaUnavailableError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1)

    all_documents = discover_documents()

    if args.document_id:
        documents = [d for d in all_documents if d.name == args.document_id]
        if not documents:
            raise SystemExit(f"Document not found: {args.document_id}")
        target_path = OUTPUT_DIR / f"criterion_expressions_{args.document_id}.jsonl"
        mode = "w"
    elif args.force:
        documents = all_documents
        target_path = OUTPUT_PATH
        mode = "w"
    else:
        done_ids = load_done_ids(OUTPUT_PATH)
        documents = [d for d in all_documents if d.name not in done_ids]
        target_path = OUTPUT_PATH
        mode = "a"

    if args.limit is not None and not args.document_id:
        documents = documents[:args.limit]

    logger.info("Processing %d documents -> %s (mode=%s)", len(documents), target_path, mode)
    print(f"Processing {len(documents)} documents -> {target_path}")

    counters = {
        "chunks_processed": 0, "items_extracted": 0, "embeddings_generated": 0,
        "errors": 0, "start_time": datetime.now(timezone.utc),
    }
    skipped_documents: list[dict] = []

    target_path.parent.mkdir(parents=True, exist_ok=True)
    output_file = open(target_path, mode, encoding="utf-8")
    anchor_cache: dict[str, list[float]] = {}

    try:
        for doc_dir in tqdm(documents, desc="documents", unit="doc"):
            document_id = doc_dir.name
            try:
                items = process_document(
                    doc_dir, args.ollama_host, args.llm_model, args.embed_model, anchor_cache, counters,
                    think=args.think,
                )
            except Exception as e:
                logger.error("[%s] document processing failed: %s", document_id, e)
                counters["errors"] += 1
                skipped_documents.append({"document_id": document_id, "error": str(e)})
                continue

            for item in items:
                output_file.write(json.dumps(item, ensure_ascii=False) + "\n")
            output_file.flush()
    finally:
        output_file.close()

    summary = {
        "documents_processed": len(documents),
        "chunks_processed": counters["chunks_processed"],
        "chunk_relevance_counts": {
            k.removeprefix("relevance_"): v for k, v in counters.items() if k.startswith("relevance_")
        },
        "items_extracted": counters["items_extracted"],
        "embeddings_generated": counters["embeddings_generated"],
        "errors": counters["errors"],
        "skipped_documents": skipped_documents,
        "started_at": counters["start_time"].isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    final_line = (
        f"Done. {summary['documents_processed']} documents, {summary['chunks_processed']} chunks, "
        f"{summary['items_extracted']} items, {summary['embeddings_generated']} embeddings, "
        f"{summary['errors']} errors."
    )
    logger.info(final_line)
    print(f"\n{final_line}")
    print(f"Output: {target_path}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
