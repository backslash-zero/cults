"""Local extraction + embedding stage (qwen3:4b + bge-m3 via Ollama).

Reads the clean-text corpus (thesis_corpus.clean_text's output), chunks each
document, sends every chunk to a locally-running Ollama (qwen3:4b by
default) for structured annotation of cult/sect-criterion expressions, then
embeds every accepted item's embedding_text and every unique entity anchor
with bge-m3 (also via Ollama), writing one consolidated
criterion_expressions.jsonl plus an extract_and_embed_summary.json.

Split into two stages with independent, disk-backed checkpoints:
  - annotate: chunk + LLM-annotate, writing each accepted item immediately
    to processed/pending_annotations.jsonl (no embedding calls at all).
  - embed: read pending_annotations.jsonl, batch-embed, write the final
    criterion_expressions.jsonl.
This means an embedding failure only costs a retry of the (fast) embedding
step, never a re-run of the (slow, expensive) LLM annotation for that
document -- annotation results are durable on disk before embedding is ever
attempted. --stage all (the default) runs both, one full pass then the
other; this is still a single pipeline with no separate curation stage in
between, just an internal checkpoint to make failures cheap to recover from.

Runs entirely against a local Ollama at http://127.0.0.1:11434 by default --
no cloud APIs, no remote host.

Usage (from thesis/corpus/):
    python -m thesis_corpus.extract_and_embed                    # full batch, both stages
    python -m thesis_corpus.extract_and_embed --limit 5           # first 5 not-yet-done documents
    python -m thesis_corpus.extract_and_embed --stage annotate    # annotation only
    python -m thesis_corpus.extract_and_embed --stage embed       # embedding only (from pending_annotations.jsonl)
    python -m thesis_corpus.extract_and_embed --document-id <id>  # one document, both stages -> separate file
    python -m thesis_corpus.extract_and_embed --force             # reprocess everything, overwrite

Deliberately out of scope: chapter selection and entity extraction as
separate curation passes -- this is a single pipeline over the corpus.
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
PENDING_PATH = OUTPUT_DIR / "pending_annotations.jsonl"
ANNOTATED_LOG_PATH = OUTPUT_DIR / "annotated_documents.txt"
SUMMARY_PATH = OUTPUT_DIR / "extract_and_embed_summary.json"
LOG_DIR = OUTPUT_DIR / "logs"
LOG_PATH = LOG_DIR / "extract_and_embed.log"

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
    """document_ids already present as full records in a criterion_expressions*.jsonl."""
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
                logger.warning("Skipping malformed line in %s: %r", path.name, line[:200])
    return done


def load_annotated_ids() -> set[str]:
    if not ANNOTATED_LOG_PATH.exists():
        return set()
    return {line.strip() for line in ANNOTATED_LOG_PATH.read_text(encoding="utf-8").splitlines() if line.strip()}


def mark_annotated(document_id: str) -> None:
    with open(ANNOTATED_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(document_id + "\n")


def load_pending_by_document() -> dict[str, list[dict]]:
    """Raw (un-embedded) annotated items from pending_annotations.jsonl, grouped by document_id."""
    by_doc: dict[str, list[dict]] = {}
    if not PENDING_PATH.exists():
        return by_doc
    with open(PENDING_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed pending line: %r", line[:200])
                continue
            by_doc.setdefault(item["document_id"], []).append(item)
    return by_doc


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


def _print_progress_summary(counters: dict) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tqdm.write(
        f"[{timestamp}] Processed {counters['chunks_processed']:,} chunks: "
        f"{counters['items_annotated']:,} items annotated, "
        f"{counters['embeddings_generated']:,} embeddings generated, "
        f"{counters['errors']:,} errors"
    )


def annotate_document(
    doc_dir: Path, host: str, llm_model: str, think: bool, pending_out, counters: dict,
) -> list[dict]:
    """Chunks and annotates one document, writing each accepted item to
    pending_out immediately. Returns the raw items (without embeddings), for
    the --document-id standalone path."""
    document_id = doc_dir.name
    pages = read_pages(doc_dir)
    chunks = build_chunks(document_id, pages)
    tqdm.write(f"[annotate] [{document_id}] {len(chunks)} chunks")

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
                _print_progress_summary(counters)
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
                record = {
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
                }
                pending_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                pending_out.flush()
                raw_items.append(record)
                counters["items_annotated"] += 1
            chunk_bar.set_postfix(items=len(raw_items))

        if counters["chunks_processed"] % SUMMARY_PRINT_INTERVAL == 0:
            _print_progress_summary(counters)

    return raw_items


def embed_document_items(
    document_id: str, raw_items: list[dict], host: str, embed_model: str,
    anchor_cache: dict[str, list[float]], counters: dict,
) -> list[dict]:
    """Batch-embeds one document's already-annotated items. Returns the
    final records (with vectors) -- empty list if embedding failed."""
    if not raw_items:
        return []

    try:
        embeddings = embed_texts(host, embed_model, [it["embedding_text"] for it in raw_items])
    except EmbeddingError as e:
        logger.error(
            "[embed] [%s] embedding batch failed (annotation is safe in %s, will retry on next run): %s",
            document_id, PENDING_PATH.name, e,
        )
        counters["errors"] += 1
        return []

    unique_anchors = sorted({a for it in raw_items for a in it["entity_anchors"]} - anchor_cache.keys())
    if unique_anchors:
        try:
            anchor_vectors = embed_texts(host, embed_model, unique_anchors)
            anchor_cache.update(zip(unique_anchors, anchor_vectors))
        except EmbeddingError as e:
            logger.error("[embed] [%s] entity anchor embedding failed: %s", document_id, e)
            counters["errors"] += 1
            # Items still get their own embedding_vector below; anchors not
            # already cached are simply omitted from entity_anchor_vectors.

    final_items = []
    for item, vector in zip(raw_items, embeddings):
        item = dict(item)
        item["embedding_vector"] = vector
        item["entity_anchor_vectors"] = {
            a: anchor_cache[a] for a in item["entity_anchors"] if a in anchor_cache
        }
        final_items.append(item)

    counters["items_embedded"] += len(final_items)
    counters["embeddings_generated"] += len(embeddings) + len(unique_anchors)
    return final_items


def run_annotate_stage(
    documents: list[Path], host: str, llm_model: str, think: bool, force: bool, limit: int | None,
    counters: dict,
) -> None:
    # A document already fully embedded (present in the main output from a
    # run predating this checkpoint file) never needs re-annotating either,
    # regardless of what annotated_documents.txt says.
    annotated_ids = set() if force else (load_annotated_ids() | load_done_ids(OUTPUT_PATH))
    todo = [d for d in documents if d.name not in annotated_ids]
    if limit is not None:
        todo = todo[:limit]

    logger.info("[annotate] %d documents to process", len(todo))
    print(f"[annotate] {len(todo)} documents -> {PENDING_PATH}")

    mode = "w" if force else "a"
    PENDING_PATH.parent.mkdir(parents=True, exist_ok=True)
    if force and ANNOTATED_LOG_PATH.exists():
        ANNOTATED_LOG_PATH.unlink()

    with open(PENDING_PATH, mode, encoding="utf-8") as pending_out:
        for doc_dir in tqdm(todo, desc="annotate", unit="doc"):
            document_id = doc_dir.name
            try:
                annotate_document(doc_dir, host, llm_model, think, pending_out, counters)
            except Exception as e:
                logger.error("[annotate] [%s] document failed: %s", document_id, e)
                counters["errors"] += 1
                continue
            mark_annotated(document_id)


def run_embed_stage(host: str, embed_model: str, force: bool, limit: int | None, counters: dict) -> None:
    pending_by_doc = load_pending_by_document()
    done_ids = set() if force else load_done_ids(OUTPUT_PATH)
    todo_ids = sorted(d for d in pending_by_doc if d not in done_ids)
    if limit is not None:
        todo_ids = todo_ids[:limit]

    logger.info("[embed] %d documents to process", len(todo_ids))
    print(f"[embed] {len(todo_ids)} documents -> {OUTPUT_PATH}")

    mode = "w" if force else "a"
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    anchor_cache: dict[str, list[float]] = {}

    with open(OUTPUT_PATH, mode, encoding="utf-8") as output_file:
        for document_id in tqdm(todo_ids, desc="embed", unit="doc"):
            raw_items = pending_by_doc[document_id]
            tqdm.write(f"[embed] [{document_id}] {len(raw_items)} items")
            final_items = embed_document_items(document_id, raw_items, host, embed_model, anchor_cache, counters)
            for item in final_items:
                output_file.write(json.dumps(item, ensure_ascii=False) + "\n")
            output_file.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["all", "annotate", "embed"], default="all",
                         help="Run only the annotation stage, only the embedding stage, or both (default).")
    parser.add_argument("--limit", type=int, default=None,
                         help="Process only the first N not-yet-done documents for the running stage(s) (for testing).")
    parser.add_argument("--document-id", default=None,
                         help="Process only this document end-to-end (annotate + embed), writing to a separate "
                              "criterion_expressions_<document_id>.jsonl instead of the main files.")
    parser.add_argument("--force", action="store_true",
                         help="Reprocess everything for the running stage(s), overwriting their checkpoint files.")
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

    counters = {
        "chunks_processed": 0, "items_annotated": 0, "items_embedded": 0,
        "embeddings_generated": 0, "errors": 0, "start_time": datetime.now(timezone.utc),
    }

    if args.document_id:
        documents = [d for d in discover_documents() if d.name == args.document_id]
        if not documents:
            raise SystemExit(f"Document not found: {args.document_id}")
        target_path = OUTPUT_DIR / f"criterion_expressions_{args.document_id}.jsonl"
        print(f"Processing document {args.document_id} -> {target_path}")

        # A throwaway scratch file for annotate_document's immediate per-item
        # writes -- target_path itself only ever receives the final, embedded
        # records, written once at the end.
        scratch_path = OUTPUT_DIR / f".pending_{args.document_id}.jsonl"
        with open(scratch_path, "w", encoding="utf-8") as scratch_out:
            raw_items = annotate_document(documents[0], args.ollama_host, args.llm_model, args.think, scratch_out, counters)
        scratch_path.unlink()

        anchor_cache: dict[str, list[float]] = {}
        final_items = embed_document_items(args.document_id, raw_items, args.ollama_host, args.embed_model, anchor_cache, counters)
        with open(target_path, "w", encoding="utf-8") as f:
            for item in final_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"\nDone. {len(final_items)} items written to {target_path}")
        print("This file is separate from the main criterion_expressions.jsonl -- "
              "merge, replace, or discard it by hand after inspecting it.")
        return

    documents = discover_documents()

    if args.stage in ("all", "annotate"):
        run_annotate_stage(documents, args.ollama_host, args.llm_model, args.think, args.force, args.limit, counters)

    if args.stage in ("all", "embed"):
        run_embed_stage(args.ollama_host, args.embed_model, args.force, args.limit, counters)

    summary = {
        "stage": args.stage,
        "chunks_processed": counters["chunks_processed"],
        "chunk_relevance_counts": {
            k.removeprefix("relevance_"): v for k, v in counters.items() if k.startswith("relevance_")
        },
        "items_annotated": counters["items_annotated"],
        "items_embedded": counters["items_embedded"],
        "embeddings_generated": counters["embeddings_generated"],
        "errors": counters["errors"],
        "started_at": counters["start_time"].isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    final_line = (
        f"Done ({args.stage}). {summary['chunks_processed']} chunks, "
        f"{summary['items_annotated']} items annotated, {summary['items_embedded']} items embedded, "
        f"{summary['embeddings_generated']} embeddings, {summary['errors']} errors."
    )
    logger.info(final_line)
    print(f"\n{final_line}")
    print(f"Pending annotations: {PENDING_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
