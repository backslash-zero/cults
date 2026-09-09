"""Embed a v2 extraction run with bge-m3 into a v1-shaped archive.

Reads processed/v2/<corpus>/run_<tag>/expressions_v2.jsonl (the judge-
accepted final rows) and writes processed/v2/<corpus>/run_<tag>/
criterion_expressions.jsonl with the same field names v1's archive has --
so build_shared_space, reduce_embeddings, translate_miviludes_expressions
and geometric_analysis_common can read it with only a path change -- plus
every v2 field carried through:

  document_id, chunk_index, page_range, source_quote (= verbatim_expression),
  embedding_text (line breaks already folded by the screen), entity_anchors,
  claim_mode, epistemic_status, attribution, context_window,
  embedding_vector, entity_anchor_vectors, embedding_model, ... v2 fields ...

Checkpointed per document (embedded_documents.txt), resumable, refuses a
different embed model on resume, never modifies expressions_v2.jsonl.
Uses ollama_client.embed_texts exactly as v1's embed stage did.

Usage (from thesis/corpus/, Ollama host):
    python -m thesis_corpus.embed_v2 --corpus literature --run-tag 20260910
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from thesis_corpus.pilot_v2_literature import PROCESSED_ROOT, git_commit_hash, read_jsonl, sha256_file, write_json

DEFAULT_EMBED_MODEL = "bge-m3"
logger = logging.getLogger("thesis_corpus.embed_v2")


def append_jsonl(path: Path, rows) -> int:
    n = 0
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def to_archive_record(record: dict, vector: list[float], anchor_vectors: dict[str, list[float]], embed_model: str) -> dict:
    out = dict(record)
    out["source_quote"] = record["verbatim_expression"]
    out["embedding_model"] = embed_model
    out["embedding_vector"] = vector
    out["entity_anchor_vectors"] = {a: anchor_vectors[a] for a in record.get("entity_anchors", []) if a in anchor_vectors}
    return out


def run(args, embed=None, check=None) -> Path:
    from thesis_corpus.ollama_client import EmbeddingError, OllamaUnavailableError, check_available, embed_texts
    embed = embed or embed_texts
    check = check or check_available

    run_dir = args.out_root / args.corpus / f"run_{args.run_tag}"
    source = run_dir / "expressions_v2.jsonl"
    if not source.exists():
        raise SystemExit(f"{source} missing -- run extract_v2 first")
    if not (run_dir / "documents_done.txt").exists():
        raise SystemExit(f"{run_dir} has no documents_done.txt -- extraction not finished/checkpointed")
    out_path = run_dir / "criterion_expressions.jsonl"
    done_path = run_dir / "embedded_documents.txt"
    config_path = run_dir / "embed_config.json"
    source_sha = sha256_file(source)
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing["embed_model"] != args.embed_model:
            raise SystemExit(f"Refusing to resume: embed model {existing['embed_model']} != {args.embed_model}")
        if existing["expressions_v2_sha256"] != source_sha:
            raise SystemExit("Refusing to resume: expressions_v2.jsonl changed since embedding started (delete the embed outputs to restart)")
    else:
        write_json(config_path, {"script": "thesis_corpus.embed_v2", "corpus": args.corpus, "run_tag": args.run_tag,
                                 "embed_model": args.embed_model, "expressions_v2_sha256": source_sha,
                                 "git_commit": git_commit_hash(), "platform": sys.platform,
                                 "created_at": datetime.now(timezone.utc).isoformat(), "ollama_host": args.ollama_host})
    try:
        check(args.ollama_host)
    except OllamaUnavailableError as e:
        raise SystemExit(str(e))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(run_dir / "embed.log", encoding="utf-8"), logging.StreamHandler()])

    by_doc: dict[str, list[dict]] = defaultdict(list)
    for rec in read_jsonl(source):
        by_doc[rec["document_id"]].append(rec)
    done = set(done_path.read_text(encoding="utf-8").split()) if done_path.exists() else set()
    todo = [d for d in sorted(by_doc) if d not in done]
    if args.limit is not None:
        todo = todo[:args.limit]
    logger.info("[%s] %d documents with expressions, %d done, %d to embed", args.corpus, len(by_doc), len(done), len(todo))

    anchor_cache: dict[str, list[float]] = {}
    for i, document_id in enumerate(todo, start=1):
        records = by_doc[document_id]
        try:
            vectors = embed(args.ollama_host, args.embed_model, [r["embedding_text"] for r in records])
            anchors = sorted({a for r in records for a in r.get("entity_anchors", [])} - anchor_cache.keys())
            if anchors:
                anchor_cache.update(zip(anchors, embed(args.ollama_host, args.embed_model, anchors)))
        except EmbeddingError as e:
            logger.error("[%s] embedding failed (nothing written for this document; rerun to retry): %s", document_id, e)
            continue
        if len(vectors) != len(records):
            raise SystemExit(f"{document_id}: {len(vectors)} vectors for {len(records)} records")
        append_jsonl(out_path, (to_archive_record(r, v, anchor_cache, args.embed_model) for r, v in zip(records, vectors)))
        with open(done_path, "a", encoding="utf-8", newline="\n") as f:
            f.write(document_id + "\n")
        logger.info("[%d/%d] %s: %d expressions, %d new anchors embedded", i, len(todo), document_id, len(records), len(anchors))

    total_out = sum(1 for _ in open(out_path, encoding="utf-8")) if out_path.exists() else 0
    summary = {"corpus": args.corpus, "run_tag": args.run_tag, "embed_model": args.embed_model,
               "expressions_in": sum(len(v) for v in by_doc.values()), "documents_in": len(by_doc),
               "documents_embedded": len(set(done_path.read_text(encoding="utf-8").split())) if done_path.exists() else 0,
               "records_out": total_out, "complete": total_out == sum(len(v) for v in by_doc.values()),
               "updated_at": datetime.now(timezone.utc).isoformat()}
    write_json(run_dir / "embed_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", choices=("literature", "miviludes", "interviews"), required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-root", type=Path, default=PROCESSED_ROOT / "v2")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
