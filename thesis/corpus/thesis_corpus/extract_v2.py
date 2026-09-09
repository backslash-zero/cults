"""Extraction v2 -- full-corpus run (extract + screen + judge), resumable.

The production counterpart of pilot_v2_literature: every document of a
corpus, every chunk, one structured extraction call (qwen3:4b), the
deterministic screen (screen_v2), then the second-model judge (judge_v2)
over the screen-retained expressions of that document. Everything is
checkpointed per document, so a crash or an interrupted overnight run
resumes where it stopped and never repeats a model call for a finished
document.

Never touches processed/<corpus>/* (v1) or any shared-space / analysis
output. Reads processed/<corpus>/documents/*/pages.jsonl (Stage 1, byte-
mirrored between machines) and the Stage-1 integrity audit under
processed/audits/ (single source of document flags / corrupted regions).

Output: processed/v2/<corpus>/run_<tag>/
  config.json               models, prompt hashes, screening constants, audit provenance,
                            pre-screen config; a resume refuses if any of these differ
  documents_done.txt        checkpoint, one document_id per line
  documents_manifest.jsonl  per document: pages sha256, chunk count, counts, flags
  chunk_index.jsonl         per chunk: key, hashes, flags, skip code, counts (no text)
  skipped_chunks.jsonl      pre-screen skips
  model_responses.jsonl     raw extractor replies
  model_failures.jsonl      extractor failures after retry (chunk contributes nothing)
  screen_rejected.jsonl     candidates rejected by the screen, with codes
  judge_verdicts.jsonl      every judge verdict
  judge_rejected.jsonl      judge_rejected / judge_failed expressions
  chunk_terms.jsonl         domain terms (diagnostic inventory only)
  expressions_v2.jsonl      FINAL archive: judge-accepted expressions (+ judge_* fields)
  summary.json              accounting, rewritten after every document
  run.log

Usage (from thesis/corpus/, Ollama host):
    python -m thesis_corpus.extract_v2 --corpus literature --run-tag 20260910 --model qwen3:4b --judge-model qwen3:8b
    python -m thesis_corpus.extract_v2 --corpus literature --run-tag 20260910 ... --limit 3   # smoke test: 3 documents
    (rerun the same command to resume)
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from thesis_corpus import extraction_v2_schema as schema
from thesis_corpus import judge_v2
from thesis_corpus import screen_v2 as sv
from thesis_corpus.chunking import build_chunks
from thesis_corpus.pilot_v2_literature import (
    AUDITS_DIR, CORPUS_DIR, PROCESSED_ROOT, git_commit_hash, latest_audit_csv, read_jsonl, sha256_file,
    write_json, write_jsonl, model_tag,
)

SCRIPT_VERSION = "1.0.0"
SUPPORTED_CORPORA = ("literature", "miviludes", "interviews")
logger = logging.getLogger("thesis_corpus.extract_v2")
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def parse_envelope(raw: str) -> schema.ChunkEnvelopeV2:
    match = _CODE_FENCE_RE.match(raw.strip())
    return schema.ChunkEnvelopeV2.model_validate(json.loads(match.group(1) if match else raw))


def load_audit(audit_csv: Path, corpus: str) -> tuple[dict[str, list[str]], dict[str, set[str]], dict]:
    import csv
    flags: dict[str, list[str]] = {}
    with open(audit_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["corpus"] == corpus:
                flags[row["document_id"]] = [x for x in row["document_integrity_flag"].split(";") if x]
    regions_path = audit_csv.with_name(audit_csv.name.replace(".csv", ".regions.jsonl"))
    region_lines: dict[str, set[str]] = defaultdict(set)
    if regions_path.exists():
        for row in read_jsonl(regions_path):
            if row["corpus"] == corpus:
                region_lines[row["document_id"]].add(row["line_text"].strip())
    try:
        rel = str(audit_csv.resolve().relative_to(CORPUS_DIR))
    except ValueError:
        rel = None
    return flags, region_lines, {"audit_csv": str(audit_csv), "audit_csv_relative": rel,
                                 "audit_csv_sha256": sha256_file(audit_csv),
                                 "audit_regions_sha256": sha256_file(regions_path) if regions_path.exists() else None}


def append_jsonl(path: Path, rows) -> int:
    n = 0
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def extract_chunk(ctx: sv.ChunkContext, host: str, model: str, timeout: float, chat):
    """One extraction call with one retry. Returns (envelope|None, raw, mode, error)."""
    from thesis_corpus.ollama_client import AnnotationError
    user = schema.user_message(ctx.document_id, ctx.page_range, ctx.chunk_index, ctx.nfc_text)
    json_schema = schema.response_json_schema()
    error = None
    for attempt in (1, 2):
        content = user if attempt == 1 else user + "\n\nReturn only valid JSON matching the schema, no commentary."
        try:
            result = chat(host, model, schema.SYSTEM_PROMPT_V2, content, json_schema, schema.OLLAMA_OPTIONS,
                          think=schema.OLLAMA_THINK, timeout=timeout)
            return parse_envelope(result.content), result.content, result.structured_output_mode, None
        except AnnotationError as e:
            error = f"transport: {e}"
        except (json.JSONDecodeError, ValueError) as e:
            error = f"invalid JSON / envelope: {e}"
    return None, None, None, error


def run_config(args, audit_prov: dict, pre_screen: dict) -> dict:
    return {
        "script": "thesis_corpus.extract_v2", "script_version": SCRIPT_VERSION, "corpus": args.corpus,
        "model": args.model, "judge_model": args.judge_model, "judge_version": judge_v2.JUDGE_VERSION,
        "judge_prompt_sha256": judge_v2.JUDGE_PROMPT_SHA256, "judge_options": judge_v2.JUDGE_OPTIONS,
        "screening": schema.screening_constants(), "pre_screen_config": pre_screen,
        "stage1_audit": audit_prov, "text_transform_policy": "newline_to_space (screen_v2.fold_newlines); verbatim_expression untouched",
    }


IDENTITY_KEYS = ("corpus", "model", "judge_model", "judge_version", "judge_prompt_sha256", "screening", "pre_screen_config", "stage1_audit")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", choices=SUPPORTED_CORPORA, default="literature")
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--judge-model", default=judge_v2.DEFAULT_JUDGE_MODEL,
                        help="second-model judge; pass '' to skip judging (then expressions_v2.jsonl holds screen-retained rows)")
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--judge-timeout", type=float, default=240.0)
    parser.add_argument("--limit", type=int, default=None, help="process only the first N not-yet-done documents")
    parser.add_argument("--audit-csv", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=PROCESSED_ROOT / "v2")
    args = parser.parse_args()
    if args.judge_model == "":
        args.judge_model = None
    run(args)


def run(args, chat=None, check=None) -> Path:
    from thesis_corpus.ollama_client import chat_structured, check_available, OllamaUnavailableError
    chat = chat or chat_structured
    check = check or check_available

    documents_dir = PROCESSED_ROOT / args.corpus / "documents"
    if not documents_dir.exists():
        raise SystemExit(f"{documents_dir} missing -- Stage-1 tree not present on this machine (see mirror_stage1)")
    audit_csv = args.audit_csv or latest_audit_csv(AUDITS_DIR)
    doc_flags, region_lines, audit_prov = load_audit(Path(audit_csv), args.corpus)
    pre_screen = schema.PRE_SCREEN_BY_SOURCE[args.corpus]
    run_dir = args.out_root / args.corpus / f"run_{args.run_tag}"
    config_path = run_dir / "config.json"
    config = run_config(args, audit_prov, pre_screen)
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        diff = [k for k in IDENTITY_KEYS if existing.get(k) != config.get(k)]
        if diff:
            raise SystemExit(f"Refusing to resume {run_dir}: config differs on {diff}. Use a new --run-tag.")
        print(f"Resuming {run_dir}")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        config.update({"git_commit": git_commit_hash(), "platform": sys.platform,
                       "created_at": datetime.now(timezone.utc).isoformat(), "ollama_host": args.ollama_host})
        write_json(config_path, config)
    try:
        check(args.ollama_host)
    except OllamaUnavailableError as e:
        raise SystemExit(str(e))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(run_dir / "run.log", encoding="utf-8"), logging.StreamHandler()])
    judge_v2.logger.setLevel(logging.INFO)

    done_path = run_dir / "documents_done.txt"
    done = set(done_path.read_text(encoding="utf-8").split()) if done_path.exists() else set()
    all_docs = sorted(d for d in documents_dir.iterdir() if d.is_dir() and (d / "pages.jsonl").exists())
    todo = [d for d in all_docs if d.name not in done]
    if args.limit is not None:
        todo = todo[:args.limit]
    missing_audit = [d.name for d in todo if d.name not in doc_flags]
    if missing_audit:
        raise SystemExit(f"Documents absent from the Stage-1 audit {audit_csv}: {missing_audit[:5]} -- rerun the audit first")
    logger.info("[%s] %d documents total, %d done, %d to process", args.corpus, len(all_docs), len(done), len(todo))

    for doc_number, doc_dir in enumerate(todo, start=1):
        document_id = doc_dir.name
        pages = read_jsonl(doc_dir / "pages.jsonl")
        chunks = build_chunks(document_id, pages)
        logger.info("[%d/%d] %s: %d chunks", doc_number, len(todo), document_id, len(chunks))
        contexts = [sv.prepare_chunk(document_id, c.chunk_index, c.page_range, c.text, args.corpus,
                                     document_integrity_flags=doc_flags[document_id],
                                     corrupted_line_texts=frozenset(region_lines.get(document_id, ())))
                    for c in chunks]
        chunk_index_rows, skipped, responses, failures, screen_rejected, terms, screen_retained = [], [], [], [], [], [], []
        seen_texts: set[str] = set()
        for ctx in contexts:
            skip_code, skip_detail = sv.pre_screen_chunk(ctx, pre_screen)
            row = {"document_id": document_id, "chunk_index": ctx.chunk_index, "page_range": ctx.page_range,
                   "word_count": ctx.word_count, "raw_chunk_sha256": ctx.raw_sha256, "nfc_chunk_sha256": ctx.nfc_sha256,
                   "nfc_changed_codepoints": ctx.nfc_changed_codepoints, "document_integrity_flags": ctx.document_integrity_flags,
                   "corrupted_regions": ctx.corrupted_regions, "chunk_integrity_score": ctx.chunk_integrity_score,
                   "pre_screen_skip_code": skip_code, "model_status": None, "emitted": 0, "screen_retained": 0, "screen_rejected": 0}
            if skip_code:
                skipped.append({"document_id": document_id, "chunk_index": ctx.chunk_index, "skip_code": skip_code, "detail": skip_detail})
                chunk_index_rows.append(row)
                continue
            envelope, raw, mode, error = extract_chunk(ctx, args.ollama_host, args.model, args.timeout, chat)
            responses.append({"document_id": document_id, "chunk_index": ctx.chunk_index, "model": args.model,
                              "structured_output_mode": mode, "raw_content": raw, "error": error})
            if envelope is None:
                failures.append({"document_id": document_id, "chunk_index": ctx.chunk_index, "error": error})
                row["model_status"] = "failed"
                chunk_index_rows.append(row)
                logger.error("[%s:%d] extractor failure after retry: %s", document_id, ctx.chunk_index, error)
                continue
            retained, rejected = sv.screen_chunk(envelope.expressions, ctx, seen_texts)
            kept_terms, dropped_terms = sv.screen_domain_terms(envelope.domain_terms, ctx)
            for rec in retained:
                rec.update({"extraction_version": schema.EXTRACTION_VERSION, "prompt_sha256": schema.PROMPT_SHA256,
                            "model": args.model, "structured_output_mode": mode, "chunk_relevance": envelope.chunk_relevance,
                            "selection_stratum": None})
            screen_retained.extend(retained)
            screen_rejected.extend(rejected)
            terms.append({"document_id": document_id, "chunk_index": ctx.chunk_index, "domain_terms": kept_terms, "dropped_terms": dropped_terms})
            row.update({"model_status": "ok", "emitted": len(envelope.expressions), "screen_retained": len(retained), "screen_rejected": len(rejected)})
            chunk_index_rows.append(row)

        if args.judge_model and screen_retained:
            result = judge_v2.judge_records(screen_retained, args.ollama_host, args.judge_model, timeout=args.judge_timeout,
                                            chat=chat, progress_prefix=f"{document_id[:30]} judge ")
            verdicts, final, judge_rejected = result["verdict_rows"], result["accepted"], result["rejected"]
        else:
            verdicts, final, judge_rejected = [], list(screen_retained), []

        # write everything for this document, then the checkpoint line -- in that order
        append_jsonl(run_dir / "chunk_index.jsonl", chunk_index_rows)
        append_jsonl(run_dir / "skipped_chunks.jsonl", skipped)
        append_jsonl(run_dir / "model_responses.jsonl", responses)
        append_jsonl(run_dir / "model_failures.jsonl", failures)
        append_jsonl(run_dir / "screen_rejected.jsonl", screen_rejected)
        append_jsonl(run_dir / "chunk_terms.jsonl", terms)
        append_jsonl(run_dir / "judge_verdicts.jsonl", verdicts)
        append_jsonl(run_dir / "judge_rejected.jsonl", judge_rejected)
        append_jsonl(run_dir / "expressions_v2.jsonl", final)
        append_jsonl(run_dir / "documents_manifest.jsonl", [{
            "document_id": document_id, "pages_jsonl_sha256": sha256_file(doc_dir / "pages.jsonl"), "pages": len(pages),
            "chunks": len(chunks), "skipped": len(skipped), "model_failed": len(failures),
            "emitted": sum(r["emitted"] for r in chunk_index_rows), "screen_retained": len(screen_retained),
            "screen_rejected": len(screen_rejected), "judge_rejected": len(judge_rejected), "final": len(final),
            "document_integrity_flags": doc_flags[document_id], "finished_at": datetime.now(timezone.utc).isoformat()}])
        with open(done_path, "a", encoding="utf-8", newline="\n") as f:
            f.write(document_id + "\n")
        write_summary(run_dir, args)
        logger.info("[%d/%d] %s done: chunks=%d skipped=%d failed=%d emitted=%d screen_retained=%d judge_rejected=%d final=%d",
                    doc_number, len(todo), document_id, len(chunks), len(skipped), len(failures),
                    sum(r["emitted"] for r in chunk_index_rows), len(screen_retained), len(judge_rejected), len(final))
    summary = write_summary(run_dir, args)
    print(json.dumps({k: summary[k] for k in ("documents", "chunks", "candidates", "screen_rejection_codes", "judge")}, ensure_ascii=False, indent=2))
    print(f"Run dir: {run_dir}")
    return run_dir


def write_summary(run_dir: Path, args) -> dict:
    manifest = read_jsonl(run_dir / "documents_manifest.jsonl") if (run_dir / "documents_manifest.jsonl").exists() else []
    screen_rejected = read_jsonl(run_dir / "screen_rejected.jsonl") if (run_dir / "screen_rejected.jsonl").exists() else []
    verdicts = read_jsonl(run_dir / "judge_verdicts.jsonl") if (run_dir / "judge_verdicts.jsonl").exists() else []
    ok_verdicts = [v for v in verdicts if v["judge_status"] == "ok"]
    chunks_total = sum(m["chunks"] for m in manifest)
    called = chunks_total - sum(m["skipped"] for m in manifest)
    failed = sum(m["model_failed"] for m in manifest)
    emitted = sum(m["emitted"] for m in manifest)
    screen_ret = sum(m["screen_retained"] for m in manifest)
    judge_rej = sum(m["judge_rejected"] for m in manifest)
    final = sum(m["final"] for m in manifest)
    summary = {
        "corpus": args.corpus, "run_tag": args.run_tag, "model": args.model, "judge_model": args.judge_model,
        "documents": {"done": len(manifest)},
        "chunks": {"total": chunks_total, "skipped_pre_screen": chunks_total - called, "called": called,
                   "model_failed": failed, "model_failure_rate": round(failed / called, 4) if called else None},
        "candidates": {"emitted": emitted, "screen_retained": screen_ret, "screen_rejected": len(screen_rejected),
                       "reconciled": emitted == screen_ret + len(screen_rejected),
                       "judge_rejected_or_failed": judge_rej, "final": final,
                       "judge_reconciled": (screen_ret == final + judge_rej) if args.judge_model else None,
                       "final_per_called_chunk": round(final / called, 3) if called else None},
        "screen_rejection_codes": dict(Counter(r["rejection_code"] for r in screen_rejected).most_common()),
        "judge": {"judged_ok": len(ok_verdicts), "judge_failed": len(verdicts) - len(ok_verdicts),
                  "acceptance_rate": round(sum(1 for v in ok_verdicts if v["accepted"]) / len(ok_verdicts), 4) if ok_verdicts else None,
                  "extraction_issue_counts": dict(Counter(v["verdict"]["extraction_issue"] for v in ok_verdicts)),
                  "label_disagreements": dict(Counter(f for v in ok_verdicts for f in judge_v2.JUDGE_LABEL_FIELDS if not v["verdict"][f])),
                  "better_span_pointed": sum(1 for v in ok_verdicts if v["verdict"]["better_span_exists_in_chunk"])},
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(run_dir / "summary.json", summary)
    return summary


if __name__ == "__main__":
    main()
