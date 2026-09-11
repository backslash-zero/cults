"""Exhaustive interview extraction -- full-corpus run (chunk + label[+
optional judge]), resumable. Interview-only counterpart of extract_v2.py,
but with a coverage GUARANTEE: every segment interview_chunking.py produces
gets exactly one archive record, whether or not the model returns a valid
label for it. There is no candidate-screening step here (see
interview_segment_labels.py's module docstring for why v1 of this driver's
verbatim-matching design couldn't guarantee that, and why it's gone).

Never touches processed/v2/interviews/ (the existing, selective v2 run) or
any shared-space / analysis output for literature/MIVILUDES. Reads
processed/interviews/documents/*/pages.jsonl -- the SAME Stage-1 tree
extract_v2.py --corpus interviews already reads, produced by
prepare_interviews.py (run that first for any newly-added interview) and
already covered by mirror_stage1.py's cross-machine transfer. Uses the same
Stage-1 integrity-audit CSV extract_v2.py uses, but -- unlike extract_v2.py
-- does not refuse to run over a document missing from that audit: the
audit exists to catch PDF/OCR corruption, essentially irrelevant to
already-reviewed, hand-typed interview transcripts, so a document absent
from it is simply treated as flag-free rather than blocking the run (see
run(), `missing_audit` handling below).

Output: processed/interviews_full/interviews/run_<tag>/
  config.json               model, prompt hash, screening constants, audit provenance;
                            a resume refuses if any of these differ
  documents_done.txt        checkpoint, one document_id per line
  documents_manifest.jsonl  per document: pages sha256, segment/label counts, flags
  chunk_index.jsonl         per unit: key, hashes, flags, counts (no text)
  model_responses.jsonl     raw extractor replies
  model_failures.jsonl      extractor failures after retry (unit contributes nothing)
  label_issues.jsonl        label-quality problems (malformed/missing/duplicate label) --
                            diagnostic only, never removes a segment from the archive
  chunk_terms.jsonl         domain terms (diagnostic inventory only; feeds emergent entities)
  judge_verdicts.jsonl      every judge verdict for model-labeled segments (only written
                            when --judge-model is given; unlabeled segments are never judged)
  expressions_v2.jsonl      FINAL archive: one record per segment (name matches v2's so
                            embed_v2.py/embed_domain_terms.py work unmodified)
  summary.json              accounting, rewritten after every document
  run.log

Usage (from thesis/corpus/, Ollama host):
    python -m thesis_corpus.extract_interviews_full --run-tag 20260912 --model qwen3:4b
    python -m thesis_corpus.extract_interviews_full --run-tag 20260912 --limit 2   # smoke test
    python -m thesis_corpus.extract_interviews_full --run-tag 20260912 --judge-model qwen3:8b   # optional, off by default
    (rerun the same command to resume)

Then:
    python -m thesis_corpus.embed_v2 --corpus interviews --run-tag 20260912 --out-root ../processed/interviews_full
    python -m thesis_corpus.embed_domain_terms --corpus interviews --run-tag 20260912 --out-root ../processed/interviews_full
    python -m thesis_corpus.export_interview_emergent_entities --run-tag 20260912
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from thesis_corpus import interview_extraction_schema as schema
from thesis_corpus import interview_segment_labels as isl
from thesis_corpus import judge_v2
from thesis_corpus.extract_v2 import ChunkProgress, append_jsonl, format_duration, load_audit
from thesis_corpus.interview_chunking import build_transcript_units
from thesis_corpus.pilot_v2_literature import (
    AUDITS_DIR, PROCESSED_ROOT, git_commit_hash, latest_audit_csv, read_jsonl, sha256_file, write_json,
)

SCRIPT_VERSION = "2.0.0"
CORPUS = "interviews"
logger = logging.getLogger("thesis_corpus.extract_interviews_full")
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def parse_envelope(raw: str) -> schema.TranscriptLabelsEnvelope:
    match = _CODE_FENCE_RE.match(raw.strip())
    return schema.TranscriptLabelsEnvelope.model_validate(json.loads(match.group(1) if match else raw))


def label_unit(ctx: isl.ChunkContext, segments: list[str], host: str, model: str, timeout: float, chat):
    """One labeling call with one retry. Returns (envelope|None, raw, mode, error)."""
    from thesis_corpus.ollama_client import AnnotationError
    user = schema.user_message(ctx.document_id, ctx.page_range, ctx.chunk_index, segments)
    json_schema = schema.response_json_schema()
    error = None
    for attempt in (1, 2):
        content = user if attempt == 1 else user + "\n\nReturn only valid JSON matching the schema, no commentary."
        try:
            result = chat(host, model, schema.SYSTEM_PROMPT_FULL, content, json_schema, schema.OLLAMA_OPTIONS_FULL,
                          think=schema.OLLAMA_THINK, timeout=timeout)
            return parse_envelope(result.content), result.content, result.structured_output_mode, None
        except AnnotationError as e:
            error = f"transport: {e}"
        except (json.JSONDecodeError, ValueError) as e:
            error = f"invalid JSON / envelope: {e}"
    return None, None, None, error


def apply_judge_labels(records: list[dict], host: str, judge_model: str, timeout: float, chat, progress_prefix: str = "") -> tuple[list[dict], list[dict]]:
    """Optional. Runs judge_v2.judge_records (unmodified) over the
    model-labeled records only (judging a "missing" placeholder label is
    meaningless) and merges its verdict fields onto those, regardless of the
    judge's own accept/reject decision -- this pipeline never drops a
    segment on a relevance/quality verdict, so judge disagreement becomes
    additional label metadata, not a rejection. Records that were never
    labeled by the model pass through unchanged (no judge_* fields).
    Returns (labeled_records, verdict_rows)."""
    labeled_records = [r for r in records if r["label_status"] == "model"]
    if not labeled_records:
        return records, []
    result = judge_v2.judge_records(labeled_records, host, judge_model, timeout=timeout, chat=chat, progress_prefix=progress_prefix)
    verdict_by_key = {(r["document_id"], r["chunk_index"], r["candidate_rank"]): r for r in result["verdict_rows"]}
    out = []
    for rec in records:
        row = verdict_by_key.get((rec["document_id"], rec["chunk_index"], rec["candidate_rank"]))
        rec = dict(rec)
        if row:
            rec["judge_model"] = judge_model
            rec["judge_status"] = row["judge_status"]
            rec["judge_verdict"] = row["verdict"]
            rec["judge_accepted"] = row["accepted"]
        out.append(rec)
    return out, result["verdict_rows"]


def run_config(args, audit_prov: dict) -> dict:
    return {
        "script": "thesis_corpus.extract_interviews_full", "script_version": SCRIPT_VERSION, "corpus": CORPUS,
        "model": args.model, "judge_model": args.judge_model,
        "judge_version": judge_v2.JUDGE_VERSION if args.judge_model else None,
        "judge_prompt_sha256": judge_v2.JUDGE_PROMPT_SHA256 if args.judge_model else None,
        "screening": schema.screening_constants(), "stage1_audit": audit_prov,
        "text_transform_policy": "newline_to_space (screen_v2.fold_newlines); verbatim_expression untouched",
    }


IDENTITY_KEYS = ("model", "judge_model", "judge_version", "judge_prompt_sha256", "screening", "stage1_audit")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--judge-model", default="",
                        help="optional second-model judge (e.g. qwen3:8b); off by default -- unlike extract_v2, "
                             "this pipeline never rejects on a judge verdict, so the judge only earns its cost "
                             "(a second LLM call per labeled segment) if you want the extra judge_* label fields")
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--judge-timeout", type=float, default=240.0)
    parser.add_argument("--limit", type=int, default=None, help="process only the first N not-yet-done documents")
    parser.add_argument("--audit-csv", type=Path, default=None)
    parser.add_argument("--max-words", type=int, default=None, help="override interview_chunking's whole-transcript word cap")
    parser.add_argument("--out-root", type=Path, default=PROCESSED_ROOT / "interviews_full")
    args = parser.parse_args()
    if args.judge_model == "":
        args.judge_model = None
    run(args)


def run(args, chat=None, check=None) -> Path:
    from thesis_corpus.ollama_client import OllamaUnavailableError, chat_structured, check_available
    chat = chat or chat_structured
    check = check or check_available

    documents_dir = PROCESSED_ROOT / CORPUS / "documents"
    if not documents_dir.exists():
        raise SystemExit(f"{documents_dir} missing -- run `python -m thesis_corpus.prepare_interviews` first")
    audit_csv = args.audit_csv or latest_audit_csv(AUDITS_DIR)
    doc_flags, region_lines, audit_prov = load_audit(Path(audit_csv), CORPUS)

    run_dir = args.out_root / CORPUS / f"run_{args.run_tag}"
    config_path = run_dir / "config.json"
    config = run_config(args, audit_prov)
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
        logger.info("%d document(s) absent from the Stage-1 audit %s (treated as flag-free -- interview "
                    "transcripts are hand-reviewed text, not OCR'd PDFs, so a missing audit row is not a "
                    "blocker here): %s", len(missing_audit), audit_csv, missing_audit)

    build_units = (lambda doc_id, pages: build_transcript_units(doc_id, pages, max_words=args.max_words)) \
        if args.max_words else build_transcript_units

    # Cheap pre-pass (pure functions, no model calls) for unit-count/ETA
    # figures from the first log line. No pre-screen for this pipeline --
    # every unit is callable.
    total_units = 0
    for doc_dir in todo:
        total_units += len(build_units(doc_dir.name, read_jsonl(doc_dir / "pages.jsonl")))
    logger.info("[%s] %d documents total, %d done, %d to process (%d units, all will call the model)",
                CORPUS, len(all_docs), len(done), len(todo), total_units)
    progress = ChunkProgress(total_units, len(todo))

    for doc_number, doc_dir in enumerate(todo, start=1):
        document_id = doc_dir.name
        pages = read_jsonl(doc_dir / "pages.jsonl")
        units = build_units(document_id, pages)
        logger.info("[%d/%d] %s: %d unit(s)", doc_number, len(todo), document_id, len(units))
        contexts = [(isl.prepare_chunk(document_id, u.chunk_index, u.page_range, u.text, "interviews_full",
                                       document_integrity_flags=doc_flags.get(document_id, []),
                                       corrupted_line_texts=frozenset(region_lines.get(document_id, ()))),
                    u.turn_roles)
                    for u in units]
        chunk_index_rows, responses, failures, label_issues, terms, all_records = [], [], [], [], [], []
        for ctx, turn_roles in contexts:
            spans = isl.segment_spans(ctx.nfc_text, turn_roles)
            segments = [ctx.nfc_text[s:e] for s, e, _role in spans]
            row = {"document_id": document_id, "chunk_index": ctx.chunk_index, "page_range": ctx.page_range,
                   "word_count": ctx.word_count, "raw_chunk_sha256": ctx.raw_sha256, "nfc_chunk_sha256": ctx.nfc_sha256,
                   "nfc_changed_codepoints": ctx.nfc_changed_codepoints, "document_integrity_flags": ctx.document_integrity_flags,
                   "corrupted_regions": ctx.corrupted_regions, "chunk_integrity_score": ctx.chunk_integrity_score,
                   "model_status": None, "segments": len(segments), "labeled": 0, "issues": 0}
            envelope, raw, mode, error = label_unit(ctx, segments, args.ollama_host, args.model, args.timeout, chat)
            progress.record_called()
            logger.info(progress.line(document_id, ctx.chunk_index, doc_number))
            responses.append({"document_id": document_id, "chunk_index": ctx.chunk_index, "model": args.model,
                              "structured_output_mode": mode, "raw_content": raw, "error": error})
            if envelope is None:
                # No labels for this unit -- the segments still all get archived (unlabeled).
                records, issues = isl.build_segment_records(ctx, turn_roles, [])
                failures.append({"document_id": document_id, "chunk_index": ctx.chunk_index, "error": error})
                row["model_status"] = "failed"
                logger.error("[%s:%d] extractor failure after retry (segments still archived, unlabeled): %s",
                            document_id, ctx.chunk_index, error)
            else:
                records, issues = isl.build_segment_records(ctx, turn_roles, envelope.segment_labels)
                kept_terms, dropped_terms = isl.screen_domain_terms(envelope.domain_terms, ctx)
                terms.append({"document_id": document_id, "chunk_index": ctx.chunk_index, "domain_terms": kept_terms, "dropped_terms": dropped_terms})
                row["model_status"] = "ok"
            for rec in records:
                rec.update({"extraction_version": schema.EXTRACTION_VERSION, "prompt_sha256": schema.PROMPT_SHA256,
                            "model": args.model, "structured_output_mode": mode})
            all_records.extend(records)
            label_issues.extend({"document_id": document_id, "chunk_index": ctx.chunk_index, **vars(issue)} for issue in issues)
            row.update({"labeled": sum(1 for r in records if r["label_status"] == "model"), "issues": len(issues)})
            chunk_index_rows.append(row)

        verdicts = []
        if args.judge_model and all_records:
            all_records, verdicts = apply_judge_labels(all_records, args.ollama_host, args.judge_model,
                                                        timeout=args.judge_timeout, chat=chat,
                                                        progress_prefix=f"{document_id[:30]} judge ")

        # write everything for this document, then the checkpoint line -- in that order
        append_jsonl(run_dir / "chunk_index.jsonl", chunk_index_rows)
        append_jsonl(run_dir / "model_responses.jsonl", responses)
        append_jsonl(run_dir / "model_failures.jsonl", failures)
        append_jsonl(run_dir / "label_issues.jsonl", label_issues)
        append_jsonl(run_dir / "chunk_terms.jsonl", terms)
        if verdicts:
            append_jsonl(run_dir / "judge_verdicts.jsonl", verdicts)
        append_jsonl(run_dir / "expressions_v2.jsonl", all_records)
        n_labeled = sum(1 for r in all_records if r["label_status"] == "model")
        append_jsonl(run_dir / "documents_manifest.jsonl", [{
            "document_id": document_id, "pages_jsonl_sha256": sha256_file(doc_dir / "pages.jsonl"), "pages": len(pages),
            "units": len(units), "model_failed": len(failures), "segments": len(all_records),
            "labeled": n_labeled, "unlabeled": len(all_records) - n_labeled, "label_issues": len(label_issues),
            "document_integrity_flags": doc_flags.get(document_id, []), "finished_at": datetime.now(timezone.utc).isoformat()}])
        with open(done_path, "a", encoding="utf-8", newline="\n") as f:
            f.write(document_id + "\n")
        write_summary(run_dir, args)
        logger.info("[%d/%d] %s done: units=%d failed=%d segments=%d labeled=%d issues=%d | %s",
                    doc_number, len(todo), document_id, len(units), len(failures),
                    len(all_records), n_labeled, len(label_issues), progress.summary())
    summary = write_summary(run_dir, args)
    print(json.dumps({k: summary[k] for k in ("documents", "chunks", "segments", "label_issue_codes")}, ensure_ascii=False, indent=2))
    print(progress.summary())
    print(f"Run dir: {run_dir}")
    return run_dir


def write_summary(run_dir: Path, args) -> dict:
    manifest = read_jsonl(run_dir / "documents_manifest.jsonl") if (run_dir / "documents_manifest.jsonl").exists() else []
    label_issues = read_jsonl(run_dir / "label_issues.jsonl") if (run_dir / "label_issues.jsonl").exists() else []
    units_total = sum(m["units"] for m in manifest)
    failed = sum(m["model_failed"] for m in manifest)
    segments = sum(m["segments"] for m in manifest)
    labeled = sum(m["labeled"] for m in manifest)
    summary = {
        "corpus": CORPUS, "run_tag": args.run_tag, "model": args.model, "judge_model": args.judge_model,
        "documents": {"done": len(manifest)},
        "chunks": {"total": units_total, "called": units_total, "model_failed": failed,
                   "model_failure_rate": round(failed / units_total, 4) if units_total else None},
        "segments": {"total": segments, "labeled": labeled, "unlabeled": segments - labeled,
                     "labeled_rate": round(labeled / segments, 4) if segments else None,
                     "per_document": round(segments / len(manifest), 3) if manifest else None},
        "label_issue_codes": dict(Counter(i["code"] for i in label_issues).most_common()),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(run_dir / "summary.json", summary)
    return summary


if __name__ == "__main__":
    main()
