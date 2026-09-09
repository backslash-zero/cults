"""Extraction v2 -- second-model judge over screen-retained expressions.

Replaces the manual review the pilot was designed around (the reviewer has
no time for it): every expression the deterministic screen retains is sent
once more to a *different, larger* local model together with its full source
chunk and labels, and that model answers exactly the questions the manual
review sheet asked. A fixed code rule (judge_accepts) turns the answers into
accept / reject; every verdict is stored and labelled model-judged. The
judge never rewrites a span: it can only accept, reject, flag a label, or
point at a better span (recorded, not applied).

Runs as its own stage over an existing arm directory (so extraction need not
be repeated) and can be chained from the run stage with --judge-model.

Outputs, under <arm_dir>/judge_<judge-model-tag>/:
  config.json                 judge model, JUDGE_VERSION, prompt sha256, decision rule
  judge_verdicts.jsonl        one row per screen-retained expression
  expressions_v2_judged.jsonl the accepted expressions (+ judge_* fields)
  judge_rejected.jsonl        rejected expressions, code judge_rejected / judge_failed
  summary.json                accounting + acceptance / issue / disagreement counts
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, StrictBool, ValidationError

from thesis_corpus import extraction_v2_schema as schema

JUDGE_VERSION = "1.0.0-pilot"
DEFAULT_JUDGE_MODEL = "qwen3:8b"
JUDGE_OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 8192}
ISSUE_VOCAB = ("none", "off_topic", "contextless_fragment", "truncated", "overlong", "multiple_claims",
               "ocr_or_encoding_error", "heading_or_name", "interviewer_question", "other")
JUDGE_BOOL_FIELDS = ("faithful", "self_contained", "cult_relevant", "textually_intelligible", "atomic")
JUDGE_LABEL_FIELDS = ("attribution_correct", "claim_mode_correct", "epistemic_status_correct")

logger = logging.getLogger("thesis_corpus.judge_v2")

JUDGE_SYSTEM_PROMPT = """\
You are checking one expression that was copied verbatim from a text, for a thesis on how
"cults", "sects", sectarian drift and new religious movements are described, defined,
disputed, exemplified and framed. You are NOT deciding whether anything is really a cult.

You receive the full source text, the extracted expression (an exact substring of it), and
the labels assigned to it. Judge the expression strictly, as a careful human reviewer would,
using only the source text. Be conservative: when in doubt, answer false or name the issue.

Answer every field:
- faithful: the expression preserves the source's actual claim, speaker and degree of
  certainty; nothing is added, and the chosen span does not change the meaning by cutting.
- self_contained: a reader understands what it claims, describes or associates without the
  previous sentence, the question that prompted it, or the surrounding paragraph.
- cult_relevant: it materially describes, defines, disputes, exemplifies, frames or
  associates cults, sects, sectarian drift, new religious movements, group authority or
  control, manipulation, spiritual abuse, a relevant named group or leader, or an everyday
  use of "cult". A sentence that merely contains a cult-related word but is about something
  else is NOT relevant. Chapter titles, author names, bibliographic lines, administrative
  or procedural statements, and sentences announcing what a chapter will do are NOT relevant.
- textually_intelligible: no garbled characters, split words, stray symbols, detached
  accents or broken syntax.
- atomic: one claim, definition, criterion, association or named example -- not several
  ideas, a long chain of claims, or a paragraph-long quotation.
- attribution_correct, claim_mode_correct, epistemic_status_correct: whether each assigned
  label is right (asserted = stated as true; qualified = stated with explicit limits or
  conditions; contested = presented as disputed; negated = denied or rejected;
  speculative = offered as possibility or uncertainty). If epistemic_status is wrong, put
  the right value in recommended_epistemic_status, otherwise "".
- better_span_exists_in_chunk: whether the source text contains a clearly more precise,
  complete and self-contained formulation of the same idea. If so, copy that span exactly,
  character for character, into better_span; otherwise "".
- extraction_issue: the single most important problem, or "none".
- reasoning_note: at most 25 words.

Return only a JSON object matching the schema."""

JUDGE_PROMPT_SHA256 = hashlib.sha256(JUDGE_SYSTEM_PROMPT.encode("utf-8")).hexdigest()


class JudgeVerdictV2(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    faithful: StrictBool
    self_contained: StrictBool
    cult_relevant: StrictBool
    textually_intelligible: StrictBool
    atomic: StrictBool
    attribution_correct: StrictBool
    claim_mode_correct: StrictBool
    epistemic_status_correct: StrictBool
    recommended_epistemic_status: Literal[("",) + schema.EPISTEMIC_STATUSES]  # type: ignore[valid-type]
    better_span_exists_in_chunk: StrictBool
    better_span: str = ""
    extraction_issue: Literal[ISSUE_VOCAB]  # type: ignore[valid-type]
    reasoning_note: str = ""


def judge_json_schema() -> dict:
    return JudgeVerdictV2.model_json_schema()


def judge_accepts(verdict: JudgeVerdictV2) -> bool:
    """The fixed decision rule: every quality boolean true and no named issue.
    Label disagreements and better-span pointers are recorded, never rejections."""
    return all(getattr(verdict, f) for f in JUDGE_BOOL_FIELDS) and verdict.extraction_issue == "none"


def judge_flags(verdict: JudgeVerdictV2) -> list[str]:
    flags = [f"judge_{f}_disagrees" for f in JUDGE_LABEL_FIELDS if not getattr(verdict, f)]
    if verdict.better_span_exists_in_chunk:
        flags.append("judge_better_span")
    return flags


def judge_user_message(record: dict) -> str:
    return (
        f"document_id: {record['document_id']}\nchunk_index: {record['chunk_index']}\n\n"
        f"Source text:\n{record['context_window']}\n\n"
        f"Extracted expression (exact substring):\n{record['verbatim_expression']}\n\n"
        f"Assigned labels: expression_kind={record['expression_kind']}; attribution={record['attribution']}; "
        f"claim_mode={record['claim_mode']}; epistemic_status={record['epistemic_status']}; "
        f"entity_anchors={record.get('entity_anchors', [])}\n\n"
        "Schema: {\"faithful\": bool, \"self_contained\": bool, \"cult_relevant\": bool, "
        "\"textually_intelligible\": bool, \"atomic\": bool, \"attribution_correct\": bool, "
        "\"claim_mode_correct\": bool, \"epistemic_status_correct\": bool, "
        "\"recommended_epistemic_status\": \"\" | " + " | ".join(f'"{s}"' for s in schema.EPISTEMIC_STATUSES) + ", "
        "\"better_span_exists_in_chunk\": bool, \"better_span\": string, "
        "\"extraction_issue\": " + " | ".join(f'"{s}"' for s in ISSUE_VOCAB) + ", "
        "\"reasoning_note\": string}"
    )


_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def parse_verdict(raw: str) -> JudgeVerdictV2:
    match = _CODE_FENCE_RE.match(raw.strip())
    content = match.group(1) if match else raw
    parsed = json.loads(content)
    if isinstance(parsed, dict) and "recommended_epistemic_status" in parsed and parsed["recommended_epistemic_status"] is None:
        parsed["recommended_epistemic_status"] = ""
    return JudgeVerdictV2.model_validate(parsed)


def model_tag(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "-", model)


def judge_arm(arm_dir: Path, host: str, judge_model: str, timeout: float = 240.0,
              chat=None, check=None) -> Path:
    """Run the judge over <arm_dir>/expressions_v2.jsonl. `chat`/`check` are
    injectable for tests; default to ollama_client.chat_structured / check_available."""
    from thesis_corpus.ollama_client import AnnotationError, chat_structured, check_available, OllamaUnavailableError
    chat = chat or chat_structured
    check = check or check_available
    from thesis_corpus.pilot_v2_literature import read_jsonl, write_jsonl, write_json, git_commit_hash, sha256_file

    expressions_path = arm_dir / "expressions_v2.jsonl"
    if not expressions_path.exists():
        raise SystemExit(f"{expressions_path} missing -- run the extraction arm first")
    judge_dir = arm_dir / f"judge_{model_tag(judge_model)}"
    if judge_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing judge directory {judge_dir}")
    try:
        check(host)
    except OllamaUnavailableError as e:
        raise SystemExit(str(e))

    records = read_jsonl(expressions_path)
    judge_dir.mkdir(parents=True)
    log_handler = logging.FileHandler(judge_dir / "judge.log", encoding="utf-8")
    logger.addHandler(log_handler)
    logger.setLevel(logging.INFO)
    started = datetime.now(timezone.utc)
    json_schema = judge_json_schema()

    verdict_rows, accepted, rejected, modes = [], [], [], Counter()
    for i, record in enumerate(records, start=1):
        key = f"{record['document_id']}:{record['chunk_index']}#{record['candidate_rank']}"
        user = judge_user_message(record)
        verdict, raw_content, mode, error = None, None, None, None
        for attempt in (1, 2):
            content = user if attempt == 1 else user + "\n\nReturn only valid JSON matching the schema, no commentary."
            try:
                result = chat(host, judge_model, JUDGE_SYSTEM_PROMPT, content, json_schema, JUDGE_OPTIONS,
                              think=False, timeout=timeout)
                raw_content, mode = result.content, result.structured_output_mode
                verdict = parse_verdict(raw_content)
                break
            except AnnotationError as e:
                error = f"transport: {e}"
            except (json.JSONDecodeError, ValueError, ValidationError) as e:
                error = f"invalid verdict: {e}"
        row = {"key": key, "document_id": record["document_id"], "chunk_index": record["chunk_index"],
               "candidate_rank": record["candidate_rank"], "verbatim_expression": record["verbatim_expression"],
               "selection_stratum": record.get("selection_stratum"), "judge_model": judge_model,
               "structured_output_mode": mode, "raw_content": raw_content, "judge_status": "ok" if verdict else "failed",
               "error": None if verdict else error}
        if verdict is None:
            row.update({"accepted": False, "verdict": None})
            verdict_rows.append(row)
            rejected.append({**record, "rejection_code": "judge_failed", "rule_detail": error, "judge_verdict": None})
            logger.error("[%d/%d] %s judge failure after retry: %s", i, len(records), key, error)
            continue
        modes[mode] += 1
        ok = judge_accepts(verdict)
        # A better_span must itself be verbatim to be worth recording.
        better_span_verbatim = bool(verdict.better_span) and verdict.better_span in record["context_window"]
        row.update({"accepted": ok, "verdict": verdict.model_dump(), "better_span_verbatim": better_span_verbatim})
        verdict_rows.append(row)
        judged = {**record, "judge_model": judge_model, "judge_version": JUDGE_VERSION,
                  "judge_accepted": ok, "judge_verdict": verdict.model_dump(),
                  "judge_flags": judge_flags(verdict), "judge_better_span_verbatim": better_span_verbatim}
        if ok:
            accepted.append(judged)
        else:
            failed = [f for f in JUDGE_BOOL_FIELDS if not getattr(verdict, f)]
            detail = f"issue={verdict.extraction_issue}; false={','.join(failed) or '-'}; note={verdict.reasoning_note}"
            rejected.append({**judged, "rejection_code": "judge_rejected", "rule_detail": detail})
        logger.info("[%d/%d] %s -> %s (issue=%s)", i, len(records), key, "accept" if ok else "reject", verdict.extraction_issue)

    write_jsonl(judge_dir / "judge_verdicts.jsonl", verdict_rows)
    write_jsonl(judge_dir / "expressions_v2_judged.jsonl", accepted)
    write_jsonl(judge_dir / "judge_rejected.jsonl", rejected)
    finished = datetime.now(timezone.utc)
    ok_rows = [r for r in verdict_rows if r["judge_status"] == "ok"]
    summary = {
        "judge_model": judge_model, "judge_version": JUDGE_VERSION,
        "input_expressions": len(records), "judged_ok": len(ok_rows),
        "judge_failed": len(records) - len(ok_rows), "accepted": len(accepted),
        "rejected_by_judge": sum(1 for r in rejected if r["rejection_code"] == "judge_rejected"),
        "reconciled": len(accepted) + len(rejected) == len(records),
        "acceptance_rate": round(len(accepted) / len(records), 4) if records else None,
        "extraction_issue_counts": dict(Counter(r["verdict"]["extraction_issue"] for r in ok_rows)),
        "false_quality_fields": dict(Counter(f for r in ok_rows for f in JUDGE_BOOL_FIELDS if not r["verdict"][f])),
        "label_disagreements": dict(Counter(f for r in ok_rows for f in JUDGE_LABEL_FIELDS if not r["verdict"][f])),
        "better_span_pointed": sum(1 for r in ok_rows if r["verdict"]["better_span_exists_in_chunk"]),
        "better_span_verbatim": sum(1 for r in ok_rows if r.get("better_span_verbatim")),
        "by_stratum": {
            s: {"input": sum(1 for r in verdict_rows if r["selection_stratum"] == s),
                "accepted": sum(1 for r in verdict_rows if r["selection_stratum"] == s and r["accepted"])}
            for s in sorted({r["selection_stratum"] for r in verdict_rows if r["selection_stratum"]})},
        "started_at": started.isoformat(), "finished_at": finished.isoformat(),
    }
    write_json(judge_dir / "summary.json", summary)
    write_json(judge_dir / "config.json", {
        "script": "thesis_corpus.judge_v2", "judge_version": JUDGE_VERSION, "judge_model": judge_model,
        "judge_prompt_sha256": JUDGE_PROMPT_SHA256, "ollama_host": host, "options": JUDGE_OPTIONS,
        "structured_output_modes_observed": dict(modes), "decision_rule":
            "accept iff faithful, self_contained, cult_relevant, textually_intelligible, atomic are all true and extraction_issue == 'none'; "
            "label disagreements and better_span are recorded as judge_flags, never applied",
        "input_expressions_sha256": sha256_file(expressions_path), "git_commit": git_commit_hash(),
        "started_at": started.isoformat(), "finished_at": finished.isoformat(),
    })
    logger.removeHandler(log_handler)
    print(json.dumps({k: summary[k] for k in ("input_expressions", "accepted", "rejected_by_judge", "judge_failed",
                                                "acceptance_rate", "extraction_issue_counts", "label_disagreements")},
                     ensure_ascii=False, indent=2))
    print(f"Judge dir: {judge_dir}")
    return judge_dir
