"""Acceptance report for an extraction-v2 pilot, from the judge's verdicts.

With no manual review available, the approved acceptance criteria are read
from the second-model judge's answers (labelled model-judged throughout).
Four sections, as approved:
  1. precision  -- judge acceptance and issue rates on the random stratum
                   (forced chunks reported separately)
  2. yield      -- final retained per chunk versus v1 items on the same chunks
  3. missed     -- better-span pointers and judge disagreements, listed
  4. regression -- for every forced chunk, whether the known v1 failure text
                   reappeared among the final retained expressions

Writes <pilot_dir>/review/acceptance_report.{json,md}. Hard-zero conditions
are re-checked independently (screen_v2.recheck_retained_record) on the
final retained set. No prose beyond the numbers.

Usage (from thesis/corpus/):
    python -m thesis_corpus.pilot_v2_evaluate --date-tag 20260909 --arm arm_qwen3-4b --judge judge_qwen3-8b
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from thesis_corpus import screen_v2 as sv
from thesis_corpus.pilot_v2_literature import (
    DEFAULT_PILOT_ROOT, FORCED_CHUNKS, read_jsonl, rebuild_context, write_json,
)

# The approved thresholds (plan section 8), applied to the random stratum.
THRESHOLDS = {
    "faithful_min": 0.98, "issue_none_min": 0.85, "off_topic_max": 0.05,
    "overlong_plus_multiple_max": 0.05, "labels_correct_min": 0.90,
    "model_failure_rate_blocker": 0.03,
}
V1_BASELINE_ISSUE_NONE = 13 / 34  # literature rows judged 'none' in the 2026-09-08 manual audit

# Known v1 failure text per forced chunk (the string the reviewer flagged).
V1_FAILURE_TEXT = {
    ("lalich-2004-bounded-choicetrue-believers-and-charismatic-cults", 213): "di Y cult",
    ("2008-the-oxford-handbook-of-new-religious-movements-1", 194): "would be called a 'cult apologist'",
    ("dawson-2009-cults-and-new-religious-movements-a-reader", 133): "when I really mean something that can be",
    ("dawson-2009-cults-and-new-religious-movements-a-reader", 135): "self 'actualization'",
    ("davis-and-hankins-2002-new-religious-movements-and-religious-liberty-in-america", 105): "Well, perhaps to some extent.",
    ("melton-2014-encyclopedic-handbook-of-cults-in-america", 10): "The Course of Growth.",
    ("tomkins-2010-the-clapham-sect-how-wilberforce-s-circle-transformed-britain", 131): "she did not like the Clapham sect",
    ("clarke-2004-encyclopedia-of-new-religious-movements", 585): "regarded as unduly authoritarian and liable to abuse",
    ("lalich-2004-bounded-choicetrue-believers-and-charismatic-cults", 239): "I was assigned leadership of the Party's publishing house",
    ("card-2019-archaeology-and-new-religious-movements", 1): "Boliǀia",
}


def rate(n: int, d: int) -> float | None:
    return round(n / d, 4) if d else None


def evaluate(pilot_dir: Path, arm_name: str, judge_name: str) -> dict:
    arm_dir = pilot_dir / arm_name
    judge_dir = arm_dir / judge_name
    chunk_rows = read_jsonl(pilot_dir / "pilot_chunks.jsonl")
    chunks = {(r["document_id"], r["chunk_index"]): r for r in chunk_rows}
    arm_summary = json.loads((arm_dir / "summary.json").read_text(encoding="utf-8"))
    judge_summary = json.loads((judge_dir / "summary.json").read_text(encoding="utf-8"))
    verdicts = read_jsonl(judge_dir / "judge_verdicts.jsonl")
    final = read_jsonl(judge_dir / "expressions_v2_judged.jsonl")
    screen_rejected = read_jsonl(arm_dir / "rejected_candidates.jsonl")
    judge_rejected = read_jsonl(judge_dir / "judge_rejected.jsonl")

    # independent hard-zero re-check on the FINAL retained set
    contexts = {k: rebuild_context(r) for k, r in chunks.items()}
    hard_zero = []
    for rec in final:
        for p in sv.recheck_retained_record(rec, contexts[(rec["document_id"], rec["chunk_index"])]):
            hard_zero.append(f"{rec['document_id']}:{rec['chunk_index']}#{rec['candidate_rank']}: {p}")
    if not judge_summary["reconciled"] or not arm_summary["candidates"]["reconciled"]:
        hard_zero.append("accounting not reconciled")

    def stratum_stats(stratum: str) -> dict:
        rows = [v for v in verdicts if v["selection_stratum"] == stratum and v["judge_status"] == "ok"]
        n = len(rows)
        vd = [v["verdict"] for v in rows]
        issues = Counter(v["extraction_issue"] for v in vd)
        stats = {
            "screen_retained": sum(1 for v in verdicts if v["selection_stratum"] == stratum),
            "judged_ok": n,
            "judge_accepted": sum(1 for v in rows if v["accepted"]),
            "judge_acceptance_rate": rate(sum(1 for v in rows if v["accepted"]), n),
            "faithful_rate": rate(sum(1 for v in vd if v["faithful"]), n),
            "issue_none_rate": rate(issues.get("none", 0), n),
            "off_topic_rate": rate(issues.get("off_topic", 0), n),
            "overlong_plus_multiple_rate": rate(issues.get("overlong", 0) + issues.get("multiple_claims", 0), n),
            "heading_name_corrupt_truncated_contextless": sum(issues.get(k, 0) for k in
                ("heading_or_name", "ocr_or_encoding_error", "truncated", "contextless_fragment")),
            "attribution_correct_rate": rate(sum(1 for v in vd if v["attribution_correct"]), n),
            "claim_mode_correct_rate": rate(sum(1 for v in vd if v["claim_mode_correct"]), n),
            "epistemic_status_correct_rate": rate(sum(1 for v in vd if v["epistemic_status_correct"]), n),
            "better_span_pointed": sum(1 for v in vd if v["better_span_exists_in_chunk"]),
            "issue_counts": dict(issues),
        }
        return stats

    random_stats = stratum_stats("random")
    forced_stats = stratum_stats("forced_audit_regression")
    r = random_stats
    checks = {
        "faithful >= 0.98": (r["faithful_rate"] or 0) >= THRESHOLDS["faithful_min"],
        "issue_none >= 0.85": (r["issue_none_rate"] or 0) >= THRESHOLDS["issue_none_min"],
        "off_topic <= 0.05": (r["off_topic_rate"] or 0) <= THRESHOLDS["off_topic_max"],
        "overlong+multiple <= 0.05": (r["overlong_plus_multiple_rate"] or 0) <= THRESHOLDS["overlong_plus_multiple_max"],
        "zero heading/name/corrupt/truncated/contextless": r["heading_name_corrupt_truncated_contextless"] == 0,
        "attribution_correct >= 0.90": (r["attribution_correct_rate"] or 0) >= THRESHOLDS["labels_correct_min"],
        "claim_mode_correct >= 0.90": (r["claim_mode_correct_rate"] or 0) >= THRESHOLDS["labels_correct_min"],
        "epistemic_status_correct >= 0.90": (r["epistemic_status_correct_rate"] or 0) >= THRESHOLDS["labels_correct_min"],
        "model failure rate <= 0.03 (extractor)": (arm_summary["model_failure_rate"] or 0) <= THRESHOLDS["model_failure_rate_blocker"],
        "judge failure rate <= 0.03": rate(judge_summary["judge_failed"], judge_summary["input_expressions"]) is None
            or judge_summary["judge_failed"] / max(judge_summary["input_expressions"], 1) <= THRESHOLDS["model_failure_rate_blocker"],
        "hard-zero conditions": not hard_zero,
    }
    # NOTE: these criteria are read from the judge's answers, i.e. the retained set AFTER
    # the judge. Judge-rejected rows are what the judge caught; they appear in 'yield'.

    # yield
    per_chunk = defaultdict(lambda: {"v1": 0, "screen_retained": 0, "final": 0, "screen_rejected": 0, "judge_rejected": 0})
    for k, row in chunks.items():
        per_chunk[k]["v1"] = row["v1_item_count"]
        per_chunk[k]["stratum"] = row["selection_stratum"]
    for v in verdicts:
        per_chunk[(v["document_id"], v["chunk_index"])]["screen_retained"] += 1
    for rec in final:
        per_chunk[(rec["document_id"], rec["chunk_index"])]["final"] += 1
    for rec in screen_rejected:
        per_chunk[(rec["document_id"], rec["chunk_index"])]["screen_rejected"] += 1
    for rec in judge_rejected:
        per_chunk[(rec["document_id"], rec["chunk_index"])]["judge_rejected"] += 1
    called = arm_summary["chunks"]["called"]
    yield_stats = {
        "chunks_called": called, "v1_items": sum(p["v1"] for p in per_chunk.values()),
        "emitted": arm_summary["candidates"]["emitted"], "screen_retained": arm_summary["candidates"]["retained"],
        "screen_rejected": arm_summary["candidates"]["rejected"], "judge_rejected": judge_summary["rejected_by_judge"],
        "judge_failed": judge_summary["judge_failed"], "final_retained": len(final),
        "final_per_chunk": rate(len(final), called), "v1_per_chunk": rate(sum(p["v1"] for p in per_chunk.values()), called),
        "final_over_v1": rate(len(final), sum(p["v1"] for p in per_chunk.values())),
        "chunks_with_zero_final": sum(1 for p in per_chunk.values() if p["final"] == 0),
        "screen_rejection_codes": arm_summary["rejection_codes"],
        "judge_issue_counts_on_rejected": dict(Counter(rec["judge_verdict"]["extraction_issue"] for rec in judge_rejected if rec.get("judge_verdict"))),
        "per_chunk": [{"document_id": d, "chunk_index": c, **p} for (d, c), p in sorted(per_chunk.items())],
    }

    # missed / diagnostics
    missed = {
        "better_span_pointers": [
            {"key": v["key"], "expression": v["verbatim_expression"], "better_span": v["verdict"]["better_span"],
             "better_span_verbatim": v.get("better_span_verbatim"), "accepted": v["accepted"]}
            for v in verdicts if v["judge_status"] == "ok" and v["verdict"]["better_span_exists_in_chunk"]],
        "label_disagreements": [
            {"key": v["key"], "expression": v["verbatim_expression"],
             "fields": [f for f in ("attribution_correct", "claim_mode_correct", "epistemic_status_correct") if not v["verdict"][f]],
             "recommended_epistemic_status": v["verdict"]["recommended_epistemic_status"], "note": v["verdict"]["reasoning_note"]}
            for v in verdicts if v["judge_status"] == "ok" and not all(v["verdict"][f] for f in ("attribution_correct", "claim_mode_correct", "epistemic_status_correct"))],
        "judge_rejections": [
            {"key": f"{rec['document_id']}:{rec['chunk_index']}#{rec['candidate_rank']}", "expression": rec["verbatim_expression"],
             "detail": rec["rule_detail"]} for rec in judge_rejected],
    }

    # regression: forced chunks
    final_by_chunk = defaultdict(list)
    for rec in final:
        final_by_chunk[(rec["document_id"], rec["chunk_index"])].append(rec["verbatim_expression"])
    regression = []
    for document_id, chunk_index, note in FORCED_CHUNKS:
        key = next((k for k in chunks if k[0] == document_id and (chunk_index == "most_corrupted" or k[1] == chunk_index)
                    and chunks[k]["selection_stratum"] == "forced_audit_regression"), None)
        if key is None:
            continue
        failure_text = V1_FAILURE_TEXT.get(key, "")
        retained_here = final_by_chunk.get(key, [])
        reappeared = any(failure_text and failure_text in t for t in retained_here)
        regression.append({"document_id": key[0], "chunk_index": key[1], "failure_class": note,
                           "v1_failure_text": failure_text, "v1_failure_reappeared_in_final": reappeared,
                           "final_retained": retained_here,
                           "screen_rejected_codes": sorted({rec["rejection_code"] for rec in screen_rejected
                                                           if (rec["document_id"], rec["chunk_index"]) == key}),
                           "judge_rejected": [rec["verbatim_expression"] for rec in judge_rejected
                                              if (rec["document_id"], rec["chunk_index"]) == key]})
    checks["no v1 failure text reappears in final retained (forced chunks)"] = not any(x["v1_failure_reappeared_in_final"] for x in regression)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(), "pilot_dir": str(pilot_dir),
        "arm": arm_name, "judge": judge_name, "judge_model": judge_summary["judge_model"],
        "basis": "model-judged (second-model verdicts), not human review",
        "thresholds": THRESHOLDS, "v1_manual_baseline_issue_none_rate": round(V1_BASELINE_ISSUE_NONE, 4),
        "checks": checks, "all_checks_pass": all(checks.values()),
        "precision_random_stratum": random_stats, "precision_forced_stratum": forced_stats,
        "yield": yield_stats, "missed_and_disagreements": missed, "regression_forced_chunks": regression,
        "hard_zero_problems": hard_zero,
    }


def to_markdown(report: dict) -> str:
    lines = [f"# v2 pilot acceptance report ({report['arm']} / {report['judge']})",
             f"Basis: {report['basis']}. Generated {report['generated_at']}.", "",
             "## Checks (random stratum unless stated)", ""]
    for name, ok in report["checks"].items():
        lines.append(f"- {'PASS' if ok else 'FAIL'} — {name}")
    lines += ["", f"**All checks pass: {report['all_checks_pass']}**", "", "## Precision", "",
              "| metric | random | forced |", "|---|---|---|"]
    for k in ("screen_retained", "judged_ok", "judge_accepted", "judge_acceptance_rate", "faithful_rate", "issue_none_rate",
              "off_topic_rate", "overlong_plus_multiple_rate", "heading_name_corrupt_truncated_contextless",
              "attribution_correct_rate", "claim_mode_correct_rate", "epistemic_status_correct_rate", "better_span_pointed"):
        lines.append(f"| {k} | {report['precision_random_stratum'][k]} | {report['precision_forced_stratum'][k]} |")
    y = report["yield"]
    lines += ["", "## Yield", "", f"- chunks called: {y['chunks_called']}; v1 items: {y['v1_items']} ({y['v1_per_chunk']}/chunk)",
              f"- emitted {y['emitted']} → screen retained {y['screen_retained']} (rejected {y['screen_rejected']}: {y['screen_rejection_codes']})",
              f"- judge rejected {y['judge_rejected']} ({y['judge_issue_counts_on_rejected']}), judge failed {y['judge_failed']}",
              f"- **final retained {y['final_retained']}** = {y['final_per_chunk']}/chunk = {y['final_over_v1']} of v1; chunks with zero final: {y['chunks_with_zero_final']}",
              "", "## Missed material and disagreements (listed for inspection)", ""]
    m = report["missed_and_disagreements"]
    lines.append(f"- better-span pointers: {len(m['better_span_pointers'])}")
    for p in m["better_span_pointers"]:
        lines.append(f"  - {p['key']} [{'accepted' if p['accepted'] else 'rejected'}]: \"{p['expression'][:90]}\" → \"{p['better_span'][:90]}\" (verbatim: {p['better_span_verbatim']})")
    lines.append(f"- label disagreements: {len(m['label_disagreements'])}")
    for d in m["label_disagreements"]:
        lines.append(f"  - {d['key']}: {d['fields']} rec={d['recommended_epistemic_status'] or '-'} — {d['note']}")
    lines.append(f"- judge rejections: {len(m['judge_rejections'])}")
    for j in m["judge_rejections"]:
        lines.append(f"  - {j['key']}: \"{j['expression'][:90]}\" — {j['detail']}")
    lines += ["", "## Forced-regression chunks", ""]
    for x in report["regression_forced_chunks"]:
        lines.append(f"- {x['document_id'][:40]}:{x['chunk_index']} — {x['failure_class']}")
        lines.append(f"  - v1 failure text \"{x['v1_failure_text']}\" reappeared: {x['v1_failure_reappeared_in_final']}")
        lines.append(f"  - final retained ({len(x['final_retained'])}): " + " || ".join(t[:80] for t in x["final_retained"]))
        if x["screen_rejected_codes"] or x["judge_rejected"]:
            lines.append(f"  - screen codes: {x['screen_rejected_codes']}; judge rejected: {[t[:60] for t in x['judge_rejected']]}")
    if report["hard_zero_problems"]:
        lines += ["", "## HARD-ZERO PROBLEMS", ""] + [f"- {p}" for p in report["hard_zero_problems"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date-tag", required=True)
    parser.add_argument("--pilot-root", type=Path, default=DEFAULT_PILOT_ROOT)
    parser.add_argument("--arm", default="arm_qwen3-4b")
    parser.add_argument("--judge", default="judge_qwen3-8b")
    parser.add_argument("--review-name", default="review")
    args = parser.parse_args()
    pilot_dir = args.pilot_root / f"pilot_{args.date_tag}"
    report = evaluate(pilot_dir, args.arm, args.judge)
    review_dir = pilot_dir / args.review_name
    review_dir.mkdir(exist_ok=True)
    write_json(review_dir / "acceptance_report.json", report)
    (review_dir / "acceptance_report.md").write_text(to_markdown(report), encoding="utf-8", newline="\n")
    print(to_markdown(report))
    print(f"Report: {review_dir / 'acceptance_report.md'}")


if __name__ == "__main__":
    main()
