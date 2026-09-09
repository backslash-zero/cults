"""End-to-end test of the judge stage, judge-aware review, and evaluation with
stubbed models (no Ollama).

Run from thesis/corpus/:  python -m unittest thesis_corpus.tests.test_judge_v2 -v
"""
import csv
import json
import unittest
from pathlib import Path
from unittest import mock

from thesis_corpus import judge_v2
from thesis_corpus import pilot_v2_literature as pilot
from thesis_corpus import pilot_v2_evaluate as evaluate
import thesis_corpus.tests.test_pilot_v2_stages as base
StubResult = base.StubResult


def _verdict(**overrides):
    base = dict(faithful=True, self_contained=True, cult_relevant=True, textually_intelligible=True, atomic=True,
                attribution_correct=True, claim_mode_correct=True, epistemic_status_correct=True,
                recommended_epistemic_status="", better_span_exists_in_chunk=False, better_span="",
                extraction_issue="none", reasoning_note="fine")
    base.update(overrides)
    return base


def fake_judge_chat(host, model, system_prompt, user_content, json_schema, options, think=False, timeout=180.0):
    if "behaviors used by NRMs" in user_content.split("Extracted expression")[1]:
        return StubResult(json.dumps(_verdict()))
    if "critics allege" in user_content.split("Extracted expression")[1]:
        return StubResult(json.dumps(_verdict(cult_relevant=False, extraction_issue="off_topic",
                                              epistemic_status_correct=False, recommended_epistemic_status="contested")))
    if "guru claims" in user_content.split("Extracted expression")[1]:
        return StubResult("not json at all")
    return StubResult(json.dumps(_verdict(better_span_exists_in_chunk=True,
                                          better_span="Cults are said to isolate members from their families")))


class JudgeStageTest(base.PilotStagesTest):
    test_run_and_review = None  # inherited fixtures only; those tests run in their own module
    test_blind_review = None

    def test_judge_review_evaluate(self):
        self._run()
        arm = self.pilot_dir / "arm_stub-1b"
        with mock.patch("thesis_corpus.ollama_client.chat_structured", fake_judge_chat), \
             mock.patch("thesis_corpus.ollama_client.check_available", lambda host: None):
            jd = judge_v2.judge_arm(arm, "http://stub", "judge:8b")
        self.assertEqual(jd.name, "judge_judge-8b")
        summary = json.loads((jd / "summary.json").read_text())
        self.assertEqual(summary["input_expressions"], 4)
        self.assertEqual(summary["accepted"], 2)
        self.assertEqual(summary["rejected_by_judge"], 1)
        self.assertEqual(summary["judge_failed"], 1)
        self.assertTrue(summary["reconciled"])
        self.assertEqual(summary["label_disagreements"], {"epistemic_status_correct": 1})
        self.assertEqual(summary["better_span_pointed"], 1)
        accepted = pilot.read_jsonl(jd / "expressions_v2_judged.jsonl")
        self.assertTrue(all(r["judge_accepted"] for r in accepted))
        self.assertIn("judge_better_span", [f for r in accepted for f in r["judge_flags"]])
        rejected = pilot.read_jsonl(jd / "judge_rejected.jsonl")
        self.assertEqual(sorted(r["rejection_code"] for r in rejected), ["judge_failed", "judge_rejected"])
        # second judge run with the same model refuses; a different model gets its own dir
        with self.assertRaises(SystemExit):
            with mock.patch("thesis_corpus.ollama_client.chat_structured", fake_judge_chat), \
                 mock.patch("thesis_corpus.ollama_client.check_available", lambda host: None):
                judge_v2.judge_arm(arm, "http://stub", "judge:8b")

        # build-review uses the judge-accepted set and fills judge columns
        pilot.stage_build_review(self.args, self.pilot_dir)
        with open(self.pilot_dir / "review" / "pilot_review.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        v2 = [r for r in rows if r["arm"].startswith("v2")]
        self.assertEqual(len(v2), 2)
        self.assertTrue(all(r["judge_model"] == "judge:8b" and r["judge_extraction_issue"] == "none" for r in v2))
        self.assertTrue(all(r["judge_model"] == "" for r in rows if r["arm"] == "v1"))
        for r in rows:
            for col in pilot.MANUAL_REVIEW_COLUMNS:
                self.assertEqual(r[col], "")
        validation = json.loads((self.pilot_dir / "review" / "validation.json").read_text())
        v = validation["validations"][0]
        self.assertEqual(v["hard_zero_problems"], [])
        self.assertEqual((v["screen_retained"], v["final_retained"], v["judge"]), (4, 2, "judge_judge-8b"))
        with open(self.pilot_dir / "review" / "pilot_rejected_candidates.csv", encoding="utf-8") as f:
            rej = list(csv.DictReader(f))
        self.assertEqual(sorted(r["rejection_code"] for r in rej),
                         sorted(["heading_or_scaffold", "missing_required_field", "short_fragment_no_referent",
                                 "judge_rejected", "judge_failed"]))

        # evaluation report
        report = evaluate.evaluate(self.pilot_dir, "arm_stub-1b", "judge_judge-8b")
        self.assertEqual(report["yield"]["final_retained"], 2)
        self.assertEqual(report["yield"]["judge_rejected"], 1)
        self.assertEqual(report["hard_zero_problems"], [])
        self.assertIn("judge failure rate <= 0.03", report["checks"])
        self.assertFalse(report["checks"]["judge failure rate <= 0.03"])  # 1/4 in this stub
        md = evaluate.to_markdown(report)
        self.assertIn("## Yield", md)


class JudgeRuleTest(unittest.TestCase):
    def test_accept_rule_and_flags(self):
        v = judge_v2.JudgeVerdictV2.model_validate(_verdict())
        self.assertTrue(judge_v2.judge_accepts(v))
        self.assertEqual(judge_v2.judge_flags(v), [])
        v = judge_v2.JudgeVerdictV2.model_validate(_verdict(atomic=False))
        self.assertFalse(judge_v2.judge_accepts(v))
        v = judge_v2.JudgeVerdictV2.model_validate(_verdict(extraction_issue="overlong"))
        self.assertFalse(judge_v2.judge_accepts(v))
        v = judge_v2.JudgeVerdictV2.model_validate(_verdict(attribution_correct=False, better_span_exists_in_chunk=True, better_span="x"))
        self.assertTrue(judge_v2.judge_accepts(v))  # label disagreement never rejects
        self.assertEqual(judge_v2.judge_flags(v), ["judge_attribution_correct_disagrees", "judge_better_span"])

    def test_strict_parsing(self):
        with self.assertRaises(Exception):
            judge_v2.parse_verdict(json.dumps(_verdict(faithful="true")))
        with self.assertRaises(Exception):
            judge_v2.parse_verdict(json.dumps(_verdict(extraction_issue="irrelevant")))
        v = judge_v2.parse_verdict("```json\n" + json.dumps(_verdict(recommended_epistemic_status=None)) + "\n```")
        self.assertEqual(v.recommended_epistemic_status, "")


if __name__ == "__main__":
    unittest.main()
