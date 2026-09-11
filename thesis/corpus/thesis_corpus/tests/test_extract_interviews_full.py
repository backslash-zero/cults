"""extract_interviews_full end-to-end with stubbed models on a temporary
corpus: exercises the driver's checkpointing/resume, the missing-audit
tolerance (unlike extract_v2, a document absent from the Stage-1 audit must
NOT block the run), and the two behaviors this pipeline exists to change --
a cult_relevant=False expression stays in the final archive, and an
interviewer turn is kept (attributed), not discarded.

Run from thesis/corpus/:  python -m unittest thesis_corpus.tests.test_extract_interviews_full -v
"""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from thesis_corpus import extract_interviews_full as eif
from thesis_corpus.tests.test_screen_interviews_full import _candidate_full


class StubResult:
    def __init__(self, content):
        self.content = content
        self.structured_output_mode = "json_schema"


def fake_chat(host, model, system_prompt, user_content, json_schema, options, think=False, timeout=180.0):
    if "document_id: docA" in user_content:
        return StubResult(json.dumps({"domain_terms": ["Illuminati"], "expressions": [
            _candidate_full("When you hear the word cult, what comes to mind?"),
            _candidate_full("Illuminati.", cult_relevant=False),
        ]}))
    if "document_id: docB" in user_content:
        return StubResult(json.dumps({"domain_terms": [], "expressions": [
            _candidate_full("Not much."),
        ]}))
    return StubResult("{ broken")


class ExtractInterviewsFullTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.processed = root / "processed"
        docs = self.processed / "interviews" / "documents"
        texts = {
            "docA": "Interviewer: When you hear the word cult, what comes to mind?\n\nInterviewee: Illuminati.\n\n",
            "docB": "Interviewer: Tell me more.\n\nInterviewee: Not much.\n\n",
        }
        for name, text in texts.items():
            d = docs / name
            d.mkdir(parents=True)
            (d / "pages.jsonl").write_text(json.dumps({"document_id": name, "page_number": 1, "text": text,
                                                      "extraction_method": "transcript", "character_count": len(text),
                                                      "warnings": []}) + "\n", encoding="utf-8")
        audits = self.processed / "audits"
        audits.mkdir()
        self.audit_csv = audits / "stage1_text_integrity_test.csv"
        with open(self.audit_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["corpus", "document_id", "document_integrity_flag"])
            w.writeheader()
            w.writerow({"corpus": "interviews", "document_id": "docA", "document_integrity_flag": ""})
            # docB deliberately absent from the audit -- must not block the run.
        self.args = SimpleNamespace(run_tag="t1", model="stub:4b", judge_model=None,
                                    ollama_host="http://stub", timeout=1.0, judge_timeout=1.0, limit=None,
                                    audit_csv=self.audit_csv, max_words=None, out_root=self.processed / "interviews_full")
        self.patches = [mock.patch.object(eif, "PROCESSED_ROOT", self.processed)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def _run(self, **overrides):
        args = SimpleNamespace(**{**vars(self.args), **overrides})
        return eif.run(args, chat=fake_chat, check=lambda host: None)

    def test_missing_audit_document_does_not_block_run(self):
        run_dir = self._run()
        summary = json.loads((run_dir / "summary.json").read_text())
        self.assertEqual(summary["documents"]["done"], 2)
        final = eif.read_jsonl(run_dir / "expressions_v2.jsonl")
        self.assertEqual({r["document_id"] for r in final}, {"docA", "docB"})

    def test_cult_relevant_false_expression_kept_in_final_archive(self):
        run_dir = self._run()
        final = eif.read_jsonl(run_dir / "expressions_v2.jsonl")
        illuminati = [r for r in final if r["verbatim_expression"] == "Illuminati."]
        self.assertEqual(len(illuminati), 1)
        self.assertFalse(illuminati[0]["cult_relevant"])

    def test_interviewer_turn_kept_with_attribution(self):
        run_dir = self._run()
        final = eif.read_jsonl(run_dir / "expressions_v2.jsonl")
        q = [r for r in final if r["verbatim_expression"] == "When you hear the word cult, what comes to mind?"]
        self.assertEqual(len(q), 1)
        self.assertEqual(q[0]["attribution"], "interviewer")

    def test_domain_terms_recorded_regardless_of_expression_outcome(self):
        run_dir = self._run()
        terms = eif.read_jsonl(run_dir / "chunk_terms.jsonl")
        docA_terms = [t for t in terms if t["document_id"] == "docA"]
        self.assertEqual(docA_terms[0]["domain_terms"], ["Illuminati"])

    def test_resume_and_no_duplication(self):
        run_dir = self._run(limit=1)
        done = (run_dir / "documents_done.txt").read_text().split()
        self.assertEqual(done, ["docA"])
        run_dir2 = self._run()
        self.assertEqual(run_dir2, run_dir)
        self.assertEqual(len((run_dir / "documents_done.txt").read_text().split()), 2)
        self._run()  # nothing left to do, must not duplicate rows
        final = eif.read_jsonl(run_dir / "expressions_v2.jsonl")
        self.assertEqual(len({r["document_id"] for r in final}), 2)

    def test_resume_refused_on_model_change(self):
        self._run(limit=1)
        with self.assertRaises(SystemExit):
            self._run(model="other:1b")

    def test_no_judge_fields_when_judge_skipped(self):
        run_dir = self._run(limit=1)
        final = eif.read_jsonl(run_dir / "expressions_v2.jsonl")
        self.assertTrue(final)
        self.assertNotIn("judge_accepted", final[0])
        self.assertFalse((run_dir / "judge_verdicts.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
