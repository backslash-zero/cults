"""extract_v2 end-to-end with stubbed models on a temporary corpus, plus the
two rules added after the pilot (subtitled-title rejection, newline folding).

Run from thesis/corpus/:  python -m unittest thesis_corpus.tests.test_extract_v2 -v
"""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from thesis_corpus import extract_v2, screen_v2 as sv, extraction_v2_schema as schema
from thesis_corpus.tests.test_screen_v2 import screen
from thesis_corpus.tests.test_judge_v2 import _verdict
from thesis_corpus.tests.test_pilot_v2_stages import StubResult, _candidate

DOC_A_TEXT = ("Cults are said to isolate members from their families, critics argue. " * 6 +
              "The leader\ndemands total obedience from adepts, according to former members. " * 4)
DOC_B_TEXT = "General history of the region with no relevant material at all. " * 12


def fake_chat(host, model, system_prompt, user_content, json_schema, options, think=False, timeout=180.0):
    if "Extracted expression" in user_content:  # judge call
        expr = user_content.split("Extracted expression (exact substring):\n")[1].split("\n\n")[0]
        if "General history" in expr:
            return StubResult(json.dumps(_verdict(cult_relevant=False, extraction_issue="off_topic")))
        return StubResult(json.dumps(_verdict()))
    if "document_id: docA" in user_content:
        return StubResult(json.dumps({"chunk_relevance": "relevant", "domain_terms": ["adepts"], "expressions": [
            _candidate("Cults are said to isolate members from their families, critics argue"),
            _candidate("The leader\ndemands total obedience from adepts, according to former members"),
        ]}))
    if "document_id: docB" in user_content:
        return StubResult(json.dumps({"chunk_relevance": "relevant", "domain_terms": [], "expressions": [
            _candidate("General history of the region with no relevant material at all", kind="association_or_framing"),
        ]}))
    return StubResult("{ broken")


class NewRulesTest(unittest.TestCase):
    def test_subtitled_title_rejected_even_when_tagged(self):
        chunk = ("55. DWP data were drawn largely from archival material, including my own study, The Cadre Ideal: "
                 "Origins and Development of a Political Cult, and interviews with former members of the cult.")
        out = screen("The Cadre Ideal: Origins and Development of a Political Cult", chunk, kind="example_or_named_group")
        self.assertEqual(out.rejection_code, "heading_or_scaffold")
        # named groups and ordinary clauses with a colon are unaffected
        self.assertIsNone(screen("International Church of Christ", "He joined the International Church of Christ, a sect.",
                                 kind="example_or_named_group").rejection_code)
        self.assertIsNone(screen("Cults share one trait: a leader who demands obedience",
                                 "As Barker notes, Cults share one trait: a leader who demands obedience from adepts.").rejection_code)

    def test_newline_folding_recorded(self):
        ctx = sv.prepare_chunk("d", 0, [1, 1], "Intro sentence about sects.\nThe leader\ndemands total obedience from adepts, according to former members.", "literature")
        retained, rejected = sv.screen_chunk([_candidate("The leader\ndemands total obedience from adepts, according to former members")], ctx, set())
        self.assertEqual(len(retained), 1)
        rec = retained[0]
        self.assertIn("\n", rec["verbatim_expression"])
        self.assertEqual(rec["embedding_text"], "The leader demands total obedience from adepts, according to former members")
        self.assertEqual(rec["text_transform"], "newline_to_space")
        self.assertEqual(sv.recheck_retained_record(rec, ctx), [])
        rec2 = dict(rec, embedding_text=rec["verbatim_expression"], text_transform="none")
        self.assertTrue(sv.recheck_retained_record(rec2, ctx))


class ExtractV2Test(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.processed = root / "processed"
        docs = self.processed / "literature" / "documents"
        for name, text in (("docA", DOC_A_TEXT), ("docB", DOC_B_TEXT), ("docC", "Cults again. " * 60)):
            d = docs / name
            d.mkdir(parents=True)
            (d / "pages.jsonl").write_text(json.dumps({"document_id": name, "page_number": 1, "text": text,
                                                      "extraction_method": "native", "character_count": len(text), "warnings": []}) + "\n", encoding="utf-8")
        audits = self.processed / "audits"
        audits.mkdir()
        self.audit_csv = audits / "stage1_text_integrity_test.csv"
        with open(self.audit_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["corpus", "document_id", "document_integrity_flag"])
            w.writeheader()
            for name in ("docA", "docB", "docC"):
                w.writerow({"corpus": "literature", "document_id": name, "document_integrity_flag": ""})
        self.args = SimpleNamespace(corpus="literature", run_tag="t1", model="stub:4b", judge_model="stub:8b",
                                    ollama_host="http://stub", timeout=1.0, judge_timeout=1.0, limit=None,
                                    audit_csv=self.audit_csv, out_root=self.processed / "v2")
        self.patches = [mock.patch.object(extract_v2, "PROCESSED_ROOT", self.processed)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def _run(self, **overrides):
        args = SimpleNamespace(**{**vars(self.args), **overrides})
        return extract_v2.run(args, chat=fake_chat, check=lambda host: None)

    def test_full_run_resume_and_accounting(self):
        run_dir = self._run(limit=2)
        summary = json.loads((run_dir / "summary.json").read_text())
        self.assertEqual(summary["documents"]["done"], 2)
        done = (run_dir / "documents_done.txt").read_text().split()
        self.assertEqual(done, ["docA", "docB"])
        # resume finishes docC (whose model reply is broken -> failure, no rows)
        run_dir2 = self._run()
        self.assertEqual(run_dir2, run_dir)
        summary = json.loads((run_dir / "summary.json").read_text())
        self.assertEqual(summary["documents"]["done"], 3)
        self.assertEqual(summary["chunks"]["model_failed"], 1)
        self.assertTrue(summary["candidates"]["reconciled"])
        self.assertTrue(summary["candidates"]["judge_reconciled"])
        final = extract_v2.read_jsonl(run_dir / "expressions_v2.jsonl")
        self.assertEqual([r["document_id"] for r in final], ["docA", "docA"])
        self.assertTrue(all(r["judge_accepted"] for r in final))
        folded = [r for r in final if r["text_transform"] == "newline_to_space"]
        self.assertEqual(len(folded), 1)
        self.assertNotIn("\n", folded[0]["embedding_text"])
        judge_rej = extract_v2.read_jsonl(run_dir / "judge_rejected.jsonl")
        self.assertEqual([r["document_id"] for r in judge_rej], ["docB"])
        self.assertEqual(summary["candidates"]["final"], 2)
        # a third run has nothing to do and must not duplicate rows
        self._run()
        self.assertEqual(len(extract_v2.read_jsonl(run_dir / "expressions_v2.jsonl")), 2)
        self.assertEqual(len((run_dir / "documents_done.txt").read_text().split()), 3)
        # resuming with a different model is refused
        with self.assertRaises(SystemExit):
            self._run(model="other:1b")

    def test_no_judge_mode(self):
        run_dir = self._run(judge_model=None, limit=1)
        final = extract_v2.read_jsonl(run_dir / "expressions_v2.jsonl")
        self.assertEqual(len(final), 2)
        self.assertNotIn("judge_accepted", final[0])
        summary = json.loads((run_dir / "summary.json").read_text())
        self.assertIsNone(summary["candidates"]["judge_reconciled"])


if __name__ == "__main__":
    unittest.main()
