"""End-to-end test of pilot_v2_literature's run and build-review stages with a
stubbed model (no Ollama), on a temporary pilot directory.

Checks: arm outputs written, candidate/chunk accounting reconciles, model
failures yield no retained rows, the review packet is under the cap with
every manual column blank, spans resolve, and re-running refuses to
overwrite.

Run from thesis/corpus/:  python -m unittest thesis_corpus.tests.test_pilot_v2_stages -v
"""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from thesis_corpus import pilot_v2_literature as pilot
from thesis_corpus import screen_v2 as sv

CHUNK_A = ("Bromley argues that behaviors used by NRMs to recruit and keep members are misleading at best and coercive at worst. "
           "The Course of Growth. While the growth of cults has not been alarming, it has been noticeable. "
           "Well, perhaps to some extent. Members surrender their savings to the sect, critics allege. " * 2)
CHUNK_B = "Cults are said to isolate members from their families, and the guru claims divine status over adepts. " * 6
CHUNK_C = "This chunk will fail at the model. Cults again. " * 8


def _chunk_row(doc, idx, text, stratum="random", v1=2, skip=None):
    ctx = sv.prepare_chunk(doc, idx, [1, 1], text, "literature")
    return {
        "document_id": doc, "chunk_index": idx, "page_range": [1, 1], "selection_stratum": stratum,
        "selection_note": "test", "raw_text": ctx.raw_text, "nfc_text": ctx.nfc_text,
        "raw_chunk_sha256": ctx.raw_sha256, "nfc_chunk_sha256": ctx.nfc_sha256,
        "nfc_changed_codepoints": ctx.nfc_changed_codepoints, "word_count": ctx.word_count,
        "document_integrity_flags": [], "corrupted_line_texts": [], "corrupted_regions": [],
        "chunk_integrity_score": ctx.chunk_integrity_score, "v1_item_count": v1, "pre_screen_skip_code": skip,
    }


def _v1_item(doc, idx, line, quote, emb, cw):
    return {"raw_archive_line": line, "document_id": doc, "chunk_index": idx, "page_range": [1, 1],
            "source_quote": quote, "embedding_text": emb, "entity_anchors": [], "claim_mode": "direct_statement",
            "epistemic_status": "asserted", "attribution": "author", "context_window": cw}


def _candidate(text, kind="claim", **kw):
    base = dict(verbatim_expression=text, expression_kind=kind, attribution="author", claim_mode="direct_statement",
                epistemic_status="asserted", entity_anchors=[], self_contained=True, cult_relevant=True,
                textually_intelligible=True, single_coherent_expression=True, relevance_note="n")
    base.update(kw)
    return base


RESPONSES = {
    ("docA", 0): {"chunk_relevance": "relevant", "domain_terms": ["NRMs", "emprise"], "expressions": [
        _candidate("behaviors used by NRMs to recruit and keep members are misleading at best and coercive at worst"),
        _candidate("The Course of Growth."),
        _candidate("Well, perhaps to some extent."),
        _candidate("Members surrender their savings to the sect, critics allege", epistemic_status="contested"),
        {"verbatim_expression": "Cults", "expression_kind": "claim"},
    ]},
    ("docB", 5): {"chunk_relevance": "relevant", "domain_terms": [], "expressions": [
        _candidate("Cults are said to isolate members from their families"),
        _candidate("the guru claims divine status over adepts"),
    ]},
}


class StubResult:
    def __init__(self, content):
        self.content = content
        self.structured_output_mode = "json_schema"


def fake_chat(host, model, system_prompt, user_content, json_schema, options, think=False, timeout=180.0):
    for (doc, idx), payload in RESPONSES.items():
        if f"document_id: {doc}\n" in user_content and f"chunk_index: {idx}\n" in user_content:
            return StubResult(json.dumps(payload))
    return StubResult("this is not json {")


class PilotStagesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.pilot_dir = Path(self.tmp.name) / "pilot_test"
        self.pilot_dir.mkdir()
        rows = [_chunk_row("docA", 0, CHUNK_A, "forced_audit_regression", v1=3),
                _chunk_row("docB", 5, CHUNK_B, v1=1),
                _chunk_row("docC", 2, CHUNK_C, v1=1),
                _chunk_row("docD", 9, "short", v1=0, skip="chunk_too_short")]
        pilot.write_jsonl(self.pilot_dir / "pilot_chunks.jsonl", rows)
        pilot.write_jsonl(self.pilot_dir / "skipped_chunks.jsonl", [{"document_id": "docD", "chunk_index": 9, "skip_code": "chunk_too_short"}])
        pilot.write_jsonl(self.pilot_dir / "pilot_v1_items.jsonl", [
            _v1_item("docA", 0, 10, "behaviors used by NRMs to recruit", "NRM recruitment is coercive", CHUNK_A),
            _v1_item("docA", 0, 11, "Well, perhaps to some extent.", "Well, perhaps to some extent.", CHUNK_A),
            _v1_item("docA", 0, 12, "The Course of Growth.", "The Course of Growth.", CHUNK_A),
            _v1_item("docB", 5, 20, "the guru claims divine status", "the guru claims divine status", CHUNK_B),
            _v1_item("docC", 2, 30, "Cults again.", "Cults again.", CHUNK_C),
        ])
        self.args = SimpleNamespace(model="stub:1b", arm_tag=None, ollama_host="http://stub", timeout=1.0, seed=42,
                                    date_tag="test", v1_sample_per_chunk=2, v2_sample_per_chunk=3,
                                    review_row_cap=150, blind=False)

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self):
        with mock.patch("thesis_corpus.ollama_client.chat_structured", fake_chat), \
             mock.patch("thesis_corpus.ollama_client.check_available", lambda host: None):
            pilot.stage_run(self.args, self.pilot_dir)

    def test_run_and_review(self):
        self._run()
        arm = self.pilot_dir / "arm_stub-1b"
        summary = json.loads((arm / "summary.json").read_text())
        retained = pilot.read_jsonl(arm / "expressions_v2.jsonl")
        rejected = pilot.read_jsonl(arm / "rejected_candidates.jsonl")
        failures = pilot.read_jsonl(arm / "model_failures.jsonl")

        self.assertEqual(summary["chunks"], {"selected": 4, "skipped_pre_screen": 1, "called": 3, "annotated": 2,
                                             "model_failed": 1, "reconciled": True})
        self.assertTrue(summary["candidates"]["reconciled"])
        self.assertEqual(summary["candidates"]["emitted"], 7)
        self.assertEqual(len(retained), 4)
        self.assertEqual(len(rejected), 3)
        self.assertEqual([f["document_id"] for f in failures], ["docC"])
        self.assertEqual(summary["retained_from_failed_chunks"], 0)
        codes = sorted(r["rejection_code"] for r in rejected)
        self.assertEqual(codes, ["heading_or_scaffold", "missing_required_field", "short_fragment_no_referent"])
        for r in retained:
            self.assertEqual(r["embedding_text"], r["verbatim_expression"])
            self.assertEqual(r["text_transform"], "none")
            self.assertEqual(r["extraction_version"], pilot.schema.EXTRACTION_VERSION)
        terms = pilot.read_jsonl(arm / "chunk_terms.jsonl")
        self.assertEqual(terms[0]["domain_terms"], ["NRMs"])
        self.assertEqual(terms[0]["dropped_terms"][0]["reason"], "term_not_in_chunk")
        config = json.loads((arm / "config.json").read_text())
        self.assertEqual(config["screening"]["prompt_sha256"], pilot.schema.PROMPT_SHA256)

        # second run refuses to overwrite the arm
        with self.assertRaises(SystemExit):
            self._run()

        # build-review
        pilot.stage_build_review(self.args, self.pilot_dir)
        review_dir = self.pilot_dir / "review"
        with open(review_dir / "pilot_review.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertLessEqual(len(rows), 150)
        arms = {r["arm"] for r in rows}
        self.assertEqual(arms, {"v1", "v2_stub-1b"})
        self.assertEqual(sum(1 for r in rows if r["arm"] == "v1"), 2 + 1 + 1)   # docA sampled to 2, docB 1, docC 1
        self.assertEqual(sum(1 for r in rows if r["arm"].startswith("v2")), 4)
        for r in rows:
            for col in pilot.MANUAL_REVIEW_COLUMNS:
                self.assertEqual(r[col], "", col)
            if r["arm"].startswith("v2"):
                self.assertEqual(r["embedding_equals_verbatim"], "yes")
                self.assertIn(r["verbatim_expression"], r["context_window"])
        # random stratum first
        self.assertEqual(rows[0]["selection_stratum"], "random")
        self.assertEqual(rows[-1]["selection_stratum"], "forced_audit_regression")
        with open(review_dir / "pilot_chunk_comparison.csv", encoding="utf-8") as f:
            comparison = list(csv.DictReader(f))
        self.assertEqual(len(comparison), 4)
        by_key = {(c["document_id"], int(c["chunk_index"])): c for c in comparison}
        self.assertEqual(by_key[("docA", 0)]["v2_stub-1b_retained_count"], "2")
        self.assertEqual(by_key[("docA", 0)]["v1_count"], "3")
        self.assertEqual(by_key[("docD", 9)]["pre_screen_skip_code"], "chunk_too_short")
        validation = json.loads((review_dir / "validation.json").read_text())
        self.assertEqual(validation["validations"][0]["hard_zero_problems"], [])
        self.assertTrue(validation["validations"][0]["failure_rate_exceeds_review_blocker"])  # 1/3 in this stub

    def test_blind_review(self):
        self.args.blind = True
        self._run()
        pilot.stage_build_review(self.args, self.pilot_dir)
        review_dir = self.pilot_dir / "review"
        with open(review_dir / "pilot_review.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertTrue(all(r["arm"] == "" for r in rows))
        with open(review_dir / "review_key.csv", encoding="utf-8") as f:
            key = list(csv.DictReader(f))
        self.assertEqual(len(key), len(rows))
        self.assertEqual({k["arm"] for k in key}, {"v1", "v2_stub-1b"})


if __name__ == "__main__":
    unittest.main()
