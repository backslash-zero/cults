"""Unit tests for sample_v2_expressions (stdlib unittest, no Ollama).

Run from thesis/corpus/:  python -m unittest thesis_corpus.tests.test_sample_v2_expressions -v
"""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from thesis_corpus import sample_v2_expressions as sve


def make_row(document_id, chunk_index, rank=1):
    return {
        "document_id": document_id, "chunk_index": chunk_index, "candidate_rank": rank,
        "expression_kind": "claim", "verbatim_expression": f"{document_id}:{chunk_index} text",
        "embedding_text": f"{document_id}:{chunk_index} text", "text_transform": "none",
        "attribution": "author", "attribution_source": "model", "model_attribution": "author",
        "claim_mode": "direct_statement", "epistemic_status": "asserted", "entity_anchors": ["x"],
        "context_window": "ctx", "screen_flags": [],
        "judge_verdict": {"faithful": True, "self_contained": True, "cult_relevant": True,
                          "textually_intelligible": True, "atomic": True, "attribution_correct": True,
                          "claim_mode_correct": True, "epistemic_status_correct": True,
                          "recommended_epistemic_status": "", "better_span_exists_in_chunk": False,
                          "better_span": "", "extraction_issue": "none", "reasoning_note": "ok"},
        "judge_model": "judge:8b",
    }


class FloorProportionalTest(unittest.TestCase):
    def test_matches_pilot_style_allocation(self):
        alloc = sve.floor_then_proportional({"a": 522, "b": 2, "c": 245}, 100)
        self.assertEqual(sum(alloc.values()), 100)
        self.assertGreaterEqual(alloc["b"], 1)  # small doc still represented
        self.assertLessEqual(alloc["a"], 522)
        self.assertLessEqual(alloc["b"], 2)
        self.assertLessEqual(alloc["c"], 245)

    def test_never_exceeds_group_size(self):
        alloc = sve.floor_then_proportional({"a": 1, "b": 1, "c": 1}, 10)
        self.assertEqual(alloc, {"a": 1, "b": 1, "c": 1})


class SampleRowsTest(unittest.TestCase):
    def setUp(self):
        self.rows = [make_row("docA", i) for i in range(80)] + [make_row("docB", i) for i in range(20)]

    def test_uniform_reproducible(self):
        a = sve.sample_rows(self.rows, 10, seed=42, stratify_by_document=False)
        b = sve.sample_rows(self.rows, 10, seed=42, stratify_by_document=False)
        self.assertEqual(a, b)
        c = sve.sample_rows(self.rows, 10, seed=7, stratify_by_document=False)
        self.assertNotEqual(a, c)

    def test_uniform_no_duplicates_and_size(self):
        sampled = sve.sample_rows(self.rows, 15, seed=1, stratify_by_document=False)
        keys = [(r["document_id"], r["chunk_index"]) for r in sampled]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(len(sampled), 15)

    def test_n_greater_than_available_caps(self):
        sampled = sve.sample_rows(self.rows[:5], 100, seed=1, stratify_by_document=False)
        self.assertEqual(len(sampled), 5)

    def test_stratified_gives_small_group_representation(self):
        sampled = sve.sample_rows(self.rows, 10, seed=42, stratify_by_document=True)
        docs = {r["document_id"] for r in sampled}
        self.assertEqual(docs, {"docA", "docB"})
        self.assertEqual(len(sampled), 10)


class ReviewRowsTest(unittest.TestCase):
    def test_columns_and_ids(self):
        rows = sve.build_review_rows([make_row("docA", 3), make_row("docA", 1)])
        self.assertEqual([r["chunk_index"] for r in rows], [1, 3])  # sorted
        self.assertEqual([r["sample_id"] for r in rows], [1, 2])
        self.assertEqual(rows[0]["judge_extraction_issue"], "none")
        self.assertTrue(rows[0]["judge_faithful"])
        for col in sve.MANUAL_REVIEW_COLUMNS:
            self.assertEqual(rows[0][col], "")


class MainTest(unittest.TestCase):
    def test_end_to_end_and_incomplete_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "v2" / "literature" / "run_t"
            run_dir.mkdir(parents=True)
            rows = [make_row("docA", i) for i in range(6)] + [make_row("docB", i) for i in range(2)]
            with open(run_dir / "expressions_v2.jsonl", "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            docs_dir = root / "literature" / "documents"
            for name in ("docA", "docB", "docC"):  # docC not in the sample source -> run incomplete
                (docs_dir / name).mkdir(parents=True)

            args = SimpleNamespace(corpus="literature", run_tag="t", n=5, seed=1, stratify_by_document=False,
                                   out_root=root / "v2", date_tag="d1")
            # Exercise the same helpers main() calls, rather than main() itself (which parses argv).
            with mock.patch.object(sve, "PROCESSED_ROOT", root):
                run_dir2 = args.out_root / args.corpus / f"run_{args.run_tag}"
                src = run_dir2 / "expressions_v2.jsonl"
                all_rows = sve.read_jsonl(src)
                sampled = sve.sample_rows(all_rows, args.n, args.seed, args.stratify_by_document)
                review_rows = sve.build_review_rows(sampled)
                self.assertEqual(len(review_rows), 5)
                docs_in_source = sorted({r["document_id"] for r in all_rows})
                total_docs = len(list((root / "literature" / "documents").iterdir()))
                self.assertEqual(docs_in_source, ["docA", "docB"])
                self.assertEqual(total_docs, 3)
                self.assertFalse(len(docs_in_source) == total_docs)  # incomplete-run condition holds


if __name__ == "__main__":
    unittest.main()
