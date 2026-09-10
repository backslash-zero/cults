"""Unit tests for embed_domain_terms.py's pure logic: the named-entity
filter heuristic, candidate-term collection from chunk_terms.jsonl, and
resumability (already-embedded terms are never re-embedded).

No Ollama host needed -- embed_texts itself is never called here.

Run from thesis/corpus/:  python -m unittest thesis_corpus.tests.test_embed_domain_terms -v
"""
import json
import tempfile
import unittest
from pathlib import Path

from thesis_corpus import embed_domain_terms as edt


class TestLooksLikeNamedEntity(unittest.TestCase):
    def test_capitalized_terms_pass(self):
        for term in ["Scientology", "Heaven's Gate", "Jehovah's Witnesses", "ISKCON", "NRMs", "UNADFI"]:
            self.assertTrue(edt.looks_like_named_entity(term), term)

    def test_all_lowercase_terms_fail(self):
        for term in ["cults", "brainwashing", "mind control", "secularization", "sect"]:
            self.assertFalse(edt.looks_like_named_entity(term), term)

    def test_mixed_case_multiword_passes(self):
        self.assertTrue(edt.looks_like_named_entity("Church of Scientology"))

    def test_leading_lowercase_but_not_all_lowercase_passes(self):
        # e.g. a term like "eBay-style recruitment" -- has an uppercase
        # letter somewhere, so str.islower() is False even though it
        # doesn't start capitalized.
        self.assertTrue(edt.looks_like_named_entity("eBay-style recruitment"))


class TestCollectCandidateTerms(unittest.TestCase):
    def _write_chunk_terms(self, rows: list[dict]) -> Path:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
        for row in rows:
            tmp.write(json.dumps(row) + "\n")
        tmp.close()
        return Path(tmp.name)

    def test_filters_to_named_entity_shaped_terms_and_counts_mentions(self):
        path = self._write_chunk_terms([
            {"document_id": "d1", "chunk_index": 0, "domain_terms": ["Scientology", "cults", "Scientology"]},
            {"document_id": "d1", "chunk_index": 1, "domain_terms": ["brainwashing", "Heaven's Gate"]},
            {"document_id": "d2", "chunk_index": 0, "domain_terms": []},
        ])
        try:
            counts = edt.collect_candidate_terms(path)
        finally:
            path.unlink()
        self.assertEqual(counts["Scientology"], 2)
        self.assertEqual(counts["Heaven's Gate"], 1)
        self.assertNotIn("cults", counts)
        self.assertNotIn("brainwashing", counts)

    def test_empty_domain_terms_produces_no_candidates(self):
        path = self._write_chunk_terms([{"document_id": "d1", "chunk_index": 0, "domain_terms": []}])
        try:
            counts = edt.collect_candidate_terms(path)
        finally:
            path.unlink()
        self.assertEqual(len(counts), 0)


class TestLoadDoneTerms(unittest.TestCase):
    def test_missing_output_file_is_empty(self):
        self.assertEqual(edt.load_done_terms(Path("/nonexistent/path/domain_term_vectors.jsonl")), set())

    def test_reads_previously_embedded_terms(self):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
        tmp.write(json.dumps({"term": "Scientology", "vector": [0.1, 0.2]}) + "\n")
        tmp.write(json.dumps({"term": "ISKCON", "vector": [0.3, 0.4]}) + "\n")
        tmp.close()
        path = Path(tmp.name)
        try:
            done = edt.load_done_terms(path)
        finally:
            path.unlink()
        self.assertEqual(done, {"Scientology", "ISKCON"})


if __name__ == "__main__":
    unittest.main()
