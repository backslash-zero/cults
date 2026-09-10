"""Unit tests for build_shared_space.py's emergent-entity quality filters:
looks_like_named_entity, normalize_anchor, load_cited_author_surnames, and
the manual-exclusion mechanism in load_corpus_points_v2.

Run from thesis/corpus/:  python -m unittest thesis_corpus.tests.test_build_shared_space -v
"""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from thesis_corpus import build_shared_space as bss


class TestLooksLikeNamedEntity(unittest.TestCase):
    def test_capitalized_terms_pass(self):
        for term in ["Scientology", "Heaven's Gate", "ISKCON"]:
            self.assertTrue(bss.looks_like_named_entity(term), term)

    def test_all_lowercase_terms_fail(self):
        for term in ["cults", "brainwashing", "charismatic leader"]:
            self.assertFalse(bss.looks_like_named_entity(term), term)

    def test_pure_numbers_fail(self):
        # str.islower() is False for a string with no cased characters at
        # all -- bare years/page numbers must not pass as if capitalized.
        for term in ["2004", "11", "1958"]:
            self.assertFalse(bss.looks_like_named_entity(term), term)


class TestNormalizeAnchor(unittest.TestCase):
    def test_merges_casing_and_whitespace_variants(self):
        self.assertEqual(bss.normalize_anchor("Charismatic  Leader"), bss.normalize_anchor("charismatic leader"))

    def test_strips_surrounding_whitespace(self):
        self.assertEqual(bss.normalize_anchor("  Scientology  "), "scientology")


class TestLoadCitedAuthorSurnames(unittest.TestCase):
    def _write_metadata(self, rows: list[dict]) -> Path:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8", newline="")
        writer = csv.DictWriter(tmp, fieldnames=["id", "authors"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        tmp.close()
        return Path(tmp.name)

    def test_extracts_surname_from_single_author(self):
        path = self._write_metadata([{"id": "x", "authors": "Richardson, James T."}])
        try:
            surnames = bss.load_cited_author_surnames(path)
        finally:
            path.unlink()
        self.assertIn("richardson", surnames)

    def test_splits_multiple_authors(self):
        path = self._write_metadata([{"id": "x", "authors": "Barker, Eileen; Beckford, James A."}])
        try:
            surnames = bss.load_cited_author_surnames(path)
        finally:
            path.unlink()
        self.assertEqual(surnames, {"barker", "beckford"})

    def test_strips_editor_suffix(self):
        path = self._write_metadata([{"id": "x", "authors": "Dawson, Lorne L. (ed.)"}])
        try:
            surnames = bss.load_cited_author_surnames(path)
        finally:
            path.unlink()
        self.assertIn("dawson", surnames)

    def test_missing_file_returns_empty_set(self):
        self.assertEqual(bss.load_cited_author_surnames(Path("/nonexistent/literature.csv")), set())

    def test_blank_authors_field_skipped(self):
        path = self._write_metadata([{"id": "x", "authors": ""}])
        try:
            surnames = bss.load_cited_author_surnames(path)
        finally:
            path.unlink()
        self.assertEqual(surnames, set())


class TestManualExclusion(unittest.TestCase):
    def _write_archive(self, rows: list[dict]) -> Path:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
        for row in rows:
            tmp.write(json.dumps(row) + "\n")
        tmp.close()
        return Path(tmp.name)

    def test_manually_excluded_key_dropped_and_counted(self):
        # "b3-aug23-1213:0" is the real, permanently-excluded "Sept?" case
        # (see MANUALLY_EXCLUDED_POOLED_KEYS's own docstring) -- this test
        # exercises the general mechanism using that real key, rather than
        # inventing a synthetic one, so it doubles as a regression check
        # that this specific exclusion stays wired up.
        self.assertIn("b3-aug23-1213:0", bss.MANUALLY_EXCLUDED_POOLED_KEYS)
        path = self._write_archive([
            {"document_id": "b3-aug23-1213", "chunk_index": 0, "embedding_text": "Sept?",
             "embedding_vector": [0.1, 0.2], "attribution": "participant",
             "claim_mode": "question_or_reflection", "epistemic_status": "speculative"},
            {"document_id": "b3-aug23-1213", "chunk_index": 1, "embedding_text": "Charles Manson",
             "embedding_vector": [0.3, 0.4], "attribution": "participant",
             "claim_mode": "direct_statement", "epistemic_status": "asserted"},
        ])
        try:
            points, removal_counts = bss.load_corpus_points_v2("interviews", path, min_expression_words=0)
        finally:
            path.unlink()
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0]["key"], "b3-aug23-1213:1")
        self.assertEqual(removal_counts["manually_excluded"], 1)

    def test_no_exclusions_when_key_absent(self):
        path = self._write_archive([
            {"document_id": "other-interview", "chunk_index": 0, "embedding_text": "Scientology",
             "embedding_vector": [0.1, 0.2], "attribution": "participant",
             "claim_mode": "direct_statement", "epistemic_status": "asserted"},
        ])
        try:
            points, removal_counts = bss.load_corpus_points_v2("interviews", path, min_expression_words=0)
        finally:
            path.unlink()
        self.assertEqual(len(points), 1)
        self.assertEqual(removal_counts["manually_excluded"], 0)


if __name__ == "__main__":
    unittest.main()
