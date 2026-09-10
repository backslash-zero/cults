"""Unit tests for build_shared_space.py's emergent-entity quality filters:
looks_like_named_entity, normalize_anchor, and load_cited_author_surnames.

Run from thesis/corpus/:  python -m unittest thesis_corpus.tests.test_build_shared_space -v
"""
import csv
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


if __name__ == "__main__":
    unittest.main()
