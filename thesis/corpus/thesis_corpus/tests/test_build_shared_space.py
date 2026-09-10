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

    def test_extracts_surname_from_no_comma_first_last_format(self):
        # Regression test: metadata/literature.csv mixes "Surname, First"
        # and plain "First Middle Last" with no comma at all (58 of 68
        # rows use the latter) -- the original comma-only extraction
        # silently returned the WHOLE name for every no-comma row,
        # caught only when a downstream module's independent clustering
        # surfaced uncaught scholar surnames as a real entity cluster.
        path = self._write_metadata([{"id": "x", "authors": "Susan J. Palmer"}])
        try:
            surnames = bss.load_cited_author_surnames(path)
        finally:
            path.unlink()
        self.assertIn("palmer", surnames)
        self.assertNotIn("susan j. palmer", surnames)

    def test_mixed_comma_and_no_comma_authors_in_one_field(self):
        path = self._write_metadata([{"id": "x", "authors": "Barker, Eileen; James T. Richardson"}])
        try:
            surnames = bss.load_cited_author_surnames(path)
        finally:
            path.unlink()
        self.assertEqual(surnames, {"barker", "richardson"})

    def test_hyphenated_surname_kept_as_one_token(self):
        path = self._write_metadata([{"id": "x", "authors": "Danièle Hervieu-Léger"}])
        try:
            surnames = bss.load_cited_author_surnames(path)
        finally:
            path.unlink()
        self.assertIn("hervieu-léger", surnames)

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


class TestLoadEmergentEntitiesEpistemicStatus(unittest.TestCase):
    """Covers epistemic_status_distribution specifically -- the rest of
    load_emergent_entities (thresholds, casing filter, author stop-list)
    already has real-data smoke-test coverage via the toolkit's own
    retained runs; this is the one genuinely new piece of logic added
    without an existing test path exercising it."""

    def _write_archive(self, path: Path, rows: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    def test_entity_anchor_mentions_tallied_by_their_own_expressions_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            lit_path = tmp_path / "literature.jsonl"
            self._write_archive(lit_path, [
                {"epistemic_status": "asserted",
                 "entity_anchor_vectors": {"Scientology": [1.0, 0.0]}},
                {"epistemic_status": "negated",
                 "entity_anchor_vectors": {"Scientology": [1.0, 0.0]}},
                {"epistemic_status": "asserted",
                 "entity_anchor_vectors": {"Scientology": [1.0, 0.0]}},
            ])
            points = bss.load_emergent_entities(
                {"literature": lit_path}, {"literature": 1},
            )
            self.assertEqual(len(points), 1)
            self.assertEqual(
                points[0]["epistemic_status_distribution"],
                {"asserted": 2, "negated": 1},
            )
            self.assertEqual(points[0]["mention_distribution"], {"literature": 3})

    def test_domain_term_mentions_fall_into_unknown_bucket(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            lit_path = tmp_path / "literature.jsonl"
            self._write_archive(lit_path, [])  # no entity_anchors mentions at all

            chunk_terms_path = tmp_path / "chunk_terms.jsonl"
            with open(chunk_terms_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"domain_terms": ["Scientology", "Scientology"]}) + "\n")

            term_vectors_path = tmp_path / "domain_term_vectors.jsonl"
            with open(term_vectors_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"term": "Scientology", "vector": [1.0, 0.0]}) + "\n")

            points = bss.load_emergent_entities(
                {"literature": lit_path}, {"literature": 1},
                domain_term_paths={"literature": (chunk_terms_path, term_vectors_path)},
            )
            self.assertEqual(len(points), 1)
            self.assertEqual(points[0]["epistemic_status_distribution"], {"unknown": 2})

    def test_mixed_entity_anchor_and_domain_term_mentions_combine(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            lit_path = tmp_path / "literature.jsonl"
            self._write_archive(lit_path, [
                {"epistemic_status": "speculative",
                 "entity_anchor_vectors": {"Scientology": [1.0, 0.0]}},
            ])

            chunk_terms_path = tmp_path / "chunk_terms.jsonl"
            with open(chunk_terms_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"domain_terms": ["Scientology"]}) + "\n")
            term_vectors_path = tmp_path / "domain_term_vectors.jsonl"
            with open(term_vectors_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"term": "Scientology", "vector": [1.0, 0.0]}) + "\n")

            points = bss.load_emergent_entities(
                {"literature": lit_path}, {"literature": 1},
                domain_term_paths={"literature": (chunk_terms_path, term_vectors_path)},
            )
            self.assertEqual(len(points), 1)
            self.assertEqual(
                points[0]["epistemic_status_distribution"],
                {"speculative": 1, "unknown": 1},
            )
            self.assertEqual(points[0]["mention_distribution"], {"literature": 2})


class TestSerializePointForOutput(unittest.TestCase):
    def test_emergent_entity_point_keeps_both_distributions(self):
        # Regression test for a real bug: epistemic_status_distribution was
        # added to load_emergent_entities' point dict but forgotten in the
        # output serialization's explicit field allowlist, so it silently
        # never reached embedding_space.jsonl until this was caught by
        # reading the actual rebuilt file.
        import numpy as np
        point = {
            "source_dataset": "emergent_entities", "point_role": "emergent",
            "key": "scientology", "label": "scientology",
            "mention_distribution": {"literature": 3},
            "epistemic_status_distribution": {"asserted": 2, "negated": 1},
        }
        out = bss.serialize_point_for_output(point, np.array([1.0, 2.0]))
        self.assertEqual(out["mention_distribution"], {"literature": 3})
        self.assertEqual(out["epistemic_status_distribution"], {"asserted": 2, "negated": 1})
        self.assertEqual(out["shared_space_vector"], [1.0, 2.0])

    def test_expression_point_missing_fields_default_to_none(self):
        import numpy as np
        point = {"source_dataset": "literature", "point_role": "expression", "key": "doc:0", "label": "text"}
        out = bss.serialize_point_for_output(point, np.array([0.0]))
        self.assertIsNone(out["mention_distribution"])
        self.assertIsNone(out["epistemic_status_distribution"])
        self.assertIsNone(out["response_rank"])


if __name__ == "__main__":
    unittest.main()
