"""Unit tests for thesis_corpus.analyze_coercive_control_groups.

Pure functions with synthetic fixtures. The point of most interest is
interpret(): the pre-registered reading of the experiment is asserted here so
the conclusion stays mechanical, rather than something chosen after seeing
the real numbers.

Run from thesis/corpus/:  python -m unittest thesis_corpus.tests.test_analyze_coercive_control_groups -v
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from thesis_corpus import analyze_coercive_control_groups as accg
from thesis_corpus import analyze_psychological_subjection as aps


class TestLoadCellGroups(unittest.TestCase):
    def test_reads_label_cell_description_and_vector(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "g.jsonl"
            with open(p, "w", encoding="utf-8") as f:
                f.write(json.dumps({"label": "Amway", "cell": "B", "description": "downline quotas",
                                    "embedding_vector": [1.0, 0.0]}) + "\n")
                f.write("\n")
                f.write(json.dumps({"label": "Sierra Club", "cell": "C", "description": "ordinary",
                                    "embedding_vector": [0.0, 1.0]}) + "\n")
            rows, vectors = accg.load_cell_groups(p)
        self.assertEqual([r["cell"] for r in rows], ["B", "C"])
        self.assertEqual(rows[0]["description"], "downline quotas")
        self.assertEqual(vectors.shape, (2, 2))


class TestCellVectors(unittest.TestCase):
    def _rows(self):
        return [
            {"label": "Amway", "cell": "B", "description": "x"},
            {"label": "Herbalife", "cell": "B", "description": "x"},
            {"label": "The National Association of Widget Makers", "cell": "C", "description": "x"},
            {"label": "Sierra Club", "cell": "C", "description": "x"},
        ]

    def test_groups_by_cell(self):
        vectors = np.eye(4)
        grouped = accg.cell_vectors(self._rows(), vectors)
        self.assertEqual(sorted(grouped), ["B", "C"])
        self.assertEqual(len(grouped["B"]), 2)

    def test_name_form_matching_drops_long_and_the_prefixed_names(self):
        # Analysis/08's artifact: a leading "The" is worth +0.061, so any
        # cross-cell comparison on NAMES has to use the matched subset.
        vectors = np.eye(4)
        grouped = accg.cell_vectors(self._rows(), vectors, name_form_matched_only=True)
        self.assertEqual(len(grouped["B"]), 2)
        self.assertEqual(len(grouped["C"]), 1)  # the "The National Association..." row is dropped

    def test_excludes_names_that_are_already_corpus_entities(self):
        vectors = np.eye(4)
        grouped = accg.cell_vectors(self._rows(), vectors, exclude_in_corpus={"Amway"})
        self.assertEqual(len(grouped["B"]), 1)


class TestTagCorpusMembership(unittest.TestCase):
    def _points(self):
        return [
            {"key": "amway", "source_dataset": "emergent_entities"},
            {"key": "church of scientology", "source_dataset": "emergent_entities"},
            {"key": "some expression", "source_dataset": "literature"},
        ]

    def test_matches_on_normalized_key_case_insensitively(self):
        vectors = np.eye(3)
        rows, _ = accg.tag_corpus_membership(["Amway", "Sierra Club"], self._points(), vectors)
        self.assertTrue(rows[0]["in_corpus_entities"])
        self.assertFalse(rows[1]["in_corpus_entities"])

    def test_strips_a_leading_the_before_matching(self):
        # "The Church of Scientology" must match the corpus's
        # "church of scientology" anchor -- otherwise the circularity control
        # silently lets a famous cult through as "novel".
        vectors = np.eye(3)
        rows, _ = accg.tag_corpus_membership(["The Church of Scientology"], self._points(), vectors)
        self.assertTrue(rows[0]["in_corpus_entities"])
        self.assertEqual(rows[0]["matched_form"], "church of scientology")

    def test_only_entity_points_count_not_expressions(self):
        vectors = np.eye(3)
        rows, entity_vectors = accg.tag_corpus_membership(["Some Expression"], self._points(), vectors)
        self.assertFalse(rows[0]["in_corpus_entities"])
        self.assertEqual(len(entity_vectors), 2)  # the literature point is not in the entity pool


class TestResidualReviewCandidates(unittest.TestCase):
    def test_flags_novel_names_that_sit_very_close_to_a_corpus_entity(self):
        rows = [
            {"label": "NXIVM", "in_corpus_entities": False, "max_cos_to_entity_pool": 0.91},
            {"label": "Knitting Circle", "in_corpus_entities": False, "max_cos_to_entity_pool": 0.40},
            {"label": "Amway", "in_corpus_entities": True, "max_cos_to_entity_pool": 1.00},
        ]
        out = accg.residual_review_candidates(rows)
        self.assertEqual([r["label"] for r in out], ["NXIVM"])


class TestInterpret(unittest.TestCase):
    """The pre-registered reading. These assertions exist so that the
    conclusion drawn from the real run is the one committed to in advance."""

    def test_descriptions_only_is_the_expected_outcome(self):
        out = accg.interpret(name_gap=0.01, desc_gap=0.20)
        self.assertIn("legible in conduct language", out)

    def test_both_views_significant(self):
        out = accg.interpret(name_gap=0.20, desc_gap=0.20)
        self.assertIn("names carry more conduct information", out)

    def test_neither_view_significant_leaves_the_retraction_standing(self):
        out = accg.interpret(name_gap=0.01, desc_gap=0.02)
        self.assertIn("retraction stands unqualified", out)

    def test_a_gap_just_inside_the_artifact_band_does_not_count(self):
        out = accg.interpret(name_gap=0.0, desc_gap=aps.ARTIFACT_BAND - 0.001)
        self.assertIn("retraction stands unqualified", out)

    def test_a_gap_just_outside_the_artifact_band_does_count(self):
        out = accg.interpret(name_gap=0.0, desc_gap=aps.ARTIFACT_BAND + 0.001)
        self.assertIn("legible in conduct language", out)

    def test_no_data(self):
        self.assertEqual(accg.interpret(None, None), "NO DATA")


class TestSpecificityRows(unittest.TestCase):
    def test_ranks_criteria_by_b_minus_c_and_flags_the_band(self):
        criteria_points = [{"key": "crit-a"}, {"key": "crit-b"}]
        criteria_vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
        grouped = {
            "B": np.array([[1.0, 0.0]]),   # aligned with crit-a
            "C": np.array([[0.0, 1.0]]),   # aligned with crit-b
        }
        rows = accg.specificity_rows(criteria_points, criteria_vectors, grouped, "names")
        self.assertEqual(rows[0]["criterion_key"], "crit-a")
        self.assertAlmostEqual(rows[0]["B_minus_C"], 1.0, places=3)
        self.assertTrue(rows[0]["clears_artifact_band"])
        self.assertEqual(rows[-1]["criterion_key"], "crit-b")

    def test_returns_empty_when_a_headline_cell_is_missing(self):
        criteria_points = [{"key": "crit-a"}]
        criteria_vectors = np.array([[1.0, 0.0]])
        rows = accg.specificity_rows(criteria_points, criteria_vectors,
                                     {"B": np.array([[1.0, 0.0]])}, "names")
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
