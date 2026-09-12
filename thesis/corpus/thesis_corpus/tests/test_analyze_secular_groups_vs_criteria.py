"""Unit tests for thesis_corpus.analyze_secular_groups_vs_criteria.

Pure functions only (name-form matching/reporting, per-criterion
comparison, nearest-criterion assignment) with small synthetic fixtures --
same style as tests/test_analyze_literature_clusters.py.

Run from thesis/corpus/:  python -m unittest thesis_corpus.tests.test_analyze_secular_groups_vs_criteria -v
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from thesis_corpus import analyze_secular_groups_vs_criteria as asgc


class TestLoadGeneratedGroups(unittest.TestCase):
    def test_reads_labels_and_vectors_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "g.jsonl"
            with open(p, "w", encoding="utf-8") as f:
                f.write(json.dumps({"label": "Greenpeace", "embedding_vector": [1.0, 0.0]}) + "\n")
                f.write("\n")  # blank line must be skipped, not crash
                f.write(json.dumps({"label": "CrossFit", "embedding_vector": [0.0, 1.0]}) + "\n")
            labels, vectors = asgc.load_generated_groups(p)
        self.assertEqual(labels, ["Greenpeace", "CrossFit"])
        self.assertEqual(vectors.shape, (2, 2))


class TestMatchBaselineNameForm(unittest.TestCase):
    def test_keeps_short_names_without_a_leading_the(self):
        labels = ["Greenpeace", "Sierra Club", "The Nature Conservancy",
                  "The American Association for the Advancement of Science",
                  "National Football League", "Union of Concerned Scientists"]
        idx = asgc.match_baseline_name_form(labels)
        self.assertEqual([labels[i] for i in idx],
                         ["Greenpeace", "Sierra Club", "National Football League"])

    def test_the_check_is_case_insensitive_and_requires_a_word_boundary(self):
        # "Theater Guilds" starts with the letters "The" but is not "The ..."
        labels = ["Theater Guilds", "the Skeptics Society"]
        idx = asgc.match_baseline_name_form(labels)
        self.assertEqual([labels[i] for i in idx], ["Theater Guilds"])


class TestNameFormReport(unittest.TestCase):
    def test_pairs_up_the_x_with_bare_x_and_reports_the_delta(self):
        labels = ["American Civil Liberties Union", "The American Civil Liberties Union", "Greenpeace"]
        sims = np.array([0.40, 0.50, 0.45])
        rows = asgc.name_form_report(labels, sims)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["bare_name"], "American Civil Liberties Union")
        self.assertAlmostEqual(rows[0]["delta"], 0.10, places=4)

    def test_returns_empty_when_no_pairs_exist(self):
        self.assertEqual(asgc.name_form_report(["Greenpeace", "CrossFit"], np.array([0.1, 0.2])), [])


class TestPerCriterionComparison(unittest.TestCase):
    def test_reports_mean_cosine_per_group_per_criterion(self):
        criteria_points = [{"key": "crit-a", "label": "A"}, {"key": "crit-b", "label": "B"}]
        criteria_vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
        group_vectors = {
            "aligned_with_a": np.array([[1.0, 0.0], [1.0, 0.0]]),
            "aligned_with_b": np.array([[0.0, 1.0]]),
        }
        rows = asgc.per_criterion_comparison(criteria_points, criteria_vectors, group_vectors)
        by_key = {r["criterion_key"]: r for r in rows}
        self.assertAlmostEqual(by_key["crit-a"]["aligned_with_a_mean_cos"], 1.0, places=3)
        self.assertAlmostEqual(by_key["crit-a"]["aligned_with_b_mean_cos"], 0.0, places=3)
        self.assertAlmostEqual(by_key["crit-b"]["aligned_with_b_mean_cos"], 1.0, places=3)

    def test_skips_empty_groups_rather_than_reporting_zero(self):
        criteria_points = [{"key": "crit-a", "label": "A"}]
        criteria_vectors = np.array([[1.0, 0.0]])
        rows = asgc.per_criterion_comparison(
            criteria_points, criteria_vectors,
            {"present": np.array([[1.0, 0.0]]), "absent": np.empty((0, 2))},
        )
        self.assertIn("present_mean_cos", rows[0])
        self.assertNotIn("absent_mean_cos", rows[0])


class TestNearestCriterionPerGroup(unittest.TestCase):
    def test_assigns_each_group_its_closest_criterion(self):
        criteria_points = [{"key": "crit-a", "label": "A"}, {"key": "crit-b", "label": "B"}]
        criteria_vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
        labels = ["Leans A", "The Leaning B Society"]
        vectors = np.array([[1.0, 0.1], [0.1, 1.0]])
        rows = asgc.nearest_criterion_per_group(labels, vectors, criteria_points, criteria_vectors)
        self.assertEqual(rows[0]["nearest_criterion_key"], "crit-a")
        self.assertEqual(rows[1]["nearest_criterion_key"], "crit-b")
        # the name-form flag travels with the row so artifact-prone rows stay visible
        self.assertTrue(rows[0]["name_form_matched"])
        self.assertFalse(rows[1]["name_form_matched"])


if __name__ == "__main__":
    unittest.main()
