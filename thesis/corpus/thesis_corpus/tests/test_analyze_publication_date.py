"""Unit tests for analyze_publication_date.py's pure functions.

Run from thesis/corpus/:  python -m unittest thesis_corpus.tests.test_analyze_publication_date -v
"""
import unittest

import numpy as np

from thesis_corpus import analyze_publication_date as apd


class TestExtractYear(unittest.TestCase):
    def test_leading_year(self):
        self.assertEqual(apd.extract_year("2016-legal-cases-new-religious-movements"), 2016)

    def test_year_after_author_surname(self):
        self.assertEqual(apd.extract_year("clarke-2004-encyclopedia-of-new-religious-movements"), 2004)

    def test_ignores_trailing_disambiguation_number(self):
        # A real case from the corpus: the real year (2004) appears before
        # a later, unrelated 4-digit disambiguation suffix (2118).
        self.assertEqual(
            apd.extract_year("panchenko-2004-new-religious-movements-and-the-study-of-folklore-the-russian-cas-2118"),
            2004,
        )

    def test_no_year_returns_none(self):
        self.assertIsNone(apd.extract_year("no-year-in-this-slug-at-all"))

    def test_nineteen_hundreds_also_matches(self):
        self.assertEqual(apd.extract_year("smith-1998-some-title"), 1998)


class TestYearsForPoints(unittest.TestCase):
    def test_resolves_and_skips_missing(self):
        points = [
            {"key": "clarke-2004-encyclopedia:0"},
            {"key": "no-year-title:1"},
            {"key": "smith-1998-some-title:2"},
        ]
        indices, years = apd.years_for_points(points)
        self.assertEqual(indices, [0, 2])
        self.assertEqual(years, [2004, 1998])


class TestYearDistanceCorrelation(unittest.TestCase):
    def test_perfect_positive_correlation(self):
        years = [2001, 2005, 2010, 2015, 2020]
        distances = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = apd.year_distance_correlation(years, distances)
        self.assertAlmostEqual(result["spearman_rho"], 1.0, places=6)
        self.assertEqual(result["n"], 5)

    def test_too_few_points_raises(self):
        with self.assertRaises(ValueError):
            apd.year_distance_correlation([2001, 2002], [1.0, 2.0])


class TestAssignTerciles(unittest.TestCase):
    def test_splits_into_three_roughly_equal_groups(self):
        years = list(range(2001, 2025))  # 24 years, evenly spread
        labels, bounds = apd.assign_terciles(years)
        self.assertEqual(len(labels), 24)
        self.assertEqual(set(labels), {"early", "mid", "late"})
        counts = {lbl: labels.count(lbl) for lbl in ("early", "mid", "late")}
        # Roughly equal thirds (8 each for 24 evenly-spread years).
        for count in counts.values():
            self.assertGreater(count, 5)
        self.assertIn("early", bounds)
        self.assertIn("late", bounds)
        self.assertLess(bounds["early"][1], bounds["late"][0])


class TestPeriodCentroids(unittest.TestCase):
    def test_one_centroid_per_period(self):
        vectors = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 10.0], [12.0, 10.0]])
        labels = ["early", "early", "late", "late"]
        centroids = apd.period_centroids(vectors, labels)
        self.assertEqual(set(centroids.keys()), {"early", "late"})
        np.testing.assert_allclose(centroids["early"], [1.0, 0.0])
        np.testing.assert_allclose(centroids["late"], [11.0, 10.0])


class TestPeriodPairwiseRows(unittest.TestCase):
    def test_one_row_per_unordered_pair(self):
        centroids = {
            "early": np.array([0.0, 0.0]),
            "mid": np.array([3.0, 0.0]),
            "late": np.array([0.0, 4.0]),
        }
        bounds = {"early": (2001, 2008), "mid": (2009, 2016), "late": (2017, 2024)}
        rows = apd.period_pairwise_rows(centroids, bounds)
        self.assertEqual(len(rows), 3)
        pairs = {(r["period_a"], r["period_b"]) for r in rows}
        self.assertEqual(pairs, {("early", "late"), ("early", "mid"), ("late", "mid")})
        early_mid = [r for r in rows if {r["period_a"], r["period_b"]} == {"early", "mid"}][0]
        self.assertAlmostEqual(early_mid["euclidean_distance"], 3.0)


class TestPeriodNearestCriterionRows(unittest.TestCase):
    def test_finds_nearest_criterion_per_period(self):
        centroids = {"early": np.array([0.0, 0.0]), "late": np.array([10.0, 0.0])}
        bounds = {"early": (2001, 2010), "late": (2011, 2024)}
        criteria_points = [
            {"key": "crit-a", "label": "Criterion A"},
            {"key": "crit-b", "label": "Criterion B"},
        ]
        criteria_vectors = np.array([[0.5, 0.0], [9.5, 0.0]])
        rows = apd.period_nearest_criterion_rows(centroids, criteria_points, criteria_vectors, bounds)
        by_period = {r["period"]: r for r in rows}
        self.assertEqual(by_period["early"]["nearest_criterion_key"], "crit-a")
        self.assertEqual(by_period["late"]["nearest_criterion_key"], "crit-b")


if __name__ == "__main__":
    unittest.main()
