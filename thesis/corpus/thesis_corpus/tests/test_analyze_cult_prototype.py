"""Unit tests for thesis_corpus.analyze_cult_prototype.

Run from thesis/corpus/:  python -m unittest thesis_corpus.tests.test_analyze_cult_prototype -v
"""
import unittest

import numpy as np

from thesis_corpus import analyze_cult_prototype as acp


def _unit(vec):
    vec = np.array(vec, dtype=np.float64)
    return vec / np.linalg.norm(vec)


class TestCultPrototypePoolIndices(unittest.TestCase):
    def test_literature_and_miviludes_always_included(self):
        points = [
            {"source_dataset": "literature", "cult_relevant": True},
            {"source_dataset": "miviludes", "cult_relevant": True},
        ]
        self.assertEqual(acp.cult_prototype_pool_indices(points), [0, 1])

    def test_interviews_filtered_to_cult_relevant_true_only(self):
        points = [
            {"source_dataset": "interviews", "cult_relevant": True},
            {"source_dataset": "interviews", "cult_relevant": False},
            {"source_dataset": "interviews", "cult_relevant": None},
        ]
        self.assertEqual(acp.cult_prototype_pool_indices(points), [0])

    def test_reference_and_entity_points_excluded(self):
        points = [
            {"source_dataset": "concept_backbone", "cult_relevant": None},
            {"source_dataset": "emergent_entities", "cult_relevant": None},
            {"source_dataset": "miviludes_criteria", "cult_relevant": None},
        ]
        self.assertEqual(acp.cult_prototype_pool_indices(points), [])

    def test_mixed_pool(self):
        points = [
            {"source_dataset": "literature", "cult_relevant": True},
            {"source_dataset": "interviews", "cult_relevant": True},
            {"source_dataset": "interviews", "cult_relevant": False},
            {"source_dataset": "emergent_entities", "cult_relevant": None},
        ]
        self.assertEqual(acp.cult_prototype_pool_indices(points), [0, 1])


class TestEqualWeightedPrototypeCentroid(unittest.TestCase):
    def test_each_corpus_weighted_equally_regardless_of_point_count(self):
        # literature has 100 points all identical to [1,0]; miviludes has 1
        # point at [0,1]; interviews (cult_relevant) has 1 point at [0,1].
        # A naive pool would be ~98% literature-dominated ([1,0]); the
        # equal-weighted centroid should instead be equidistant between
        # literature's [1,0] and the other two corpora's [0,1].
        points = [{"source_dataset": "literature", "key": f"l:{i}", "label": "lit"} for i in range(100)]
        points.append({"source_dataset": "miviludes", "key": "m:0", "label": "mivi"})
        points.append({"source_dataset": "interviews", "key": "i:0", "label": "int", "cult_relevant": True})
        points.append({"source_dataset": "interviews", "key": "i:1", "label": "int-filler", "cult_relevant": False})
        vectors = np.array([[1.0, 0.0]] * 100 + [[0.0, 1.0], [0.0, 1.0], [5.0, 5.0]])
        centroid = acp.equal_weighted_prototype_centroid(points, vectors)
        # literature sub-centroid=[1,0], miviludes=[0,1], interviews(cult_relevant only)=[0,1]
        # -> mean = [1/3, 2/3], NOT dominated by literature's 100 points
        np.testing.assert_allclose(centroid, [1 / 3, 2 / 3], atol=1e-9)


class TestRankCriteriaByPrototypeCentrality(unittest.TestCase):
    def test_full_ranking_of_all_criteria_no_k_cutoff(self):
        points = [
            {"source_dataset": "miviludes_criteria", "key": "crit-a", "label": "Criterion A"},
            {"source_dataset": "miviludes_criteria", "key": "crit-b", "label": "Criterion B"},
            {"source_dataset": "miviludes_criteria", "key": "crit-c", "label": "Criterion C"},
            {"source_dataset": "literature", "key": "lit:0", "label": "not a criterion"},
        ]
        vectors = np.array([_unit([1, 0]), _unit([0, 1]), _unit([0.9, 0.1]), _unit([1, 0.01])])
        prototype_centroid = _unit([1, 0]) * 0.5  # closest to crit-a and crit-c, farthest from crit-b
        ranking = acp.rank_criteria_by_prototype_centrality(points, vectors, prototype_centroid)
        self.assertEqual(len(ranking), 3)  # never includes the literature point
        self.assertEqual(ranking[0]["key"], "crit-a")
        self.assertEqual(ranking[-1]["key"], "crit-b")


if __name__ == "__main__":
    unittest.main()
