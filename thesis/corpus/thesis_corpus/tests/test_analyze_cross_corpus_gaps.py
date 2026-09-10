"""Unit tests for analyze_cross_corpus_gaps.py's pure functions.

Deliberately run against small, synthetic, hand-built vectors -- not a
real shared space, same convention as tests/test_analyze_entity_topology.py.

Run from thesis/corpus/:  python -m unittest thesis_corpus.tests.test_analyze_cross_corpus_gaps -v
"""
import unittest

import numpy as np

from thesis_corpus import analyze_cross_corpus_gaps as accg


def _point(key: str, label: str, **overrides) -> dict:
    p = {"key": key, "label": label, "source_dataset": "emergent_entities", "point_role": "emergent"}
    p.update(overrides)
    return p


class TestCorpusRestrictedEntities(unittest.TestCase):
    def test_only_entities_with_nonzero_mentions_for_corpus_kept(self):
        points = [
            _point("k1", "e1", mention_distribution={"literature": 3, "interviews": 0}),
            _point("k2", "e2", mention_distribution={"literature": 0, "interviews": 2}),
            _point("k3", "e3", mention_distribution={"literature": 1, "interviews": 1}),
        ]
        vectors = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        lit_points, lit_vectors = accg.corpus_restricted_entities(points, vectors, "literature")
        self.assertEqual([p["key"] for p in lit_points], ["k1", "k3"])
        np.testing.assert_allclose(lit_vectors, [[0.0, 0.0], [2.0, 0.0]])

        int_points, _ = accg.corpus_restricted_entities(points, vectors, "interviews")
        self.assertEqual([p["key"] for p in int_points], ["k2", "k3"])

    def test_missing_mention_distribution_excludes_entity(self):
        points = [_point("k1", "e1")]
        vectors = np.array([[0.0, 0.0]])
        result_points, result_vectors = accg.corpus_restricted_entities(points, vectors, "literature")
        self.assertEqual(result_points, [])
        self.assertEqual(len(result_vectors), 0)


class TestClusterProximityRows(unittest.TestCase):
    def test_reports_nearest_corpus_b_entities_per_corpus_a_cluster(self):
        a_points = [_point(f"a{i}", f"lit_entity{i}") for i in range(4)]
        a_vectors = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [50.0, 50.0]])
        a_labels = np.array([0, 0, 0, -1])  # cluster 0 near origin, 1 noise point far away

        b_points = [_point("b1", "near_b_entity"), _point("b2", "far_b_entity")]
        b_vectors = np.array([[1.0, 0.0], [1000.0, 1000.0]])

        rows = accg.cluster_proximity_rows("literature", a_points, a_vectors, a_labels, "interviews", b_points, b_vectors, k=2)
        # Only cluster 0 (noise excluded), 2 ranks (k=2).
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r["corpus_a_cluster_id"] == 0 for r in rows))
        self.assertTrue(all(r["corpus_a"] == "literature" and r["corpus_b"] == "interviews" for r in rows))
        self.assertEqual(rows[0]["rank"], 1)
        self.assertEqual(rows[0]["nearest_corpus_b_entity"], "near_b_entity")
        self.assertEqual(rows[0]["corpus_a_cluster_size"], 3)

    def test_no_clusters_produces_no_rows(self):
        a_points = [_point("a1", "e1")]
        a_vectors = np.array([[0.0, 0.0]])
        a_labels = np.array([-1])  # all noise
        b_points = [_point("b1", "e2")]
        b_vectors = np.array([[1.0, 0.0]])
        rows = accg.cluster_proximity_rows("literature", a_points, a_vectors, a_labels, "interviews", b_points, b_vectors, k=1)
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
