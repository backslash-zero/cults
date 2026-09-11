"""Unit tests for thesis_corpus.analyze_literature_clusters.

Run from thesis/corpus/:  python -m unittest thesis_corpus.tests.test_analyze_literature_clusters -v
"""
import unittest

import numpy as np

from thesis_corpus import analyze_literature_clusters as alc


def _unit(vec):
    vec = np.array(vec, dtype=np.float64)
    return vec / np.linalg.norm(vec)


def _fixture():
    # Two obvious clusters in 3-D, plus one noise-labelled point (-1).
    points = [
        {"key": "a", "label": "expr A1"}, {"key": "b", "label": "expr A2"}, {"key": "c", "label": "expr A3"},
        {"key": "d", "label": "expr B1"}, {"key": "e", "label": "expr B2"}, {"key": "f", "label": "expr B3"},
        {"key": "n", "label": "noise expr"},
    ]
    vectors = np.array([
        _unit([1, 0, 0]), _unit([0.95, 0.1, 0]), _unit([0.9, 0.15, 0]),
        _unit([0, 1, 0]), _unit([0.1, 0.95, 0]), _unit([0.15, 0.9, 0]),
        _unit([0, 0, 1]),
    ])
    labels = np.array([0, 0, 0, 1, 1, 1, -1])
    entity_points = [
        {"key": "ent-a", "label": "Entity near A"},
        {"key": "ent-b", "label": "Entity near B"},
    ]
    entity_vectors = np.array([_unit([1, 0.05, 0]), _unit([0.05, 1, 0])])
    return points, vectors, labels, entity_points, entity_vectors


class TestClusterBindingExpressions(unittest.TestCase):
    def test_returns_k_members_from_own_cluster_only(self):
        points, vectors, labels, *_ = _fixture()
        result = alc.cluster_binding_expressions(points, vectors, labels, k=2)
        self.assertEqual(set(result.keys()), {0, 1})
        cluster0_keys = {n["key"] for n in result[0]}
        self.assertTrue(cluster0_keys.issubset({"a", "b", "c"}))
        self.assertEqual(len(result[0]), 2)

    def test_noise_never_gets_a_cluster(self):
        points, vectors, labels, *_ = _fixture()
        result = alc.cluster_binding_expressions(points, vectors, labels, k=2)
        self.assertNotIn(-1, result)


class TestClusterNearestEntities(unittest.TestCase):
    def test_each_cluster_gets_its_own_nearest_entity(self):
        points, vectors, labels, entity_points, entity_vectors = _fixture()
        result = alc.cluster_nearest_entities(vectors, labels, entity_points, entity_vectors, k=1)
        self.assertEqual(result[0][0]["key"], "ent-a")
        self.assertEqual(result[1][0]["key"], "ent-b")


class TestEntityNearestCluster(unittest.TestCase):
    def test_each_entity_assigned_to_its_own_nearest_cluster(self):
        points, vectors, labels, entity_points, entity_vectors = _fixture()
        rows = alc.entity_nearest_cluster(vectors, labels, entity_points, entity_vectors)
        by_key = {r["entity_key"]: r for r in rows}
        self.assertEqual(by_key["ent-a"]["nearest_cluster_id"], "0")
        self.assertEqual(by_key["ent-b"]["nearest_cluster_id"], "1")

    def test_never_assigns_the_noise_label(self):
        points, vectors, labels, entity_points, entity_vectors = _fixture()
        rows = alc.entity_nearest_cluster(vectors, labels, entity_points, entity_vectors)
        self.assertTrue(all(r["nearest_cluster_id"] != "-1" for r in rows))


if __name__ == "__main__":
    unittest.main()
