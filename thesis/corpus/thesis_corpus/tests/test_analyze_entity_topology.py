"""Unit tests for analyze_entity_topology.py's pure computation functions.

Deliberately run against small, synthetic, hand-built vectors -- NOT
against a real shared space -- same convention as
tests/test_analyze_typicality.py.

Run from thesis/corpus/:  python -m unittest thesis_corpus.tests.test_analyze_entity_topology -v
"""
import unittest

import numpy as np

from thesis_corpus import analyze_entity_topology as aet


def _point(key: str, label: str, **overrides) -> dict:
    p = {"key": key, "label": label, "source_dataset": "emergent_entities", "point_role": "emergent"}
    p.update(overrides)
    return p


class TestComputeClusters(unittest.TestCase):
    def test_two_well_separated_dense_groups_form_two_clusters(self):
        # Two tight groups of 4 points each, far apart on the x-axis.
        group_a = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1]])
        group_b = np.array([[100.0, 0.0], [100.1, 0.0], [100.0, 0.1], [100.1, 0.1]])
        vectors = np.vstack([group_a, group_b])
        labels = aet.compute_clusters(vectors, min_cluster_size=3)
        self.assertEqual(len(labels), 8)
        # Every point in group_a shares one label, every point in group_b
        # shares a different one, and neither is noise.
        labels_a = set(labels[:4])
        labels_b = set(labels[4:])
        self.assertEqual(len(labels_a), 1)
        self.assertEqual(len(labels_b), 1)
        self.assertNotIn(-1, labels_a)
        self.assertNotIn(-1, labels_b)
        self.assertNotEqual(labels_a, labels_b)

    def test_scattered_points_are_noise(self):
        rng = np.random.RandomState(0)
        # Points spread far apart relative to their own scale -- no dense
        # neighbourhood for any of them at a min_cluster_size of 3.
        vectors = rng.uniform(0, 1000, size=(6, 2))
        labels = aet.compute_clusters(vectors, min_cluster_size=3)
        self.assertTrue((labels == -1).all())

    def test_too_few_points_raises(self):
        vectors = np.array([[0.0, 0.0], [1.0, 1.0]])
        with self.assertRaises(ValueError):
            aet.compute_clusters(vectors, min_cluster_size=3)


class TestLocalDensityScores(unittest.TestCase):
    def test_tight_cluster_scores_lower_than_isolated_point(self):
        tight = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1]])
        isolated = np.array([[1000.0, 1000.0]])
        vectors = np.vstack([tight, isolated])
        density = aet.local_density_scores(vectors, k=2)
        self.assertEqual(len(density), 5)
        # Every tight-cluster point's k=2 density score should be far
        # smaller than the isolated point's.
        self.assertTrue((density[:4] < density[4]).all())

    def test_too_few_points_raises(self):
        vectors = np.array([[0.0, 0.0], [1.0, 1.0]])
        with self.assertRaises(ValueError):
            aet.local_density_scores(vectors, k=5)


class TestGrandCentroid(unittest.TestCase):
    def test_equal_weight_regardless_of_corpus_size(self):
        # literature has many more points than miviludes/interviews, but
        # the grand centroid must weight each corpus's OWN centroid
        # equally, not each point equally.
        per_source = {
            "literature": {"centroid": np.array([0.0, 0.0])},
            "miviludes": {"centroid": np.array([3.0, 0.0])},
            "interviews": {"centroid": np.array([0.0, 3.0])},
        }
        result = aet.grand_centroid(per_source)
        np.testing.assert_allclose(result, [1.0, 1.0])


class TestEntityRows(unittest.TestCase):
    def test_row_shape_and_values(self):
        points = [_point("k1", "Scientology"), _point("k2", "cults")]
        vectors = np.array([[0.0, 0.0], [10.0, 0.0]])
        labels = np.array([0, -1])
        density = np.array([0.5, 5.0])
        centroid = np.array([0.0, 0.0])
        rows = aet.entity_rows(points, vectors, labels, density, centroid)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["cluster_id"], 0)
        self.assertFalse(rows[0]["is_noise"])
        self.assertEqual(rows[1]["cluster_id"], -1)
        self.assertTrue(rows[1]["is_noise"])
        self.assertAlmostEqual(rows[1]["distance_to_grand_centroid"], 10.0)


class TestClusterSummaryRows(unittest.TestCase):
    def test_excludes_noise_and_sorts_by_centrality(self):
        points = [_point(f"k{i}", f"entity{i}") for i in range(6)]
        # Cluster 0 near the origin (close to grand centroid); cluster 1 far away.
        vectors = np.array([
            [0.0, 0.0], [0.1, 0.0], [0.0, 0.1],
            [50.0, 50.0], [50.1, 50.0],
            [999.0, 999.0],  # noise
        ])
        labels = np.array([0, 0, 0, 1, 1, -1])
        centroid = np.array([0.0, 0.0])
        rows = aet.cluster_summary_rows(points, vectors, labels, centroid)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["cluster_id"], 0)
        self.assertEqual(rows[0]["size"], 3)
        self.assertEqual(rows[1]["cluster_id"], 1)
        self.assertEqual(rows[1]["size"], 2)
        self.assertLess(rows[0]["distance_to_grand_centroid"], rows[1]["distance_to_grand_centroid"])


class TestClusterCentroidNearestRows(unittest.TestCase):
    def test_ranks_clusters_by_size_and_retrieves_k_nearest_candidates(self):
        points = [_point(f"k{i}", f"entity{i}") for i in range(7)]
        # Cluster 0: 4 members near origin. Cluster 1: 2 members far away.
        # 1 noise point.
        vectors = np.array([
            [0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1],
            [50.0, 50.0], [50.1, 50.0],
            [999.0, 999.0],
        ])
        labels = np.array([0, 0, 0, 0, 1, 1, -1])
        candidates = [_point("c1", "candidate_near_origin"), _point("c2", "candidate_far")]
        candidate_vectors = np.array([[0.0, 0.0], [1000.0, 1000.0]])

        rows = aet.cluster_centroid_nearest_rows(
            points, vectors, labels, candidates, candidate_vectors, "expression", top_n_clusters=2, k=2,
        )
        cluster_ids = {r["cluster_id"] for r in rows}
        self.assertEqual(cluster_ids, {0, 1})
        # Cluster 0 (4 members) is bigger than cluster 1 (2 members).
        sizes = {r["cluster_id"]: r["cluster_size"] for r in rows}
        self.assertEqual(sizes[0], 4)
        self.assertEqual(sizes[1], 2)
        # Every row carries the candidate_kind passed in.
        self.assertTrue(all(r["candidate_kind"] == "expression" for r in rows))
        # Cluster 0's nearest candidate (centroid near origin) should be
        # "candidate_near_origin" at rank 1.
        cluster_0_rank_1 = [r for r in rows if r["cluster_id"] == 0 and r["rank"] == 1][0]
        self.assertEqual(cluster_0_rank_1["neighbor_label"], "candidate_near_origin")

    def test_top_n_clusters_caps_output(self):
        points = [_point(f"k{i}", f"entity{i}") for i in range(9)]
        vectors = np.array([[float(i) * 100, 0.0] for i in range(9)])
        # 3 clusters of size 3 each (well-separated on x-axis).
        labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        candidates = [_point("c1", "only_candidate")]
        candidate_vectors = np.array([[0.0, 0.0]])
        rows = aet.cluster_centroid_nearest_rows(
            points, vectors, labels, candidates, candidate_vectors, "entity", top_n_clusters=1, k=1,
        )
        self.assertEqual(len({r["cluster_id"] for r in rows}), 1)


if __name__ == "__main__":
    unittest.main()
