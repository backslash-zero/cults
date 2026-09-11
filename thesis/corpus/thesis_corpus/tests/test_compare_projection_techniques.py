"""Unit tests for thesis_corpus.compare_projection_techniques.

Run from thesis/corpus/:  python -m unittest thesis_corpus.tests.test_compare_projection_techniques -v
"""
import unittest

import numpy as np

from thesis_corpus import compare_projection_techniques as cpt


class TestMeanPairwiseCosine(unittest.TestCase):
    def test_identical_vectors_give_similarity_one(self):
        vectors = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        self.assertAlmostEqual(cpt.mean_pairwise_cosine(vectors), 1.0, places=6)

    def test_orthogonal_vectors_give_similarity_zero(self):
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.assertAlmostEqual(cpt.mean_pairwise_cosine(vectors), 0.0, places=6)

    def test_opposite_vectors_give_similarity_minus_one(self):
        vectors = np.array([[1.0, 0.0], [-1.0, 0.0]])
        self.assertAlmostEqual(cpt.mean_pairwise_cosine(vectors), -1.0, places=6)

    def test_ignores_vector_magnitude(self):
        vectors = np.array([[1.0, 0.0], [5.0, 0.0]])
        self.assertAlmostEqual(cpt.mean_pairwise_cosine(vectors), 1.0, places=6)


class TestProjectAll(unittest.TestCase):
    def test_every_technique_returns_one_row_per_point_plus_centroid(self):
        rng = np.random.default_rng(0)
        local_vectors = rng.normal(size=(12, 8))
        local_vectors /= np.linalg.norm(local_vectors, axis=1, keepdims=True)
        centroid = local_vectors.mean(axis=0)
        projections = cpt.project_all(local_vectors, centroid)
        self.assertEqual(set(projections.keys()), {"pca_local", "mds", "umap_default", "umap_tight", "tsne"})
        for technique, coords in projections.items():
            self.assertEqual(coords.shape, (13, 2), technique)


if __name__ == "__main__":
    unittest.main()
