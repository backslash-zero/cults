"""Unit tests for generate_voronoi_projections.py's pure functions:
gap_grid_rows, cluster_centroid_seeds, corpus_centroid_seeds.

Run from thesis/corpus/:  python -m unittest thesis_corpus.tests.test_generate_voronoi_projections -v
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from thesis_corpus import generate_voronoi_projections as gvp


class TestGapGridRows(unittest.TestCase):
    def test_flags_empty_cell_inside_hull_as_gap(self):
        # A ring of points around the origin, with a hole in the middle --
        # the centre cell should be inside the hull but empty (a gap);
        # cells near the ring itself should have points and not be gaps.
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        ring = np.stack([10 * np.cos(angles), 10 * np.sin(angles)], axis=1)
        rows = gvp.gap_grid_rows(ring, bins=5)
        center_cells = [r for r in rows if abs(r["x_center"]) < 2 and abs(r["y_center"]) < 2]
        self.assertTrue(any(r["is_gap"] for r in center_cells), "expected the empty centre to be flagged as a gap")

    def test_cells_outside_hull_never_flagged_as_gaps(self):
        ring = np.array([[0.0, 10.0], [10.0, 0.0], [0.0, -10.0], [-10.0, 0.0]])
        rows = gvp.gap_grid_rows(ring, bins=5)
        # Corner cells (far outside the diamond hull) must be
        # inside_convex_hull=False and therefore never is_gap=True.
        corner_cells = [r for r in rows if r["cell_x"] == 0 and r["cell_y"] == 0]
        for r in corner_cells:
            if not r["inside_convex_hull"]:
                self.assertFalse(r["is_gap"])

    def test_cell_with_points_is_never_a_gap(self):
        cluster = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [10.0, 10.0], [10.0, -10.0], [-10.0, 0.0]])
        rows = gvp.gap_grid_rows(cluster, bins=3)
        occupied = [r for r in rows if r["count"] > 0]
        self.assertTrue(all(not r["is_gap"] for r in occupied))

    def test_too_few_points_raises(self):
        with self.assertRaises(ValueError):
            gvp.gap_grid_rows(np.array([[0.0, 0.0], [1.0, 1.0]]), bins=5)


class TestClusterCentroidSeeds(unittest.TestCase):
    def test_excludes_noise_and_averages_members(self):
        entity_vectors = np.array([
            [0.0, 0.0], [2.0, 0.0],  # cluster 0 -> centroid (1, 0)
            [10.0, 10.0], [10.0, 12.0], [10.0, 11.0],  # cluster 1 -> centroid (10, 11)
            [999.0, 999.0],  # noise, cluster_id -1
        ])
        cluster_rows = [
            {"cluster_id": "0"}, {"cluster_id": "0"},
            {"cluster_id": "1"}, {"cluster_id": "1"}, {"cluster_id": "1"},
            {"cluster_id": "-1"},
        ]
        points, vectors = gvp.cluster_centroid_seeds(entity_vectors, cluster_rows)
        self.assertEqual(len(points), 2)
        np.testing.assert_allclose(vectors[0], [1.0, 0.0])
        np.testing.assert_allclose(vectors[1], [10.0, 11.0])
        self.assertIn("2 entities", points[0]["label"])
        self.assertIn("3 entities", points[1]["label"])


class TestCorpusCentroidSeeds(unittest.TestCase):
    def test_one_seed_per_expression_corpus(self):
        per_source = {
            "literature": {"centroid": np.array([1.0, 0.0])},
            "miviludes": {"centroid": np.array([0.0, 1.0])},
            "interviews": {"centroid": np.array([0.0, 0.0])},
        }
        points, vectors = gvp.corpus_centroid_seeds(per_source)
        self.assertEqual(len(points), 3)
        self.assertEqual({p["source_dataset"] for p in points}, {"literature", "miviludes", "interviews"})


class TestCoordsPathFromManifestPath(unittest.TestCase):
    def test_strips_manifest_json_not_just_json(self):
        manifest_path = Path("/out/umap_voronoi_criteria_seeded_n20_d0.2.manifest.json")
        result = gvp.coords_path_from_manifest_path(manifest_path)
        self.assertEqual(result, Path("/out/umap_voronoi_criteria_seeded_n20_d0.2.jsonl"))


class TestReadMemberCoords2D(unittest.TestCase):
    def test_reads_only_member_rows(self):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
        tmp.write(json.dumps({"point_kind": "member", "umap_2d": [1.0, 2.0]}) + "\n")
        tmp.write(json.dumps({"point_kind": "member", "umap_2d": [3.0, 4.0]}) + "\n")
        tmp.write(json.dumps({"point_kind": "centroid_overlay", "umap_2d": [99.0, 99.0]}) + "\n")
        tmp.close()
        path = Path(tmp.name)
        try:
            coords = gvp._read_member_coords_2d(path)
        finally:
            path.unlink()
        self.assertEqual(coords.shape, (2, 2))
        np.testing.assert_allclose(coords, [[1.0, 2.0], [3.0, 4.0]])


if __name__ == "__main__":
    unittest.main()
