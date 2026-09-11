"""Unit tests for thesis_corpus.find_secular_group_mentions.

Run from thesis/corpus/:  python -m unittest thesis_corpus.tests.test_find_secular_group_mentions -v
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from thesis_corpus import find_secular_group_mentions as fsg


def _unit(vec):
    vec = np.array(vec, dtype=np.float64)
    return vec / np.linalg.norm(vec)


class TestFindEntitiesByKey(unittest.TestCase):
    def test_matches_case_insensitively(self):
        points = [
            {"source_dataset": "emergent_entities", "key": "Amway", "label": "Amway"},
            {"source_dataset": "emergent_entities", "key": "nazism", "label": "Nazism"},
            {"source_dataset": "literature", "key": "amway", "label": "not an entity point"},
        ]
        idx = fsg.find_entities_by_key(points, {"amway"})
        self.assertEqual(idx, [0])  # only the emergent_entities point, not the literature one

    def test_only_entities_never_other_source_datasets(self):
        points = [
            {"source_dataset": "literature", "key": "amway", "label": "amway mentioned in prose"},
        ]
        self.assertEqual(fsg.find_entities_by_key(points, {"amway"}), [])

    def test_no_match_returns_empty(self):
        points = [{"source_dataset": "emergent_entities", "key": "heaven's gate", "label": "Heaven's Gate"}]
        self.assertEqual(fsg.find_entities_by_key(points, {"amway"}), [])


class TestDistancesToCentroids(unittest.TestCase):
    def test_reports_one_row_per_index_per_centroid_pair(self):
        points = [
            {"key": "amway", "label": "Amway"},
            {"key": "nazism", "label": "Nazism"},
        ]
        vectors = np.array([_unit([1, 0]), _unit([0, 1])])
        centroids = {"criteria_list": _unit([1, 0]) * 0.5, "prototype": _unit([0, 1]) * 0.5}
        rows = fsg.distances_to_centroids(points, vectors, [0, 1], centroids)
        self.assertEqual(len(rows), 2)
        self.assertIn("criteria_list_cosine_similarity", rows[0])
        self.assertIn("prototype_cosine_similarity", rows[0])
        # amway ([1,0]) should be much closer to the criteria_list centroid ([1,0]-direction)
        self.assertGreater(rows[0]["criteria_list_cosine_similarity"], rows[0]["prototype_cosine_similarity"])


class TestLoadGeneratedGroupDistances(unittest.TestCase):
    def test_missing_file_returns_empty_list_not_an_error(self):
        centroids = {"criteria_list": _unit([1, 0])}
        self.assertEqual(fsg.load_generated_group_distances(Path("/nonexistent/path.jsonl"), centroids), [])

    def test_present_file_computes_distances_and_tags_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "generated.jsonl"
            with open(path, "w") as f:
                f.write(json.dumps({"label": "CrossFit", "embedding_vector": [1.0, 0.0]}) + "\n")
            centroids = {"criteria_list": _unit([1, 0]) * 0.5}
            rows = fsg.load_generated_group_distances(path, centroids)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["group"], "generated_secular")
            self.assertIn("criteria_list_cosine_similarity", rows[0])


if __name__ == "__main__":
    unittest.main()
