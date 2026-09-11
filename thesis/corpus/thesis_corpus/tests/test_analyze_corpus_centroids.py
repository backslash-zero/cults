"""Unit tests for thesis_corpus.analyze_corpus_centroids.

Run from thesis/corpus/:  python -m unittest thesis_corpus.tests.test_analyze_corpus_centroids -v
"""
import unittest

import numpy as np

from thesis_corpus import analyze_corpus_centroids as acc


def _point(source_dataset, key, label, mention_distribution=None):
    return {"source_dataset": source_dataset, "key": key, "label": label,
            "mention_distribution": mention_distribution}


def _fake_space():
    points = [
        _point("miviludes", f"{acc.MIVILUDES_REPORT_DOCUMENT_ID}:0", "report expr 1"),
        _point("miviludes", f"{acc.MIVILUDES_REPORT_DOCUMENT_ID}:1", "report expr 2"),
        _point("miviludes", "raw-transcript:0", "raw transcript expr -- not in rapport"),
        _point("miviludes_criteria", "crit-a", "Criterion A"),
        _point("miviludes_criteria", "crit-b", "Criterion B"),
        _point("literature", "lit-doc:0", "lit expr 1"),
        _point("literature", "lit-doc:1", "lit expr 2"),
        _point("interviews", "b1:0", "interview expr 1"),
        _point("concept_backbone", "cb-1", "concept 1"),
        _point("structural_concepts", "sc-1", "structural 1"),
        _point("conceptnet_concepts", "cn-1", "conceptnet 1"),
        _point("emergent_entities", "scientology", "Scientology", {"literature": 5, "miviludes": 0, "interviews": 2}),
        _point("emergent_entities", "raelism", "Raelism", {"literature": 0, "miviludes": 3, "interviews": 0}),
        _point("emergent_entities", "unmentioned", "never mentioned", {"literature": 0, "miviludes": 0, "interviews": 0}),
    ]
    vectors = np.arange(len(points) * 4, dtype=np.float64).reshape(len(points), 4)
    return points, vectors


class TestGroupIndices(unittest.TestCase):
    def setUp(self):
        self.points, self.vectors = _fake_space()

    def test_rapport_excludes_raw_transcript_document(self):
        idx = acc.group_indices(self.points, "rapport")
        labels = {self.points[i]["label"] for i in idx}
        self.assertEqual(labels, {"report expr 1", "report expr 2"})

    def test_sectarian_drift_list_is_criteria_source_dataset(self):
        idx = acc.group_indices(self.points, "sectarian_drift_list")
        self.assertEqual(len(idx), 2)
        self.assertTrue(all(self.points[i]["source_dataset"] == "miviludes_criteria" for i in idx))

    def test_dictionary_pools_three_reference_source_datasets(self):
        idx = acc.group_indices(self.points, "dictionary")
        datasets = {self.points[i]["source_dataset"] for i in idx}
        self.assertEqual(datasets, {"concept_backbone", "structural_concepts", "conceptnet_concepts"})

    def test_entities_all_includes_every_emergent_entity(self):
        idx = acc.group_indices(self.points, "entities_all")
        self.assertEqual(len(idx), 3)

    def test_entities_per_corpus_filters_by_nonzero_mention(self):
        lit_idx = acc.group_indices(self.points, "entities_literature")
        self.assertEqual({self.points[i]["label"] for i in lit_idx}, {"Scientology"})
        mivi_idx = acc.group_indices(self.points, "entities_miviludes")
        self.assertEqual({self.points[i]["label"] for i in mivi_idx}, {"Raelism"})
        interview_idx = acc.group_indices(self.points, "entities_interviews")
        self.assertEqual({self.points[i]["label"] for i in interview_idx}, {"Scientology"})

    def test_epistemic_status_group_filters_by_corpus_and_status(self):
        points = [
            {"source_dataset": "literature", "key": "l:0", "label": "asserted claim", "epistemic_status": "asserted"},
            {"source_dataset": "literature", "key": "l:1", "label": "contested claim", "epistemic_status": "contested"},
            {"source_dataset": "interviews", "key": "i:0", "label": "asserted interview", "epistemic_status": "asserted"},
        ]
        idx = acc.group_indices(points, "literature__asserted")
        self.assertEqual({points[i]["label"] for i in idx}, {"asserted claim"})

    def test_straight_source_dataset_fallback(self):
        idx = acc.group_indices(self.points, "literature")
        self.assertEqual(len(idx), 2)
        idx = acc.group_indices(self.points, "interviews")
        self.assertEqual(len(idx), 1)


class TestCriterionNeighbors(unittest.TestCase):
    def test_one_row_per_criterion_per_pool_rank(self):
        points, vectors = _fake_space()
        rows = acc.criterion_neighbors(points, vectors, entity_k=2, literature_k=2, interviews_k=2)
        # 2 criteria x (2 entity rows + 2 literature rows + 1 interview row -- the fixture
        # only has 1 interview point, so nearest_points returns just that one, not 2).
        self.assertEqual(len(rows), 10)
        self.assertEqual({r["criterion_key"] for r in rows}, {"crit-a", "crit-b"})
        self.assertEqual({r["pool"] for r in rows}, {"entities", "literature", "interviews"})
        crit_a_entities = [r for r in rows if r["criterion_key"] == "crit-a" and r["pool"] == "entities"]
        self.assertEqual(len(crit_a_entities), 2)
        self.assertEqual([r["rank"] for r in crit_a_entities], [1, 2])

    def test_entity_pool_never_returns_a_literature_or_interview_point(self):
        points, vectors = _fake_space()
        rows = acc.criterion_neighbors(points, vectors, entity_k=3, literature_k=0, interviews_k=0)
        entity_rows = [r for r in rows if r["pool"] == "entities"]
        entity_keys = {p["key"] for p in points if p["source_dataset"] == "emergent_entities"}
        self.assertTrue(all(r["key"] in entity_keys for r in entity_rows))


class TestDiscoverEpistemicStatusGroups(unittest.TestCase):
    def test_finds_groups_meeting_min_n_only(self):
        points = (
            [{"source_dataset": "literature", "epistemic_status": "asserted"}] * 5
            + [{"source_dataset": "literature", "epistemic_status": "contested"}] * 2  # below MIN
            + [{"source_dataset": "interviews", "epistemic_status": "asserted"}] * 3
            + [{"source_dataset": "concept_backbone", "epistemic_status": None}] * 10  # never a candidate
        )
        groups = acc.discover_epistemic_status_groups(points)
        self.assertEqual(set(groups), {"literature__asserted", "interviews__asserted"})

    def test_null_epistemic_status_excluded(self):
        points = [{"source_dataset": "literature", "epistemic_status": None}] * 5
        self.assertEqual(acc.discover_epistemic_status_groups(points), [])


class TestTruncateLabel(unittest.TestCase):
    def test_short_label_unchanged(self):
        self.assertEqual(acc.truncate_label("Scientology"), "Scientology")

    def test_long_label_truncated_with_ellipsis(self):
        label = "a" * 50
        out = acc.truncate_label(label, max_len=10)
        self.assertEqual(len(out), 10)
        self.assertTrue(out.endswith("…"))

    def test_collapses_internal_whitespace(self):
        self.assertEqual(acc.truncate_label("a   b\n c"), "a b c")


if __name__ == "__main__":
    unittest.main()
