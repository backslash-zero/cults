"""Unit tests for thesis_corpus.analyze_psychological_subjection.

Pure functions with synthetic fixtures, plus one REGRESSION TEST against the
real on-disk data for the correction this module exists to make (the
criterion-text-length confound). That one is deliberately coupled to real
data: the whole point of the module is that a previously-published claim was
a length artifact, so the assertion has to hold against the actual corpus,
not a fixture that could be tuned to agree.

Run from thesis/corpus/:  python -m unittest thesis_corpus.tests.test_analyze_psychological_subjection -v
"""
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from thesis_corpus import analyze_psychological_subjection as aps


class TestCriterionLengthControl(unittest.TestCase):
    def _rows(self):
        # absolute score falls as word count rises (the confound);
        # diff is flat with respect to length (so it is not confounded).
        return [
            {"criterion_key": "a", "generated_name_matched_mean_cos": 0.50, "matched_minus_religious": 0.01},
            {"criterion_key": "b", "generated_name_matched_mean_cos": 0.40, "matched_minus_religious": 0.05},
            {"criterion_key": "c", "generated_name_matched_mean_cos": 0.30, "matched_minus_religious": 0.02},
            {"criterion_key": "d", "generated_name_matched_mean_cos": 0.20, "matched_minus_religious": 0.04},
        ], {"a": 3, "b": 10, "c": 20, "d": 30}

    def test_detects_a_pure_length_confound_on_the_absolute_measure(self):
        rows, wc = self._rows()
        out, stats = aps.criterion_length_control(rows, wc, focus_key="d")
        self.assertEqual(stats["spearman_length_vs_absolute"], -1.0)
        self.assertEqual(stats["focus_fr_word_count"], 30)
        self.assertEqual([r["criterion_fr_word_count"] for r in out], [3, 10, 20, 30])

    def test_the_diff_measure_is_reported_separately_from_the_absolute_one(self):
        rows, wc = self._rows()
        _, stats = aps.criterion_length_control(rows, wc, focus_key="d")
        # "d" is last on absolute score but mid-pack on the diff measure --
        # exactly the discrepancy that forced the retraction.
        self.assertEqual(stats["focus_rank_absolute_of_17"], 1)
        self.assertEqual(stats["focus_rank_diff_of_17"], 3)
        self.assertLess(abs(stats["spearman_length_vs_diff"]), 0.5)

    def test_adds_a_length_residualized_diff_column(self):
        rows, wc = self._rows()
        out, _ = aps.criterion_length_control(rows, wc, focus_key="d")
        self.assertTrue(all("diff_residual_on_length" in r for r in out))
        self.assertAlmostEqual(sum(r["diff_residual_on_length"] for r in out), 0.0, places=3)


class TestCriterionLengthControlAgainstRealData(unittest.TestCase):
    """The regression guard for the retraction itself. If these assertions
    ever fail, either the corpus changed or the correction was undone --
    both need a human look, not a silently-updated number."""

    def setUp(self):
        csv_path = (aps.bss.PROCESSED_DIR / "analysis_raw" / "secular_groups"
                    / "criteria_vs_secular_groups.csv")
        if not csv_path.exists():
            self.skipTest(f"{csv_path} not present; run analyze_secular_groups_vs_criteria first.")
        with open(csv_path, encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))

    def test_mental_destabilization_is_the_longest_criterion(self):
        wc = aps.load_criterion_word_counts()
        self.assertEqual(wc[aps.SUBJECTION_CRITERION_KEY], max(wc.values()))

    def test_absolute_measure_is_length_confounded_but_the_diff_measure_is_not(self):
        _, stats = aps.criterion_length_control(self.rows, aps.load_criterion_word_counts())
        self.assertLess(stats["spearman_length_vs_absolute"], -0.5)
        self.assertLess(abs(stats["spearman_length_vs_diff"]), 0.15)

    def test_the_retracted_rank_and_the_corrected_rank(self):
        _, stats = aps.criterion_length_control(self.rows, aps.load_criterion_word_counts())
        self.assertEqual(stats["focus_rank_absolute_of_17"], 1)   # the retracted "dead last of 17"
        self.assertEqual(stats["focus_rank_diff_of_17"], 6)       # the length-robust truth
        self.assertEqual(stats["n_criteria"], 17)


class TestLoadClusterCoreProxies(unittest.TestCase):
    def _write_csv(self, path, rows):
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["cluster_id", "rank", "key", "label",
                                              "euclidean_distance", "cosine_similarity"])
            w.writeheader()
            w.writerows(rows)

    def test_flags_a_degenerate_cluster_whose_members_are_all_the_same_text(self):
        # Reproduces cluster 40's real shape: every binding expression is the
        # literal token "brainwashing", so the "proxy centroid" carries far
        # less information than its 8 rows suggest.
        points = [{"key": f"doc:{i}"} for i in range(3)]
        vectors = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "b.csv"
            self._write_csv(p, [
                {"cluster_id": 40, "rank": 1, "key": "doc:0", "label": "brainwashing",
                 "euclidean_distance": 0.43, "cosine_similarity": 0.9208},
                {"cluster_id": 40, "rank": 2, "key": "doc:1", "label": "brainwashing",
                 "euclidean_distance": 0.43, "cosine_similarity": 0.9208},
            ])
            proxies = aps.load_cluster_core_proxies(p, (40,), points, vectors)
        self.assertEqual(proxies[40]["n_distinct_vectors"], 1)
        self.assertEqual(proxies[40]["n_resolved"], 2)
        self.assertTrue(aps.proxy_fidelity_rows(proxies)[0]["degenerate"])

    def test_reports_unresolved_keys_rather_than_silently_dropping_them(self):
        points = [{"key": "doc:0"}]
        vectors = np.array([[1.0, 0.0]])
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "b.csv"
            self._write_csv(p, [
                {"cluster_id": 43, "rank": 1, "key": "doc:0", "label": "x",
                 "euclidean_distance": 0.5, "cosine_similarity": 0.87},
                {"cluster_id": 43, "rank": 2, "key": "doc:missing", "label": "y",
                 "euclidean_distance": 0.6, "cosine_similarity": 0.80},
            ])
            proxies = aps.load_cluster_core_proxies(p, (43,), points, vectors)
        self.assertEqual(proxies[43]["missing_keys"], ["doc:missing"])
        self.assertEqual(proxies[43]["n_resolved"], 1)
        self.assertEqual(proxies[43]["member_cos_to_true_centroid_min"], 0.80)

    def test_averages_distinct_member_vectors(self):
        points = [{"key": "a"}, {"key": "b"}]
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "b.csv"
            self._write_csv(p, [
                {"cluster_id": 48, "rank": 1, "key": "a", "label": "x",
                 "euclidean_distance": 0.4, "cosine_similarity": 0.9},
                {"cluster_id": 48, "rank": 2, "key": "b", "label": "y",
                 "euclidean_distance": 0.4, "cosine_similarity": 0.8},
            ])
            proxies = aps.load_cluster_core_proxies(p, (48,), points, vectors)
        np.testing.assert_allclose(proxies[48]["vector"], [0.5, 0.5])
        self.assertEqual(proxies[48]["n_distinct_vectors"], 2)


class TestDiscriminance(unittest.TestCase):
    def test_auc_is_one_when_a_dominates_b(self):
        d = aps.discriminance(np.array([0.9, 0.8]), np.array([0.1, 0.2]))
        self.assertEqual(d["auc"], 1.0)
        self.assertTrue(d["clears_artifact_band"])

    def test_auc_is_half_when_samples_are_identical(self):
        d = aps.discriminance(np.array([0.5, 0.5]), np.array([0.5, 0.5]))
        self.assertEqual(d["auc"], 0.5)
        self.assertEqual(d["gap"], 0.0)
        self.assertFalse(d["clears_artifact_band"])

    def test_hand_computed_partial_overlap(self):
        # a = [0.6, 0.4], b = [0.5]  ->  one win, one loss  ->  AUC 0.5
        d = aps.discriminance(np.array([0.6, 0.4]), np.array([0.5]))
        self.assertEqual(d["auc"], 0.5)
        self.assertAlmostEqual(d["gap"], 0.0, places=6)

    def test_a_real_but_sub_artifact_gap_is_flagged_as_not_clearing(self):
        # +0.02 is a genuine mean difference but smaller than the +0.061
        # "The"-token artifact, so it must not be reported as interpretable.
        d = aps.discriminance(np.array([0.52, 0.52]), np.array([0.50, 0.50]))
        self.assertAlmostEqual(d["gap"], 0.02, places=6)
        self.assertEqual(d["auc"], 1.0)
        self.assertFalse(d["clears_artifact_band"])


class TestSummarizeEntityNearestCluster(unittest.TestCase):
    def test_flags_membership_of_the_subjection_clusters_per_group_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "enc.csv"
            with open(p, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["entity_key", "entity_label",
                                                  "nearest_cluster_id", "euclidean_distance",
                                                  "cosine_similarity"])
                w.writeheader()
                w.writerows([
                    {"entity_key": "heaven's gate", "entity_label": "heaven's gate",
                     "nearest_cluster_id": "4", "euclidean_distance": 0.1, "cosine_similarity": 0.985},
                    {"entity_key": "maoist thought reform", "entity_label": "maoist thought reform",
                     "nearest_cluster_id": "43", "euclidean_distance": 0.5, "cosine_similarity": 0.587},
                    {"entity_key": "unrelated", "entity_label": "unrelated",
                     "nearest_cluster_id": "43", "euclidean_distance": 0.5, "cosine_similarity": 0.5},
                ])
            rows = aps.summarize_entity_nearest_cluster(
                p, (40, 43, 48),
                {"religious": {"heaven's gate"}, "secular": {"maoist thought reform"}},
            )
        self.assertEqual(len(rows), 2)  # "unrelated" is in neither named set
        by_set = {r["group_set"]: r for r in rows}
        self.assertFalse(by_set["religious"]["is_subjection_cluster"])
        self.assertTrue(by_set["secular"]["is_subjection_cluster"])


if __name__ == "__main__":
    unittest.main()
