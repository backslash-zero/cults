"""Unit tests for geometric_analysis_common's typicality/borderline
additions (centroid_and_dispersion_for_indices, ranked_points,
epistemic_status_subgroup_indices, borderline_ranking) and for
analyze_typicality.py itself.

Deliberately run against small, synthetic, hand-built points/vectors --
NOT against the real v1 shared space, throwaway or otherwise. This module
is not executed against real data until v2's shared space exists; see
analyze_typicality.py's module docstring.

Run from thesis/corpus/:  python -m unittest thesis_corpus.tests.test_analyze_typicality -v
"""
import itertools
import unittest
from types import SimpleNamespace

import numpy as np

from thesis_corpus import analyze_typicality as at
from thesis_corpus import geometric_analysis_common as gac


# ---------------------------------------------------------------------------
# Synthetic fixture: 3 expression corpora, well-separated on the x-axis,
# each with several epistemic-status subgroups of varying (deliberately
# thin, in miviludes' case) size -- mirrors the real corpus asymmetry
# (literature: 5 statuses, miviludes: 3, interviews: 4) called out in the plan.
# ---------------------------------------------------------------------------

def _point(key, source_dataset, epistemic_status, x, **overrides):
    p = {
        "key": key, "source_dataset": source_dataset, "epistemic_status": epistemic_status,
        "label": f"label:{key}", "attribution": "author", "claim_mode": "direct_statement",
    }
    p.update(overrides)
    return p, [x, 0.0]


def build_synthetic_corpus():
    rows = []
    # literature: 5 statuses, 2 points each, clustered near x=0.5..9.5
    # (never an exact [0, 0] vector -- that would make cosine_similarity
    # undefined for that point, which no real 394-D embedding ever is).
    lit_statuses = ["asserted", "qualified", "contested", "negated", "speculative"]
    for i, status in enumerate(lit_statuses):
        rows.append(_point(f"lit_doc:{2 * i}", "literature", status, 0.5 + 2 * i))
        rows.append(_point(f"lit_doc:{2 * i + 1}", "literature", status, 1.5 + 2 * i))

    # miviludes: 3 statuses, deliberately uneven sizes (n=1 for contested --
    # thinnest possible subgroup, must still be computed, never dropped).
    rows.append(_point("miv_doc:0", "miviludes", "asserted", 100.0))
    rows.append(_point("miv_doc:1", "miviludes", "asserted", 101.0))
    rows.append(_point("miv_doc:2", "miviludes", "contested", 102.0))
    rows.append(_point("miv_doc:3", "miviludes", "speculative", 103.0))
    rows.append(_point("miv_doc:4", "miviludes", "speculative", 104.0))

    # interviews: 4 statuses, 1-2 points each, clustered near x=200
    int_statuses = ["asserted", "contested", "negated", "speculative"]
    for i, status in enumerate(int_statuses):
        rows.append(_point(f"int_doc:{2 * i}", "interviews", status, 200.0 + 2 * i))
        rows.append(_point(f"int_doc:{2 * i + 1}", "interviews", status, 200.0 + 2 * i + 1))

    # a reference-vocabulary point, to make sure it's never swept into any
    # expression-corpus grouping (epistemic_status is null, as for every
    # non-expression source_dataset in the real pipeline).
    rows.append(_point("concept:0", "concept_backbone", None, 500.0))

    points = [r[0] for r in rows]
    vectors = np.array([r[1] for r in rows], dtype=np.float64)
    return points, vectors


class CentroidAndDispersionForIndicesTest(unittest.TestCase):
    def test_matches_manual_computation(self):
        vectors = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], dtype=np.float64)
        result = gac.centroid_and_dispersion_for_indices(vectors, [0, 1, 2])
        np.testing.assert_allclose(result["centroid"], [2.0, 0.0])
        self.assertAlmostEqual(result["dispersion"], (2.0 + 0.0 + 2.0) / 3)
        self.assertEqual(result["n"], 3)

    def test_per_source_centroids_and_dispersion_byte_identical_after_refactor(self):
        # per_source_centroids_and_dispersion loops over every entry of
        # ALL_SOURCE_DATASETS internally, so this fixture needs >=1 point
        # for each of them (not just the 3 expression corpora) to exercise
        # the refactored function at all.
        points, vectors = build_synthetic_corpus()
        extra_points = []
        extra_vectors = []
        for i, source in enumerate(("miviludes_criteria", "structural_concepts", "emergent_entities", "conceptnet_concepts")):
            extra_points.append({"key": f"{source}:0", "label": source, "source_dataset": source, "epistemic_status": None})
            extra_vectors.append([1000.0 + i, 0.0])
        points = points + extra_points
        vectors = np.concatenate([vectors, np.array(extra_vectors, dtype=np.float64)], axis=0)

        shared_space = SimpleNamespace(points=points, vectors=vectors)
        via_module = gac.per_source_centroids_and_dispersion(shared_space)
        for source in gac.ALL_SOURCE_DATASETS:
            idxs = gac.source_dataset_indices(points, source)
            direct = gac.centroid_and_dispersion_for_indices(vectors, idxs)
            np.testing.assert_array_equal(via_module[source]["centroid"], direct["centroid"])
            self.assertEqual(via_module[source]["dispersion"], direct["dispersion"])
            self.assertEqual(via_module[source]["n"], direct["n"])


class RankedPointsTest(unittest.TestCase):
    def setUp(self):
        self.points = [{"key": f"k{i}", "label": f"l{i}"} for i in range(5)]
        # All-nonzero and off-grid so there are no exact distance ties and
        # cosine_similarity stays well-defined throughout (a zero-norm
        # query OR candidate vector would divide by zero -- never the case
        # for a real 394-D embedding, but easy to hit by accident here).
        self.vectors = np.array([[0.5], [1.5], [4.5], [2.5], [3.5]], dtype=np.float64)
        self.query = np.array([0.1])

    def test_ranked_ascending_covers_every_candidate(self):
        ranked = gac.ranked_points(self.query, self.points, self.vectors)
        self.assertEqual(len(ranked), 5)
        distances = [r["euclidean_distance"] for r in ranked]
        self.assertEqual(distances, sorted(distances))
        self.assertEqual([r["rank"] for r in ranked], [1, 2, 3, 4, 5])
        # nearest-first order by key: k0(0), k1(1), k3(2), k4(3), k2(4)
        self.assertEqual([r["key"] for r in ranked], ["k0", "k1", "k3", "k4", "k2"])

    def test_nearest_points_equals_ranked_points_sliced(self):
        for k in (1, 3, 5):
            self.assertEqual(
                gac.nearest_points(self.query, self.points, self.vectors, k),
                gac.ranked_points(self.query, self.points, self.vectors)[:k],
            )

    def test_atypical_is_reversed_ranked(self):
        ranked = gac.ranked_points(self.query, self.points, self.vectors)
        atypical_keys = [r["key"] for r in list(reversed(ranked))[:2]]
        self.assertEqual(atypical_keys, ["k2", "k4"])  # farthest two: k2(4), k4(3)

    def test_exclude_self(self):
        # Both "self" and "dup" sit exactly at the query's own coordinate --
        # exclude_self can only detect "numerically identical to the query",
        # not "the query's own originating point" specifically, so BOTH are
        # skipped, leaving only "far".
        vectors = np.array([[0.0], [0.0], [5.0]], dtype=np.float64)
        points = [{"key": "self", "label": "self"}, {"key": "dup", "label": "dup"}, {"key": "far", "label": "far"}]
        ranked = gac.ranked_points(np.array([0.0]), points, vectors, exclude_self=True)
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0]["key"], "far")


class EpistemicStatusSubgroupIndicesTest(unittest.TestCase):
    def test_dynamic_discovery_matches_data(self):
        points, _vectors = build_synthetic_corpus()
        lit = gac.epistemic_status_subgroup_indices(points, "literature")
        self.assertEqual(set(lit.keys()), {"asserted", "qualified", "contested", "negated", "speculative"})
        miv = gac.epistemic_status_subgroup_indices(points, "miviludes")
        self.assertEqual(set(miv.keys()), {"asserted", "contested", "speculative"})
        self.assertEqual(len(miv["contested"]), 1)  # thinnest subgroup, still present
        interviews = gac.epistemic_status_subgroup_indices(points, "interviews")
        self.assertEqual(set(interviews.keys()), {"asserted", "contested", "negated", "speculative"})

    def test_raises_on_zero_subgroups(self):
        points = [{"key": "x", "source_dataset": "concept_backbone", "epistemic_status": None}]
        with self.assertRaises(ValueError):
            gac.epistemic_status_subgroup_indices(points, "literature")


class BorderlineRankingTest(unittest.TestCase):
    def test_equidistant_point_ranks_first(self):
        # centroid_a at x=0, centroid_b at x=10; a pool point exactly at x=5
        # is perfectly equidistant (ambiguity_ratio == 0) and must rank first.
        points = [
            {"key": "mid", "label": "mid"},
            {"key": "near_a", "label": "near_a"},
            {"key": "near_b", "label": "near_b"},
        ]
        vectors = np.array([[5.0, 0.0], [1.0, 0.0], [9.0, 0.0]], dtype=np.float64)
        centroid_a = np.array([0.0, 0.0])
        centroid_b = np.array([10.0, 0.0])
        rows = gac.borderline_ranking(points, vectors, [0, 1, 2], centroid_a, centroid_b)
        self.assertEqual(rows[0]["key"], "mid")
        self.assertAlmostEqual(rows[0]["ambiguity_ratio"], 0.0)
        self.assertEqual([r["rank"] for r in rows], [1, 2, 3])
        for r in rows:
            self.assertGreaterEqual(r["ambiguity_ratio"], 0.0)
            expected_side = "a" if r["distance_to_a"] <= r["distance_to_b"] else "b"
            self.assertEqual(r["nominally_closer_to"], expected_side)

    def test_pool_restricted_to_given_indices_only(self):
        points = [{"key": f"k{i}", "label": f"l{i}"} for i in range(4)]
        vectors = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
        rows = gac.borderline_ranking(points, vectors, [0, 2], np.array([0.0]), np.array([2.0]))
        self.assertEqual({r["key"] for r in rows}, {"k0", "k2"})

    def test_min_distance_to_either_centroid_is_diagnostic_not_filter(self):
        points = [{"key": "far_but_equidistant", "label": "x"}]
        vectors = np.array([[1000.0, 0.0]], dtype=np.float64)
        centroid_a = np.array([0.0, 0.0])
        centroid_b = np.array([2000.0, 0.0])
        rows = gac.borderline_ranking(points, vectors, [0], centroid_a, centroid_b)
        self.assertAlmostEqual(rows[0]["ambiguity_ratio"], 0.0)
        self.assertAlmostEqual(rows[0]["min_distance_to_either_centroid"], 1000.0)
        # still present despite being far from both -- diagnostic, not a filter.


class TypicalityRowsForGroupTest(unittest.TestCase):
    def test_typical_and_atypical_are_reverses_of_one_ranking(self):
        points = [{"key": f"doc:{i}", "label": f"l{i}", "source_dataset": "literature",
                   "epistemic_status": "asserted", "attribution": "author", "claim_mode": "direct_statement"}
                  for i in range(5)]
        vectors = np.array([[0.0], [1.0], [2.0], [3.0], [10.0]], dtype=np.float64)
        indices = list(range(5))
        rows, pooled_indices = at.typicality_rows_for_group(points, vectors, indices, k=2, extra={"corpus": "literature"})

        typical = [r for r in rows if r["direction"] == "typical"]
        atypical = [r for r in rows if r["direction"] == "atypical"]
        self.assertEqual(len(typical), 2)
        self.assertEqual(len(atypical), 2)
        # centroid = mean([0,1,2,3,10]) = 3.2 -> distances: doc:3=0.2, doc:2=1.2, doc:1=2.2, doc:0=3.2, doc:4=6.8
        # nearest two (typical) are doc:3,doc:2; farthest two (atypical) are doc:4,doc:0.
        self.assertEqual([r["pooled_key"] for r in typical], ["doc:3", "doc:2"])
        self.assertEqual([r["pooled_key"] for r in atypical], ["doc:4", "doc:0"])
        for r in rows:
            self.assertEqual(r["n_in_group"], 5)
            self.assertIsNone(r["context_window"])  # not yet attached

    def test_thin_group_never_dropped(self):
        points = [{"key": "only:0", "label": "only", "source_dataset": "miviludes",
                   "epistemic_status": "contested", "attribution": "author", "claim_mode": "direct_statement"}]
        vectors = np.array([[5.0]], dtype=np.float64)
        rows, _pooled = at.typicality_rows_for_group(points, vectors, [0], k=10, extra={"corpus": "miviludes"})
        self.assertEqual(len(rows), 2)  # 1 typical + 1 atypical, both the same lone point
        self.assertTrue(all(r["n_in_group"] == 1 for r in rows))


class AttachContextWindowsTest(unittest.TestCase):
    def test_fills_fields_from_fake_resolution(self):
        points = [{"key": "doc:0", "source_dataset": "literature", "label": "x"}]
        rows = [{"pooled_key": "doc:0"}]
        resolutions = {
            "literature": SimpleNamespace(
                context_window_by_pooled_index={0: "the context"},
                occurrence_key_by_pooled_index={0: "doc:0:0"},
            ),
        }
        at.attach_context_windows(points, rows, [0], resolutions)
        self.assertEqual(rows[0]["context_window"], "the context")
        self.assertEqual(rows[0]["occurrence_key"], "doc:0:0")

    def test_raises_on_missing_context_window_for_selected_row(self):
        points = [{"key": "doc:0", "source_dataset": "literature", "label": "x"}]
        rows = [{"pooled_key": "doc:0"}]
        resolutions = {
            "literature": SimpleNamespace(
                context_window_by_pooled_index={},  # unmatched -- must hard-fail
                occurrence_key_by_pooled_index={0: "doc:0:0"},
            ),
        }
        with self.assertRaises(ValueError):
            at.attach_context_windows(points, rows, [0], resolutions)


class ComputeTypicalityIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.points, self.vectors = build_synthetic_corpus()
        self.result = at.compute_typicality(self.points, self.vectors, k=3)

    def test_groupings_summary_matches_dynamic_discovery(self):
        summary = self.result.groupings_summary
        self.assertEqual(set(summary["per_corpus"]["literature"]["statuses"]),
                          {"asserted", "qualified", "contested", "negated", "speculative"})
        self.assertEqual(set(summary["per_corpus"]["miviludes"]["statuses"]),
                          {"asserted", "contested", "speculative"})
        self.assertEqual(summary["per_corpus"]["miviludes"]["statuses"]["contested"], 1)
        self.assertEqual(set(summary["per_corpus"]["interviews"]["statuses"]),
                          {"asserted", "contested", "negated", "speculative"})

    def test_cross_epistemology_pair_count(self):
        self.assertEqual(len(self.result.groupings_summary["cross_epistemology_pairs"]), 3)  # C(3,2)

    def test_cross_status_pair_counts(self):
        pairs = self.result.groupings_summary["cross_status_pairs"]
        lit_pairs = [p for p in pairs if p["corpus"] == "literature"]
        miv_pairs = [p for p in pairs if p["corpus"] == "miviludes"]
        int_pairs = [p for p in pairs if p["corpus"] == "interviews"]
        self.assertEqual(len(lit_pairs), 10)  # C(5,2)
        self.assertEqual(len(miv_pairs), 3)   # C(3,2)
        self.assertEqual(len(int_pairs), 6)   # C(4,2)

    def test_reference_vocab_point_never_appears(self):
        for row in self.result.typicality_by_corpus_rows + self.result.typicality_by_corpus_and_status_rows:
            self.assertNotEqual(row["pooled_key"], "concept:0")
        for row in self.result.borderline_cross_epistemology_rows + self.result.borderline_cross_status_within_corpus_rows:
            self.assertNotEqual(row["pooled_key"], "concept:0")

    def test_validations_pass_on_well_formed_output(self):
        at.run_all_validations(self.result, self.points)  # must not raise


class ValidationFailureModesTest(unittest.TestCase):
    """Deliberately corrupt otherwise-valid rows to confirm each validator
    actually catches the invariant it claims to check."""

    def test_rank_gap_is_caught(self):
        rows = [
            {"corpus": "literature", "direction": "typical", "rank": 1, "euclidean_distance_to_centroid": 1.0},
            {"corpus": "literature", "direction": "typical", "rank": 3, "euclidean_distance_to_centroid": 2.0},
        ]
        with self.assertRaises(AssertionError):
            at.validate_typicality_ranks_and_monotonicity(rows, ("corpus",))

    def test_non_monotonic_typical_distance_is_caught(self):
        rows = [
            {"corpus": "literature", "direction": "typical", "rank": 1, "euclidean_distance_to_centroid": 5.0},
            {"corpus": "literature", "direction": "typical", "rank": 2, "euclidean_distance_to_centroid": 1.0},
        ]
        with self.assertRaises(AssertionError):
            at.validate_typicality_ranks_and_monotonicity(rows, ("corpus",))

    def test_wrong_n_in_group_is_caught(self):
        points, _vectors = build_synthetic_corpus()
        rows = [{"corpus": "literature", "epistemic_status_group": None, "n_in_group": 999}]
        with self.assertRaises(AssertionError):
            at.validate_n_in_group_matches_direct_count(rows, points)

    def test_wrong_n_pool_is_caught(self):
        points, _vectors = build_synthetic_corpus()
        rows = [{"corpus_a": "literature", "corpus_b": "miviludes", "n_pool": 1}]
        with self.assertRaises(AssertionError):
            at.validate_n_pool_matches_direct_count(rows, points)

    def test_negative_ambiguity_ratio_is_caught(self):
        rows = [{"ambiguity_ratio": -0.1, "corpus_a": "literature", "corpus_b": "miviludes",
                 "distance_to_a": 1.0, "distance_to_b": 2.0, "nominally_closer_to": "literature"}]
        with self.assertRaises(AssertionError):
            at.validate_borderline_rows(rows)

    def test_inconsistent_nominally_closer_to_is_caught(self):
        rows = [{"ambiguity_ratio": 0.1, "corpus_a": "literature", "corpus_b": "miviludes",
                 "distance_to_a": 1.0, "distance_to_b": 2.0, "nominally_closer_to": "miviludes"}]
        with self.assertRaises(AssertionError):
            at.validate_borderline_rows(rows)


class UnorderedPairsTest(unittest.TestCase):
    def test_matches_itertools_combinations(self):
        items = ["a", "b", "c", "d"]
        self.assertEqual(at.unordered_pairs(items), list(itertools.combinations(items, 2)))


if __name__ == "__main__":
    unittest.main()
