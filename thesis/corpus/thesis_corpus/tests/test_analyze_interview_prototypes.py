"""Unit tests for thesis_corpus.analyze_interview_prototypes.

Run from thesis/corpus/:  python -m unittest thesis_corpus.tests.test_analyze_interview_prototypes -v
"""
import unittest

import numpy as np

from thesis_corpus import analyze_interview_prototypes as aip


class TestBestMatchInDocument(unittest.TestCase):
    def test_exact_containment_scores_high(self):
        candidates = [
            {"index": 0, "label": "[laughs] Okay — it's a local community of people growing local tomatoes."},
            {"index": 1, "label": "It's a group?"},
        ]
        match, ratio = aip.best_match_in_document(["it's a local community of people growing local tomatoes."], candidates)
        self.assertEqual(match["index"], 0)
        self.assertGreaterEqual(ratio, 0.8)

    def test_picks_the_closer_of_two_partial_candidates(self):
        candidates = [
            {"index": 0, "label": "this AI thing you told me about, this AI cult."},
            {"index": 1, "label": "completely unrelated sentence about something else entirely."},
        ]
        match, ratio = aip.best_match_in_document(["AI cult"], candidates)
        self.assertEqual(match["index"], 0)

    def test_no_candidates_returns_none(self):
        match, ratio = aip.best_match_in_document(["anything"], [])
        self.assertIsNone(match)
        self.assertEqual(ratio, 0.0)

    def test_second_query_text_rescues_a_bad_first_match(self):
        # Mirrors the real bug found on b2-aug13-1832: the composite
        # exemplar text matches poorly everywhere, but the precise label
        # is an exact hit -- trying both and keeping the best must find it.
        candidates = [
            {"index": 0, "label": "cult films are usually films that were not very economically successful"},
            {"index": 1, "label": "some other sentence from the same document"},
        ]
        composite = "one thing that comes to my mind ... they will call it a 'cult' bicycle part ... cult films are usually films that were not very economically successful ... you would use 'cult' as an adjective."
        label = "cult films are usually films that were not very economically successful"
        match, ratio = aip.best_match_in_document([label, composite], candidates)
        self.assertEqual(match["index"], 0)
        self.assertGreaterEqual(ratio, 0.99)


class TestResolvePrototypes(unittest.TestCase):
    def test_resolves_within_same_document_only(self):
        points = [
            {"source_dataset": "interviews", "key": "docA:0", "label": "Tomato cult! It's a local community."},
            {"source_dataset": "interviews", "key": "docB:0", "label": "Something else about docB entirely."},
        ]
        vectors = np.zeros((2, 3))
        exemplar_rows = [{
            "document_id": "docA", "transcript_initial_exemplar_text": "It's a local community.",
            "source_expression_label": "It's a local community.",
        }]
        resolved = aip.resolve_prototypes(exemplar_rows, points, vectors)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0]["matched_index"], 0)
        self.assertFalse(resolved[0]["low_confidence"])

    def test_wrong_document_never_matched(self):
        points = [
            {"source_dataset": "interviews", "key": "docB:0", "label": "the exact right text lives only here"},
        ]
        vectors = np.zeros((1, 3))
        exemplar_rows = [{
            "document_id": "docA", "transcript_initial_exemplar_text": "the exact right text lives only here",
            "source_expression_label": "the exact right text lives only here",
        }]
        resolved = aip.resolve_prototypes(exemplar_rows, points, vectors)
        self.assertIsNone(resolved[0]["matched_index"])
        self.assertTrue(resolved[0]["low_confidence"])


if __name__ == "__main__":
    unittest.main()
