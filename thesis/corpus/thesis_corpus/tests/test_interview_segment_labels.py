"""Unit tests for thesis_corpus.interview_segment_labels -- the coverage
guarantee this pipeline exists for: every segment gets exactly one archive
record, whether or not a valid model label exists for it.

Run from thesis/corpus/:  python -m unittest thesis_corpus.tests.test_interview_segment_labels -v
"""
import unittest

from thesis_corpus import interview_segment_labels as isl

TRANSCRIPT = ('So, first and only question — when you hear the word "cult," what comes to your mind?\n\n'
              "Illuminati.\n\n"
              "Okay.\n\n"
              "Well, usually following a leadership that tells you what's right and what's wrong.")
TRANSCRIPT_ROLES = ["interviewer", "participant", "participant", "participant"]


def _label(segment_index, cult_relevant=True, **overrides):
    base = dict(segment_index=segment_index, expression_kind="claim", claim_mode="direct_statement",
                epistemic_status="asserted", entity_anchors=[], cult_relevant=cult_relevant, relevance_note="test")
    base.update(overrides)
    return base


def ctx(text=TRANSCRIPT, document_id="doc1"):
    return isl.prepare_chunk(document_id, 0, [1, 1], text, "interviews_full")


class SegmentSpansTests(unittest.TestCase):
    def test_pairs_paragraphs_with_roles_in_order(self):
        spans = isl.segment_spans(TRANSCRIPT, TRANSCRIPT_ROLES)
        self.assertEqual([role for _s, _e, role in spans], TRANSCRIPT_ROLES)
        for (start, end, _role), expected in zip(spans, TRANSCRIPT.split("\n\n")):
            self.assertEqual(TRANSCRIPT[start:end], expected)


class CoverageGuaranteeTests(unittest.TestCase):
    """The core property this module exists for."""

    def test_every_segment_gets_a_record_even_with_no_labels_at_all(self):
        records, issues = isl.build_segment_records(ctx(), TRANSCRIPT_ROLES, [])
        self.assertEqual(len(records), 4)
        self.assertTrue(all(r["label_status"] == "missing" for r in records))
        self.assertTrue(all(r["cult_relevant"] is None for r in records))
        self.assertEqual([i.code for i in issues], ["segment_unlabeled"] * 4)

    def test_every_segment_gets_a_record_with_partial_labels(self):
        raw = [_label(0), _label(2, cult_relevant=False)]
        records, issues = isl.build_segment_records(ctx(), TRANSCRIPT_ROLES, raw)
        self.assertEqual(len(records), 4)
        statuses = [r["label_status"] for r in records]
        self.assertEqual(statuses, ["model", "missing", "model", "missing"])
        self.assertEqual(records[2]["cult_relevant"], False)
        unlabeled_codes = [i.code for i in issues]
        self.assertEqual(unlabeled_codes.count("segment_unlabeled"), 2)

    def test_verbatim_expression_is_always_the_real_segment_text(self):
        # no verbatim-matching involved at all -- the text IS the segment,
        # guaranteed to be an exact, real substring of the transcript.
        records, _ = isl.build_segment_records(ctx(), TRANSCRIPT_ROLES, [_label(1)])
        self.assertEqual(records[1]["verbatim_expression"], "Illuminati.")
        self.assertEqual(records[1]["attribution"], "participant")

    def test_malformed_label_does_not_remove_the_segment(self):
        raw = [_label(1, cult_relevant="not-a-boolean")]  # otherwise complete, fails StrictBool
        records, issues = isl.build_segment_records(ctx(), TRANSCRIPT_ROLES, raw)
        self.assertEqual(len(records), 4)
        self.assertEqual(records[1]["label_status"], "missing")
        self.assertIn("invalid_label_shape", [i.code for i in issues])

    def test_out_of_range_segment_index_reported_not_fatal(self):
        raw = [_label(99)]
        records, issues = isl.build_segment_records(ctx(), TRANSCRIPT_ROLES, raw)
        self.assertEqual(len(records), 4)  # still exactly one record per real segment
        self.assertTrue(all(r["label_status"] == "missing" for r in records))
        self.assertIn("segment_index_out_of_range", [i.code for i in issues])

    def test_duplicate_segment_index_first_one_kept(self):
        raw = [_label(0, relevance_note="first"), _label(0, relevance_note="second")]
        records, issues = isl.build_segment_records(ctx(), TRANSCRIPT_ROLES, raw)
        self.assertEqual(records[0]["relevance_note"], "first")
        self.assertIn("duplicate_segment_index", [i.code for i in issues])

    def test_filler_segment_labeled_and_retained(self):
        raw = [_label(2, cult_relevant=False, relevance_note="filler")]
        records, _ = isl.build_segment_records(ctx(), TRANSCRIPT_ROLES, raw)
        self.assertEqual(records[2]["verbatim_expression"], "Okay.")
        self.assertEqual(records[2]["label_status"], "model")
        self.assertFalse(records[2]["cult_relevant"])

    def test_no_attribution_source_or_v1_fields_leak_into_records(self):
        records, _ = isl.build_segment_records(ctx(), TRANSCRIPT_ROLES, [_label(0)])
        for legacy_field in ("attribution_source", "model_attribution", "self_contained",
                             "textually_intelligible", "single_coherent_expression", "chunk_relevance",
                             "candidate_rank_source"):
            self.assertNotIn(legacy_field, records[0])


if __name__ == "__main__":
    unittest.main()
