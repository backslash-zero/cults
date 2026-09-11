"""Unit tests for thesis_corpus.screen_interviews_full -- the deterministic
screen for the exhaustive ("extract everything, label cult-relevance")
interview pipeline. These fixtures directly encode the requirements that
motivated this module: short/no-lexicon and cult_relevant=False candidates
must be RETAINED (never rejected), interviewer speech is retained with
attribution="interviewer" instead of hard-rejected, and -- since
interview_chunking.py now strips all `Interviewer:`/`Interviewee:` labels
and the transcript header/notes before this module ever sees the text --
attribution comes entirely from the parallel `turn_roles` list, paired up
positionally via `turn_spans()`.

Run from thesis/corpus/:  python -m unittest thesis_corpus.tests.test_screen_interviews_full -v
"""
import unittest

from thesis_corpus import interview_extraction_schema as schema
from thesis_corpus import screen_interviews_full as sif

# Label-free, exactly as interview_chunking.build_transcript_units would
# produce it: turns joined by blank lines, header/notes already dropped.
TRANSCRIPT = ('So, first and only question — when you hear the word "cult," what comes to your mind?\n\n'
              "Honestly, some kind of new wave band — Blue Öyster Cult. And then secondly, religious cults.\n\n"
              "And what is culty about them?\n\n"
              "Well, usually following a leadership that tells you what's right and what's wrong.")
TRANSCRIPT_ROLES = ["interviewer", "participant", "interviewer", "participant"]

SHORT_TRANSCRIPT = "When you hear the word cult, what comes to mind?\n\nIlluminati.\n\nAnything else?\n\nOkay."
SHORT_TRANSCRIPT_ROLES = ["interviewer", "participant", "interviewer", "participant"]


def _candidate_full(text, kind="claim", **overrides):
    base = dict(verbatim_expression=text, expression_kind=kind, claim_mode="direct_statement",
                epistemic_status="asserted", entity_anchors=[], cult_relevant=True, relevance_note="test")
    base.update(overrides)
    return base


def ctx(text=TRANSCRIPT, document_id="b1-aug12-1231"):
    return sif.prepare_chunk(document_id, 0, [1, 1], text, "interviews_full")


def screen(text, text_ctx=TRANSCRIPT, roles=TRANSCRIPT_ROLES, **kw):
    cand, code, detail = schema.validate_candidate(_candidate_full(text, **kw))
    assert cand is not None, (code, detail)
    return sif.screen_candidate(cand, ctx(text_ctx), roles)


def screen_short(text, **kw):
    return screen(text, text_ctx=SHORT_TRANSCRIPT, roles=SHORT_TRANSCRIPT_ROLES, **kw)


class ExhaustivenessTests(unittest.TestCase):
    """Guards the two rules that were deliberately dropped from screen_v2.py:
    the short-fragment-no-referent rule, and the cult-relevance gate."""

    def test_short_no_lexicon_candidate_retained(self):
        out = screen_short("Illuminati.")
        self.assertIsNone(out.rejection_code, out.detail)
        self.assertEqual(out.attribution, "participant")

    def test_filler_retained(self):
        out = screen_short("Okay.")
        self.assertIsNone(out.rejection_code, out.detail)

    def test_cult_relevant_false_candidate_retained(self):
        out = screen_short("Illuminati.", cult_relevant=False)
        self.assertIsNone(out.rejection_code, out.detail)

    def test_standalone_bare_name_not_rejected(self):
        # screen_v2's S11 (standalone_personal_name) has no equivalent here --
        # a bare named answer is exactly what exhaustive extraction should keep.
        out = screen("Blue Öyster Cult", kind="example_or_named_group")
        self.assertIsNone(out.rejection_code, out.detail)


class AttributionTests(unittest.TestCase):
    def test_interviewer_turn_retained_with_attribution(self):
        out = screen('when you hear the word "cult," what comes to your mind?')
        self.assertIsNone(out.rejection_code, out.detail)
        self.assertEqual(out.attribution, "interviewer")

    def test_participant_turn_retained_with_attribution(self):
        out = screen("following a leadership that tells you what's right and what's wrong")
        self.assertIsNone(out.rejection_code, out.detail)
        self.assertEqual(out.attribution, "participant")

    def test_second_interviewer_turn_also_attributed_correctly(self):
        out = screen("what is culty about them?")
        self.assertIsNone(out.rejection_code, out.detail)
        self.assertEqual(out.attribution, "interviewer")

    def test_span_crossing_turns_rejected(self):
        out = screen("religious cults.\n\nAnd what is culty about them?")
        self.assertEqual(out.rejection_code, "span_crosses_speaker_turn")

    def test_span_outside_known_turn_when_roles_dont_cover_text(self):
        # defensive fallback: if turn_roles is shorter than the actual
        # number of paragraphs (should never happen with chunking-produced
        # input), a span in the uncovered tail is rejected, not mis-attributed.
        out = screen("following a leadership that tells you what's right and what's wrong",
                     roles=["interviewer", "participant"])  # only covers the first two turns
        self.assertEqual(out.rejection_code, "span_outside_known_turn")


class StructuralCorrectnessTests(unittest.TestCase):
    """Rules explicitly KEPT because they are correctness properties, not
    relevance judgments -- verbatim fidelity, boundary sanity, integrity."""

    def test_not_verbatim_rejected(self):
        out = screen("something never said in this transcript")
        self.assertEqual(out.rejection_code, "not_verbatim")

    def test_dangling_boundary_rejected(self):
        out = screen("And then secondly, religious cults")
        self.assertEqual(out.rejection_code, "dangling_boundary")

    def test_hard_corruption_rejected(self):
        text = "What comes to mind?\n\nThe le�der demands obedience.\n\n"
        out = screen("The le�der demands obedience", text_ctx=text, roles=["interviewer", "participant"])
        self.assertEqual(out.rejection_code, "integrity_replacement_char")


class DedupTests(unittest.TestCase):
    def test_duplicate_candidates_deduped(self):
        text = "following a leadership that tells you what's right and what's wrong"
        raw = [_candidate_full(text), _candidate_full(text)]
        retained, rejected = sif.screen_unit(raw, ctx(), set(), TRANSCRIPT_ROLES)
        self.assertEqual(len(retained), 1)
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["rejection_code"], "duplicate_or_overlapping_span")

    def test_no_per_unit_cap(self):
        # unlike screen_v2 (MAX_PER_CHUNK=4), this pipeline has no cap
        raw = [_candidate_full("Illuminati."), _candidate_full("Okay.")]
        retained, rejected = sif.screen_unit(raw, ctx(SHORT_TRANSCRIPT), set(), SHORT_TRANSCRIPT_ROLES)
        self.assertEqual(len(retained), 2)
        self.assertEqual(rejected, [])


class RetainedRecordShapeTests(unittest.TestCase):
    def test_retained_record_has_no_attribution_source_or_validation_flags(self):
        raw = [_candidate_full("Illuminati.", cult_relevant=False)]
        retained, _ = sif.screen_unit(raw, ctx(SHORT_TRANSCRIPT), set(), SHORT_TRANSCRIPT_ROLES)
        rec = retained[0]
        self.assertEqual(rec["attribution"], "participant")
        self.assertEqual(rec["cult_relevant"], False)
        for legacy_field in ("attribution_source", "model_attribution", "self_contained",
                             "textually_intelligible", "single_coherent_expression", "chunk_relevance"):
            self.assertNotIn(legacy_field, rec)

    def test_context_window_never_contains_labels_or_header(self):
        raw = [_candidate_full("Illuminati.")]
        retained, _ = sif.screen_unit(raw, ctx(SHORT_TRANSCRIPT), set(), SHORT_TRANSCRIPT_ROLES)
        for marker in ("Interviewer:", "Interviewee:", "Interview Notes:", "---"):
            self.assertNotIn(marker, retained[0]["context_window"])


class TurnSpansTests(unittest.TestCase):
    def test_pairs_paragraphs_with_roles_in_order(self):
        spans = sif.turn_spans(TRANSCRIPT, TRANSCRIPT_ROLES)
        self.assertEqual([role for _s, _e, role in spans], TRANSCRIPT_ROLES)
        for (start, end, _role), expected_para in zip(spans, TRANSCRIPT.split("\n\n")):
            self.assertEqual(TRANSCRIPT[start:end], expected_para)


if __name__ == "__main__":
    unittest.main()
