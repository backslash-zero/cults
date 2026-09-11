"""Unit tests for thesis_corpus.interview_chunking.

Run from thesis/corpus/:  python -m unittest thesis_corpus.tests.test_interview_chunking -v
"""
import unittest

from thesis_corpus.interview_chunking import build_transcript_units

HEADER = "INTERVIEW — Aug 5, 16:50\n\nInterviewee: 30, female.\n\n---\n\n"


def _turn(role, text):
    label = {"interviewer": "Interviewer", "participant": "Interviewee"}[role]
    return f"{label}: {text}\n\n"


class WholeTranscriptTests(unittest.TestCase):
    def test_short_transcript_is_one_unit_with_labels_stripped(self):
        text = HEADER + _turn("interviewer", "When you hear cult, what comes to mind?") + _turn("participant", "Tomatoes.")
        pages = [{"page_number": 1, "text": text}]
        units = build_transcript_units("doc1", pages, max_words=1800)
        self.assertEqual(len(units), 1)
        unit = units[0]
        self.assertEqual(unit.chunk_index, 0)
        self.assertEqual(unit.page_range, [1, 1])
        self.assertEqual(unit.text, "When you hear cult, what comes to mind?\n\nTomatoes.")
        self.assertEqual(unit.turn_roles, ["interviewer", "participant"])
        for marker in ("Interviewer", "Interviewee", "---", "30, female"):
            self.assertNotIn(marker, unit.text)

    def test_header_and_demographics_dropped(self):
        text = HEADER + _turn("interviewer", "Q?") + _turn("participant", "A.")
        pages = [{"page_number": 1, "text": text}]
        unit = build_transcript_units("doc1", pages)[0]
        self.assertNotIn("16:50", unit.text)
        self.assertNotIn("female", unit.text)

    def test_interview_notes_dropped_not_just_relabeled(self):
        text = (HEADER + _turn("interviewer", "Q?") + _turn("participant", "A.")
                + "Interview Notes: name unclear, ASR corrected.\n\n")
        pages = [{"page_number": 1, "text": text}]
        unit = build_transcript_units("doc1", pages)[0]
        self.assertEqual(unit.text, "Q?\n\nA.")
        self.assertEqual(unit.turn_roles, ["interviewer", "participant"])
        self.assertNotIn("name unclear", unit.text)

    def test_trailing_double_dash_stripped(self):
        # real transcript b3-aug16-1517 ends its last turn in a stray "--"
        # rather than the usual "---" -- must still be treated as the
        # separator convention, not left dangling in the extracted text.
        text = HEADER + _turn("interviewer", "Q?") + "Interviewee: A.\n\n--\n"
        pages = [{"page_number": 1, "text": text}]
        unit = build_transcript_units("doc1", pages)[0]
        self.assertEqual(unit.text, "Q?\n\nA.")

    def test_trailing_separator_before_notes_stripped(self):
        text = (HEADER + _turn("interviewer", "Q?") + "Interviewee: A.\n\n---\n\n"
                + "Interview Notes: whatever.\n\n")
        pages = [{"page_number": 1, "text": text}]
        unit = build_transcript_units("doc1", pages)[0]
        self.assertEqual(unit.text, "Q?\n\nA.")

    def test_multi_sentence_turn_split_into_separate_segments(self):
        long_answer = "First idea about cults. Second, unrelated idea about leaders. Third idea?"
        text = HEADER + _turn("interviewer", "Tell me about cults.") + _turn("participant", long_answer)
        pages = [{"page_number": 1, "text": text}]
        unit = build_transcript_units("doc1", pages)[0]
        paragraphs = unit.text.split("\n\n")
        self.assertEqual(paragraphs, [
            "Tell me about cults.",
            "First idea about cults.",
            "Second, unrelated idea about leaders.",
            "Third idea?",
        ])
        self.assertEqual(unit.turn_roles, ["interviewer", "participant", "participant", "participant"])

    def test_no_turns_at_all_returns_no_units(self):
        pages = [{"page_number": 1, "text": "Just some header text with no Interviewer label."}]
        self.assertEqual(build_transcript_units("doc1", pages), [])

    def test_no_pages_returns_no_units(self):
        self.assertEqual(build_transcript_units("doc1", []), [])


class SplitFallbackTests(unittest.TestCase):
    def _long_transcript(self, n_turns=12):
        parts = [HEADER]
        for i in range(n_turns):
            parts.append(_turn("interviewer", f"Question number {i} about cults and groups and leaders?"))
            parts.append(_turn("participant", " ".join(["word"] * 40) + f" answer {i}."))
        return "".join(parts)

    def test_split_at_interviewer_turn_boundary_labels_stripped(self):
        text = self._long_transcript(n_turns=12)
        pages = [{"page_number": 1, "text": text}]
        units = build_transcript_units("doc1", pages, max_words=50)
        self.assertEqual(len(units), 2)
        self.assertEqual([u.chunk_index for u in units], [0, 1])
        for unit in units:
            for marker in ("Interviewer:", "Interviewee:"):
                self.assertNotIn(marker, unit.text)
        # roles line up 1:1 with blank-line-separated paragraphs in each unit
        for unit in units:
            self.assertEqual(len(unit.text.split("\n\n")), len(unit.turn_roles))
        # nothing lost or duplicated across the split
        combined_words = sum(len(u.text.split()) for u in units)
        original_words = sum(len(u.text.split()) for u in build_transcript_units("doc1", pages, max_words=100000))
        self.assertEqual(combined_words, original_words)

    def test_no_interviewer_boundary_keeps_one_oversized_unit(self):
        # a single interviewer turn followed by one huge participant turn --
        # no second interviewer turn to split at
        text = HEADER + _turn("interviewer", "Tell me everything.") + _turn("participant", " ".join(["word"] * 200))
        pages = [{"page_number": 1, "text": text}]
        units = build_transcript_units("doc1", pages, max_words=50)
        self.assertEqual(len(units), 1)


if __name__ == "__main__":
    unittest.main()
