"""Unit tests for thesis_corpus.translate_interview_segments.

Run from thesis/corpus/:  python -m unittest thesis_corpus.tests.test_translate_interview_segments -v
"""
import json
import tempfile
import unittest
from pathlib import Path

from thesis_corpus import translate_interview_segments as tis


def _segment(document_id, segment_index, text, chunk_index=0):
    return {"document_id": document_id, "segment_index": segment_index, "chunk_index": chunk_index,
            "verbatim_expression": text}


class TranslateInterviewSegmentsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.source_path = root / "expressions_v2.jsonl"
        self.output_path = root / "segment_translations_en.jsonl"
        segments = [
            _segment("b4-sep10-1838", 0, "Religion."),
            _segment("b4-sep10-1838", 1, "Ouais, religion."),
            _segment("b1-aug05-1650", 0, "Tomato cult!"),  # English -- must be skipped
        ]
        with open(self.source_path, "w", encoding="utf-8") as f:
            for s in segments:
                f.write(json.dumps(s) + "\n")
        self.languages = {"b4-sep10-1838": "French", "b1-aug05-1650": "English"}

    def tearDown(self):
        self.tmp.cleanup()

    def _fake_translate(self, host, model, text, target_language="English", source_language=None):
        return f"[EN] {text}"

    def test_only_non_english_documents_translated(self):
        tis.run(self.source_path, self.output_path, "http://stub", "stub:4b",
               force=False, limit=None, languages=self.languages, translate=self._fake_translate)
        rows = tis.load_jsonl(self.output_path)
        self.assertEqual(len(rows), 2)
        self.assertEqual({r["document_id"] for r in rows}, {"b4-sep10-1838"})
        self.assertEqual(rows[0]["translation_en"], "[EN] Religion.")
        self.assertEqual(rows[0]["source_language"], "French")
        self.assertEqual(rows[0]["text_source"], "Religion.")

    def test_source_archive_never_modified(self):
        before = self.source_path.read_text(encoding="utf-8")
        tis.run(self.source_path, self.output_path, "http://stub", "stub:4b",
               force=False, limit=None, languages=self.languages, translate=self._fake_translate)
        self.assertEqual(self.source_path.read_text(encoding="utf-8"), before)

    def test_resume_does_not_duplicate(self):
        tis.run(self.source_path, self.output_path, "http://stub", "stub:4b",
               force=False, limit=None, languages=self.languages, translate=self._fake_translate)
        tis.run(self.source_path, self.output_path, "http://stub", "stub:4b",
               force=False, limit=None, languages=self.languages, translate=self._fake_translate)
        rows = tis.load_jsonl(self.output_path)
        self.assertEqual(len(rows), 2)

    def test_force_reprocesses(self):
        tis.run(self.source_path, self.output_path, "http://stub", "stub:4b",
               force=False, limit=None, languages=self.languages, translate=self._fake_translate)
        tis.run(self.source_path, self.output_path, "http://stub", "stub:4b",
               force=True, limit=None, languages=self.languages, translate=self._fake_translate)
        rows = tis.load_jsonl(self.output_path)
        self.assertEqual(len(rows), 2)

    def test_limit_restricts_batch(self):
        tis.run(self.source_path, self.output_path, "http://stub", "stub:4b",
               force=False, limit=1, languages=self.languages, translate=self._fake_translate)
        rows = tis.load_jsonl(self.output_path)
        self.assertEqual(len(rows), 1)

    def test_translation_failure_recorded_and_skipped(self):
        from thesis_corpus.ollama_client import TranslationError

        def failing_translate(host, model, text, target_language="English", source_language=None):
            raise TranslationError("stub failure")

        tis.run(self.source_path, self.output_path, "http://stub", "stub:4b",
               force=False, limit=None, languages=self.languages, translate=failing_translate)
        rows = tis.load_jsonl(self.output_path)
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
