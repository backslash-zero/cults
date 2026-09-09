"""Interview speaker-turn rules and the embed_v2 runner (stubbed embeddings).

Run from thesis/corpus/:  python -m unittest thesis_corpus.tests.test_interviews_and_embed -v
"""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from thesis_corpus import screen_v2 as sv, extraction_v2_schema as schema, embed_v2
from thesis_corpus.tests.test_pilot_v2_stages import _candidate

TRANSCRIPT = ("INTERVIEW — Aug 12, starting 12:31\n\nInterviewee: age 34, gender female, main language German.\n\n---\n\n"
              'Interviewer: So, first and only question — when you hear the word "cult," what comes to your mind?\n\n'
              "Interviewee: Honestly, some kind of new wave band — Blue Öyster Cult. And then secondly, religious cults.\n\n"
              "Interviewer : And what is culty about them?\n\n"
              "Interviewee : Well, usually following a leadership that tells you what's right and what's wrong.\n\n"
              "Interview Notes: name unclear at 12:40, ASR corrected.")


def ctx(text=TRANSCRIPT):
    return sv.prepare_chunk("b1-aug12-1231", 0, [1, 1], text, "interviews")


def screen(text, **kw):
    cand, code, detail = schema.validate_candidate(_candidate(text, **kw))
    assert cand is not None, (code, detail)
    return sv.screen_candidate(cand, ctx())


class SpeakerTurnTests(unittest.TestCase):
    def test_turns(self):
        roles = [r for _, _, r in sv.speaker_turns(TRANSCRIPT)]
        self.assertEqual(roles, ["header", "interviewer", "participant", "interviewer", "participant", "notes"])

    def test_interviewer_question_rejected_even_if_model_says_participant(self):
        out = screen('when you hear the word "cult," what comes to your mind?', attribution="participant",
                     kind="question_or_reflection", claim_mode="question_or_reflection")
        self.assertEqual(out.rejection_code, "interviewer_utterance")
        self.assertEqual(out.detail, "transcript speaker turn")

    def test_participant_answer_attribution_forced(self):
        out = screen("following a leadership that tells you what's right and what's wrong", attribution="unspecified")
        self.assertIsNone(out.rejection_code, out.detail)
        self.assertEqual(out.attribution_override, "participant")
        self.assertIn("attribution_from_transcript", out.flags)
        retained, _ = sv.screen_chunk([_candidate("following a leadership that tells you what's right and what's wrong",
                                                  attribution="unspecified")], ctx(), set())
        self.assertEqual(retained[0]["attribution"], "participant")
        self.assertEqual(retained[0]["attribution_source"], "transcript")
        self.assertEqual(retained[0]["model_attribution"], "unspecified")

    def test_short_association_in_participant_turn(self):
        out = screen("religious cults", attribution="participant", kind="association_or_framing")
        self.assertIsNone(out.rejection_code, out.detail)

    def test_span_crossing_turns_rejected(self):
        out = screen("religious cults.\n\nInterviewer : And what is culty about them?", attribution="participant")
        self.assertIn(out.rejection_code, ("span_includes_speaker_label", "span_crosses_speaker_turn"))

    def test_header_and_notes_rejected(self):
        out = screen("age 34, gender female, main language German", attribution="participant", kind="association_or_framing")
        self.assertEqual(out.rejection_code, "span_in_transcript_header")
        out = screen("name unclear at 12:40, ASR corrected", attribution="participant", kind="association_or_framing")
        self.assertEqual(out.rejection_code, "interviewer_utterance")

    def test_literature_unaffected(self):
        lit = sv.prepare_chunk("d", 0, [1, 1], "Interviewer: is a role in ethnography. Cults recruit by deception, critics argue.", "literature")
        cand, _, _ = schema.validate_candidate(_candidate("Cults recruit by deception, critics argue"))
        self.assertIsNone(sv.screen_candidate(cand, lit).rejection_code)


class EmbedV2Test(unittest.TestCase):
    def test_embed_run_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "v2" / "literature" / "run_t"
            run_dir.mkdir(parents=True)
            rows = [
                {"document_id": "docA", "chunk_index": 0, "page_range": [1, 1], "candidate_rank": 1,
                 "verbatim_expression": "Cults isolate\nmembers", "embedding_text": "Cults isolate members",
                 "text_transform": "newline_to_space", "entity_anchors": ["Peoples Temple"], "claim_mode": "direct_statement",
                 "epistemic_status": "asserted", "attribution": "author", "context_window": "x", "judge_accepted": True},
                {"document_id": "docB", "chunk_index": 2, "page_range": [3, 3], "candidate_rank": 1,
                 "verbatim_expression": "the guru", "embedding_text": "the guru", "text_transform": "none",
                 "entity_anchors": [], "claim_mode": "direct_statement", "epistemic_status": "asserted",
                 "attribution": "author", "context_window": "y", "judge_accepted": True},
            ]
            with open(run_dir / "expressions_v2.jsonl", "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            (run_dir / "documents_done.txt").write_text("docA\ndocB\n")
            calls = []

            def fake_embed(host, model, texts):
                calls.append(list(texts))
                return [[float(len(t)), 1.0] for t in texts]

            args = SimpleNamespace(corpus="literature", run_tag="t", embed_model="stub-embed", ollama_host="h",
                                   limit=1, out_root=root / "v2")
            out = embed_v2.run(args, embed=fake_embed, check=lambda h: None)
            recs = [json.loads(l) for l in open(out, encoding="utf-8")]
            self.assertEqual(len(recs), 1)
            self.assertEqual(recs[0]["source_quote"], "Cults isolate\nmembers")
            self.assertEqual(recs[0]["embedding_vector"], [21.0, 1.0])
            self.assertEqual(list(recs[0]["entity_anchor_vectors"]), ["Peoples Temple"])
            self.assertEqual(recs[0]["embedding_model"], "stub-embed")
            args.limit = None
            embed_v2.run(args, embed=fake_embed, check=lambda h: None)
            recs = [json.loads(l) for l in open(out, encoding="utf-8")]
            self.assertEqual([r["document_id"] for r in recs], ["docA", "docB"])
            summary = json.loads((run_dir / "embed_summary.json").read_text())
            self.assertTrue(summary["complete"])
            embed_v2.run(args, embed=fake_embed, check=lambda h: None)  # nothing to do, no duplicates
            self.assertEqual(len([l for l in open(out, encoding="utf-8")]), 2)
            args.embed_model = "other"
            with self.assertRaises(SystemExit):
                embed_v2.run(args, embed=fake_embed, check=lambda h: None)


if __name__ == "__main__":
    unittest.main()
