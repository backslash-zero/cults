"""Unit tests for qa_interview_entity_trace.py's pure resolution functions.

Run from thesis/corpus/:  python -m unittest thesis_corpus.tests.test_qa_interview_entity_trace -v
"""
import unittest

from thesis_corpus import qa_interview_entity_trace as qa


class TestResolveExpression(unittest.TestCase):
    def test_ordinary_expression_pooled(self):
        self.assertEqual(qa.resolve_expression("some-doc", 0), "pooled")

    def test_manually_excluded_key_flagged(self):
        # "b3-aug23-1213:0" is the real, permanently-excluded "Sept?" case.
        self.assertEqual(qa.resolve_expression("b3-aug23-1213", 0), "manually_excluded")


class TestResolveDomainTerm(unittest.TestCase):
    def test_lowercase_term_filtered_not_named_entity_shaped(self):
        result = qa.resolve_domain_term("cult", {"cult"}, set(), {})
        self.assertEqual(result["status"], "filtered_not_named_entity_shaped")

    def test_named_entity_not_in_embedded_set_filtered_not_embedded(self):
        result = qa.resolve_domain_term("Scientology", set(), set(), {})
        self.assertEqual(result["status"], "filtered_not_embedded")

    def test_cited_author_surname_filtered(self):
        result = qa.resolve_domain_term("Palmer", {"Palmer"}, {"palmer"}, {})
        self.assertEqual(result["status"], "filtered_cited_author_surname")
        self.assertEqual(result["resolved_key"], "palmer")

    def test_below_mention_threshold_filtered(self):
        result = qa.resolve_domain_term("Obscure Group", {"Obscure Group"}, set(), {})
        self.assertEqual(result["status"], "filtered_below_mention_threshold")
        self.assertEqual(result["resolved_key"], "obscure group")

    def test_resolved_to_pooled_entity(self):
        emergent = {"scientology": {"label": "scientology", "mention_distribution": {"literature": 3, "interviews": 1}}}
        result = qa.resolve_domain_term("Scientology", {"Scientology"}, set(), emergent)
        self.assertEqual(result["status"], "pooled")
        self.assertEqual(result["resolved_label"], "scientology")
        self.assertEqual(result["mention_distribution"], {"literature": 3, "interviews": 1})


class TestBuildTraceRows(unittest.TestCase):
    def test_interleaves_expressions_and_their_chunks_domain_terms(self):
        expressions_by_doc = {
            "doc1": [
                {"chunk_index": 0, "verbatim_expression": "Tomato cult!", "attribution": "participant",
                 "claim_mode": "direct_statement", "epistemic_status": "asserted"},
            ],
        }
        domain_terms_by_doc_chunk = {("doc1", 0): ["Tomato cult", "cult"]}
        embedded_terms = {"Tomato cult"}
        emergent = {"tomato cult": {"label": "tomato cult", "mention_distribution": {"interviews": 1}}}

        rows = qa.build_trace_rows(expressions_by_doc, domain_terms_by_doc_chunk, embedded_terms, set(), emergent)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["item_type"], "expression")
        self.assertEqual(rows[0]["resolution_status"], "pooled")
        self.assertEqual(rows[1]["item_type"], "domain_term")
        self.assertEqual(rows[1]["text"], "Tomato cult")
        self.assertEqual(rows[1]["resolution_status"], "pooled")
        self.assertEqual(rows[1]["resolved_entity"], "tomato cult")
        self.assertEqual(rows[2]["text"], "cult")
        self.assertEqual(rows[2]["resolution_status"], "filtered_not_named_entity_shaped")


if __name__ == "__main__":
    unittest.main()
