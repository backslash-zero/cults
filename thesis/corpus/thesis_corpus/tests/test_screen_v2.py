"""Unit tests for thesis_corpus.screen_v2 -- fixtures agreed in the v2 plan.

Run from thesis/corpus/:  python -m unittest thesis_corpus.tests.test_screen_v2 -v
"""
import unittest

from thesis_corpus import screen_v2 as sv
from thesis_corpus import extraction_v2_schema as schema

BASE = dict(
    attribution="author", claim_mode="direct_statement", epistemic_status="asserted",
    entity_anchors=[], self_contained=True, cult_relevant=True,
    textually_intelligible=True, single_coherent_expression=True, relevance_note="test",
)


def cand(text, kind="claim", **overrides):
    raw = {**BASE, "verbatim_expression": text, "expression_kind": kind, **overrides}
    return raw


def ctx(text, flags=(), regions=(), source="literature", document_id="doc", chunk_index=0):
    return sv.prepare_chunk(document_id, chunk_index, [1, 1], text, source,
                            document_integrity_flags=list(flags), corrupted_line_texts=frozenset(regions))


def screen(text, chunk_text, **kw):
    c = ctx(chunk_text, flags=kw.pop("flags", ()), regions=kw.pop("regions", ()))
    validated, code, detail = schema.validate_candidate(cand(text, **kw))
    assert validated is not None, (code, detail)
    return sv.screen_candidate(validated, c)


CULT_CONTEXT = ("Interviewer: When you hear the word cult, what comes to your mind?\n"
                "Interviewee: {span} That is my first image. Then maybe rituals.")
PLAIN_CONTEXT = "The weather report said rain. {span} The market opened at nine."


class RejectFixtures(unittest.TestCase):
    def test_contextless_acknowledgement(self):
        chunk = "So, does President Bush recognize the limitations of government? Well, perhaps to some extent. He proposes to prohibit it."
        out = screen("Well, perhaps to some extent.", chunk)
        self.assertEqual(out.rejection_code, "short_fragment_no_referent")

    def test_truncated_name(self):
        chunk = "la formation d'une emprise psychologique via l'idéalisation du leader. François-Xavier Bauduin, sociologue, présente ses recherches."
        out = screen("François-Xavier B", chunk)
        self.assertEqual(out.rejection_code, "span_cuts_word")  # "B" is cut from "Bauduin"

    def test_lone_capital_end_when_not_cut(self):
        chunk = "The report was signed by François-Xavier B and filed."
        out = screen("The report was signed by François-Xavier B", chunk)
        self.assertEqual(out.rejection_code, "dangling_boundary")

    def test_ligature_substitution_in_flagged_document(self):
        chunk = ("One conclusion is obvious. Yet another conclusion is that it might be more di Y cult for members "
                 "in groups with a softer style of authority to see through the veil of power.")
        span = "it might be more di Y cult for members in groups with a softer style of authority to see through the veil of power"
        out = screen(span, chunk, flags=("ligature_substitution",))
        self.assertEqual(out.rejection_code, "integrity_ligature_substitution")
        # Same text in an unflagged document: only a flag.
        out2 = screen(span, chunk)
        self.assertIsNone(out2.rejection_code)
        self.assertIn("isolated_capital", out2.flags)

    def test_run_in_heading(self):
        chunk = ("remain as members as of 1990.\nThe Course of Growth. While the growth of cults has not been alarming, "
                 "it has been noticeable and raised questions of its causes.")
        out = screen("The Course of Growth.", chunk)
        self.assertEqual(out.rejection_code, "heading_or_scaffold")

    def test_all_caps_heading_on_own_line(self):
        chunk = "text before.\nCHAPTER THREE\nThe cultic milieu is a concept introduced by Campbell."
        out = screen("CHAPTER THREE", chunk)
        self.assertEqual(out.rejection_code, "heading_or_scaffold")
        # Heading keyword on its own line rejects even with a domain term in it.
        chunk2 = "text before.\nPART II THE CULTIC MILIEU\nCampbell introduced the term."
        out2 = screen("PART II THE CULTIC MILIEU", chunk2)
        self.assertEqual(out2.rejection_code, "heading_or_scaffold")
        # All-caps with a domain term, embedded in a sentence: retained, no rejection on capitalization.
        chunk3 = "Critics wrote that DESTRUCTIVE CULTS recruit by deception, a claim Barker disputed."
        out3 = screen("DESTRUCTIVE CULTS recruit by deception", chunk3)
        self.assertIsNone(out3.rejection_code, out3.detail)

    def test_cipher_region(self):
        line = "The ancient citadel and urban center of Tiwanaku iŶ Boliǀia's highlaŶd plateau the ǁhite ŵiŶoƌitǇ ruled."
        chunk = "A clean sentence about cults.\n" + line
        out = screen("urban center of Tiwanaku", chunk, flags=("cmap_cipher",), regions=(line,))
        self.assertEqual(out.rejection_code, "integrity_cipher_region")

    def test_not_verbatim_paraphrase(self):
        chunk = "By demanding that the self be 'actualized' prior to reaching the 20s, society is handing children a double-edged sword."
        out = screen("Society handing children a double-edged sword by demanding self 'actualization' prior to 20s", chunk)
        self.assertEqual(out.rejection_code, "not_verbatim")

    def test_translation_is_not_verbatim(self):
        chunk = "Les techniques d'investigation mise en œuvre depuis 2009 par la CAIMADES se sont enrichies."
        out = screen("The investigative techniques implemented by CAIMADES since 2:09 have been enriched", chunk)
        self.assertEqual(out.rejection_code, "not_verbatim")

    def test_model_truncation_mid_word_is_caught(self):
        chunk = "There is no word that easily substitutes, so I will have to speak of rebellion when I really mean something that can be, and often is, acceptable."
        out = screen("There is no word that easily substitutes, so I will have to speak of rebellion when I really mean something that can be", chunk)
        self.assertEqual(out.rejection_code, "dangling_boundary")

    def test_meta_discourse(self):
        chunk = "Why the current interest has developed will be the subject of this chapter, which adopts a sociological approach."
        out = screen("Why the current interest has developed will be the subject of this chapter", chunk)
        self.assertEqual(out.rejection_code, "meta_discourse")

    def test_standalone_personal_name(self):
        chunk = "The volume was edited by Lorne L. Dawson and published in 2009 for readers interested in cults."
        out = screen("Lorne L. Dawson", chunk)
        self.assertEqual(out.rejection_code, "standalone_personal_name")

    def test_interviewer_rejected(self):
        chunk = 'Interviewer: When you hear the word "cult," what comes to your mind?\nInterviewee: A leader.'
        out = screen('When you hear the word "cult," what comes to your mind?', chunk, attribution="interviewer",
                     kind="question_or_reflection", claim_mode="question_or_reflection")
        self.assertEqual(out.rejection_code, "interviewer_utterance")

    def test_too_long(self):
        words = " ".join(["cult"] + ["word"] * 55)
        out = screen(words, "Intro. " + words + " End.")
        self.assertEqual(out.rejection_code, "too_long")

    def test_validation_flag_false(self):
        out = screen("sectes destructrices", "Il parle des sectes destructrices.", cult_relevant=False)
        self.assertEqual(out.rejection_code, "validation_flag_false")

    def test_citation_dominated(self):
        chunk = "cults (Barker 1984; Richardson 1993; Introvigne 2001) were studied."
        out = screen("cults (Barker 1984; Richardson 1993; Introvigne 2001)", chunk)
        self.assertEqual(out.rejection_code, "citation_dominated")


class RetainFixtures(unittest.TestCase):
    def test_tomato_cult(self):
        out = screen("Tomato cult!", CULT_CONTEXT.format(span="Tomato cult!"),
                     kind="association_or_framing", attribution="participant")
        self.assertIsNone(out.rejection_code, out.detail)

    def test_la_scientologie(self):
        chunk = "Interviewer: un exemple ?\nInterviewee: la Scientologie, clairement."
        out = screen("la Scientologie", chunk, kind="example_or_named_group", attribution="participant")
        self.assertIsNone(out.rejection_code, out.detail)

    def test_sectes_destructrices(self):
        out = screen("sectes destructrices", "Le rapport évoque les sectes destructrices et leurs adeptes.")
        self.assertIsNone(out.rejection_code, out.detail)

    def test_ideologie_martelee(self):
        chunk = "Interviewer: pourquoi ce groupe-là tu dirais que c'est une secte ?\nInterviewee: idéologie martelée très fort, ça implique un truc négatif."
        out = screen("idéologie martelée très fort", chunk, kind="association_or_framing", attribution="participant")
        self.assertIsNone(out.rejection_code, out.detail)
        self.assertIn("short_association", out.flags)

    def test_people_dressed_in_white_with_cult_context(self):
        out = screen("People dressed in white", CULT_CONTEXT.format(span="People dressed in white."),
                     kind="association_or_framing", attribution="participant")
        self.assertIsNone(out.rejection_code, out.detail)

    def test_people_dressed_in_white_without_cult_context(self):
        out = screen("People dressed in white", PLAIN_CONTEXT.format(span="People dressed in white."),
                     kind="association_or_framing", attribution="participant")
        self.assertEqual(out.rejection_code, "short_fragment_no_referent")

    def test_miviludes_all_caps_named_group(self):
        chunk = "Les signalements sont transmis. MIVILUDES\nLa mission publie un rapport sur les dérives sectaires."
        out = screen("MIVILUDES", chunk, kind="example_or_named_group", attribution="institution")
        self.assertIsNone(out.rejection_code, out.detail)

    def test_iskcon_inside_sentence(self):
        chunk = "In the 1980s ISKCON was accused of brainwashing its recruits by several courts."
        out = screen("ISKCON was accused of brainwashing its recruits", chunk)
        self.assertIsNone(out.rejection_code, out.detail)
        self.assertNotIn("possible_heading_or_acronym", out.flags)

    def test_untagged_all_caps_acronym_in_sentence_is_flag_only(self):
        chunk = "Members joined the NRM and later the CIA investigated the group for fraud."
        out = screen("CIA", chunk)
        # not on its own line -> flag only (S11 name rule: single all-caps token is not a Title-Case name)
        self.assertIsNone(out.rejection_code, out.detail)
        self.assertIn("possible_heading_or_acronym", out.flags)

    def test_regular_claim(self):
        chunk = "Bromley argues that behaviors used by NRMs to recruit and keep members are misleading at best and coercive at worst."
        out = screen("behaviors used by NRMs to recruit and keep members are misleading at best and coercive at worst", chunk)
        self.assertIsNone(out.rejection_code, out.detail)
        self.assertEqual(out.flags, [])

    def test_spaced_accent_is_flag_only(self):
        chunk = "Hervieu-Le ´ger's theories on Europe, secularization, and 'cults' play a central role in European sociology of religion."
        out = screen("Hervieu-Le ´ger's theories on Europe, secularization, and 'cults' play a central role in European sociology of religion", chunk)
        self.assertIsNone(out.rejection_code, out.detail)
        self.assertIn("spaced_accent", out.flags)

    def test_nbsp_and_greek_are_flags(self):
        chunk = "The term derives from the Greek θρησκεία and is applied to cults by scholars."
        out = screen("The term derives from the Greek θρησκεία and is applied to cults by scholars", chunk)
        self.assertIsNone(out.rejection_code, out.detail)
        self.assertIn("unusual_script", out.flags)
        self.assertIn("contains_nbsp", out.flags)


class ChunkLevelTests(unittest.TestCase):
    def test_validation_codes_and_cap_and_overlap(self):
        chunk = ("Cults are said to isolate members from their families. Cults are said to isolate members from their families. "
                 "Leaders demand total obedience from adepts. Recruitment relies on systematic deception, several critics argue. "
                 "The guru claims divine status. Members surrender their savings to the sect.")
        raws = [
            cand("Cults are said to isolate members from their families"),
            cand("said to isolate members from their families"),               # overlaps rank 1
            {"verbatim_expression": "Leaders demand total obedience from adepts", "expression_kind": "claim"},  # missing fields
            {**cand("Recruitment relies on systematic deception, several critics argue"), "self_contained": "true"},  # string bool
            cand("The guru claims divine status"),
            cand("Members surrender their savings to the sect"),
            cand("Leaders demand total obedience from adepts"),
            cand("Recruitment relies on systematic deception, several critics argue"),  # 5th valid -> cap
        ]
        c = ctx(chunk)
        seen = set()
        retained, rejected = sv.screen_chunk(raws, c, seen)
        self.assertEqual(len(retained), 4)
        codes = sorted(r["rejection_code"] for r in rejected)
        self.assertEqual(codes, sorted(["duplicate_or_overlapping_span", "missing_required_field",
                                        "non_boolean_validation_field", "exceeds_per_chunk_cap"]))
        self.assertEqual(len(raws), len(retained) + len(rejected))
        for r in retained:
            self.assertEqual(r["embedding_text"], r["verbatim_expression"])
            self.assertEqual(c.nfc_text[r["span_start"]:r["span_end"]], r["verbatim_expression"])
            self.assertEqual(sv.recheck_retained_record(r, c), [])
        # document-level duplicate across chunks
        c2 = ctx("Cults are said to isolate members from their families. More text about the sect.", chunk_index=1)
        retained2, rejected2 = sv.screen_chunk([cand("Cults are said to isolate members from their families")], c2, seen)
        self.assertEqual(retained2, [])
        self.assertEqual(rejected2[0]["rejection_code"], "duplicate_or_overlapping_span")

    def test_pre_screen(self):
        cfg = schema.PRE_SCREEN_BY_SOURCE["literature"]
        self.assertEqual(sv.pre_screen_chunk(ctx("too short"), cfg)[0], "chunk_too_short")
        bib = "\n".join([
            "Barker, Eileen. 1984. The Making of a Moonie. Oxford: Blackwell.",
            "Richardson, James T. 1993. Definitions of Cult. Review of Religious Research 34: 348-356.",
            "Introvigne, Massimo. 2001. Brainwashing Theories. Nova Religio 4: 1-20.",
            "Dawson, Lorne L. 2006. Comprehending Cults. Oxford: Oxford University Press.",
        ] + ["word"] * 0)
        bib_ctx = ctx(bib + " " + " ".join(["filler"] * 40))
        self.assertEqual(sv.pre_screen_chunk(bib_ctx, cfg)[0], "chunk_is_bibliography_or_index")
        self.assertEqual(sv.pre_screen_chunk(ctx(" ".join(["Cults are groups."] * 20)), cfg)[0], None)
        self.assertEqual(sv.pre_screen_chunk(ctx("short turn"), schema.PRE_SCREEN_BY_SOURCE["interviews"])[0], None)
        # Clarke 605 false positive: prose sentences starting with proper nouns and containing years.
        prose = "\n".join([
            "Several aspects of State Shinto posed difficult contradictions for adherents of new religious movements " * 2,
            "State Shinto shaped the suppression of religions through charges of lèse-majesté. In a celebrated event of 1891 a teacher refused to bow.",
            "The Peace Preservation Law of 1925, which gave the state wide-ranging powers to restrict speech and public assembly, was used against sects.",
        ])
        self.assertEqual(sv.pre_screen_chunk(ctx(prose), cfg)[0], None)
        # Clarke 120: a short "Further reading" list followed by a real entry body -> not skipped.
        entry = "\n".join([
            "Further reading",
            "Grayson, J.H. (1989) Korea: A Religious History, Oxford: Clarendon Press.",
            "Harvey, P. (1990) An Introduction to Buddhism. Teachings, History and Practices, Cambridge: University Press.",
            "PETER B.CLARKE",
            "CHAOS MAGICK",
            "Chaos Magick could be described as the union of traditional occult ideas with applied psychology. " * 2,
            "Chaos Magick is the creation of two magicians, Peter J.Carroll and Ray Sherwin, though Carroll is seen as its main theorist. " * 2,
        ])
        self.assertEqual(sv.pre_screen_chunk(ctx(entry), cfg)[0], None)
        # ...but a chunk that is almost entirely a reference list after the heading is skipped.
        ref_list = "\n".join(["Further reading"] + [f"Author{i}, A. ({1980 + i}) Some Title, London: Routledge." for i in range(6)])
        self.assertEqual(sv.pre_screen_chunk(ctx(ref_list + " " + " ".join(["x"] * 40)), cfg)[0], "chunk_is_bibliography_or_index")

    def test_domain_terms(self):
        c = ctx("La justice restaurative protège les citoyens contre l'emprise. Les sectes sont visées.")
        kept, dropped = sv.screen_domain_terms(["emprise", "Emprise", "brainwashing", ""], c)
        self.assertEqual(kept, ["emprise"])
        self.assertEqual([d["reason"] for d in dropped], ["term_not_in_chunk", "empty_or_non_string"])

    def test_nfc_recorded(self):
        c = ctx("dérive sectaire dans le groupe")
        self.assertEqual(c.nfc_text, "dérive sectaire dans le groupe")
        self.assertGreater(c.nfc_changed_codepoints, 0)


if __name__ == "__main__":
    unittest.main()
