"""Unit tests for thesis_corpus.text_integrity (stdlib unittest; no Ollama).

Fixtures are the exact strings surfaced by the 2026-09-08 fidelity audit
and by tracing them back to Stage-1 pages.jsonl.

Run from thesis/corpus/:  python -m unittest thesis_corpus.tests.test_text_integrity -v
"""
import unittest

from thesis_corpus import text_integrity as ti

LALICH_LIGATURE = "it might be more di Y cult for members in groups with a softer style of authority"
LALICH_LIGATURE_2 = "a soft-sell approach tends to have a more W rmly binding e V ect."
TOMKINS_KERNING = "he moved to the W est and then to W estminster"
OXFORD_ACCENT = "Hervieu-Le ´ger is not what, in certain quarters, would be called a 'cult apologist'"
CARD_CIPHER = ("The ancient citadel and urban center of Tiwanaku (c.300 -ϭϭϬϬADͿ iŶ Boliǀia's "
               "highlaŶd plateau deployed by both Boliǀia's ǁhite ŵiŶoƌitǇ aŶd its iŶdigeŶous ŵajoƌitǇ")
PUA_RUN = "  the press"
CLEAN_FR = "Les dérives sectaires en matière de santé et de bien-être ne se résument pas à des pratiques de soins"
GREEK_QUOTE = "the term derives from the Greek θρησκεία (thrēskeia), meaning worship"
PORTUGUESE_OK = "LEGIÃO DA BOA VONTADE is a Brazilian movement"


class NfcTests(unittest.TestCase):
    def test_nfc_noop_on_nfc_text(self):
        text, changed = ti.nfc_normalize(CLEAN_FR)
        self.assertEqual(text, CLEAN_FR)
        self.assertEqual(changed, 0)

    def test_nfc_counts_changed_codepoints(self):
        decomposed = "dérive"  # e + combining acute
        text, changed = ti.nfc_normalize(decomposed)
        self.assertEqual(text, "dérive")
        self.assertGreaterEqual(changed, 1)
        self.assertNotEqual(len(text), len(decomposed))


class CharacterClassTests(unittest.TestCase):
    def test_clean_text_has_no_flags(self):
        counts = ti.count_character_classes(CLEAN_FR)
        for key in ti.CHARACTER_CLASS_KEYS:
            self.assertEqual(counts[key], 0, key)

    def test_ligature_substitution_pattern(self):
        self.assertEqual(ti.count_character_classes(LALICH_LIGATURE)["ligature_pattern"], 1)
        self.assertEqual(ti.count_character_classes(LALICH_LIGATURE_2)["ligature_pattern"], 2)

    def test_kerning_split_also_matches_pattern_but_is_a_document_level_call(self):
        # "the W est" matches the same regex; only the document rate decides.
        self.assertEqual(ti.count_character_classes(TOMKINS_KERNING)["ligature_pattern"], 2)

    def test_isolated_capital_excludes_I_and_A(self):
        self.assertEqual(ti.count_character_classes("something which I call doubling")["isolated_capital"], 0)
        self.assertEqual(ti.count_character_classes("in a group")["isolated_capital"], 0)

    def test_spaced_accent(self):
        self.assertEqual(ti.count_character_classes(OXFORD_ACCENT)["spaced_accent"], 1)
        self.assertEqual(ti.count_character_classes("Hervieu-Léger")["spaced_accent"], 0)

    def test_cipher_text_mixed_script_tokens(self):
        counts = ti.count_character_classes(CARD_CIPHER)
        self.assertGreaterEqual(counts["mixed_script_tokens"], 6)
        self.assertGreater(counts["latin_ext_b_letters"], 0)
        self.assertGreater(counts["greek_letters"], 0)

    def test_private_use_run(self):
        counts = ti.count_character_classes(PUA_RUN)
        self.assertEqual(counts["private_use"], 11)
        self.assertEqual(counts["private_use_max_run"], 9)

    def test_mojibake_sequences(self):
        self.assertEqual(ti.count_character_classes("dÃ©rive sectaire â€“ emprise")["mojibake_seq"], 2)
        self.assertEqual(ti.count_character_classes(PORTUGUESE_OK)["mojibake_seq"], 0)

    def test_replacement_and_control(self):
        self.assertEqual(ti.count_character_classes("cult�")["replacement_char"], 1)
        self.assertEqual(ti.count_character_classes("cult\x0bsect")["control_char"], 1)
        self.assertEqual(ti.count_character_classes("cult\tsect\nx")["control_char"], 0)

    def test_residue_counts(self):
        counts = ti.count_character_classes("a b c­d re-\nligious")
        self.assertEqual(counts["nbsp"], 1)
        self.assertEqual(counts["soft_hyphen"], 1)
        self.assertEqual(counts["hyphen_linebreak"], 1)


class CipherLineTests(unittest.TestCase):
    def test_card_line_flagged(self):
        flagged = ti.find_cipher_lines("clean line about cults here\n" + CARD_CIPHER)
        self.assertEqual([f[0] for f in flagged], [1])
        self.assertGreater(flagged[0][1], 0.05)

    def test_low_density_paragraph_flagged_by_mixed_tokens(self):
        # Real Card 2019 lines are paragraph-long with only 1-3% suspicious
        # letters; the mixed-token trigger must catch them.
        # (Card's cipher also uses Latin Extended-A letters such as Ŷ and ŵ;
        # those are deliberately not counted -- see text_integrity.py -- so
        # the fixture uses the Extended-B glyphs ǀ ǁ ƌ Ǉ that do trigger.)
        clean = "The ancient citadel and urban center of Tiwanaku is a major site. " * 8
        line = clean + "in Boliǀia's plateau the ǁhite minoƌitǇ studied the site."
        flagged = ti.find_cipher_lines(line)
        self.assertEqual(len(flagged), 1)
        self.assertLess(flagged[0][1], 0.05)
        self.assertGreaterEqual(flagged[0][2], 2)

    def test_latin_extended_a_is_not_suspicious(self):
        self.assertEqual(ti.mixed_script_tokens("la mise en œuvre de Prabhupāda et Šaban"), [])

    def test_single_foreign_token_not_a_region(self):
        flagged = ti.find_cipher_lines("the Serbian word крст means cross in this context of a long clean sentence")
        self.assertEqual(flagged, [])

    def test_greek_quotation_line_flagged_at_line_level_only(self):
        # Line-level detection fires; document_flags decides it is not a cipher.
        flagged = ti.find_cipher_lines(GREEK_QUOTE)
        self.assertEqual(len(flagged), 1)
        counts = ti.count_character_classes(GREEK_QUOTE)
        hard, notes = ti.document_flags(counts, words=50_000, cipher_lines=1)
        self.assertNotIn("cmap_cipher", hard)
        self.assertIn("non_latin_letters_present", notes)


class DocumentFlagTests(unittest.TestCase):
    def _counts(self, **overrides):
        counts = {key: 0 for key in ti.CHARACTER_CLASS_KEYS}
        counts.update(overrides)
        return counts

    def test_ligature_rate_threshold_separates_lalich_from_tomkins(self):
        lalich, _ = ti.document_flags(self._counts(ligature_pattern=1075, isolated_capital=1075), 129_531, 0)
        self.assertIn("ligature_substitution", lalich)
        tomkins, notes = ti.document_flags(self._counts(ligature_pattern=108, isolated_capital=108), 106_751, 0)
        self.assertNotIn("ligature_substitution", tomkins)
        self.assertIn("spaced_capital_splits", notes)

    def test_cipher_flags(self):
        hard, _ = ti.document_flags(self._counts(mixed_script_tokens=40), 2_500, 5)
        self.assertIn("cmap_cipher", hard)
        hard, _ = ti.document_flags(self._counts(private_use=60, private_use_max_run=60), 100_000, 0)
        self.assertIn("private_use_cipher", hard)

    def test_detached_accents(self):
        hard, _ = ti.document_flags(self._counts(spaced_accent=328), 233_295, 0)
        self.assertIn("detached_accents", hard)
        hard, _ = ti.document_flags(self._counts(spaced_accent=2), 233_295, 0)
        self.assertNotIn("detached_accents", hard)

    def test_clean_document(self):
        hard, notes = ti.document_flags(self._counts(), 50_000, 0)
        self.assertEqual(hard, [])
        self.assertEqual(notes, [])


if __name__ == "__main__":
    unittest.main()
