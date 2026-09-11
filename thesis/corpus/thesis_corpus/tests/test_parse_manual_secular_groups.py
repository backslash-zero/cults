"""Unit tests for thesis_corpus.parse_manual_secular_groups.

Run from thesis/corpus/:  python -m unittest thesis_corpus.tests.test_parse_manual_secular_groups -v
"""
import tempfile
import unittest
from pathlib import Path

from thesis_corpus import parse_manual_secular_groups as pmsg


class TestParseManualGroupList(unittest.TestCase):
    def test_skips_header_lines_and_blanks(self):
        raw = "prompt: give me 100 groups\nmodel: qwen3:8b\n\nGreenpeace (action climatique)\n"
        self.assertEqual(pmsg.parse_manual_group_list(raw), ["Greenpeace"])

    def test_skips_category_headers_with_no_separator(self):
        # Real lines from the actual source files -- confirmed by direct
        # inspection that every such heading has neither "(" nor a spaced
        # dash anywhere in it, unlike every real entry.
        raw = "\n".join([
            "Humanist & Secular Organizations",
            "American Humanist Association (AHA) – Promotes humanism and secular ethics.",
            "1. Environnement et durabilité",
            "Greenpeace (action climatique)",
        ])
        self.assertEqual(
            pmsg.parse_manual_group_list(raw),
            ["American Humanist Association", "Greenpeace"],
        )

    def test_extracts_name_before_en_dash_description(self):
        raw = "Freedom From Religion Foundation (FFRF) – Advocates for the separation of church and state."
        self.assertEqual(pmsg.parse_manual_group_list(raw), ["Freedom From Religion Foundation"])

    def test_extracts_name_before_parenthetical_description_no_dash(self):
        raw = "Yoga Alliance (pratique du yoga)"
        self.assertEqual(pmsg.parse_manual_group_list(raw), ["Yoga Alliance"])

    def test_domain_name_with_a_leading_number_is_not_mistaken_for_a_numbered_list_marker(self):
        # A real failure, reproduced: "350.org" (a climate org's actual
        # domain-name-as-name) was being stripped down to "org" because
        # the numbering-marker regex matched "350." with no required
        # space after it, same as it would for "1. Item".
        raw = "350.org – Climate action."
        self.assertEqual(pmsg.parse_manual_group_list(raw), ["350.org"])

    def test_hyphenated_name_without_spaces_is_not_mistaken_for_a_separator(self):
        # "Anti-Defamation League" has a bare hyphen with no surrounding
        # spaces -- must not be treated as the " - " description separator.
        raw = "The Anti-Defamation League (ADL) – Combats anti-Semitism and hate."
        self.assertEqual(pmsg.parse_manual_group_list(raw), ["The Anti-Defamation League"])


class TestLoadAllGroupNames(unittest.TestCase):
    def test_dedups_case_insensitively_across_files_keeping_first_seen_casing(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "a.txt").write_text("Greenpeace (action climatique)\nMeetup.com (loisirs)\n", encoding="utf-8")
            (d / "b.txt").write_text("GREENPEACE (env) – dup.\nRed Cross (aide) – urgences.\n", encoding="utf-8")
            names = pmsg.load_all_group_names(d)
        self.assertEqual(names, ["Greenpeace", "Meetup.com", "Red Cross"])

    def test_ignores_non_txt_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "a.txt").write_text("Greenpeace (env)\n", encoding="utf-8")
            (d / "readme.md").write_text("Should Not Be Read (x)\n", encoding="utf-8")
            names = pmsg.load_all_group_names(d)
        self.assertEqual(names, ["Greenpeace"])


if __name__ == "__main__":
    unittest.main()
