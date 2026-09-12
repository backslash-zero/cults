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


class TestParseManualGroupListWithCells(unittest.TestCase):
    RAW = "\n".join([
        "prompt: four blocks",
        "model: qwen3:8b",
        "",
        "**Block A — widely known as cults**",
        "NXIVM – recruited women into a coercive master/slave hierarchy",
        "Anonymous (hacker collective) – decentralised, no coercion claim",
        "",
        "Block B — ordinary-sounding but coercive",
        "1. Amway – escalating financial demands on a downline",
        "Herbalife (aggressive recruitment quotas)",
        "Some Stray Heading With No Separator",
        "",
        "Block C — ordinary organizations",
        "Sierra Club – ordinary membership conservation body",
    ])

    def test_assigns_each_entry_to_its_block(self):
        out = pmsg.parse_manual_group_list_with_cells(self.RAW)
        self.assertEqual([(e["name"], e["cell"]) for e in out], [
            ("NXIVM", "A"),
            ("Anonymous (hacker collective)", "A"),
            ("Amway", "B"),
            ("Herbalife", "B"),
            ("Sierra Club", "C"),
        ])

    def test_keeps_the_description(self):
        out = pmsg.parse_manual_group_list_with_cells(self.RAW)
        by_name = {e["name"]: e["description"] for e in out}
        self.assertEqual(by_name["Amway"], "escalating financial demands on a downline")
        self.assertEqual(by_name["Herbalife"], "aggressive recruitment quotas")

    def test_prefers_the_dash_separator_so_a_name_keeps_its_own_parentheses(self):
        # The plain parser splits at "(" and would truncate this to
        # "Anonymous"; with descriptions in play the dash has to win.
        out = pmsg.parse_manual_group_list_with_cells(self.RAW)
        entry = next(e for e in out if e["name"].startswith("Anonymous"))
        self.assertEqual(entry["name"], "Anonymous (hacker collective)")
        self.assertEqual(entry["description"], "decentralised, no coercion claim")

    def test_skips_headers_numbering_and_separatorless_strays(self):
        out = pmsg.parse_manual_group_list_with_cells(self.RAW)
        names = [e["name"] for e in out]
        self.assertNotIn("Some Stray Heading With No Separator", names)
        self.assertIn("Amway", names)  # numbering stripped, not skipped

    def test_entries_before_any_block_header_get_a_none_cell(self):
        out = pmsg.parse_manual_group_list_with_cells("Orphan Group – no block yet")
        self.assertEqual(out[0]["cell"], None)


class TestBlockHeaderTolerance(unittest.TestCase):
    def test_accepts_the_ways_a_model_actually_decorates_a_heading(self):
        for header, expected in [
            ("Block A — widely known as cults", "A"),
            ("**Block A —**", "A"),
            ("### Block A:", "A"),
            ("BLOCK B - coercive", "B"),
            ("Bloc C", "C"),
            ("Block D.", "D"),
            ("> Block B", "B"),
        ]:
            with self.subTest(header=header):
                entries = pmsg.parse_manual_group_list_with_cells(f"{header}\nName – desc")
                self.assertEqual(entries[0]["cell"], expected)

    def test_skips_hash_comments_without_breaking_markdown_headers(self):
        # Hand-curated source files carry "#" provenance comments. Those must
        # not be embedded as group names, but "### Block A" must still parse
        # as a header -- so the comment check has to come after the header one.
        raw = "\n".join([
            "# construction rule: real named organizations only",
            "### Block B - coercive",
            "# this one is a comment, not an entry",
            "Herbalife - FTC settlement over income misrepresentation",
        ])
        entries = pmsg.parse_manual_group_list_with_cells(raw)
        self.assertEqual([e["name"] for e in entries], ["Herbalife"])
        self.assertEqual(entries[0]["cell"], "B")

    def test_rejects_a_bare_letter_heading_because_it_looks_like_an_entry(self):
        # "A — cults" is indistinguishable from "Name - description", so
        # treating it as a header would silently swallow a real entry.
        entries = pmsg.parse_manual_group_list_with_cells("A — cults\nName – desc")
        self.assertEqual(entries[0]["name"], "A")
        self.assertIsNone(entries[0]["cell"])


class TestDiagnoseEmptyParse(unittest.TestCase):
    """Guards the error message that replaced a dead end: a bare "No cell
    entries parsed" gave no way to tell a missing file from an unrecognised
    format."""

    def test_names_the_missing_directory(self):
        from thesis_corpus.embed_manual_secular_groups import diagnose_empty_parse
        msg = diagnose_empty_parse(Path("/nonexistent/coercive-control-groups"))
        self.assertIn("does not exist", msg)

    def test_says_when_only_ignored_extensions_are_present(self):
        from thesis_corpus.embed_manual_secular_groups import diagnose_empty_parse
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "PROMPT.md").write_text("instructions, not data", encoding="utf-8")
            msg = diagnose_empty_parse(d)
        self.assertIn("no .txt/.tx file is present", msg)
        self.assertIn("PROMPT.md", msg)

    def test_shows_the_offending_lines_when_a_txt_exists_but_nothing_matched(self):
        from thesis_corpus.embed_manual_secular_groups import diagnose_empty_parse
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "out.txt").write_text("Block A\nNoSeparatorHere\nAlsoNothing\n", encoding="utf-8")
            msg = diagnose_empty_parse(d)
        self.assertIn("no entry lines matched", msg)
        self.assertIn("NoSeparatorHere", msg)


class TestLoadAllCellEntries(unittest.TestCase):
    def test_keeps_the_same_name_in_two_different_cells(self):
        # A cell-assignment conflict is a finding to surface, not something
        # to silently resolve -- so dedup is on (cell, name), not name.
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "a.txt").write_text(
                "Block A — cults\nAmway – listed as a cult\n"
                "Block B — coercive\nAmway – listed as merely coercive\n", encoding="utf-8")
            out = pmsg.load_all_cell_entries(d)
        self.assertEqual([(e["name"], e["cell"]) for e in out], [("Amway", "A"), ("Amway", "B")])

    def test_dedups_exact_repeats_within_one_cell(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "a.txt").write_text(
                "Block B — coercive\nAmway – x\nAMWAY – y\nHerbalife – z\n", encoding="utf-8")
            out = pmsg.load_all_cell_entries(d)
        self.assertEqual([e["name"] for e in out], ["Amway", "Herbalife"])


if __name__ == "__main__":
    unittest.main()
