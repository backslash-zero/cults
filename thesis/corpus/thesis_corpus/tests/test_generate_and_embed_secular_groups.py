"""Unit tests for thesis_corpus.generate_and_embed_secular_groups.

Only tests parse_group_list -- the one pure function in this module; the
chat/embed calls need a live Ollama server (unreachable on this machine,
see module docstring) and are exercised for real on the Windows machine.

Run from thesis/corpus/:  python -m unittest thesis_corpus.tests.test_generate_and_embed_secular_groups -v
"""
import unittest

from thesis_corpus import generate_and_embed_secular_groups as gesg


class TestParseGroupList(unittest.TestCase):
    def test_plain_lines(self):
        raw = "Amway\nCrossFit\nNXIVM"
        self.assertEqual(gesg.parse_group_list(raw), ["Amway", "CrossFit", "NXIVM"])

    def test_strips_numbered_list_markers(self):
        raw = "1. Amway\n2. CrossFit\n3) NXIVM"
        self.assertEqual(gesg.parse_group_list(raw), ["Amway", "CrossFit", "NXIVM"])

    def test_strips_bullet_markers(self):
        raw = "- Amway\n* CrossFit"
        self.assertEqual(gesg.parse_group_list(raw), ["Amway", "CrossFit"])

    def test_strips_surrounding_quotes_and_blank_lines(self):
        raw = '"Amway"\n\n\'CrossFit\'\n'
        self.assertEqual(gesg.parse_group_list(raw), ["Amway", "CrossFit"])

    def test_empty_input_returns_empty_list(self):
        self.assertEqual(gesg.parse_group_list(""), [])


if __name__ == "__main__":
    unittest.main()
