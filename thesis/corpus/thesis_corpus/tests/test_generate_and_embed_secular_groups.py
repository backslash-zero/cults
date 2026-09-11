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


class TestDefaultChatTimeout(unittest.TestCase):
    def test_small_n_uses_the_120s_floor(self):
        self.assertEqual(gesg.default_chat_timeout(10), 120.0)

    def test_large_n_scales_up(self):
        # real case that timed out: n=100 against qwen3:8b at a fixed 120s
        self.assertEqual(gesg.default_chat_timeout(100), 600.0)
        self.assertGreater(gesg.default_chat_timeout(100), 120.0)


if __name__ == "__main__":
    unittest.main()
