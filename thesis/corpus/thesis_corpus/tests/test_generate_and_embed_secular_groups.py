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

    def test_rejects_lines_longer_than_max_name_length(self):
        # Padded with enough good lines that one bad one stays under the
        # rejection-fraction safety net (tested separately, below).
        good = [f"Group {i}" for i in range(8)]
        raw = "\n".join(good[:4] + [("x" * 100)] + good[4:])
        self.assertEqual(gesg.parse_group_list(raw), good)

    def test_rejects_markdown_and_code_artifacts(self):
        good = [f"Group {i}" for i in range(8)]
        raw = "\n".join(good[:4] + ["```python", "### Header"] + good[4:])
        self.assertEqual(gesg.parse_group_list(raw), good)

    def test_raises_when_the_model_goes_completely_off_topic(self):
        # A real failure, reproduced: a live run asked for a name list and
        # instead got a full markdown tutorial about scraping Reddit
        # subreddits (headers, code blocks, a fake data table, a
        # follow-up question) -- mostly-rejected lines must raise, not
        # silently return whatever few short lines happened to survive.
        raw = "\n".join([
            "To create a list of the top 100 most popular Reddit subreddits, you can use a combination of tools.",
            "## Step-by-Step Guide",
            "### 1. Understand What Makes a Subreddit Popular",
            "```python",
            "import praw",
            "reddit = praw.Reddit(client_id='YOUR_CLIENT_ID')",
            "```",
            "| Rank | Subreddit | Subscribers |",
            "|------|-----------|-------------|",
            "| 1 | r/worldnews | 10M+ |",
            "Would you like help writing a Python script to fetch and rank subreddits?",
        ])
        with self.assertRaises(ValueError):
            gesg.parse_group_list(raw)


class TestDefaultChatTimeout(unittest.TestCase):
    def test_small_n_uses_the_120s_floor(self):
        self.assertEqual(gesg.default_chat_timeout(10), 120.0)

    def test_large_n_scales_up(self):
        # real case that timed out: n=100 against qwen3:8b at a fixed 120s
        self.assertEqual(gesg.default_chat_timeout(100), 600.0)
        self.assertGreater(gesg.default_chat_timeout(100), 120.0)


if __name__ == "__main__":
    unittest.main()
