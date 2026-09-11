"""Unit tests for thesis_corpus.generate_and_embed_secular_groups.

Only tests parse_group_list -- the one pure function in this module; the
chat/embed calls need a live Ollama server (unreachable on this machine,
see module docstring) and are exercised for real on the Windows machine.

Run from thesis/corpus/:  python -m unittest thesis_corpus.tests.test_generate_and_embed_secular_groups -v
"""
import unittest
from unittest import mock

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

    def test_raises_on_generic_clarifying_question_non_answer(self):
        # A real failure, reproduced verbatim: a live run against qwen3:8b
        # returned exactly this -- four short, markdown-free lines that
        # none of the length/artifact checks caught -- instead of
        # attempting the task at all. 4/4 rejected must raise.
        raw = "\n".join([
            "Are you asking about a specific topic?",
            "Do you need help with a task or problem?",
            "Are you looking for information or guidance?",
            "Let me know, and I'll be happy to assist!",
        ])
        with self.assertRaises(ValueError):
            gesg.parse_group_list(raw)

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


class TestGenerateGroupNamesWithRetries(unittest.TestCase):
    def test_returns_names_on_first_try_when_good(self):
        good_raw = "\n".join([f"Group {i}" for i in range(10)])
        with mock.patch.object(gesg, "generate_group_names_raw", return_value=good_raw) as m:
            names = gesg.generate_group_names_with_retries("host", "model", n=10, max_attempts=3)
        self.assertEqual(len(names), 10)
        m.assert_called_once()

    def test_retries_and_recovers_from_a_transient_non_answer(self):
        # First attempt: the real "only 4 names" failure (plausible-looking
        # but far too few relative to n=10, even though none individually
        # trip the length/markdown checks) -- MIN_NAMES_FRACTION must catch
        # this even when parse_group_list itself doesn't raise.
        bad_raw = "\n".join(["Group A", "Group B"])
        good_raw = "\n".join([f"Group {i}" for i in range(10)])
        with mock.patch.object(gesg, "generate_group_names_raw", side_effect=[bad_raw, good_raw]) as m:
            names = gesg.generate_group_names_with_retries("host", "model", n=10, max_attempts=3)
        self.assertEqual(len(names), 10)
        self.assertEqual(m.call_count, 2)

    def test_raises_after_exhausting_all_attempts(self):
        bad_raw = "\n".join(["Group A", "Group B"])
        with mock.patch.object(gesg, "generate_group_names_raw", return_value=bad_raw) as m:
            with self.assertRaises(RuntimeError):
                gesg.generate_group_names_with_retries("host", "model", n=10, max_attempts=3)
        self.assertEqual(m.call_count, 3)

    def test_writes_raw_response_to_out_dir_on_every_attempt(self):
        import tempfile
        from pathlib import Path

        bad_raw = "Are you asking about a specific topic?"
        good_raw = "\n".join([f"Group {i}" for i in range(10)])
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            with mock.patch.object(gesg, "generate_group_names_raw", side_effect=[bad_raw, good_raw]):
                gesg.generate_group_names_with_retries("host", "model", n=10, max_attempts=3, out_dir=out_dir)
            raw_path = out_dir / "generated_secular_groups_raw_response.txt"
            self.assertEqual(raw_path.read_text(encoding="utf-8"), good_raw)


if __name__ == "__main__":
    unittest.main()
