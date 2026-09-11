"""Generates a comparison list of well-known non-religious social groups
and structures via an LLM chat call, embeds each with bge-m3, and writes
raw 1024-d vectors -- the C2 half of testing the central Hypothesis
(the corpus-native half is find_secular_group_mentions.py, C1, already run
and reported in Analysis/08_Secular_Groups_Hypothesis.md).

CODE-ONLY ON THIS MACHINE. Ollama is unreachable on this Mac right now
(confirmed: curl to 127.0.0.1:11434 refused, `ollama` binary not on PATH).
Per the user's explicit choice, this script is written now and run later
on the Windows/Ollama machine -- same cross-machine pattern already
established earlier in this project (mirror_stage1.py, the interview
extraction/embedding workflow): write code here, run it there, copy the
output JSONL back to this path, then re-run analyze_secular_groups.py
(once written) to fold it into the comparison already done for C1.

Two Ollama calls, same primitives the rest of this codebase already uses
(ollama_client.py) -- no new dependency:
  1. A bare chat completion (same "think": False + strip-</think> pattern
     as ollama_client.translate_text, not the JSON-schema chat_structured
     path -- this just needs free text back, one group name per line, not
     a validated schema) asking for N well-known non-religious social
     groups/structures across varied domains (MLMs, corporate/workplace
     culture, fandoms, sports fanbases, wellness/fitness brands, startups,
     political movements, fraternities) -- deliberately varied domains, not
     just one category, so the resulting comparison set isn't accidentally
     narrow.
  2. ollama_client.embed_texts(host, "bge-m3", names) -- the exact same
     embedding call embed_v2.py/embed_domain_terms.py already use, so the
     resulting vectors are directly comparable to everything else in this
     corpus with zero additional transformation.

Usage (on the Windows/Ollama machine, from thesis/corpus/):
    python -m thesis_corpus.generate_and_embed_secular_groups --n 30
    python -m thesis_corpus.generate_and_embed_secular_groups --n 30 --ollama-host http://127.0.0.1:11434
Then copy processed/analysis_raw/secular_groups/generated_secular_groups.jsonl
back to this Mac at the same path, and re-run analyze_secular_groups.py.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import httpx

from thesis_corpus import build_shared_space as bss

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("thesis_corpus.generate_and_embed_secular_groups")

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_CHAT_MODEL = "qwen3:4b"
DEFAULT_EMBED_MODEL = "bge-m3"
DEFAULT_N = 30

SYSTEM_PROMPT = (
    "You are a knowledgeable assistant helping with academic research on group dynamics. "
    "List {n} well-known, real, NON-religious social groups, organizations, or structures -- "
    "spread across varied domains: multi-level-marketing (MLM) companies, corporate/workplace "
    "culture examples, fan communities/fandoms, sports fan bases, wellness or fitness brands, "
    "startups known for intense internal culture, fraternities/sororities, and secular "
    "political or ideological movements. Do NOT include any religious groups, churches, cults, "
    "or new religious movements -- every entry must be unambiguously secular. "
    "Return ONLY the list, one short name per line -- no numbering, no markdown formatting "
    "(no headers, no bold, no code blocks, no tables, no links), no commentary before or after "
    "the list, no follow-up questions. Each line must be a bare name only, a few words at most. "
    "Example of the exact format expected (do not reuse these specific examples in your answer):\n"
    "Amway\n"
    "CrossFit\n"
    "Anonymous (hacker collective)"
)

# A real name is always short. Checked directly against a real failure: a
# generation that went completely off-topic (wrote a markdown tutorial
# about scraping Reddit instead of a name list) produced "names" up to
# 270 characters and full of "```"/"|"/"http"/markdown headers -- this
# threshold and character check catch that shape of failure without
# needing to inspect content semantically.
MAX_NAME_LENGTH = 60
SUSPICIOUS_SUBSTRINGS = ("```", "http://", "https://", "| ", "###", "##")
MAX_REJECTED_FRACTION = 0.3  # if more than this fraction of lines are rejected, something is wrong -- fail loudly


def parse_group_list(raw_text: str) -> list[str]:
    """Splits the model's line-per-name response into a clean list --
    strips numbering ("1. ", "- ", "* "), blank lines, surrounding
    whitespace/quotes, and any line that's clearly not a bare name (too
    long, or containing markdown/code/link artifacts -- see
    MAX_NAME_LENGTH/SUSPICIOUS_SUBSTRINGS docstring above). Pure function,
    testable without Ollama."""
    names = []
    rejected = 0
    for line in raw_text.splitlines():
        line = line.strip().strip("\"'")
        if not line:
            continue
        # strip leading "1.", "1)", "-", "*" list markers
        for prefix_len in range(min(len(line), 4), 0, -1):
            prefix = line[:prefix_len]
            if prefix.rstrip(". )-*").isdigit() or prefix.strip() in ("-", "*"):
                line = line[prefix_len:].strip()
                break
        if not line:
            continue
        if len(line) > MAX_NAME_LENGTH or any(s in line for s in SUSPICIOUS_SUBSTRINGS):
            rejected += 1
            continue
        names.append(line)

    total = len(names) + rejected
    if total and rejected / total > MAX_REJECTED_FRACTION:
        raise ValueError(
            f"{rejected}/{total} lines rejected as not-a-bare-name (too long or markdown-like) -- "
            "the model likely went off-topic or ignored the format instructions rather than "
            "returning a clean list. Inspect the raw response before retrying, not just this "
            "filtered output."
        )
    return names


def default_chat_timeout(n: int) -> float:
    """Scales with `n` -- a fixed 120s default was enough for n=30 but timed
    out on a real n=100 request against qwen3:8b (ReadTimeout at exactly
    120s, mid-generation, not a crash). ~6s/name is a generous per-item
    budget for an 8B model generating a longer structured list on typical
    consumer hardware; still overridable via --chat-timeout for a slower
    machine or a much larger n."""
    return max(120.0, n * 6.0)


def generate_group_names_raw(host: str, model: str, n: int, timeout: float | None = None) -> str:
    """Just the chat call + </think>-stripping -- returns raw text, not yet
    parsed/validated. Split out from generate_group_names so main() can
    always save the raw response to disk BEFORE parsing, regardless of
    whether parsing then succeeds or raises -- the real failure this
    guards against (a response that's an off-topic markdown essay, not a
    list) is exactly the case where you most want the raw text preserved
    to inspect, not just a traceback."""
    if timeout is None:
        timeout = default_chat_timeout(n)
    resp = httpx.post(
        f"{host}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT.format(n=n)},
                {"role": "user", "content": f"List {n} non-religious social groups/structures now."},
            ],
            "stream": False,
            "think": False,
            # Low, not zero -- some variation across n items is fine/wanted,
            # but 0.7 measurably increased the risk of the model drifting
            # off-task into free-form writing on a real n=100 run.
            "options": {"temperature": 0.2},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    content = resp.json()["message"]["content"].strip()
    if "</think>" in content:
        content = content.rsplit("</think>", 1)[-1].strip()
    return content


def generate_group_names(host: str, model: str, n: int, timeout: float | None = None) -> list[str]:
    return parse_group_list(generate_group_names_raw(host, model, n, timeout))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST)
    parser.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--chat-timeout", type=float, default=None,
                         help="Seconds to wait for the chat call. Defaults to max(120, n*6) -- "
                              "a fixed 120s was enough at n=30 but timed out mid-generation at "
                              "n=100 against qwen3:8b. Raise this further for a slower machine.")
    parser.add_argument("--out-dir", type=Path, default=bss.PROCESSED_DIR / "analysis_raw" / "secular_groups")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from thesis_corpus.ollama_client import OllamaUnavailableError, check_available, embed_texts
    try:
        check_available(args.ollama_host)
    except OllamaUnavailableError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1)

    chat_timeout = args.chat_timeout if args.chat_timeout is not None else default_chat_timeout(args.n)
    logger.info("Generating %d non-religious group names via %s (timeout=%.0fs)...", args.n, args.chat_model, chat_timeout)
    raw_response = generate_group_names_raw(args.ollama_host, args.chat_model, args.n, timeout=chat_timeout)

    # Always saved BEFORE parsing -- if parse_group_list rejects the
    # response as off-topic/malformed (see its own docstring: this is
    # exactly what caught a real failure, a full markdown tutorial about
    # scraping Reddit instead of a name list), this file is what you
    # inspect, not a traceback with no record of what the model actually said.
    raw_path = args.out_dir / "generated_secular_groups_raw_response.txt"
    raw_path.write_text(raw_response, encoding="utf-8")
    logger.info("Raw model response saved to %s (%d chars)", raw_path, len(raw_response))

    try:
        names = parse_group_list(raw_response)
    except ValueError as e:
        print(f"ERROR: {e}\nRaw response is in {raw_path} -- read it before retrying.", file=sys.stderr)
        raise SystemExit(1)
    logger.info("Got %d names: %s", len(names), names)
    if not names:
        raise SystemExit(f"Model returned zero parseable names -- inspect {raw_path} before retrying.")

    logger.info("Embedding %d names via %s...", len(names), args.embed_model)
    vectors = embed_texts(args.ollama_host, args.embed_model, names)

    out_path = args.out_dir / "generated_secular_groups.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for name, vector in zip(names, vectors):
            f.write(json.dumps({"label": name, "embedding_vector": vector}, ensure_ascii=False) + "\n")

    print(f"Done. {out_path} ({len(names)} names). Copy this file back to the Mac at the same "
          f"relative path, then run analyze_secular_groups.py.")


if __name__ == "__main__":
    main()
