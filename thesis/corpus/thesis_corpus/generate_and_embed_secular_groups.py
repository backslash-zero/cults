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
    "Return ONLY the list, one name per line, no numbering, no commentary, no explanation."
)


def parse_group_list(raw_text: str) -> list[str]:
    """Splits the model's line-per-name response into a clean list --
    strips numbering ("1. ", "- ", "* "), blank lines, and surrounding
    whitespace/quotes. Pure function, testable without Ollama."""
    names = []
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
        if line:
            names.append(line)
    return names


def generate_group_names(host: str, model: str, n: int, timeout: float = 120.0) -> list[str]:
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
            "options": {"temperature": 0.7},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    content = resp.json()["message"]["content"].strip()
    if "</think>" in content:
        content = content.rsplit("</think>", 1)[-1].strip()
    return parse_group_list(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST)
    parser.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--out-dir", type=Path, default=bss.PROCESSED_DIR / "analysis_raw" / "secular_groups")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from thesis_corpus.ollama_client import OllamaUnavailableError, check_available, embed_texts
    try:
        check_available(args.ollama_host)
    except OllamaUnavailableError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1)

    logger.info("Generating %d non-religious group names via %s...", args.n, args.chat_model)
    names = generate_group_names(args.ollama_host, args.chat_model, args.n)
    logger.info("Got %d names: %s", len(names), names)
    if not names:
        raise SystemExit("Model returned zero parseable names -- inspect its raw response before retrying.")

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
