"""Embeds the manually-curated non-religious-group names
(parse_manual_secular_groups.py's output) with bge-m3 and writes them in
the exact format find_secular_group_mentions.py's
load_generated_group_distances() already expects -- the C2 half of testing
the central Hypothesis (C1, the corpus-native half, is
find_secular_group_mentions.py itself, already run and reported in
Analysis/08_Secular_Groups_Hypothesis.md).

This REPLACES the live-generation approach (generate_and_embed_secular_groups.py):
three rounds of driving a local Ollama model to both generate AND embed the
list live, over a slow manual Windows-run/copy-back cycle, hit three
different failure modes (timeout, off-topic essay, generic non-answer).
The user instead generated the raw candidate names themselves (same
prompt, run directly, reviewed by eye) and pasted the output into
dictionaries/non-religious-groups/*.txt. Parsing that text is deterministic
and runs fine on this Mac (parse_manual_secular_groups.py); only embedding
still needs Ollama's bge-m3, which is unreachable here -- confirmed via
`curl 127.0.0.1:11434` refused, `ollama` binary not on PATH -- so this
script is code-only on this machine, run on the Windows/Ollama machine,
same cross-machine pattern as every other Ollama-dependent step in this
project.

Usage (on the Windows/Ollama machine, from thesis/corpus/):
    python -m thesis_corpus.embed_manual_secular_groups
Then copy processed/analysis_raw/secular_groups/generated_secular_groups.jsonl
back to this Mac at the same path, and re-run find_secular_group_mentions.py.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from thesis_corpus import build_shared_space as bss
from thesis_corpus.parse_manual_secular_groups import DEFAULT_SOURCE_DIR, load_all_group_names

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("thesis_corpus.embed_manual_secular_groups")

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_EMBED_MODEL = "bge-m3"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR,
                         help="Directory of manually-curated .txt/.tx name lists to parse and embed.")
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST)
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

    names = load_all_group_names(args.source_dir)
    if not names:
        raise SystemExit(f"No group names parsed from {args.source_dir} -- nothing to embed.")
    logger.info("Embedding %d manually-curated group names via %s...", len(names), args.embed_model)
    vectors = embed_texts(args.ollama_host, args.embed_model, names)

    out_path = args.out_dir / "generated_secular_groups.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for name, vector in zip(names, vectors):
            f.write(json.dumps({"label": name, "embedding_vector": vector}, ensure_ascii=False) + "\n")

    print(f"Done. {out_path} ({len(names)} names). Copy this file back to the Mac at the same "
          f"relative path, then run find_secular_group_mentions.py.")


if __name__ == "__main__":
    main()
