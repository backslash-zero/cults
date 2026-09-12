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
from thesis_corpus.parse_manual_secular_groups import (
    DEFAULT_SOURCE_DIR,
    load_all_cell_entries,
    load_all_group_names,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("thesis_corpus.embed_manual_secular_groups")

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_EMBED_MODEL = "bge-m3"


def diagnose_empty_parse(source_dir: Path) -> str:
    """Says WHY nothing parsed, rather than just that nothing did.

    Written after a real dead end: the run reported "No cell entries parsed"
    and gave no way to tell from the message whether the source file was
    missing, had the wrong extension, or had headers the parser didn't
    recognise. Each of those needs a different fix, so each gets named
    here."""
    if not source_dir.exists():
        return f"{source_dir} does not exist. Create it and save the model's raw output there as a .txt file."

    listing = sorted(p for p in source_dir.iterdir() if p.is_file())
    parsed = [p for p in listing if p.suffix in (".txt", ".tx")]
    ignored = [p for p in listing if p.suffix not in (".txt", ".tx")]

    lines = [f"No cell entries parsed from {source_dir}.", ""]
    lines.append(f"Files the parser READS (.txt/.tx): {len(parsed)}")
    for p in parsed:
        lines.append(f"  {p.name} ({p.stat().st_size} bytes)")
    lines.append(f"Files IGNORED (wrong extension): {len(ignored)}")
    for p in ignored:
        lines.append(f"  {p.name}")

    if not parsed:
        lines += [
            "",
            "=> Nothing to parse: no .txt/.tx file is present.",
            "   The prompt has to be RUN and its raw output SAVED here first:",
            f"     1. run the prompt in {source_dir / 'PROMPT.md'}",
            f"     2. save the model's raw reply as e.g. {source_dir / 'coercive-groups-1.txt'}",
            "     3. re-run this command",
            "   (PROMPT.md itself is ignored on purpose -- it is the instructions, not the data.)",
        ]
        return "\n".join(lines)

    lines += ["", "=> Files are present but no entry lines matched. First 8 lines of each:"]
    for p in parsed:
        lines.append(f"  --- {p.name} ---")
        for raw_line in p.read_text(encoding="utf-8", errors="replace").splitlines()[:8]:
            lines.append(f"    {raw_line[:100]!r}")
    lines += [
        "",
        "   An entry must be 'Name - description' on one line (hyphen, en dash or em dash with",
        "   spaces around it), or 'Name (description)'. A line with no separator is treated as a",
        "   heading and skipped. Check the lines above against that.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR,
                         help="Directory of manually-curated .txt/.tx name lists to parse and embed.")
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--out-dir", type=Path, default=bss.PROCESSED_DIR / "analysis_raw" / "secular_groups")
    parser.add_argument("--out-name", default="generated_secular_groups.jsonl",
                         help="Output filename inside --out-dir. Override so a second list can sit "
                              "beside the first instead of overwriting it.")
    parser.add_argument("--cells", action="store_true",
                         help="Parse the 4-cell coercive-control format (Block A..D headers, "
                              "'Name - description' entries) instead of a plain name list. Writes a "
                              "`cell` and `description` field per row, and -- unless "
                              "--names-only is set -- a second JSONL of the DESCRIPTIONS embedded "
                              "separately, since bare names carry no information about conduct "
                              "(see Analysis/10).")
    parser.add_argument("--names-only", action="store_true",
                         help="With --cells: skip the description embedding pass.")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from thesis_corpus.ollama_client import OllamaUnavailableError, check_available, embed_texts
    try:
        check_available(args.ollama_host)
    except OllamaUnavailableError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1)

    if args.cells:
        entries = load_all_cell_entries(args.source_dir)
        if not entries:
            raise SystemExit(diagnose_empty_parse(args.source_dir))
        counts: dict[str, int] = {}
        for e in entries:
            counts[str(e["cell"])] = counts.get(str(e["cell"]), 0) + 1
        logger.info("Parsed %d entries across cells: %s", len(entries), dict(sorted(counts.items())))
        if set(counts) == {"None"}:
            raise SystemExit(
                f"Parsed {len(entries)} entries but NONE got a cell -- no 'Block A'..'Block D' "
                f"header line was recognised in {args.source_dir}.\n"
                "The cell assignment is the entire experiment, so this is not embedded as-is.\n"
                "Fix: make each block header its own line containing the word 'Block' and the "
                "letter, e.g. 'Block B - ordinary-sounding but coercive'. Then re-run."
            )

        names = [e["name"] for e in entries]
        logger.info("Embedding %d names via %s...", len(names), args.embed_model)
        name_vectors = embed_texts(args.ollama_host, args.embed_model, names)
        out_path = args.out_dir / args.out_name
        with open(out_path, "w", encoding="utf-8") as f:
            for e, vector in zip(entries, name_vectors):
                f.write(json.dumps({"label": e["name"], "cell": e["cell"],
                                    "description": e["description"],
                                    "embedding_vector": vector}, ensure_ascii=False) + "\n")
        written = [out_path]

        if not args.names_only:
            described = [e for e in entries if e["description"]]
            if described:
                logger.info("Embedding %d descriptions via %s...", len(described), args.embed_model)
                desc_vectors = embed_texts(args.ollama_host, args.embed_model,
                                           [e["description"] for e in described])
                desc_path = args.out_dir / args.out_name.replace(".jsonl", "_descriptions.jsonl")
                with open(desc_path, "w", encoding="utf-8") as f:
                    for e, vector in zip(described, desc_vectors):
                        f.write(json.dumps({"label": e["name"], "cell": e["cell"],
                                            "description": e["description"],
                                            "embedding_vector": vector}, ensure_ascii=False) + "\n")
                written.append(desc_path)
            else:
                logger.warning("No descriptions found -- skipping the description pass.")

        print("Done. " + "\n      ".join(str(p) for p in written))
        print(f"({len(entries)} entries, cells {dict(sorted(counts.items()))}). Copy these back to "
              f"the Mac at the same relative paths, then run analyze_coercive_control_groups.py.")
        return

    names = load_all_group_names(args.source_dir)
    if not names:
        raise SystemExit(f"No group names parsed from {args.source_dir} -- nothing to embed.")
    logger.info("Embedding %d manually-curated group names via %s...", len(names), args.embed_model)
    vectors = embed_texts(args.ollama_host, args.embed_model, names)

    out_path = args.out_dir / args.out_name
    with open(out_path, "w", encoding="utf-8") as f:
        for name, vector in zip(names, vectors):
            f.write(json.dumps({"label": name, "embedding_vector": vector}, ensure_ascii=False) + "\n")

    print(f"Done. {out_path} ({len(names)} names). Copy this file back to the Mac at the same "
          f"relative path, then run find_secular_group_mentions.py.")


if __name__ == "__main__":
    main()
