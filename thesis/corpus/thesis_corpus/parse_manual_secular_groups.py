"""Parses the manually-curated non-religious-group name lists in
dictionaries/non-religious-groups/ into a clean, deduplicated name list --
the C2 comparison set for the central Hypothesis test
(Analysis/08_Secular_Groups_Hypothesis.md), replacing the earlier
on-the-fly generate_and_embed_secular_groups.py approach.

Why this replaces live generation: three straight rounds of driving a
local Ollama model (generate_and_embed_secular_groups.py) to produce this
list live hit three different failure modes (timeout, off-topic essay,
generic non-answer) across a slow manual Windows-run/copy-back cycle. The
user instead ran the same kind of prompt themselves, reviewed the output,
and pasted the results into dictionaries/non-religious-groups/*.txt -- a
one-time, human-reviewed, reproducible input, which is arguably more
defensible for a thesis than stochastic live generation anyway (see the
Hypothesis writeup for that framing). Only the deterministic parsing step
lives here; embedding still needs Ollama's bge-m3 (unreachable on this
Mac), so that stays a separate cross-machine step -- see
embed_manual_secular_groups.py.

Source format (both files are raw LLM chat transcripts, "prompt: ..." /
"model: ..." header lines, then real entries mixed with category-header
lines): every real entry has a name followed by a separator introducing a
description -- an English file uses "Name (ACRONYM) -- Description."
(en-dash), the French file uses "Name (description)." (parenthetical only,
no dash). Category headers (e.g. "Humanist & Secular Organizations",
"1. Environnement et durabilité") have neither a dash-with-spaces nor a
parenthesis anywhere in the line -- confirmed by inspecting every line in
both source files, not assumed. That's the one discriminating feature this
parser relies on to tell a real entry from a section heading.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = HERE.parent / "dictionaries" / "non-religious-groups"

# Requires whitespace after the delimiter -- "1. Item" is a list marker,
# but "350.org" (a real entry, a domain name) is not: a real failure this
# guards against, first version of this regex (with `\s*` instead of
# `\s+`) stripped "350." off of "350.org -- Climate action." and produced
# the bare, meaningless name "org".
_LEADING_NUMBERING_RE = re.compile(r"^\d+[.)]\s+")
_SEPARATOR_RE = re.compile(r"\(| [–—-] ")


def parse_manual_group_list(raw_text: str) -> list[str]:
    """Extracts bare group names from one source file's raw text, in
    order, without deduplicating (dedup happens once across all source
    files combined -- see load_all_group_names). Skips blank lines, the
    "prompt:"/"model:" transcript header lines, and category-header lines
    (recognized by having no name/description separator at all -- see
    module docstring)."""
    names = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line or line.lower().startswith(("prompt:", "model:")):
            continue
        line = _LEADING_NUMBERING_RE.sub("", line)
        if not line:
            continue
        match = _SEPARATOR_RE.search(line)
        if match is None:
            continue  # category header, not an entry
        name = line[: match.start()].strip()
        if name:
            names.append(name)
    return names


# Tolerant of how a model actually decorates a heading: markdown heading
# hashes, bold/italic asterisks or underscores, blockquote markers, and any
# punctuation between the word and the letter ("### Block A:", "**Block
# A —**", "Bloc B -"). The word block/bloc is REQUIRED: a bare "A — ..."
# line is indistinguishable from a "Name - description" entry, so accepting
# it would silently eat a real entry.
_BLOCK_HEADER_RE = re.compile(r"^[#*_>\s]*(?:block|bloc)\s*[:.\-–—]?\s*([A-D])\b", re.IGNORECASE)
_DASH_SEPARATOR_RE = re.compile(r" [–—-] ")


def parse_manual_group_list_with_cells(raw_text: str) -> list[dict]:
    """Variant of parse_manual_group_list for the 4-cell coercive-control
    lists, which differ from the plain non-religious lists in two ways that
    both matter: the DESCRIPTION is kept rather than discarded, and the
    block headers are data (the experimental cell) rather than noise to skip.

    Returns `{"name", "description", "cell"}` per entry, where `cell` is
    "A".."D" from the most recent `Block X` header (None for entries before
    any header). Descriptions are kept because bare names carry no
    information about what an organization does -- the limitation established
    in Analysis/10, where the groups nearest "difficulty leaving the group"
    turned out to be a mountain-biking group and two sports clubs, matched on
    the word "group". Scoring names and descriptions separately turns that
    limitation into a measurement.

    Prefers the dash separator over the parenthesis one, unlike
    parse_manual_group_list: here the requested format is explicitly
    `Name – description`, so a name that itself contains parentheses
    ("Anonymous (hacker collective) – ...") must not be split at the bracket."""
    entries = []
    cell = None
    for line in raw_text.splitlines():
        line = line.strip()
        if not line or line.lower().startswith(("prompt:", "model:")):
            continue

        header = _BLOCK_HEADER_RE.match(line)
        if header:
            cell = header.group(1).upper()
            continue

        # Checked AFTER the header match, so a markdown-style "### Block A"
        # is still read as a header. Hand-curated source files carry "#"
        # provenance comments explaining inclusion rules, and those must not
        # be embedded as group names.
        if line.startswith("#"):
            continue

        line = _LEADING_NUMBERING_RE.sub("", line).strip()
        if not line:
            continue

        match = _DASH_SEPARATOR_RE.search(line)
        if match:
            name, description = line[: match.start()], line[match.end():]
        else:
            paren = line.find("(")
            if paren == -1:
                continue  # no separator at all -> a stray heading, not an entry
            name, description = line[:paren], line[paren + 1:].rstrip(")")

        name = name.strip().strip("\"'")
        if name:
            entries.append({"name": name, "description": description.strip(), "cell": cell})
    return entries


def load_all_cell_entries(source_dir: Path) -> list[dict]:
    """parse_manual_group_list_with_cells over every .txt/.tx file in
    source_dir, deduplicated on (cell, casefolded name). A name appearing in
    TWO different cells is deliberately kept twice -- that's a genuine
    cell-assignment conflict the analysis should surface, not silently
    resolve (e.g. an MLM the model puts in both "famous cults" and
    "ordinary-sounding but coercive")."""
    seen = set()
    entries = []
    for path in sorted(source_dir.iterdir()):
        if path.suffix not in (".txt", ".tx"):
            continue
        for entry in parse_manual_group_list_with_cells(path.read_text(encoding="utf-8")):
            key = (entry["cell"], entry["name"].casefold())
            if key not in seen:
                seen.add(key)
                entries.append(entry)
    return entries


def load_all_group_names(source_dir: Path = DEFAULT_SOURCE_DIR) -> list[str]:
    """Reads every .txt/.tx file in source_dir, parses each, and
    deduplicates case-insensitively across all of them combined (the same
    org appears under multiple category headings within and across files --
    e.g. "The Southern Poverty Law Center (SPLC)" appears twice in the
    English file alone). Keeps the first-seen casing/order."""
    seen: dict[str, str] = {}
    for path in sorted(source_dir.iterdir()):
        if path.suffix not in (".txt", ".tx"):
            continue
        for name in parse_manual_group_list(path.read_text(encoding="utf-8")):
            key = name.casefold()
            if key not in seen:
                seen[key] = name
    return list(seen.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_SOURCE_DIR / "parsed_group_names.txt")
    args = parser.parse_args()

    names = load_all_group_names(args.source_dir)
    args.out.write_text("\n".join(names) + "\n", encoding="utf-8")
    print(f"Parsed {len(names)} unique group names from {args.source_dir} -> {args.out}")


if __name__ == "__main__":
    main()
