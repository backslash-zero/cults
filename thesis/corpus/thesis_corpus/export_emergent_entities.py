"""Ranks emergent-entity anchors by total mention count and writes two views
of the same ranking: a full CSV (all entities, for anyone who wants the
complete data) and a LaTeX \\input-able table fragment (top N only, for the
"Emergent Entities" appendix, 04_Appendix/6_Appendix.tex).

Reads processed/shared_space/embedding_space.jsonl -- never modifies it,
same "never mutate the source, only write new files" convention as
balanced_analysis.py and build_shared_space.py. Rerun after any
build_shared_space.py rerun that could change mention counts.

Usage:
    python -m thesis_corpus.export_emergent_entities
    python -m thesis_corpus.export_emergent_entities --top-n 50
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent.parent
EMBEDDING_SPACE_PATH = CORPUS_DIR / "processed" / "shared_space" / "embedding_space.jsonl"
CSV_OUTPUT_PATH = CORPUS_DIR / "processed" / "shared_space" / "emergent_entities_ranked.csv"
LATEX_OUTPUT_PATH = (
    CORPUS_DIR.parent / "04_Appendix" / "Emergent_Entities" / "top_100_rows.tex"
)

DEFAULT_TOP_N = 100

_LATEX_ESCAPES = [
    ("\\", r"\textbackslash{}"),
    ("&", r"\&"),
    ("%", r"\%"),
    ("$", r"\$"),
    ("#", r"\#"),
    ("_", r"\_"),
    ("{", r"\{"),
    ("}", r"\}"),
    ("~", r"\textasciitilde{}"),
    ("^", r"\textasciicircum{}"),
]


def _escape_latex(text: str) -> str:
    for old, new in _LATEX_ESCAPES:
        text = text.replace(old, new)
    return text


def _latex_number(n: int) -> str:
    """44325 -> "44{,}325" -- this codebase's thousands-separator convention
    throughout Methods.tex/ANALYSIS_OVERVIEW.md (plain commas can trigger
    inconsistent spacing in LaTeX)."""
    return f"{n:,}".replace(",", "{,}")


def rank_emergent_entities(embedding_space_path: Path = EMBEDDING_SPACE_PATH) -> list[dict]:
    """Returns every emergent_entities point, sorted by total mention count
    descending, as {label, total, literature, miviludes, interviews}."""
    rows = []
    with open(embedding_space_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("source_dataset") != "emergent_entities":
                continue
            mentions = item.get("mention_distribution") or {}
            rows.append({
                "label": item["label"],
                "total": sum(mentions.values()),
                "literature": mentions.get("literature", 0),
                "miviludes": mentions.get("miviludes", 0),
                "interviews": mentions.get("interviews", 0),
            })
    rows.sort(key=lambda r: r["total"], reverse=True)
    return rows


def write_csv(rows: list[dict], path: Path = CSV_OUTPUT_PATH) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "entity", "total_mentions", "literature", "miviludes", "interviews"])
        for i, r in enumerate(rows, 1):
            writer.writerow([i, r["label"], r["total"], r["literature"], r["miviludes"], r["interviews"]])


def write_latex_rows(rows: list[dict], top_n: int, path: Path = LATEX_OUTPUT_PATH) -> None:
    lines = []
    for i, r in enumerate(rows[:top_n], 1):
        lines.append(
            f"{i} & {_escape_latex(r['label'])} & {_latex_number(r['total'])} & "
            f"{_latex_number(r['literature'])} & {_latex_number(r['miviludes'])} & "
            f"{_latex_number(r['interviews'])} \\\\"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    args = parser.parse_args()

    rows = rank_emergent_entities()
    write_csv(rows)
    write_latex_rows(rows, args.top_n)

    top_n_total = sum(r["total"] for r in rows[: args.top_n])
    grand_total = sum(r["total"] for r in rows)
    print(f"{len(rows)} emergent entities ranked.")
    print(f"Wrote full ranking -> {CSV_OUTPUT_PATH}")
    print(f"Wrote top {args.top_n} LaTeX rows -> {LATEX_OUTPUT_PATH}")
    print(
        f"Top {args.top_n} ({100 * args.top_n / len(rows):.1f}% of entities) account for "
        f"{top_n_total:,} of {grand_total:,} total mentions ({100 * top_n_total / grand_total:.1f}%)."
    )


if __name__ == "__main__":
    main()
