"""Prepare interview segments for the layered analysis.

Keeps participant segments, flags fillers (segments with no content word), and
links each hand-reviewed entity to the segments that name it. Entity patterns
live in ``interviews/metadata/entity_aliases.csv`` so they can be edited by hand.
No model is called.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
from pathlib import Path

CORPUS = Path(__file__).resolve().parents[1]
RUN_DIR = CORPUS / "processed/interviews_full/interviews/run_20260912"
ALIASES = CORPUS / "interviews/metadata/entity_aliases.csv"
OUT_DIR = CORPUS / "processed/analysis_raw/layers"

# Function words and discourse markers. A segment made only of these is a filler.
FILLER_WORDS = frozenset("""
a about absolutely actually all also am an and any anything anyway are as at be but by called can check could
did do does don dont else even exactly first for from get go going good got guess had happy has have he hear
heard her here him his how huh i if in is it its just kind know let like look maybe me mean might more most my
name names no non not nothing now of oh ok okay on one or other our out people perhaps probably random re really
right s said same say says secondly see she sell so some something sorry sort sure t than thank thanks that the
their them then there these they thing things think this those thought time to too uh um understand us very
want was way we well were what when where which who why will with would wrong yeah yep yes you your
à ah alors au aussi aux avec bah bien c ça ce cela ces cest comment comprends comprenais dans de des dire dis
donc du elle en encore est et être eu euh faire fait il ils j je jai l la le les leur lui m ma mais me même mes
mon n ne non nous on ou où ouais oui par pas peut peu plus pour pourquoi putain qu que quelque quelques qui quoi
s sa sais sans se ses si sinon sont suis sur t ta te toujours tous tout tu un une va vais vient vois voilà vous
y comprends sait esprit forcément vraiment
""".split())

WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def is_filler(text: str) -> bool:
    """True when the text contains no word outside FILLER_WORDS."""
    return all(word in FILLER_WORDS for word in WORD.findall(text.lower()))


def load_aliases(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["regex"] = re.compile(row["pattern"], re.IGNORECASE)
    return rows


def match_entities(text: str, aliases: list[dict]) -> list[str]:
    return [row["entity"] for row in aliases if row["regex"].search(text)]


def load_segments(run_dir: Path) -> list[dict]:
    with (run_dir / "criterion_expressions.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def segment_key(segment: dict) -> str:
    return f"{segment['document_id']}:{segment['chunk_index']}:{segment['segment_index']}"


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--aliases", type=Path, default=ALIASES)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--review-dir", type=Path, help="Also write the review CSVs here.")
    args = parser.parse_args()

    aliases = load_aliases(args.aliases)
    participant = [s for s in load_segments(args.run_dir) if s["attribution"] == "participant"]
    rows, fillers = [], []
    by_entity: dict[str, list[dict]] = collections.defaultdict(list)
    for segment in participant:
        text = segment["verbatim_expression"]
        record = {"key": segment_key(segment), "document_id": segment["document_id"], "text": text}
        if is_filler(text):
            fillers.append(record)
            continue
        entities = match_entities(text, aliases)
        rows.append({**record, "cult_relevant": segment["cult_relevant"], "entities": "; ".join(entities)})
        for entity in entities:
            by_entity[entity].append(record)

    summary = []
    for alias in aliases:
        hits = by_entity[alias["entity"]]
        summary.append({
            "entity": alias["entity"], "type": alias["type"], "segments": len(hits),
            "interviews": len({h["document_id"] for h in hits}),
            "examples": " || ".join(h["text"][:110] for h in hits[:3]),
        })
    summary.sort(key=lambda r: (-r["interviews"], -r["segments"]))

    write_csv(args.out_dir / "participant_segments.csv", rows, ["key", "document_id", "cult_relevant", "entities", "text"])
    entity_fields = ["entity", "type", "segments", "interviews", "examples"]
    write_csv(args.out_dir / "entity_summary.csv", summary, entity_fields)
    write_csv(args.out_dir / "fillers_removed.csv", fillers, ["key", "document_id", "text"])
    if args.review_dir:
        write_csv(args.review_dir / "entities_review.csv", summary, entity_fields)
        write_csv(args.review_dir / "fillers_removed.csv", fillers, ["key", "document_id", "text"])
    print(json.dumps({
        "participant": len(participant), "fillers": len(fillers), "kept": len(rows),
        "kept_with_entity": sum(bool(r["entities"]) for r in rows),
        "entities_3plus_segments": sum(r["segments"] >= 3 for r in summary),
        "entities_2plus_interviews": sum(r["interviews"] >= 2 for r in summary),
    }))


if __name__ == "__main__":
    main()
