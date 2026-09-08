"""Draws a stratified, reproducible manual-review sample of extracted
expressions from the three raw criterion_expressions.jsonl archives, for
a human fidelity check against their own source context. Produces a
review sheet ONLY -- this script makes no judgement about whether any
expression is faithful, or any label correct; every manual-review column
in its output is left blank, to be filled in by hand.

Reads the three raw archives DIRECTLY (geometric_analysis_common.load_raw_archive,
ARCHIVE_PATH_BY_SOURCE) -- no dependency on the shared space
(embedding_space.jsonl), the persisted PCA transform, or any
processed/analysis/<run-id>/ output. This audit is independent of, and
unaffected by, any correction or rerun of the geometric-analysis toolkit.

Sampling design:
  - Fixed per-source target allocation (literature=34, miviludes=33,
    interviews=33, total=100) -- deliberately NOT proportional to each
    source's population size, since literature is ~97% of pooled
    expression points and a proportional sample would be almost entirely
    literature, defeating the purpose of a cross-corpus fidelity check.
    If a source's eligible population is smaller than its target
    allocation, this is a hard error (SystemExit) -- the script never
    silently reallocates the shortfall to another source.
  - Within each source, sampling is loosely stratified by epistemic_status
    (proportional allocation via the same largest-remainder rounding
    geometric_analysis_common.stratified_sample_by_document already uses
    for document-level stratification, generalized here to status groups,
    with each group's allocation additionally capped at its own
    population size and any resulting shortfall redistributed to
    remaining groups) -- rare statuses are not force-balanced to equal
    counts (which would misrepresent the archive's real composition) but
    are not entirely excluded either, as a single random draw ignoring
    status could do by chance.
  - "Eligible" = has non-empty embedding_text AND non-empty context_window
    (both required review-sheet fields) -- no other restriction; every
    epistemic_status value is eligible, unlike the geometric-analysis
    toolkit's own asserted_qualified filter.
  - Every random draw uses a deterministic sub-seed derived from the base
    --seed plus a fixed (source, status) enumeration order -- never
    Python's hash() (not stable across runs/processes) -- so the full
    sample, and its row order, are exactly reproducible.

Usage (from thesis/corpus/):
    python -m thesis_corpus.audit_expression_fidelity_sample
    python -m thesis_corpus.audit_expression_fidelity_sample --seed 42 --total 100
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from thesis_corpus import geometric_analysis_common as gac

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("thesis_corpus.audit_expression_fidelity_sample")

SCRIPT_VERSION = "1.0.0"
DEFAULT_SEED = 42
DEFAULT_ALLOCATION = {"literature": 34, "miviludes": 33, "interviews": 33}
AUDITS_DIR = gac.PROCESSED_DIR / "audits"

REVIEW_SHEET_COLUMNS = [
    "sample_id", "source_dataset", "document_id", "chunk_index",
    "embedding_text", "context_window", "attribution", "claim_mode",
    "epistemic_status", "raw_archive_path", "raw_archive_line",
    # Manual-review columns -- always blank in this script's own output.
    "expression_faithful", "attribution_correct", "claim_mode_correct",
    "epistemic_status_correct", "recommended_epistemic_status", "reviewer_notes",
]
MANUAL_REVIEW_COLUMNS = [
    "expression_faithful", "attribution_correct", "claim_mode_correct",
    "epistemic_status_correct", "recommended_epistemic_status", "reviewer_notes",
]

README_TEMPLATE = """\
# Expression fidelity sample -- manual review sheet

Generated {timestamp} (seed={seed}, script version {script_version}, git commit {git_commit}).

## What this is

A stratified, reproducible sample of {total} extracted expressions drawn
directly from the three raw source archives ({sources}), for a manual
fidelity check: does each machine-extracted expression, read against its
own surrounding context, actually say what its metadata claims it says?

**This is a targeted quality-control sample, not a statistical estimate
of universal pipeline accuracy, and not a replacement for close reading
in the thesis.** {total} items, drawn once, cannot establish an error
rate for the ~40,000-expression archive as a whole -- they can surface
whether the extraction/tagging pipeline is producing plausible output on
a cross-section of real cases, and catch any systematic problem worth
knowing about before relying on the pipeline's output further.

Sample allocation (deliberately non-proportional to each source's real
size, so no single source dominates the review purely by candidate-pool
size): {allocation}.

## How to review each row

For every row in `{csv_filename}`, read `embedding_text` against its own
`context_window` (the surrounding source text the expression was
extracted from), then fill in the six manual-review columns:

- **`expression_faithful`** -- does `embedding_text` accurately and
  adequately represent the relevant claim in the surrounding
  `context_window`? (e.g. `yes` / `no` / `partial` -- use whatever
  convention you prefer, this script does not prescribe one)
- **`attribution_correct`** -- is the statement attributed to the
  correct speaker or source (`attribution` column)?
- **`claim_mode_correct`** -- is the claim-mode label (`claim_mode`)
  plausible when read in context?
- **`epistemic_status_correct`** -- is the epistemic label
  (`epistemic_status`) correct when read in context?
- **`recommended_epistemic_status`** -- only fill this in if the current
  status is clearly wrong; otherwise leave blank. Do not fill this in
  merely because you'd have phrased the label differently.
- **`reviewer_notes`** -- briefly state the reason for a "no", any
  ambiguity, or a proposed correction. Leave blank if there's nothing to
  add.

Every manual-review column starts blank in this script's own output --
nothing here has been prefilled, predicted, or automatically scored.

## Source-by-epistemic-status distribution

Full counts (eligible population and selected sample, per source and
per status) are recorded in `{config_filename}`, not repeated here.

## Scope note

This script draws the sample and leaves every judgement column blank.
It does not decide, suggest, or imply whether any row is faithful or
correctly labelled -- that determination is yours alone.
"""


def eligible_items(indexed_archive_items: list[tuple[int, dict]]) -> list[tuple[int, dict]]:
    """Non-empty embedding_text AND non-empty context_window -- the two
    fields the review sheet requires -- with no other restriction. Every
    epistemic_status value is eligible; this is deliberately broader than
    the geometric-analysis toolkit's own asserted_qualified filter.
    Operates on (line_index, item) pairs throughout, rather than bare
    items, so a sampled item's exact source line is always carried
    alongside it -- no separate identity-based lookup needed later."""
    return [
        (i, item) for i, item in indexed_archive_items
        if (item.get("embedding_text") or "").strip() and (item.get("context_window") or "").strip()
    ]


def stratified_allocation_by_status(groups: dict[str, list], n: int) -> dict[str, int]:
    """Allocation of n items across `groups` ({status: [items]}), in two
    stages:

    1. A guaranteed floor of 1 for every group that has at least one
       eligible item, up to as many groups as fit within n (smallest
       groups first, so a genuinely rare status is the one guaranteed
       representation, not crowded out by an already-common one) --
       this is what keeps a rare status from being entirely absent
       purely because pure proportional rounding sends it to zero (a
       real, expected outcome here: literature is ~98% `asserted`, so a
       naive proportional 34-item draw would allocate exactly 34 to
       `asserted` and 0 to every other status).
    2. The remainder (n minus the floor already placed) is allocated
       proportionally to each group's remaining population share, via
       the same largest-remainder rounding
       geometric_analysis_common.stratified_sample_by_document uses for
       document-level stratification, generalized here to status
       groups.

    Every group's final allocation is capped at its own population size
    (a rare status's floor-of-1 plus its tiny proportional share can
    never together exceed how many such items actually exist); any
    shortfall this produces is redistributed to groups with spare
    capacity, largest first. This does NOT force equal counts per status
    -- a common status still receives far more than a rare one, just
    never zero when at least one real item exists. Deterministic given
    the same `groups` and `n` -- no randomness here, only which/how-many
    items per group; the actual item-level draw within each group
    happens separately, under the caller's own seed."""
    total = sum(len(items) for items in groups.values())
    if n > total:
        raise ValueError(f"Cannot allocate {n} items across groups totaling only {total}.")

    by_size_asc = sorted(groups.items(), key=lambda kv: (len(kv[1]), kv[0]))
    by_size_desc = list(reversed(by_size_asc))

    floor: dict[str, int] = {status: 0 for status in groups}
    floor_budget = n
    for status, items in by_size_asc:
        if floor_budget <= 0:
            break
        floor[status] = 1
        floor_budget -= 1

    remaining_n = n - sum(floor.values())
    remaining_capacity = {status: len(items) - floor[status] for status, items in groups.items()}
    remaining_total = sum(remaining_capacity.values())

    proportional: dict[str, int] = {status: 0 for status in groups}
    if remaining_n > 0 and remaining_total > 0:
        for status, items in by_size_desc:
            share = remaining_capacity[status] / remaining_total if remaining_total else 0
            proportional[status] = min(int(share * remaining_n), remaining_capacity[status])
        shortfall = remaining_n - sum(proportional.values())
        while shortfall > 0:
            progressed = False
            for status, items in by_size_desc:
                if shortfall == 0:
                    break
                if proportional[status] < remaining_capacity[status]:
                    proportional[status] += 1
                    shortfall -= 1
                    progressed = True
            if not progressed:
                break  # remaining_capacity fully exhausted everywhere; shouldn't happen given n <= total

    allocations = {status: floor[status] + proportional[status] for status in groups}

    # Final safety net: cap at population, redistribute any residual
    # shortfall (should not trigger given the logic above, kept as a
    # defensive invariant rather than assumed unreachable).
    for status, items in groups.items():
        allocations[status] = min(allocations[status], len(items))
    shortfall = n - sum(allocations.values())
    while shortfall > 0:
        progressed = False
        for status, items in by_size_desc:
            if shortfall == 0:
                break
            if allocations[status] < len(items):
                allocations[status] += 1
                shortfall -= 1
                progressed = True
        if not progressed:
            raise AssertionError(
                f"stratified_allocation_by_status: could not place remaining shortfall={shortfall} "
                f"across groups totaling {total} for requested n={n} -- this should be unreachable "
                "given the n > total check above; investigate."
            )
    return allocations


def sample_source(
    source: str, archive_items: list[dict], target_n: int, base_seed: int, source_index: int,
) -> tuple[list[tuple[int, dict]], dict]:
    """Returns (sampled, diagnostics) for one source, where `sampled` is a
    list of (line_index, item) pairs -- the line_index is each item's
    0-based position in `archive_items` (== its line number in the raw
    archive file, per gac.load_raw_archive's own file-order guarantee),
    carried through explicitly rather than re-derived later. `diagnostics`
    carries the eligible-population and achieved-sample counts per
    epistemic_status, for config.json."""
    indexed = list(enumerate(archive_items))
    eligible = eligible_items(indexed)
    if len(eligible) < target_n:
        raise SystemExit(
            f"audit_expression_fidelity_sample: source={source!r} has only {len(eligible)} eligible "
            f"records (non-empty embedding_text and context_window), fewer than its required "
            f"allocation of {target_n}. Refusing to silently reduce this source's allocation or "
            f"reallocate the shortfall elsewhere -- stop and report, per the audit's own design. "
            f"Total raw archive records for this source: {len(archive_items)}."
        )

    groups: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for line_index, item in eligible:
        groups[item.get("epistemic_status") or "(missing)"].append((line_index, item))
    population_counts = {status: len(items) for status, items in groups.items()}

    allocations = stratified_allocation_by_status(groups, target_n)

    ordered_statuses = sorted(groups.keys())
    sampled: list[tuple[int, dict]] = []
    sample_counts: dict[str, int] = {}
    for status_index, status in enumerate(ordered_statuses):
        group_items = groups[status]
        k = allocations[status]
        sample_counts[status] = k
        if k == 0:
            continue
        sub_seed = base_seed + source_index * 1000 + status_index
        # index into group_items, not a value from the items themselves --
        # random.Random(...).sample over a range is what gac.simple_random_sample
        # does over a list of indices; reuse that exact function for
        # consistency with the rest of the toolkit's sampling code.
        chosen_positions = gac.simple_random_sample(list(range(len(group_items))), k, sub_seed)
        for pos in sorted(chosen_positions):
            sampled.append(group_items[pos])

    diagnostics = {
        "eligible_population": len(eligible),
        "total_raw_records": len(archive_items),
        "population_by_status": population_counts,
        "requested_allocation": target_n,
        "achieved_allocation": len(sampled),
        "sample_by_status": sample_counts,
    }
    return sampled, diagnostics


def build_review_rows(source: str, sampled: list[tuple[int, dict]], archive_path: Path) -> list[dict]:
    rows = []
    for line_index, item in sampled:
        rows.append({
            "sample_id": None,  # assigned once, globally, after all sources are combined
            "source_dataset": source,
            "document_id": item.get("document_id"),
            "chunk_index": item.get("chunk_index"),
            "embedding_text": item.get("embedding_text"),
            "context_window": item.get("context_window"),
            "attribution": item.get("attribution"),
            "claim_mode": item.get("claim_mode"),
            "epistemic_status": item.get("epistemic_status"),
            "raw_archive_path": str(archive_path),
            "raw_archive_line": line_index,
            "expression_faithful": "", "attribution_correct": "", "claim_mode_correct": "",
            "epistemic_status_correct": "", "recommended_epistemic_status": "", "reviewer_notes": "",
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--total", type=int, default=sum(DEFAULT_ALLOCATION.values()))
    parser.add_argument(
        "--allocation", type=str, default=None,
        help='Override the default per-source allocation, e.g. \'{"literature":34,"miviludes":33,"interviews":33}\'. '
             "Must sum to --total.",
    )
    parser.add_argument("--out-dir", type=Path, default=AUDITS_DIR)
    parser.add_argument("--date-tag", type=str, default=None, help="Override the YYYYMMDD tag in output filenames (default: today, UTC).")
    args = parser.parse_args()

    allocation = DEFAULT_ALLOCATION if args.allocation is None else json.loads(args.allocation)
    if sum(allocation.values()) != args.total:
        raise SystemExit(f"Allocation {allocation} sums to {sum(allocation.values())}, not --total={args.total}.")
    if set(allocation.keys()) != set(gac.EXPRESSION_CORPORA):
        raise SystemExit(f"Allocation keys {sorted(allocation.keys())} must exactly match {sorted(gac.EXPRESSION_CORPORA)}.")

    date_tag = args.date_tag or datetime.now(timezone.utc).strftime("%Y%m%d")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"expression_fidelity_sample_{date_tag}.csv"
    readme_path = args.out_dir / f"expression_fidelity_sample_{date_tag}.README.md"
    config_path = args.out_dir / f"expression_fidelity_sample_{date_tag}.config.json"

    all_rows: list[dict] = []
    diagnostics_by_source: dict[str, dict] = {}
    archive_paths: dict[str, str] = {}

    for source_index, source in enumerate(gac.EXPRESSION_CORPORA):
        archive_path = gac.ARCHIVE_PATH_BY_SOURCE[source]
        archive_paths[source] = str(archive_path)
        logger.info("Loading raw archive for %s: %s", source, archive_path)
        archive_items = gac.load_raw_archive(archive_path)

        target_n = allocation[source]
        sampled, diagnostics = sample_source(source, archive_items, target_n, args.seed, source_index)
        diagnostics_by_source[source] = diagnostics
        logger.info(
            "%s: %d eligible / %d total raw records; sampled %d (target %d); by-status sample: %s",
            source, diagnostics["eligible_population"], diagnostics["total_raw_records"],
            len(sampled), target_n, diagnostics["sample_by_status"],
        )
        all_rows.extend(build_review_rows(source, sampled, archive_path))

    # Deterministic row order: by source (in gac.EXPRESSION_CORPORA order,
    # already the iteration order above), then by raw_archive_line
    # ascending within each source -- reproducible independent of any
    # dict/set iteration order.
    all_rows.sort(key=lambda r: (gac.EXPRESSION_CORPORA.index(r["source_dataset"]), r["raw_archive_line"]))
    for i, row in enumerate(all_rows, start=1):
        row["sample_id"] = i

    gac.write_csv(csv_path, all_rows)
    logger.info("Wrote %d rows -> %s", len(all_rows), csv_path)

    achieved_allocation = {source: diagnostics_by_source[source]["achieved_allocation"] for source in gac.EXPRESSION_CORPORA}
    config = {
        "script": "audit_expression_fidelity_sample.py",
        "script_version": SCRIPT_VERSION,
        "git_commit": gac.git_commit_hash(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "requested_allocation": allocation,
        "achieved_allocation": achieved_allocation,
        "total_sample_size": len(all_rows),
        "raw_archive_paths": archive_paths,
        "diagnostics_by_source": diagnostics_by_source,
        "review_sheet_columns": REVIEW_SHEET_COLUMNS,
        "manual_review_columns_left_blank": MANUAL_REVIEW_COLUMNS,
        "csv_path": str(csv_path),
        "readme_path": str(readme_path),
    }
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote config -> %s", config_path)

    readme_text = README_TEMPLATE.format(
        timestamp=config["generated_at"],
        seed=args.seed,
        script_version=SCRIPT_VERSION,
        git_commit=config["git_commit"],
        total=len(all_rows),
        sources=", ".join(f"{s} ({archive_paths[s]})" for s in gac.EXPRESSION_CORPORA),
        allocation=", ".join(f"{s}={achieved_allocation[s]}" for s in gac.EXPRESSION_CORPORA),
        csv_filename=csv_path.name,
        config_filename=config_path.name,
    )
    readme_path.write_text(readme_text, encoding="utf-8")
    logger.info("Wrote README -> %s", readme_path)

    print(f"\nDone. {len(all_rows)} rows -> {csv_path}")


if __name__ == "__main__":
    main()
