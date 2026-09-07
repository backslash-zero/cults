"""Assembles a single neutral Markdown draft from a completed analysis
run's saved output -- reads only, computes nothing new, and does not
interpret or synthesize findings. That synthesis is the user's own work;
this script's job is to lay out what each module actually produced,
consistently and completely, so a person can write the interpretation on
top of it.

Explicitly never repeats "order of mention is a standard salience proxy"
or any similar claim -- interviews provide exploratory initial-exemplar
evidence (analyze_initial_exemplars.py), not a ranked free-list measure
(see audit_free_listing_rank.py's verdict, always surfaced first if
present).

Requires an existing --run-id, same as generate_figures.py -- never falls
back to "latest".

Usage (from thesis/corpus/):
    python -m thesis_corpus.generate_geometric_draft_report --run-id 20260101-120000
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

from thesis_corpus import geometric_analysis_common as gac

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("thesis_corpus.generate_geometric_draft_report")

MODULE_NAME = "generate_geometric_draft_report"


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def md_table(rows: list[dict], columns: list[str], max_rows: int = 20) -> str:
    if not rows:
        return "_(no rows)_\n"
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows[:max_rows]:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    if len(rows) > max_rows:
        lines.append(f"\n_(showing {max_rows} of {len(rows)} rows -- see the run's own CSV for the rest)_")
    return "\n".join(lines)


def section_global_structure(run_dir: Path) -> str:
    d = run_dir / "analyze_global_structure"
    if not d.exists():
        return ""
    pairwise = read_csv_rows(d / "pairwise_source_centroid_matrix.csv")
    reference = read_csv_rows(d / "reference_to_combined_expression_centroid.csv")
    return f"""## Global structure

Pairwise source-centroid distances (mode-independent, Euclidean primary; full table in `analyze_global_structure/pairwise_source_centroid_matrix.csv`):

{md_table(pairwise, ["source_a", "source_b", "euclidean_distance", "cosine_similarity"], max_rows=15)}

Reference/emergent centroids' distance to the combined-expression reference point, across all three combined-reference variants (never merged):

{md_table(reference, ["mode", "reference_dataset", "euclidean_distance_to_combined_expression"], max_rows=15)}

Nearest terms from each of the 3 reference vocabularies (concept_backbone, structural_concepts, conceptnet_concepts) per expression corpus are in `analyze_global_structure/nearest_reference_terms_raw.csv` (raw retrieval, NOT size-adjusted -- the three vocabularies have very different candidate-pool sizes: 3,000/1,500/195). An equal-size-controlled version of the same comparison (every set down-sampled to the smallest, currently 195, repeated with bootstrap mean/std/95% CI) is in the separate `nearest_reference_terms_equal_size.csv` -- **this controls for pool-size bias only; it is not evidence the three vocabularies are otherwise interchangeable** (they differ in source, curation, coverage, and purpose). The two files are never merged.
"""


def section_cluster_structure(run_dir: Path) -> str:
    d = run_dir / "analyze_cluster_structure"
    if not d.exists():
        return ""
    equal_n = read_csv_rows(d / "knn_composition_equal_n_expression.csv")
    silhouette = json.loads((d / "silhouette_equal_n_expression.json").read_text(encoding="utf-8")) if (d / "silhouette_equal_n_expression.json").exists() else {}
    umap_manifests = sorted(d.glob("*.manifest.json"))
    return f"""## Cluster structure

**Primary controlled comparison** (`equal_n_expression`, bootstrapped): k-NN neighbour composition, mean fraction +/- 95% empirical interval across repetitions:

{md_table(equal_n, ["k", "query_corpus", "neighbor_corpus", "mean", "std", "ci95_low", "ci95_high"], max_rows=18)}

Silhouette (`equal_n_expression`, bootstrapped): mean={silhouette.get('mean', 'n/a')}, std={silhouette.get('std', 'n/a')}, 95% CI [{silhouette.get('ci95_low', 'n/a')}, {silhouette.get('ci95_high', 'n/a')}].

`full` and `reduced_literature` (descriptive/sensitivity views, not the primary comparison) are in `analyze_cluster_structure/knn_composition_descriptive.csv` and `silhouette_descriptive.csv`.

2-D UMAP fits this run produced ({len(umap_manifests)} manifest(s)):
{chr(10).join(f"- `{m.name}`" for m in umap_manifests)}
See `generate_figures/` for the rendered scatter plots, if that module has been run for this run-id.
"""


def section_rank_audit(run_dir: Path) -> str:
    d = run_dir / "audit_free_listing_rank"
    if not d.exists():
        return ""
    verdict = json.loads((d / "verdict.json").read_text(encoding="utf-8"))
    return f"""## response_rank audit (read this before any interview-prototype claim)

**salience_proxy_valid: {verdict['salience_proxy_valid']}**

{verdict['reason']}

Narrow check ("{verdict['narrow_check']}"): {verdict['n_passing_narrow_check']}/{verdict['n_documents']} documents pass ({verdict['narrow_check_pass_rate']:.1%}).

`response_rank` plays no role in any other section of this report. Interview-side geometric evidence comes exclusively from the manually-reviewed initial-exemplar workflow below.
"""


def section_initial_exemplars(run_dir: Path) -> str:
    d = run_dir / "analyze_initial_exemplars"
    if not d.exists():
        return ""
    summary = read_csv_rows(d / "exemplar_summary.csv")
    n_resolved = sum(1 for r in summary if r.get("status") == "resolved")
    n_unavailable = sum(1 for r in summary if r.get("status") == "unavailable")
    return f"""## Initial exemplars (interviews)

**n={len(summary)} interviews total: {n_resolved} resolved (have a geometric position), {n_unavailable} unavailable (no qualifying claim in the transcript, no distances computed). Exploratory/descriptive only -- no inferential statistics, no claim of representativeness.**

Each interview's first participant claim in response to the opening prompt (manually verified against the transcript -- see `interviews/metadata/initial_exemplars.csv`), analyzed geometrically: distance to the 17 MIVILUDES criteria, to the reference vocabularies, to each expression-corpus centroid, and nearest literature/MIVILUDES/other-interview expressions. Grouped by `exemplar_type` (a descriptive code, not an asserted natural category). Full tables: `analyze_initial_exemplars/*.csv`.

{md_table(summary, ["document_id", "exemplar_type", "status", "source_expression_label"], max_rows=26)}
"""


def section_criterion_neighbours(run_dir: Path) -> str:
    d = run_dir / "analyze_criterion_neighbours"
    if not d.exists():
        return ""
    ordering = read_csv_rows(d / "language_sensitivity_ordering_comparison.csv")
    by_pair: dict[str, list[dict]] = {}
    for r in ordering:
        by_pair.setdefault(r.get("comparison_pair", "unknown"), []).append(r)

    pair_summaries = []
    for pair, rows in sorted(by_pair.items()):
        n_sensitive = sum(1 for r in rows if r.get("language_sensitive") == "True")
        pair_summaries.append(f"- `{pair}`: {n_sensitive}/{len(rows)} criteria show a different nearest corpus.")

    return f"""## Criterion neighbours

Three separate representations are analyzed, never merged into one heatmap or summary statistic:
- `french_primary_shared_space` -- authoritative, current 394-D shared PCA space.
- `english_sensitivity_shared_space` -- the stored English criterion embeddings, projected into the SAME 394-D space via the persisted transform (same coordinates/metric/corpus vectors/controlled samples as French-primary, so this is the primary sensitivity check).
- `english_raw_embedding_cosine` -- optional secondary diagnostic, raw 1024-d bge-m3 cosine (not comparable in scale to the two 394-D representations above).

Full qualitative-retrieval and controlled-comparison tables per representation are in `analyze_criterion_neighbours/qualitative_retrieval_<representation>.csv` and `controlled_comparison_<representation>.csv`. Nearest reference-vocabulary terms (raw + equal-size-controlled) are in `nearest_reference_terms_raw_<representation>.csv` and `reference_comparison_equal_size_<representation>.csv`, for the two 394-D representations only (a reference-vocabulary comparison isn't meaningful in raw bge-m3 space). See `generate_figures/criterion_corpus_heatmap.png` for the criterion x corpus heatmap (built from the French-primary controlled comparison, not the raw qualitative distances).

**Language-representation sensitivity audit** (ordering comparison only -- never an averaged distance):
{chr(10).join(pair_summaries)}

{md_table([r for r in ordering if r.get("language_sensitive") == "True"], ["criterion_key", "comparison_pair"] + [c for c in (ordering[0].keys() if ordering else []) if c.startswith("nearest_corpus_")])}
"""


def section_emergent_entities(run_dir: Path) -> str:
    d = run_dir / "analyze_emergent_entities"
    if not d.exists():
        return ""
    provenance = read_csv_rows(d / "provenance_category_counts.csv")
    return f"""## Emergent entities

Provenance categories (which corpora mention each entity):

{md_table(provenance, ["provenance_category", "n_entities"])}

Full per-entity table (rank, mentions, provenance category, nearest MIVILUDES criterion, nearest term from each of the 3 reference vocabularies) is in `analyze_emergent_entities/emergent_entities_full.csv`. An equal-size-controlled reference comparison across all ranked entities (controls for the vocabularies' very different sizes -- not evidence they're otherwise interchangeable) is in the separate `reference_comparison_equal_size.csv`. This analysis does not classify entities as cult/mainstream or apply any other normative label -- see `analyze_emergent_entities/SCOPE_NOTICE.txt`.
"""


def section_focused_projections(run_dir: Path) -> str:
    d = run_dir / "generate_focused_projections"
    if not d.exists():
        return ""
    index_path = d / "populations_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    rows = [
        {"population": name, **{k: v for k, v in info.items() if k != "shortages"},
         "shortages": "yes" if info.get("shortages") else "no"}
        for name, info in sorted(index.items())
    ]
    return f"""## Focused projections

Small, curated 2-D projections (UMAP + PCA-2D slice each) built for readable figures -- distinct from the whole-space UMAP populations above, which are too dense to read any one comparison off of. `n_points_fit` is what UMAP/PCA actually fit on; `n_points_overlay` (corpus centroids, embedded into the fitted layout after fitting, never part of the fit itself) and `n_points_rendered` (`= fit + overlay`) are reported separately. A `shortages`=yes population had fewer usable candidates than requested for at least one of its source pools (logged in that population's own manifest) -- distinct from and never caused by the UMAP `n_neighbors` clamp for small populations.

{md_table(rows, ["population", "n_points_fit", "n_points_overlay", "n_points_rendered", "shortages"], max_rows=30)}

See `generate_figures/` for the rendered scatter plots (both `umap_*` and `pca_*` variants, colored by `source_dataset` and by `point_role`).
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", type=str, required=True)
    args = parser.parse_args()

    run_dir = gac.existing_run_dir(args.run_id)
    manifest_path = run_dir / "RUN_MANIFEST.json"
    if not manifest_path.exists():
        raise SystemExit(f"No RUN_MANIFEST.json in {run_dir} -- this doesn't look like a toolkit run.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    sections = [
        section_rank_audit(run_dir),
        section_global_structure(run_dir),
        section_cluster_structure(run_dir),
        section_criterion_neighbours(run_dir),
        section_initial_exemplars(run_dir),
        section_emergent_entities(run_dir),
        section_focused_projections(run_dir),
    ]
    body = "\n".join(s for s in sections if s)

    header = f"""# Geometric analysis -- draft report (run `{args.run_id}`)

**This is a neutral working draft, not a finished analysis.** It lays out
what each module in the geometric-analysis toolkit produced for this run --
tables, summary statistics, figure references -- without synthesis or
interpretation. Writing the thesis's own interpretation of these results
is a separate, subsequent step, done by the researcher, not by this
script.

Generated from `processed/analysis/{args.run_id}/RUN_MANIFEST.json`
(input SHA-256 `{manifest['input_sha256'][:16]}...`, git commit
`{(manifest.get('git_commit') or 'unknown')[:12]}`). Completed modules:
{', '.join(sorted(k for k, v in manifest['modules'].items() if v.get('status') == 'completed'))}.

"""

    out_dir = gac.module_run_dir(run_dir, MODULE_NAME)
    report_path = out_dir / "draft_report.md"
    report_path.write_text(header + body, encoding="utf-8")

    gac.write_module_config(out_dir, run_id=args.run_id, git_commit=gac.git_commit_hash())
    gac.update_run_manifest(run_dir, MODULE_NAME, "completed", output_dir=out_dir)

    print(f"\nDone. Draft report -> {report_path}")


if __name__ == "__main__":
    main()
