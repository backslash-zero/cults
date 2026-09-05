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

Nearest concept_backbone/structural_concepts terms per expression corpus are in `analyze_global_structure/nearest_concept_backbone_terms.csv` and `nearest_structural_concepts_terms.csv` (kept as two separate tables, not merged).
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

**n={len(summary)} interviews reviewed: {n_resolved} resolved, {n_unavailable} unavailable. Exploratory/descriptive only -- no inferential statistics, no claim of representativeness.**

Each interview's first participant claim in response to the opening prompt (manually verified against the transcript -- see `interviews/metadata/initial_exemplars.csv`), analyzed geometrically: distance to the 17 MIVILUDES criteria, to the reference vocabularies, to each expression-corpus centroid, and nearest literature/MIVILUDES/other-interview expressions. Grouped by `exemplar_type` (a descriptive code, not an asserted natural category). Full tables: `analyze_initial_exemplars/*.csv`.

{md_table(summary, ["document_id", "exemplar_type", "status", "source_expression_label"], max_rows=26)}
"""


def section_criterion_neighbours(run_dir: Path) -> str:
    d = run_dir / "analyze_criterion_neighbours"
    if not d.exists():
        return ""
    ordering = read_csv_rows(d / "language_sensitivity_ordering_comparison.csv")
    n_sensitive = sum(1 for r in ordering if r.get("language_sensitive") == "True")
    return f"""## Criterion neighbours

Full qualitative-retrieval and controlled-comparison tables (French-primary, 394-D shared space) are in `analyze_criterion_neighbours/qualitative_retrieval_french_primary.csv` and `controlled_comparison_french_primary.csv`. See `generate_figures/criterion_corpus_heatmap.png` for the criterion x corpus heatmap (built from the controlled comparison, not the raw qualitative distances).

**Language-representation sensitivity audit**: {n_sensitive}/{len(ordering)} criteria show a *different* nearest corpus between the French-primary (shared-space, Euclidean) and English-sensitivity (raw bge-m3, cosine) representations. French-primary and English-sensitivity results are kept in separate files throughout (`*_french_primary.csv` vs. `*_english_sensitivity.csv`) and never averaged into one number.

{md_table([r for r in ordering if r.get("language_sensitive") == "True"], ["criterion_key", "nearest_corpus_french_primary", "nearest_corpus_english_sensitivity"])}
"""


def section_emergent_entities(run_dir: Path) -> str:
    d = run_dir / "analyze_emergent_entities"
    if not d.exists():
        return ""
    provenance = read_csv_rows(d / "provenance_category_counts.csv")
    return f"""## Emergent entities

Provenance categories (which corpora mention each entity):

{md_table(provenance, ["provenance_category", "n_entities"])}

Full per-entity table (rank, mentions, provenance category, nearest MIVILUDES criterion, nearest structural concept) is in `analyze_emergent_entities/emergent_entities_full.csv`. This analysis does not classify entities as cult/mainstream or apply any other normative label -- see `analyze_emergent_entities/SCOPE_NOTICE.txt`.
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
