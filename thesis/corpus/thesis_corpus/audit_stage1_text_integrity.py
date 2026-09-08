"""Read-only Stage-1 text-integrity audit over every corpus's pages.jsonl.

Motivation: the 100-item fidelity review found garbled text ("di Y cult",
"Hervieu-Le ´ger") and tracing it showed the corruption is already present
in Stage 1's pages.jsonl -- born-digital PDFs whose fonts decode ligature
or accent glyphs wrongly, plus three documents whose ToUnicode maps are
broken outright (Latin-Extended-B/Greek cipher text, Private-Use-Area
runs). Stage 1's only quality gate (extraction.is_text_plausible) counts
characters, so all of these passed as "processed"/"native". Nothing in the
existing metadata records text *quality*.

This script measures it, per document, and is the single source of truth
for the document-level integrity flags the v2 extraction pilot reads
(pilot_v2_literature.py never hardcodes a known-problem list). It reads
processed/<corpus>/documents/<id>/pages.jsonl and metadata.json only, and
writes only under processed/audits/. It never modifies pages.jsonl, any
archive, or any retained run.

Line-level detection runs on the same text the chunker sees
(writer.collapse_whitespace applied per page, then NFC), so a flagged line
can be matched exactly against a chunk's lines later.

Outputs (refuses to overwrite an existing date tag):
  processed/audits/stage1_text_integrity_<date>.csv          one row per document
  processed/audits/stage1_text_integrity_<date>.config.json  thresholds, inputs, hashes
  processed/audits/stage1_text_integrity_<date>.regions.jsonl
      one row per flagged cipher line in documents flagged cmap_cipher /
      private_use_cipher -- the lookup source for chunk-level corrupted regions

Usage (from thesis/corpus/):
    python -m thesis_corpus.audit_stage1_text_integrity
    python -m thesis_corpus.audit_stage1_text_integrity --corpora literature --date-tag 20260909
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from thesis_corpus import text_integrity as ti
from thesis_corpus.writer import collapse_whitespace

SCRIPT_VERSION = "1.0.0"
CORPUS_DIR = Path(__file__).resolve().parent.parent
PROCESSED_ROOT = CORPUS_DIR / "processed"
AUDITS_DIR = PROCESSED_ROOT / "audits"
DEFAULT_CORPORA = ("literature", "miviludes", "interviews")

RATE_KEYS = ("ligature_pattern", "isolated_capital", "spaced_accent", "mixed_script_tokens")

CSV_COLUMNS = [
    "corpus", "document_id", "language", "extraction_method", "pages", "empty_pages",
    "chars", "words", "non_nfc_codepoints",
    *ti.CHARACTER_CLASS_KEYS,
    "cipher_lines",
    *[f"{k}_per_1k_words" for k in RATE_KEYS],
    "document_integrity_flag", "notes", "pages_jsonl_sha256",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=CORPUS_DIR, capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def read_pages(pages_path: Path) -> list[dict]:
    pages = []
    with open(pages_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pages.append(json.loads(line))
    return pages


def read_metadata(doc_dir: Path) -> dict:
    path = doc_dir / "metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def audit_document(corpus: str, doc_dir: Path) -> tuple[dict, list[dict]]:
    """Returns (csv_row, region_rows) for one document."""
    pages = read_pages(doc_dir / "pages.jsonl")
    metadata = read_metadata(doc_dir)

    totals = {key: 0 for key in ti.CHARACTER_CLASS_KEYS}
    chars = words = empty_pages = non_nfc = 0
    cipher_rows: list[dict] = []

    for page in sorted(pages, key=lambda p: p["page_number"]):
        raw = page.get("text") or ""
        if not raw.strip():
            empty_pages += 1
            continue
        collapsed = collapse_whitespace(raw)
        text, changed = ti.nfc_normalize(collapsed)
        non_nfc += changed
        chars += len(text)
        words += len(text.split())
        counts = ti.count_character_classes(text)
        for key, value in counts.items():
            if key == "private_use_max_run":
                totals[key] = max(totals[key], value)
            else:
                totals[key] += value
        for line_index, share, mixed, line_text in ti.find_cipher_lines(text):
            cipher_rows.append({
                "corpus": corpus,
                "document_id": doc_dir.name,
                "page_number": page["page_number"],
                "line_index_in_page": line_index,
                "suspicious_letter_share": round(share, 4),
                "mixed_script_tokens": mixed,
                "line_text": line_text,
            })

    hard_flags, notes = ti.document_flags(totals, words, len(cipher_rows))
    per_1k = 1000.0 / max(words, 1)

    row = {
        "corpus": corpus,
        "document_id": doc_dir.name,
        "language": metadata.get("language") or "",
        "extraction_method": metadata.get("extraction_method") or "",
        "pages": len(pages),
        "empty_pages": empty_pages,
        "chars": chars,
        "words": words,
        "non_nfc_codepoints": non_nfc,
        **totals,
        "cipher_lines": len(cipher_rows),
        **{f"{k}_per_1k_words": round(totals[k] * per_1k, 3) for k in RATE_KEYS},
        "document_integrity_flag": ";".join(hard_flags),
        "notes": ";".join(notes),
        "pages_jsonl_sha256": sha256_file(doc_dir / "pages.jsonl"),
    }
    # Regions are only meaningful (and only used downstream) for documents
    # whose flags say the cipher signal is real, not a Greek quotation.
    region_rows = cipher_rows if ({"cmap_cipher", "private_use_cipher"} & set(hard_flags)) else []
    return row, region_rows


def discover_documents(corpus_dir: Path) -> list[Path]:
    documents_dir = corpus_dir / "documents"
    if not documents_dir.exists():
        return []
    return sorted(d for d in documents_dir.iterdir() if d.is_dir() and (d / "pages.jsonl").exists())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--out-dir", type=Path, default=AUDITS_DIR)
    parser.add_argument("--corpora", nargs="+", default=list(DEFAULT_CORPORA))
    parser.add_argument("--date-tag", default=datetime.now(timezone.utc).strftime("%Y%m%d"))
    args = parser.parse_args()

    stem = f"stage1_text_integrity_{args.date_tag}"
    csv_path = args.out_dir / f"{stem}.csv"
    config_path = args.out_dir / f"{stem}.config.json"
    regions_path = args.out_dir / f"{stem}.regions.jsonl"
    for path in (csv_path, config_path, regions_path):
        if path.exists():
            raise SystemExit(f"Refusing to overwrite existing audit output: {path} (pick another --date-tag)")

    rows: list[dict] = []
    regions: list[dict] = []
    per_corpus_counts: dict[str, int] = {}
    for corpus in args.corpora:
        corpus_dir = args.processed_root / corpus
        documents = discover_documents(corpus_dir)
        if not documents:
            raise SystemExit(f"No documents with pages.jsonl under {corpus_dir}")
        per_corpus_counts[corpus] = len(documents)
        for doc_dir in documents:
            row, region_rows = audit_document(corpus, doc_dir)
            rows.append(row)
            regions.extend(region_rows)
            print(f"[{corpus}] {doc_dir.name[:70]:<70} words={row['words']:>7} "
                  f"flags={row['document_integrity_flag'] or '-'} notes={row['notes'] or '-'}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with open(regions_path, "w", encoding="utf-8") as f:
        for region in regions:
            f.write(json.dumps(region, ensure_ascii=False) + "\n")

    flag_summary: dict[str, list[str]] = {}
    for row in rows:
        for flag in filter(None, row["document_integrity_flag"].split(";")):
            flag_summary.setdefault(flag, []).append(f"{row['corpus']}/{row['document_id']}")

    config = {
        "script": "thesis_corpus.audit_stage1_text_integrity",
        "script_version": SCRIPT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit_hash(),
        "processed_root": str(args.processed_root),
        "corpora": args.corpora,
        "documents_per_corpus": per_corpus_counts,
        "read_only": True,
        "text_seen_by_detectors": "writer.collapse_whitespace(page_text) then unicodedata NFC -- identical to what chunking.build_chunks feeds the model, so regions match chunk lines exactly",
        "document_flag_thresholds": ti.DOCUMENT_FLAG_THRESHOLDS,
        "cipher_line_detection": {
            "min_letters": ti.CIPHER_LINE_MIN_LETTERS,
            "suspicious_letter_share_threshold": ti.CIPHER_LINE_SHARE_THRESHOLD,
            "min_mixed_script_tokens": ti.CIPHER_LINE_MIN_MIXED_TOKENS,
            "rule": "line flagged if letters >= min_letters and (share >= threshold or mixed tokens >= min_mixed_script_tokens)",
            "suspicious_blocks": "Latin Extended-B U+0180-024F, Greek U+0370-03FF, Private Use Area U+E000-F8FF",
            "regions_written_only_for_flags": ["cmap_cipher", "private_use_cipher"],
        },
        "hard_flag_order": ti.HARD_FLAG_ORDER,
        "flag_summary": flag_summary,
        "outputs": {
            "csv": str(csv_path), "regions_jsonl": str(regions_path),
            "csv_sha256": sha256_file(csv_path), "regions_sha256": sha256_file(regions_path),
            "rows": len(rows), "region_rows": len(regions),
        },
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{len(rows)} documents audited; {len(regions)} cipher-region lines recorded.")
    print("Flag summary:")
    for flag in ti.HARD_FLAG_ORDER:
        docs = flag_summary.get(flag, [])
        print(f"  {flag:<24} {len(docs):>3}  {', '.join(d.split('/')[1][:40] for d in docs[:6])}{' ...' if len(docs) > 6 else ''}")
    print(f"CSV:     {csv_path}\nRegions: {regions_path}\nConfig:  {config_path}")


if __name__ == "__main__":
    main()
