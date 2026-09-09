"""Extraction v2 -- literature-only pilot (one model arm, ~28 chunks).

Three stages, run in order:

  --stage dry-run      (Mac, no Ollama) select the pilot chunks from the v1
                       pages.jsonl with the unchanged chunker, resolve
                       integrity flags / corrupted regions from the Stage-1
                       audit output, pre-screen, and write
                       pilot_<date>/{pilot_selection.json, pilot_chunks.jsonl,
                       skipped_chunks.jsonl, pilot_v1_items.jsonl}.
  --stage rebuild-chunks (Ollama host, after a git pull) regenerates
                       pilot_chunks.jsonl -- gitignored because it carries
                       verbatim excerpts -- from the tracked
                       pilot_selection.json plus this machine's pages.jsonl,
                       and verifies its sha256 against the Mac build.
  --stage run          (Windows/Ollama host) one structured model call per
                       non-skipped chunk, deterministic screening, write
                       pilot_<date>/arm_<model-tag>/{config.json,
                       model_responses.jsonl, expressions_v2.jsonl,
                       rejected_candidates.jsonl, chunk_terms.jsonl,
                       model_failures.jsonl, summary.json}. Refuses to
                       overwrite an existing arm directory.
  --stage build-review (Mac) independent re-check of every arm, then the
                       manual-review packet under pilot_<date>/review/.

Never touches processed/literature/*, the shared space, retained analysis
runs, or thesis prose. Reads the v1 archive only to copy the v1 items of
the selected chunks (for the side-by-side review).

Usage (from thesis/corpus/):
    python -m thesis_corpus.pilot_v2_literature --stage dry-run --date-tag 20260909
    python -m thesis_corpus.pilot_v2_literature --stage run --date-tag 20260909 --model qwen3:4b
    python -m thesis_corpus.pilot_v2_literature --stage build-review --date-tag 20260909
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from thesis_corpus import extraction_v2_schema as schema
from thesis_corpus import screen_v2 as sv
from thesis_corpus.chunking import build_chunks

SCRIPT_VERSION = "1.0.0"
CORPUS_DIR = Path(__file__).resolve().parent.parent
PROCESSED_ROOT = CORPUS_DIR / "processed"
V1_LITERATURE_DIR = PROCESSED_ROOT / "literature"
V1_ARCHIVE_PATH = V1_LITERATURE_DIR / "criterion_expressions.jsonl"
AUDITS_DIR = PROCESSED_ROOT / "audits"
DEFAULT_PILOT_ROOT = PROCESSED_ROOT / "v2" / "literature"
SOURCE = "literature"
DEFAULT_MODEL = "qwen3:4b"
DEFAULT_SEED = 42
REVIEW_ROW_CAP = 150

# ---------------------------------------------------------------------------
# Pilot selection (approved plan, section 6)
# ---------------------------------------------------------------------------
DOC = {
    "lalich": "lalich-2004-bounded-choicetrue-believers-and-charismatic-cults",
    "oxford": "2008-the-oxford-handbook-of-new-religious-movements-1",
    "dawson": "dawson-2009-cults-and-new-religious-movements-a-reader",
    "davis": "davis-and-hankins-2002-new-religious-movements-and-religious-liberty-in-america",
    "melton": "melton-2014-encyclopedic-handbook-of-cults-in-america",
    "tomkins": "tomkins-2010-the-clapham-sect-how-wilberforce-s-circle-transformed-britain",
    "clarke": "clarke-2004-encyclopedia-of-new-religious-movements",
    "card": "card-2019-archaeology-and-new-religious-movements",
    "richardson": "richardson-2018-definitions-of-cult-from-sociological-technical-to-popular-negat",
}

# (document_id, chunk_index or "most_corrupted", failure class tested)
FORCED_CHUNKS = [
    (DOC["lalich"], 213, "ligature substitution ('di Y cult') -> integrity_ligature_substitution"),
    (DOC["oxford"], 194, "detached accent ('Hervieu-Le ´ger', flag) + off-topic 'cult apologist' sentence"),
    (DOC["dawson"], 133, "model truncation of a complete quote"),
    (DOC["dawson"], 135, "model rewrite (\"self 'actualization'\")"),
    (DOC["davis"], 105, "contextless acknowledgement ('Well, perhaps to some extent.')"),
    (DOC["melton"], 10, "heading extracted as expression ('The Course of Growth.')"),
    (DOC["tomkins"], 131, "relevance boundary (historical 'sect')"),
    (DOC["clarke"], 585, "encyclopedia entry-style short fragments"),
    (DOC["lalich"], 239, "overlong autobiographical / multiple claims"),
    (DOC["card"], "most_corrupted", "broken-CMap cipher text -> deterministic rejection only"),
]

# (document_id, random quota, role)
RANDOM_DOCS = [
    (DOC["richardson"], 3, "clean text, short article directly on defining 'cult'"),
    (DOC["dawson"], 3, "clean text, multi-author reader"),
    (DOC["lalich"], 3, "known font-encoding/ligature problem document"),
    (DOC["oxford"], 3, "detached-accent document, edited handbook"),
    (DOC["davis"], 3, "relevance-boundary (legal/policy prose)"),
    (DOC["clarke"], 3, "entry-style prose, v1's largest short-fragment source"),
]

MANUAL_REVIEW_COLUMNS = [
    "faithful", "self_contained", "cult_relevant", "textually_intelligible", "atomic",
    "attribution_correct", "claim_mode_correct", "epistemic_status_correct",
    "recommended_epistemic_status", "better_span_exists_in_chunk", "extraction_issue", "reviewer_notes",
]
EXTRACTION_ISSUE_VOCAB = ("none", "off_topic", "contextless_fragment", "truncated", "overlong", "multiple_claims",
                          "ocr_or_encoding_error", "heading_or_name", "interviewer_question", "other")
REVIEW_PREFILLED_COLUMNS = [
    "review_id", "arm", "selection_stratum", "document_id", "chunk_index", "page_range", "candidate_rank",
    "expression_kind", "verbatim_expression", "embedding_text", "embedding_equals_verbatim", "context_window",
    "attribution", "claim_mode", "epistemic_status", "entity_anchors", "relevance_note", "screen_flags",
    "chunk_integrity_score", "document_integrity_flag", "raw_archive_line",
]

logger = logging.getLogger("thesis_corpus.pilot_v2_literature")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_commit_hash() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=CORPUS_DIR, capture_output=True, text=True, timeout=5)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows) -> int:
    # newline="\n": on Windows, text mode would otherwise write "\r\n" and the
    # file's sha256 would differ from the Mac's although every row is identical.
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")


def sha256_lf_normalized(path: Path) -> str:
    """sha256 of the file with CRLF folded to LF -- used only to diagnose a
    mismatch as a line-ending difference rather than a content difference."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def model_tag(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "-", model)


def latest_audit_csv(audits_dir: Path) -> Path:
    candidates = sorted(audits_dir.glob("stage1_text_integrity_*.csv"))
    if not candidates:
        raise SystemExit(f"No stage1_text_integrity_*.csv under {audits_dir} -- run audit_stage1_text_integrity first")
    return candidates[-1]


def load_audit(audit_csv: Path) -> tuple[dict[str, list[str]], dict[str, set[str]], dict]:
    """document_id -> hard flags; document_id -> corrupted line texts; provenance."""
    flags: dict[str, list[str]] = {}
    with open(audit_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["corpus"] != SOURCE:
                continue
            flags[row["document_id"]] = [x for x in row["document_integrity_flag"].split(";") if x]
    regions_path = audit_csv.with_name(audit_csv.name.replace(".csv", ".regions.jsonl"))
    region_lines: dict[str, set[str]] = defaultdict(set)
    if regions_path.exists():
        for row in read_jsonl(regions_path):
            if row["corpus"] == SOURCE:
                region_lines[row["document_id"]].add(row["line_text"].strip())
    provenance = {
        "audit_csv": str(audit_csv), "audit_csv_sha256": sha256_file(audit_csv),
        "audit_regions_jsonl": str(regions_path) if regions_path.exists() else None,
        "audit_regions_sha256": sha256_file(regions_path) if regions_path.exists() else None,
    }
    return flags, region_lines, provenance


def read_pages(doc_dir: Path) -> list[dict]:
    return read_jsonl(doc_dir / "pages.jsonl")


# ---------------------------------------------------------------------------
# Stage: dry-run
# ---------------------------------------------------------------------------

def scan_v1_archive(document_ids: set[str], archive_path: Path) -> dict[str, dict[int, list[dict]]]:
    """v1 items (vectors dropped) for the given documents, keyed doc -> chunk -> items in archive order."""
    by_doc: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    with open(archive_path, encoding="utf-8") as f:
        for line_index, line in enumerate(f):
            # cheap prefilter on the document_id prefix of the JSON line
            head = line[:400]
            if not any(f'"document_id": "{d}"' in head for d in document_ids):
                continue
            item = json.loads(line)
            if item["document_id"] not in document_ids:
                continue
            by_doc[item["document_id"]][item["chunk_index"]].append({
                "raw_archive_line": line_index,
                "document_id": item["document_id"],
                "chunk_index": item["chunk_index"],
                "page_range": item.get("page_range"),
                "source_quote": item["source_quote"],
                "embedding_text": item["embedding_text"],
                "entity_anchors": item.get("entity_anchors", []),
                "claim_mode": item.get("claim_mode"),
                "epistemic_status": item.get("epistemic_status"),
                "attribution": item.get("attribution"),
                "context_window": item["context_window"],
            })
    return by_doc


def stage_dry_run(args, pilot_dir: Path) -> None:
    if pilot_dir.exists() and any(pilot_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty pilot directory {pilot_dir} (pick another --date-tag)")
    audit_csv = args.audit_csv or latest_audit_csv(AUDITS_DIR)
    doc_flags, region_lines, audit_prov = load_audit(audit_csv)
    document_ids = {d for d, _, _ in FORCED_CHUNKS} | {d for d, _, _ in RANDOM_DOCS}
    for document_id in sorted(document_ids):
        if document_id not in doc_flags:
            raise SystemExit(f"{document_id} not present in the Stage-1 audit {audit_csv}")
        if not (V1_LITERATURE_DIR / "documents" / document_id / "pages.jsonl").exists():
            raise SystemExit(f"pages.jsonl missing for {document_id}")

    print(f"Scanning v1 archive for {len(document_ids)} documents: {V1_ARCHIVE_PATH}")
    v1_by_doc = scan_v1_archive(document_ids, V1_ARCHIVE_PATH)

    contexts: dict[str, dict[int, sv.ChunkContext]] = {}
    chunk_counts: dict[str, int] = {}
    pages_sha: dict[str, str] = {}
    for document_id in sorted(document_ids):
        doc_dir = V1_LITERATURE_DIR / "documents" / document_id
        pages_sha[document_id] = sha256_file(doc_dir / "pages.jsonl")
        chunks = build_chunks(document_id, read_pages(doc_dir))
        chunk_counts[document_id] = len(chunks)
        contexts[document_id] = {
            c.chunk_index: sv.prepare_chunk(
                document_id, c.chunk_index, c.page_range, c.text, SOURCE,
                document_integrity_flags=doc_flags[document_id],
                corrupted_line_texts=frozenset(region_lines.get(document_id, ())),
            ) for c in chunks
        }
        v1_max = max(v1_by_doc[document_id].keys(), default=-1)
        if v1_max >= len(chunks):
            raise SystemExit(f"{document_id}: v1 archive references chunk {v1_max} but the chunker produced {len(chunks)} chunks")

    selected: list[tuple[str, int, str, str]] = []  # (doc, chunk, stratum, note)
    forced_keys: set[tuple[str, int]] = set()
    for document_id, chunk_index, note in FORCED_CHUNKS:
        if chunk_index == "most_corrupted":
            ranked = sorted(contexts[document_id].values(), key=lambda c: (-len(c.corrupted_regions), c.chunk_index))
            if not ranked or not ranked[0].corrupted_regions:
                raise SystemExit(f"{document_id}: no chunk with corrupted regions -- audit regions did not resolve")
            chunk_index = ranked[0].chunk_index
            note = f"{note} (resolved chunk {chunk_index}: {len(ranked[0].corrupted_regions)} corrupted lines)"
        if chunk_index not in contexts[document_id]:
            raise SystemExit(f"{document_id}: forced chunk {chunk_index} does not exist")
        selected.append((document_id, chunk_index, "forced_audit_regression", note))
        forced_keys.add((document_id, chunk_index))

    for doc_index, (document_id, quota, role) in enumerate(RANDOM_DOCS):
        pool = sorted(ci for ci in v1_by_doc[document_id] if (document_id, ci) not in forced_keys)
        if len(pool) < quota:
            raise SystemExit(f"{document_id}: only {len(pool)} v1 chunks available for a random quota of {quota}")
        rng = random.Random(args.seed * 1000 + doc_index)
        for chunk_index in sorted(rng.sample(pool, quota)):
            selected.append((document_id, chunk_index, "random", role))

    selected_meta = []
    v1_rows, mismatches = [], []
    for document_id, chunk_index, stratum, note in selected:
        ctx = contexts[document_id][chunk_index]
        v1_items = v1_by_doc[document_id].get(chunk_index, [])
        for item in v1_items:
            if ctx.nfc_changed_codepoints == 0 and item["context_window"] != ctx.raw_text:
                mismatches.append((document_id, chunk_index))
                break
        selected_meta.append({"document_id": document_id, "chunk_index": chunk_index, "selection_stratum": stratum,
                              "selection_note": note, "v1_item_count": len(v1_items)})
        v1_rows.extend(v1_items)
    if mismatches:
        raise SystemExit(f"Chunk text differs from v1 context_window for {mismatches} -- chunker no longer reproduces v1")

    chunk_rows, skipped_rows = build_chunk_rows(selected_meta, contexts, region_lines)
    pilot_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(pilot_dir / "pilot_chunks.jsonl", chunk_rows)
    write_jsonl(pilot_dir / "skipped_chunks.jsonl", skipped_rows)
    write_jsonl(pilot_dir / "pilot_v1_items.jsonl", v1_rows)
    try:
        audit_prov["audit_csv_relative"] = str(Path(audit_csv).resolve().relative_to(CORPUS_DIR))
    except ValueError:
        audit_prov["audit_csv_relative"] = None
    selection = {
        "script": "thesis_corpus.pilot_v2_literature", "script_version": SCRIPT_VERSION, "stage": "dry-run",
        "generated_at": datetime.now(timezone.utc).isoformat(), "git_commit": git_commit_hash(),
        "extraction_version": schema.EXTRACTION_VERSION, "prompt_sha256": schema.PROMPT_SHA256,
        "source": SOURCE, "seed": args.seed,
        "stage1_audit": audit_prov,
        "pre_screen_config": schema.PRE_SCREEN_BY_SOURCE[SOURCE],
        "forced_chunks": [{"document_id": d, "chunk_index": c, "note": n} for d, c, s, n in selected if s == "forced_audit_regression"],
        "random_docs": [{"document_id": d, "quota": q, "role": r, "sub_seed": args.seed * 1000 + i,
                         "pool": "chunk indices with >= 1 v1 item, forced chunks excluded"}
                        for i, (d, q, r) in enumerate(RANDOM_DOCS)],
        "selected_chunks": selected_meta,
        "pilot_chunks_sha256": sha256_file(pilot_dir / "pilot_chunks.jsonl"),
        "skipped_chunks_sha256": sha256_file(pilot_dir / "skipped_chunks.jsonl"),
        "rebuild": "on a machine without pilot_chunks.jsonl (it is gitignored: verbatim excerpts), run "
                   "--stage rebuild-chunks; it regenerates the file from pages.jsonl + this selection and "
                   "verifies the sha256 above",
        "n_selected": len(selected), "n_skipped_pre_screen": len(skipped_rows),
        "n_forced": sum(1 for s in selected if s[2] == "forced_audit_regression"),
        "n_random": sum(1 for s in selected if s[2] == "random"),
        "documents": {d: {"pages_jsonl_sha256": pages_sha[d], "chunk_count": chunk_counts[d],
                          "document_integrity_flags": doc_flags[d], "v1_chunks_with_items": len(v1_by_doc[d])}
                      for d in sorted(document_ids)},
        "v1_archive": {"path": str(V1_ARCHIVE_PATH), "items_copied": len(v1_rows)},
        "chunk_text_equals_v1_context_window": "verified for every selected chunk with nfc_changed_codepoints == 0",
    }
    write_json(pilot_dir / "pilot_selection.json", selection)

    print(f"\nSelected {len(selected)} chunks ({selection['n_forced']} forced, {selection['n_random']} random); "
          f"{len(skipped_rows)} skipped by pre-screen; {len(v1_rows)} v1 items copied.")
    for row in chunk_rows:
        print(f"  [{row['selection_stratum'][:6]}] {row['document_id'][:45]:<45} chunk {row['chunk_index']:>4} "
              f"words={row['word_count']:>4} v1={row['v1_item_count']:>2} flags={';'.join(row['document_integrity_flags']) or '-'} "
              f"regions={len(row['corrupted_regions'])} nfc_changed={row['nfc_changed_codepoints']} skip={row['pre_screen_skip_code'] or '-'}")
    print(f"Pilot dir: {pilot_dir}")


def build_chunk_rows(selected_meta: list[dict], contexts: dict[str, dict[int, sv.ChunkContext]],
                     region_lines: dict[str, set[str]]) -> tuple[list[dict], list[dict]]:
    """Deterministic pilot_chunks.jsonl / skipped_chunks.jsonl rows from the
    selection metadata and rebuilt chunk contexts (shared by dry-run and
    rebuild-chunks so both machines produce byte-identical files)."""
    pre_screen_config = schema.PRE_SCREEN_BY_SOURCE[SOURCE]
    chunk_rows, skipped_rows = [], []
    for meta in selected_meta:
        document_id, chunk_index = meta["document_id"], meta["chunk_index"]
        ctx = contexts[document_id][chunk_index]
        skip_code, skip_detail = sv.pre_screen_chunk(ctx, pre_screen_config)
        chunk_rows.append({
            "document_id": document_id, "chunk_index": chunk_index, "page_range": ctx.page_range,
            "selection_stratum": meta["selection_stratum"], "selection_note": meta["selection_note"],
            "raw_text": ctx.raw_text, "nfc_text": ctx.nfc_text,
            "raw_chunk_sha256": ctx.raw_sha256, "nfc_chunk_sha256": ctx.nfc_sha256,
            "nfc_changed_codepoints": ctx.nfc_changed_codepoints,
            "word_count": ctx.word_count,
            "document_integrity_flags": ctx.document_integrity_flags,
            "corrupted_line_texts": sorted(region_lines.get(document_id, ())),
            "corrupted_regions": ctx.corrupted_regions,
            "chunk_integrity_score": ctx.chunk_integrity_score,
            "v1_item_count": meta["v1_item_count"],
            "pre_screen_skip_code": skip_code,
        })
        if skip_code:
            skipped_rows.append({"document_id": document_id, "chunk_index": chunk_index, "skip_code": skip_code,
                                 "detail": skip_detail, "selection_stratum": meta["selection_stratum"]})
    return chunk_rows, skipped_rows


# ---------------------------------------------------------------------------
# Stage: rebuild-chunks (any machine with pages.jsonl; no v1 archive needed)
# ---------------------------------------------------------------------------

def stage_rebuild_chunks(args, pilot_dir: Path) -> None:
    selection_path = pilot_dir / "pilot_selection.json"
    if not selection_path.exists():
        raise SystemExit(f"{selection_path} missing -- pull the repository (it is tracked) or run --stage dry-run on the Mac")
    if (pilot_dir / "pilot_chunks.jsonl").exists():
        raise SystemExit(f"{pilot_dir / 'pilot_chunks.jsonl'} already exists; nothing to rebuild")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    audit_rel = selection["stage1_audit"].get("audit_csv_relative")
    audit_csv = args.audit_csv or (CORPUS_DIR / audit_rel if audit_rel else None)
    if audit_csv is None or not Path(audit_csv).exists():
        raise SystemExit(f"Stage-1 audit CSV not found ({audit_csv}); pass --audit-csv")
    if sha256_file(Path(audit_csv)) != selection["stage1_audit"]["audit_csv_sha256"]:
        raise SystemExit(f"{audit_csv} sha256 differs from the one recorded in pilot_selection.json")
    doc_flags, region_lines, _ = load_audit(Path(audit_csv))

    contexts: dict[str, dict[int, sv.ChunkContext]] = {}
    for document_id, info in selection["documents"].items():
        doc_dir = V1_LITERATURE_DIR / "documents" / document_id
        if not (doc_dir / "pages.jsonl").exists():
            raise SystemExit(f"pages.jsonl missing for {document_id} under {doc_dir}")
        actual = sha256_file(doc_dir / "pages.jsonl")
        if actual != info["pages_jsonl_sha256"]:
            raise SystemExit(f"{document_id}: pages.jsonl sha256 {actual[:12]} != recorded {info['pages_jsonl_sha256'][:12]} "
                             "-- this machine's Stage-1 output differs from the Mac's")
        if doc_flags.get(document_id, []) != info["document_integrity_flags"]:
            raise SystemExit(f"{document_id}: audit flags differ from pilot_selection.json")
        chunks = build_chunks(document_id, read_pages(doc_dir))
        if len(chunks) != info["chunk_count"]:
            raise SystemExit(f"{document_id}: chunker produced {len(chunks)} chunks, selection recorded {info['chunk_count']}")
        contexts[document_id] = {
            c.chunk_index: sv.prepare_chunk(document_id, c.chunk_index, c.page_range, c.text, SOURCE,
                                            document_integrity_flags=doc_flags[document_id],
                                            corrupted_line_texts=frozenset(region_lines.get(document_id, ())))
            for c in chunks
        }
    chunk_rows, skipped_rows = build_chunk_rows(selection["selected_chunks"], contexts, region_lines)
    write_jsonl(pilot_dir / "pilot_chunks.jsonl", chunk_rows)
    write_jsonl(pilot_dir / "skipped_chunks.jsonl", skipped_rows)
    for name, key in (("pilot_chunks.jsonl", "pilot_chunks_sha256"), ("skipped_chunks.jsonl", "skipped_chunks_sha256")):
        actual = sha256_file(pilot_dir / name)
        if actual != selection[key]:
            hint = ("line endings only (CRLF vs LF) -- update the code, this writer must emit LF"
                    if sha256_lf_normalized(pilot_dir / name) == selection[key] else "content differs")
            (pilot_dir / name).unlink()
            raise SystemExit(f"Rebuilt {name} sha256 {actual[:12]} != recorded {selection[key][:12]} ({hint}); "
                             "file removed -- do not run the pilot")
    print(f"Rebuilt {len(chunk_rows)} chunks ({len(skipped_rows)} skipped) -> {pilot_dir / 'pilot_chunks.jsonl'}; sha256 verified against pilot_selection.json")


# ---------------------------------------------------------------------------
# Stage: run (model calls + screening)
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def _parse_envelope(raw: str):
    match = _CODE_FENCE_RE.match(raw.strip())
    content = match.group(1) if match else raw
    parsed = json.loads(content)
    return schema.ChunkEnvelopeV2.model_validate(parsed)


def rebuild_context(row: dict) -> sv.ChunkContext:
    ctx = sv.prepare_chunk(
        row["document_id"], row["chunk_index"], row["page_range"], row["raw_text"], SOURCE,
        document_integrity_flags=row["document_integrity_flags"],
        corrupted_line_texts=frozenset(row["corrupted_line_texts"]),
    )
    if ctx.nfc_sha256 != row["nfc_chunk_sha256"] or ctx.raw_sha256 != row["raw_chunk_sha256"]:
        raise SystemExit(f"{row['document_id']}:{row['chunk_index']}: chunk text hash mismatch against pilot_chunks.jsonl")
    if ctx.corrupted_regions != [tuple(r) for r in row["corrupted_regions"]]:
        raise SystemExit(f"{row['document_id']}:{row['chunk_index']}: corrupted regions differ from pilot_chunks.jsonl")
    return ctx


def stage_run(args, pilot_dir: Path) -> None:
    from thesis_corpus.ollama_client import AnnotationError, chat_structured, check_available, OllamaUnavailableError

    chunks_path = pilot_dir / "pilot_chunks.jsonl"
    if not chunks_path.exists():
        raise SystemExit(f"{chunks_path} missing -- run --stage dry-run (Mac) or --stage rebuild-chunks (this machine) first")
    arm_dir = pilot_dir / f"arm_{args.arm_tag or model_tag(args.model)}"
    if arm_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing arm directory {arm_dir}")
    try:
        check_available(args.ollama_host)
    except OllamaUnavailableError as e:
        raise SystemExit(str(e))

    rows = read_jsonl(chunks_path)
    todo = [r for r in rows if not r["pre_screen_skip_code"]]
    arm_dir.mkdir(parents=True)
    log_path = arm_dir / "run.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()])
    started = datetime.now(timezone.utc)
    json_schema = schema.response_json_schema()

    responses, retained_all, rejected_all, terms_all, failures = [], [], [], [], []
    seen_by_doc: dict[str, set[str]] = defaultdict(set)
    modes = Counter()
    for i, row in enumerate(todo, start=1):
        ctx = rebuild_context(row)
        user = schema.user_message(ctx.document_id, ctx.page_range, ctx.chunk_index, ctx.nfc_text)
        envelope, raw_content, mode, error = None, None, None, None
        for attempt in (1, 2):
            content = user if attempt == 1 else user + "\n\nReturn only valid JSON matching the schema, no commentary."
            try:
                result = chat_structured(args.ollama_host, args.model, schema.SYSTEM_PROMPT_V2, content,
                                         json_schema, schema.OLLAMA_OPTIONS, think=schema.OLLAMA_THINK, timeout=args.timeout)
                raw_content, mode = result.content, result.structured_output_mode
                envelope = _parse_envelope(raw_content)
                break
            except AnnotationError as e:
                error = f"transport: {e}"
            except (json.JSONDecodeError, ValueError) as e:
                error = f"invalid JSON / envelope: {e}"
        responses.append({"document_id": ctx.document_id, "chunk_index": ctx.chunk_index, "model": args.model,
                          "structured_output_mode": mode, "raw_content": raw_content, "error": None if envelope else error})
        if envelope is None:
            failures.append({"document_id": ctx.document_id, "chunk_index": ctx.chunk_index, "error": error})
            logger.error("[%s:%d] model failure after retry: %s", ctx.document_id, ctx.chunk_index, error)
            continue
        modes[mode] += 1
        retained, rejected = sv.screen_chunk(envelope.expressions, ctx, seen_by_doc[ctx.document_id])
        kept_terms, dropped_terms = sv.screen_domain_terms(envelope.domain_terms, ctx)
        for record in retained:
            record.update({"extraction_version": schema.EXTRACTION_VERSION, "prompt_sha256": schema.PROMPT_SHA256,
                           "model": args.model, "structured_output_mode": mode,
                           "selection_stratum": row["selection_stratum"], "chunk_relevance": envelope.chunk_relevance})
        for record in rejected:
            record["selection_stratum"] = row["selection_stratum"]
        retained_all.extend(retained)
        rejected_all.extend(rejected)
        terms_all.append({"document_id": ctx.document_id, "chunk_index": ctx.chunk_index,
                          "domain_terms": kept_terms, "dropped_terms": dropped_terms})
        logger.info("[%d/%d] %s:%d relevance=%s emitted=%d retained=%d rejected=%d terms=%d mode=%s",
                    i, len(todo), ctx.document_id[:40], ctx.chunk_index, envelope.chunk_relevance,
                    len(envelope.expressions), len(retained), len(rejected), len(kept_terms), mode)

    write_jsonl(arm_dir / "model_responses.jsonl", responses)
    write_jsonl(arm_dir / "expressions_v2.jsonl", retained_all)
    write_jsonl(arm_dir / "rejected_candidates.jsonl", rejected_all)
    write_jsonl(arm_dir / "chunk_terms.jsonl", terms_all)
    write_jsonl(arm_dir / "model_failures.jsonl", failures)
    summary = build_summary(rows, todo, retained_all, rejected_all, failures, args.model)
    summary["started_at"], summary["finished_at"] = started.isoformat(), datetime.now(timezone.utc).isoformat()
    write_json(arm_dir / "summary.json", summary)
    config = {
        "script": "thesis_corpus.pilot_v2_literature", "script_version": SCRIPT_VERSION, "stage": "run",
        "model": args.model, "arm": arm_dir.name, "ollama_host": args.ollama_host, "timeout_s": args.timeout,
        "structured_output_modes_observed": dict(modes),
        "git_commit": git_commit_hash(), "platform": sys.platform,
        "pilot_chunks_sha256": sha256_file(chunks_path), "n_chunks_called": len(todo),
        "screening": schema.screening_constants(),
        "seed": args.seed, "started_at": started.isoformat(), "finished_at": summary["finished_at"],
    }
    write_json(arm_dir / "config.json", config)
    print(json.dumps({k: summary[k] for k in ("chunks", "candidates", "rejection_codes", "model_failure_count", "model_failure_rate")},
                     ensure_ascii=False, indent=2))
    print(f"Arm dir: {arm_dir}")


def build_summary(rows, todo, retained, rejected, failures, model) -> dict:
    per_chunk = {(r["document_id"], r["chunk_index"]): {"v1": r["v1_item_count"], "v2_retained": 0, "v2_rejected": 0,
                                                       "stratum": r["selection_stratum"], "skipped": bool(r["pre_screen_skip_code"])}
                 for r in rows}
    for r in retained:
        per_chunk[(r["document_id"], r["chunk_index"])]["v2_retained"] += 1
    for r in rejected:
        per_chunk[(r["document_id"], r["chunk_index"])]["v2_rejected"] += 1
    failed_keys = {(f["document_id"], f["chunk_index"]) for f in failures}
    emitted = len(retained) + len(rejected)
    per_doc = defaultdict(lambda: {"chunks": 0, "v1": 0, "v2_retained": 0, "v2_rejected": 0})
    for (doc, _), v in per_chunk.items():
        d = per_doc[doc]
        d["chunks"] += 1
        d["v1"] += v["v1"]
        d["v2_retained"] += v["v2_retained"]
        d["v2_rejected"] += v["v2_rejected"]
    called = len(todo)
    return {
        "model": model,
        "chunks": {"selected": len(rows), "skipped_pre_screen": len(rows) - called, "called": called,
                   "annotated": called - len(failures), "model_failed": len(failures),
                   "reconciled": (len(rows) - called) + (called - len(failures)) + len(failures) == len(rows)},
        "candidates": {"emitted": emitted, "retained": len(retained), "rejected": len(rejected),
                       "reconciled": emitted == len(retained) + len(rejected)},
        "rejection_codes": dict(Counter(r["rejection_code"] for r in rejected).most_common()),
        "screen_flags_on_retained": dict(Counter(f for r in retained for f in r["screen_flags"]).most_common()),
        "expression_kinds_retained": dict(Counter(r["expression_kind"] for r in retained)),
        "model_failure_count": len(failures),
        "model_failure_rate": round(len(failures) / called, 4) if called else None,
        "model_failure_rate_review_blocker_threshold": 0.03,
        "retained_from_failed_chunks": sum(1 for r in retained if (r["document_id"], r["chunk_index"]) in failed_keys),
        "per_chunk": [{"document_id": d, "chunk_index": c, **v} for (d, c), v in sorted(per_chunk.items())],
        "per_document": dict(per_doc),
        "retained_per_called_chunk": round(len(retained) / max(called - len(failures), 1), 3),
        "v1_items_on_called_chunks": sum(v["v1"] for k, v in per_chunk.items() if not v["skipped"]),
    }


# ---------------------------------------------------------------------------
# Stage: build-review (independent re-check + packet)
# ---------------------------------------------------------------------------

def validate_arm(arm_dir: Path, chunk_rows: list[dict]) -> dict:
    """Hard-zero conditions re-evaluated from scratch (never trusting the run)."""
    contexts = {(r["document_id"], r["chunk_index"]): rebuild_context(r) for r in chunk_rows}
    retained = read_jsonl(arm_dir / "expressions_v2.jsonl")
    rejected = read_jsonl(arm_dir / "rejected_candidates.jsonl")
    failures = read_jsonl(arm_dir / "model_failures.jsonl")
    responses = read_jsonl(arm_dir / "model_responses.jsonl")
    summary = json.loads((arm_dir / "summary.json").read_text(encoding="utf-8"))
    problems: list[str] = []
    for record in retained:
        ctx = contexts[(record["document_id"], record["chunk_index"])]
        for p in sv.recheck_retained_record(record, ctx):
            problems.append(f"{record['document_id']}:{record['chunk_index']} rank {record['candidate_rank']}: {p}")
        # independent re-screen of the retained candidate
        cand, code, detail = schema.validate_candidate({
            k: record[k] for k in ("expression_kind", "attribution", "claim_mode", "epistemic_status", "entity_anchors",
                                   "self_contained", "cult_relevant", "textually_intelligible", "single_coherent_expression",
                                   "relevance_note")} | {"verbatim_expression": record["verbatim_expression"]})
        outcome = sv.screen_candidate(cand, ctx) if cand else None
        if cand is None or outcome.rejection_code:
            problems.append(f"{record['document_id']}:{record['chunk_index']} rank {record['candidate_rank']}: "
                            f"re-screen rejects with {code or outcome.rejection_code}")
    failed_keys = {(f["document_id"], f["chunk_index"]) for f in failures}
    for record in retained:
        if (record["document_id"], record["chunk_index"]) in failed_keys:
            problems.append(f"retained row from failed chunk {record['document_id']}:{record['chunk_index']}")
    # accounting
    emitted_from_responses = 0
    for resp in responses:
        if resp["error"] is None:
            emitted_from_responses += len(_parse_envelope(resp["raw_content"]).expressions)
    if emitted_from_responses != len(retained) + len(rejected):
        problems.append(f"accounting: {emitted_from_responses} emitted in responses != {len(retained)} retained + {len(rejected)} rejected")
    called = sum(1 for r in chunk_rows if not r["pre_screen_skip_code"])
    if len(responses) != called:
        problems.append(f"accounting: {len(responses)} responses != {called} called chunks")
    if len([r for r in responses if r["error"]]) != len(failures):
        problems.append("accounting: failed responses != model_failures.jsonl")
    for f in failures:
        if not any(r["document_id"] == f["document_id"] and r["chunk_index"] == f["chunk_index"] for r in responses):
            problems.append(f"failure {f} not in model_responses")
    return {
        "arm": arm_dir.name, "retained": len(retained), "rejected": len(rejected), "responses": len(responses),
        "model_failures": len(failures), "model_failure_rate": summary.get("model_failure_rate"),
        "failure_rate_exceeds_review_blocker": (summary.get("model_failure_rate") or 0) > 0.03,
        "hard_zero_problems": problems,
    }


def stage_build_review(args, pilot_dir: Path) -> None:
    chunk_rows = read_jsonl(pilot_dir / "pilot_chunks.jsonl")
    v1_items = read_jsonl(pilot_dir / "pilot_v1_items.jsonl")
    arms = sorted(p for p in pilot_dir.glob("arm_*") if (p / "expressions_v2.jsonl").exists())
    if not arms:
        raise SystemExit(f"No arm_*/expressions_v2.jsonl under {pilot_dir} -- run --stage run first")
    review_dir = pilot_dir / "review"
    if review_dir.exists() and any(review_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite existing review packet {review_dir}")

    validations = [validate_arm(arm, chunk_rows) for arm in arms]
    for v in validations:
        for p in v["hard_zero_problems"]:
            print("HARD-ZERO VIOLATION:", p)
    if any(v["hard_zero_problems"] for v in validations):
        raise SystemExit("Hard-zero validation failed; review packet not built.")

    chunk_index_map = {(r["document_id"], r["chunk_index"]): r for r in chunk_rows}
    order = {("random", ): 0, ("forced_audit_regression", ): 1}
    rng = random.Random(args.seed)

    v1_by_chunk: dict[tuple, list[dict]] = defaultdict(list)
    for item in v1_items:
        v1_by_chunk[(item["document_id"], item["chunk_index"])].append(item)
    arm_rows: dict[str, dict[tuple, list[dict]]] = {}
    for arm in arms:
        by_chunk: dict[tuple, list[dict]] = defaultdict(list)
        for rec in read_jsonl(arm / "expressions_v2.jsonl"):
            by_chunk[(rec["document_id"], rec["chunk_index"])].append(rec)
        arm_rows[arm.name] = by_chunk

    def v1_sample(items: list[dict]) -> list[dict]:
        if len(items) <= args.v1_sample_per_chunk:
            return items
        rng_local = random.Random(args.seed * 7919 + items[0]["raw_archive_line"])
        return sorted(rng_local.sample(items, args.v1_sample_per_chunk), key=lambda x: x["raw_archive_line"])

    # v2 rows: all retained unless the packet would exceed the cap
    total_v2 = sum(len(v) for by in arm_rows.values() for v in by.values())
    total_v1 = sum(len(v1_sample(v)) for v in v1_by_chunk.values())
    v2_cap_per_chunk = None
    if total_v2 + total_v1 > args.review_row_cap:
        v2_cap_per_chunk = args.v2_sample_per_chunk

    review_rows: list[dict] = []
    sampling_note = []
    sorted_chunks = sorted(chunk_rows, key=lambda r: (0 if r["selection_stratum"] == "random" else 1, r["document_id"], r["chunk_index"]))
    for crow in sorted_chunks:
        key = (crow["document_id"], crow["chunk_index"])
        rows_here: list[tuple[str, dict]] = []
        for item in v1_sample(v1_by_chunk.get(key, [])):
            rows_here.append(("v1", {
                "candidate_rank": "", "expression_kind": "", "verbatim_expression": item["source_quote"],
                "embedding_text": item["embedding_text"], "context_window": item["context_window"],
                "attribution": item["attribution"], "claim_mode": item["claim_mode"], "epistemic_status": item["epistemic_status"],
                "entity_anchors": "; ".join(item.get("entity_anchors") or []), "relevance_note": "", "screen_flags": "",
                "raw_archive_line": item["raw_archive_line"], "page_range": item.get("page_range"),
            }))
        for arm_name, by_chunk in arm_rows.items():
            recs = by_chunk.get(key, [])
            if v2_cap_per_chunk is not None and len(recs) > v2_cap_per_chunk:
                recs = sorted(random.Random(args.seed * 104729 + crow["chunk_index"]).sample(recs, v2_cap_per_chunk),
                              key=lambda x: x["candidate_rank"])
                sampling_note.append(f"{arm_name} {key}: {len(by_chunk[key])} -> {v2_cap_per_chunk}")
            for rec in recs:
                rows_here.append((f"v2_{arm_name[4:]}", {
                    "candidate_rank": rec["candidate_rank"], "expression_kind": rec["expression_kind"],
                    "verbatim_expression": rec["verbatim_expression"], "embedding_text": rec["embedding_text"],
                    "context_window": rec["context_window"], "attribution": rec["attribution"], "claim_mode": rec["claim_mode"],
                    "epistemic_status": rec["epistemic_status"], "entity_anchors": "; ".join(rec["entity_anchors"]),
                    "relevance_note": rec["relevance_note"], "screen_flags": ";".join(rec["screen_flags"]),
                    "raw_archive_line": "", "page_range": rec["page_range"],
                }))
        for arm_label, payload in rows_here:
            review_rows.append({
                "review_id": None, "arm": arm_label, "selection_stratum": crow["selection_stratum"],
                "document_id": crow["document_id"], "chunk_index": crow["chunk_index"],
                "page_range": "-".join(str(p) for p in (payload["page_range"] or crow["page_range"])),
                **{k: payload[k] for k in ("candidate_rank", "expression_kind", "verbatim_expression", "embedding_text")},
                "embedding_equals_verbatim": "yes" if payload["embedding_text"] == payload["verbatim_expression"] else "no",
                "context_window": payload["context_window"],
                **{k: payload[k] for k in ("attribution", "claim_mode", "epistemic_status", "entity_anchors", "relevance_note", "screen_flags")},
                "chunk_integrity_score": crow["chunk_integrity_score"],
                "document_integrity_flag": ";".join(crow["document_integrity_flags"]),
                "raw_archive_line": payload["raw_archive_line"],
                **{c: "" for c in MANUAL_REVIEW_COLUMNS},
            })
    for i, row in enumerate(review_rows, start=1):
        row["review_id"] = i
    if len(review_rows) > args.review_row_cap:
        raise SystemExit(f"Review packet has {len(review_rows)} rows > cap {args.review_row_cap} even after v2 sampling")

    review_dir.mkdir(parents=True)
    key_rows = []
    columns = REVIEW_PREFILLED_COLUMNS + MANUAL_REVIEW_COLUMNS
    if args.blind:
        key_rows = [{"review_id": r["review_id"], "arm": r["arm"]} for r in review_rows]
        for r in review_rows:
            r["arm"] = ""
        with open(review_dir / "review_key.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["review_id", "arm"])
            w.writeheader()
            w.writerows(key_rows)
    with open(review_dir / "pilot_review.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(review_rows)

    # chunk comparison
    comparison_rows = []
    for crow in sorted_chunks:
        key = (crow["document_id"], crow["chunk_index"])
        row = {"document_id": key[0], "chunk_index": key[1], "selection_stratum": crow["selection_stratum"],
               "selection_note": crow["selection_note"], "pre_screen_skip_code": crow["pre_screen_skip_code"] or "",
               "document_integrity_flag": ";".join(crow["document_integrity_flags"]),
               "chunk_integrity_score": crow["chunk_integrity_score"], "corrupted_region_lines": len(crow["corrupted_regions"]),
               "v1_count": len(v1_by_chunk.get(key, [])),
               "v1_texts": " || ".join(i["embedding_text"] for i in v1_by_chunk.get(key, []))}
        for arm in arms:
            rejected = [r for r in read_jsonl(arm / "rejected_candidates.jsonl") if (r["document_id"], r["chunk_index"]) == key]
            recs = arm_rows[arm.name].get(key, [])
            label = arm.name[4:]
            row[f"v2_{label}_retained_count"] = len(recs)
            row[f"v2_{label}_rejected_count"] = len(rejected)
            row[f"v2_{label}_rejection_codes"] = ";".join(sorted({r["rejection_code"] for r in rejected}))
            row[f"v2_{label}_texts"] = " || ".join(r["verbatim_expression"] for r in recs)
        row["missed_expression"] = ""
        row["notes"] = ""
        comparison_rows.append(row)
    with open(review_dir / "pilot_chunk_comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
        w.writeheader()
        w.writerows(comparison_rows)

    arm_counts = Counter(r["arm"] if not args.blind else k["arm"] for r, k in zip(review_rows, key_rows or review_rows))
    readme = README_TEMPLATE.format(
        date_tag=args.date_tag, n_rows=len(review_rows), n_chunks=len(chunk_rows),
        arm_counts=", ".join(f"{k}: {v}" for k, v in sorted(arm_counts.items())),
        blind="yes -- arm column blanked, mapping in review_key.csv" if args.blind else "no",
        v1_sample=args.v1_sample_per_chunk,
        v2_sampling=("none (every retained v2 row is included)" if v2_cap_per_chunk is None
                     else f"seeded sample of {v2_cap_per_chunk} per chunk: " + "; ".join(sampling_note)),
        issue_vocab=" | ".join(EXTRACTION_ISSUE_VOCAB),
        validations=json.dumps(validations, indent=2),
    )
    n_rejected = write_rejected_csv(pilot_dir, arms, review_dir / "pilot_rejected_candidates.csv")
    (review_dir / "README.md").write_text(readme, encoding="utf-8")
    write_json(review_dir / "validation.json", {"validations": validations, "review_rows": len(review_rows),
                                                "arm_row_counts": dict(arm_counts), "blind": args.blind,
                                                "generated_at": datetime.now(timezone.utc).isoformat()})
    print(f"Review packet: {len(review_rows)} rows ({dict(arm_counts)}) -> {review_dir}")
    print(f"Rejected-candidate sheet: {n_rejected} rows -> {review_dir / 'pilot_rejected_candidates.csv'}")


README_TEMPLATE = """# Extraction v2 literature pilot -- manual review packet ({date_tag})

**What this is.** A side-by-side manual review of the frozen v1 extraction and the conservative v2 extraction on the same {n_chunks} literature chunks. Nothing here has been judged automatically: every manual-review column is blank and is yours to fill.

**Files**
- `pilot_review.csv` -- {n_rows} rows ({arm_counts}). Ordered random stratum first, then forced-regression chunks; within a stratum by document, chunk, arm, rank. Blind: {blind}.
- `pilot_chunk_comparison.csv` -- one row per chunk: v1 count and texts, v2 retained/rejected counts, rejection codes, and two blank columns (`missed_expression`, `notes`) for anything clearly useful that v2 omitted.
- `pilot_rejected_candidates.csv` -- every candidate the deterministic screen rejected, with its rejection code, the model's fields, the chunk text, and blank `rejection_correct` / `should_have_been_retained` / `reviewer_notes` columns.
- `validation.json` -- the independent hard-zero re-check of every arm (span resolution, verbatim = embedding, no interviewer rows, no high-confidence corruption, candidate/chunk accounting).

**Row budget.** v1 arm: seeded sample of at most {v1_sample} items per chunk (the full v1 list of every chunk is in the comparison file). v2 arm: {v2_sampling}.

**How to read a row.** `verbatim_expression` is the exact source span (for v1 rows, the v1 `source_quote`). `embedding_text` is what was / would be embedded; for v2 it is identical by construction (`embedding_equals_verbatim`). `context_window` is the full chunk the span comes from. `screen_flags` are non-fatal observations from the deterministic screen (e.g. `long`, `spaced_accent`, `isolated_capital`, `possible_heading_or_acronym`, `short_association`) -- they are hints for you, not judgements.

**Manual columns** (leave blank if not applicable):
- `faithful`, `self_contained`, `cult_relevant`, `textually_intelligible`, `atomic`: yes / no.
- `attribution_correct`, `claim_mode_correct`, `epistemic_status_correct`: yes / no / unclear; `recommended_epistemic_status` if you would change it.
- `better_span_exists_in_chunk`: yes / no -- is there a clearly better expression in `context_window` than the one selected?
- `extraction_issue`: {issue_vocab} (several allowed, separated by `;`).
- `reviewer_notes`: free text.

**What the pilot is for.** A decision gate: precision (are retained expressions usable?), yield (how many remain per chunk?), missed valuable material, and whether the known v1 failure cases in the forced stratum disappeared. No accuracy claim is made from it.

**Validation output**
```
{validations}
```
"""


# ---------------------------------------------------------------------------
# Rejected-candidate CSV (review aid; no judgement columns pre-filled)
# ---------------------------------------------------------------------------

REJECTED_CSV_COLUMNS = [
    "rejected_id", "arm", "selection_stratum", "document_id", "chunk_index", "candidate_rank",
    "rejection_code", "rule_detail", "model_verbatim_expression", "resolved_text", "span_start", "span_end",
    "expression_kind", "attribution", "claim_mode", "epistemic_status",
    "self_contained", "cult_relevant", "textually_intelligible", "single_coherent_expression",
    "relevance_note", "entity_anchors", "context_window",
    "rejection_correct", "should_have_been_retained", "reviewer_notes",
]


def write_rejected_csv(pilot_dir: Path, arms: list[Path], out_path: Path) -> int:
    """One row per rejected candidate across the given arms, joined to its
    chunk text. The three trailing columns are blank for the reviewer."""
    chunk_rows = {(r["document_id"], r["chunk_index"]): r for r in read_jsonl(pilot_dir / "pilot_chunks.jsonl")}
    rows = []
    for arm in arms:
        for rec in read_jsonl(arm / "rejected_candidates.jsonl"):
            raw = rec.get("raw_candidate") or {}
            raw = raw if isinstance(raw, dict) else {"verbatim_expression": str(raw)}
            chunk = chunk_rows.get((rec["document_id"], rec["chunk_index"]), {})
            anchors = raw.get("entity_anchors")
            rows.append({
                "rejected_id": None, "arm": f"v2_{arm.name[4:]}",
                "selection_stratum": rec.get("selection_stratum", chunk.get("selection_stratum", "")),
                "document_id": rec["document_id"], "chunk_index": rec["chunk_index"],
                "candidate_rank": rec["candidate_rank"], "rejection_code": rec["rejection_code"],
                "rule_detail": rec.get("rule_detail", ""),
                "model_verbatim_expression": raw.get("verbatim_expression", ""),
                "resolved_text": rec.get("resolved_text") or "",
                "span_start": rec.get("span_start") if rec.get("span_start") is not None else "",
                "span_end": rec.get("span_end") if rec.get("span_end") is not None else "",
                **{k: raw.get(k, "") for k in ("expression_kind", "attribution", "claim_mode", "epistemic_status",
                                                "self_contained", "cult_relevant", "textually_intelligible",
                                                "single_coherent_expression", "relevance_note")},
                "entity_anchors": "; ".join(anchors) if isinstance(anchors, list) else (anchors or ""),
                "context_window": chunk.get("nfc_text", ""),
                "rejection_correct": "", "should_have_been_retained": "", "reviewer_notes": "",
            })
    rows.sort(key=lambda r: (0 if r["selection_stratum"] == "random" else 1, r["document_id"], r["chunk_index"], r["candidate_rank"]))
    for i, row in enumerate(rows, start=1):
        row["rejected_id"] = i
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=REJECTED_CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def stage_rejected_csv(args, pilot_dir: Path) -> None:
    arms = sorted(p for p in pilot_dir.glob("arm_*") if (p / "rejected_candidates.jsonl").exists())
    if not arms:
        raise SystemExit(f"No arm_*/rejected_candidates.jsonl under {pilot_dir}")
    review_dir = pilot_dir / "review"
    review_dir.mkdir(exist_ok=True)
    out = review_dir / "pilot_rejected_candidates.csv"
    if out.exists():
        raise SystemExit(f"Refusing to overwrite {out}")
    n = write_rejected_csv(pilot_dir, arms, out)
    print(f"Rejected-candidate sheet: {n} rows -> {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["dry-run", "rebuild-chunks", "run", "build-review", "rejected-csv"], required=True)
    parser.add_argument("--date-tag", default=datetime.now(timezone.utc).strftime("%Y%m%d"))
    parser.add_argument("--pilot-root", type=Path, default=DEFAULT_PILOT_ROOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--audit-csv", type=Path, default=None, help="Stage-1 integrity audit CSV (default: latest under processed/audits/)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--arm-tag", default=None, help="Override the arm directory suffix (e.g. a throwaway reproducibility rerun)")
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--v1-sample-per-chunk", type=int, default=2)
    parser.add_argument("--v2-sample-per-chunk", type=int, default=3)
    parser.add_argument("--review-row-cap", type=int, default=REVIEW_ROW_CAP)
    parser.add_argument("--blind", action="store_true")
    args = parser.parse_args()

    pilot_dir = args.pilot_root / f"pilot_{args.date_tag}"
    if args.stage == "dry-run":
        stage_dry_run(args, pilot_dir)
    elif args.stage == "rebuild-chunks":
        stage_rebuild_chunks(args, pilot_dir)
    elif args.stage == "run":
        stage_run(args, pilot_dir)
    elif args.stage == "rejected-csv":
        stage_rejected_csv(args, pilot_dir)
    else:
        stage_build_review(args, pilot_dir)


if __name__ == "__main__":
    main()
