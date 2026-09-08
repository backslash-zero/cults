"""Pack / verify a mirror of the Stage-1 text tree between machines.

Stage 1 (clean_text / prepare_interviews / prepare_text_files) runs on the
Mac and writes processed/<corpus>/documents/<id>/{pages.jsonl, extracted.md,
extracted.txt, metadata.json} plus corpus_manifest.csv. All of that is
gitignored (verbatim copyrighted text), so the Ollama machine only has
whatever was copied by hand -- and it turned out to be incomplete (the
Oxford handbook's pages.jsonl was missing there). This script makes the
copy explicit and checkable:

  --pack     (source machine) zip every file under processed/<corpus>/documents/
             and each corpus_manifest.csv, together with MIRROR_MANIFEST.json
             (relative path -> sha256 + size, per-corpus document counts,
             git commit, timestamp).
  --verify   (target machine, after unzipping into processed/) compare the
             local tree against MIRROR_MANIFEST.json: reports missing files,
             sha256 mismatches, and local documents that the source does not
             have. Never deletes or overwrites anything.

Usage (from thesis/corpus/):
    python -m thesis_corpus.mirror_stage1 --pack --out processed/stage1_mirror_<date>.zip
    # copy the zip, unzip -o into thesis/corpus/processed/ on the other machine, then:
    python -m thesis_corpus.mirror_stage1 --verify --manifest processed/MIRROR_MANIFEST.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent.parent
PROCESSED_ROOT = CORPUS_DIR / "processed"
DEFAULT_CORPORA = ("literature", "miviludes", "interviews")
MANIFEST_NAME = "MIRROR_MANIFEST.json"


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


def collect_files(processed_root: Path, corpora) -> list[Path]:
    files: list[Path] = []
    for corpus in corpora:
        corpus_dir = processed_root / corpus
        manifest_csv = corpus_dir / "corpus_manifest.csv"
        if manifest_csv.exists():
            files.append(manifest_csv)
        documents_dir = corpus_dir / "documents"
        if documents_dir.exists():
            files.extend(sorted(p for p in documents_dir.rglob("*") if p.is_file() and p.name != ".DS_Store"))
    return files


def pack(processed_root: Path, corpora, out_zip: Path) -> None:
    if out_zip.exists():
        raise SystemExit(f"Refusing to overwrite {out_zip}")
    files = collect_files(processed_root, corpora)
    entries = {}
    docs_per_corpus: dict[str, set[str]] = {c: set() for c in corpora}
    for path in files:
        rel = path.relative_to(processed_root).as_posix()
        entries[rel] = {"sha256": sha256_file(path), "size": path.stat().st_size}
        parts = rel.split("/")
        if len(parts) >= 3 and parts[1] == "documents":
            docs_per_corpus[parts[0]].add(parts[2])
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit_hash(),
        "source_processed_root": str(processed_root),
        "corpora": list(corpora),
        "documents_per_corpus": {c: sorted(d) for c, d in docs_per_corpus.items()},
        "file_count": len(entries),
        "total_bytes": sum(e["size"] for e in entries.values()),
        "files": entries,
    }
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in files:
            zf.write(path, path.relative_to(processed_root).as_posix())
        zf.writestr(MANIFEST_NAME, json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Packed {len(files)} files ({manifest['total_bytes'] / 1e6:.1f} MB) from "
          f"{', '.join(f'{c}: {len(d)} docs' for c, d in docs_per_corpus.items())} -> {out_zip} "
          f"({out_zip.stat().st_size / 1e6:.1f} MB)")
    print(f"zip sha256: {sha256_file(out_zip)}")


def verify(processed_root: Path, manifest_path: Path) -> None:
    if manifest_path.suffix == ".zip":
        with zipfile.ZipFile(manifest_path) as zf:
            manifest = json.loads(zf.read(MANIFEST_NAME))
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing, mismatched, ok = [], [], 0
    for rel, info in manifest["files"].items():
        path = processed_root / rel
        if not path.exists():
            missing.append(rel)
        elif sha256_file(path) != info["sha256"]:
            mismatched.append(rel)
        else:
            ok += 1
    extras: dict[str, list[str]] = {}
    for corpus, docs in manifest["documents_per_corpus"].items():
        documents_dir = processed_root / corpus / "documents"
        if documents_dir.exists():
            local = {d.name for d in documents_dir.iterdir() if d.is_dir()}
            extra = sorted(local - set(docs))
            if extra:
                extras[corpus] = extra
    print(f"Verified against manifest from {manifest['generated_at']} (git {manifest['git_commit']}):")
    print(f"  identical: {ok}/{manifest['file_count']}")
    print(f"  missing:   {len(missing)}" + (f"  e.g. {missing[:5]}" if missing else ""))
    print(f"  mismatch:  {len(mismatched)}" + (f"  e.g. {mismatched[:5]}" if mismatched else ""))
    for corpus, extra in extras.items():
        print(f"  local-only documents in {corpus} (not on the source; left untouched): {extra}")
    if missing or mismatched:
        raise SystemExit("Mirror is NOT complete.")
    print("Mirror complete: every Stage-1 file matches the source byte for byte.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pack", action="store_true")
    mode.add_argument("--verify", action="store_true")
    parser.add_argument("--processed-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--corpora", nargs="+", default=list(DEFAULT_CORPORA))
    parser.add_argument("--out", type=Path, default=None, help="--pack: zip path to write")
    parser.add_argument("--manifest", type=Path, default=None, help="--verify: MIRROR_MANIFEST.json or the zip itself")
    args = parser.parse_args()
    if args.pack:
        out = args.out or (PROCESSED_ROOT / f"stage1_mirror_{datetime.now(timezone.utc).strftime('%Y%m%d')}.zip")
        pack(args.processed_root, args.corpora, out)
    else:
        manifest = args.manifest or (args.processed_root / MANIFEST_NAME)
        if not manifest.exists():
            raise SystemExit(f"Manifest not found: {manifest}")
        verify(args.processed_root, manifest)


if __name__ == "__main__":
    main()
