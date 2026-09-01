"""Deterministic document IDs derived from PDF filenames.

Real filename collisions exist in the source corpus (e.g. two different
Zotero storage folders both containing a PDF literally named "Barker - 2020 -
Even new religious movements have legacies.pdf"), so normalization alone
isn't enough. On collision, the source folder name (the Zotero storage key)
is appended — deterministic across reruns, unlike an order-dependent counter.
"""
import re
from pathlib import Path

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
MAX_SLUG_LEN = 80


def normalize_filename(filename: str) -> str:
    """PDF filename (with or without extension) -> lowercase hyphen slug."""
    stem = Path(filename).stem
    slug = _NON_ALNUM_RE.sub("-", stem.lower()).strip("-")
    return slug[:MAX_SLUG_LEN].strip("-") or "untitled"


def make_document_id(pdf_path: Path, source_root: Path, used_ids: set[str]) -> str:
    """Deterministic, collision-safe document_id for a PDF.

    Derived from the PDF's own filename (the human-legible identifier, not
    the numeric Zotero storage-key folder it happens to live in). On
    collision, the parent folder name relative to source_root is appended.
    """
    base = normalize_filename(pdf_path.name)
    if base not in used_ids:
        used_ids.add(base)
        return base

    parent_key = pdf_path.parent.name
    candidate = f"{base}-{parent_key}"
    if candidate not in used_ids:
        used_ids.add(candidate)
        return candidate

    # Still colliding (e.g. two PDFs with the same name in the same folder,
    # which shouldn't happen on a real filesystem, but stay deterministic
    # rather than crash): fall back to the full relative path, slugified.
    rel = pdf_path.relative_to(source_root)
    candidate = normalize_filename(str(rel))
    used_ids.add(candidate)
    return candidate
