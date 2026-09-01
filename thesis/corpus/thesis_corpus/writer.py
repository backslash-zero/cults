"""Per-document output files and the corpus-level manifest."""
import csv
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_WHITESPACE_RUN_RE = re.compile(r"[ \t]+")
_BLANK_LINE_RUN_RE = re.compile(r"\n{3,}")

MANIFEST_FIELDS = [
    "document_id", "source_relative_path", "source_filename", "page_count",
    "extraction_method", "ocr_used", "processing_status", "error_message",
    "output_directory",
]


@dataclass
class PageRecord:
    document_id: str
    page_number: int
    text: str
    extraction_method: str
    character_count: int
    warnings: list[str] = field(default_factory=list)


@dataclass
class DocumentResult:
    document_id: str
    source_relative_path: str
    source_filename: str
    page_count: int | None
    extraction_method: str | None  # "native" | "ocr" | None
    ocr_used: bool
    processing_status: str  # "processed" | "processed_with_ocr" | "unreadable" | "failed"
    error_message: str
    pages: list[PageRecord] = field(default_factory=list)
    markdown: str = ""


def collapse_whitespace(text: str) -> str:
    """Collapse runs of horizontal whitespace and excessive blank lines,
    keeping paragraph breaks (a single blank line) and page markers intact."""
    lines = [_WHITESPACE_RUN_RE.sub(" ", line).strip() for line in text.splitlines()]
    collapsed = "\n".join(lines)
    return _BLANK_LINE_RUN_RE.sub("\n\n", collapsed).strip() + "\n"


def page_marker(page_number: int) -> str:
    return f"<!-- PAGE: {page_number} -->"


def build_plain_text(pages: list[PageRecord]) -> str:
    parts = []
    for p in pages:
        parts.append(page_marker(p.page_number))
        parts.append(p.text)
    return collapse_whitespace("\n".join(parts))


def write_document_outputs(result: DocumentResult, doc_dir: Path) -> None:
    doc_dir.mkdir(parents=True, exist_ok=True)

    (doc_dir / "extracted.md").write_text(result.markdown, encoding="utf-8")
    (doc_dir / "extracted.txt").write_text(
        build_plain_text(result.pages) if result.pages else "", encoding="utf-8"
    )

    with open(doc_dir / "pages.jsonl", "w", encoding="utf-8") as f:
        for p in result.pages:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")

    metadata = {
        "document_id": result.document_id,
        "source_relative_path": result.source_relative_path,
        "source_filename": result.source_filename,
        "page_count": result.page_count,
        "extraction_method": result.extraction_method,
        "ocr_used": result.ocr_used,
        "processing_status": result.processing_status,
        "error_message": result.error_message,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }
    (doc_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


class ManifestWriter:
    """Appends one row per document as processing proceeds, so a crash
    partway through the batch doesn't lose already-processed rows."""

    def __init__(self, manifest_path: Path):
        self.manifest_path = manifest_path
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(manifest_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=MANIFEST_FIELDS)
        self._writer.writeheader()

    def add(self, result: DocumentResult, output_directory: str) -> None:
        self._writer.writerow({
            "document_id": result.document_id,
            "source_relative_path": result.source_relative_path,
            "source_filename": result.source_filename,
            "page_count": result.page_count if result.page_count is not None else "",
            "extraction_method": result.extraction_method or "",
            "ocr_used": result.ocr_used,
            "processing_status": result.processing_status,
            "error_message": result.error_message,
            "output_directory": output_directory,
        })
        self._file.flush()

    def close(self) -> None:
        self._file.close()
