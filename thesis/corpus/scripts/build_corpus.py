#!/usr/bin/env python3
"""Build the unified corpus databases and LaTeX appendices.

Four corpus types share this script: interviews, literature, dictionary/
thesaurus entries, and custom terms. Each type reads a hand-authored YAML
source file, computes any derived fields, and writes a JSON database, a flat
CSV export, and generated LaTeX for the thesis appendix.

Requires PyYAML. Run with: python3 thesis/corpus/scripts/build_corpus.py
(from the repo root) — corpus/ lives inside thesis/, alongside the LaTeX
source it generates appendices into.
"""
import csv
import json
import re
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
CORPUS_DIR = SCRIPT_DIR.parent
THESIS_DIR = CORPUS_DIR.parent
REPO_ROOT = THESIS_DIR.parent
APPENDIX_ROOT = THESIS_DIR / "04_Appendix"

# --- Interviews (migrated from thesis/Interviews/Corpus/) -------------------
INTERVIEWS_DIR = CORPUS_DIR / "interviews"
INTERVIEWS_SOURCE_YAML = INTERVIEWS_DIR / "metadata" / "interviews_source.yaml"
INTERVIEWS_CLEANED_DIR = INTERVIEWS_DIR / "cleaned"
INTERVIEWS_DATABASE_JSON = INTERVIEWS_DIR / "metadata" / "database.json"
INTERVIEWS_DATABASE_CSV = INTERVIEWS_DIR / "metadata" / "database.csv"
INTERVIEWS_APPENDIX_DIR = APPENDIX_ROOT / "Interview_Corpus"
INTERVIEWS_APPENDIX_TEX = INTERVIEWS_APPENDIX_DIR / "appendix.tex"
INTERVIEWS_TABLE_TEX = INTERVIEWS_APPENDIX_DIR / "database_table.tex"

# --- Literature / dictionaries / custom terms (new, shared metadata dir) ----
METADATA_DIR = CORPUS_DIR / "metadata"
LITERATURE_SOURCE_YAML = METADATA_DIR / "literature_source.yaml"
LITERATURE_JSON = METADATA_DIR / "literature.json"
LITERATURE_CSV = METADATA_DIR / "literature.csv"
DICTIONARIES_SOURCE_YAML = METADATA_DIR / "dictionaries_source.yaml"
DICTIONARIES_JSON = METADATA_DIR / "dictionaries.json"
DICTIONARIES_CSV = METADATA_DIR / "dictionaries.csv"
CUSTOM_TERMS_SOURCE_YAML = METADATA_DIR / "custom_terms_source.yaml"
CUSTOM_TERMS_JSON = METADATA_DIR / "custom_terms.json"
CUSTOM_TERMS_CSV = METADATA_DIR / "custom_terms.csv"

LITERATURE_APPENDIX_DIR = APPENDIX_ROOT / "Literature_Corpus"
LITERATURE_LIST_TEX = LITERATURE_APPENDIX_DIR / "reference_list.tex"
DICTIONARIES_TABLE_TEX = LITERATURE_APPENDIX_DIR / "dictionaries_table.tex"
CUSTOM_TERMS_TABLE_TEX = LITERATURE_APPENDIX_DIR / "custom_terms_table.tex"
LITERATURE_RAW_DIR = CORPUS_DIR / "literature" / "raw"
KEYWORD_QUERIES_TXT = CORPUS_DIR / "literature" / "keyword-queries.txt"
KEYWORD_QUERIES_TABLE_TEX = LITERATURE_APPENDIX_DIR / "keyword_queries_table.tex"

LATEX_SPECIAL = {
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
    "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    "\\": r"\textbackslash{}",
}


def escape_latex(s):
    if s is None:
        return ""
    return "".join(LATEX_SPECIAL.get(ch, ch) for ch in str(s))


def load_yaml_list(path, key):
    """Load a YAML source file's list under `key`; tolerate an empty/missing
    file (no content collected yet) by returning an empty list."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not data:
        return []
    return data.get(key) or []


def write_json(path, key, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({key: records}, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# =============================================================================
# Interviews
# =============================================================================

INTERVIEWER_RE = re.compile(r"^Interviewer\s*:")
INTERVIEWEE_RE = re.compile(r"^Interviewee\s*:")

# Trailing editorial/methodological blocks stripped from the *published LaTeX*
# transcript body only (never from cleaned/<id>/transcript.txt itself).
TRAILING_NOTE_MARKERS = [
    "TRANSCRIPTION / EDITING NOTES",
    "TRANSCRIPTION/EDITING NOTES",
    "Interview Notes:",
]
SEPARATOR_LINE_RE = re.compile(r"^-{2,}$")
LABEL_RE = re.compile(r"^(Interviewer|Interviewee)(\s*):", re.MULTILINE)


def dialogue_only(text):
    """Return just the Interviewer:/Interviewee: turns from a transcript.

    Drops everything before the first turn (source's own ad hoc header/
    demographics line — replaced by format_header() instead) and any trailing
    TRANSCRIPTION/EDITING NOTES or Instagram "Interview Notes:" footer, plus a
    dangling separator line left at the end either way.
    """
    lines = text.splitlines()
    start_idx = 0
    for i, line in enumerate(lines):
        if INTERVIEWER_RE.match(line.strip()):
            start_idx = i
            break
    joined = "\n".join(lines[start_idx:])

    cut_idx = len(joined)
    for marker in TRAILING_NOTE_MARKERS:
        idx = joined.find(marker)
        if idx != -1:
            cut_idx = min(cut_idx, idx)
    joined = joined[:cut_idx]

    body_lines = joined.splitlines()
    while body_lines and (
        body_lines[-1].strip() == "" or SEPARATOR_LINE_RE.match(body_lines[-1].strip())
    ):
        body_lines.pop()
    return "\n".join(body_lines)


def format_header(record):
    """Build a uniform header, generated from the metadata fields rather than
    copied from each source file's own ad hoc formatting, so batch 1-3 and
    Instagram interviews read identically."""
    interviewee = record.get("interviewee", {})
    age_val = interviewee.get("age", "unknown")
    age_str = f"{age_val} years old" if isinstance(age_val, int) else f"age {age_val}"

    dt = record["date_time"]
    if record.get("date_time_precision"):
        dt = f"{dt} ({record['date_time_precision']})"

    header_line = f"INTERVIEW — {dt} ({record['method']})"
    demo_line = (
        f"Interviewee: {age_str}, {interviewee.get('gender', 'unknown')}, "
        f"main language {interviewee.get('main_language', 'unknown')}, "
        f"speaking in {interviewee.get('language_spoken', 'unknown')}, "
        f"lives in {record.get('location', 'unknown')}, "
        f"nationality {interviewee.get('nationality', 'unknown')}."
    )
    return f"{header_line}\n\n{demo_line}"


def bold_labels(escaped_text):
    """Bold Interviewer:/Interviewee: (and the French 'Interviewer :' spacing
    variant) at the start of each line. Must run on already-escaped text."""
    return LABEL_RE.sub(lambda m: r"\textbf{%s%s:}" % (m.group(1), m.group(2)), escaped_text)


def count_turns(text, is_instagram):
    """Count Interviewer:/Interviewee: turns.

    Batch 1-3 files open with a demographics line labeled "Interviewee:"
    before the dialogue starts, which is not itself an answer turn — it is
    subtracted out. Instagram files use a different "Age:, Gender:, ..."
    header with no such label, so no offset is needed there.
    """
    n_interviewer = 0
    n_interviewee = 0
    for line in text.splitlines():
        stripped = line.strip()
        if INTERVIEWER_RE.match(stripped):
            n_interviewer += 1
        elif INTERVIEWEE_RE.match(stripped):
            n_interviewee += 1
    header_offset = 0 if is_instagram else 1
    return n_interviewer, max(n_interviewee - header_offset, 0)


def word_count(text):
    return len(text.split())


def build_interview_records():
    records = []
    for entry in load_yaml_list(INTERVIEWS_SOURCE_YAML, "interviews"):
        iid = entry["id"]
        idir = INTERVIEWS_CLEANED_DIR / iid
        transcript_path = idir / "transcript.txt"
        translation_path = idir / "translation_en.txt"
        corrections_path = idir / "corrections.log"

        text = transcript_path.read_text(encoding="utf-8")
        is_instagram = entry["batch"] == "Instagram Batch"
        n_q, n_a = count_turns(text, is_instagram)

        translation_text = None
        if entry.get("translated") and translation_path.exists():
            translation_text = translation_path.read_text(encoding="utf-8")

        record = dict(entry)
        record["text"] = text
        record["translation_text"] = translation_text
        record["n_questions"] = n_q
        record["n_answers"] = n_a
        record["total_word_count"] = word_count(text)
        record["corrections_log"] = str(corrections_path.relative_to(INTERVIEWS_DIR))
        records.append(record)
    return records


INTERVIEW_CSV_FIELDS = [
    "id", "batch", "source_file", "date_time", "date_time_precision", "method",
    "language", "translated", "translation_language", "interviewer",
    "participant_id", "location", "age", "gender", "nationality",
    "main_language", "language_spoken", "n_questions", "n_answers",
    "total_word_count", "notes",
]


def write_interviews_csv(records):
    rows = []
    for r in records:
        interviewee = r.get("interviewee", {})
        rows.append({
            "id": r["id"],
            "batch": r["batch"],
            "source_file": r["source_file"],
            "date_time": r["date_time"],
            "date_time_precision": r.get("date_time_precision", ""),
            "method": r["method"],
            "language": r["language"],
            "translated": r["translated"],
            "translation_language": r.get("translation_language", ""),
            "interviewer": r["interviewer"],
            "participant_id": r["participant_id"],
            "location": r.get("location", ""),
            "age": interviewee.get("age", ""),
            "gender": interviewee.get("gender", ""),
            "nationality": interviewee.get("nationality", ""),
            "main_language": interviewee.get("main_language", ""),
            "language_spoken": interviewee.get("language_spoken", ""),
            "n_questions": r["n_questions"],
            "n_answers": r["n_answers"],
            "total_word_count": r["total_word_count"],
            "notes": r.get("notes", ""),
        })
    write_csv(INTERVIEWS_DATABASE_CSV, INTERVIEW_CSV_FIELDS, rows)


PAREN_SUFFIX_RE = re.compile(r"\s*\([^)]*\)\s*$")


def short_field(s):
    """Strip a trailing parenthetical caveat (e.g. '(see notes — ...)') for the
    compact table view; the full string is still in the metadata slip/JSON."""
    return PAREN_SUFFIX_RE.sub("", str(s)).strip()


def write_interviews_table_tex(records):
    INTERVIEWS_APPENDIX_DIR.mkdir(parents=True, exist_ok=True)
    header_row = (
        r"ID & Date/time & Method & Language & Age & Gender & Nationality & "
        r"Main language & Q & A & Words \\"
    )
    lines = [
        r"\begin{landscape}",
        r"\begin{longtable}{@{}p{2.2cm}p{2.0cm}p{1.6cm}p{1.4cm}p{0.8cm}p{1.3cm}p{2.4cm}p{1.8cm}p{0.6cm}p{0.6cm}p{1.0cm}@{}}",
        r"\caption{Interview corpus metadata}\label{tab:interview_corpus} \\",
        r"\toprule",
        header_row,
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        header_row,
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    for r in records:
        interviewee = r.get("interviewee", {})
        lines.append(" & ".join([
            r"\ref{int:%s}" % r["id"],
            escape_latex(r["date_time"]),
            escape_latex(r["method"]),
            escape_latex(r["language"]),
            escape_latex(short_field(interviewee.get("age", "unknown"))),
            escape_latex(short_field(interviewee.get("gender", "unknown"))),
            escape_latex(short_field(interviewee.get("nationality", "unknown"))),
            escape_latex(short_field(interviewee.get("main_language", "unknown"))),
            str(r["n_questions"]),
            str(r["n_answers"]),
            str(r["total_word_count"]),
        ]) + r" \\")
    lines.append(r"\end{longtable}")
    lines.append(r"\end{landscape}")
    INTERVIEWS_TABLE_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_interviews_appendix_tex(records):
    INTERVIEWS_APPENDIX_DIR.mkdir(parents=True, exist_ok=True)
    parts = []
    for r in records:
        iid = r["id"]
        interviewee = r.get("interviewee", {})
        parts.append(r"\subsection{Interview %s}\label{int:%s}" % (escape_latex(iid), iid))
        parts.append(r"\subsubsection*{Metadata}")
        parts.append(r"\begin{itemize}")
        parts.append(r"  \item \textbf{ID}: %s" % escape_latex(iid))
        dt = escape_latex(r["date_time"])
        if r.get("date_time_precision"):
            dt += f" ({escape_latex(r['date_time_precision'])})"
        parts.append(r"  \item \textbf{Date/time}: %s" % dt)
        parts.append(r"  \item \textbf{Method}: %s" % escape_latex(r["method"]))
        lang = escape_latex(r["language"])
        if r.get("translated"):
            lang += f", translated to {escape_latex(r.get('translation_language', ''))}"
        parts.append(r"  \item \textbf{Language}: %s" % lang)
        parts.append(r"  \item \textbf{Location}: %s" % escape_latex(r.get("location", "unknown")))
        parts.append(r"  \item \textbf{Interviewee}: age %s, %s, %s nationality, main language %s" % (
            escape_latex(interviewee.get("age", "unknown")),
            escape_latex(interviewee.get("gender", "unknown")),
            escape_latex(interviewee.get("nationality", "unknown")),
            escape_latex(interviewee.get("main_language", "unknown")),
        ))
        parts.append(r"  \item \textbf{Questions / Answers}: %d / %d" % (r["n_questions"], r["n_answers"]))
        parts.append(r"  \item \textbf{Total word count}: %d" % r["total_word_count"])
        parts.append(r"\end{itemize}")
        # Note: corrections.log path and free-text `notes` are deliberately not
        # rendered here (kept in database.json/csv and the log files themselves)
        # — the published appendix stays reading-copy clean.

        parts.append(r"\subsubsection*{Transcript}")
        is_french = r.get("language") == "French"
        transcript_body = format_header(r) + "\n\n" + dialogue_only(r["text"])
        if is_french:
            parts.append(r"\begin{otherlanguage}{french}")
        parts.append(bold_labels(escape_latex(transcript_body)))
        if is_french:
            parts.append(r"\end{otherlanguage}")

        if r.get("translation_text"):
            parts.append(r"\subsubsection*{English translation}")
            parts.append(r"\textit{Translation note: produced directly by the assistant compiling this corpus, not a certified translation.}")
            parts.append("")
            parts.append(bold_labels(escape_latex(dialogue_only(r["translation_text"]))))

        parts.append("")
    INTERVIEWS_APPENDIX_TEX.write_text("\n".join(parts) + "\n", encoding="utf-8")


def build_interviews():
    records = build_interview_records()
    write_json(INTERVIEWS_DATABASE_JSON, "interviews", records)
    write_interviews_csv(records)
    write_interviews_table_tex(records)
    write_interviews_appendix_tex(records)
    return records


# =============================================================================
# Literature
# =============================================================================

LITERATURE_CSV_FIELDS = [
    "id", "title", "authors", "year", "type", "source", "doi", "isbn",
    "language", "tags", "raw_file", "date_added", "abstract", "notes",
]


# --- Minimal hand-rolled BibTeX parser --------------------------------------
# No third-party dependency; handles the standard Zotero-export shape (brace-
# or quote-delimited field values, nested braces for case-protected titles).

def _parse_bib_fields(body):
    """body is the entry content after 'key,' -- returns {field: raw_value}."""
    fields = {}
    i, n = 0, len(body)
    while i < n:
        while i < n and body[i] in " \t\n\r,":
            i += 1
        if i >= n:
            break
        eq = body.find("=", i)
        if eq == -1:
            break
        name = body[i:eq].strip().lower()
        i = eq + 1
        while i < n and body[i] in " \t\n\r":
            i += 1
        if i < n and body[i] == "{":
            depth, i = 1, i + 1
            start = i
            while i < n and depth > 0:
                if body[i] == "{":
                    depth += 1
                elif body[i] == "}":
                    depth -= 1
                i += 1
            value = body[start:i - 1]
        elif i < n and body[i] == '"':
            i += 1
            start = i
            while i < n and body[i] != '"':
                i += 1
            value = body[start:i]
            i += 1
        else:
            start = i
            while i < n and body[i] not in ",\n":
                i += 1
            value = body[start:i]
        # strip nested protective braces (e.g. "{{Oxford}}" -> "Oxford") and
        # collapse whitespace from multi-line values
        fields[name] = re.sub(r"\s+", " ", value.replace("{", "").replace("}", "")).strip()
        i += 1
    return fields


def parse_bib_file(path):
    text = path.read_text(encoding="utf-8")
    entries = []
    i, n = 0, len(text)
    while True:
        at = text.find("@", i)
        if at == -1:
            break
        brace = text.find("{", at)
        if brace == -1:
            break
        entry_type = text[at + 1:brace].strip().lower()
        depth, j = 1, brace + 1
        while j < n and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        body = text[brace + 1:j - 1]
        i = j
        comma = body.find(",")
        if comma == -1:
            continue
        key = body[:comma].strip()
        fields = _parse_bib_fields(body[comma + 1:])
        entries.append({"type": entry_type, "key": key, "fields": fields})
    return entries


BIB_TYPE_MAP = {"book": "book", "incollection": "book_chapter", "article": "article",
                "inproceedings": "report", "report": "report", "phdthesis": "other"}


def bib_entry_to_record(entry):
    f = entry["fields"]
    authors_raw = f.get("author") or f.get("editor") or ""
    authors = [a.strip() for a in authors_raw.split(" and ") if a.strip()]
    is_editor = "author" not in f and "editor" in f
    return {
        "id": entry["key"],
        "title": f.get("title", entry["key"]),
        "authors": [f"{a} (ed.)" for a in authors] if is_editor and authors else authors,
        "year": f.get("date", f.get("year", ""))[:4] if f.get("date") or f.get("year") else "",
        "type": BIB_TYPE_MAP.get(entry["type"], "other"),
        "source": ", ".join(x for x in (f.get("publisher"), f.get("location")) if x),
        "doi": f.get("doi", ""),
        "isbn": f.get("isbn", ""),
        "language": f.get("langid", f.get("language", "")),
        "tags": [],
        "raw_file": "",
        "date_added": "",
        "abstract": f.get("abstract", ""),
        "notes": "",
    }


def load_bib_records():
    records = []
    if not LITERATURE_RAW_DIR.exists():
        return records
    for path in sorted(LITERATURE_RAW_DIR.glob("*.bib")):
        for entry in parse_bib_file(path):
            records.append(bib_entry_to_record(entry))
    return records


def build_literature_records():
    records = [dict(entry) for entry in load_yaml_list(LITERATURE_SOURCE_YAML, "literature")]
    seen_ids = {r["id"] for r in records}
    for rec in load_bib_records():
        if rec["id"] in seen_ids:
            continue  # a hand-authored YAML entry with the same id takes precedence
        records.append(rec)
        seen_ids.add(rec["id"])
    return records


def write_literature_csv(records):
    rows = []
    for r in records:
        rows.append({
            "id": r["id"],
            "title": r.get("title", ""),
            "authors": "; ".join(r.get("authors", []) or []),
            "year": r.get("year", ""),
            "type": r.get("type", ""),
            "source": r.get("source", ""),
            "doi": r.get("doi", ""),
            "isbn": r.get("isbn", ""),
            "language": r.get("language", ""),
            "tags": "; ".join(r.get("tags", []) or []),
            "raw_file": r.get("raw_file", ""),
            "date_added": r.get("date_added", ""),
            "abstract": r.get("abstract", ""),
            "notes": r.get("notes", ""),
        })
    write_csv(LITERATURE_CSV, LITERATURE_CSV_FIELDS, rows)


def write_literature_list_tex(records):
    LITERATURE_APPENDIX_DIR.mkdir(parents=True, exist_ok=True)
    if not records:
        LITERATURE_LIST_TEX.write_text(
            r"\textit{No literature entries collected yet.}" + "\n", encoding="utf-8")
        return
    parts = []
    for r in records:
        rid = r["id"]
        authors = ", ".join(r.get("authors", []) or []) or "Unknown author"
        year = r.get("year") or "n.d."
        title = r.get("title", "untitled")
        parts.append(r"\subsection{%s}\label{lit:%s}" % (escape_latex(title), escape_latex(rid)))
        parts.append(r"\subsubsection*{Metadata}")
        parts.append(r"\begin{itemize}")
        parts.append(r"  \item \textbf{Citation key}: \texttt{%s}" % escape_latex(rid))
        parts.append(r"  \item \textbf{Authors/Editors}: %s" % escape_latex(authors))
        parts.append(r"  \item \textbf{Year}: %s" % escape_latex(year))
        if r.get("type"):
            parts.append(r"  \item \textbf{Type}: %s" % escape_latex(r["type"]))
        if r.get("source"):
            parts.append(r"  \item \textbf{Publisher}: %s" % escape_latex(r["source"]))
        if r.get("isbn"):
            parts.append(r"  \item \textbf{ISBN}: %s" % escape_latex(r["isbn"]))
        if r.get("doi"):
            parts.append(r"  \item \textbf{DOI}: %s" % escape_latex(r["doi"]))
        if r.get("tags"):
            parts.append(r"  \item \textbf{Tags}: %s" % escape_latex(", ".join(r["tags"])))
        parts.append(r"\end{itemize}")
        if r.get("abstract"):
            parts.append(r"\subsubsection*{Abstract}")
            parts.append(escape_latex(r["abstract"]))
        parts.append("")
    LITERATURE_LIST_TEX.write_text("\n".join(parts) + "\n", encoding="utf-8")


def build_literature():
    records = build_literature_records()
    write_json(LITERATURE_JSON, "literature", records)
    write_literature_csv(records)
    write_literature_list_tex(records)
    return records


# =============================================================================
# Keyword queries (corpus construction protocol)
# =============================================================================

def load_keyword_queries():
    """Parse corpus/literature/keyword-queries.txt: one query per non-blank,
    non-comment line. `#` starts a comment (whole-line only, since queries
    never contain '#')."""
    if not KEYWORD_QUERIES_TXT.exists():
        return []
    queries = []
    for line in KEYWORD_QUERIES_TXT.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        queries.append(stripped)
    return queries


def write_keyword_queries_table_tex(queries):
    LITERATURE_APPENDIX_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{longtable}{@{}p{10cm}@{}}",
        r"\caption{Keyword queries used for corpus retrieval}\label{tab:keyword_queries_corpus} \\",
        r"\toprule",
        r"Query \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Query \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    if not queries:
        lines.append(r"\textit{No queries collected yet.} \\")
    for q in queries:
        lines.append(r"\texttt{%s} \\" % escape_latex(q))
    lines.append(r"\end{longtable}")
    KEYWORD_QUERIES_TABLE_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_keyword_queries():
    queries = load_keyword_queries()
    write_keyword_queries_table_tex(queries)
    return queries


# =============================================================================
# Dictionary / thesaurus entries
# =============================================================================

DICTIONARY_CSV_FIELDS = ["id", "source", "term", "language", "definition", "citation", "date_added"]


def build_dictionary_records():
    return [dict(entry) for entry in load_yaml_list(DICTIONARIES_SOURCE_YAML, "dictionaries")]


def write_dictionaries_csv(records):
    rows = [{
        "id": r["id"],
        "source": r.get("source", ""),
        "term": r.get("term", ""),
        "language": r.get("language", ""),
        "definition": r.get("definition", ""),
        "citation": r.get("citation", ""),
        "date_added": r.get("date_added", ""),
    } for r in records]
    write_csv(DICTIONARIES_CSV, DICTIONARY_CSV_FIELDS, rows)


def write_glossary_table(path, records, caption, label, ref_prefix, columns):
    """Generic two/three-column longtable glossary writer.

    `columns` is a list of (header, field_getter) where field_getter(record)
    returns the already-plain-text (not yet escaped) cell value.
    """
    LITERATURE_APPENDIX_DIR.mkdir(parents=True, exist_ok=True)
    colspec = "@{}" + "p{4cm}" + "p{8cm}" * (len(columns) - 1) + "@{}"
    header_row = " & ".join(h for h, _ in columns) + r" \\"
    lines = [
        r"\begin{longtable}{%s}" % colspec,
        r"\caption{%s}\label{%s} \\" % (caption, label),
        r"\toprule",
        header_row,
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        header_row,
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    if not records:
        lines.append(r"\multicolumn{%d}{@{}l}{\textit{No entries collected yet.}} \\" % len(columns))
    for r in records:
        cells = []
        for i, (_, getter) in enumerate(columns):
            val = escape_latex(getter(r))
            if i == 0:
                val = r"\ref{%s:%s}~(\texttt{%s})" % (ref_prefix, r["id"], val)
            cells.append(val)
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\end{longtable}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_dictionaries():
    records = build_dictionary_records()
    write_json(DICTIONARIES_JSON, "dictionaries", records)
    write_dictionaries_csv(records)
    write_glossary_table(
        DICTIONARIES_TABLE_TEX, records,
        "Dictionary and thesaurus entries", "tab:dictionary_corpus", "dict",
        [("Term", lambda r: r.get("term", "")),
         ("Definition (source)", lambda r: f"{r.get('definition', '')} — {r.get('source', '')}")],
    )
    return records


# =============================================================================
# Custom terms
# =============================================================================

CUSTOM_TERM_CSV_FIELDS = ["id", "term", "definition", "related_terms", "date_added", "notes"]


def build_custom_term_records():
    return [dict(entry) for entry in load_yaml_list(CUSTOM_TERMS_SOURCE_YAML, "custom_terms")]


def write_custom_terms_csv(records):
    rows = [{
        "id": r["id"],
        "term": r.get("term", ""),
        "definition": r.get("definition", ""),
        "related_terms": "; ".join(r.get("related_terms", []) or []),
        "date_added": r.get("date_added", ""),
        "notes": r.get("notes", ""),
    } for r in records]
    write_csv(CUSTOM_TERMS_CSV, CUSTOM_TERM_CSV_FIELDS, rows)


def build_custom_terms():
    records = build_custom_term_records()
    write_json(CUSTOM_TERMS_JSON, "custom_terms", records)
    write_custom_terms_csv(records)
    write_glossary_table(
        CUSTOM_TERMS_TABLE_TEX, records,
        "Custom terms", "tab:custom_terms_corpus", "term",
        [("Term", lambda r: r.get("term", "")),
         ("Definition", lambda r: r.get("definition", ""))],
    )
    return records


# =============================================================================

def main():
    interviews = build_interviews()
    literature = build_literature()
    dictionaries = build_dictionaries()
    custom_terms = build_custom_terms()
    keyword_queries = build_keyword_queries()
    print(f"Built {len(interviews)} interviews, {len(literature)} literature "
          f"entries, {len(dictionaries)} dictionary entries, "
          f"{len(custom_terms)} custom terms, {len(keyword_queries)} "
          f"keyword queries.")
    for p in (INTERVIEWS_DATABASE_JSON, INTERVIEWS_DATABASE_CSV,
              INTERVIEWS_TABLE_TEX, INTERVIEWS_APPENDIX_TEX,
              LITERATURE_JSON, LITERATURE_CSV, LITERATURE_LIST_TEX,
              DICTIONARIES_JSON, DICTIONARIES_CSV, DICTIONARIES_TABLE_TEX,
              CUSTOM_TERMS_JSON, CUSTOM_TERMS_CSV, CUSTOM_TERMS_TABLE_TEX,
              KEYWORD_QUERIES_TABLE_TEX):
        print(f"  {p.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
