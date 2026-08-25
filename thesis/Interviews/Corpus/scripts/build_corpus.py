#!/usr/bin/env python3
"""Build the interview corpus database and LaTeX appendix.

Reads Corpus/metadata/interviews_source.yaml plus the cleaned/translated
transcript files under Corpus/cleaned/<id>/, computes derived counts, and
writes:
  - Corpus/metadata/database.json   (canonical: text + translation + metadata)
  - Corpus/metadata/database.csv    (flat metadata export, no full text)
  - 04_Appendix/Interview_Corpus/database_table.tex  (longtable, from the JSON)
  - 04_Appendix/Interview_Corpus/appendix.tex        (one subsection per interview)

Requires PyYAML. Run with: python3 build_corpus.py
"""
import csv
import json
import re
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
CORPUS_DIR = SCRIPT_DIR.parent
INTERVIEWS_DIR = CORPUS_DIR.parent
THESIS_DIR = INTERVIEWS_DIR.parent
APPENDIX_DIR = THESIS_DIR / "04_Appendix" / "Interview_Corpus"

SOURCE_YAML = CORPUS_DIR / "metadata" / "interviews_source.yaml"
CLEANED_DIR = CORPUS_DIR / "cleaned"
DATABASE_JSON = CORPUS_DIR / "metadata" / "database.json"
DATABASE_CSV = CORPUS_DIR / "metadata" / "database.csv"
APPENDIX_TEX = APPENDIX_DIR / "appendix.tex"
TABLE_TEX = APPENDIX_DIR / "database_table.tex"

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


def load_interviews():
    with open(SOURCE_YAML, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["interviews"]


def build_records():
    records = []
    for entry in load_interviews():
        iid = entry["id"]
        idir = CLEANED_DIR / iid
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
        record["corrections_log"] = str(corrections_path.relative_to(CORPUS_DIR))
        records.append(record)
    return records


def write_json(records):
    DATABASE_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(DATABASE_JSON, "w", encoding="utf-8") as f:
        json.dump({"interviews": records}, f, ensure_ascii=False, indent=2)
        f.write("\n")


CSV_FIELDS = [
    "id", "batch", "source_file", "date_time", "date_time_precision", "method",
    "language", "translated", "translation_language", "interviewer",
    "participant_id", "location", "age", "gender", "nationality",
    "main_language", "language_spoken", "n_questions", "n_answers",
    "total_word_count", "notes",
]


def write_csv(records):
    DATABASE_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(DATABASE_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in records:
            interviewee = r.get("interviewee", {})
            writer.writerow({
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


LATEX_SPECIAL = {
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
    "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    "\\": r"\textbackslash{}",
}


def escape_latex(s):
    if s is None:
        return ""
    return "".join(LATEX_SPECIAL.get(ch, ch) for ch in str(s))


PAREN_SUFFIX_RE = re.compile(r"\s*\([^)]*\)\s*$")


def short_field(s):
    """Strip a trailing parenthetical caveat (e.g. '(see notes — ...)') for the
    compact table view; the full string is still in the metadata slip/JSON."""
    return PAREN_SUFFIX_RE.sub("", str(s)).strip()


def write_table_tex(records):
    APPENDIX_DIR.mkdir(parents=True, exist_ok=True)
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
    TABLE_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_appendix_tex(records):
    APPENDIX_DIR.mkdir(parents=True, exist_ok=True)
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
    APPENDIX_TEX.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main():
    records = build_records()
    write_json(records)
    write_csv(records)
    write_table_tex(records)
    write_appendix_tex(records)
    print(f"Built {len(records)} interview records.")
    for p in (DATABASE_JSON, DATABASE_CSV, TABLE_TEX, APPENDIX_TEX):
        print(f"  {p.relative_to(THESIS_DIR)}")


if __name__ == "__main__":
    main()
