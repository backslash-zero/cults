"""Chunking for the exhaustive interview extraction pipeline: one unit per
whole transcript wherever it fits the model's context window, with a
turn-boundary-aligned split as a fallback for longer transcripts.

Unlike the raw transcript file, a unit's `text` contains no structural
markup at all: no `Interviewer:` / `Interviewee:` / `Interview Notes:`
labels, no `---` separators, and no transcript header (date, demographics).
Those labels are real structure -- they're what decide the turn boundaries
and, downstream, the `attribution` on every expression -- but they are not
anyone's speech, so they should never be extractable text or sit inside an
`embedding_text`/`context_window`. This module parses them once, here, and
throws the label strings themselves away.

Granularity: a raw `Interviewer:`/`Interviewee:` turn is very often several
sentences covering several distinct ideas in a row (a documentary, then a
named group, then a definitional claim) -- far too coarse a unit to hand the
model as "one turn," since exhaustive segmentation then has to do all the
work of finding those boundaries inside one large blob. So each raw turn is
first split into sentences (a plain punctuation heuristic, imperfect on
informal speech but a real improvement over one block per turn), and it is
those SENTENCES, not raw turns, that become the blank-line-separated blocks
in `text` -- several in a row can belong to the same speaker's reply.
`turn_roles` carries one role ("interviewer" or "participant") per block, in
order, for `interview_segment_labels.py` to pair back up positionally (see
`segment_spans()` there) without ever needing to see a label again -- that
pairing is also the authoritative list of what gets embedded, guaranteeing
one archive record per segment regardless of what the model does with it.
`Interview Notes:` turns (transcriber commentary -- never a speaker's own
words) and the transcript header are dropped entirely, not merely excluded
downstream.

Deliberately not a parameterization of chunking.py: chunking.py's
2-5-paragraph / 300-700-word packer is tuned for literature prose density
and, applied to an interview, arbitrarily fragments the conversation into
2-5-turn windows -- exactly the shape that made the existing v2 interview
pipeline unable to see a short reply next to the question that prompted it.
Interviews are short (measured directly on the current 28-interview corpus:
111-1536 words, avg 435) and want the model to see the WHOLE conversation
at once wherever that fits.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Generous headroom over the observed 1536-word max in the current corpus,
# so ordinary corpus growth doesn't immediately hit the split fallback.
DEFAULT_MAX_WORDS = 1800

_LABEL_RE = re.compile(r"^(Interviewer|Interviewee|Interview Notes)\s*:\s*", re.MULTILINE | re.IGNORECASE)
_ROLE_BY_LABEL = {"interviewer": "interviewer", "interviewee": "participant", "interview notes": "notes"}
# A "---" separator line some transcripts use between the header/demographics
# block and the first turn, or trailing the last turn before an
# "Interview Notes:" section or end of file -- structural markup, never
# something a speaker said, stripped the same way the labels themselves are.
# Threshold is 2+ dashes, not 3+: at least one real transcript
# (b3-aug16-1517) ends in a stray "--" rather than the usual "---", almost
# certainly the same separator convention with a dropped character, not
# anyone's actual words -- nobody ends a spoken answer in bare hyphens.
_TRAILING_SEPARATOR_RE = re.compile(r"(?:\n\s*)?-{2,}\s*$")
# Same sentence-boundary heuristic screen_v2.py's own (private)
# _SENTENCE_SPLIT_RE uses -- duplicated rather than imported, since it's not
# a public function there.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

KEPT_ROLES = ("interviewer", "participant")


@dataclass
class TranscriptUnit:
    document_id: str
    chunk_index: int
    page_range: list[int]
    text: str
    turn_roles: list[str] = field(default_factory=list)


def _parse_turns(raw_text: str) -> list[tuple[str, str]]:
    """Returns [(role, content), ...] for every Interviewer:/Interviewee:/
    Interview Notes: turn in `raw_text`, in order. Text before the first
    Interviewer: label (the transcript header) is dropped -- it is never
    anyone's speech, the same rule screen_v2.speaker_turns() applies."""
    labels = list(_LABEL_RE.finditer(raw_text))
    first_interviewer = next((m for m in labels if m.group(1).lower() == "interviewer"), None)
    if first_interviewer is None:
        return []
    active = [m for m in labels if m.start() >= first_interviewer.start()]
    turns = []
    for i, m in enumerate(active):
        content_end = active[i + 1].start() if i + 1 < len(active) else len(raw_text)
        content = raw_text[m.end():content_end].strip()
        content = _TRAILING_SEPARATOR_RE.sub("", content).strip()
        turns.append((_ROLE_BY_LABEL[m.group(1).lower()], content))
    return turns


def _split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text)]
    return [p for p in parts if p]


def _sentence_groups(raw_text: str) -> list[list[tuple[str, str]]]:
    """One group per raw Interviewer:/Interviewee: turn (skipping "notes"
    and empty turns), each group being that turn's own sentences as
    (role, sentence) tuples -- role is the same for every sentence in a
    group. Grouping is kept (rather than flattening immediately) so the
    split fallback below can still find "the start of a new interviewer
    turn" at the right granularity, not just "the start of any sentence
    that happens to be attributed to the interviewer.\""""
    groups = []
    for role, content in _parse_turns(raw_text):
        if role not in KEPT_ROLES or not content:
            continue
        sentences = _split_sentences(content) or [content]
        groups.append([(role, s) for s in sentences])
    return groups


def _join(entries: list[tuple[str, str]]) -> tuple[str, list[str]]:
    return "\n\n".join(content for _role, content in entries), [role for role, _content in entries]


def build_transcript_units(
    document_id: str, pages: list[dict], max_words: int = DEFAULT_MAX_WORDS,
) -> list[TranscriptUnit]:
    """Interviews are always a single page (see prepare_interviews.py). One
    unit covers the whole transcript's sentences when they fit `max_words`;
    otherwise it is split into two units at the sentence where a new
    INTERVIEWER turn begins (other than the very first turn, which would
    leave a near-empty first unit) -- never inside a turn, and never inside
    a sentence.
    """
    if not pages:
        return []
    page = sorted(pages, key=lambda p: p["page_number"])[0]
    groups = _sentence_groups(page["text"] or "")
    if not groups:
        return []
    page_range = [page["page_number"], page["page_number"]]

    flat = [entry for group in groups for entry in group]
    text, roles = _join(flat)
    if len(text.split()) <= max_words:
        return [TranscriptUnit(document_id=document_id, chunk_index=0, page_range=page_range, text=text, turn_roles=roles)]

    group_starts, idx = [], 0
    for group in groups:
        group_starts.append(idx)
        idx += len(group)
    candidates = [group_starts[i] for i, group in enumerate(groups) if i > 0 and group[0][0] == "interviewer"]
    if not candidates:
        # No second interviewer turn to split at (shouldn't happen on a real
        # transcript) -- keep one oversized unit rather than cutting mid-turn.
        return [TranscriptUnit(document_id=document_id, chunk_index=0, page_range=page_range, text=text, turn_roles=roles)]

    cumulative_words, total = [], 0
    for _role, content in flat:
        total += len(content.split())
        cumulative_words.append(total)
    midpoint = cumulative_words[-1] / 2
    split_at = min(candidates, key=lambda i: abs(cumulative_words[i - 1] - midpoint))

    first_text, first_roles = _join(flat[:split_at])
    second_text, second_roles = _join(flat[split_at:])
    return [
        TranscriptUnit(document_id=document_id, chunk_index=0, page_range=page_range, text=first_text, turn_roles=first_roles),
        TranscriptUnit(document_id=document_id, chunk_index=1, page_range=page_range, text=second_text, turn_roles=second_roles),
    ]
