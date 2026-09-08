"""Deterministic text-integrity detection shared by the Stage-1 integrity
audit (audit_stage1_text_integrity.py) and the v2 extraction screen
(screen_v2.py).

Everything here is a pure function over a string. Nothing repairs,
transliterates, or rewrites text -- the only transformation offered is
Unicode NFC normalization, and callers always get the count of codepoints
it changed so the fact can be recorded rather than hidden.

Character classes detected (see CHARACTER_CLASS_KEYS):
  replacement_char   U+FFFD
  private_use        U+E000-F8FF (broken PDF ToUnicode maps emit these)
  control_char       C0/C1 controls other than \\t \\n \\r
  mojibake_seq       classic UTF-8-read-as-Latin-1 sequences (Ã©, â€™, Â )
  spaced_accent      a spacing accent glyph (´ ˆ ˜ ¨ ¸) next to a letter,
                     e.g. "Hervieu-Le ´ger" (detached accent in the PDF font)
  isolated_capital   "di Y cult"-style: single capital (not I/A) between
                     lowercase fragments -- ligature substitution OR a
                     kerning split ("the W est"); which one is a
                     document-level judgement, see document_flags()
  ligature_pattern   the subset of isolated_capital using W/V/Y, the letters
                     Expert-font ligature glyphs decode to
  latin_ext_b_letters / greek_letters / cyrillic_letters
                     letters from those blocks (legitimate in names, quotes,
                     transliteration; only a cipher signal when mixed into
                     Latin words -- see mixed_script_tokens / cipher_lines)
  mixed_script_tokens tokens containing ASCII letters AND Latin-Ext-B /
                     Greek / PUA characters ("Boliǀia's", "ǁhite")
  nbsp, soft_hyphen, hyphen_linebreak   whitespace/hyphenation residue

All codepoint ranges are written as \\uXXXX escapes on purpose: several of
these characters are invisible or render as boxes, and a literal that a
tool silently drops would turn a character class into a stray hyphen.

Latin Extended-A (U+0100-017F) is deliberately NOT in the suspicious set:
it holds œ, ā, ī, š, ž, ş, ğ, ı ... -- ordinary French, transliterated
Sanskrit/Japanese and Central-European names (MIVILUDES alone has 118
œ-tokens). The three broken-CMap documents in this corpus are caught by
their Extended-B / Greek / Private-Use glyphs (Card 2019: ǀ ǁ ƌ Ǉ ϭ Ϭ ͞ ͟).
"""
from __future__ import annotations

import re
import unicodedata

CHARACTER_CLASS_KEYS = (
    "replacement_char", "private_use", "private_use_max_run", "control_char",
    "mojibake_seq", "spaced_accent", "isolated_capital", "ligature_pattern",
    "latin_ext_b_letters", "greek_letters", "cyrillic_letters",
    "mixed_script_tokens", "nbsp", "soft_hyphen", "hyphen_linebreak",
)

PUA_START, PUA_END = "", ""

_REPLACEMENT_RE = re.compile("�")
_PRIVATE_USE_RE = re.compile("[-]")
_PRIVATE_USE_RUN_RE = re.compile("[-]+")
_CONTROL_RE = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_MOJIBAKE_RE = re.compile("Ã[-¿]|â€.|Â[  ]")
_SPACING_ACCENTS = "´ˆ˜¨¸"  # ´ ˆ ˜ ¨ ¸
_SPACED_ACCENT_RE = re.compile(
    rf"[^\W\d_] ?[{_SPACING_ACCENTS}](?= ?[^\W\d_])|[{_SPACING_ACCENTS}] ?[^\W\d_]"
)
# Single capital other than I/A between two lowercase fragments, each
# fragment itself not part of a longer word.
_ISOLATED_CAPITAL_RE = re.compile(r"(?<![A-Za-z])[a-z]+ [B-HJ-Z] [a-z]+(?![A-Za-z])")
_LIGATURE_PATTERN_RE = re.compile(r"(?<![A-Za-z])[a-z]+ [WVY] [a-z]+(?![A-Za-z])")
_LATIN_EXT_B_RE = re.compile("[ƀ-ɏ]")
_GREEK_RE = re.compile("[Ͱ-Ͽ]")
_CYRILLIC_RE = re.compile("[Ѐ-ӿ]")
_SUSPICIOUS_SCRIPT_RE = re.compile("[ƀ-ɏͰ-Ͽ-]")
_ASCII_LETTER_RE = re.compile("[A-Za-z]")
_TOKEN_RE = re.compile(r"\S+")
_NBSP_RE = re.compile(" ")
_SOFT_HYPHEN_RE = re.compile("­")
_HYPHEN_LINEBREAK_RE = re.compile(r"\w-\n\w")

CIPHER_LINE_MIN_LETTERS = 10
CIPHER_LINE_SHARE_THRESHOLD = 0.05
# A broken ToUnicode map typically substitutes only the glyphs of one font
# subset, so a fully corrupted paragraph can show a suspicious-letter share
# of just 1-3% (Card 2019: 27 mixed tokens in a 1,111-letter line at 3.4%).
# Docling emits one line per text item, i.e. paragraph-long lines, so the
# share test alone misses these; two or more mixed-script tokens on a line
# is the second, independent trigger.
CIPHER_LINE_MIN_MIXED_TOKENS = 2


def nfc_normalize(text: str) -> tuple[str, int]:
    """NFC-normalize `text`; returns (normalized, changed_codepoints) where
    changed_codepoints is the number of codepoints in the *original* string
    that are not reproduced identically at the same position after
    normalization (0 when the text was already NFC)."""
    normalized = unicodedata.normalize("NFC", text)
    if normalized == text:
        return normalized, 0
    changed = 0
    for original_char, normalized_char in zip(text, normalized):
        if original_char != normalized_char:
            changed += 1
    changed += abs(len(text) - len(normalized))
    return normalized, changed


def _letters_only(pattern: re.Pattern, text: str) -> int:
    return sum(1 for match in pattern.finditer(text) if match.group(0).isalpha())


def mixed_script_tokens(text: str) -> list[str]:
    """Tokens mixing ASCII letters with Latin-Ext-B / Greek / PUA characters."""
    return [
        token for token in _TOKEN_RE.findall(text)
        if _ASCII_LETTER_RE.search(token) and _SUSPICIOUS_SCRIPT_RE.search(token)
    ]


def count_character_classes(text: str) -> dict[str, int]:
    runs = [len(m.group(0)) for m in _PRIVATE_USE_RUN_RE.finditer(text)]
    return {
        "replacement_char": len(_REPLACEMENT_RE.findall(text)),
        "private_use": len(_PRIVATE_USE_RE.findall(text)),
        "private_use_max_run": max(runs) if runs else 0,
        "control_char": len(_CONTROL_RE.findall(text)),
        "mojibake_seq": len(_MOJIBAKE_RE.findall(text)),
        "spaced_accent": len(_SPACED_ACCENT_RE.findall(text)),
        "isolated_capital": len(_ISOLATED_CAPITAL_RE.findall(text)),
        "ligature_pattern": len(_LIGATURE_PATTERN_RE.findall(text)),
        "latin_ext_b_letters": _letters_only(_LATIN_EXT_B_RE, text),
        "greek_letters": _letters_only(_GREEK_RE, text),
        "cyrillic_letters": _letters_only(_CYRILLIC_RE, text),
        "mixed_script_tokens": len(mixed_script_tokens(text)),
        "nbsp": len(_NBSP_RE.findall(text)),
        "soft_hyphen": len(_SOFT_HYPHEN_RE.findall(text)),
        "hyphen_linebreak": len(_HYPHEN_LINEBREAK_RE.findall(text)),
    }


def has_hard_corruption(text: str) -> str | None:
    """The span-level hard-reject classes shared with screen_v2 (everything
    that is corruption with high confidence regardless of document). Returns
    the rejection code or None."""
    if _REPLACEMENT_RE.search(text):
        return "integrity_replacement_char"
    if _PRIVATE_USE_RE.search(text):
        return "integrity_private_use"
    if _CONTROL_RE.search(text):
        return "integrity_control_char"
    if _MOJIBAKE_RE.search(text):
        return "integrity_mojibake"
    return None


def ligature_substitution_present(text: str) -> bool:
    return bool(_LIGATURE_PATTERN_RE.search(text))


def soft_flags(text: str) -> list[str]:
    """Non-fatal integrity flags for a span (never a rejection)."""
    flags = []
    if _SPACED_ACCENT_RE.search(text):
        flags.append("spaced_accent")
    if _LATIN_EXT_B_RE.search(text) or _GREEK_RE.search(text) or _CYRILLIC_RE.search(text):
        flags.append("unusual_script")
    if _ISOLATED_CAPITAL_RE.search(text):
        flags.append("isolated_capital")
    if _NBSP_RE.search(text):
        flags.append("contains_nbsp")
    if _SOFT_HYPHEN_RE.search(text):
        flags.append("contains_soft_hyphen")
    if _HYPHEN_LINEBREAK_RE.search(text):
        flags.append("contains_hyphen_linebreak")
    return flags


def cipher_line_share(line: str) -> tuple[int, float]:
    """(letter_count, share of letters from Latin-Ext-B / Greek / PUA)."""
    letters = [c for c in line if c.isalpha() or PUA_START <= c <= PUA_END]
    if not letters:
        return 0, 0.0
    suspicious = sum(1 for c in letters if _SUSPICIOUS_SCRIPT_RE.match(c))
    return len(letters), suspicious / len(letters)


def find_cipher_lines(
    text: str,
    share_threshold: float = CIPHER_LINE_SHARE_THRESHOLD,
    min_letters: int = CIPHER_LINE_MIN_LETTERS,
    min_mixed_tokens: int = CIPHER_LINE_MIN_MIXED_TOKENS,
) -> list[tuple[int, float, int, str]]:
    """Lines (index, share, mixed_token_count, text) showing the
    broken-ToUnicode-map signature: either the Latin-Ext-B / Greek / PUA
    letter share reaches `share_threshold`, or the line holds at least
    `min_mixed_tokens` tokens that mix ASCII letters with those scripts. A
    document-level decision (document_flags) still has to say whether these
    lines are a cipher region or, e.g., a legitimate Greek quotation."""
    flagged = []
    for index, line in enumerate(text.split("\n")):
        letters, share = cipher_line_share(line)
        mixed = len(mixed_script_tokens(line))
        if letters >= min_letters and (share >= share_threshold or mixed >= min_mixed_tokens):
            flagged.append((index, share, mixed, line))
    return flagged


# Document-level flag thresholds. Rates are per 1,000 words. Calibrated on
# the current corpus (see audit_stage1_text_integrity.py's config output):
# Lalich 2004 = 8.3 ligature-pattern hits / 1k words (Expert-font ligature
# glyphs decoded as W/V/Y); Tomkins 2010 = 1.0 and Werbner & Basu 2002 = 0.57,
# both of which are kerning splits of real capitalised words ("the W est",
# "der V eer") rather than ligature substitution -- hence the 2.0 cut for the
# hard flag and a separate informational note at 0.5.
DOCUMENT_FLAG_THRESHOLDS = {
    "private_use_cipher_min_chars": 10,
    "cmap_cipher_min_mixed_script_tokens": 10,
    "cmap_cipher_min_cipher_lines": 3,
    "ligature_substitution_min_rate_per_1k": 2.0,
    "spaced_capital_splits_min_rate_per_1k": 0.5,
    "detached_accents_min_rate_per_1k": 0.5,
}

HARD_FLAG_ORDER = (
    "replacement_chars", "control_chars", "mojibake", "private_use_cipher",
    "cmap_cipher", "ligature_substitution", "detached_accents",
)


def document_flags(
    counts: dict[str, int], words: int, cipher_lines: int,
    thresholds: dict[str, float] = DOCUMENT_FLAG_THRESHOLDS,
) -> tuple[list[str], list[str]]:
    """Returns (hard_flags, notes). Hard flags name a defect the v2 screen
    acts on; notes are informational."""
    per_1k = 1000.0 / max(words, 1)
    hard: list[str] = []
    notes: list[str] = []
    if counts["replacement_char"] > 0:
        hard.append("replacement_chars")
    if counts["control_char"] > 0:
        hard.append("control_chars")
    if counts["mojibake_seq"] > 0:
        hard.append("mojibake")
    if counts["private_use"] >= thresholds["private_use_cipher_min_chars"]:
        hard.append("private_use_cipher")
    if (counts["mixed_script_tokens"] >= thresholds["cmap_cipher_min_mixed_script_tokens"]
            or cipher_lines >= thresholds["cmap_cipher_min_cipher_lines"]):
        hard.append("cmap_cipher")
    ligature_rate = counts["ligature_pattern"] * per_1k
    isolated_rate = counts["isolated_capital"] * per_1k
    if ligature_rate >= thresholds["ligature_substitution_min_rate_per_1k"]:
        hard.append("ligature_substitution")
    elif max(ligature_rate, isolated_rate) >= thresholds["spaced_capital_splits_min_rate_per_1k"]:
        notes.append("spaced_capital_splits")
    if counts["spaced_accent"] * per_1k >= thresholds["detached_accents_min_rate_per_1k"]:
        hard.append("detached_accents")
    if counts["hyphen_linebreak"] > 0:
        notes.append("hyphen_linebreaks_present")
    if counts["nbsp"] > 0:
        notes.append("nbsp_present")
    if counts["soft_hyphen"] > 0:
        notes.append("soft_hyphens_present")
    if counts["greek_letters"] + counts["cyrillic_letters"] + counts["latin_ext_b_letters"] > 0 \
            and "cmap_cipher" not in hard:
        notes.append("non_latin_letters_present")
    return hard, notes
