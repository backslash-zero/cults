#!/usr/bin/env python3
"""Run the corpus construction protocol against the OpenAlex API
(thesis/03_Content/3_Methods.tex, "Corpus construction protocol").

1. Search: run every keyword query in thesis/corpus/literature/keyword-queries.txt
   against OpenAlex, restricted to publication_year:<FROM_YEAR>-<TO_YEAR>.
   Direct label-term and framing-term queries (cult/sect/new religious
   movements/extreme beliefs — everything in TITLE_ONLY_MARKERS) search the
   *title* only (filter=title.search:<query>) — searching the abstract for
   these lets through anything that merely mentions the phrase in passing
   (confirmed for "extreme beliefs": 50 hits at the abstract level, mostly
   generic voting-behavior/wellbeing-psychology papers, vs. 43 uniformly
   on-topic hits when restricted to the title). "NRM" is deliberately not
   searched as a bare acronym — it collides with "Natural Resource
   Management" and "National Resistance Movement", both of which fall under
   the same allowed fields as genuine hits, so only the full phrase "new
   religious movements" is used. Only "sectarian drifts" remains a framing
   term searched at the title-and-abstract level, since it's less likely to
   appear in a title verbatim. None of the title-only queries carry a
   disciplinary anchor (earlier versions paired each label term with "AND
   sociology"/"AND history"/etc., but that just requires the literal
   discipline word in the title — a poor, overly strict proxy for
   disciplinary relevance that's redundant with step 3's field/subfield
   screening, and it produced queries with zero hits, e.g. "sect" AND
   anthropology). Because a single unconditioned query's top results skew
   toward whatever sense has the highest raw citation counts (historically,
   ancient-religion/archaeology uses of "cult", most of which get removed at
   screening), the title-only queries use a much larger PER_PAGE (200,
   OpenAlex's max) than the one remaining framing-term query (50), so
   there's still enough depth left after screening to populate every
   surviving discipline.
2. Merge & deduplicate: pool every query's hits into one dataset, first by
   OpenAlex work ID (falling back to DOI), then by normalized title +
   first-author surname — this second pass catches near-duplicates that
   don't share a DOI (reprints/translations, or the same work indexed under
   several DOIs, e.g. versioned Zenodo preprints).
3. Screen: keep only book / book-chapter / article types (Type inclusions),
   AND only works whose OpenAlex primary_topic.field is one of
   FIELD_ALLOWLIST (Social Sciences, Arts and Humanities, Psychology) — this
   drops works OpenAlex has classified into unrelated fields (Medicine,
   Engineering, Agricultural and Biological Sciences, etc.). Within those
   fields, further exclusions apply, each targeting a specific recurring
   false-positive class rather than an arbitrary blacklist:
     - primary_topic.subfield in SUBFIELD_EXCLUDE (History, Archeology,
       Classics — all definitionally about antiquity, so "cult"/"sect"
       there is reliably the ancient-ritual sense, not the cults/NRMs-as-
       social-phenomenon sense this corpus targets; Literature and Literary
       Theory for a related reason — it analyses fictional/mythological
       texts, so "cult" there is a literary/mythological reference, not a
       real-world group);
     - primary_topic.display_name in TOPIC_EXCLUDE ("Biblical Studies and
       Interpretation", "Folklore, Mythology, and Literature Studies" — the
       same ancient/legendary-content rationale as the subfield exclusions
       above, but at the finer topic level, since these can occur under
       subfields not otherwise excluded, e.g. Religious studies);
     - any primary_topic.display_name containing "history" (case-
       insensitive) — generalizes the History subfield exclusion to catch
       cases where the *topic* is historically framed but the *subfield*
       is something else entirely (e.g. "Mao Cult", topic "Chinese history
       and philosophy", subfield Sociology and Political Science; "Dragon
       Cults in The Tale of the Heike", topic "Japanese History and
       Culture", subfield Cultural Studies);
     - titles matching TITLE_EXCLUDE_PATTERNS ("cult of ...", "personality
       cult", "cult following"/"classic"/"status"/"collectors"/"phenomenon"/
       "television"/"tv"/"film"/"movie"/"cinema", "fitness cult", "culting"
       (as in "the culting of brands" — the marketing-devotion sense),
       "cult(ure)"/"cult and culture"/"cults and cultures", "academic sect"
       — all variants of the fame/devotion metaphor sense, \cultfame in
       thesis/03_Content/2_Theory.tex, a title pun, or "sect" used for an
       academic/ideological faction rather than a religious group);
     - the taxonomic-abbreviation regex (bare "sect." as in botanical/
       zoological nomenclature, e.g. "Sophora sect. Edwardsia");
     - the all-caps acronym regex ("SECT"/"SECTS" as an acronym, e.g. the
       "Scale for Effective Communication in Team Sports (SECTS)");
     - the ancient-civilization/period-marker list (ANCIENT_MARKERS) —
       addresses the same ancient-ritual sense as the subfield exclusion
       above, but at the title level, because it keeps resurfacing under
       different, unpredictable subfields each run (Demography,
       Communication, Anthropology, Religious studies have all produced an
       ancient-cult hit at one point or another) — chasing it subfield by
       subfield doesn't scale.
   Everything excluded is logged with its specific reason.
4. Rank: within each of {book, article} (book-chapter counts as book), score
   = normalized cited_by_count (min-max within category) + a bonus per extra
   distinct query that retrieved the record (cross-channel presence).
5. Stratify: bucket by OpenAlex's primary_topic.subfield.display_name (the
   level that actually corresponds to the disciplines named in the corpus
   construction protocol — its coarser `field` level produced nonsensical
   single-paper "disciplines" in an earlier version of this script), and
   keep the top N per discipline per category — but only entries with some
   corroborating evidence (cited_by_count > 0 OR found by more than one
   query); a bucket that would otherwise be empty falls back to the
   unfiltered ranking rather than disappearing, since a thin bucket is more
   informative than a hidden one. Without this floor, thin buckets get
   padded with unvalidated, single-source noise just to reach N (this is
   how a triplicated self-published Zenodo book once "ranked" in a
   psychology bucket — three different DOIs for the same 0-citation book).

By default this is a *retrieval* step only — output is a plain-text report
(thesis/corpus/literature/openalex_search_results.txt) for manual review (thesis
author approval) before any record is folded into the approved literature
corpus. Passing --approve additionally writes every stratified candidate
(the exact set shown in that report's BOOKS/ARTICLES sections — not the
excluded log) to thesis/corpus/literature/raw/OpenAlex_corpus.bib, in the same
Zotero-export .bib format thesis/corpus/scripts/build_corpus.py already parses for
Books.bib, so a subsequent `build_corpus.py` run folds them into
literature.json/.csv and the reference_list.tex appendix. Re-running with
--approve overwrites that file with the current run's candidates — it is
not additive across runs.

Authentication: since 13 Feb 2026 OpenAlex requires an API key on every
request (the old anonymous/"polite pool" tier is now far more restricted).
Get a free key at openalex.org/settings/api (10x the daily credit budget of
the unauthenticated tier) and set it as the OPENALEX_API_KEY environment
variable before running this script — never hardcode it here or commit it.

Usage:
  OPENALEX_API_KEY=... python3 thesis/corpus/scripts/fetch_openalex_corpus.py             # review only
  OPENALEX_API_KEY=... python3 thesis/corpus/scripts/fetch_openalex_corpus.py --approve   # also write the .bib
Requires: requests (`pip install requests`)
"""
import os
import re
import sys
import time
import unicodedata
import urllib.parse
from collections import defaultdict
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
CORPUS_DIR = SCRIPT_DIR.parent
KEYWORD_QUERIES_TXT = CORPUS_DIR / "literature" / "keyword-queries.txt"
OUTPUT_TXT = CORPUS_DIR / "literature" / "openalex_search_results.txt"
LITERATURE_RAW_DIR = CORPUS_DIR / "literature" / "raw"
APPROVED_BIB_PATH = LITERATURE_RAW_DIR / "OpenAlex_corpus.bib"

API_BASE = "https://api.openalex.org/works"
MAILTO = "celestin.meunier@gmail.com"  # legacy "polite pool" identifier — kept
                                       # alongside api_key, doesn't hurt
API_KEY = os.environ.get("OPENALEX_API_KEY")
FROM_YEAR = 2001
TO_YEAR = 2026
PER_PAGE_DIRECT = 200  # direct label-term queries (title-search, no
                       # disciplinary anchor) — see module docstring step 1
PER_PAGE_FRAMING = 50  # adjacent/framing-term queries (title-and-abstract)
REQUEST_PAUSE_S = 2.0

MAX_RETRIES = 3
MAX_RETRY_WAIT_S = 20  # cap on any single backoff sleep, regardless of what
                       # the server's Retry-After header asks for — failing
                       # the request and moving on is preferable to hanging
                       # indefinitely. With an API_KEY set this should rarely
                       # trigger at all (100k credits/day vs. ~5 requests per
                       # full run); it mainly guards the unauthenticated
                       # fallback path.

TYPE_INCLUDE = {"book", "book-chapter", "article"}
BOOK_TYPES = {"book", "book-chapter"}
TOP_N_PER_DISCIPLINE = 5

# Direct label-term queries are searched in the title only (see module
# docstring, step 1); anything else (adjacent/framing terms) is searched in
# title-and-abstract. Matched as a substring against the (lowercased) query.
# "extreme beliefs" was moved here after direct API testing showed the
# title-and-abstract version pulls in generic voting-behavior/wellbeing-
# psychology noise (50 hits) that title-only search avoids (43 hits, all
# genuinely about extremism/radicalization/delusion).
TITLE_ONLY_MARKERS = ('"cult"', '"sect"', "new religious movements", '"nrm"',
                       "extreme beliefs")

# primary_topic.field values a work must fall under to be kept — OpenAlex's
# 26 top-level fields, restricted to the ones relevant to cults/sects/NRM
# scholarship.
FIELD_ALLOWLIST = {"Social Sciences", "Arts and Humanities", "Psychology"}

# primary_topic.subfield values excluded even within FIELD_ALLOWLIST: in
# practice these subfields' "cult"/"sect" hits are reliably about ancient
# ritual practice or artifacts (e.g. "the cult of Mithras", votive "cult
# objects") rather than the cults/NRMs-as-social-phenomenon sense this
# corpus targets. Classics joins History/Archeology for the same reason: the
# field is by definition about ancient Greek/Roman studies. Literature and
# Literary Theory is excluded for a related but distinct reason: it analyses
# fictional/mythological texts (e.g. Isis and Osiris in Plutarch, Jane
# Austen's novels), so "cult" there is reliably a literary or mythological
# reference, not a real-world group — even though this same false-positive
# class also turns up classified under other subfields (see the
# topic-level rules below, which catch those cases instead).
SUBFIELD_EXCLUDE = {"History", "Archeology", "Classics", "Literature and Literary Theory"}

# primary_topic.display_name (the most granular OpenAlex classification,
# one level finer than subfield) values excluded outright: both are
# reliably about ancient/legendary religious content regardless of which
# subfield they happen to be filed under.
TOPIC_EXCLUDE = {"Biblical Studies and Interpretation",
                  "Folklore, Mythology, and Literature Studies"}

# Any primary_topic.display_name containing "history" or "antiquity" is
# excluded, the same way the History subfield is: this catches the
# recurring ancient/period-specific sense of "cult"/"sect" when OpenAlex has
# filed the *subfield* as something else entirely — e.g. "Mao Cult" (topic
# "Chinese history and philosophy", subfield Sociology and Political
# Science), "Dragon Cults in The Tale of the Heike" (topic "Japanese
# History and Culture", subfield Cultural Studies), a hip-hop-conspiracy-
# theory paper topic-classified as "Music History and Culture", or "The
# Rise of the Hero Cult and the New Simonides" (topic "Classical Antiquity
# Studies", subfield Anthropology — caught by "antiquity", since it has no
# "history" in its topic name at all). Matched case-insensitively as a
# substring.
_TOPIC_HISTORICAL_MARKERS = ("history", "antiquity")

# Title substrings that reliably signal the *wrong* sense of "cult" — either
# the fame/devotion metaphor (\cultfame in thesis/03_Content/2_Theory.tex:
# "the cult of celebrity"), its "personality cult" (political leader-
# worship) and fandom/pop-culture variants (following/classic/status/
# collectors/television/film/fitness), or a title pun where "cult" is
# typographically or lexically paired with "culture" (e.g. "The Confidence
# Cult(ure)", "Between Cult and Culture"). Also covers "sect" used
# metaphorically for an academic/ideological faction rather than a
# religious group (e.g. "Academic Sects as Impediments to Understanding").
# Matched case-insensitively as a plain substring.
TITLE_EXCLUDE_PATTERNS = (
    "cult of", "personality cult", "cult following", "cult classic",
    "cult status", "cult collectors", "cult phenomenon", "cult television",
    "cult tv", "cult film", "cult movie", "cult cinema", "fitness cult",
    "culting", "cult(ure)", "cult and culture", "cults and cultures",
    "academic sect",
)

# Regex-based title exclusions, for patterns substring matching can't
# express. Each entry is (compiled_pattern, reason, case_sensitive_input) —
# case_sensitive_input selects whether the pattern is matched against the
# raw title or its lowercased form.
_TAXONOMIC_SECT_RE = re.compile(r"\bsect\.\s")  # botanical/zoological
                                                 # "section" abbreviation,
                                                 # e.g. "Sophora sect.
                                                 # Edwardsia" — "NOT section"
                                                 # doesn't catch this since
                                                 # "section" isn't spelled
                                                 # out.
_ACRONYM_SECTS_RE = re.compile(r"\bSECTS?\b")   # matched against the raw
                                                 # (unlowercased) title: an
                                                 # all-caps "SECT"/"SECTS" is
                                                 # reliably an acronym (e.g.
                                                 # "Scale for Effective
                                                 # Communication in Team
                                                 # Sports (SECTS)"), not the
                                                 # English word.

# Ancient-civilization/period markers: this cluster of terms keeps
# resurfacing the ancient-ritual sense of "cult"/"sect" under a different,
# unpredictable subfield almost every run (Classics, Demography via "Cultes
# et sanctuaires de l'île de Cos", Communication via "Classic Maya Community
# Cults", Anthropology via "the cult of Mithras", Religious studies via
# "Hezekiah's Alleged Cultic Centralization") — chasing it subfield by
# subfield is whack-a-mole, so it's addressed here directly at the title
# level instead. Word-boundary matched against the lowercased title. Trade-
# off: a rare modern paper that names an ancient civilization in its title
# (e.g. a comparative study) could be wrongly excluded.
ANCIENT_MARKERS = (
    "ancient", "antiquity", "roman", "greek", "hellenistic", "homeric",
    "hero cult", "hero-cult", "mesopotamia", "mesopotamian", "byzantine",
    "medieval", "bronze age", "iron age", "neolithic", "prehistoric", "maya",
    "aztec", "egyptian", "pharaonic",
)
_ANCIENT_MARKER_RE = re.compile(
    r"\b(" + "|".join(re.escape(m) for m in ANCIENT_MARKERS) + r")\b"
)


def warn_if_no_api_key():
    if not API_KEY:
        print("WARNING: no OPENALEX_API_KEY set — running on the "
              "unauthenticated tier, which OpenAlex rate-limits heavily as "
              "of Feb 2026. Get a free key at openalex.org/settings/api and "
              "`export OPENALEX_API_KEY=...` before rerunning.")


def load_queries():
    queries = []
    for line in KEYWORD_QUERIES_TXT.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        queries.append(stripped)
    return queries


def is_title_only(query):
    lowered = query.lower()
    return any(marker in lowered for marker in TITLE_ONLY_MARKERS)


def http_get_with_retry(url):
    """GET url, retrying on HTTP 429 with capped exponential backoff.
    Raises for any other error status. Returns the parsed JSON body."""
    backoff = 5
    for attempt in range(1, MAX_RETRIES + 1):
        resp = requests.get(url, timeout=30)
        if resp.status_code == 429 and attempt < MAX_RETRIES:
            retry_after = min(int(resp.headers.get("Retry-After", backoff)), MAX_RETRY_WAIT_S)
            print(f"    rate-limited, retrying in {retry_after}s "
                  f"(attempt {attempt}/{MAX_RETRIES})...")
            time.sleep(retry_after)
            backoff *= 2
            continue
        resp.raise_for_status()
        return resp.json()
    return {"results": []}


def fetch_works(filter_clauses, sort="cited_by_count:desc", per_page=25):
    params = {
        "filter": ",".join(filter_clauses),
        "sort": sort,
        "per-page": str(per_page),
        "mailto": MAILTO,
    }
    if API_KEY:
        params["api_key"] = API_KEY
    url = f"{API_BASE}?{urllib.parse.urlencode(params)}"
    return http_get_with_retry(url).get("results", [])


def run_query(query):
    title_only = is_title_only(query)
    search_field = "title.search" if title_only else "title_and_abstract.search"
    per_page = PER_PAGE_DIRECT if title_only else PER_PAGE_FRAMING
    return fetch_works([
        f"{search_field}:{query}",
        f"publication_year:{FROM_YEAR}-{TO_YEAR}",
    ], per_page=per_page)


def field_of(work):
    topic = work.get("primary_topic") or {}
    return (topic.get("field") or {}).get("display_name")


def discipline_of(work):
    """Finer-grained bucket for stratification: OpenAlex's primary_topic.
    field (26 top-level categories) is too coarse to distinguish e.g.
    sociology from anthropology from history — that granularity lives one
    level down, at primary_topic.subfield."""
    topic = work.get("primary_topic") or {}
    subfield = (topic.get("subfield") or {}).get("display_name")
    return subfield or "Unclassified"


def topic_of(work):
    """The most granular OpenAlex classification level (one step finer than
    subfield) — used for TOPIC_EXCLUDE and the topic-history rule, since it
    can distinguish e.g. "Biblical Studies and Interpretation" from other
    Religious studies topics, or catch a historically-framed paper whose
    *subfield* isn't History at all (see module docstring step 3)."""
    topic = work.get("primary_topic") or {}
    return topic.get("display_name") or ""


def add_to_pool(pool, work, channel):
    wid = work.get("id") or work.get("doi")
    if wid not in pool:
        pool[wid] = {"work": work, "channels": set()}
    pool[wid]["channels"].add(channel)


def _merge_entry(target, source):
    """Merge source into target in place, keeping target's work record if
    it's more cited, and always unioning channels."""
    if source["work"].get("cited_by_count", 0) > target["work"].get("cited_by_count", 0):
        source["channels"] |= target["channels"]
        target["work"] = source["work"]
        target["channels"] = source["channels"]
    else:
        target["channels"] |= source["channels"]


def dedupe_by_doi(pool):
    """OpenAlex IDs are already unique per record, but two records can share
    a DOI (e.g. a book and its OpenAlex-derived chapter stub) — collapse
    those, keeping the one with the higher citation count and merging
    channels."""
    by_doi = {}
    deduped = {}
    for wid, entry in pool.items():
        doi = entry["work"].get("doi")
        key = doi if doi else wid
        if key in by_doi:
            _merge_entry(deduped[by_doi[key]], entry)
        else:
            by_doi[key] = wid
            deduped[wid] = entry
    return deduped


_TITLE_NORM_RE = re.compile(r"[^a-z0-9 ]")


def _normalize_title(title):
    return _TITLE_NORM_RE.sub("", (title or "").lower()).strip()


def _first_author_surname(work):
    authorships = work.get("authorships") or []
    if not authorships:
        return ""
    name = (authorships[0].get("author") or {}).get("display_name", "")
    return name.split()[-1].lower() if name else ""


def dedupe_by_title(deduped):
    """Second dedup pass: catches near-duplicates that don't share a DOI —
    reprints/translations indexed as separate records, or the same work
    given several DOIs (e.g. versioned Zenodo preprints). Merges records
    sharing a normalized title AND first-author surname."""
    by_title_author = {}
    result = {}
    for wid, entry in deduped.items():
        title = entry["work"].get("title") or entry["work"].get("display_name") or ""
        key = (_normalize_title(title), _first_author_surname(entry["work"]))
        if key[0] and key in by_title_author:
            _merge_entry(result[by_title_author[key]], entry)
        else:
            by_title_author[key] = wid
            result[wid] = entry
    return result


def screen(deduped):
    included, excluded = [], []
    counts = defaultdict(int)
    for entry in deduped.values():
        work = entry["work"]
        wtype = work.get("type") or "unknown"
        if wtype not in TYPE_INCLUDE:
            excluded.append((entry, f"type exclusion: OpenAlex type={wtype!r}"))
            counts["type"] += 1
            continue
        field = field_of(work)
        if field not in FIELD_ALLOWLIST:
            excluded.append((entry, f"field exclusion: primary_topic.field={field!r}"))
            counts["field"] += 1
            continue
        subfield = discipline_of(work)
        if subfield in SUBFIELD_EXCLUDE:
            excluded.append((entry, f"subfield exclusion: primary_topic.subfield={subfield!r}"))
            counts["subfield"] += 1
            continue
        topic = topic_of(work)
        if topic in TOPIC_EXCLUDE:
            excluded.append((entry, f"topic exclusion: primary_topic={topic!r}"))
            counts["topic"] += 1
            continue
        if any(m in topic.lower() for m in _TOPIC_HISTORICAL_MARKERS):
            excluded.append((entry, f"topic-history exclusion: primary_topic={topic!r}"))
            counts["topic_history"] += 1
            continue
        raw_title = work.get("title") or work.get("display_name") or ""
        title = raw_title.lower()
        matched_pattern = next((p for p in TITLE_EXCLUDE_PATTERNS if p in title), None)
        if matched_pattern:
            excluded.append((entry, f"title exclusion: contains {matched_pattern!r}"))
            counts["title"] += 1
            continue
        if _TAXONOMIC_SECT_RE.search(title):
            excluded.append((entry, "taxonomic-abbreviation exclusion: 'sect.' "
                                     "(botanical/zoological section, not the word 'sect')"))
            counts["taxonomic"] += 1
            continue
        if _ACRONYM_SECTS_RE.search(raw_title):
            excluded.append((entry, "acronym exclusion: all-caps 'SECT(S)' "
                                     "(reliably an acronym, not the word 'sect')"))
            counts["acronym"] += 1
            continue
        ancient_match = _ANCIENT_MARKER_RE.search(title)
        if ancient_match:
            excluded.append((entry, f"ancient-marker exclusion: title contains "
                                     f"{ancient_match.group(1)!r}"))
            counts["ancient"] += 1
            continue
        included.append(entry)
    return included, excluded, counts


def score_and_categorize(included):
    categories = {"book": [], "article": []}
    for entry in included:
        cat = "book" if entry["work"]["type"] in BOOK_TYPES else "article"
        categories[cat].append(entry)

    for entries in categories.values():
        counts = [e["work"].get("cited_by_count", 0) for e in entries]
        lo, hi = (min(counts), max(counts)) if counts else (0, 0)
        for e in entries:
            c = e["work"].get("cited_by_count", 0)
            norm = (c - lo) / (hi - lo) if hi > lo else 0.0
            cross_channel_bonus = min(len(e["channels"]) - 1, 5) * 0.05
            e["score"] = norm + cross_channel_bonus
    return categories


def _has_evidence(entry):
    return entry["work"].get("cited_by_count", 0) > 0 or len(entry["channels"]) > 1


def stratify(categories):
    """Bucket each category by discipline (primary_topic.subfield) and keep
    the top TOP_N_PER_DISCIPLINE per bucket, ranked by score. Entries with
    no corroborating evidence (0 citations and a single retrieval channel)
    are dropped first, unless that would empty the bucket entirely — a thin
    bucket is more informative than one silently hidden."""
    stratified = {"book": defaultdict(list), "article": defaultdict(list)}
    for cat, entries in categories.items():
        for e in entries:
            stratified[cat][discipline_of(e["work"])].append(e)
    for cat in stratified:
        for disc in stratified[cat]:
            bucket = stratified[cat][disc]
            bucket.sort(key=lambda e: e["score"], reverse=True)
            validated = [e for e in bucket if _has_evidence(e)]
            stratified[cat][disc] = (validated or bucket)[:TOP_N_PER_DISCIPLINE]
    return stratified


def fmt_work(entry, rank=None):
    work = entry["work"]
    authors = ", ".join(
        (a.get("author") or {}).get("display_name", "?")
        for a in (work.get("authorships") or [])[:4]
    )
    if len(work.get("authorships") or []) > 4:
        authors += " et al."
    year = work.get("publication_year", "n.d.")
    title = work.get("title") or work.get("display_name") or "(untitled)"
    cited = work.get("cited_by_count", 0)
    wtype = work.get("type", "unknown")
    doi = work.get("doi") or ""
    oaid = work.get("id") or ""
    n_channels = len(entry["channels"])
    score = entry.get("score")
    prefix = f"{rank}. " if rank is not None else "- "
    lines = [f"{prefix}{title} ({year}) — {authors or 'Unknown author'}"]
    lines.append(f"     type={wtype}  cited_by={cited}  channels={n_channels}"
                  + (f"  score={score:.3f}" if score is not None else ""))
    lines.append(f"     channels: {', '.join(sorted(entry['channels']))}")
    if doi:
        lines.append(f"     doi: {doi}")
    lines.append(f"     openalex: {oaid}")
    return "\n".join(lines)


def write_report(queries, query_hit_counts, deduped, included, excluded, stratified):
    lines = [
        "OpenAlex corpus construction — keyword-query results for review",
        "=" * 70,
        f"Publication window: {FROM_YEAR}-{TO_YEAR}",
        f"Queries run: {len(queries)} (see thesis/corpus/literature/keyword-queries.txt)",
        f"Raw pool (union of query hits): {sum(query_hit_counts.values())} "
        f"hits across queries -> {len(deduped)} unique records",
        f"Included after screening: {len(included)}",
    ]
    lines.append("")
    lines.append("NOTE: this is a candidate list awaiting manual approval — no")
    lines.append("record here has been added to the approved literature corpus yet.")
    lines.append("")

    lines.append("-" * 70)
    lines.append("Query hit counts")
    lines.append("-" * 70)
    for q in queries:
        lines.append(f"  {q!r}: {query_hit_counts.get(q, 'FAILED')}")
    lines.append("")

    for cat, label in (("book", "BOOKS"), ("article", "ARTICLES")):
        lines.append("=" * 70)
        lines.append(f"{label} — stratified by discipline, top {TOP_N_PER_DISCIPLINE} each")
        lines.append("=" * 70)
        disciplines = sorted(stratified[cat].keys())
        if not disciplines:
            lines.append("  (none)")
        for disc in disciplines:
            entries = stratified[cat][disc]
            lines.append("")
            lines.append(f"-- {disc} ({len(entries)}) --")
            for rank, e in enumerate(entries, 1):
                lines.append(fmt_work(e, rank))
        lines.append("")

    lines.append("=" * 70)
    lines.append(f"EXCLUDED RECORDS ({len(excluded)}) — screening log")
    lines.append("=" * 70)
    for entry, reason in excluded:
        lines.append(f"[{reason}]")
        lines.append(fmt_work(entry))
        lines.append("")

    OUTPUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
# --approve: export the stratified candidates to a .bib file
# =============================================================================

BIB_TYPE_FOR_OPENALEX = {"book": "book", "book-chapter": "incollection", "article": "article"}
_BIB_KEY_RE = re.compile(r"@\w+\{([^,\s]+)\s*,")


# Greek/Cyrillic letters that are visually indistinguishable from a Latin
# letter, seen in practice as OpenAlex metadata glitches rather than
# intentional non-Latin script (e.g. author name "Jan Ν. Bremmer" —
# a Greek capital Nu standing in for a Latin "N" in a Dutch scholar's
# name) — normalized here rather than typeset, since preserving them as
# "Greek text" would misrepresent a plain data-entry error.
_HOMOGLYPHS = str.maketrans({
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H",
    "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O",
    "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
})


def _bib_escape(value):
    # NFC-normalize first: OpenAlex sometimes returns accented letters as a
    # base letter plus a separate combining diacritic (NFD) rather than the
    # single precomposed character (NFC) — e.g. "e" + U+0302 instead of
    # "ê" — which pdfLaTeX's inputenc cannot handle as a standalone glyph.
    normalized = unicodedata.normalize("NFC", str(value))
    return normalized.translate(_HOMOGLYPHS).replace("{", "").replace("}", "")


def _slugify_title(title, max_words=4):
    words = re.findall(r"[a-zA-Z0-9]+", (title or "").lower())[:max_words]
    return "-".join(words) or "untitled"


def existing_bib_keys():
    """Scan every .bib already in thesis/corpus/literature/raw/ (except the one
    this script writes) for citation keys already in use, so newly
    generated IDs don't collide with them."""
    keys = set()
    if not LITERATURE_RAW_DIR.exists():
        return keys
    for path in LITERATURE_RAW_DIR.glob("*.bib"):
        if path == APPROVED_BIB_PATH:
            continue
        for m in _BIB_KEY_RE.finditer(path.read_text(encoding="utf-8")):
            keys.add(m.group(1))
    return keys


def make_lit_id(work, used_ids):
    """lit-<firstauthor><year>-<title-slug>, matching the corpus-wide ID
    scheme (corpus/README.md) — disambiguated with a letter suffix on
    collision (e.g. two same-author-same-year works)."""
    surname = re.sub(r"[^a-z]", "", _first_author_surname(work) or "unknown")
    year = work.get("publication_year") or "nd"
    slug = _slugify_title(work.get("title") or work.get("display_name"))
    base = f"lit-{surname}{year}-{slug}"
    candidate = base
    suffix = ord("a")
    while candidate in used_ids:
        candidate = f"{base}-{chr(suffix)}"
        suffix += 1
    used_ids.add(candidate)
    return candidate


def reconstruct_abstract(work):
    """OpenAlex gives abstracts as an inverted index (word -> [positions])
    rather than plain text, to sidestep some publishers' redistribution
    terms; reassemble it back into a plain-text string."""
    inv = work.get("abstract_inverted_index")
    if not inv:
        return ""
    positions = {}
    for word, idxs in inv.items():
        for i in idxs:
            positions[i] = word
    return " ".join(positions[i] for i in sorted(positions))


def work_to_bibtex(work, lit_id):
    btype = BIB_TYPE_FOR_OPENALEX.get(work.get("type"), "article")
    lines = [f"@{btype}{{{lit_id},"]
    title = work.get("title") or work.get("display_name") or ""
    lines.append(f"  title = {{{_bib_escape(title)}}},")
    authors = [
        (a.get("author") or {}).get("display_name", "")
        for a in (work.get("authorships") or [])
    ]
    authors = [a for a in authors if a]
    if authors:
        lines.append(f"  author = {{{_bib_escape(' and '.join(authors))}}},")
    year = work.get("publication_year")
    if year:
        lines.append(f"  date = {{{year}}},")
    source_name = ((work.get("primary_location") or {}).get("source") or {}).get("display_name")
    if source_name:
        lines.append(f"  publisher = {{{_bib_escape(source_name)}}},")
    doi = work.get("doi") or ""
    if doi:
        lines.append(f"  doi = {{{doi.replace('https://doi.org/', '')}}},")
    abstract = reconstruct_abstract(work)
    if abstract:
        lines.append(f"  abstract = {{{_bib_escape(abstract)}}},")
    cited = work.get("cited_by_count", 0)
    lines.append(f"  note = {{Retrieved via the OpenAlex corpus construction protocol "
                 f"(thesis Methods, Corpus construction protocol); {cited} citations "
                 f"at retrieval time.}},")
    lines.append("}")
    return "\n".join(lines)


def write_approved_bib(stratified):
    """Write every stratified candidate (the exact set shown in the report's
    BOOKS/ARTICLES sections) to thesis/corpus/literature/raw/OpenAlex_corpus.bib,
    for build_corpus.py to fold into the approved literature corpus on its
    next run. Overwrites any previous export from this script."""
    used_ids = existing_bib_keys()
    entries = []
    for cat in ("book", "article"):
        for disc in sorted(stratified[cat].keys()):
            for e in stratified[cat][disc]:
                lit_id = make_lit_id(e["work"], used_ids)
                entries.append(work_to_bibtex(e["work"], lit_id))
    LITERATURE_RAW_DIR.mkdir(parents=True, exist_ok=True)
    APPROVED_BIB_PATH.write_text("\n\n".join(entries) + "\n", encoding="utf-8")
    return len(entries)


def main():
    warn_if_no_api_key()
    if not KEYWORD_QUERIES_TXT.exists():
        sys.exit(f"Missing {KEYWORD_QUERIES_TXT}")
    queries = load_queries()
    print(f"Running {len(queries)} keyword queries against OpenAlex "
          f"(publication_year:{FROM_YEAR}-{TO_YEAR})...")

    pool = {}
    query_hit_counts = {}
    for i, query in enumerate(queries, 1):
        try:
            results = run_query(query)
        except requests.RequestException as e:
            print(f"  [{i}/{len(queries)}] {query!r}: FAILED ({e})")
            continue
        query_hit_counts[query] = len(results)
        print(f"  [{i}/{len(queries)}] {query!r}: {len(results)} hits")
        for work in results:
            add_to_pool(pool, work, query)
        time.sleep(REQUEST_PAUSE_S)

    print(f"\nMerged pool: {len(pool)} unique records "
          f"(before deduplication).")

    deduped = dedupe_by_title(dedupe_by_doi(pool))
    print(f"After deduplication (ID/DOI + normalized title/author): "
          f"{len(deduped)} unique records.")

    included, excluded, counts = screen(deduped)
    print(f"Screening: {len(included)} included, "
          f"{counts['type']} excluded by type, {counts['field']} excluded "
          f"by field, {counts['subfield']} excluded by subfield, "
          f"{counts['topic']} excluded by topic, "
          f"{counts['topic_history']} excluded by topic-history, "
          f"{counts['title']} excluded by title pattern, "
          f"{counts['taxonomic']} excluded by taxonomic abbreviation, "
          f"{counts['acronym']} excluded by acronym, "
          f"{counts['ancient']} excluded by ancient marker.")

    categories = score_and_categorize(included)
    stratified = stratify(categories)

    write_report(queries, query_hit_counts, deduped, included, excluded, stratified)
    print(f"\nReport written to {OUTPUT_TXT.relative_to(CORPUS_DIR.parent)}")

    if "--approve" in sys.argv:
        n = write_approved_bib(stratified)
        print(f"Approved: wrote {n} candidates to "
              f"{APPROVED_BIB_PATH.relative_to(CORPUS_DIR.parent)}")
        print("Run thesis/corpus/scripts/build_corpus.py to fold them into "
              "literature.json/.csv and the reference_list.tex appendix.")
    else:
        print("Review it, then re-run with --approve before folding any "
              "entries into the approved literature corpus.")


if __name__ == "__main__":
    main()
