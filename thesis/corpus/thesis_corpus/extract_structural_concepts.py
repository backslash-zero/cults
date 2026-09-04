"""Extract candidate structural/generic vocabulary from the corpora' own
expression text, for the "structural concepts" reference point-set (see
build_shared_space.py, kept alongside the original WordNet concept_backbone
as a second, corpus-informed reference subset -- not a replacement).

Why not mine entity_anchors (like emergent_entities does): checked first,
and entity_anchors turns out to essentially never tag abstract
social-structure vocabulary. Even summed across the whole corpus with no
threshold at all, "control" appears 4 times, "authority" 12, "obedience" 3,
"manipulation" 3, "harm" 0 as tagged anchors -- entity_anchors captures
concrete named things ("Scientology", "charismatic leader"), not the kind
of generic vocabulary this point-set needs, at any scale.

Instead, this script tokenizes the actual expression prose (`embedding_text`)
across all three corpora and keeps only tokens that are valid Open English
WordNet (oewn:2024) lemmas -- the same lexicon build_concept_backbone_en.py
already uses, so no new dependency. This does two jobs at once:
  - Discards proper nouns, typos, and non-English tokens (MIVILUDES is
    mostly French, interviews partly so) essentially for free, since none
    of those are WordNet entries.
  - Supplies a ready-made English gloss for every surviving candidate, per
    the same "<term>: <gloss>" convention as the WordNet backbone.

A word is counted once per *expression* that contains it (not once per raw
occurrence), so one repetitive sentence can't inflate a word's count the
way raw token counting would.

Output: dictionaries/structural_concepts_candidates.csv
  concept_id, concept_en, gloss_en, total_mentions, literature_mentions,
  miviludes_mentions, interviews_mentions, n_corpora, is_generic

`is_generic` starts as an automatic guess (see AUTO_EXCLUDE below) but this
is meant to be reviewed by hand afterward -- this script's own candidate
ranking already does the heavy lifting (WordNet membership excludes proper
nouns structurally), so the remaining judgment call is narrower: pruning
overly generic/meaningless function-adjacent words ("thing", "way", "said")
that are real WordNet entries but useless as interpretive landmarks.

Usage (from thesis/corpus/):
    python -m thesis_corpus.extract_structural_concepts
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import wn

CORPUS_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = CORPUS_DIR / "processed"
OUTPUT_PATH = CORPUS_DIR / "dictionaries" / "structural_concepts_candidates.csv"

ARCHIVES = {
    "literature": PROCESSED_DIR / "literature" / "criterion_expressions.jsonl",
    "miviludes": PROCESSED_DIR / "miviludes" / "criterion_expressions.jsonl",
    "interviews": PROCESSED_DIR / "interviews" / "criterion_expressions.jsonl",
}

WORDNET_LEXICON = "oewn:2024"
MIN_TOKEN_LEN = 3
CANDIDATE_POOL_SIZE = 10000  # ranked pool inspected for WordNet membership before domain filtering
TARGET_SIZE = 1500  # how many domain-filtered candidates to keep -- the practical ceiling before
                     # quality degrades further (proper-noun leakage picks up past ~2000, each
                     # term backed by fewer real mentions); checked directly against the full
                     # 7,053-candidate pool before settling here, not picked blind

# Restricts candidates to lexicographer files plausibly about social
# relations, belief, psychology, or group dynamics -- matching the intent
# behind the example vocabulary (control, authority, belief, manipulation,
# isolation, loyalty, obedience, ...), and excluding concrete/physical
# domains (places, objects, body parts, substances, dates, quantities) that
# pass plain WordNet-membership but aren't "structural concepts" in the
# relevant sense. Verbs are excluded entirely: the target vocabulary is
# nouns/adjectives naming a concept, attribute, or role, not an action.
INCLUDE_LEXFILES = {
    "adj.all", "adj.pert",
    "noun.person", "noun.cognition", "noun.act", "noun.communication",
    "noun.group", "noun.attribute", "noun.state", "noun.relation",
    "noun.possession", "noun.motive", "noun.feeling", "noun.phenomenon",
    "noun.process", "noun.Tops",
}

# A handful of very-high-frequency words whose top WordNet sense (by our
# naive "first synset for pos n/a/v" pick) is clearly the wrong one for how
# the corpora actually use them -- e.g. "religious" picks the noun "monk"
# sense, not the adjective. Overridden by hand since getting these right
# matters most (they're the most visible entries in the final set).
GLOSS_OVERRIDES = {
    "religious": "concerned with sacred matters or religion or the church",
    "movement": "a group of people with a common ideology who try together to achieve certain general goals",
    "spiritual": "concerned with sacred matters or religion or the sacred",
    "state": "the group of people comprising the government of a sovereign state",
    "order": "a group of people who are joined by profession or interest or principle",
    "acts": "something that people do or cause to happen",
    "dominant": "exercising influence or control",
    "hostile": "not belonging to your own country's forces or those of an ally",
    "close": "in intimate association or accord",
    "turn": "a change in direction",
    "group": "any number of entities (members) considered as a unit",
    "god": "the supernatural being conceived as the perfect and omnipotent and omniscient originator and ruler of the universe",
    "lay": "characteristic of those who are not members of the clergy",
    # Found extending the target size to 1500: a real WordNet sense exists
    # and is a genuinely useful concept, but the top-ranked sense picked was
    # a specific US-federal-department/institution instance instead of the
    # general one.
    "education": "the activities of educating or instructing; activities that impart knowledge or skill",
    "justice": "the quality of being just or fair",
    "energy": "enterprising or ambitious drive; forceful exertion",
    "defense": "the act of defending against harm",
    "key": "a solution or remedy for a difficult problem",
    "nation": "a politically organized body of people under a single government",
    "post": "a job in an organization",
    "service": "a public act of religious worship",
    "balance": "a state of equilibrium",
    "private": "confined to particular persons or groups",
}

# Spot-checked and excluded: a WordNet sense exists and cleared the domain
# filter, but it's either a proper-noun leak the instance_hyponym check
# missed, an acronym coinciding with a common word (e.g. "LET" = Lashkar-e-
# Taiba, "SHAPE" = NATO's Supreme Headquarters), an overly narrow/wrong
# sense not fixable with a one-line override, or a nationality/demonym
# that isn't a structural concept in the relevant sense. Two specific-named
# leaks are worth flagging explicitly: "hubbard" (L. Ron Hubbard) and
# "iskcon" (a specific religious sect, the Hare Krishnas) are exactly the
# kind of proper-noun/named-group leak this whole domain filter exists to
# keep out, even though they're topically on-point -- structural concepts
# should describe roles/dynamics, not name specific people or groups (that
# job belongs to emergent_entities). Not exhaustive -- a pool this size
# will have residual noise beyond what a spot-check catches; documented
# here rather than claimed fully clean.
EXCLUDE_WORDS = {
    "truth",  # matches "Sojourner Truth" (proper-noun leak)
    "back",  # matches a football position, not the general sense
    "anti",  # standalone token artifact of hyphenated "anti-cult" etc.
    "well",  # matches "an abundant source", an unrelated noun sense
    "times",  # matches an arithmetic operation, not plural of "time"
    "japanese",  # nationality/demonym, not a structural concept
    "roman",  # nationality/demonym, not a structural concept
    "writings",  # matches a specific Hebrew Scriptures sense
    # Found extending the target size from 600 to 1500 (systematic sweep
    # for biographical-looking glosses, plus manual spot-checks):
    "broad", "asian", "english", "prime", "peter", "elijah", "ted",
    "american", "western", "young", "major", "black", "day", "eastern",
    "best", "france", "born", "hubbard", "jones", "iskcon", "john",
    "masters", "land", "base", "hope", "twenty", "begin", "singer", "low",
    "joseph", "james", "drew", "richardson", "smith", "berg", "eight",
    "charles", "london", "usa", "manson", "bond", "king", "southern",
    "thornton", "muhammad", "weber", "sessions", "army", "henry", "lewis",
    "robert", "robbins", "let", "ron", "shape", "pas", "italian",
    "russian", "simon", "palmer",
}

_TOKEN_RE = re.compile(r"[a-zA-Z']+")

# Standard English function words -- WordNet contains entries for several of
# these ("be", "have", "can" are real verb senses), so stopword exclusion is
# needed on top of WordNet membership, not instead of it.
STOPWORDS = {
    "the", "a", "an", "of", "in", "to", "and", "or", "is", "was", "were", "be", "been", "being",
    "that", "this", "these", "those", "with", "for", "as", "by", "at", "from", "it", "its", "it's",
    "are", "am", "has", "have", "had", "not", "no", "but", "can", "cannot", "could", "will", "would",
    "should", "shall", "may", "might", "must", "which", "who", "whom", "whose", "what", "when",
    "where", "how", "why", "also", "more", "most", "other", "others", "some", "such", "than", "then",
    "there", "their", "theirs", "they", "them", "he", "she", "his", "her", "hers", "him", "we", "our",
    "ours", "us", "you", "your", "yours", "i", "my", "mine", "me", "if", "do", "does", "did", "done",
    "so", "just", "very", "too", "only", "own", "same", "each", "any", "all", "both", "few", "many",
    "much", "into", "onto", "upon", "about", "over", "under", "again", "further", "once", "here",
    "up", "down", "out", "on", "off", "above", "below", "between", "through", "during", "before",
    "after", "while", "because", "since", "though", "although", "yet", "still", "even", "one", "two",
    "three", "first", "second", "third", "new", "old", "like", "one's", "get", "gets", "got", "make",
    "makes", "made", "way", "ways", "thing", "things", "something", "someone", "anything", "anyone",
    "everything", "everyone", "nothing", "nobody", "lot", "lots", "kind", "kinds", "sort", "sorts",
}


def tokenize(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text) if len(t) >= MIN_TOKEN_LEN}


def domain_synset(en: wn.Wordnet, word: str) -> "wn.Synset | None":
    """First noun/adjective synset (verbs excluded -- see INCLUDE_LEXFILES)
    whose lexicographer file is in-domain, and which isn't a named-entity
    instance (WordNet's instance_hyponym mechanism, same check
    build_concept_backbone_en.py uses for the WordNet backbone) -- filters
    out proper-noun leakage like "York" or "James" that happens to also be
    a common WordNet lemma."""
    for pos in ("n", "a"):
        for synset in en.synsets(word, pos=pos):
            if synset.lexfile() not in INCLUDE_LEXFILES:
                continue
            if synset.get_related("instance_hyponym"):
                continue
            return synset
    return None


def main() -> None:
    mentions_by_corpus: dict[str, Counter[str]] = defaultdict(Counter)

    for corpus_name, path in ARCHIVES.items():
        if not path.exists():
            raise SystemExit(f"Missing corpus archive: {path}")
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                tokens = tokenize(item.get("embedding_text", ""))
                for token in tokens:
                    if token in STOPWORDS:
                        continue
                    mentions_by_corpus[corpus_name][token] += 1

    total_mentions: Counter[str] = Counter()
    for corpus_counts in mentions_by_corpus.values():
        total_mentions.update(corpus_counts)

    print(f"Tokenized expression text: {len(total_mentions)} unique (post-stopword) tokens across all corpora.")

    en = wn.Wordnet(WORDNET_LEXICON)

    candidates = []
    for word, count in total_mentions.most_common(CANDIDATE_POOL_SIZE):
        if word in EXCLUDE_WORDS:
            continue
        synset = domain_synset(en, word)
        if synset is None:
            continue
        gloss = GLOSS_OVERRIDES.get(word, synset.definition())
        n_corpora = sum(1 for c in mentions_by_corpus.values() if c.get(word, 0) > 0)
        candidates.append({
            "concept_en": word,
            "gloss_en": gloss,
            "total_mentions": count,
            "literature_mentions": mentions_by_corpus["literature"].get(word, 0),
            "miviludes_mentions": mentions_by_corpus["miviludes"].get(word, 0),
            "interviews_mentions": mentions_by_corpus["interviews"].get(word, 0),
            "n_corpora": n_corpora,
        })
        if len(candidates) >= TARGET_SIZE:
            break

    print(f"{len(candidates)} candidates kept (in-domain WordNet lexfile, ranked by expression-frequency).")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "concept_id", "concept_en", "gloss_en", "total_mentions",
        "literature_mentions", "miviludes_mentions", "interviews_mentions",
        "n_corpora", "is_generic",
    ]
    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, c in enumerate(candidates, 1):
            writer.writerow({
                "concept_id": f"sc_{i:04d}",
                "concept_en": c["concept_en"],
                "gloss_en": c["gloss_en"],
                "total_mentions": c["total_mentions"],
                "literature_mentions": c["literature_mentions"],
                "miviludes_mentions": c["miviludes_mentions"],
                "interviews_mentions": c["interviews_mentions"],
                "n_corpora": c["n_corpora"],
                "is_generic": "true",  # domain-filtered already; flip individual outliers to false on manual review
            })

    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
