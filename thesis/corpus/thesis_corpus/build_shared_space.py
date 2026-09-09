"""Build one shared, cross-corpus embedding space (PCA on a pooled matrix).

Every prior embedding step (extract_and_embed's three corpora,
embed_miviludes_criteria, embed_concept_backbone) produces vectors in the
same raw 1024-d bge-m3 space, but reduce_embeddings.py's downsampling fits
an INDEPENDENT PCA per corpus -- literature's own 100-d space, MIVILUDES's
own, interviews' own. Those are unrelated coordinate systems: a point's
position in one is not comparable to a point's position in another. This
script instead pools every vector from every dataset, standardizes, and
fits ONE PCA on the pooled matrix -- so every item, from every dataset,
ends up in the same shared coordinate system, comparable to every other.

This is what operationalizes the concept backbone's stated purpose (see
Methods.tex, "A Generic Concept Backbone"): it is not a static reference
list sitting outside the analysis, but an active participant in fitting
this shared space, alongside every corpus item and the MIVILUDES criteria.

Every point additionally carries a `point_role`, distinguishing three kinds
of thing this space pools together (see Methods.tex, "A Shared Cross-Corpus
Space"):
  - `"expression"`: a criterion expression extracted from a text (the three
    corpora) or a MIVILUDES criterion -- something a source actually said.
  - `"reference"`: a backdrop vocabulary point, not itself a claim any
    source makes. Three distinct subsets share this role, deliberately kept
    separate rather than merged into one, since they buy different things:
    `concept_backbone` (WordNet, topic-neutral -- not derived from any
    corpus, a fixed independent yardstick), `structural_concepts`
    (corpus-derived generic/structural vocabulary -- extracted from the
    corpora's own expression text, geometrically closer to the data by
    construction, but not topic-neutral: it exists *because* the corpora
    use this vocabulary), and `conceptnet_concepts` (see
    extract_conceptnet_concepts.py -- ConceptNet's associative neighborhood
    of the `structural_concepts` seed words, filtered to in-domain WordNet
    senses and then hand-pruned of hub words; a smoother generalization of
    `structural_concepts`, seeded from the corpus but not restricted to
    exactly what it says). Use `concept_backbone` when independence from the
    corpus matters; use `structural_concepts`/`conceptnet_concepts` when
    proximity/interpretive relevance matters more than neutrality.
  - `"emergent"`: a named entity/group/concept mentioned BY the corpora
    themselves (emergent entities, below) -- corpus-derived like an
    expression point, but a recurring reference object rather than a claim
    about one.

Pooling (eight source_dataset values; total point count is logged at
runtime, not asserted against a hardcoded constant -- it changes whenever
the corpus grows or the emergent-entity threshold below is adjusted):
  - Each corpus item (literature/miviludes/interviews) contributes ONE
    point (`point_role="expression"`): its embedding_vector, plus its
    claim_mode/epistemic_status/attribution tags carried through unchanged
    for later faceting -- except exact-duplicate expressions within the
    same document (keeping the first occurrence) and expressions under
    MIN_EXPRESSION_WORDS words, both filtered here at pooling time rather
    than in the archive (see load_corpus_points). MIVILUDES items use their
    English translation (translate_miviludes_expressions.py) as the
    embedded vector/label instead of the French original, closing a
    measured language-asymmetry gap against the English-only reference
    vocabularies; French moves to a `label_fr` field. Interview items
    additionally carry response_rank --
    a 1-indexed extraction-order position, *not* a free-listing rank or a
    cognitive-salience proxy (the interview protocol elicits a single
    first-association example plus justification/probes, not a ranked
    list; see thesis_corpus.audit_free_listing_rank and
    thesis_corpus.analyze_initial_exemplars for the actual, manually
    reviewed interview-side geometric analysis).
  - Each MIVILUDES criterion contributes ONE point (`point_role="expression"`):
    its French embedding (the official original). The English translation is
    kept only as a display label (`label_en`) on the same point, not
    embedded separately -- translation fidelity is instead checked directly,
    once, via raw-space cosine similarity between the French and English
    embeddings (see `check_miviludes_translation_fidelity`), rather than by
    spending a second near-duplicate point in the shared analytical space.
  - Each concept-backbone entry contributes ONE point
    (`point_role="reference"`, `source_dataset="concept_backbone"`): its
    embedding_vector.
  - Each structural-concept entry (see extract_structural_concepts.py)
    contributes ONE point (`point_role="reference"`,
    `source_dataset="structural_concepts"`): its embedding_vector, plus
    `mention_distribution` (per-corpus mention counts, same provenance
    role as on emergent-entity points) reconstructed from the CSV's
    mention-count columns.
  - Each ConceptNet-derived concept kept on manual review (see
    extract_conceptnet_concepts.py: `is_generic == "true"` is the default,
    hand-flipped to `"false"` per row for generic hub words that survived
    the automated ranking -- so kept rows are the ones still `"true"` after
    review) contributes ONE point (`point_role="reference"`,
    `source_dataset="conceptnet_concepts"`): its embedding_vector. Only the
    hand-reviewed, kept subset is ever embedded/pooled (see
    filter_conceptnet_concepts.py) -- the 1,150 rows flagged as generic hub
    words during manual review never leave the candidates CSV.
  - Each emergent entity mentioned at least `--entity-anchor-min-mentions`
    times across all three corpora contributes ONE point
    (`point_role="emergent"`): a per-unique (normalized) entity-anchor
    embedding already computed in Stage 2 (`entity_anchor_vectors`), never
    before pooled into any space. Gives named entities/dimensions (e.g.
    "Scientology", "charismatic leader") an actual position relative to
    corpus expressions and the concept backbone. Carries
    `mention_distribution` (per-corpus mention counts) as provenance
    metadata, not used in the PCA fit.

Preprocessing: StandardScaler (zero mean, unit variance per dimension)
before PCA. Every vector already comes from the same embedding model, but
the sources differ a lot in register (academic prose, government French,
casual interview speech, bare word+gloss dictionary entries) and could
plausibly carry different per-dimension distributions -- this is a
defensive measure against any one dataset dominating the fit purely due to
scale, not a claim that such an imbalance is known to exist.

Dimensionality is not a fixed constant: PCA is first fit at full rank to
get the complete explained-variance curve (saved as a diagnostic, not just
asserted), and the smallest k reaching 95% cumulative variance is chosen
from that curve.

This needs no Ollama -- only numpy/scikit-learn/matplotlib on data that's
already local. Never modifies any of the source files; only ever writes new
files under processed/.

Usage (from thesis/corpus/):
    python -m thesis_corpus.build_shared_space
    python -m thesis_corpus.build_shared_space --entity-anchor-min-mentions 5

    # v2 (a thesis_corpus.extract_v2 + embed_v2 run for all three corpora):
    # writes to processed/shared_space_v2/, never touches v1's own archives
    # or processed/shared_space/. --min-expression-words 0 because v2 already
    # screens short fragments at extraction time with a referent test a bare
    # word-count floor can't express (see load_corpus_points_v2's docstring).
    python -m thesis_corpus.build_shared_space --run-tag 20260910 --min-expression-words 0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

CORPUS_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = CORPUS_DIR / "processed"
# All cross-corpus (not per-corpus) outputs live under their own
# processed/shared_space/ subdirectory, as a sibling of processed/<corpus>/
# -- keeps "one corpus's own pipeline output" and "output that spans every
# corpus" visually and structurally distinct, and gives any later step
# (e.g. a further 2D/3D reduction for visualization) an obvious home next
# to the space it would be reducing.
SHARED_SPACE_DIR = PROCESSED_DIR / "shared_space"

CORPUS_ARCHIVES = {
    "literature": PROCESSED_DIR / "literature" / "criterion_expressions.jsonl",
    "miviludes": PROCESSED_DIR / "miviludes" / "criterion_expressions.jsonl",
    "interviews": PROCESSED_DIR / "interviews" / "criterion_expressions.jsonl",
}
MIVILUDES_CRITERIA_PATH = CORPUS_DIR / "metadata" / "miviludes_criteria_embedded.jsonl"
CONCEPT_BACKBONE_PATH = CORPUS_DIR / "dictionaries" / "concept_backbone_embedded.jsonl"
# Produced by extract_structural_concepts.py + embedding on the Windows/Ollama
# machine (see thesis_corpus/README.md) -- python -m thesis_corpus.embed_concept_backbone
# --input dictionaries/structural_concepts_candidates.csv --output <this path>
# (the existing embed_concept_backbone.py is fully generic over its
# --input/--output CSV, no separate embed script needed).
STRUCTURAL_CONCEPTS_PATH = CORPUS_DIR / "dictionaries" / "structural_concepts_embedded.jsonl"
# Produced by extract_conceptnet_concepts.py, hand-reviewed (is_generic
# flipped to "false" on 1,150 of 1,345 rows -- generic hub words), then
# filter_conceptnet_concepts.py + embed_concept_backbone.py on the
# Windows/Ollama machine (see filter_conceptnet_concepts.py's docstring
# for the exact command). Only the 195 rows kept on review are embedded.
CONCEPTNET_CONCEPTS_PATH = CORPUS_DIR / "dictionaries" / "conceptnet_concepts_embedded.jsonl"
# Produced by translate_miviludes_expressions.py on the Ollama machine (see
# thesis_corpus/README.md) -- translates MIVILUDES's own expressions to
# English so they're embedded on the same footing as the English-only
# reference vocabularies (concept_backbone, structural_concepts). Required,
# not optional: the whole point is to close a measured language-asymmetry
# gap, so silently falling back to French if this ever went missing would
# let that gap reopen unnoticed.
MIVILUDES_EXPRESSION_TRANSLATIONS_PATH = PROCESSED_DIR / "miviludes" / "expression_translations_embedded.jsonl"

OUTPUT_PATH = SHARED_SPACE_DIR / "embedding_space.jsonl"
VARIANCE_CSV_PATH = SHARED_SPACE_DIR / "variance_curve.csv"
VARIANCE_PLOT_PATH = SHARED_SPACE_DIR / "variance_curve.png"
VARIANCE_JSON_PATH = SHARED_SPACE_DIR / "variance_curve.json"
PCA_TRANSFORM_PATH = SHARED_SPACE_DIR / "pca_transform.joblib"
PCA_TRANSFORM_METADATA_PATH = SHARED_SPACE_DIR / "pca_transform_metadata.json"

VARIANCE_THRESHOLD = 0.95
EMBEDDING_DIM = 1024
ENTITY_ANCHOR_MIN_MENTIONS = 3
# Below this, a FR/EN pair is treated as a likely mistranslation worth
# inspecting by hand. NOT 0.90: the MIVILUDES criteria check's own first
# run flagged crit-legal-disputes at 0.50 under a flat <0.90 rule, and
# hand inspection found it was actually an accurate translation -- short
# official phrases just embed less stably cross-lingually than length
# alone would suggest. 0.70 is calibrated to catch genuine mistranslations
# without flagging that kind of expected short-phrase noise.
COSINE_FLAG_THRESHOLD = 0.70
# Expressions shorter than this (naive whitespace word count) are dropped
# when pooling -- see MIN_EXPRESSION_WORDS' use in load_corpus_points.
MIN_EXPRESSION_WORDS = 5

_WHITESPACE_RE = re.compile(r"\s+")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("thesis_corpus.build_shared_space")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit_hash() -> str | None:
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=CORPUS_DIR,
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def normalize_anchor(anchor: str) -> str:
    """Lowercase, strip, and collapse internal whitespace runs to one space
    -- so "Charismatic  Leader" and "charismatic leader" merge to the same
    entity rather than becoming two near-duplicate points."""
    return _WHITESPACE_RE.sub(" ", anchor.strip().lower())


def load_corpus_points(
    corpus_name: str, path: Path, translations: dict[str, dict] | None = None,
) -> tuple[list[dict], dict[str, int]]:
    """Two known-and-deferred extraction issues (documented in Methods.tex,
    "Vectorising Scholarly Work") get filtered here, at pooling time, rather
    than by mutating the archive: the LLM occasionally emits the same
    expression twice within one chunk's response, and occasionally emits a
    short, low-content phrase -- already captured as an entity anchor on a
    fuller expression -- as though it were an independent expression.
    Filtering the pooled space (not the archive) matches this codebase's
    "never modify the source, only write new files" convention -- the
    archive stays the full, traceable record of what the LLM actually
    produced, flaws included. Filtering (duplicate/length checks) always
    operates on the original archive text, regardless of `translations` --
    the notion of "a duplicate" or "a short fragment" is about what was
    actually extracted, not which language ends up embedded.

    `translations` (only ever passed for `corpus_name == "miviludes"`): a
    load_miviludes_translations() lookup, keyed by
    `document_id:chunk_index:occurrence` (see that function for why a bare
    `document_id:chunk_index` isn't unique here). The matching occurrence
    counter is rebuilt below from the archive's own read order -- reset per
    (document_id, chunk_index) pair and incremented for every raw item,
    filtered or not, since the translations file was built from the full,
    unfiltered 914-item archive and its own occurrence numbering has no
    knowledge of which items this function later drops. When given, each
    surviving point's `vector`/`label` become the English translation's
    `embedding_vector_en`/`text_en`, and the original French moves to a new
    `label_fr` field -- closing the language-asymmetry gap against the
    English-only reference vocabularies (concept_backbone,
    structural_concepts). Fails loudly if any pooled MIVILUDES expression
    has no matching translation, or if a matched translation's French text
    doesn't match the archive's (a guard against the occurrence-counter
    reconstruction silently drifting out of sync), rather than silently
    keeping French or pairing the wrong translation for just that one
    point.

    response_rank is computed BEFORE filtering, over every item in original
    archive order, so a dropped item doesn't shift the rank of items after
    it -- "3rd mentioned" should reflect what was actually said 3rd, not a
    renumbering among whatever survives the filter.

    Returns (points, removal_counts) -- removal_counts is
    {"duplicates": N, "short_fragments": M}, for the "X duplicates removed,
    Y short fragments removed" log line and Methods.tex documentation."""
    points = []
    response_rank_by_document: dict[str, int] = defaultdict(int)
    for_interviews = corpus_name == "interviews"
    seen_text_by_document: dict[str, set[str]] = defaultdict(set)
    occurrence_by_chunk: dict[tuple[str, int], int] = defaultdict(int)
    duplicates_removed = 0
    short_fragments_removed = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            document_id = item["document_id"]
            chunk_index = item["chunk_index"]
            text = item["embedding_text"]
            key = f"{document_id}:{chunk_index}"

            translation_key = None
            if translations is not None:
                chunk = (document_id, chunk_index)
                translation_key = f"{document_id}:{chunk_index}:{occurrence_by_chunk[chunk]}"
                occurrence_by_chunk[chunk] += 1

            response_rank = None
            if for_interviews:
                response_rank_by_document[document_id] += 1
                response_rank = response_rank_by_document[document_id]

            if text in seen_text_by_document[document_id]:
                duplicates_removed += 1
                continue
            seen_text_by_document[document_id].add(text)

            if len(text.split()) < MIN_EXPRESSION_WORDS:
                short_fragments_removed += 1
                continue

            label, label_fr, vector = text, None, item["embedding_vector"]
            if translations is not None:
                translation = translations.get(translation_key)
                if translation is None:
                    raise SystemExit(
                        f"Missing translation for MIVILUDES expression {translation_key} -- "
                        f"{MIVILUDES_EXPRESSION_TRANSLATIONS_PATH} exists but doesn't "
                        "cover every pooled expression. Rerun translate_miviludes_expressions.py."
                    )
                if translation["text_fr"] != text:
                    raise SystemExit(
                        f"Translation mismatch for MIVILUDES expression {translation_key}: "
                        f"archive text {text!r} != translation's text_fr "
                        f"{translation['text_fr']!r}. The occurrence-based join has drifted "
                        "out of sync with the archive -- do not trust this pipeline run."
                    )
                label = translation["text_en"]
                label_fr = translation["text_fr"]
                vector = translation["embedding_vector_en"]

            points.append({
                "source_dataset": corpus_name,
                "point_role": "expression",
                "key": key,
                "label": label,
                "label_fr": label_fr,
                "attribution": item.get("attribution"),
                "claim_mode": item.get("claim_mode"),
                "epistemic_status": item.get("epistemic_status"),
                "response_rank": response_rank,
                "vector": vector,
            })
    return points, {"duplicates": duplicates_removed, "short_fragments": short_fragments_removed}


def load_corpus_points_v2(
    corpus_name: str, path: Path, min_expression_words: int,
) -> tuple[list[dict], dict[str, int]]:
    """v2-archive counterpart of load_corpus_points, for a run produced by
    thesis_corpus.extract_v2 (literature/interviews only -- MIVILUDES needs
    load_miviludes_points_v2 below for its translation join). Unlike v1,
    no exact-duplicate-within-document filtering: v2's own screen already
    rejects exact duplicates within a chunk (S14); an identical short
    named-entity mention recurring across two different chunks of the same
    document (e.g. "la Scientologie" said twice) is real signal, not an
    extraction artefact, so it's kept.

    `min_expression_words` is accepted as a parameter (not hardcoded)
    because v2 archives already screen short fragments at extraction time
    with a referent test a bare word-count floor can't express (see
    screen_v2.py's S7 -- "Tomato cult!" is 2 words but meaningful); v2 call
    sites are expected to pass 0 so this never double-filters.

    Returns (points, removal_counts) -- same {"duplicates", "short_fragments"}
    shape as load_corpus_points for uniform logging; "duplicates" is
    always 0 here."""
    points = []
    response_rank_by_document: dict[str, int] = defaultdict(int)
    for_interviews = corpus_name == "interviews"
    short_fragments_removed = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            document_id = item["document_id"]
            text = item["embedding_text"]
            key = f"{document_id}:{item['chunk_index']}"

            response_rank = None
            if for_interviews:
                response_rank_by_document[document_id] += 1
                response_rank = response_rank_by_document[document_id]

            if len(text.split()) < min_expression_words:
                short_fragments_removed += 1
                continue

            points.append({
                "source_dataset": corpus_name,
                "point_role": "expression",
                "key": key,
                "label": text,
                "label_fr": None,
                "attribution": item.get("attribution"),
                "claim_mode": item.get("claim_mode"),
                "epistemic_status": item.get("epistemic_status"),
                "response_rank": response_rank,
                "vector": item["embedding_vector"],
            })
    return points, {"duplicates": 0, "short_fragments": short_fragments_removed}


def load_miviludes_points_v2(
    path: Path, translations: dict[str, dict], min_expression_words: int,
) -> tuple[list[dict], dict[str, int]]:
    """v2 MIVILUDES counterpart of load_corpus_points: same
    English-translation-as-primary-vector policy as v1 (closing the same
    measured language-asymmetry gap, Methods.tex "Language asymmetry"),
    keyed on v2's own occurrence numbering (rebuilt here from the v2
    archive's own read order -- independent of, and not comparable to,
    v1's numbering over a different set of expressions). No
    duplicate-within-document filter, same reasoning as
    load_corpus_points_v2."""
    points = []
    occurrence_by_chunk: dict[tuple[str, int], int] = defaultdict(int)
    short_fragments_removed = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            document_id = item["document_id"]
            chunk_index = item["chunk_index"]
            text = item["embedding_text"]
            key = f"{document_id}:{chunk_index}"
            chunk = (document_id, chunk_index)
            translation_key = f"{document_id}:{chunk_index}:{occurrence_by_chunk[chunk]}"
            occurrence_by_chunk[chunk] += 1

            if len(text.split()) < min_expression_words:
                short_fragments_removed += 1
                continue

            translation = translations.get(translation_key)
            if translation is None:
                raise SystemExit(
                    f"Missing translation for v2 MIVILUDES expression {translation_key} -- "
                    "the v2 translations file doesn't cover every pooled expression. Rerun "
                    "translate_miviludes_expressions.py against the v2 archive."
                )
            if translation["text_fr"] != text:
                raise SystemExit(
                    f"Translation mismatch for v2 MIVILUDES expression {translation_key}: "
                    f"archive text {text!r} != translation's text_fr {translation['text_fr']!r}. "
                    "The occurrence-based join has drifted out of sync with the v2 archive -- "
                    "do not trust this pipeline run."
                )

            points.append({
                "source_dataset": "miviludes",
                "point_role": "expression",
                "key": key,
                "label": translation["text_en"],
                "label_fr": translation["text_fr"],
                "attribution": item.get("attribution"),
                "claim_mode": item.get("claim_mode"),
                "epistemic_status": item.get("epistemic_status"),
                "response_rank": None,
                "vector": translation["embedding_vector_en"],
            })
    return points, {"duplicates": 0, "short_fragments": short_fragments_removed}


def _report_translation_fidelity(label: str, similarities: list[tuple[str, float]]) -> None:
    """Shared reporting for both the 17-criteria and (once wired in) the
    914-expression translation-fidelity checks: full distribution, not
    just mean/min -- a flat low-percentile number is expected for short
    phrases (see COSINE_FLAG_THRESHOLD) and shouldn't itself read as a
    problem. Only pairs below COSINE_FLAG_THRESHOLD are named individually,
    as candidates for manual inspection, not as confirmed errors."""
    values = np.array([c for _, c in similarities], dtype=np.float64)
    print(f"\n{label} FR vs EN raw-embedding cosine similarity "
          f"(diagnostic; only FR is pooled into the shared space):")
    print(f"  n={len(values)}  mean={values.mean():.4f}  median={np.median(values):.4f}  "
          f"p10={np.percentile(values, 10):.4f}  min={values.min():.4f}")
    flagged = sorted((k, c) for k, c in similarities if c < COSINE_FLAG_THRESHOLD)
    if flagged:
        print(f"  {len(flagged)} pair(s) below {COSINE_FLAG_THRESHOLD:.2f} -- candidates for manual "
              f"inspection (not automatically errors; short phrases embed noisily cross-lingually):")
        for key, cosine in flagged:
            print(f"    {key:<40} cosine={cosine:.4f}")
        logger.warning(
            "%d %s FR/EN pair(s) below the %.2f flag threshold -- inspect by hand before treating as an error.",
            len(flagged), label, COSINE_FLAG_THRESHOLD,
        )


def check_miviludes_translation_fidelity(path: Path) -> None:
    """Diagnostic only: cosine similarity between each criterion's raw
    (pre-PCA) French and English embeddings. Only the French embedding is
    pooled into the shared space (see load_miviludes_criteria_points) --
    this replaces the old approach of pooling both and comparing their
    shared-space distance from the origin, which spent a second
    near-duplicate point on a check obtainable directly from the raw
    vectors."""
    similarities = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            fr = np.array(item["embedding_vector_fr"], dtype=np.float64)
            en = np.array(item["embedding_vector_en"], dtype=np.float64)
            cosine = float(np.dot(fr, en) / (np.linalg.norm(fr) * np.linalg.norm(en)))
            similarities.append((item["id"], cosine))

    _report_translation_fidelity("MIVILUDES criteria", similarities)


def load_miviludes_translations(path: Path) -> dict[str, dict]:
    """`document_id:chunk_index` is NOT a unique key per expression --
    checked directly: a chunk can and does yield multiple expressions (e.g.
    MIVILUDES's 914 expressions span only 118 unique document_id:chunk_index
    pairs; literature's 39,236 span only 5,146), a pre-existing, previously
    harmless property of this pipeline that becomes a real bug the moment
    something (this function) tries to use that pair as a dict key -- an
    earlier version of this function did exactly that and silently
    collapsed 914 translations down to 118, each one then handed out to
    every expression sharing its chunk regardless of which French text it
    actually translated.

    Fixed by keying on `document_id:chunk_index:occurrence`, where
    `occurrence` is "the Nth time this document_id:chunk_index pair has
    been seen so far" -- unique by construction, and reconstructible from
    load_corpus_points' own read of the source archive because both files
    are traversed in on-disk order and that order was verified identical
    (translate_miviludes_expressions.py processes the source archive
    sequentially and never reorders it)."""
    translations = {}
    occurrence_by_chunk: dict[tuple[str, int], int] = defaultdict(int)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            chunk = (item["document_id"], item["chunk_index"])
            occurrence = occurrence_by_chunk[chunk]
            occurrence_by_chunk[chunk] += 1
            translations[f"{item['document_id']}:{item['chunk_index']}:{occurrence}"] = item
    return translations


def check_miviludes_expression_translation_fidelity(translations: dict[str, dict]) -> None:
    """Diagnostic only: same raw-embedding cosine-similarity check as
    check_miviludes_translation_fidelity above, applied to MIVILUDES's 914
    expressions instead of the 17 criteria. Uses the self-contained
    embedding_vector_fr/_en pair translate_miviludes_expressions.py already
    stores side by side, so this never needs to reopen the (large,
    gitignored) main archive."""
    similarities = []
    for key, item in translations.items():
        fr = np.array(item["embedding_vector_fr"], dtype=np.float64)
        en = np.array(item["embedding_vector_en"], dtype=np.float64)
        cosine = float(np.dot(fr, en) / (np.linalg.norm(fr) * np.linalg.norm(en)))
        similarities.append((key, cosine))

    _report_translation_fidelity("MIVILUDES expressions", similarities)


def load_miviludes_criteria_points(path: Path) -> list[dict]:
    points = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            points.append({
                "source_dataset": "miviludes_criteria",
                "point_role": "expression",
                "key": item["id"],
                "label": item["criterion_fr"],
                "label_en": item["criterion_en"],
                "vector": item["embedding_vector_fr"],
            })
    return points


def load_concept_backbone_points(path: Path) -> list[dict]:
    points = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            points.append({
                "source_dataset": "concept_backbone",
                "point_role": "reference",
                "key": item["concept_id"],
                "label": item["concept_en"],
                "vector": item["embedding_vector"],
            })
    return points


def load_structural_concepts_points(path: Path) -> list[dict]:
    """Reads structural_concepts_embedded.jsonl (extract_structural_concepts.py's
    candidates CSV, embedded via the existing embed_concept_backbone.py --
    same "<term>: <gloss>" format, just a different input/output path). Also
    a `point_role="reference"` point-set, but corpus-derived rather than
    topic-neutral -- see the module docstring's "reference" bullet for the
    distinction from `concept_backbone`."""
    points = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            points.append({
                "source_dataset": "structural_concepts",
                "point_role": "reference",
                "key": item["concept_id"],
                "label": item["concept_en"],
                "mention_distribution": {
                    "literature": int(item.get("literature_mentions", 0)),
                    "miviludes": int(item.get("miviludes_mentions", 0)),
                    "interviews": int(item.get("interviews_mentions", 0)),
                },
                "vector": item["embedding_vector"],
            })
    return points


def load_conceptnet_concepts_points(path: Path) -> list[dict]:
    """Reads conceptnet_concepts_embedded.jsonl -- the hand-reviewed subset
    of extract_conceptnet_concepts.py's candidates (is_generic=="true" rows
    only; filter_conceptnet_concepts.py produces the CSV that gets embedded)
    -- via the same embed_concept_backbone.py pipeline as concept_backbone
    and structural_concepts. A `point_role="reference"` point-set, seeded
    from structural_concepts but generalized via ConceptNet's associative
    graph rather than drawn directly from corpus text -- see the module
    docstring's "reference" bullet for how this differs from the other two
    reference subsets."""
    points = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("is_generic") != "true":
                raise ValueError(
                    f"{path} contains a row with is_generic != 'true' "
                    f"({item['concept_id']!r}) -- only hand-reviewed-kept "
                    "rows should ever reach this file; re-run "
                    "filter_conceptnet_concepts.py before re-embedding."
                )
            points.append({
                "source_dataset": "conceptnet_concepts",
                "point_role": "reference",
                "key": item["concept_id"],
                "label": item["concept_en"],
                "vector": item["embedding_vector"],
            })
    return points


def load_emergent_entities(archive_paths: dict[str, Path], min_mentions: int) -> list[dict]:
    """Pools one point per unique (normalized) entity anchor mentioned at
    least `min_mentions` times across all three corpus archives, using the
    per-anchor embedding Stage 2 already computed (entity_anchor_vectors) --
    never before pooled into any space. First-seen vector per normalized
    anchor is kept (the embedding is a function of the literal anchor text
    alone, so occurrences of the same normalized string carry equivalent
    vectors modulo casing/whitespace, already normalized away here).

    These are "emergent entities" (`point_role="emergent"`): named
    entities/groups/concepts mentioned BY the corpora themselves, as
    distinct from an "expression" point (a claim a source makes) or a
    "reference" point (the corpus-independent concept backbone).

    Each point also carries `mention_distribution`: a per-corpus mention
    count (e.g. {"literature": 820, "miviludes": 15, "interviews": 12}) --
    provenance metadata only, not used in the PCA fit, so an anchor
    overwhelmingly mentioned in one corpus can be told apart from one
    mentioned evenly across all three."""
    vector_by_anchor: dict[str, list[float]] = {}
    mentions_by_corpus: dict[str, Counter[str]] = defaultdict(Counter)

    for corpus_name, path in archive_paths.items():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                raw_vectors = item.get("entity_anchor_vectors") or {}
                for raw_anchor, vector in raw_vectors.items():
                    key = normalize_anchor(raw_anchor)
                    if not key:
                        continue
                    mentions_by_corpus[corpus_name][key] += 1
                    vector_by_anchor.setdefault(key, vector)

    total_mentions: Counter[str] = Counter()
    for corpus_counts in mentions_by_corpus.values():
        total_mentions.update(corpus_counts)

    def distribution_for(anchor: str) -> dict[str, int]:
        return {corpus_name: mentions_by_corpus[corpus_name].get(anchor, 0) for corpus_name in archive_paths}

    print(f"\nEmergent entities: {len(total_mentions)} unique (normalized) across all corpora; "
          f"top 20 by mention count:")
    for anchor, count in total_mentions.most_common(20):
        print(f"  {anchor}: {count} total mentions -- {distribution_for(anchor)}")

    points = []
    for anchor, count in total_mentions.items():
        if count < min_mentions:
            continue
        points.append({
            "source_dataset": "emergent_entities",
            "point_role": "emergent",
            "key": anchor,
            "label": anchor,
            "mention_distribution": distribution_for(anchor),
            "vector": vector_by_anchor[anchor],
        })
    return points


def sanity_checks(points: list[dict], coords: np.ndarray) -> None:
    norms_by_dataset = defaultdict(list)
    for p, c in zip(points, coords):
        norms_by_dataset[p["source_dataset"]].append(float(np.linalg.norm(c)))

    print("\nMean shared-space vector norm by source_dataset (diagnostic, not a pass/fail check):")
    for dataset, norms in sorted(norms_by_dataset.items()):
        print(f"  {dataset:<24} n={len(norms):<6} mean_norm={sum(norms)/len(norms):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variance-threshold", type=float, default=VARIANCE_THRESHOLD)
    parser.add_argument("--entity-anchor-min-mentions", type=int, default=ENTITY_ANCHOR_MIN_MENTIONS,
                         help="Minimum times an entity anchor must be mentioned across all corpora to get its own point.")
    parser.add_argument("--run-tag", default=None,
                         help="If given, pool the v2 extraction run with this tag instead of v1: reads "
                              "processed/v2/<corpus>/run_<tag>/criterion_expressions.jsonl for each corpus and "
                              "processed/v2/miviludes/run_<tag>/expression_translations_embedded.jsonl for the "
                              "MIVILUDES translation join, and defaults --output-dir to "
                              "processed/shared_space_v2/ instead of processed/shared_space/. v1's own archives "
                              "and processed/shared_space/ are never opened for writing when this is set.")
    parser.add_argument("--output-dir", type=Path, default=None,
                         help="Override the output directory (default: processed/shared_space/, or "
                              "processed/shared_space_v2/ if --run-tag is given).")
    parser.add_argument("--min-expression-words", type=int, default=MIN_EXPRESSION_WORDS,
                         help="Pooling-time short-expression filter (word count). v1 default is 5. v2 archives "
                              "already screen short fragments at extraction time with a referent test a bare "
                              "word-count floor can't express -- pass 0 for a v2 run so a short but meaningful "
                              "item like 'Tomato cult!' isn't double-filtered.")
    args = parser.parse_args()

    if args.run_tag:
        v2_root = PROCESSED_DIR / "v2"
        corpus_archives = {
            corpus: v2_root / corpus / f"run_{args.run_tag}" / "criterion_expressions.jsonl"
            for corpus in CORPUS_ARCHIVES
        }
        miviludes_translations_path = v2_root / "miviludes" / f"run_{args.run_tag}" / "expression_translations_embedded.jsonl"
        output_dir = args.output_dir or (SHARED_SPACE_DIR.parent / "shared_space_v2")
    else:
        corpus_archives = CORPUS_ARCHIVES
        miviludes_translations_path = MIVILUDES_EXPRESSION_TRANSLATIONS_PATH
        output_dir = args.output_dir or SHARED_SPACE_DIR

    output_path = output_dir / "embedding_space.jsonl"
    variance_csv_path = output_dir / "variance_curve.csv"
    variance_plot_path = output_dir / "variance_curve.png"
    variance_json_path = output_dir / "variance_curve.json"
    pca_transform_path = output_dir / "pca_transform.joblib"
    pca_transform_metadata_path = output_dir / "pca_transform_metadata.json"

    if not miviludes_translations_path.exists():
        raise SystemExit(
            f"Missing: {miviludes_translations_path} -- run "
            "translate_miviludes_expressions.py (on the Ollama machine, pointed at the matching "
            "--source archive), then copy the output back to this path."
        )
    miviludes_translations = load_miviludes_translations(miviludes_translations_path)
    check_miviludes_expression_translation_fidelity(miviludes_translations)

    points: list[dict] = []
    counts: dict[str, int] = {}
    removed: dict[str, dict[str, int]] = {}

    for corpus_name, path in corpus_archives.items():
        if not path.exists():
            raise SystemExit(f"Missing corpus archive: {path}")
        if args.run_tag:
            if corpus_name == "miviludes":
                new_points, removal_counts = load_miviludes_points_v2(path, miviludes_translations, args.min_expression_words)
            else:
                new_points, removal_counts = load_corpus_points_v2(corpus_name, path, args.min_expression_words)
        else:
            translations = miviludes_translations if corpus_name == "miviludes" else None
            new_points, removal_counts = load_corpus_points(corpus_name, path, translations)
        counts[corpus_name] = len(new_points)
        removed[corpus_name] = removal_counts
        points.extend(new_points)

    logger.info("Filtered during pooling (duplicates / short fragments, per corpus): %s", removed)
    total_duplicates = sum(r["duplicates"] for r in removed.values())
    total_short = sum(r["short_fragments"] for r in removed.values())
    print(f"\nFiltered during pooling: {total_duplicates} exact-duplicate expressions removed "
          f"(per document, keeping first occurrence), {total_short} short fragments removed "
          f"(under {args.min_expression_words} words). Per corpus: {removed}")

    if not MIVILUDES_CRITERIA_PATH.exists():
        raise SystemExit(f"Missing: {MIVILUDES_CRITERIA_PATH}")
    check_miviludes_translation_fidelity(MIVILUDES_CRITERIA_PATH)
    criteria_points = load_miviludes_criteria_points(MIVILUDES_CRITERIA_PATH)
    counts["miviludes_criteria"] = len(criteria_points)
    points.extend(criteria_points)

    if not CONCEPT_BACKBONE_PATH.exists():
        raise SystemExit(f"Missing: {CONCEPT_BACKBONE_PATH}")
    concept_points = load_concept_backbone_points(CONCEPT_BACKBONE_PATH)
    counts["concept_backbone"] = len(concept_points)
    points.extend(concept_points)

    if not STRUCTURAL_CONCEPTS_PATH.exists():
        raise SystemExit(
            f"Missing: {STRUCTURAL_CONCEPTS_PATH} -- run "
            "extract_structural_concepts.py, then embed it (on the Ollama "
            "machine) via embed_concept_backbone.py --input "
            "dictionaries/structural_concepts_candidates.csv --output "
            f"{STRUCTURAL_CONCEPTS_PATH}"
        )
    structural_concept_points = load_structural_concepts_points(STRUCTURAL_CONCEPTS_PATH)
    counts["structural_concepts"] = len(structural_concept_points)
    points.extend(structural_concept_points)

    if not CONCEPTNET_CONCEPTS_PATH.exists():
        raise SystemExit(
            f"Missing: {CONCEPTNET_CONCEPTS_PATH} -- hand-review "
            "dictionaries/conceptnet_concepts_candidates.csv, then run "
            "filter_conceptnet_concepts.py + embed_concept_backbone.py "
            "(on the Ollama machine) to produce it -- see "
            "filter_conceptnet_concepts.py's docstring for the exact commands."
        )
    conceptnet_concept_points = load_conceptnet_concepts_points(CONCEPTNET_CONCEPTS_PATH)
    counts["conceptnet_concepts"] = len(conceptnet_concept_points)
    points.extend(conceptnet_concept_points)

    emergent_entity_points = load_emergent_entities(corpus_archives, args.entity_anchor_min_mentions)
    counts["emergent_entities"] = len(emergent_entity_points)
    points.extend(emergent_entity_points)

    logger.info("Pooled point counts: %s", counts)
    logger.info("Total pooled points: %d", len(points))

    vectors = np.array([p["vector"] for p in points], dtype=np.float64)
    for p in points:
        del p["vector"]
    if vectors.shape[1] != EMBEDDING_DIM:
        raise SystemExit(f"Expected {EMBEDDING_DIM}-d vectors, got {vectors.shape[1]}")

    logger.info("Standardizing pooled matrix (zero mean, unit variance per dimension) before PCA...")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(vectors)

    logger.info("Fitting full-rank PCA to get the complete explained-variance curve...")
    full_pca = PCA(random_state=args.seed)
    full_coords = full_pca.fit_transform(scaled)
    cumulative_variance = np.cumsum(full_pca.explained_variance_ratio_)

    k = int(np.searchsorted(cumulative_variance, args.variance_threshold) + 1)
    k = min(k, len(cumulative_variance))
    variance_at_k = float(cumulative_variance[k - 1])
    logger.info("%.1f%% cumulative variance at k=%d (threshold: %.0f%%)", variance_at_k * 100, k, args.variance_threshold * 100)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(variance_csv_path, "w", encoding="utf-8") as f:
        f.write("n_components,cumulative_variance\n")
        for i, v in enumerate(cumulative_variance, 1):
            f.write(f"{i},{v}\n")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance)
    plt.axhline(args.variance_threshold, color="gray", linestyle="--", linewidth=1)
    plt.axvline(k, color="gray", linestyle="--", linewidth=1)
    plt.scatter([k], [variance_at_k], color="red", zorder=5, label=f"k={k}, {variance_at_k*100:.1f}%")
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title("Shared cross-corpus PCA: explained variance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(variance_plot_path, dpi=150)
    plt.close()

    variance_json_path.write_text(
        json.dumps({
            "curve": [float(v) for v in cumulative_variance],
            "chosen_k": k,
            "variance_at_k": variance_at_k,
            "threshold": args.variance_threshold,
        }, indent=2),
        encoding="utf-8",
    )

    shared_coords = full_coords[:, :k]

    logger.info("Writing %s ...", output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        for p, coord in zip(points, shared_coords):
            out = {
                "source_dataset": p["source_dataset"],
                "point_role": p["point_role"],
                "key": p["key"],
                "label": p["label"],
                "label_en": p.get("label_en"),
                "label_fr": p.get("label_fr"),
                "attribution": p.get("attribution"),
                "claim_mode": p.get("claim_mode"),
                "epistemic_status": p.get("epistemic_status"),
                "response_rank": p.get("response_rank"),
                "mention_distribution": p.get("mention_distribution"),
                "shared_space_vector": coord.tolist(),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    sanity_checks(points, shared_coords)

    logger.info("Persisting the fitted StandardScaler + PCA (%s) ...", pca_transform_path)
    output_sha256 = _sha256_file(output_path)
    joblib.dump({"scaler": scaler, "pca": full_pca}, pca_transform_path)
    pca_transform_metadata_path.write_text(
        json.dumps({
            "k": k,
            "embedding_dim": EMBEDDING_DIM,
            "variance_threshold": args.variance_threshold,
            "variance_at_k": variance_at_k,
            "seed": args.seed,
            "n_points_fit": len(points),
            "embedding_space_sha256": output_sha256,
            "git_commit": _git_commit_hash(),
        }, indent=2),
        encoding="utf-8",
    )

    print(f"\nDone. {len(points)} points, k={k} ({variance_at_k*100:.1f}% variance).")
    print(f"Output: {output_path}")
    print(f"Variance curve: {variance_csv_path}, {variance_plot_path}, {variance_json_path}")
    print(f"Persisted transform: {pca_transform_path}, {pca_transform_metadata_path}")


if __name__ == "__main__":
    main()
