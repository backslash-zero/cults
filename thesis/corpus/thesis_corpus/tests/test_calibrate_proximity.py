import re

import numpy as np

from thesis_corpus.calibrate_proximity import mask_text, masked_entity_test, retrieval, summarise, triplets


def test_mask_text_replaces_every_alias_match():
    regex = re.compile(r"scientolog", re.I)
    assert mask_text("Scientology, la Scientologie", regex) == "[name], la [name]"
    assert mask_text("the Church of Scientology", regex) == "the Church of [name]"


def test_mask_text_removes_first_names_and_merges():
    regex = re.compile(r"manson|kasabian", re.I)
    assert mask_text("Charles Manson and Linda Kasabian", regex, r"charles|linda") == "[name] and [name]"


def test_retrieval_perfect_and_chance():
    pool = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    own = np.array([True, True, False, False])
    assert retrieval(np.array([1.0, 0.0]), pool, own) == {"precision_at_k": 1.0, "mean_rank": 1.5}
    assert retrieval(np.array([0.0, 1.0]), pool, own) == {"precision_at_k": 0.0, "mean_rank": 3.5}


def test_masking_lowers_retrieval_when_the_name_carried_it():
    keys = ["a", "b", "c", "d"]
    pool = np.array([[1.0, 0.0], [0.95, 0.05], [0.0, 1.0], [0.1, 0.9]])
    masked = [{"entity": "X", "key": "a", "document_id": "i1", "vector": np.array([-1.0, 0.1])},
              {"entity": "X", "key": "b", "document_id": "i2", "vector": np.array([-0.9, 0.2])}]
    row = masked_entity_test(keys, pool, masked, {"X": np.array([1.0, 0.0])})[0]
    assert row["precision_unmasked"] == 1.0
    assert row["precision_masked"] == 0.0
    assert row["chance_precision"] == 0.5


def test_triplets_compare_variant_with_same_topic_partner():
    rows = [
        {"contrast": "7_same_topic_different_claim", "base_id": "N1|N2", "domain": "cult", "register": "neutral", "cosine": 0.6},
        {"contrast": "1_paraphrase", "base_id": "N1", "domain": "cult", "register": "neutral", "cosine": 0.9},
        {"contrast": "3_negation", "base_id": "N1", "domain": "cult", "register": "neutral", "cosine": 0.5},
    ]
    out = {t["contrast"]: t["variant_closer"] for t in triplets(rows)}
    assert out == {"1_paraphrase": True, "3_negation": False}


def test_summarise_groups_by_contrast_and_domain():
    rows = [{"contrast": "1_paraphrase", "domain": "cult", "cosine": c} for c in (0.7, 0.8, 0.9)]
    s = summarise(rows)[0]
    assert (s["n"], s["median"], s["min"], s["max"]) == (3, 0.8, 0.7, 0.9)
