import re

from thesis_corpus.prepare_interview_segments import is_filler, match_entities


def test_discourse_only_segments_are_fillers():
    for text in ["Yeah, yeah.", "I don't know.", "Yeah, I don't have any names.", "what was it called?", "..."]:
        assert is_filler(text)


def test_short_content_segments_are_kept():
    for text in ["Manipulation, groupe, idéologie.", "Gourou.", "Religion", "Tomato cult!", "Qui suit des règles."]:
        assert not is_filler(text)


def test_match_entities_merges_variants():
    aliases = [{"entity": "Scientology", "regex": re.compile("scientolog", re.I)}]
    assert match_entities("La Scientologie, le truc de Raël", aliases) == ["Scientology"]
    assert match_entities("Church of Scientology", aliases) == ["Scientology"]
    assert match_entities("Mormons", aliases) == []
