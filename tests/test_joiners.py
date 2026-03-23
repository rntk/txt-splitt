"""Tests for group joiners."""

import pytest

from txt_splitt.sentences.joiners import (
    AdjacentSameTopicJoiner,
    join_sentences_by_groups,
)
from txt_splitt.sentences.types import Sentence, SentenceGroup, SentenceRange


def _sentences(n: int) -> list[Sentence]:
    return [
        Sentence(index=i, start=i * 10, end=i * 10 + 5, text=f"S{i}.") for i in range(n)
    ]


def test_adjacent_same_topic_joiner_merges_adjacent_groups() -> None:
    joiner = AdjacentSameTopicJoiner()
    groups = [
        SentenceGroup(label=("Tech",), ranges=(SentenceRange(start=0, end=1),)),
        SentenceGroup(label=("Tech",), ranges=(SentenceRange(start=2, end=3),)),
    ]

    result = joiner.join(groups, _sentences(4))

    assert len(result) == 1
    assert result[0].label == ("Tech",)
    assert result[0].ranges == (SentenceRange(start=0, end=3),)


def test_adjacent_same_topic_joiner_does_not_merge_non_adjacent_groups() -> None:
    joiner = AdjacentSameTopicJoiner()
    groups = [
        SentenceGroup(label=("Tech",), ranges=(SentenceRange(start=0, end=0),)),
        SentenceGroup(label=("Tech",), ranges=(SentenceRange(start=2, end=2),)),
    ]

    result = joiner.join(groups, _sentences(3))

    assert len(result) == 2


def test_adjacent_same_topic_joiner_does_not_merge_different_topics() -> None:
    joiner = AdjacentSameTopicJoiner()
    groups = [
        SentenceGroup(label=("Tech",), ranges=(SentenceRange(start=0, end=1),)),
        SentenceGroup(label=("Science",), ranges=(SentenceRange(start=2, end=3),)),
    ]

    result = joiner.join(groups, _sentences(4))

    assert len(result) == 2
    assert result[0].label == ("Tech",)
    assert result[1].label == ("Science",)


def test_join_sentences_by_groups_creates_joined_sentences() -> None:
    groups = [
        SentenceGroup(label=("TopicA",), ranges=(SentenceRange(start=0, end=1),)),
        SentenceGroup(label=("TopicB",), ranges=(SentenceRange(start=2, end=3),)),
    ]

    joined_sentences, remapped_groups = join_sentences_by_groups(groups, _sentences(4))

    assert [s.text for s in joined_sentences] == ["S0. S1.", "S2. S3."]
    assert remapped_groups[0].ranges == (SentenceRange(start=0, end=0),)
    assert remapped_groups[1].ranges == (SentenceRange(start=1, end=1),)


def test_join_sentences_by_groups_raises_on_out_of_bounds_range() -> None:
    groups = [
        SentenceGroup(label=("TopicA",), ranges=(SentenceRange(start=0, end=9),)),
    ]

    with pytest.raises(ValueError, match="exceeds sentence count"):
        join_sentences_by_groups(groups, _sentences(3))
