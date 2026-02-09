"""Tests for group joiners."""

from txt_splitt.joiners import AdjacentSameTopicJoiner
from txt_splitt.types import Sentence, SentenceGroup, SentenceRange


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
