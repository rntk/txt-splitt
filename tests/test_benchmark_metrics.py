"""Tests for descriptive grouping measurements."""

import json

import pytest

from txt_splitt.benchmark_metrics import measure_result
from txt_splitt.sentences.types import (
    Sentence,
    SentenceGroup,
    SentenceRange,
    SplitResult,
)


def _sentences(count: int) -> tuple[Sentence, ...]:
    # Source offsets intentionally differ from clean text lengths.
    return tuple(
        Sentence(i, i * 100, i * 100 + 90, "x" * (i + 1)) for i in range(count)
    )


def test_duplicate_paths_union_positions_and_measure_distinct_assignments() -> None:
    result = SplitResult(
        _sentences(5),
        (
            SentenceGroup(("A", "one"), (SentenceRange(0, 1), SentenceRange(1, 2))),
            SentenceGroup(("A", "one"), (SentenceRange(2, 2),)),
            SentenceGroup(("A", "two"), (SentenceRange(2, 3),)),
            SentenceGroup(("B",), (SentenceRange(3, 3),)),
        ),
    )
    metrics = measure_result(result)
    assert metrics["sentence_count"] == 5
    assert metrics["clean_chars"] == 15
    assert metrics["group_count"] == 4
    assert metrics["topic_count"] == 3
    assert metrics["top_level_topic_count"] == 2
    assert metrics["range_count"] == 5
    assert metrics["topics"] == [
        {
            "label": ["A", "one"],
            "sentence_count": 3,
            "clean_chars": 6,
            "range_count": 3,
        },
        {
            "label": ["A", "two"],
            "sentence_count": 2,
            "clean_chars": 7,
            "range_count": 1,
        },
        {"label": ["B"], "sentence_count": 1, "clean_chars": 4, "range_count": 1},
    ]
    assert metrics["coverage_rate"] == 0.8
    assert metrics["uncovered_sentence_count"] == 1
    assert metrics["uncovered_rate"] == 0.2
    assert metrics["multiply_assigned_sentence_count"] == 2
    assert metrics["multiply_assigned_rate"] == 0.4
    assert metrics["singleton_topic_fraction"] == pytest.approx(1 / 3)
    assert metrics["singleton_range_fraction"] == 0.4
    assert metrics["hierarchy"] == [
        {"prefix": [], "child_count": 2},
        {"prefix": ["A"], "child_count": 2},
        {"prefix": ["A", "one"], "child_count": 0},
        {"prefix": ["A", "two"], "child_count": 0},
        {"prefix": ["B"], "child_count": 0},
    ]
    assert metrics["topic_sentence_size"] == {
        "count": 3,
        "min": 1,
        "max": 3,
        "mean": 2.0,
        "median": 2,
    }
    json.dumps(metrics, allow_nan=False)


def test_invalid_ranges_are_reported_and_excluded_without_clipping() -> None:
    result = SplitResult(
        _sentences(3),
        (
            SentenceGroup(
                ("A",), (SentenceRange(-1, 1), SentenceRange(2, 1), SentenceRange(1, 3))
            ),
        ),
    )
    metrics = measure_result(result)
    assert metrics["invalid_range_count"] == 3
    assert metrics["covered_sentence_count"] == 0
    assert metrics["topics"][0]["sentence_count"] == 0
    assert all(not record["valid"] for record in metrics["ranges"])
    assert metrics["range_sentence_size"]["max"] == 0


def test_ten_bins_count_topics_intersecting_positions() -> None:
    metrics = measure_result(
        SplitResult(
            _sentences(20),
            (
                SentenceGroup(("A",), (SentenceRange(0, 19),)),
                SentenceGroup(("B",), (SentenceRange(1, 2),)),
            ),
        )
    )
    bins = metrics["position_bins"]
    assert len(bins) == 10
    assert [(record["start"], record["end"]) for record in bins] == [
        (index * 2, index * 2 + 2) for index in range(10)
    ]
    assert [record["topic_count"] for record in bins] == [2, 2] + [1] * 8
    assert bins[0]["topics"] == [["A"], ["B"]]


def test_empty_result_has_finite_zero_statistics_and_ten_empty_bins() -> None:
    metrics = measure_result(SplitResult((), ()))
    assert metrics["coverage_rate"] == 0.0
    assert metrics["singleton_topic_fraction"] == 0.0
    assert metrics["topic_sentence_size"] == {
        "count": 0,
        "min": 0,
        "max": 0,
        "mean": 0.0,
        "median": 0.0,
    }
    assert len(metrics["position_bins"]) == 10
    assert all(record["topic_count"] == 0 for record in metrics["position_bins"])
    json.dumps(metrics, allow_nan=False)


def test_empty_label_is_a_topic_without_a_top_level_label() -> None:
    metrics = measure_result(
        SplitResult(_sentences(1), (SentenceGroup((), (SentenceRange(0, 0),)),))
    )
    assert metrics["topic_count"] == 1
    assert metrics["top_level_topic_count"] == 0
    assert metrics["covered_sentence_count"] == 1
    assert sum(record["topic_count"] for record in metrics["position_bins"]) == 1
