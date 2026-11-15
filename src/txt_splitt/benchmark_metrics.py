"""Descriptive measurements of sentence grouping; these are not quality scores.

Ranges use inclusive, zero-based sentence positions. A range is invalid if reversed
or outside the result's sentence tuple; invalid ranges contribute no assignments.
Topics are unique full label paths, including the empty path. Topic sizes union
positions across all groups with the same label. Coverage and multiple assignment
count distinct topics, so duplicate ranges do not inflate these measurements.
Character sizes sum sentence text lengths, never restored source-offset spans.
"""

from __future__ import annotations

from statistics import median
from typing import Any

from txt_splitt.sentences.types import SplitResult


def _summary(values: list[int]) -> dict[str, int | float]:
    """Summarize observations, using zero for all empty-sample statistics."""
    return {
        "count": len(values),
        "min": min(values, default=0),
        "max": max(values, default=0),
        "mean": sum(values) / len(values) if values else 0.0,
        "median": median(values) if values else 0.0,
    }


def measure_result(result: SplitResult) -> dict[str, Any]:
    """Return JSON-compatible raw records, counts, rates and size summaries.

    Rates use total sentence count; singleton fractions use topic/range count.
    Empty denominators yield zero. Invalid ranges remain in the range denominator
    and raw records, with zero sizes. Hierarchy records include the root and every
    observed prefix (leaves have zero children); children are distinct next labels.
    Ten bins partition sentence positions using floor(i * sentence_count / 10),
    with half-open bounds. Empty bins have no topics. A topic counts in each bin
    containing at least one of its assigned positions. Labels and prefixes sort
    lexicographically, while range records retain input order.
    """
    sentence_count = len(result.sentences)
    lengths = [len(sentence.text) for sentence in result.sentences]
    topic_positions: dict[tuple[str, ...], set[int]] = {}
    topic_range_counts: dict[tuple[str, ...], int] = {}
    range_records: list[dict[str, Any]] = []
    invalid_range_count = 0
    for group_index, group in enumerate(result.groups):
        positions = topic_positions.setdefault(group.label, set())
        topic_range_counts[group.label] = topic_range_counts.get(group.label, 0) + len(
            group.ranges
        )
        for range_index, span in enumerate(group.ranges):
            valid = 0 <= span.start <= span.end < sentence_count
            indices = range(span.start, span.end + 1) if valid else range(0)
            positions.update(indices)
            invalid_range_count += int(not valid)
            range_records.append(
                {
                    "group_index": group_index,
                    "range_index": range_index,
                    "label": list(group.label),
                    "start": span.start,
                    "end": span.end,
                    "valid": valid,
                    "sentence_count": len(indices),
                    "clean_chars": sum(lengths[index] for index in indices),
                }
            )

    assignments = [0] * sentence_count
    topic_records: list[dict[str, Any]] = []
    children: dict[tuple[str, ...], set[str]] = {(): set()}
    for label, positions in sorted(topic_positions.items()):
        for index in positions:
            assignments[index] += 1
        topic_records.append(
            {
                "label": list(label),
                "sentence_count": len(positions),
                "clean_chars": sum(lengths[index] for index in positions),
                "range_count": topic_range_counts[label],
            }
        )
        for depth in range(len(label)):
            children.setdefault(label[:depth], set()).add(label[depth])
        children.setdefault(label, set())
    hierarchy: list[dict[str, Any]] = [
        {"prefix": list(prefix), "child_count": len(names)}
        for prefix, names in sorted(children.items())
    ]
    bins: list[dict[str, Any]] = []
    for bin_index in range(10):
        start = bin_index * sentence_count // 10
        end = (bin_index + 1) * sentence_count // 10
        labels = [
            list(label)
            for label, positions in sorted(topic_positions.items())
            if any(start <= index < end for index in positions)
        ]
        bins.append(
            {
                "bin_index": bin_index,
                "start": start,
                "end": end,
                "topic_count": len(labels),
                "topics": labels,
            }
        )
    covered = sum(count > 0 for count in assignments)
    multiple = sum(count > 1 for count in assignments)
    topic_sizes = [len(positions) for positions in topic_positions.values()]
    range_sizes = [record["sentence_count"] for record in range_records]
    return {
        "sentence_count": sentence_count,
        "clean_chars": sum(lengths),
        "group_count": len(result.groups),
        "topic_count": len(topic_positions),
        "top_level_topic_count": len(children[()]),
        "range_count": len(range_records),
        "covered_sentence_count": covered,
        "coverage_rate": covered / sentence_count if sentence_count else 0.0,
        "uncovered_sentence_count": sentence_count - covered,
        "uncovered_rate": (sentence_count - covered) / sentence_count
        if sentence_count
        else 0.0,
        "multiply_assigned_sentence_count": multiple,
        "multiply_assigned_rate": multiple / sentence_count if sentence_count else 0.0,
        "invalid_range_count": invalid_range_count,
        "singleton_topic_fraction": topic_sizes.count(1) / len(topic_sizes)
        if topic_sizes
        else 0.0,
        "singleton_range_fraction": range_sizes.count(1) / len(range_sizes)
        if range_sizes
        else 0.0,
        "topic_sentence_size": _summary(topic_sizes),
        "topic_char_size": _summary(
            [record["clean_chars"] for record in topic_records]
        ),
        "range_sentence_size": _summary(range_sizes),
        "range_char_size": _summary(
            [record["clean_chars"] for record in range_records]
        ),
        "hierarchy_child_count": _summary(
            [record["child_count"] for record in hierarchy]
        ),
        "position_bin_topic_count": _summary(
            [record["topic_count"] for record in bins]
        ),
        "topics": topic_records,
        "ranges": range_records,
        "hierarchy": hierarchy,
        "position_bins": bins,
    }
