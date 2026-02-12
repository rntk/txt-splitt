"""Response parser implementations."""

import re

from txt_splitt.errors import ParseError
from txt_splitt.types import SentenceGroup, SentenceRange

# Compiled regex for parsing range strings like "0-5" or "10"
_RANGE_PATTERN = re.compile(r"(\d+)\s*-\s*(\d+)")
_SINGLE_NUMBER_PATTERN = re.compile(r"(\d+)")


class TopicRangeParser:
    """Parse LLM topic-range responses into sentence groups.

    Expected format per line:
        Category>Subcategory>Topic: 0-5, 10-15

    Labels are split by '>' into a tuple.
    Ranges are clamped to [0, sentence_count-1] and sorted by start.
    Does NOT fill gaps or validate coverage (that's the GapHandler's job).
    """

    def parse(self, response: str, sentence_count: int) -> list[SentenceGroup]:
        if sentence_count <= 0:
            raise ParseError("sentence_count must be positive")

        max_index = sentence_count - 1
        lines = [ln.strip() for ln in response.strip().splitlines() if ln.strip()]
        grouped_ranges: dict[tuple[str, ...], list[SentenceRange]] = {}
        label_order: list[tuple[str, ...]] = []

        for ln in lines:
            if ":" not in ln:
                continue

            topic_path, ranges_str = ln.split(":", 1)
            topic_path = topic_path.strip()
            ranges_str = ranges_str.strip()

            if not topic_path:
                continue

            label = tuple(
                part.strip() for part in topic_path.split(">") if part.strip()
            )
            if not label:
                continue

            parsed_ranges = _parse_range_string(ranges_str)
            clamped: list[SentenceRange] = []

            for start, end in parsed_ranges:
                start = max(0, min(start, max_index))
                end = max(0, min(end, max_index))
                if start > end:
                    start, end = end, start
                clamped.append(SentenceRange(start=start, end=end))

            if clamped:
                if label not in grouped_ranges:
                    grouped_ranges[label] = []
                    label_order.append(label)
                grouped_ranges[label].extend(clamped)

        groups: list[SentenceGroup] = []
        for label in label_order:
            merged_ranges = _merge_ranges(grouped_ranges[label])
            groups.append(SentenceGroup(label=label, ranges=tuple(merged_ranges)))

        if not groups:
            raise ParseError("No valid topic ranges found in response")

        return groups


def _parse_range_string(ranges_str: str) -> list[tuple[int, int]]:
    """Parse range string like '0-5, 10-15, 20' into (start, end) tuples."""
    results: list[tuple[int, int]] = []
    parts = [p.strip() for p in ranges_str.split(",")]

    for part in parts:
        if "-" in part and not part.startswith("-"):
            match = _RANGE_PATTERN.match(part)
            if match:
                results.append((int(match.group(1)), int(match.group(2))))
                continue

        match = _SINGLE_NUMBER_PATTERN.match(part)
        if match:
            n = int(match.group(1))
            results.append((n, n))

    return results


def _merge_ranges(ranges: list[SentenceRange]) -> list[SentenceRange]:
    """Sort and coalesce overlapping or adjacent ranges."""
    if not ranges:
        return []

    ordered = sorted(ranges, key=lambda r: (r.start, r.end))
    coalesced: list[SentenceRange] = [ordered[0]]

    for current in ordered[1:]:
        last = coalesced[-1]
        if current.start <= last.end + 1:
            coalesced[-1] = SentenceRange(
                start=last.start,
                end=max(last.end, current.end),
            )
            continue
        coalesced.append(current)

    return coalesced
