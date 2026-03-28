"""Response parser for insight extraction."""

import re

from txt_splitt.errors import ParseError
from txt_splitt.insights.types import Insight
from txt_splitt.sentences.types import SentenceRange

__all__ = ["InsightParser"]

# Reuse the same range-parsing patterns used in sentences/parsers.py
_RANGE_PATTERN = re.compile(r"(\d+)\s*-\s*(\d+)")
_SINGLE_NUMBER_PATTERN = re.compile(r"(\d+)")

# Greedy match: grab longest label, expect colon + digit-starting ranges.
# Same pattern as sentences/parsers.py — handles colons inside insight names.
_INSIGHT_LINE_PATTERN = re.compile(r"^(.+):\s*(\d[\d\s,\-]*)\s*$")


class InsightParser:
    """Parse LLM insight responses into a list of Insight objects.

    Expected format per line (text mode):
        Insight Name: 0-5, 10-15

    Labels are plain strings (no hierarchy separator).
    Insights with the same normalized name are merged — this allows the same
    insight to appear across multiple chunks and be combined into one.
    Ranges are clamped to [0, sentence_count-1].
    """

    def parse(self, response: str, sentence_count: int) -> list[Insight]:
        if sentence_count <= 0:
            raise ParseError("sentence_count must be positive")
        return _parse_text_insights(response, sentence_count)


def _normalize_name(name: str) -> str:
    """Normalize an insight name for deduplication (lowercase, collapsed whitespace)."""
    return " ".join(name.lower().split())


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


def _clamp_range(start: int, end: int, max_index: int) -> SentenceRange | None:
    if max_index < 0:
        return None
    start = max(0, min(start, max_index))
    end = max(0, min(end, max_index))
    if start > end:
        start, end = end, start
    return SentenceRange(start=start, end=end)


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


def _build_insights(
    grouped_ranges: dict[str, list[SentenceRange]],
    name_order: list[str],
    canonical_names: dict[str, str],
) -> list[Insight]:
    insights: list[Insight] = []
    for norm_name in name_order:
        merged = _merge_ranges(grouped_ranges[norm_name])
        if merged:
            insights.append(
                Insight(name=canonical_names[norm_name], ranges=tuple(merged))
            )
    if not insights:
        raise ParseError("No valid insights found in response")
    return insights


def _parse_text_insights(response: str, sentence_count: int) -> list[Insight]:
    max_index = sentence_count - 1
    lines = [ln.strip() for ln in response.strip().splitlines() if ln.strip()]

    grouped_ranges: dict[str, list[SentenceRange]] = {}
    name_order: list[str] = []
    canonical_names: dict[str, str] = {}

    for ln in lines:
        match = _INSIGHT_LINE_PATTERN.match(ln)
        if match:
            name = match.group(1).strip()
            ranges_str = match.group(2).strip()
        elif ":" in ln:
            name, ranges_str = ln.split(":", 1)
            name = name.strip()
            ranges_str = ranges_str.strip()
        else:
            continue

        if not name:
            continue

        norm = _normalize_name(name)
        parsed_ranges = _parse_range_string(ranges_str)
        clamped: list[SentenceRange] = []
        for start, end in parsed_ranges:
            clamped_range = _clamp_range(start, end, max_index)
            if clamped_range is not None:
                clamped.append(clamped_range)

        if clamped:
            if norm not in grouped_ranges:
                grouped_ranges[norm] = []
                name_order.append(norm)
                canonical_names[norm] = name
            grouped_ranges[norm].extend(clamped)

    return _build_insights(grouped_ranges, name_order, canonical_names)
