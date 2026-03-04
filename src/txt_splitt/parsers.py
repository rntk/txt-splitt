"""Response parser implementations."""

import json
import re
from collections.abc import Iterable
from typing import Literal

from txt_splitt.errors import ParseError
from txt_splitt.types import SentenceGroup, SentenceRange

# Compiled regex for parsing range strings like "0-5" or "10"
_RANGE_PATTERN = re.compile(r"(\d+)\s*-\s*(\d+)")
_SINGLE_NUMBER_PATTERN = re.compile(r"(\d+)")

# Greedy match for topic line: grabs the longest possible topic path,
# then expects a colon followed by digit-starting range content.
# Handles colons in topic names like "Apple: Mac Mini production: 6-20".
_TOPIC_LINE_PATTERN = re.compile(r"^(.+):\s*(\d[\d\s,\-]*)\s*$")


class TopicRangeParser:
    """Parse LLM topic-range responses into sentence groups.

    Expected format per line:
        Category>Subcategory>Topic: 0-5, 10-15

    Labels are split by '>' into a tuple.
    Any ':' inside a label segment is normalized into an additional '>' level.
    Ranges are clamped to [0, sentence_count-1] and sorted by start.
    Does NOT fill gaps or validate coverage (that's the GapHandler's job).
    """

    def __init__(self, *, input_mode: Literal["text", "json", "auto"] = "text") -> None:
        if input_mode not in {"text", "json", "auto"}:
            msg = f"input_mode must be 'text', 'json', or 'auto', got {input_mode!r}"
            raise ValueError(msg)
        self._input_mode = input_mode

    @property
    def supported_response_formats(self) -> frozenset[str]:
        if self._input_mode == "json":
            return frozenset({"json"})
        if self._input_mode == "auto":
            return frozenset({"text", "json"})
        return frozenset({"text"})

    def parse(self, response: str, sentence_count: int) -> list[SentenceGroup]:
        if sentence_count <= 0:
            raise ParseError("sentence_count must be positive")
        if self._input_mode == "text":
            return _parse_text_groups(response, sentence_count)
        if self._input_mode == "json":
            return _parse_json_groups(response, sentence_count)

        # auto mode: prefer JSON when possible, then fallback to text.
        try:
            return _parse_json_groups(response, sentence_count)
        except ParseError:
            return _parse_text_groups(response, sentence_count)


def _parse_text_groups(response: str, sentence_count: int) -> list[SentenceGroup]:
    max_index = sentence_count - 1
    lines = [ln.strip() for ln in response.strip().splitlines() if ln.strip()]
    grouped_ranges: dict[tuple[str, ...], list[SentenceRange]] = {}
    label_order: list[tuple[str, ...]] = []

    for ln in lines:
        # Try greedy regex first (handles colons in topic names)
        match = _TOPIC_LINE_PATTERN.match(ln)
        if match:
            topic_path = match.group(1).strip()
            ranges_str = match.group(2).strip()
        elif ":" in ln:
            # Fallback for lines with non-digit range content (e.g. "nope, 2-3")
            topic_path, ranges_str = ln.split(":", 1)
            topic_path = topic_path.strip()
            ranges_str = ranges_str.strip()
        else:
            continue

        if not topic_path:
            continue

        label = _normalize_label_parts(topic_path.split(">"))
        if not label:
            continue

        parsed_ranges = _parse_range_string(ranges_str)
        clamped: list[SentenceRange] = []

        for start, end in parsed_ranges:
            clamped_range = _clamp_range(start, end, max_index)
            if clamped_range is not None:
                clamped.append(clamped_range)

        if clamped:
            if label not in grouped_ranges:
                grouped_ranges[label] = []
                label_order.append(label)
            grouped_ranges[label].extend(clamped)

    return _build_groups(grouped_ranges, label_order)


def _parse_json_groups(response: str, sentence_count: int) -> list[SentenceGroup]:
    max_index = sentence_count - 1
    grouped_ranges: dict[tuple[str, ...], list[SentenceRange]] = {}
    label_order: list[tuple[str, ...]] = []

    for root in _parse_json_documents(response):
        topics = root.get("topics")
        if not isinstance(topics, list):
            continue
        for topic in topics:
            if not isinstance(topic, dict):
                continue
            label_obj = topic.get("label")
            if not isinstance(label_obj, list):
                continue
            label = _normalize_label_parts(
                part for part in label_obj if isinstance(part, str)
            )
            if not label:
                continue

            ranges_obj = topic.get("ranges")
            if not isinstance(ranges_obj, list):
                continue

            clamped: list[SentenceRange] = []
            for range_obj in ranges_obj:
                if not isinstance(range_obj, dict):
                    continue
                start = range_obj.get("start")
                end = range_obj.get("end")
                if not isinstance(start, int) or not isinstance(end, int):
                    continue
                clamped_range = _clamp_range(start, end, max_index)
                if clamped_range is not None:
                    clamped.append(clamped_range)

            if clamped:
                if label not in grouped_ranges:
                    grouped_ranges[label] = []
                    label_order.append(label)
                grouped_ranges[label].extend(clamped)

    return _build_groups(grouped_ranges, label_order)


def _parse_json_documents(response: str) -> list[dict[str, object]]:
    decoder = json.JSONDecoder()
    idx = 0
    documents: list[dict[str, object]] = []

    while idx < len(response):
        while idx < len(response) and response[idx].isspace():
            idx += 1
        if idx >= len(response):
            break
        try:
            payload, next_idx = decoder.raw_decode(response, idx)
        except json.JSONDecodeError as e:
            raise ParseError(f"Invalid JSON response: {e.msg}") from e
        idx = next_idx
        _collect_root_documents(payload, documents)

    return documents


def _collect_root_documents(
    payload: object, documents: list[dict[str, object]]
) -> None:
    if isinstance(payload, dict):
        documents.append(payload)
        return
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                documents.append(item)


def _build_groups(
    grouped_ranges: dict[tuple[str, ...], list[SentenceRange]],
    label_order: list[tuple[str, ...]],
) -> list[SentenceGroup]:
    groups: list[SentenceGroup] = []
    for label in label_order:
        merged_ranges = _merge_ranges(grouped_ranges[label])
        groups.append(SentenceGroup(label=label, ranges=tuple(merged_ranges)))

    if not groups:
        raise ParseError("No valid topic ranges found in response")

    return groups


def _normalize_label_parts(parts: Iterable[str]) -> tuple[str, ...]:
    """Normalize label parts to always use ">" as the hierarchy separator.

    Any ':' characters inside label segments are treated as sub-level separators.
    This keeps parsing tolerant of model outputs like "Apple: Mac Mini production"
    while canonicalizing final labels to ("Apple", "Mac Mini production").
    """
    normalized: list[str] = []
    for raw_part in parts:
        part = raw_part.strip()
        if not part:
            continue
        for sub_part in part.split(":"):
            sub = sub_part.strip()
            if sub:
                normalized.append(sub)
    return tuple(normalized)


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


def _clamp_range(start: int, end: int, max_index: int) -> SentenceRange | None:
    if max_index < 0:
        return None
    start = max(0, min(start, max_index))
    end = max(0, min(end, max_index))
    if start > end:
        start, end = end, start
    return SentenceRange(start=start, end=end)
