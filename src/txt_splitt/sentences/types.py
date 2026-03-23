"""Sentence-oriented data types."""

from dataclasses import dataclass

from txt_splitt.types import OffsetMapping, OffsetSegment

__all__ = [
    "MarkedText",
    "OffsetMapping",
    "OffsetSegment",
    "PreparedChunk",
    "PreparedDocument",
    "Sentence",
    "SentenceGroup",
    "SentenceRange",
    "SplitResult",
    "_indices_to_ranges",
]


@dataclass(frozen=True, slots=True)
class Sentence:
    """A sentence extracted from source text."""

    index: int  # 0-based position
    start: int  # char offset in source text
    end: int  # char offset (exclusive, slice convention)
    text: str  # the actual sentence text


@dataclass(frozen=True, slots=True)
class MarkedText:
    """Text with sentence markers applied."""

    tagged_text: str  # formatted string with {N} markers
    sentence_count: int


@dataclass(frozen=True, slots=True)
class PreparedChunk:
    """A chunk of marked text ready to send to an LLM."""

    chunk_id: int
    tagged_text: str
    sentence_count: int
    marker_start: int | None
    marker_end: int | None


@dataclass(frozen=True, slots=True)
class PreparedDocument:
    """Prepared single-document state for two-stage pipeline processing."""

    original_text: str
    prepared_text: str
    sentences: tuple[Sentence, ...]
    marked_text: MarkedText
    chunks: tuple[PreparedChunk, ...]
    offset_mapping: OffsetMapping | None


@dataclass(frozen=True, slots=True)
class SentenceRange:
    """A range of sentence indices (both inclusive)."""

    start: int  # 0-based sentence index (inclusive)
    end: int  # 0-based sentence index (inclusive)


@dataclass(frozen=True, slots=True)
class SentenceGroup:
    """A group of sentences sharing a topic label."""

    label: tuple[str, ...]  # e.g. ("Technology", "AI", "GPT-4")
    ranges: tuple[SentenceRange, ...]


@dataclass(frozen=True, slots=True)
class SplitResult:
    """Final result of the text splitting pipeline."""

    sentences: tuple[Sentence, ...]
    groups: tuple[SentenceGroup, ...]


def _indices_to_ranges(indices: list[int]) -> list[SentenceRange]:
    """Convert sorted sentence indices into minimal contiguous ranges."""
    if not indices:
        return []
    ranges: list[SentenceRange] = []
    start = indices[0]
    end = indices[0]
    for idx in indices[1:]:
        if idx == end + 1:
            end = idx
        else:
            ranges.append(SentenceRange(start=start, end=end))
            start = idx
            end = idx
    ranges.append(SentenceRange(start=start, end=end))
    return ranges
