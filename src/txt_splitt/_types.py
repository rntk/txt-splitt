"""Data types for the text splitter pipeline."""

from dataclasses import dataclass


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
