"""Data types for the text splitter pipeline."""

from __future__ import annotations

import bisect
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


@dataclass(frozen=True, slots=True)
class OffsetSegment:
    """A contiguous text segment preserved during HTML cleaning.

    Maps a range in clean text back to the corresponding range in original text.
    """

    clean_offset: int
    original_offset: int
    length: int


@dataclass(frozen=True, slots=True)
class OffsetMapping:
    """Mapping between clean-text and original-HTML-text positions.

    Built from a sequence of ``OffsetSegment`` instances representing the
    non-tag portions of the original text.
    """

    segments: tuple[OffsetSegment, ...]
    original_length: int
    clean_length: int

    def to_original(self, clean_pos: int) -> int:
        """Map a position in clean text to the corresponding original-text position.

        For positions at segment boundaries the result points to the start of
        the corresponding original segment, so sentence spans may include HTML
        tags that fall within their range.
        """
        if clean_pos < 0:
            msg = f"clean_pos must be non-negative, got {clean_pos}"
            raise ValueError(msg)
        if clean_pos > self.clean_length:
            msg = f"clean_pos {clean_pos} exceeds clean_length {self.clean_length}"
            raise ValueError(msg)
        if clean_pos == self.clean_length:
            return self.original_length
        if not self.segments:
            msg = "cannot map position: no segments in mapping"
            raise ValueError(msg)

        offsets = [seg.clean_offset for seg in self.segments]
        idx = bisect.bisect_right(offsets, clean_pos) - 1
        seg = self.segments[idx]
        return seg.original_offset + (clean_pos - seg.clean_offset)
