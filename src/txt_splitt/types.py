"""Shared data types for the package."""

from __future__ import annotations

import bisect
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OffsetSegment:
    """A contiguous text segment preserved during HTML cleaning."""

    clean_offset: int
    original_offset: int
    length: int


@dataclass(frozen=True, slots=True)
class OffsetMapping:
    """Mapping between clean-text and original-text positions."""

    segments: tuple[OffsetSegment, ...]
    original_length: int
    clean_length: int

    def to_original(self, clean_pos: int) -> int:
        """Map a position in clean text back to the original text."""
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

        offsets = [segment.clean_offset for segment in self.segments]
        idx = bisect.bisect_right(offsets, clean_pos) - 1
        segment = self.segments[idx]
        return segment.original_offset + (clean_pos - segment.clean_offset)
