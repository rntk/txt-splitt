"""Data types for the keyword extraction pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Word:
    """A word extracted from source text."""

    index: int  # 0-based position
    start: int  # char offset in source text (inclusive)
    end: int  # char offset in source text (exclusive, slice convention)
    text: str  # the actual word text


@dataclass(frozen=True, slots=True)
class MarkedWords:
    """Text with word markers applied."""

    tagged_text: str  # formatted string with {N} markers inline
    word_count: int


@dataclass(frozen=True, slots=True)
class Keyword:
    """A single extracted keyword or phrase."""

    text: str  # the keyword text (may span multiple words)
    start: int  # char offset in source text (inclusive)
    end: int  # char offset in source text (exclusive, slice convention)


@dataclass(frozen=True, slots=True)
class KeywordResult:
    """Final result of the keyword extraction pipeline."""

    keywords: tuple[Keyword, ...]
    words: tuple[Word, ...]
