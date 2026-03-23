"""Keyword pipeline result helpers."""

from __future__ import annotations

from txt_splitt.keywords.types import Keyword, KeywordResult, Word
from txt_splitt.types import OffsetMapping


def build_keywords(
    index_ranges: list[tuple[int, int]],
    words: list[Word],
    text: str,
) -> list[Keyword]:
    """Build Keyword objects from parsed word-index ranges."""
    keywords: list[Keyword] = []
    for start_idx, end_idx in index_ranges:
        if start_idx >= len(words) or end_idx >= len(words):
            continue
        start_char = words[start_idx].start
        end_char = words[end_idx].end
        keywords.append(
            Keyword(text=text[start_char:end_char], start=start_char, end=end_char)
        )
    return keywords


def remap_keywords(
    keywords: list[Keyword],
    mapping: OffsetMapping,
) -> list[Keyword]:
    """Remap keyword offsets from clean text back to the original text."""
    return [
        Keyword(
            text=keyword.text,
            start=mapping.to_original(keyword.start),
            end=mapping.to_original(keyword.end),
        )
        for keyword in keywords
    ]


def remap_words(
    words: list[Word],
    mapping: OffsetMapping,
) -> list[Word]:
    """Remap word offsets from clean text back to the original text."""
    return [
        Word(
            index=word.index,
            start=mapping.to_original(word.start),
            end=mapping.to_original(word.end),
            text=word.text,
        )
        for word in words
    ]


def build_result(
    keywords: list[Keyword],
    words: list[Word],
) -> KeywordResult:
    """Build the final keyword result object."""
    return KeywordResult(keywords=tuple(keywords), words=tuple(words))
