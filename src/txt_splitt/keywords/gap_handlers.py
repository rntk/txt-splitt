"""Gap handlers for keyword extraction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from txt_splitt.keywords.results import build_keywords
from txt_splitt.keywords.types import Keyword, Word
from txt_splitt.tracer import NoOpTracer

if TYPE_CHECKING:
    from txt_splitt.keywords.protocols import (
        KeywordLLMStrategy,
        KeywordParser,
        WordMarkerStrategy,
        WordSplitter,
    )
    from txt_splitt.tracer import Tracer


class RepairingGapHandler:
    """Detect large uncovered gaps and re-query the LLM on those regions."""

    def __init__(
        self,
        *,
        splitter: WordSplitter,
        marker: WordMarkerStrategy,
        llm: KeywordLLMStrategy,
        parser: KeywordParser,
        min_gap_words: int = 20,
        tracer: Tracer | None = None,
    ) -> None:
        if min_gap_words < 1:
            raise ValueError("min_gap_words must be >= 1")
        self._splitter = splitter
        self._marker = marker
        self._llm = llm
        self._parser = parser
        self.min_gap_words = min_gap_words
        self._tracer = tracer if tracer is not None else NoOpTracer()

    def handle(
        self,
        keywords: list[Keyword],
        words: list[Word],
        text: str,
    ) -> list[Keyword]:
        gaps = _find_gap_word_ranges(keywords, words)
        large_gaps = [gap for gap in gaps if len(gap) >= self.min_gap_words]
        if not large_gaps:
            return keywords

        new_keywords: list[Keyword] = []
        for gap_words in large_gaps:
            gap_start_char = gap_words[0].start
            gap_end_char = gap_words[-1].end
            gap_text = text[gap_start_char:gap_end_char]

            with self._tracer.span(
                "gap_handler.heal_gap",
                gap_word_count=len(gap_words),
                gap_start=gap_start_char,
                gap_end=gap_end_char,
            ) as span:
                sub_words = self._splitter.split(gap_text)
                if not sub_words:
                    continue

                marked = self._marker.mark(gap_text, sub_words)
                response = self._llm.query(marked)
                index_ranges = self._parser.parse(response, marked.word_count)
                found = build_keywords(index_ranges, sub_words, gap_text)
                span.attributes["llm_response"] = response
                span.attributes["new_keywords_found"] = len(found)

            for keyword in found:
                new_keywords.append(
                    Keyword(
                        text=keyword.text,
                        start=keyword.start + gap_start_char,
                        end=keyword.end + gap_start_char,
                    )
                )

        if not new_keywords:
            return keywords
        return _merge_keywords(keywords, new_keywords)

    def validate(
        self,
        keywords: list[Keyword],
        words: list[Word],
        text: str,
    ) -> list[Keyword]:
        """Backward-compatible alias for the pre-refactor method name."""
        return self.handle(keywords, words, text)


GapHandler = RepairingGapHandler


def _find_gap_word_ranges(
    keywords: list[Keyword],
    words: list[Word],
) -> list[list[Word]]:
    """Return contiguous runs of words not covered by any keyword."""
    if not words:
        return []

    covered: set[int] = set()
    for keyword in keywords:
        for word in words:
            if not (word.end <= keyword.start or word.start >= keyword.end):
                covered.add(word.index)

    gaps: list[list[Word]] = []
    current: list[Word] = []
    for word in words:
        if word.index not in covered:
            current.append(word)
        elif current:
            gaps.append(current)
            current = []
    if current:
        gaps.append(current)
    return gaps


def _merge_keywords(existing: list[Keyword], new: list[Keyword]) -> list[Keyword]:
    """Merge keyword lists, dropping new keywords that overlap existing ones."""
    result = list(existing)
    for candidate in new:
        overlaps = any(
            not (candidate.end <= keyword.start or candidate.start >= keyword.end)
            for keyword in result
        )
        if not overlaps:
            result.append(candidate)
    result.sort(key=lambda keyword: keyword.start)
    return result
