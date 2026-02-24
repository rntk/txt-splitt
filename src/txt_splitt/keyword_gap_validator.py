"""Gap validator with self-healing for the keyword extraction pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from txt_splitt.keyword_pipeline import _build_keywords
from txt_splitt.keyword_types import Keyword, Word
from txt_splitt.tracer import NoOpTracer

if TYPE_CHECKING:
    from txt_splitt.keyword_protocols import (
        KeywordLLMStrategy,
        KeywordParser,
        WordMarkerStrategy,
        WordSplitter,
    )
    from txt_splitt.tracer import Tracer


class KeywordGapValidator:
    """Detects large uncovered gaps in keyword extraction and re-queries the LLM.

    After the main pipeline extracts keywords, this validator:
    1. Finds contiguous runs of words not covered by any keyword.
    2. If a run has >= min_gap_words words, re-runs stages 1-5 on that text slice.
    3. Merges newly found keywords into the existing result (deduplicating by overlap).
    """

    def __init__(
        self,
        *,
        word_splitter: WordSplitter,
        marker: WordMarkerStrategy,
        llm: KeywordLLMStrategy,
        parser: KeywordParser,
        min_gap_words: int = 20,
        tracer: Tracer | None = None,
    ) -> None:
        if min_gap_words < 1:
            raise ValueError("min_gap_words must be >= 1")
        self._word_splitter = word_splitter
        self._marker = marker
        self._llm = llm
        self._parser = parser
        self.min_gap_words = min_gap_words
        self._tracer = tracer if tracer is not None else NoOpTracer()

    def validate(
        self,
        keywords: list[Keyword],
        words: list[Word],
        text: str,
    ) -> list[Keyword]:
        """Return an updated keyword list after healing coverage gaps.

        Args:
            keywords: Keywords found so far (char offsets into *text*).
            words: All words in *text* (from the word splitter).
            text: The clean source text.

        Returns:
            Merged keyword list (original + any newly found ones), sorted by start.
        """
        gaps = _find_gap_word_ranges(keywords, words)
        large_gaps = [g for g in gaps if len(g) >= self.min_gap_words]

        if not large_gaps:
            return keywords

        new_keywords: list[Keyword] = []
        for gap_words in large_gaps:
            gap_start_char = gap_words[0].start
            gap_end_char = gap_words[-1].end
            gap_text = text[gap_start_char:gap_end_char]

            with self._tracer.span(
                "gap_validator.heal_gap",
                gap_word_count=len(gap_words),
                gap_start=gap_start_char,
                gap_end=gap_end_char,
            ) as span:
                # Re-run stages 1-5 on the gap region
                sub_words = self._word_splitter.split(gap_text)
                if not sub_words:
                    continue

                marked = self._marker.mark(gap_text, sub_words)
                response = self._llm.query(marked)
                index_ranges = self._parser.parse(response, marked.word_count)
                found = _build_keywords(index_ranges, sub_words, gap_text)

                span.attributes["llm_response"] = response
                span.attributes["new_keywords_found"] = len(found)

                # Shift char offsets from gap-local to full-text coordinates
                for kw in found:
                    new_keywords.append(
                        Keyword(
                            text=kw.text,
                            start=kw.start + gap_start_char,
                            end=kw.end + gap_start_char,
                        )
                    )

        if not new_keywords:
            return keywords

        merged = _merge_keywords(keywords, new_keywords)
        return merged


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_gap_word_ranges(
    keywords: list[Keyword],
    words: list[Word],
) -> list[list[Word]]:
    """Return contiguous runs of words not covered by any keyword.

    A word is considered *covered* if its char range overlaps with any keyword's
    char range, or if the word is fully contained within a keyword span.
    Adjacency check: a keyword that spans the entire gap explains the gap, so
    words that fall inside a long keyword are not considered uncovered.
    """
    if not words:
        return []

    # Build a set of covered word indices using keyword char spans
    covered: set[int] = set()
    for kw in keywords:
        for w in words:
            # Word overlaps keyword if not (w.end <= kw.start or w.start >= kw.end)
            if not (w.end <= kw.start or w.start >= kw.end):
                covered.add(w.index)

    # Collect contiguous runs of uncovered words
    gaps: list[list[Word]] = []
    current: list[Word] = []
    for w in words:
        if w.index not in covered:
            current.append(w)
        else:
            if current:
                gaps.append(current)
                current = []
    if current:
        gaps.append(current)

    return gaps


def _merge_keywords(
    existing: list[Keyword],
    new: list[Keyword],
) -> list[Keyword]:
    """Merge two keyword lists, dropping new keywords that overlap existing ones."""
    result = list(existing)
    for candidate in new:
        overlaps = any(
            not (candidate.end <= kw.start or candidate.start >= kw.end)
            for kw in result
        )
        if not overlaps:
            result.append(candidate)
    result.sort(key=lambda k: k.start)
    return result
