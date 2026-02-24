"""Keyword extraction pipeline orchestrator."""

from __future__ import annotations

from typing import TYPE_CHECKING, final

from txt_splitt.keyword_types import Keyword, KeywordResult, Word
from txt_splitt.tracer import NoOpTracer

if TYPE_CHECKING:
    from txt_splitt.keyword_protocols import (
        KeywordGapValidatorStrategy,
        KeywordLLMStrategy,
        KeywordParser,
        WordMarkerStrategy,
        WordSplitter,
    )
    from txt_splitt.protocols import HtmlCleaner
    from txt_splitt.tracer import Tracer
    from txt_splitt.types import OffsetMapping


@final
class KeywordPipeline:
    """Orchestrates the keyword extraction pipeline.

    Stages:
    - Stage 0 (optional): HTML clean → offset mapping
    - Stage 1: Split text into words
    - Stage 2: Mark words → MarkedWords
    - Stage 3: LLM query → raw response string
    - Stage 4: Parse response → list of (start_idx, end_idx) pairs
    - Stage 5: Build Keyword objects from word offsets
    - Stage 6 (optional): Back-map offsets via mapping.to_original()
    """

    def __init__(
        self,
        *,
        word_splitter: WordSplitter,
        marker: WordMarkerStrategy,
        llm: KeywordLLMStrategy,
        parser: KeywordParser,
        html_cleaner: HtmlCleaner | None = None,
        gap_validator: KeywordGapValidatorStrategy | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self._word_splitter = word_splitter
        self._marker = marker
        self._llm = llm
        self._parser = parser
        self._html_cleaner = html_cleaner
        self._gap_validator = gap_validator
        self._tracer = tracer if tracer is not None else NoOpTracer()

    def run(self, text: str) -> KeywordResult:
        """Run the full keyword extraction pipeline on input text.

        Exceptions from any stage propagate directly to the caller.
        """
        with self._tracer.span("keyword_pipeline.run", text_length=len(text)):
            # Stage 0 (optional): Clean HTML tags
            mapping: OffsetMapping | None = None
            clean_text = text
            if self._html_cleaner is not None:
                with self._tracer.span("stage0.html_clean"):
                    clean_text, mapping = self._html_cleaner.clean(text)

            # Stage 1: Split into words
            with self._tracer.span("stage1.word_split") as s1:
                words = self._word_splitter.split(clean_text)
                s1.attributes["word_count"] = len(words)

            # Stage 2: Apply word markers
            with self._tracer.span("stage2.mark_words") as s2:
                marked = self._marker.mark(clean_text, words)
                s2.attributes["tagged_length"] = len(marked.tagged_text)

            # Stage 3: Query LLM
            with self._tracer.span("stage3.llm_query") as s3:
                response = self._llm.query(marked)
                s3.attributes["response"] = response

            # Stage 4: Parse response → list of (start_idx, end_idx) pairs
            with self._tracer.span("stage4.parse") as s4:
                index_ranges = self._parser.parse(response, marked.word_count)
                s4.attributes["keyword_count"] = len(index_ranges)

            # Stage 5: Build Keyword objects from word offsets
            with self._tracer.span("stage5.build_keywords"):
                keywords = _build_keywords(index_ranges, words, clean_text)

            # Stage 5.5 (optional): Gap validation and self-healing
            if self._gap_validator is not None:
                with self._tracer.span("stage5_5.gap_validate") as s55:
                    keywords = self._gap_validator.validate(keywords, words, clean_text)
                    s55.attributes["keyword_count_after_healing"] = len(keywords)

            # Stage 6 (optional): Back-map offsets via mapping.to_original()
            if mapping is not None:
                with self._tracer.span("stage6.remap_offsets"):
                    keywords = _remap_keywords(keywords, mapping)
                    words = _remap_words(words, mapping)

        return KeywordResult(
            keywords=tuple(keywords),
            words=tuple(words),
        )


def _build_keywords(
    index_ranges: list[tuple[int, int]],
    words: list[Word],
    text: str,
) -> list[Keyword]:
    """Build Keyword objects from (start_idx, end_idx) pairs and word list."""
    keywords: list[Keyword] = []
    for start_idx, end_idx in index_ranges:
        if start_idx >= len(words) or end_idx >= len(words):
            continue
        start_char = words[start_idx].start
        end_char = words[end_idx].end
        keyword_text = text[start_char:end_char]
        keywords.append(Keyword(text=keyword_text, start=start_char, end=end_char))
    return keywords


def _remap_keywords(
    keywords: list[Keyword],
    mapping: OffsetMapping,
) -> list[Keyword]:
    """Remap keyword char offsets from clean-text to original-text positions."""
    remapped: list[Keyword] = []
    for kw in keywords:
        orig_start = mapping.to_original(kw.start)
        orig_end = mapping.to_original(kw.end)
        remapped.append(Keyword(text=kw.text, start=orig_start, end=orig_end))
    return remapped


def _remap_words(
    words: list[Word],
    mapping: OffsetMapping,
) -> list[Word]:
    """Remap word char offsets from clean-text to original-text positions."""
    remapped: list[Word] = []
    for w in words:
        orig_start = mapping.to_original(w.start)
        orig_end = mapping.to_original(w.end)
        remapped.append(
            Word(index=w.index, start=orig_start, end=orig_end, text=w.text)
        )
    return remapped
