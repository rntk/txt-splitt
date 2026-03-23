"""Tests for the keyword extraction pipeline."""

from __future__ import annotations

import re

import pytest

from txt_splitt.errors import ParseError
from txt_splitt.html_cleaners import TagStripCleaner
from txt_splitt.keywords.builders import build_pipeline
from txt_splitt.keywords.chunkers import WordOverlapChunker, _split_tokens
from txt_splitt.keywords.gap_handlers import (
    RepairingGapHandler,
    _find_gap_word_ranges,
    _merge_keywords,
)
from txt_splitt.keywords.llm import _merge_responses
from txt_splitt.keywords.markers import WordBracketMarker
from txt_splitt.keywords.parsers import KeywordIndexParser
from txt_splitt.keywords.splitters import RegexWordSplitter
from txt_splitt.keywords.types import Keyword, MarkedWords, Word

# ---------------------------------------------------------------------------
# Word splitting
# ---------------------------------------------------------------------------


class TestRegexWordSplitter:
    def test_basic_split(self) -> None:
        splitter = RegexWordSplitter()
        words = splitter.split("hello world foo")
        assert len(words) == 3
        assert words[0] == Word(index=0, start=0, end=5, text="hello")
        assert words[1] == Word(index=1, start=6, end=11, text="world")
        assert words[2] == Word(index=2, start=12, end=15, text="foo")

    def test_offsets_with_extra_spaces(self) -> None:
        splitter = RegexWordSplitter()
        words = splitter.split("  hello   world  ")
        assert len(words) == 2
        assert words[0].start == 2
        assert words[0].end == 7
        assert words[1].start == 10
        assert words[1].end == 15

    def test_empty_string(self) -> None:
        splitter = RegexWordSplitter()
        assert splitter.split("") == []

    def test_single_word(self) -> None:
        splitter = RegexWordSplitter()
        words = splitter.split("python")
        assert len(words) == 1
        assert words[0] == Word(index=0, start=0, end=6, text="python")

    def test_indices_are_sequential(self) -> None:
        splitter = RegexWordSplitter()
        words = splitter.split("a b c d e")
        assert [w.index for w in words] == [0, 1, 2, 3, 4]

    def test_text_slices_match(self) -> None:
        text = "machine learning is cool"
        splitter = RegexWordSplitter()
        words = splitter.split(text)
        for w in words:
            assert text[w.start : w.end] == w.text


# ---------------------------------------------------------------------------
# Word marking
# ---------------------------------------------------------------------------


class TestWordBracketMarker:
    def test_basic_mark(self) -> None:
        marker = WordBracketMarker()
        words = [
            Word(index=0, start=0, end=5, text="hello"),
            Word(index=1, start=6, end=11, text="world"),
        ]
        result = marker.mark("hello world", words)
        assert result.tagged_text == "{0} hello {1} world"
        assert result.word_count == 2

    def test_empty_words(self) -> None:
        marker = WordBracketMarker()
        result = marker.mark("", [])
        assert result.tagged_text == ""
        assert result.word_count == 0

    def test_single_word(self) -> None:
        marker = WordBracketMarker()
        words = [Word(index=0, start=0, end=4, text="test")]
        result = marker.mark("test", words)
        assert result.tagged_text == "{0} test"

    def test_marker_format(self) -> None:
        marker = WordBracketMarker()
        words = [
            Word(index=0, start=0, end=1, text="a"),
            Word(index=1, start=2, end=3, text="b"),
            Word(index=2, start=4, end=5, text="c"),
        ]
        result = marker.mark("a b c", words)
        assert "{0}" in result.tagged_text
        assert "{1}" in result.tagged_text
        assert "{2}" in result.tagged_text


# ---------------------------------------------------------------------------
# Index parsing
# ---------------------------------------------------------------------------


class TestKeywordIndexParser:
    def test_single_indices(self) -> None:
        parser = KeywordIndexParser()
        result = parser.parse("0, 3, 7", word_count=10)
        assert result == [(0, 0), (3, 3), (7, 7)]

    def test_range(self) -> None:
        parser = KeywordIndexParser()
        result = parser.parse("7-10", word_count=15)
        assert result == [(7, 10)]

    def test_mixed(self) -> None:
        parser = KeywordIndexParser()
        result = parser.parse("0, 3, 7-10, 15", word_count=20)
        assert result == [(0, 0), (3, 3), (7, 10), (15, 15)]

    def test_deduplication(self) -> None:
        parser = KeywordIndexParser()
        result = parser.parse("3, 3, 5, 5", word_count=10)
        assert result == [(3, 3), (5, 5)]

    def test_out_of_bounds_filtered(self) -> None:
        parser = KeywordIndexParser()
        result = parser.parse("0, 5, 100", word_count=10)
        assert result == [(0, 0), (5, 5)]

    def test_empty_response(self) -> None:
        parser = KeywordIndexParser()
        result = parser.parse("", word_count=10)
        assert result == []

    def test_no_numbers(self) -> None:
        parser = KeywordIndexParser()
        result = parser.parse("no keywords found", word_count=10)
        assert result == []

    def test_sorted_output(self) -> None:
        parser = KeywordIndexParser()
        result = parser.parse("7, 2, 5", word_count=10)
        assert result == [(2, 2), (5, 5), (7, 7)]

    def test_negative_word_count_raises(self) -> None:
        parser = KeywordIndexParser()
        with pytest.raises(ParseError):
            parser.parse("0", word_count=-1)

    def test_range_reversed_normalized(self) -> None:
        parser = KeywordIndexParser()
        result = parser.parse("10-7", word_count=15)
        assert result == [(7, 10)]

    def test_range_at_boundary(self) -> None:
        parser = KeywordIndexParser()
        # end == word_count-1 is valid
        result = parser.parse("8-9", word_count=10)
        assert result == [(8, 9)]

    def test_range_exceeds_boundary(self) -> None:
        parser = KeywordIndexParser()
        # end >= word_count is invalid
        result = parser.parse("8-10", word_count=10)
        assert result == []


# ---------------------------------------------------------------------------
# End-to-end pipeline with mock LLM
# ---------------------------------------------------------------------------


class MockLLMCallable:
    def __init__(self, response: str) -> None:
        self._response = response

    def call(self, prompt: str, temperature: float) -> str:
        return self._response


class MockKeywordLLM:
    def __init__(self, response: str) -> None:
        self._response = response

    def query(self, marked: MarkedWords) -> str:
        return self._response


class TestKeywordPipeline:
    def test_basic_pipeline(self) -> None:
        pipeline = build_pipeline(
            splitter=RegexWordSplitter(),
            marker=WordBracketMarker(),
            llm=MockKeywordLLM("0, 2"),
            parser=KeywordIndexParser(),
        )
        result = pipeline.run("hello world foo")
        assert len(result.words) == 3
        assert len(result.keywords) == 2
        kw_texts = {kw.text for kw in result.keywords}
        assert "hello" in kw_texts
        assert "foo" in kw_texts

    def test_pipeline_with_phrase(self) -> None:
        pipeline = build_pipeline(
            splitter=RegexWordSplitter(),
            marker=WordBracketMarker(),
            llm=MockKeywordLLM("0-1"),
            parser=KeywordIndexParser(),
        )
        result = pipeline.run("machine learning rocks")
        assert len(result.keywords) == 1
        assert result.keywords[0].text == "machine learning"

    def test_pipeline_empty_response(self) -> None:
        pipeline = build_pipeline(
            splitter=RegexWordSplitter(),
            marker=WordBracketMarker(),
            llm=MockKeywordLLM(""),
            parser=KeywordIndexParser(),
        )
        result = pipeline.run("hello world")
        assert len(result.keywords) == 0
        assert len(result.words) == 2

    def test_pipeline_keyword_offsets(self) -> None:
        text = "hello world"
        pipeline = build_pipeline(
            splitter=RegexWordSplitter(),
            marker=WordBracketMarker(),
            llm=MockKeywordLLM("1"),
            parser=KeywordIndexParser(),
        )
        result = pipeline.run(text)
        assert len(result.keywords) == 1
        kw = result.keywords[0]
        assert kw.text == "world"
        assert text[kw.start : kw.end] == "world"

    def test_pipeline_preserves_words(self) -> None:
        text = "the quick brown fox"
        pipeline = build_pipeline(
            splitter=RegexWordSplitter(),
            marker=WordBracketMarker(),
            llm=MockKeywordLLM("1, 3"),
            parser=KeywordIndexParser(),
        )
        result = pipeline.run(text)
        assert len(result.words) == 4
        assert result.words[0].text == "the"
        assert result.words[3].text == "fox"


# ---------------------------------------------------------------------------
# HTML cleaning + offset back-mapping
# ---------------------------------------------------------------------------


class TestKeywordPipelineHtml:
    def test_html_offset_backmap(self) -> None:
        html = "<b>hello</b> world"
        pipeline = build_pipeline(
            splitter=RegexWordSplitter(),
            marker=WordBracketMarker(),
            llm=MockKeywordLLM("0"),
            parser=KeywordIndexParser(),
            html_cleaner=TagStripCleaner(),
        )
        result = pipeline.run(html)
        # "hello world" after cleaning
        assert len(result.keywords) == 1
        kw = result.keywords[0]
        assert kw.text == "hello"
        # start offset maps back to where "hello" begins in the original HTML
        assert kw.start == html.index("hello")
        assert html[kw.start] == "h"

    def test_html_multi_keyword(self) -> None:
        html = "<p>machine learning</p>"
        pipeline = build_pipeline(
            splitter=RegexWordSplitter(),
            marker=WordBracketMarker(),
            llm=MockKeywordLLM("0-1"),
            parser=KeywordIndexParser(),
            html_cleaner=TagStripCleaner(),
        )
        result = pipeline.run(html)
        assert len(result.keywords) == 1
        kw = result.keywords[0]
        assert kw.text == "machine learning"


# ---------------------------------------------------------------------------
# Chunker tests
# ---------------------------------------------------------------------------


class TestWordOverlapChunker:
    def test_no_chunking_needed(self) -> None:
        chunker = WordOverlapChunker(max_chars=1000)
        marked = MarkedWords(tagged_text="{0} hello {1} world", word_count=2)
        result = chunker.chunk(marked)
        assert result == [marked]

    def test_split_tokens(self) -> None:
        tokens = _split_tokens("{0} hello {1} world {2} foo")
        assert tokens == ["{0} hello", "{1} world", "{2} foo"]

    def test_split_tokens_empty(self) -> None:
        tokens = _split_tokens("")
        assert tokens == []

    def test_chunking_produces_multiple(self) -> None:
        inline = " ".join(f"{{{i}}} word{i}" for i in range(50))
        marked = MarkedWords(tagged_text=inline, word_count=50)
        chunker = WordOverlapChunker(max_chars=100, overlap_words=2)
        chunks = chunker.chunk(marked)
        assert len(chunks) > 1


# ---------------------------------------------------------------------------
# Merge responses helper
# ---------------------------------------------------------------------------


class TestMergeResponses:
    def test_merge_deduplicates(self) -> None:
        result = _merge_responses(["0, 1, 2", "2, 3, 4"])
        # 2 appears in both but should appear once
        tokens = [t.strip() for t in result.split(",")]
        assert tokens.count("2") == 1

    def test_merge_single(self) -> None:
        result = _merge_responses(["0, 1, 2"])
        assert result == "0, 1, 2"

    def test_merge_with_ranges(self) -> None:
        result = _merge_responses(["0-2, 5", "5, 7-9"])
        assert "0-2" in result or "0 - 2" in result or "0-2" in result.replace(" ", "")
        tokens_normalized = re.sub(r"\s+", "", result)
        assert "5" in tokens_normalized


# ---------------------------------------------------------------------------
# Gap validator helpers
# ---------------------------------------------------------------------------


def _make_words(texts: list[str]) -> tuple[str, list[Word]]:
    """Build a joined text and corresponding Word list for testing."""
    full_text = " ".join(texts)
    words: list[Word] = []
    offset = 0
    for idx, t in enumerate(texts):
        words.append(Word(index=idx, start=offset, end=offset + len(t), text=t))
        offset += len(t) + 1  # +1 for space
    return full_text, words


class TestFindGapWordRanges:
    def test_no_keywords_all_gap(self) -> None:
        _, words = _make_words(["a", "b", "c"])
        gaps = _find_gap_word_ranges([], words)
        assert len(gaps) == 1
        assert len(gaps[0]) == 3

    def test_all_covered(self) -> None:
        text, words = _make_words(["hello", "world"])
        kws = [Keyword(text="hello world", start=0, end=len(text))]
        gaps = _find_gap_word_ranges(kws, words)
        assert gaps == []

    def test_gap_in_middle(self) -> None:
        text, words = _make_words(["a", "b", "c", "d", "e"])
        # cover first and last word only
        kws = [
            Keyword(text="a", start=words[0].start, end=words[0].end),
            Keyword(text="e", start=words[4].start, end=words[4].end),
        ]
        gaps = _find_gap_word_ranges(kws, words)
        assert len(gaps) == 1
        assert len(gaps[0]) == 3  # b, c, d

    def test_empty_words(self) -> None:
        assert _find_gap_word_ranges([], []) == []


class TestMergeKeywords:
    def test_no_overlap_adds_all(self) -> None:
        existing = [Keyword(text="a", start=0, end=1)]
        new = [Keyword(text="b", start=10, end=11)]
        result = _merge_keywords(existing, new)
        assert len(result) == 2

    def test_overlapping_new_dropped(self) -> None:
        existing = [Keyword(text="hello", start=0, end=5)]
        new = [Keyword(text="hell", start=0, end=4)]  # overlaps
        result = _merge_keywords(existing, new)
        assert len(result) == 1
        assert result[0].text == "hello"

    def test_result_sorted_by_start(self) -> None:
        existing = [Keyword(text="z", start=10, end=11)]
        new = [Keyword(text="a", start=0, end=1)]
        result = _merge_keywords(existing, new)
        assert result[0].start == 0
        assert result[1].start == 10


# ---------------------------------------------------------------------------
# RepairingGapHandler end-to-end
# ---------------------------------------------------------------------------


def _make_gap_llm(response: str) -> MockKeywordLLM:
    return MockKeywordLLM(response)


class TestRepairingGapHandler:
    def _build_text_and_words(self, n: int) -> tuple[str, list[Word]]:
        """Build a text of n single-char words: 'w0 w1 w2 ...'"""
        labels = [f"w{i}" for i in range(n)]
        return _make_words(labels)

    def test_no_gap_below_threshold(self) -> None:
        text, words = self._build_text_and_words(10)
        # cover words 0 and 9, gap in between = 8 words (below 20)
        kws = [
            Keyword(text=words[0].text, start=words[0].start, end=words[0].end),
            Keyword(text=words[9].text, start=words[9].start, end=words[9].end),
        ]
        validator = RepairingGapHandler(
            splitter=RegexWordSplitter(),
            marker=WordBracketMarker(),
            llm=_make_gap_llm("0"),  # would add one keyword if called
            parser=KeywordIndexParser(),
            min_gap_words=20,
        )
        result = validator.validate(kws, words, text)
        # Should be unchanged — gap is only 8 words
        assert len(result) == 2

    def test_large_gap_triggers_requery(self) -> None:
        text, words = self._build_text_and_words(30)
        # cover only word 0; words 1-29 form a gap of 29 words
        kws = [Keyword(text=words[0].text, start=words[0].start, end=words[0].end)]
        # The re-query LLM will say "0" (first word of the gap sub-text = words[1])
        validator = RepairingGapHandler(
            splitter=RegexWordSplitter(),
            marker=WordBracketMarker(),
            llm=_make_gap_llm("0"),
            parser=KeywordIndexParser(),
            min_gap_words=20,
        )
        result = validator.validate(kws, words, text)
        # Should have gained at least one new keyword from the gap
        assert len(result) > 1

    def test_pipeline_integration_with_gap_validator(self) -> None:
        """Full pipeline run: main LLM covers only first word, validator heals gap."""
        labels = [f"w{i}" for i in range(30)]
        text = " ".join(labels)

        main_llm = MockKeywordLLM("0")  # covers only w0
        heal_llm = MockKeywordLLM("0")  # covers first word of each gap slice

        validator = RepairingGapHandler(
            splitter=RegexWordSplitter(),
            marker=WordBracketMarker(),
            llm=heal_llm,
            parser=KeywordIndexParser(),
            min_gap_words=20,
        )
        pipeline = build_pipeline(
            splitter=RegexWordSplitter(),
            marker=WordBracketMarker(),
            llm=main_llm,
            parser=KeywordIndexParser(),
            gap_handler=validator,
        )
        result = pipeline.run(text)
        assert len(result.keywords) > 1

    def test_invalid_min_gap_words_raises(self) -> None:
        with pytest.raises(ValueError):
            RepairingGapHandler(
                splitter=RegexWordSplitter(),
                marker=WordBracketMarker(),
                llm=MockKeywordLLM(""),
                parser=KeywordIndexParser(),
                min_gap_words=0,
            )
