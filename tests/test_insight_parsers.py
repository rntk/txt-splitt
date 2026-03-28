"""Tests for InsightParser."""

import pytest

from txt_splitt.errors import ParseError
from txt_splitt.insights.parsers import InsightParser
from txt_splitt.sentences.types import SentenceRange


class TestInsightParserText:
    def setup_method(self) -> None:
        self.parser = InsightParser()

    def test_zero_sentence_count_raises(self) -> None:
        with pytest.raises(ParseError, match="sentence_count must be positive"):
            self.parser.parse("Insight: 0-5", sentence_count=0)

    def test_single_insight_single_range(self) -> None:
        result = self.parser.parse("Context Compaction at 50K: 0-5", sentence_count=10)
        assert len(result) == 1
        assert result[0].name == "Context Compaction at 50K"
        assert result[0].ranges == (SentenceRange(start=0, end=5),)

    def test_multiple_insights(self) -> None:
        response = "First Insight: 0-5\nSecond Insight: 6-9"
        result = self.parser.parse(response, sentence_count=10)
        assert len(result) == 2
        assert result[0].name == "First Insight"
        assert result[1].name == "Second Insight"

    def test_multiple_ranges_per_insight(self) -> None:
        result = self.parser.parse("My Insight: 0-5, 10-15", sentence_count=20)
        assert len(result) == 1
        assert result[0].ranges == (
            SentenceRange(start=0, end=5),
            SentenceRange(start=10, end=15),
        )

    def test_single_sentence_number(self) -> None:
        result = self.parser.parse("My Insight: 7", sentence_count=10)
        assert result[0].ranges == (SentenceRange(start=7, end=7),)

    def test_mixed_ranges(self) -> None:
        result = self.parser.parse("My Insight: 2, 4, 6-9", sentence_count=10)
        assert result[0].ranges == (
            SentenceRange(start=2, end=2),
            SentenceRange(start=4, end=4),
            SentenceRange(start=6, end=9),
        )

    def test_same_name_merges_across_lines(self) -> None:
        """Same name on separate lines (e.g. different chunks) merges."""
        response = "Key Finding: 0-3\nKey Finding: 10-12"
        result = self.parser.parse(response, sentence_count=20)
        assert len(result) == 1
        assert result[0].name == "Key Finding"
        assert result[0].ranges == (
            SentenceRange(start=0, end=3),
            SentenceRange(start=10, end=12),
        )

    def test_name_normalization_for_merge(self) -> None:
        """Merging is case-insensitive and whitespace-normalized."""
        response = "Key Finding: 0-3\nkey finding: 10-12"
        result = self.parser.parse(response, sentence_count=20)
        assert len(result) == 1
        assert result[0].ranges == (
            SentenceRange(start=0, end=3),
            SentenceRange(start=10, end=12),
        )

    def test_adjacent_ranges_coalesced_on_merge(self) -> None:
        response = "Key Finding: 0-3\nKey Finding: 4-7"
        result = self.parser.parse(response, sentence_count=10)
        assert len(result) == 1
        # 0-3 and 4-7 are adjacent → coalesced to 0-7
        assert result[0].ranges == (SentenceRange(start=0, end=7),)

    def test_range_clamped_to_sentence_count(self) -> None:
        result = self.parser.parse("My Insight: 0-100", sentence_count=10)
        assert result[0].ranges == (SentenceRange(start=0, end=9),)

    def test_inverted_range_swapped(self) -> None:
        result = self.parser.parse("My Insight: 5-2", sentence_count=10)
        assert result[0].ranges == (SentenceRange(start=2, end=5),)

    def test_lines_without_colon_skipped(self) -> None:
        response = "This line has no colon\nReal Insight: 0-5"
        result = self.parser.parse(response, sentence_count=10)
        assert len(result) == 1
        assert result[0].name == "Real Insight"

    def test_empty_response_raises(self) -> None:
        with pytest.raises(ParseError, match="No valid insights"):
            self.parser.parse("", sentence_count=10)

    def test_response_with_no_valid_lines_raises(self) -> None:
        with pytest.raises(ParseError, match="No valid insights"):
            self.parser.parse("no colon here\nalso no colon", sentence_count=10)

    def test_insight_name_with_colon_uses_last_colon_for_ranges(self) -> None:
        """Name containing a colon — greedy match should handle this."""
        result = self.parser.parse("Speed: 10x Faster: 0-5", sentence_count=10)
        assert len(result) == 1
        assert result[0].ranges == (SentenceRange(start=0, end=5),)

    def test_insertion_order_preserved(self) -> None:
        response = "Insight C: 8-9\nInsight A: 0-2\nInsight B: 5-6"
        result = self.parser.parse(response, sentence_count=10)
        assert [i.name for i in result] == ["Insight C", "Insight A", "Insight B"]
