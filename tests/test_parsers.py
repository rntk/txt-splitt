"""Tests for response parsing."""

import pytest

from txt_splitt.errors import ParseError
from txt_splitt.parsers import TopicRangeParser
from txt_splitt.types import SentenceRange


class TestTopicRangeParser:
    def setup_method(self) -> None:
        self.parser = TopicRangeParser()

    def test_single_topic_single_range(self) -> None:
        response = "Technology>AI>GPT-4: 0-5"
        result = self.parser.parse(response, sentence_count=10)
        assert len(result) == 1
        assert result[0].label == ("Technology", "AI", "GPT-4")
        assert result[0].ranges == (SentenceRange(start=0, end=5),)

    def test_multiple_topics(self) -> None:
        response = "Technology>AI>GPT-4: 0-5\nSport>Football>England: 6-9"
        result = self.parser.parse(response, sentence_count=10)
        assert len(result) == 2
        assert result[0].label == ("Technology", "AI", "GPT-4")
        assert result[1].label == ("Sport", "Football", "England")

    def test_multiple_ranges_per_topic(self) -> None:
        response = "Technology>Database>PostgreSQL: 0-5, 10-15"
        result = self.parser.parse(response, sentence_count=20)
        assert len(result) == 1
        assert result[0].ranges == (
            SentenceRange(start=0, end=5),
            SentenceRange(start=10, end=15),
        )

    def test_single_sentence_range(self) -> None:
        response = "Technology>AI>GPT-4: 5"
        result = self.parser.parse(response, sentence_count=10)
        assert result[0].ranges == (SentenceRange(start=5, end=5),)

    def test_mixed_ranges(self) -> None:
        response = "Sport>Football>England: 2, 4, 6-9"
        result = self.parser.parse(response, sentence_count=10)
        assert result[0].ranges == (
            SentenceRange(start=2, end=2),
            SentenceRange(start=4, end=4),
            SentenceRange(start=6, end=9),
        )

    def test_label_split_by_separator(self) -> None:
        response = "Science>Climate>IPCC Report: 0-3"
        result = self.parser.parse(response, sentence_count=5)
        assert result[0].label == ("Science", "Climate", "IPCC Report")

    def test_two_level_label(self) -> None:
        response = "Technology>AI: 0-3"
        result = self.parser.parse(response, sentence_count=5)
        assert result[0].label == ("Technology", "AI")

    def test_flat_label_accepted(self) -> None:
        response = "PostgreSQL: 0-3"
        result = self.parser.parse(response, sentence_count=5)
        assert result[0].label == ("PostgreSQL",)

    def test_clamping_to_max_index(self) -> None:
        response = "Technology>AI>GPT-4: 0-100"
        result = self.parser.parse(response, sentence_count=10)
        assert result[0].ranges == (SentenceRange(start=0, end=9),)

    def test_ranges_sorted_by_start(self) -> None:
        response = "Technology>AI>GPT-4: 10-15, 0-5"
        result = self.parser.parse(response, sentence_count=20)
        assert result[0].ranges == (
            SentenceRange(start=0, end=5),
            SentenceRange(start=10, end=15),
        )

    def test_skips_lines_without_colon(self) -> None:
        response = "Some junk line\nTechnology>AI>GPT-4: 0-5"
        result = self.parser.parse(response, sentence_count=10)
        assert len(result) == 1

    def test_empty_response_raises(self) -> None:
        with pytest.raises(ParseError):
            self.parser.parse("", sentence_count=10)

    def test_no_valid_ranges_raises(self) -> None:
        with pytest.raises(ParseError):
            self.parser.parse("just some text without ranges", sentence_count=10)

    def test_zero_sentence_count_raises(self) -> None:
        with pytest.raises(ParseError):
            self.parser.parse("Technology>AI: 0-5", sentence_count=0)

    def test_negative_sentence_count_raises(self) -> None:
        with pytest.raises(ParseError):
            self.parser.parse("Technology>AI: 0-5", sentence_count=-1)

    def test_whitespace_in_label_trimmed(self) -> None:
        response = " Technology > AI > GPT-4 : 0-5"
        result = self.parser.parse(response, sentence_count=10)
        assert result[0].label == ("Technology", "AI", "GPT-4")

    def test_inverted_range_swapped(self) -> None:
        response = "Technology>AI: 5-0"
        result = self.parser.parse(response, sentence_count=10)
        assert result[0].ranges == (SentenceRange(start=0, end=5),)
