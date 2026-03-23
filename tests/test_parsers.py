"""Tests for response parsing."""

from typing import Literal, cast

import pytest

from txt_splitt.errors import ParseError
from txt_splitt.sentences.parsers import TopicRangeParser
from txt_splitt.sentences.types import SentenceRange


class TestTopicRangeParser:
    def setup_method(self) -> None:
        self.parser = TopicRangeParser()

    def test_invalid_input_mode_raises(self) -> None:
        invalid_mode = cast(Literal["text", "json", "auto"], "yaml")
        with pytest.raises(ValueError, match="input_mode must be"):
            TopicRangeParser(input_mode=invalid_mode)

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

    def test_ignores_invalid_range_parts_but_keeps_valid_ones(self) -> None:
        response = "Technology>AI: nope, 2-3, ???, 5"
        result = self.parser.parse(response, sentence_count=10)
        assert result[0].ranges == (
            SentenceRange(start=2, end=3),
            SentenceRange(start=5, end=5),
        )

    def test_line_with_only_invalid_ranges_is_skipped(self) -> None:
        response = "Technology>AI: nope, ???\nScience>Climate: 0-1"
        result = self.parser.parse(response, sentence_count=5)
        assert len(result) == 1
        assert result[0].label == ("Science", "Climate")

    def test_all_lines_with_invalid_ranges_raise(self) -> None:
        response = "Technology>AI: nope\nScience>Climate: ???"
        with pytest.raises(ParseError):
            self.parser.parse(response, sentence_count=5)

    def test_duplicate_topics_are_merged(self) -> None:
        response = (
            "Technology>AI>GPT-4: 0-2\n"
            "Technology>AI>GPT-4: 4-5\n"
            "Science>Climate>IPCC Report: 3"
        )
        result = self.parser.parse(response, sentence_count=6)

        assert len(result) == 2
        assert result[0].label == ("Technology", "AI", "GPT-4")
        assert result[0].ranges == (
            SentenceRange(start=0, end=2),
            SentenceRange(start=4, end=5),
        )
        assert result[1].label == ("Science", "Climate", "IPCC Report")
        assert result[1].ranges == (SentenceRange(start=3, end=3),)

    def test_duplicate_topics_ranges_are_coalesced(self) -> None:
        response = (
            "Technology>Database>PostgreSQL: 0-2\n"
            "Technology>Database>PostgreSQL: 2-4\n"
            "Technology>Database>PostgreSQL: 5"
        )
        result = self.parser.parse(response, sentence_count=8)

        assert len(result) == 1
        assert result[0].label == ("Technology", "Database", "PostgreSQL")
        assert result[0].ranges == (SentenceRange(start=0, end=5),)

    def test_json_mode_single_document(self) -> None:
        parser = TopicRangeParser(input_mode="json")
        response = (
            '{"topics":['
            '{"label":["Technology","AI","GPT-4"],"ranges":[{"start":0,"end":2}]},'
            '{"label":["Science","Climate"],"ranges":[{"start":3,"end":5}]}'
            "]}"
        )
        result = parser.parse(response, sentence_count=6)

        assert len(result) == 2
        assert result[0].label == ("Technology", "AI", "GPT-4")
        assert result[0].ranges == (SentenceRange(start=0, end=2),)
        assert result[1].label == ("Science", "Climate")
        assert result[1].ranges == (SentenceRange(start=3, end=5),)

    def test_json_mode_parses_newline_delimited_documents(self) -> None:
        parser = TopicRangeParser(input_mode="json")
        response = (
            '{"topics":[{"label":["Technology","AI"],"ranges":[{"start":0,"end":1}]}]}\n'
            '{"topics":[{"label":["Technology","AI"],"ranges":[{"start":2,"end":3}]},'
            '{"label":["Science","Climate"],"ranges":[{"start":4,"end":5}]}]}'
        )
        result = parser.parse(response, sentence_count=6)

        assert len(result) == 2
        assert result[0].label == ("Technology", "AI")
        assert result[0].ranges == (SentenceRange(start=0, end=3),)
        assert result[1].label == ("Science", "Climate")
        assert result[1].ranges == (SentenceRange(start=4, end=5),)

    def test_json_mode_parses_array_of_documents(self) -> None:
        parser = TopicRangeParser(input_mode="json")
        response = (
            '[{"topics":[{"label":["Technology","AI"],"ranges":[{"start":0,"end":1}]}]},'
            '{"topics":[{"label":["Science","Climate"],"ranges":[{"start":2,"end":3}]}]}]'
        )
        result = parser.parse(response, sentence_count=4)

        assert len(result) == 2
        assert result[0].label == ("Technology", "AI")
        assert result[1].label == ("Science", "Climate")

    def test_json_mode_invalid_json_raises(self) -> None:
        parser = TopicRangeParser(input_mode="json")
        with pytest.raises(ParseError, match="Invalid JSON response"):
            parser.parse('{"topics":[', sentence_count=3)

    def test_auto_mode_prefers_json(self) -> None:
        parser = TopicRangeParser(input_mode="auto")
        response = (
            '{"topics":[{"label":["Technology","AI"],"ranges":[{"start":0,"end":2}]}]}'
        )
        result = parser.parse(response, sentence_count=3)

        assert len(result) == 1
        assert result[0].label == ("Technology", "AI")
        assert result[0].ranges == (SentenceRange(start=0, end=2),)

    def test_colon_in_topic_name_single_range(self) -> None:
        response = "Technology>Hardware>Apple: Mac Mini production: 6-20"
        result = self.parser.parse(response, sentence_count=25)
        assert len(result) == 1
        assert result[0].label == (
            "Technology",
            "Hardware",
            "Apple",
            "Mac Mini production",
        )
        assert result[0].ranges == (SentenceRange(start=6, end=20),)

    def test_colon_in_topic_name_multiple_ranges(self) -> None:
        response = "Technology>AI>GPT-4: overview: 0-5, 10-15"
        result = self.parser.parse(response, sentence_count=20)
        assert len(result) == 1
        assert result[0].label == ("Technology", "AI", "GPT-4", "overview")
        assert result[0].ranges == (
            SentenceRange(start=0, end=5),
            SentenceRange(start=10, end=15),
        )

    def test_colon_in_topic_name_mixed_with_normal(self) -> None:
        response = (
            "Technology>Hardware>Apple: Mac Mini production: 6-20\nScience>Climate: 0-5"
        )
        result = self.parser.parse(response, sentence_count=25)
        assert len(result) == 2
        assert result[0].label == (
            "Technology",
            "Hardware",
            "Apple",
            "Mac Mini production",
        )
        assert result[1].label == ("Science", "Climate")

    def test_multiple_colons_expand_to_multiple_levels(self) -> None:
        response = "Technology>Cloud>AWS: Security: IAM: 1-2"
        result = self.parser.parse(response, sentence_count=5)
        assert len(result) == 1
        assert result[0].label == ("Technology", "Cloud", "AWS", "Security", "IAM")
        assert result[0].ranges == (SentenceRange(start=1, end=2),)

    def test_auto_mode_falls_back_to_text(self) -> None:
        parser = TopicRangeParser(input_mode="auto")
        result = parser.parse("Technology>AI: 0-2", sentence_count=3)

        assert len(result) == 1
        assert result[0].label == ("Technology", "AI")
