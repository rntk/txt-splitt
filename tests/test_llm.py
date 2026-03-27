"""Unit tests for the LLM stage."""

# ruff: noqa: E501

import asyncio
import json
from typing import Literal, cast
from unittest.mock import MagicMock

import pytest

from txt_splitt.errors import LLMError
from txt_splitt.retry import RetryConfig
from txt_splitt.sentences.llm import (
    HierarchicalTopicRangeLLM,
    TopicListLLM,
    TopicRangeAssignmentLLM,
    TopicRangeLLM,
    _extract_lines_by_range,
)
from txt_splitt.sentences.types import MarkedText, SentenceRange
from txt_splitt.tracer import Tracer


class TestTopicRangeLLM:
    def test_successful_query(self) -> None:
        client = MagicMock()
        client.call.return_value = "  Technology>AI: 0-2  "
        llm = TopicRangeLLM(client)

        marked_text = MarkedText(tagged_text="[0] AI is fast.", sentence_count=1)
        response = llm.query(marked_text)

        assert response == "Technology>AI: 0-2"
        client.call.assert_called_once()

    def test_empty_response_raises_error(self) -> None:
        client = MagicMock()
        client.call.return_value = ""
        llm = TopicRangeLLM(client)

        marked_text = MarkedText(tagged_text="...", sentence_count=1)
        with pytest.raises(LLMError, match="Empty LLM response"):
            llm.query(marked_text)

    def test_whitespace_response_raises_error(self) -> None:
        client = MagicMock()
        client.call.return_value = "   "
        llm = TopicRangeLLM(client)

        marked_text = MarkedText(tagged_text="...", sentence_count=1)
        with pytest.raises(LLMError, match="Empty LLM response"):
            llm.query(marked_text)

    def test_none_response_raises_error(self) -> None:
        client = MagicMock()
        client.call.return_value = None
        llm = TopicRangeLLM(client)

        marked_text = MarkedText(tagged_text="...", sentence_count=1)
        with pytest.raises(LLMError, match="Empty LLM response"):
            llm.query(marked_text)

    def test_client_exception_wrapped(self) -> None:
        client = MagicMock()
        client.call.side_effect = Exception("Network error")
        llm = TopicRangeLLM(client)

        marked_text = MarkedText(tagged_text="...", sentence_count=1)
        with pytest.raises(LLMError, match="LLM call failed: Network error"):
            llm.query(marked_text)

    def test_llm_error_propagates(self) -> None:
        client = MagicMock()
        client.call.side_effect = LLMError("Custom LLM error")
        llm = TopicRangeLLM(client)

        marked_text = MarkedText(tagged_text="...", sentence_count=1)
        with pytest.raises(LLMError, match="Custom LLM error"):
            llm.query(marked_text)

    def test_prompt_contains_tagged_text_and_content_tags(self) -> None:
        client = MagicMock()
        client.call.return_value = "Technology>AI: 0"
        llm = TopicRangeLLM(client)

        tagged_text = "{0} Unique marker text"
        marked_text = MarkedText(tagged_text=tagged_text, sentence_count=1)
        llm.query(marked_text)

        args, kwargs = client.call.call_args
        prompt = args[0]
        assert tagged_text in prompt
        assert "<content>" in prompt
        assert "</content>" in prompt
        assert kwargs["temperature"] == 0.0

    def test_prompt_includes_output_contract_rules(self) -> None:
        client = MagicMock()
        client.call.return_value = "Technology>AI: 0"
        llm = TopicRangeLLM(client)

        marked_text = MarkedText(tagged_text="{0} Text", sentence_count=1)
        llm.query(marked_text)

        args, _kwargs = client.call.call_args
        prompt = args[0]
        assert 'Use 2-4 levels separated by ">"' in prompt
        assert "final answer must contain ONLY topic lines." in prompt
        assert 'Use ":" only once per line' in prompt
        marker_rule = (
            "Every marker ID shown in <content> must belong to exactly one topic line."
        )
        assert marker_rule in prompt

    def test_prompt_guides_digest_story_separation(self) -> None:
        client = MagicMock()
        client.call.return_value = "Technology>AI: 0"
        llm = TopicRangeLLM(client)

        marked_text = MarkedText(tagged_text="{0} Text", sentence_count=1)
        llm.query(marked_text)

        args, _kwargs = client.call.call_args
        prompt = args[0]
        digest_rule = "For digest-style article blurbs, use one topic per story/article"
        assert digest_rule in prompt
        assert "split them into" in prompt
        assert "separate sections with DISTINCT topic labels" in prompt
        specificity_rule = (
            "Prefer the specific story, comparison, release, review, company move,"
        )
        assert specificity_rule in prompt
        assert "Do NOT treat every newline as a topic boundary." in prompt

    def test_custom_temperature_is_forwarded(self) -> None:
        client = MagicMock()
        client.call.return_value = "Technology>AI: 0"
        llm = TopicRangeLLM(client, temperature=0.7)

        marked_text = MarkedText(tagged_text="{0} Text", sentence_count=1)
        llm.query(marked_text)

        args, kwargs = client.call.call_args
        assert isinstance(args[0], str)
        assert kwargs["temperature"] == 0.7

    def test_json_output_mode_updates_prompt_with_schema(self) -> None:
        client = MagicMock()
        client.call.return_value = '{"topics":[]}'
        llm = TopicRangeLLM(client, output_mode="json")

        marked_text = MarkedText(tagged_text="{0} Text", sentence_count=1)
        llm.query(marked_text)

        args, _kwargs = client.call.call_args
        prompt = args[0]
        assert "JSON SCHEMA:" in prompt
        assert '"topics"' in prompt
        assert '"label"' in prompt
        assert '"ranges"' in prompt
        assert "Return ONLY valid JSON" in prompt
        assert llm.response_format == "json"

    def test_invalid_output_mode_raises(self) -> None:
        client = MagicMock()
        invalid_mode = cast(Literal["text", "json"], "yaml")
        with pytest.raises(ValueError, match="output_mode must be"):
            TopicRangeLLM(client, output_mode=invalid_mode)

    def test_invalid_max_response_chars_raises(self) -> None:
        client = MagicMock()
        with pytest.raises(ValueError, match="max_response_chars must be > 0"):
            TopicRangeLLM(client, max_response_chars=0)

    def test_overly_large_response_raises_error(self) -> None:
        client = MagicMock()
        client.call.return_value = "x" * 101
        llm = TopicRangeLLM(client, max_response_chars=100)

        marked_text = MarkedText(tagged_text="{0} Text", sentence_count=1)
        with pytest.raises(LLMError, match="LLM response too large"):
            llm.query(marked_text)

    def test_repetitive_response_raises_error(self) -> None:
        client = MagicMock()
        repeated_phrase = "article 5 is 520-521 and article 6 is 521-521 duplicate"
        client.call.return_value = " ".join([repeated_phrase] * 40)
        llm = TopicRangeLLM(client)

        marked_text = MarkedText(tagged_text="{0} Text", sentence_count=1)
        with pytest.raises(LLMError, match="LLM response appears repetitive"):
            llm.query(marked_text)


class TestTopicRangeLLMWithChunker:
    def test_query_without_chunker_unchanged(self) -> None:
        client = MagicMock()
        client.call.return_value = "Technology>AI: 0-2"
        llm = TopicRangeLLM(client)

        mt = MarkedText(tagged_text="{0} A\n{1} B\n{2} C", sentence_count=3)
        result = llm.query(mt)

        assert result == "Technology>AI: 0-2"
        client.call.assert_called_once()

    def test_chunker_concatenates_responses(self) -> None:
        client = MagicMock()
        client.call.side_effect = [
            "Technology>AI: 0-2",
            "Science>Climate: 3-5",
        ]

        chunker = MagicMock()
        chunk_a = MarkedText(tagged_text="{0} A\n{1} B\n{2} C", sentence_count=3)
        chunk_b = MarkedText(tagged_text="{3} D\n{4} E\n{5} F", sentence_count=3)
        chunker.chunk.return_value = [chunk_a, chunk_b]

        llm = TopicRangeLLM(client, chunker=chunker)
        mt = MarkedText(
            tagged_text="{0} A\n{1} B\n{2} C\n{3} D\n{4} E\n{5} F",
            sentence_count=6,
        )
        result = llm.query(mt)

        assert result == "Technology>AI: 0-2\nScience>Climate: 3-5"
        assert client.call.call_count == 2
        chunker.chunk.assert_called_once_with(mt)

    def test_chunker_single_chunk(self) -> None:
        client = MagicMock()
        client.call.return_value = "Technology>AI: 0-2"

        chunker = MagicMock()
        mt = MarkedText(tagged_text="{0} A\n{1} B\n{2} C", sentence_count=3)
        chunker.chunk.return_value = [mt]

        llm = TopicRangeLLM(client, chunker=chunker)
        result = llm.query(mt)

        assert result == "Technology>AI: 0-2"
        client.call.assert_called_once()

    def test_chunker_error_in_second_chunk(self) -> None:
        client = MagicMock()
        client.call.side_effect = [
            "Technology>AI: 0-2",
            Exception("Network error"),
        ]

        chunker = MagicMock()
        chunk_a = MarkedText(tagged_text="{0} A", sentence_count=1)
        chunk_b = MarkedText(tagged_text="{1} B", sentence_count=1)
        chunker.chunk.return_value = [chunk_a, chunk_b]

        llm = TopicRangeLLM(client, chunker=chunker)
        mt = MarkedText(tagged_text="{0} A\n{1} B", sentence_count=2)

        with pytest.raises(LLMError, match="LLM call failed"):
            llm.query(mt)

    def test_chunker_empty_response_raises(self) -> None:
        client = MagicMock()
        client.call.return_value = ""

        chunker = MagicMock()
        mt = MarkedText(tagged_text="{0} A", sentence_count=1)
        chunker.chunk.return_value = [mt]

        llm = TopicRangeLLM(client, chunker=chunker)
        with pytest.raises(LLMError, match="Empty LLM response"):
            llm.query(mt)

    def test_chunker_concatenates_json_responses(self) -> None:
        client = MagicMock()
        client.call.side_effect = [
            '{"topics":[{"label":["Technology","AI"],"ranges":[{"start":0,"end":2}]}]}',
            '{"topics":[{"label":["Science","Climate"],"ranges":[{"start":3,"end":5}]}]}',
        ]

        chunker = MagicMock()
        chunk_a = MarkedText(tagged_text="{0} A\n{1} B\n{2} C", sentence_count=3)
        chunk_b = MarkedText(tagged_text="{3} D\n{4} E\n{5} F", sentence_count=3)
        chunker.chunk.return_value = [chunk_a, chunk_b]

        llm = TopicRangeLLM(client, chunker=chunker, output_mode="json")
        mt = MarkedText(
            tagged_text="{0} A\n{1} B\n{2} C\n{3} D\n{4} E\n{5} F",
            sentence_count=6,
        )
        result = llm.query(mt)

        assert result == (
            '{"topics":[{"label":["Technology","AI"],"ranges":[{"start":0,"end":2}]}]}\n'
            '{"topics":[{"label":["Science","Climate"],"ranges":[{"start":3,"end":5}]}]}'
        )


class TestTopicListLLM:
    def test_successful_extraction(self) -> None:
        client = MagicMock()
        client.call.return_value = "Technology>AI>GPT-4\nSport>Football>England"
        llm = TopicListLLM(client)

        mt = MarkedText(tagged_text="{0} AI.\n{1} Football.", sentence_count=2)
        topics = llm.extract(mt)

        assert topics == ["Technology>AI>GPT-4", "Sport>Football>England"]
        client.call.assert_called_once()

    def test_strips_whitespace_from_topics(self) -> None:
        client = MagicMock()
        client.call.return_value = "  Technology>AI  \n  Sport>Football  \n"
        llm = TopicListLLM(client)

        mt = MarkedText(tagged_text="{0} Text", sentence_count=1)
        topics = llm.extract(mt)

        assert topics == ["Technology>AI", "Sport>Football"]

    def test_empty_response_raises_error(self) -> None:
        client = MagicMock()
        client.call.return_value = ""
        llm = TopicListLLM(client)

        mt = MarkedText(tagged_text="...", sentence_count=1)
        with pytest.raises(LLMError, match="Empty LLM response"):
            llm.extract(mt)

    def test_whitespace_only_response_raises_error(self) -> None:
        client = MagicMock()
        client.call.return_value = "   \n\n  "
        llm = TopicListLLM(client)

        mt = MarkedText(tagged_text="...", sentence_count=1)
        with pytest.raises(LLMError, match="Empty LLM response"):
            llm.extract(mt)

    def test_client_exception_wrapped(self) -> None:
        client = MagicMock()
        client.call.side_effect = Exception("Network error")
        llm = TopicListLLM(client)

        mt = MarkedText(tagged_text="...", sentence_count=1)
        with pytest.raises(LLMError, match="LLM call failed: Network error"):
            llm.extract(mt)

    def test_llm_error_propagates(self) -> None:
        client = MagicMock()
        client.call.side_effect = LLMError("Custom error")
        llm = TopicListLLM(client)

        mt = MarkedText(tagged_text="...", sentence_count=1)
        with pytest.raises(LLMError, match="Custom error"):
            llm.extract(mt)

    def test_prompt_contains_content_tags_and_no_ranges(self) -> None:
        client = MagicMock()
        client.call.return_value = "Technology>AI"
        llm = TopicListLLM(client)

        tagged_text = "{0} Unique marker text"
        mt = MarkedText(tagged_text=tagged_text, sentence_count=1)
        llm.extract(mt)

        args, kwargs = client.call.call_args
        prompt = args[0]
        assert tagged_text in prompt
        assert "<content>" in prompt
        assert "</content>" in prompt
        assert "Do NOT assign sentence ranges" in prompt
        assert kwargs["temperature"] == 0.0

    def test_custom_temperature_is_forwarded(self) -> None:
        client = MagicMock()
        client.call.return_value = "Technology>AI"
        llm = TopicListLLM(client, temperature=0.5)

        mt = MarkedText(tagged_text="{0} Text", sentence_count=1)
        llm.extract(mt)

        _args, kwargs = client.call.call_args
        assert kwargs["temperature"] == 0.5

    def test_invalid_max_response_chars_raises(self) -> None:
        client = MagicMock()
        with pytest.raises(ValueError, match="max_response_chars must be > 0"):
            TopicListLLM(client, max_response_chars=0)

    def test_overly_large_response_raises_error(self) -> None:
        client = MagicMock()
        client.call.return_value = "x" * 101
        llm = TopicListLLM(client, max_response_chars=100)

        mt = MarkedText(tagged_text="{0} Text", sentence_count=1)
        with pytest.raises(LLMError, match="LLM response too large"):
            llm.extract(mt)

    def test_repetitive_response_raises_error(self) -> None:
        client = MagicMock()
        repeated_phrase = "article 5 is 520-521 and article 6 is 521-521 duplicate"
        client.call.return_value = " ".join([repeated_phrase] * 40)
        llm = TopicListLLM(client)

        mt = MarkedText(tagged_text="{0} Text", sentence_count=1)
        with pytest.raises(LLMError, match="LLM response appears repetitive"):
            llm.extract(mt)

    def test_chunker_deduplicates_topics(self) -> None:
        client = MagicMock()
        client.call.side_effect = [
            "Technology>AI>GPT-4\nSport>Football",
            "Technology>AI>GPT-4\nScience>Climate",
        ]

        chunker = MagicMock()
        chunk_a = MarkedText(tagged_text="{0} A\n{1} B", sentence_count=2)
        chunk_b = MarkedText(tagged_text="{2} C\n{3} D", sentence_count=2)
        chunker.chunk.return_value = [chunk_a, chunk_b]

        llm = TopicListLLM(client, chunker=chunker)
        mt = MarkedText(tagged_text="{0} A\n{1} B\n{2} C\n{3} D", sentence_count=4)
        topics = llm.extract(mt)

        assert topics == [
            "Technology>AI>GPT-4",
            "Sport>Football",
            "Science>Climate",
        ]
        assert client.call.call_count == 2

    def test_chunker_error_in_second_chunk(self) -> None:
        client = MagicMock()
        client.call.side_effect = [
            "Technology>AI",
            Exception("Network error"),
        ]

        chunker = MagicMock()
        chunk_a = MarkedText(tagged_text="{0} A", sentence_count=1)
        chunk_b = MarkedText(tagged_text="{1} B", sentence_count=1)
        chunker.chunk.return_value = [chunk_a, chunk_b]

        llm = TopicListLLM(client, chunker=chunker)
        mt = MarkedText(tagged_text="{0} A\n{1} B", sentence_count=2)

        with pytest.raises(LLMError, match="LLM call failed"):
            llm.extract(mt)


class TestTopicRangeAssignmentLLM:
    def test_successful_assignment(self) -> None:
        client = MagicMock()
        client.call.side_effect = ["0-2", "3-5"]
        llm = TopicRangeAssignmentLLM(client)
        topics = ["Technology>AI>GPT-4", "Sport>Football>England"]

        mt = MarkedText(tagged_text="{0} A\n{1} B\n{2} C", sentence_count=3)
        result = llm.assign(mt, topics)

        assert result == "Technology>AI>GPT-4: 0-2\nSport>Football>England: 3-5"
        assert client.call.call_count == 2

    def test_prompt_contains_single_topic(self) -> None:
        client = MagicMock()
        client.call.side_effect = ["0", "1"]
        topics = ["Technology>AI", "Sport>Football"]
        llm = TopicRangeAssignmentLLM(client)

        mt = MarkedText(tagged_text="{0} Text", sentence_count=1)
        llm.assign(mt, topics)

        # First call should contain first topic only
        first_prompt = client.call.call_args_list[0][0][0]
        assert "Technology>AI" in first_prompt
        assert "Sport>Football" not in first_prompt
        assert "<content>" in first_prompt
        assert "</content>" in first_prompt

        # Second call should contain second topic only
        second_prompt = client.call.call_args_list[1][0][0]
        assert "Sport>Football" in second_prompt

    def test_kv_cache_common_prefix(self) -> None:
        """Prompts for different topics share the same prefix for KV cache."""
        client = MagicMock()
        client.call.side_effect = ["0-1", "2-3"]
        topics = ["Technology>AI", "Sport>Football"]
        llm = TopicRangeAssignmentLLM(client)

        mt = MarkedText(tagged_text="{0} A\n{1} B", sentence_count=2)
        llm.assign(mt, topics)

        prompt_1 = client.call.call_args_list[0][0][0]
        prompt_2 = client.call.call_args_list[1][0][0]

        # Everything before the topic line should be identical
        prefix_1 = prompt_1.rsplit("Assign marker ranges for this topic:\n", 1)[0]
        prefix_2 = prompt_2.rsplit("Assign marker ranges for this topic:\n", 1)[0]
        assert prefix_1 == prefix_2

    def test_none_response_skips_topic(self) -> None:
        client = MagicMock()
        client.call.side_effect = ["NONE", "3-5"]
        llm = TopicRangeAssignmentLLM(client)

        mt = MarkedText(tagged_text="{0} A", sentence_count=1)
        result = llm.assign(mt, ["Technology>AI", "Sport>Football"])

        assert result == "Sport>Football: 3-5"

    def test_empty_response_skips_topic(self) -> None:
        client = MagicMock()
        client.call.side_effect = ["", "3-5"]
        llm = TopicRangeAssignmentLLM(client)

        mt = MarkedText(tagged_text="{0} A", sentence_count=1)
        result = llm.assign(mt, ["Technology>AI", "Sport>Football"])

        assert result == "Sport>Football: 3-5"

    def test_all_topics_empty_returns_empty(self) -> None:
        client = MagicMock()
        client.call.side_effect = ["NONE", ""]
        llm = TopicRangeAssignmentLLM(client)

        mt = MarkedText(tagged_text="{0} A", sentence_count=1)
        result = llm.assign(mt, ["Technology>AI", "Sport>Football"])

        assert result == ""

    def test_client_exception_wrapped(self) -> None:
        client = MagicMock()
        client.call.side_effect = Exception("Network error")
        llm = TopicRangeAssignmentLLM(client)

        mt = MarkedText(tagged_text="...", sentence_count=1)
        with pytest.raises(LLMError, match="LLM call failed: Network error"):
            llm.assign(mt, ["Technology>AI"])

    def test_llm_error_propagates(self) -> None:
        client = MagicMock()
        client.call.side_effect = LLMError("Custom LLM error")
        llm = TopicRangeAssignmentLLM(client)

        mt = MarkedText(tagged_text="...", sentence_count=1)
        with pytest.raises(LLMError, match="Custom LLM error"):
            llm.assign(mt, ["Technology>AI"])

    def test_json_output_mode(self) -> None:
        client = MagicMock()
        client.call.side_effect = [
            '[{"start": 0, "end": 2}]',
            '[{"start": 3, "end": 5}]',
        ]
        llm = TopicRangeAssignmentLLM(client, output_mode="json")
        topics = ["Technology>AI", "Sport>Football"]

        mt = MarkedText(tagged_text="{0} Text", sentence_count=1)
        result = llm.assign(mt, topics)

        parsed = json.loads(result)
        assert len(parsed["topics"]) == 2
        assert parsed["topics"][0]["label"] == ["Technology", "AI"]
        assert parsed["topics"][0]["ranges"] == [{"start": 0, "end": 2}]
        assert parsed["topics"][1]["label"] == ["Sport", "Football"]
        assert parsed["topics"][1]["ranges"] == [{"start": 3, "end": 5}]

        # Check JSON prompt format
        args, _kwargs = client.call.call_args_list[0]
        prompt = args[0]
        assert "JSON array" in prompt
        assert llm.response_format == "json"

    def test_json_none_topic_skipped(self) -> None:
        client = MagicMock()
        client.call.side_effect = ["[]", '[{"start": 3, "end": 5}]']
        llm = TopicRangeAssignmentLLM(client, output_mode="json")

        mt = MarkedText(tagged_text="{0} Text", sentence_count=1)
        result = llm.assign(mt, ["Technology>AI", "Sport>Football"])

        parsed = json.loads(result)
        assert len(parsed["topics"]) == 1
        assert parsed["topics"][0]["label"] == ["Sport", "Football"]

    def test_invalid_output_mode_raises(self) -> None:
        client = MagicMock()
        invalid_mode = cast(Literal["text", "json"], "yaml")
        with pytest.raises(ValueError, match="output_mode must be"):
            TopicRangeAssignmentLLM(client, output_mode=invalid_mode)

    def test_invalid_max_response_chars_raises(self) -> None:
        client = MagicMock()
        with pytest.raises(ValueError, match="max_response_chars must be > 0"):
            TopicRangeAssignmentLLM(client, max_response_chars=0)

    def test_overly_large_response_raises_error(self) -> None:
        client = MagicMock()
        client.call.return_value = "x" * 101
        llm = TopicRangeAssignmentLLM(client, max_response_chars=100)

        mt = MarkedText(tagged_text="{0} Text", sentence_count=1)
        with pytest.raises(LLMError, match="LLM response too large"):
            llm.assign(mt, ["Technology>AI"])

    def test_custom_temperature_is_forwarded(self) -> None:
        client = MagicMock()
        client.call.return_value = "0"
        llm = TopicRangeAssignmentLLM(client, temperature=0.3)

        mt = MarkedText(tagged_text="{0} Text", sentence_count=1)
        llm.assign(mt, ["Technology>AI"])

        _args, kwargs = client.call.call_args
        assert kwargs["temperature"] == 0.3

    def test_chunker_concatenates_responses(self) -> None:
        client = MagicMock()
        # chunk1: topic1=0-2, topic2=NONE; chunk2: topic1=NONE, topic2=3-5
        client.call.side_effect = [
            "0-2",  # chunk1, Technology>AI
            "NONE",  # chunk1, Sport>Football
            "NONE",  # chunk2, Technology>AI
            "3-5",  # chunk2, Sport>Football
        ]

        chunker = MagicMock()
        chunk_a = MarkedText(tagged_text="{0} A\n{1} B\n{2} C", sentence_count=3)
        chunk_b = MarkedText(tagged_text="{3} D\n{4} E\n{5} F", sentence_count=3)
        chunker.chunk.return_value = [chunk_a, chunk_b]

        topics = ["Technology>AI", "Sport>Football"]
        llm = TopicRangeAssignmentLLM(client, chunker=chunker)
        mt = MarkedText(
            tagged_text="{0} A\n{1} B\n{2} C\n{3} D\n{4} E\n{5} F",
            sentence_count=6,
        )
        result = llm.assign(mt, topics)

        assert result == "Technology>AI: 0-2\nSport>Football: 3-5"
        assert client.call.call_count == 4  # 2 chunks x 2 topics

    def test_per_topic_prompt_has_content_tags(self) -> None:
        client = MagicMock()
        client.call.side_effect = ["0-2", "3-5"]

        topics = ["Technology>AI", "Sport>Football"]
        llm = TopicRangeAssignmentLLM(client)
        mt = MarkedText(tagged_text="{0} A\n{1} B", sentence_count=2)
        llm.assign(mt, topics)

        # Each per-topic prompt should include content and its own topic
        for i, call_args in enumerate(client.call.call_args_list):
            prompt = call_args[0][0]
            assert "<content>" in prompt
            assert "</content>" in prompt
            assert "{0} A" in prompt
            assert topics[i] in prompt

    def test_async_client_executes_in_parallel(self) -> None:
        """Test that async clients are detected and executed in parallel."""
        import asyncio

        class AsyncClient:
            def __init__(self) -> None:
                self.call_count = 0

            async def call(self, prompt: str, temperature: float) -> str:
                self.call_count += 1
                await asyncio.sleep(0.01)  # Simulate async I/O
                if "Technology>AI" in prompt:
                    return "0-2"
                return "3-5"

        async def run_test() -> None:
            client = AsyncClient()
            llm = TopicRangeAssignmentLLM(client)
            topics = ["Technology>AI", "Sport>Football"]

            mt = MarkedText(tagged_text="{0} A\n{1} B\n{2} C", sentence_count=3)
            result = await llm.assign_async(mt, topics)

            assert result == "Technology>AI: 0-2\nSport>Football: 3-5"
            assert client.call_count == 2

        asyncio.run(run_test())

    def test_assign_async_traces_response(self) -> None:
        """Async assignment writes merged response into tracer span."""
        import asyncio

        class AsyncClient:
            async def call(self, prompt: str, temperature: float) -> str:
                await asyncio.sleep(0.01)
                if "Technology>AI" in prompt:
                    return "0-2"
                return "3-5"

        async def run_test() -> None:
            tracer = Tracer()
            client = AsyncClient()
            llm = TopicRangeAssignmentLLM(client, tracer=tracer)
            topics = ["Technology>AI", "Sport>Football"]

            mt = MarkedText(tagged_text="{0} A\n{1} B\n{2} C", sentence_count=3)
            result = await llm.assign_async(mt, topics)

            assert result == "Technology>AI: 0-2\nSport>Football: 3-5"
            assert len(tracer.spans) == 1
            span = tracer.spans[0]
            assert span.name == "topic_range_assignment_llm.assign_async"
            assert span.attributes["response"] == result
            assert span.attributes["response_length"] == len(result)
            assert len(span.children) == 2
            for child in span.children:
                assert child.name == "llm.call"
                assert "<content>" in child.attributes["prompt"]
                assert "</content>" in child.attributes["prompt"]
                assert (
                    "Assign marker ranges for this topic:" in child.attributes["prompt"]
                )

        asyncio.run(run_test())

    def test_async_client_json_mode(self) -> None:
        """Test async client with JSON output mode."""
        import asyncio

        class AsyncClient:
            async def call(self, prompt: str, temperature: float) -> str:
                await asyncio.sleep(0.01)
                if "Technology>AI" in prompt:
                    return '[{"start": 0, "end": 2}]'
                return '[{"start": 3, "end": 5}]'

        async def run_test() -> None:
            client = AsyncClient()
            llm = TopicRangeAssignmentLLM(client, output_mode="json")
            topics = ["Technology>AI", "Sport>Football"]

            mt = MarkedText(tagged_text="{0} Text", sentence_count=1)
            result = await llm.assign_async(mt, topics)

            parsed = json.loads(result)
            assert len(parsed["topics"]) == 2
            assert parsed["topics"][0]["label"] == ["Technology", "AI"]
            assert parsed["topics"][1]["label"] == ["Sport", "Football"]

        asyncio.run(run_test())

    def test_async_client_with_chunker(self) -> None:
        """Test async client works correctly with chunking."""
        import asyncio

        class AsyncClient:
            def __init__(self) -> None:
                self.call_count = 0

            async def call(self, prompt: str, temperature: float) -> str:
                self.call_count += 1
                await asyncio.sleep(0.01)
                # chunk1: topic1=0-2, topic2=NONE; chunk2: topic1=NONE, topic2=3-5
                if "{0} A" in prompt and "Technology>AI" in prompt:
                    return "0-2"
                if "{0} A" in prompt and "Sport>Football" in prompt:
                    return "NONE"
                if "{3} D" in prompt and "Technology>AI" in prompt:
                    return "NONE"
                if "{3} D" in prompt and "Sport>Football" in prompt:
                    return "3-5"
                return "NONE"

        async def run_test() -> None:
            chunker = MagicMock()
            chunk_a = MarkedText(tagged_text="{0} A\n{1} B\n{2} C", sentence_count=3)
            chunk_b = MarkedText(tagged_text="{3} D\n{4} E\n{5} F", sentence_count=3)
            chunker.chunk.return_value = [chunk_a, chunk_b]

            client = AsyncClient()
            topics = ["Technology>AI", "Sport>Football"]
            llm = TopicRangeAssignmentLLM(client, chunker=chunker)
            mt = MarkedText(
                tagged_text="{0} A\n{1} B\n{2} C\n{3} D\n{4} E\n{5} F",
                sentence_count=6,
            )
            result = await llm.assign_async(mt, topics)

            assert result == "Technology>AI: 0-2\nSport>Football: 3-5"
            # 2 chunks x 2 topics (parallelized per chunk)
            assert client.call_count == 4

        asyncio.run(run_test())

    def test_sync_client_with_assign_async(self) -> None:
        """Test that sync clients work with assign_async()."""
        import asyncio

        client = MagicMock()
        client.call.side_effect = ["0-2", "3-5"]
        llm = TopicRangeAssignmentLLM(client)
        topics = ["Technology>AI", "Sport>Football"]

        async def run_test() -> None:
            mt = MarkedText(tagged_text="{0} A\n{1} B\n{2} C", sentence_count=3)
            result = await llm.assign_async(mt, topics)
            assert result == "Technology>AI: 0-2\nSport>Football: 3-5"

        asyncio.run(run_test())

    def test_async_client_with_sync_assign_raises(self) -> None:
        """Test that using assign() with async client raises clear error."""

        class AsyncClient:
            async def call(self, prompt: str, temperature: float) -> str:
                return "0-2"

        client = AsyncClient()
        llm = TopicRangeAssignmentLLM(client)
        mt = MarkedText(tagged_text="{0} A", sentence_count=1)

        with pytest.raises(RuntimeError, match="Cannot use assign.*async client"):
            llm.assign(mt, ["Technology>AI"])


class TestRetryConfig:
    def test_max_attempts_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="max_attempts must be >= 1"):
            RetryConfig(max_attempts=0)

    def test_next_returns_none_when_exhausted(self) -> None:
        policy = RetryConfig(max_attempts=2)
        assert policy.next(2, "p", 0.0, LLMError("x")) is None

    def test_next_returns_same_params_by_default(self) -> None:
        policy = RetryConfig(max_attempts=3)
        result = policy.next(0, "my prompt", 0.5, LLMError("x"))
        assert result == ("my prompt", 0.5)

    def test_temperature_schedule_applied(self) -> None:
        policy = RetryConfig(max_attempts=3, temperature_schedule=[0.2, 0.7])
        assert policy.next(0, "p", 0.0, LLMError("x")) == ("p", 0.2)
        assert policy.next(1, "p", 0.0, LLMError("x")) == ("p", 0.7)
        # Beyond schedule length: falls back to passed temperature
        assert policy.next(2, "p", 0.0, LLMError("x")) == ("p", 0.0)

    def test_prompt_modifier_applied(self) -> None:
        policy = RetryConfig(
            max_attempts=3,
            prompt_modifier=lambda p, attempt: f"{p} retry={attempt}",
        )
        result = policy.next(1, "base", 0.0, LLMError("x"))
        assert result == ("base retry=1", 0.0)

    def test_both_modifiers_combined(self) -> None:
        policy = RetryConfig(
            max_attempts=2,
            temperature_schedule=[0.9],
            prompt_modifier=lambda p, _: p + " HINT",
        )
        result = policy.next(0, "original", 0.0, LLMError("x"))
        assert result == ("original HINT", 0.9)


class TestTopicRangeLLMWithRetry:
    def test_retries_on_empty_response(self) -> None:
        client = MagicMock()
        client.call.side_effect = ["", "Technology>AI: 0-2"]
        policy = RetryConfig(max_attempts=1)
        llm = TopicRangeLLM(client, retry_policy=policy)

        mt = MarkedText(tagged_text="{0} AI is fast.", sentence_count=1)
        result = llm.query(mt)

        assert result == "Technology>AI: 0-2"
        assert client.call.call_count == 2

    def test_raises_after_max_attempts_exhausted(self) -> None:
        client = MagicMock()
        client.call.return_value = ""
        policy = RetryConfig(max_attempts=2)
        llm = TopicRangeLLM(client, retry_policy=policy)

        mt = MarkedText(tagged_text="{0} Text", sentence_count=1)
        with pytest.raises(LLMError, match="Empty LLM response"):
            llm.query(mt)

        assert client.call.call_count == 3  # 1 initial + 2 retries

    def test_temperature_schedule_forwarded_on_retry(self) -> None:
        client = MagicMock()
        client.call.side_effect = ["", "Technology>AI: 0"]
        policy = RetryConfig(max_attempts=1, temperature_schedule=[0.9])
        llm = TopicRangeLLM(client, retry_policy=policy)

        mt = MarkedText(tagged_text="{0} Text", sentence_count=1)
        llm.query(mt)

        first_temp = client.call.call_args_list[0][1]["temperature"]
        retry_temp = client.call.call_args_list[1][1]["temperature"]
        assert first_temp == 0.0
        assert retry_temp == 0.9

    def test_prompt_modifier_applied_on_retry(self) -> None:
        client = MagicMock()
        client.call.side_effect = ["", "Technology>AI: 0"]
        policy = RetryConfig(
            max_attempts=1,
            prompt_modifier=lambda p, _: p + " RETRY_HINT",
        )
        llm = TopicRangeLLM(client, retry_policy=policy)

        mt = MarkedText(tagged_text="{0} Text", sentence_count=1)
        llm.query(mt)

        retry_prompt = client.call.call_args_list[1][0][0]
        assert retry_prompt.endswith(" RETRY_HINT")

    def test_no_retry_without_policy(self) -> None:
        client = MagicMock()
        client.call.return_value = ""
        llm = TopicRangeLLM(client)  # no retry_policy

        mt = MarkedText(tagged_text="{0} Text", sentence_count=1)
        with pytest.raises(LLMError, match="Empty LLM response"):
            llm.query(mt)

        assert client.call.call_count == 1


class TestTopicListLLMWithRetry:
    def test_retries_on_empty_response(self) -> None:
        client = MagicMock()
        client.call.side_effect = ["", "Technology>AI>GPT-4"]
        policy = RetryConfig(max_attempts=1)
        llm = TopicListLLM(client, retry_policy=policy)

        mt = MarkedText(tagged_text="{0} Text", sentence_count=1)
        result = llm.extract(mt)

        assert result == ["Technology>AI>GPT-4"]
        assert client.call.call_count == 2

    def test_raises_after_max_attempts_exhausted(self) -> None:
        client = MagicMock()
        client.call.return_value = ""
        policy = RetryConfig(max_attempts=2)
        llm = TopicListLLM(client, retry_policy=policy)

        mt = MarkedText(tagged_text="{0} Text", sentence_count=1)
        with pytest.raises(LLMError, match="Empty LLM response"):
            llm.extract(mt)

        assert client.call.call_count == 3  # 1 initial + 2 retries


class TestTopicRangeAssignmentLLMWithRetry:
    def test_retries_on_oversized_response(self) -> None:
        client = MagicMock()
        client.call.side_effect = ["x" * 101, "0-2"]
        policy = RetryConfig(max_attempts=1)
        llm = TopicRangeAssignmentLLM(
            client,
            max_response_chars=100,
            retry_policy=policy,
        )

        mt = MarkedText(tagged_text="{0} Text", sentence_count=1)
        result = llm.assign(mt, ["Technology>AI"])

        assert result == "Technology>AI: 0-2"
        assert client.call.call_count == 2

    def test_raises_after_max_attempts_exhausted(self) -> None:
        client = MagicMock()
        client.call.return_value = "x" * 101
        policy = RetryConfig(max_attempts=2)
        llm = TopicRangeAssignmentLLM(
            client,
            max_response_chars=100,
            retry_policy=policy,
        )

        mt = MarkedText(tagged_text="{0} Text", sentence_count=1)
        with pytest.raises(LLMError, match="LLM response too large"):
            llm.assign(mt, ["Technology>AI"])

        assert client.call.call_count == 3  # 1 initial + 2 retries

    def test_empty_none_response_skipped_without_retry(self) -> None:
        """NONE/empty responses are valid (no sentences for topic) and never retried."""
        client = MagicMock()
        client.call.side_effect = ["NONE", "3-5"]
        policy = RetryConfig(max_attempts=3)
        llm = TopicRangeAssignmentLLM(client, retry_policy=policy)

        mt = MarkedText(tagged_text="{0} Text", sentence_count=1)
        result = llm.assign(mt, ["Technology>AI", "Sport>Football"])

        assert result == "Sport>Football: 3-5"
        assert client.call.call_count == 2  # no retry triggered for NONE

    def test_async_retries_on_oversized_response(self) -> None:
        call_count = 0

        class AsyncClient:
            async def call(self, prompt: str, temperature: float) -> str:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return "x" * 101
                return "0-2"

        async def run_test() -> None:
            client = AsyncClient()
            policy = RetryConfig(max_attempts=1)
            llm = TopicRangeAssignmentLLM(
                client, max_response_chars=100, retry_policy=policy
            )
            mt = MarkedText(tagged_text="{0} Text", sentence_count=1)
            result = await llm.assign_async(mt, ["Technology>AI"])
            assert result == "Technology>AI: 0-2"
            assert call_count == 2

        asyncio.run(run_test())

    def test_temperature_schedule_forwarded_on_retry(self) -> None:
        received_temps: list[float] = []

        client = MagicMock()

        def side_effect(prompt: str, temperature: float) -> str:
            received_temps.append(temperature)
            if len(received_temps) == 1:
                return "x" * 101  # triggers LLMError on first call
            return "0-2"

        client.call.side_effect = side_effect
        policy = RetryConfig(max_attempts=1, temperature_schedule=[0.8])
        llm = TopicRangeAssignmentLLM(
            client,
            max_response_chars=100,
            retry_policy=policy,
        )

        mt = MarkedText(tagged_text="{0} Text", sentence_count=1)
        llm.assign(mt, ["Technology>AI"])

        assert received_temps == [0.0, 0.8]


class TestExtractLinesByRange:
    def test_extracts_matching_lines(self) -> None:
        tagged = "{0} First\n{1} Second\n{2} Third\n{3} Fourth"
        ranges = [SentenceRange(start=1, end=2)]
        result = _extract_lines_by_range(tagged, ranges)
        assert result == "{1} Second\n{2} Third"

    def test_multiple_ranges(self) -> None:
        tagged = "{0} A\n{1} B\n{2} C\n{3} D\n{4} E"
        ranges = [SentenceRange(start=0, end=1), SentenceRange(start=3, end=4)]
        result = _extract_lines_by_range(tagged, ranges)
        assert result == "{0} A\n{1} B\n{3} D\n{4} E"

    def test_empty_ranges_returns_empty(self) -> None:
        tagged = "{0} Text"
        assert _extract_lines_by_range(tagged, []) == ""

    def test_preserves_original_marker_ids(self) -> None:
        tagged = "{10} Ten\n{11} Eleven\n{12} Twelve"
        ranges = [SentenceRange(start=11, end=12)]
        result = _extract_lines_by_range(tagged, ranges)
        assert result == "{11} Eleven\n{12} Twelve"
        assert "{10}" not in result

    def test_no_matching_markers_returns_empty(self) -> None:
        tagged = "{0} A\n{1} B"
        ranges = [SentenceRange(start=5, end=10)]
        result = _extract_lines_by_range(tagged, ranges)
        assert result == ""

    def test_single_marker_range(self) -> None:
        tagged = "{0} A\n{1} B\n{2} C"
        ranges = [SentenceRange(start=1, end=1)]
        result = _extract_lines_by_range(tagged, ranges)
        assert result == "{1} B"


class TestHierarchicalTopicRangeLLM:
    def _make_large_marked_text(self, n: int = 100) -> MarkedText:
        lines = [f"{{{i}}} Sentence {i}." for i in range(n)]
        return MarkedText(tagged_text="\n".join(lines), sentence_count=n)

    def test_small_doc_uses_single_stage(self) -> None:
        """Below the threshold, falls back to a standard single-stage call."""
        client = MagicMock()
        client.call.return_value = "Technology>AI: 0-4"
        llm = HierarchicalTopicRangeLLM(client, min_sentences_for_hierarchical=10)

        mt = MarkedText(tagged_text="{0} A\n{1} B", sentence_count=2)
        result = llm.query(mt)

        assert result == "Technology>AI: 0-4"
        assert client.call.call_count == 1

    def test_small_doc_uses_full_detail_prompt(self) -> None:
        """Single-stage fallback uses the full detail prompt (not the coarse one)."""
        client = MagicMock()
        client.call.return_value = "Technology>AI: 0"
        llm = HierarchicalTopicRangeLLM(client, min_sentences_for_hierarchical=10)

        mt = MarkedText(tagged_text="{0} Text", sentence_count=1)
        llm.query(mt)

        prompt = client.call.call_args[0][0]
        # Full detail prompt has granular topic naming rules, not the coarse heading
        assert "chapter-level" not in prompt
        assert "Category>BroadTopic" not in prompt

    def test_two_stage_produces_merged_output(self) -> None:
        """Stage 1 gives coarse groups; stage 2 refines each into subtopics."""
        coarse_response = "Technology>AI: 0-49\nBusiness>Finance: 50-99"
        ai_fine = "Technology>AI>LLMs: 0-24\nTechnology>AI>Agents: 25-49"
        finance_fine = "Business>Finance>Stocks: 50-74\nBusiness>Finance>Bonds: 75-99"

        client = MagicMock()
        client.call.side_effect = [coarse_response, ai_fine, finance_fine]

        mt = self._make_large_marked_text(100)
        llm = HierarchicalTopicRangeLLM(client, min_sentences_for_hierarchical=10)
        result = llm.query(mt)

        assert "Technology>AI>LLMs: 0-24" in result
        assert "Technology>AI>Agents: 25-49" in result
        assert "Business>Finance>Stocks: 50-74" in result
        assert "Business>Finance>Bonds: 75-99" in result
        assert client.call.call_count == 3

    def test_stage2_receives_only_subset_lines(self) -> None:
        """Stage 2 prompts contain only the lines from the coarse group."""
        coarse_response = "Technology>AI: 0-1\nBusiness>Finance: 2-3"
        client = MagicMock()
        client.call.side_effect = [
            coarse_response,
            "Technology>AI>LLMs: 0-1",
            "Business>Finance>Stocks: 2-3",
        ]

        tagged = (
            "{0} AI sentence.\n{1} More AI.\n{2} Finance sentence.\n{3} More finance."
        )
        mt = MarkedText(tagged_text=tagged, sentence_count=4)
        llm = HierarchicalTopicRangeLLM(client, min_sentences_for_hierarchical=2)
        llm.query(mt)

        # Second call (first stage-2) should only contain markers 0-1
        second_prompt = client.call.call_args_list[1][0][0]
        assert "{0}" in second_prompt
        assert "{1}" in second_prompt
        assert "{2}" not in second_prompt
        assert "{3}" not in second_prompt

        # Third call (second stage-2) should only contain markers 2-3
        third_prompt = client.call.call_args_list[2][0][0]
        assert "{2}" in third_prompt
        assert "{3}" in third_prompt
        assert "{0}" not in third_prompt
        assert "{1}" not in third_prompt

    def test_stage2_prompt_includes_parent_topic(self) -> None:
        """Refinement prompt tells the LLM the parent topic context."""
        coarse_response = "Technology>AI: 0-49"
        client = MagicMock()
        client.call.side_effect = [
            coarse_response,
            "Technology>AI>LLMs: 0-49",
        ]

        mt = self._make_large_marked_text(50)
        llm = HierarchicalTopicRangeLLM(client, min_sentences_for_hierarchical=10)
        llm.query(mt)

        refine_prompt = client.call.call_args_list[1][0][0]
        assert "Technology>AI" in refine_prompt

    def test_ensure_parent_prefix_added_when_missing(self) -> None:
        """Lines without parent prefix in stage 2 response get it prepended."""
        coarse_response = "Technology>AI: 0-49"
        # Stage 2 returns without parent prefix
        client = MagicMock()
        client.call.side_effect = [
            coarse_response,
            "LLMs: 0-24\nAgents: 25-49",
        ]

        mt = self._make_large_marked_text(50)
        llm = HierarchicalTopicRangeLLM(client, min_sentences_for_hierarchical=10)
        result = llm.query(mt)

        assert "Technology>AI>LLMs: 0-24" in result
        assert "Technology>AI>Agents: 25-49" in result

    def test_empty_subset_skipped(self) -> None:
        """Coarse groups whose marker IDs are absent from tagged_text are skipped."""
        # tagged_text only has marker {0}; coarse assigns {0} to AI and {1} to Finance.
        # Marker {1} is absent from tagged_text, so Finance subset is empty → skipped.
        coarse_response = "Technology>AI: 0-0\nBusiness>Finance: 1-1"
        client = MagicMock()
        client.call.side_effect = [
            coarse_response,
            "Technology>AI>LLMs: 0-0",
            # Finance stage 2 should NOT be called (empty subset)
        ]

        tagged = "{0} AI sentence only."
        mt = MarkedText(tagged_text=tagged, sentence_count=2)
        llm = HierarchicalTopicRangeLLM(client, min_sentences_for_hierarchical=1)
        result = llm.query(mt)

        assert "Technology>AI>LLMs: 0-0" in result
        assert client.call.call_count == 2  # coarse + 1 stage-2 only

    def test_coarse_parse_failure_raises_llm_error(self) -> None:
        """Unparseable coarse response raises LLMError."""
        client = MagicMock()
        client.call.return_value = "this is not a valid topic range line"

        mt = self._make_large_marked_text(100)
        llm = HierarchicalTopicRangeLLM(client, min_sentences_for_hierarchical=10)
        with pytest.raises(LLMError, match="Failed to parse coarse LLM response"):
            llm.query(mt)

    def test_json_output_mode_two_stage(self) -> None:
        """JSON mode merges stage-2 topic arrays into a single JSON response."""
        coarse_json = '{"topics": [{"label": ["Technology", "AI"], "ranges": [{"start": 0, "end": 49}]}, {"label": ["Business", "Finance"], "ranges": [{"start": 50, "end": 99}]}]}'
        ai_fine_json = '{"topics": [{"label": ["Technology", "AI", "LLMs"], "ranges": [{"start": 0, "end": 24}]}, {"label": ["Technology", "AI", "Agents"], "ranges": [{"start": 25, "end": 49}]}]}'
        finance_fine_json = '{"topics": [{"label": ["Business", "Finance", "Stocks"], "ranges": [{"start": 50, "end": 74}]}, {"label": ["Business", "Finance", "Bonds"], "ranges": [{"start": 75, "end": 99}]}]}'

        client = MagicMock()
        client.call.side_effect = [coarse_json, ai_fine_json, finance_fine_json]

        mt = self._make_large_marked_text(100)
        llm = HierarchicalTopicRangeLLM(
            client, output_mode="json", min_sentences_for_hierarchical=10
        )
        result = llm.query(mt)

        parsed = json.loads(result)
        assert len(parsed["topics"]) == 4
        labels = [t["label"] for t in parsed["topics"]]
        assert ["Technology", "AI", "LLMs"] in labels
        assert ["Technology", "AI", "Agents"] in labels
        assert ["Business", "Finance", "Stocks"] in labels
        assert ["Business", "Finance", "Bonds"] in labels
        assert llm.response_format == "json"

    def test_json_stage2_parent_prefix_enforced(self) -> None:
        """JSON mode prepends parent label segments when missing from stage-2 output."""
        coarse_json = '{"topics": [{"label": ["Technology", "AI"], "ranges": [{"start": 0, "end": 49}]}]}'
        # Stage 2 returns labels without parent prefix
        fine_json = (
            '{"topics": [{"label": ["LLMs"], "ranges": [{"start": 0, "end": 24}]}]}'
        )

        client = MagicMock()
        client.call.side_effect = [coarse_json, fine_json]

        mt = self._make_large_marked_text(50)
        llm = HierarchicalTopicRangeLLM(
            client, output_mode="json", min_sentences_for_hierarchical=10
        )
        result = llm.query(mt)

        parsed = json.loads(result)
        assert parsed["topics"][0]["label"] == ["Technology", "AI", "LLMs"]

    def test_chunker_applied_to_stage1(self) -> None:
        """Chunker is used for the coarse stage-1 call."""
        chunker = MagicMock()
        chunk_a = MarkedText(tagged_text="{0} A\n{1} B", sentence_count=2)
        chunk_b = MarkedText(tagged_text="{2} C\n{3} D", sentence_count=2)
        chunker.chunk.return_value = [chunk_a, chunk_b]

        client = MagicMock()
        # Two coarse chunks → two stage-2 calls (one per coarse group)
        client.call.side_effect = [
            "Technology>AI: 0-1",  # chunk 1 coarse
            "Business>Finance: 2-3",  # chunk 2 coarse
            "Technology>AI>LLMs: 0-1",
            "Business>Finance>Stocks: 2-3",
        ]

        tagged = "{0} A\n{1} B\n{2} C\n{3} D"
        mt = MarkedText(tagged_text=tagged, sentence_count=4)
        llm = HierarchicalTopicRangeLLM(
            client, chunker=chunker, min_sentences_for_hierarchical=2
        )
        llm.query(mt)

        chunker.chunk.assert_called_once_with(mt)
        # Coarse prompts come from the chunks, not the full text
        first_prompt = client.call.call_args_list[0][0][0]
        assert "{0} A" in first_prompt
        assert "{2} C" not in first_prompt

    def test_invalid_output_mode_raises(self) -> None:
        client = MagicMock()
        invalid_mode = cast(Literal["text", "json"], "xml")
        with pytest.raises(ValueError, match="output_mode must be"):
            HierarchicalTopicRangeLLM(client, output_mode=invalid_mode)

    def test_invalid_max_response_chars_raises(self) -> None:
        client = MagicMock()
        with pytest.raises(ValueError, match="max_response_chars must be > 0"):
            HierarchicalTopicRangeLLM(client, max_response_chars=0)

    def test_invalid_min_sentences_raises(self) -> None:
        client = MagicMock()
        with pytest.raises(
            ValueError, match="min_sentences_for_hierarchical must be > 0"
        ):
            HierarchicalTopicRangeLLM(client, min_sentences_for_hierarchical=0)

    def test_response_format_property(self) -> None:
        client = MagicMock()
        assert (
            HierarchicalTopicRangeLLM(client, output_mode="text").response_format
            == "text"
        )
        assert (
            HierarchicalTopicRangeLLM(client, output_mode="json").response_format
            == "json"
        )

    def test_client_exception_wrapped(self) -> None:
        client = MagicMock()
        client.call.side_effect = Exception("Network error")
        llm = HierarchicalTopicRangeLLM(client, min_sentences_for_hierarchical=10)

        mt = self._make_large_marked_text(100)
        with pytest.raises(LLMError, match="LLM call failed: Network error"):
            llm.query(mt)

    def test_coarse_prompt_uses_broad_heading(self) -> None:
        """Coarse stage uses a prompt that requests high-level grouping."""
        coarse_response = "Technology>AI: 0-49\nBusiness>Finance: 50-99"
        client = MagicMock()
        client.call.side_effect = [
            coarse_response,
            "Technology>AI>LLMs: 0-49",
            "Business>Finance>Stocks: 50-99",
        ]

        mt = self._make_large_marked_text(100)
        llm = HierarchicalTopicRangeLLM(client, min_sentences_for_hierarchical=10)
        llm.query(mt)

        coarse_prompt = client.call.call_args_list[0][0][0]
        assert "chapter-level" in coarse_prompt
        assert "3-10 sections" in coarse_prompt

    def test_custom_coarse_prompt_builder(self) -> None:
        """Custom coarse_prompt_builder is used for stage 1."""
        client = MagicMock()
        client.call.side_effect = [
            "Technology>AI: 0-49",
            "Technology>AI>LLMs: 0-49",
        ]

        custom_prompt_calls: list[str] = []

        def custom_coarse(text: str) -> str:
            custom_prompt_calls.append(text)
            return "custom coarse prompt"

        mt = self._make_large_marked_text(50)
        llm = HierarchicalTopicRangeLLM(
            client,
            min_sentences_for_hierarchical=10,
            coarse_prompt_builder=custom_coarse,
        )
        llm.query(mt)

        assert len(custom_prompt_calls) == 1
        assert client.call.call_args_list[0][0][0] == "custom coarse prompt"

    def test_custom_refine_prompt_builder(self) -> None:
        """Custom refine_prompt_builder is used for stage 2."""
        client = MagicMock()
        client.call.side_effect = [
            "Technology>AI: 0-49",
            "Technology>AI>LLMs: 0-49",
        ]

        refine_calls: list[tuple[str, str]] = []

        def custom_refine(text: str, parent: str) -> str:
            refine_calls.append((text, parent))
            return "custom refine prompt"

        mt = self._make_large_marked_text(50)
        llm = HierarchicalTopicRangeLLM(
            client,
            min_sentences_for_hierarchical=10,
            refine_prompt_builder=custom_refine,
        )
        llm.query(mt)

        assert len(refine_calls) == 1
        assert refine_calls[0][1] == "Technology>AI"
        assert client.call.call_args_list[1][0][0] == "custom refine prompt"
