"""Unit tests for the LLM stage."""

import asyncio
import json
from typing import Literal, cast
from unittest.mock import MagicMock

import pytest

from txt_splitt.errors import LLMError
from txt_splitt.retry import RetryConfig
from txt_splitt.sentences.llm import (
    TopicListLLM,
    TopicRangeAssignmentLLM,
    TopicRangeLLM,
)
from txt_splitt.sentences.types import MarkedText
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
        assert "Output ONLY the final topic lines." in prompt
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
        assert "separate sections even if they are thematically related" in prompt
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
        prefix_1 = prompt_1.rsplit("Assign sentence ranges for this topic:\n", 1)[0]
        prefix_2 = prompt_2.rsplit("Assign sentence ranges for this topic:\n", 1)[0]
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
                    "Assign sentence ranges for this topic:"
                    in child.attributes["prompt"]
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
