"""Unit tests for the LLM stage."""

from typing import Literal, cast
from unittest.mock import MagicMock

import pytest

from txt_splitt.errors import LLMError
from txt_splitt.llm import TopicListLLM, TopicRangeAssignmentLLM, TopicRangeLLM
from txt_splitt.types import MarkedText


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
        client.call.return_value = (
            "Technology>AI>GPT-4: 0-2\nSport>Football>England: 3-5"
        )
        llm = TopicRangeAssignmentLLM(client)
        topics = ["Technology>AI>GPT-4", "Sport>Football>England"]

        mt = MarkedText(tagged_text="{0} A\n{1} B\n{2} C", sentence_count=3)
        result = llm.assign(mt, topics)

        assert result == "Technology>AI>GPT-4: 0-2\nSport>Football>England: 3-5"
        client.call.assert_called_once()

    def test_prompt_contains_topics(self) -> None:
        client = MagicMock()
        client.call.return_value = "Technology>AI: 0"
        topics = ["Technology>AI", "Sport>Football"]
        llm = TopicRangeAssignmentLLM(client)

        mt = MarkedText(tagged_text="{0} Text", sentence_count=1)
        llm.assign(mt, topics)

        args, _kwargs = client.call.call_args
        prompt = args[0]
        assert "- Technology>AI" in prompt
        assert "- Sport>Football" in prompt
        assert "<content>" in prompt
        assert "</content>" in prompt
        assert "TOPICS:" in prompt

    def test_empty_response_raises_error(self) -> None:
        client = MagicMock()
        client.call.return_value = ""
        llm = TopicRangeAssignmentLLM(client)

        mt = MarkedText(tagged_text="...", sentence_count=1)
        with pytest.raises(LLMError, match="Empty LLM response"):
            llm.assign(mt, ["Technology>AI"])

    def test_client_exception_wrapped(self) -> None:
        client = MagicMock()
        client.call.side_effect = Exception("Network error")
        llm = TopicRangeAssignmentLLM(client)

        mt = MarkedText(tagged_text="...", sentence_count=1)
        with pytest.raises(LLMError, match="LLM call failed: Network error"):
            llm.assign(mt, ["Technology>AI"])

    def test_json_output_mode(self) -> None:
        client = MagicMock()
        client.call.return_value = '{"topics":[]}'
        llm = TopicRangeAssignmentLLM(client, output_mode="json")

        mt = MarkedText(tagged_text="{0} Text", sentence_count=1)
        llm.assign(mt, ["Technology>AI"])

        args, _kwargs = client.call.call_args
        prompt = args[0]
        assert "JSON SCHEMA:" in prompt
        assert "Return ONLY valid JSON" in prompt
        assert llm.response_format == "json"

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
        client.call.return_value = "Technology>AI: 0"
        llm = TopicRangeAssignmentLLM(client, temperature=0.3)

        mt = MarkedText(tagged_text="{0} Text", sentence_count=1)
        llm.assign(mt, ["Technology>AI"])

        _args, kwargs = client.call.call_args
        assert kwargs["temperature"] == 0.3

    def test_chunker_concatenates_responses(self) -> None:
        client = MagicMock()
        client.call.side_effect = [
            "Technology>AI: 0-2",
            "Sport>Football: 3-5",
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
        assert client.call.call_count == 2

    def test_chunker_passes_all_topics_to_each_chunk(self) -> None:
        client = MagicMock()
        client.call.side_effect = [
            "Technology>AI: 0-2",
            "Sport>Football: 3-5",
        ]

        chunker = MagicMock()
        chunk_a = MarkedText(tagged_text="{0} A", sentence_count=1)
        chunk_b = MarkedText(tagged_text="{1} B", sentence_count=1)
        chunker.chunk.return_value = [chunk_a, chunk_b]

        topics = ["Technology>AI", "Sport>Football"]
        llm = TopicRangeAssignmentLLM(client, chunker=chunker)
        mt = MarkedText(tagged_text="{0} A\n{1} B", sentence_count=2)
        llm.assign(mt, topics)

        # Both calls should contain all topics in the prompt
        for call_args in client.call.call_args_list:
            prompt = call_args[0][0]
            assert "- Technology>AI" in prompt
            assert "- Sport>Football" in prompt
