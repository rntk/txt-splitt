"""Unit tests for the LLM stage."""

from unittest.mock import MagicMock

import pytest

from txt_splitt.errors import LLMError
from txt_splitt.llm import TopicRangeLLM
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
