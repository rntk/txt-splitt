"""Tests for LLMRepairingGapHandler."""

from unittest.mock import MagicMock

import pytest

from txt_splitt.errors import GapError
from txt_splitt.gap_handlers import LLMRepairingGapHandler
from txt_splitt.types import Sentence, SentenceGroup, SentenceRange


def _make_sentences(n: int) -> list[Sentence]:
    return [
        Sentence(index=i, start=i * 10, end=i * 10 + 8, text=f"Sentence {i}.")
        for i in range(n)
    ]


class TestLLMRepairingGapHandler:
    def test_gap_sentence_assigned_to_previous(self) -> None:
        client = MagicMock()
        client.call.return_value = "PREVIOUS"
        handler = LLMRepairingGapHandler(client)

        groups = [
            SentenceGroup(label=("A",), ranges=(SentenceRange(0, 0),)),
            SentenceGroup(label=("B",), ranges=(SentenceRange(2, 2),)),
        ]
        result = handler.handle(groups, sentence_count=3, sentences=_make_sentences(3))

        assert result[0].ranges == (SentenceRange(0, 1),)
        assert result[1].ranges == (SentenceRange(2, 2),)
        client.call.assert_called_once()

    def test_gap_sentence_assigned_to_next(self) -> None:
        client = MagicMock()
        client.call.return_value = "NEXT"
        handler = LLMRepairingGapHandler(client)

        groups = [
            SentenceGroup(label=("A",), ranges=(SentenceRange(0, 0),)),
            SentenceGroup(label=("B",), ranges=(SentenceRange(2, 2),)),
        ]
        result = handler.handle(groups, sentence_count=3, sentences=_make_sentences(3))

        assert result[0].ranges == (SentenceRange(0, 0),)
        assert result[1].ranges == (SentenceRange(1, 2),)
        client.call.assert_called_once()

    def test_gap_sentence_assigned_to_new_group(self) -> None:
        client = MagicMock()
        client.call.return_value = "NEW: Technology > AI > Agentic Workflows"
        handler = LLMRepairingGapHandler(client)

        groups = [
            SentenceGroup(label=("A",), ranges=(SentenceRange(0, 0),)),
            SentenceGroup(label=("B",), ranges=(SentenceRange(2, 2),)),
        ]
        result = handler.handle(groups, sentence_count=3, sentences=_make_sentences(3))

        assert len(result) == 3
        assert result[2].label == ("Technology", "AI", "Agentic Workflows")
        assert result[2].ranges == (SentenceRange(1, 1),)

    def test_ambiguous_response_defaults_to_previous(self) -> None:
        client = MagicMock()
        client.call.return_value = "I cannot decide"
        handler = LLMRepairingGapHandler(client)

        groups = [
            SentenceGroup(label=("A",), ranges=(SentenceRange(0, 0),)),
            SentenceGroup(label=("B",), ranges=(SentenceRange(2, 2),)),
        ]
        result = handler.handle(groups, sentence_count=3, sentences=_make_sentences(3))

        assert result[0].ranges == (SentenceRange(0, 1),)
        assert result[1].ranges == (SentenceRange(2, 2),)

    def test_start_and_end_gaps_use_nearest_group_without_llm(self) -> None:
        client = MagicMock()
        handler = LLMRepairingGapHandler(client)

        groups = [
            SentenceGroup(label=("A",), ranges=(SentenceRange(2, 2),)),
        ]
        result = handler.handle(groups, sentence_count=5, sentences=_make_sentences(5))

        assert len(result) == 1
        assert result[0].ranges == (SentenceRange(0, 4),)
        client.call.assert_not_called()

    def test_missing_sentences_context_raises(self) -> None:
        client = MagicMock()
        handler = LLMRepairingGapHandler(client)
        groups = [SentenceGroup(label=("A",), ranges=(SentenceRange(0, 0),))]

        with pytest.raises(GapError, match="requires sentences context"):
            handler.handle(groups, sentence_count=1)
