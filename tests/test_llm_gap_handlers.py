"""Tests for LLMRepairingGapHandler."""

from unittest.mock import MagicMock

import pytest

from txt_splitt.errors import GapError
from txt_splitt.sentences.gap_handlers import (
    _GAP_PROMPT_PREFIX,
    LLMRepairingGapHandler,
    _build_gap_prompt,
    _parse_gap_response,
)
from txt_splitt.sentences.types import Sentence, SentenceGroup, SentenceRange
from txt_splitt.tracer import Tracer


def _make_sentences(n: int) -> list[Sentence]:
    return [
        Sentence(index=i, start=i * 10, end=i * 10 + 8, text=f"Sentence {i}.")
        for i in range(n)
    ]


class TestLLMRepairingGapHandler:
    def test_gap_prompt_forbids_placeholder_new_labels(self) -> None:
        prompt = _build_gap_prompt(
            sentence_text="Gap sentence.",
            prev_label=("Technology", "AI"),
            prev_context=["Previous context."],
            next_label=("Business", "Automation"),
            next_context=["Next context."],
        )

        assert "NEW is rare." in prompt
        assert "2-4 levels separated by '>'." in prompt
        assert "top level should be a broad domain such as" in prompt
        assert "lowest level should identify the specific subject" in prompt
        assert "Use official capitalization and canonical names" in prompt
        assert "Reply with exactly one line and no explanation." in prompt
        assert "NEW: Actual>Concrete>TopicPath" in prompt
        assert "NEW: Level1>Level2>Topic" in prompt
        assert "NEW: Category>Subcategory>Topic" in prompt

    def test_gap_prompt_uses_stable_prefix_for_kv_cache(self) -> None:
        prompt = _build_gap_prompt(
            sentence_text="Gap sentence.",
            prev_label=("Technology", "AI"),
            prev_context=["Previous context."],
            next_label=("Business", "Automation"),
            next_context=["Next context."],
        )

        assert prompt.startswith(_GAP_PROMPT_PREFIX)
        suffix = prompt.removeprefix(_GAP_PROMPT_PREFIX)
        assert '<OPTION_A label="Technology>AI">' in suffix
        assert "<GAP>Gap sentence.</GAP>" in suffix
        assert '<OPTION_B label="Business>Automation">' in suffix

    def test_gap_between_same_neighbor_owner_skips_llm(self) -> None:
        client = MagicMock()
        handler = LLMRepairingGapHandler(client)

        groups = [
            SentenceGroup(
                label=("A",),
                ranges=(SentenceRange(0, 0), SentenceRange(2, 2)),
            ),
        ]
        result = handler.handle(groups, sentence_count=3, sentences=_make_sentences(3))

        assert len(result) == 1
        assert result[0].ranges == (SentenceRange(0, 2),)
        client.call.assert_not_called()

    def test_html_entity_invisible_gap_sentence_skips_llm(self) -> None:
        client = MagicMock()
        handler = LLMRepairingGapHandler(client)

        groups = [
            SentenceGroup(label=("A",), ranges=(SentenceRange(0, 0),)),
            SentenceGroup(label=("B",), ranges=(SentenceRange(2, 2),)),
        ]
        sentences = [
            Sentence(index=0, start=0, end=8, text="Sentence 0."),
            Sentence(index=1, start=10, end=18, text="&#173;&#847;"),
            Sentence(index=2, start=20, end=28, text="Sentence 2."),
        ]

        result = handler.handle(groups, sentence_count=3, sentences=sentences)

        assert result[0].ranges == (SentenceRange(0, 1),)
        assert result[1].ranges == (SentenceRange(2, 2),)
        client.call.assert_not_called()

    def test_gap_context_omits_trash_sentences(self) -> None:
        client = MagicMock()
        client.call.return_value = "PREVIOUS"
        handler = LLMRepairingGapHandler(client)

        groups = [
            SentenceGroup(label=("A",), ranges=(SentenceRange(0, 1),)),
            SentenceGroup(label=("B",), ranges=(SentenceRange(3, 4),)),
        ]
        sentences = [
            Sentence(index=0, start=0, end=8, text="Meaningful previous context."),
            Sentence(index=1, start=10, end=18, text="&nbsp;"),
            Sentence(index=2, start=20, end=28, text="Gap sentence."),
            Sentence(index=3, start=30, end=38, text="&nbsp;"),
            Sentence(index=4, start=40, end=48, text="Meaningful next context."),
        ]

        handler.handle(groups, sentence_count=5, sentences=sentences)

        client.call.assert_called_once()
        prompt = client.call.call_args.args[0]
        assert "&nbsp;" not in prompt
        assert "Meaningful previous context." in prompt
        assert "Meaningful next context." in prompt

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

    def test_parse_gap_response_keeps_non_empty_new_label(self) -> None:
        decision, label = _parse_gap_response("NEW: Category>Subcategory>Topic")

        assert decision == "new"
        assert label == ("Category", "Subcategory", "Topic")

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

    def test_tracer_captures_gap_resolution_details(self) -> None:
        client = MagicMock()
        client.call.return_value = "I cannot decide"
        tracer = Tracer()
        handler = LLMRepairingGapHandler(client, tracer=tracer)

        groups = [
            SentenceGroup(label=("A",), ranges=(SentenceRange(0, 0),)),
            SentenceGroup(label=("B",), ranges=(SentenceRange(2, 2),)),
        ]
        result = handler.handle(groups, sentence_count=3, sentences=_make_sentences(3))

        assert result[0].ranges == (SentenceRange(0, 1),)
        assert result[1].ranges == (SentenceRange(2, 2),)

        assert len(tracer.spans) == 1
        root = tracer.spans[0]
        assert root.name == "gap_handler.llm_repair"
        assert root.attributes["gap_count"] == 1
        assert root.attributes["output_group_count"] == 2

        gap_span = root.children[0]
        assert gap_span.name == "gap_handler.llm_repair.gap"
        assert gap_span.attributes["gap_start"] == 1
        assert gap_span.attributes["gap_end"] == 1

        resolve_span = gap_span.children[0]
        assert resolve_span.name == "gap_handler.llm_repair.resolve_sentence"
        assert resolve_span.attributes["parsed_decision"] == "unknown"
        assert resolve_span.attributes["fallback"] == "previous"
