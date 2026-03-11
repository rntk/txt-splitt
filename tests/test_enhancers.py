"""Tests for the enhancer stage."""

from unittest.mock import MagicMock

import pytest

from txt_splitt.enhancers import (
    ShortSentenceEnhancer,
    _parse_reassignment_response,
)
from txt_splitt.errors import EnhancerError
from txt_splitt.types import Sentence, SentenceGroup, SentenceRange, _indices_to_ranges


def _make_sentence(index: int, text: str) -> Sentence:
    return Sentence(
        index=index, start=index * 50, end=index * 50 + len(text), text=text
    )


def _all_covered_indices(groups: list[SentenceGroup]) -> set[int]:
    indices: set[int] = set()
    for g in groups:
        for r in g.ranges:
            for i in range(r.start, r.end + 1):
                indices.add(i)
    return indices


class TestShortSentenceEnhancer:
    def setup_method(self) -> None:
        self.client = MagicMock()
        self.enhancer = ShortSentenceEnhancer(self.client, min_length=20)

    def test_no_short_sentences_returns_unchanged(self) -> None:
        sentences = [
            _make_sentence(0, "This is a fairly long sentence about AI."),
            _make_sentence(1, "Another long sentence about climate change."),
            _make_sentence(2, "Yet another long sentence about technology."),
        ]
        groups = [
            SentenceGroup(label=("Technology",), ranges=(SentenceRange(0, 1),)),
            SentenceGroup(label=("Science",), ranges=(SentenceRange(2, 2),)),
        ]

        result = self.enhancer.enhance(groups, sentences)

        self.client.call.assert_not_called()
        assert len(result) == 2
        assert result[0].ranges == (SentenceRange(0, 1),)
        assert result[1].ranges == (SentenceRange(2, 2),)

    def test_short_sentence_at_boundary_reassigned_to_next(self) -> None:
        sentences = [
            _make_sentence(0, "This is a fairly long sentence about AI."),
            _make_sentence(1, "Indeed."),  # short, at end of group A
            _make_sentence(2, "Climate change is a major global concern."),
        ]
        groups = [
            SentenceGroup(label=("Technology", "AI"), ranges=(SentenceRange(0, 1),)),
            SentenceGroup(label=("Science", "Climate"), ranges=(SentenceRange(2, 2),)),
        ]
        self.client.call.return_value = "NEXT"

        result = self.enhancer.enhance(groups, sentences)

        self.client.call.assert_called_once()
        assert result[0].ranges == (SentenceRange(0, 0),)
        assert result[1].ranges == (SentenceRange(1, 2),)

    def test_short_sentence_at_boundary_reassigned_to_previous(self) -> None:
        sentences = [
            _make_sentence(0, "This is a fairly long sentence about AI."),
            _make_sentence(1, "Yes, sure."),  # short, at start of group B
            _make_sentence(2, "Climate change is a major global concern."),
        ]
        groups = [
            SentenceGroup(label=("Technology", "AI"), ranges=(SentenceRange(0, 0),)),
            SentenceGroup(label=("Science", "Climate"), ranges=(SentenceRange(1, 2),)),
        ]
        self.client.call.return_value = "PREVIOUS"

        result = self.enhancer.enhance(groups, sentences)

        self.client.call.assert_called_once()
        assert result[0].ranges == (SentenceRange(0, 1),)
        assert result[1].ranges == (SentenceRange(2, 2),)

    def test_ambiguous_llm_response_keeps_original(self) -> None:
        sentences = [
            _make_sentence(0, "This is a fairly long sentence about AI."),
            _make_sentence(1, "OK then."),  # short
            _make_sentence(2, "Climate change is a major global concern."),
        ]
        groups = [
            SentenceGroup(label=("Technology",), ranges=(SentenceRange(0, 1),)),
            SentenceGroup(label=("Science",), ranges=(SentenceRange(2, 2),)),
        ]
        self.client.call.return_value = "I'm not sure about this one"

        result = self.enhancer.enhance(groups, sentences)

        assert result[0].ranges == (SentenceRange(0, 1),)
        assert result[1].ranges == (SentenceRange(2, 2),)

    def test_single_group_no_llm_calls(self) -> None:
        sentences = [
            _make_sentence(0, "Short."),
            _make_sentence(1, "Also short."),
        ]
        groups = [
            SentenceGroup(label=("Technology",), ranges=(SentenceRange(0, 1),)),
        ]

        result = self.enhancer.enhance(groups, sentences)

        self.client.call.assert_not_called()
        assert result == groups

    def test_single_sentence_no_llm_calls(self) -> None:
        sentences = [_make_sentence(0, "Short.")]
        groups = [
            SentenceGroup(label=("Technology",), ranges=(SentenceRange(0, 0),)),
        ]

        self.enhancer.enhance(groups, sentences)

        self.client.call.assert_not_called()

    def test_group_emptied_by_reassignment_is_dropped(self) -> None:
        sentences = [
            _make_sentence(0, "Long sentence about artificial intelligence and ML."),
            _make_sentence(1, "Yes."),  # short, sole member of group B
            _make_sentence(2, "More about deep learning and neural networks."),
        ]
        groups = [
            SentenceGroup(label=("Technology",), ranges=(SentenceRange(0, 0),)),
            SentenceGroup(label=("Misc",), ranges=(SentenceRange(1, 1),)),
            SentenceGroup(label=("Science",), ranges=(SentenceRange(2, 2),)),
        ]
        # First candidate: sentence 1 (short) at boundary 0|1 → LLM says PREVIOUS
        # Second candidate: sentence 1 (short) at boundary 1|2
        # skipped (already reassigned).
        self.client.call.return_value = "PREVIOUS"

        result = self.enhancer.enhance(groups, sentences)

        assert len(result) == 2
        assert result[0].label == ("Technology",)
        assert result[0].ranges == (SentenceRange(0, 1),)
        assert result[1].label == ("Science",)

    def test_llm_call_failure_raises_enhancer_error(self) -> None:
        sentences = [
            _make_sentence(0, "This is a fairly long sentence about AI."),
            _make_sentence(1, "OK."),  # short
            _make_sentence(2, "Climate change is a major global concern."),
        ]
        groups = [
            SentenceGroup(label=("Technology",), ranges=(SentenceRange(0, 1),)),
            SentenceGroup(label=("Science",), ranges=(SentenceRange(2, 2),)),
        ]
        self.client.call.side_effect = Exception("Network error")

        with pytest.raises(EnhancerError, match="LLM call failed"):
            self.enhancer.enhance(groups, sentences)

    def test_enhancer_error_passthrough(self) -> None:
        sentences = [
            _make_sentence(0, "This is a fairly long sentence about AI."),
            _make_sentence(1, "OK."),  # short
            _make_sentence(2, "Climate change is a major global concern."),
        ]
        groups = [
            SentenceGroup(label=("Technology",), ranges=(SentenceRange(0, 1),)),
            SentenceGroup(label=("Science",), ranges=(SentenceRange(2, 2),)),
        ]
        self.client.call.side_effect = EnhancerError("Custom error")

        with pytest.raises(EnhancerError, match="Custom error"):
            self.enhancer.enhance(groups, sentences)

    def test_coverage_invariant_maintained(self) -> None:
        sentences = [
            _make_sentence(0, "This is a fairly long sentence about AI."),
            _make_sentence(1, "OK."),  # short
            _make_sentence(2, "Sure."),  # short
            _make_sentence(3, "Climate change is a major global concern."),
            _make_sentence(4, "Yeah."),  # short
            _make_sentence(5, "The economy is growing at a steady pace now."),
        ]
        groups = [
            SentenceGroup(label=("Technology",), ranges=(SentenceRange(0, 1),)),
            SentenceGroup(label=("Science",), ranges=(SentenceRange(2, 3),)),
            SentenceGroup(label=("Economy",), ranges=(SentenceRange(4, 5),)),
        ]
        self.client.call.return_value = "NEXT"

        result = self.enhancer.enhance(groups, sentences)

        covered = _all_covered_indices(result)
        assert covered == set(range(6))

    def test_short_sentence_in_middle_of_group_ignored(self) -> None:
        sentences = [
            _make_sentence(0, "This is a fairly long sentence about AI."),
            _make_sentence(1, "Yes."),  # short, but in middle of group
            _make_sentence(2, "This is another long sentence about AI."),
            _make_sentence(3, "Climate change is a major global concern."),
        ]
        groups = [
            SentenceGroup(label=("Technology",), ranges=(SentenceRange(0, 2),)),
            SentenceGroup(label=("Science",), ranges=(SentenceRange(3, 3),)),
        ]

        result = self.enhancer.enhance(groups, sentences)

        # Sentence 1 is short but NOT at a boundary — no LLM call for it
        self.client.call.assert_not_called()
        assert result[0].ranges == (SentenceRange(0, 2),)

    def test_custom_min_length_threshold(self) -> None:
        sentences = [
            _make_sentence(
                0, "This is a fairly long sentence about artificial intelligence."
            ),
            _make_sentence(1, "A medium-length sentence here."),  # 30 chars
            _make_sentence(
                2, "Climate change is a major global concern right now today."
            ),
        ]
        groups = [
            SentenceGroup(label=("Technology",), ranges=(SentenceRange(0, 1),)),
            SentenceGroup(label=("Science",), ranges=(SentenceRange(2, 2),)),
        ]

        # With min_length=20 (default for this test class),
        # 30-char sentence is not short.
        self.enhancer.enhance(groups, sentences)
        self.client.call.assert_not_called()

        # With min_length=50, 30-char sentence IS short (but 56+ char sentences are not)
        client2 = MagicMock()
        client2.call.return_value = "NEXT"
        enhancer50 = ShortSentenceEnhancer(client2, min_length=50)
        enhancer50.enhance(groups, sentences)
        client2.call.assert_called_once()

    def test_custom_temperature_forwarded(self) -> None:
        sentences = [
            _make_sentence(0, "This is a fairly long sentence about AI."),
            _make_sentence(1, "OK."),  # short
            _make_sentence(2, "Climate change is a major global concern."),
        ]
        groups = [
            SentenceGroup(label=("Technology",), ranges=(SentenceRange(0, 1),)),
            SentenceGroup(label=("Science",), ranges=(SentenceRange(2, 2),)),
        ]
        enhancer = ShortSentenceEnhancer(self.client, min_length=20, temperature=0.5)
        self.client.call.return_value = "NEXT"

        enhancer.enhance(groups, sentences)

        _, kwargs = self.client.call.call_args
        assert kwargs["temperature"] == 0.5

    def test_preserves_group_order(self) -> None:
        sentences = [
            _make_sentence(0, "Climate change is a major global concern."),
            _make_sentence(1, "OK."),  # short
            _make_sentence(2, "This is a fairly long sentence about AI."),
        ]
        groups = [
            SentenceGroup(label=("Science",), ranges=(SentenceRange(0, 1),)),
            SentenceGroup(label=("Technology",), ranges=(SentenceRange(2, 2),)),
        ]
        self.client.call.return_value = "NEXT"

        result = self.enhancer.enhance(groups, sentences)

        assert result[0].label == ("Science",)
        assert result[1].label == ("Technology",)

    def test_prompt_contains_sentence_and_labels(self) -> None:
        sentences = [
            _make_sentence(0, "This is a fairly long sentence about AI."),
            _make_sentence(1, "OK."),  # short
            _make_sentence(2, "Climate change is a major global concern."),
        ]
        groups = [
            SentenceGroup(label=("Technology", "AI"), ranges=(SentenceRange(0, 1),)),
            SentenceGroup(label=("Science", "Climate"), ranges=(SentenceRange(2, 2),)),
        ]
        self.client.call.return_value = "NEXT"

        self.enhancer.enhance(groups, sentences)

        prompt = self.client.call.call_args[0][0]
        assert "OK." in prompt
        assert "Technology > AI" in prompt
        assert "Science > Climate" in prompt

    def test_non_contiguous_ranges_handled(self) -> None:
        sentences = [
            _make_sentence(0, "Long sentence about AI and machine learning."),
            _make_sentence(1, "OK."),  # short, boundary with group B
            _make_sentence(2, "Climate change is really concerning these days."),
            _make_sentence(3, "More about artificial intelligence and stuff."),
        ]
        groups = [
            SentenceGroup(
                label=("Technology",),
                ranges=(SentenceRange(0, 1), SentenceRange(3, 3)),
            ),
            SentenceGroup(label=("Science",), ranges=(SentenceRange(2, 2),)),
        ]
        self.client.call.return_value = "NEXT"

        result = self.enhancer.enhance(groups, sentences)

        covered = _all_covered_indices(result)
        assert covered == set(range(4))

    def test_llm_response_with_both_words_is_ambiguous(self) -> None:
        sentences = [
            _make_sentence(0, "This is a fairly long sentence about AI."),
            _make_sentence(1, "OK."),
            _make_sentence(2, "Climate change is a major global concern."),
        ]
        groups = [
            SentenceGroup(label=("Technology",), ranges=(SentenceRange(0, 1),)),
            SentenceGroup(label=("Science",), ranges=(SentenceRange(2, 2),)),
        ]
        self.client.call.return_value = "Could be PREVIOUS or NEXT, hard to say"

        result = self.enhancer.enhance(groups, sentences)

        # Ambiguous — sentence stays
        assert result[0].ranges == (SentenceRange(0, 1),)
        assert result[1].ranges == (SentenceRange(2, 2),)


class TestParseReassignmentResponse:
    def test_previous(self) -> None:
        assert _parse_reassignment_response("PREVIOUS") == "previous"

    def test_next(self) -> None:
        assert _parse_reassignment_response("NEXT") == "next"

    def test_previous_lowercase(self) -> None:
        assert _parse_reassignment_response("previous") == "previous"

    def test_next_with_extra_text(self) -> None:
        assert _parse_reassignment_response("I think NEXT") == "next"

    def test_both_words_returns_none(self) -> None:
        assert _parse_reassignment_response("PREVIOUS or NEXT") is None

    def test_neither_word_returns_none(self) -> None:
        assert _parse_reassignment_response("I don't know") is None

    def test_empty_returns_none(self) -> None:
        assert _parse_reassignment_response("") is None


class TestIndicesToRanges:
    def test_contiguous(self) -> None:
        assert _indices_to_ranges([0, 1, 2, 3]) == [SentenceRange(0, 3)]

    def test_non_contiguous(self) -> None:
        assert _indices_to_ranges([0, 1, 4, 5]) == [
            SentenceRange(0, 1),
            SentenceRange(4, 5),
        ]

    def test_single(self) -> None:
        assert _indices_to_ranges([3]) == [SentenceRange(3, 3)]

    def test_empty(self) -> None:
        assert _indices_to_ranges([]) == []

    def test_all_individual(self) -> None:
        assert _indices_to_ranges([0, 2, 4]) == [
            SentenceRange(0, 0),
            SentenceRange(2, 2),
            SentenceRange(4, 4),
        ]
