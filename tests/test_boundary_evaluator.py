"""Tests for the BoundaryEvaluator enhancer."""

from unittest.mock import MagicMock

import pytest

from txt_splitt.boundary_evaluator import (
    BoundaryEvaluator,
    _build_boundary_prompt,
    _gather_boundary_context,
    _parse_boundary_response,
)
from txt_splitt.errors import EnhancerError
from txt_splitt.types import Sentence, SentenceGroup, SentenceRange


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


# Shared short sentence texts used across tests
_AI_0 = "AI is transforming many industries today."
_AI_1 = "ML needs large training datasets to work."
_ENV_0 = "Climate change threatens food security."
_ENV_1 = "Sea levels endanger coastal communities."


class TestBoundaryEvaluator:
    def setup_method(self) -> None:
        self.client = MagicMock()
        self.evaluator = BoundaryEvaluator(self.client)

    def test_single_group_returns_unchanged(self) -> None:
        sentences = [
            _make_sentence(0, _AI_0),
            _make_sentence(1, _AI_1),
        ]
        groups = [
            SentenceGroup(label=("Technology",), ranges=(SentenceRange(0, 1),)),
        ]

        result = self.evaluator.enhance(groups, sentences)

        self.client.call.assert_not_called()
        assert result == groups

    def test_single_sentence_returns_unchanged(self) -> None:
        sentences = [_make_sentence(0, "Just one sentence.")]
        groups = [
            SentenceGroup(label=("Technology",), ranges=(SentenceRange(0, 0),)),
        ]

        result = self.evaluator.enhance(groups, sentences)

        self.client.call.assert_not_called()
        assert result == groups

    def test_correct_boundary_no_change(self) -> None:
        sentences = [
            _make_sentence(0, _AI_0),
            _make_sentence(1, _AI_1),
            _make_sentence(2, _ENV_0),
            _make_sentence(3, _ENV_1),
        ]
        groups = [
            SentenceGroup(label=("Technology",), ranges=(SentenceRange(0, 1),)),
            SentenceGroup(label=("Environment",), ranges=(SentenceRange(2, 3),)),
        ]
        self.client.call.return_value = "CORRECT"

        result = self.evaluator.enhance(groups, sentences)

        self.client.call.assert_called_once()
        assert result[0].ranges == (SentenceRange(0, 1),)
        assert result[1].ranges == (SentenceRange(2, 3),)

    def test_shift_left_moves_sentences_to_left_group(self) -> None:
        sentences = [
            _make_sentence(0, _AI_0),
            _make_sentence(1, _AI_1),
            _make_sentence(2, _ENV_0),
            _make_sentence(3, _ENV_1),
        ]
        groups = [
            SentenceGroup(label=("Technology",), ranges=(SentenceRange(0, 1),)),
            SentenceGroup(label=("Environment",), ranges=(SentenceRange(2, 3),)),
        ]
        self.client.call.return_value = "SHIFT_LEFT 1"

        result = self.evaluator.enhance(groups, sentences)

        assert result[0].ranges == (SentenceRange(0, 2),)
        assert result[1].ranges == (SentenceRange(3, 3),)

    def test_shift_right_moves_sentences_to_right_group(self) -> None:
        sentences = [
            _make_sentence(0, _AI_0),
            _make_sentence(1, _AI_1),
            _make_sentence(2, _ENV_0),
            _make_sentence(3, _ENV_1),
        ]
        groups = [
            SentenceGroup(label=("Technology",), ranges=(SentenceRange(0, 1),)),
            SentenceGroup(label=("Environment",), ranges=(SentenceRange(2, 3),)),
        ]
        self.client.call.return_value = "SHIFT_RIGHT 1"

        result = self.evaluator.enhance(groups, sentences)

        assert result[0].ranges == (SentenceRange(0, 0),)
        assert result[1].ranges == (SentenceRange(1, 3),)

    def test_shift_clamped_to_max_shift(self) -> None:
        sentences = [
            _make_sentence(0, _AI_0),
            _make_sentence(1, _AI_1),
            _make_sentence(2, _ENV_0),
            _make_sentence(3, _ENV_1),
        ]
        groups = [
            SentenceGroup(label=("Technology",), ranges=(SentenceRange(0, 1),)),
            SentenceGroup(label=("Environment",), ranges=(SentenceRange(2, 3),)),
        ]
        # Request shift of 10, but max_shift=2 → only 2 sentences move
        evaluator = BoundaryEvaluator(self.client, max_shift=2)
        self.client.call.return_value = "SHIFT_LEFT 10"

        result = evaluator.enhance(groups, sentences)

        # Only 2 sentences move (clamped to max_shift=2, right group has 2)
        assert result[0].ranges == (SentenceRange(0, 3),)
        assert len(result) == 1  # right group is now empty → dropped

    def test_shift_clamped_to_group_size(self) -> None:
        sentences = [
            _make_sentence(0, _AI_0),
            _make_sentence(1, _AI_1),
            _make_sentence(2, _ENV_0),
        ]
        groups = [
            SentenceGroup(label=("Technology",), ranges=(SentenceRange(0, 1),)),
            SentenceGroup(label=("Environment",), ranges=(SentenceRange(2, 2),)),
        ]
        # SHIFT_LEFT 2 but right group only has 1 sentence → only 1 moves
        evaluator = BoundaryEvaluator(self.client, max_shift=5)
        self.client.call.return_value = "SHIFT_LEFT 2"

        result = evaluator.enhance(groups, sentences)

        # Right group had only 1 sentence → it moves left, right group dropped
        assert result[0].ranges == (SentenceRange(0, 2),)
        assert len(result) == 1

    def test_coverage_invariant_maintained(self) -> None:
        sentences = [
            _make_sentence(i, f"Sentence {i} with some content here.") for i in range(6)
        ]
        groups = [
            SentenceGroup(label=("A",), ranges=(SentenceRange(0, 1),)),
            SentenceGroup(label=("B",), ranges=(SentenceRange(2, 3),)),
            SentenceGroup(label=("C",), ranges=(SentenceRange(4, 5),)),
        ]
        self.client.call.return_value = "SHIFT_LEFT 1"

        result = self.evaluator.enhance(groups, sentences)

        covered = _all_covered_indices(result)
        assert covered == set(range(6))

    def test_multiple_boundaries_each_evaluated(self) -> None:
        sentences = [
            _make_sentence(i, f"Sentence {i} with some content here.") for i in range(6)
        ]
        groups = [
            SentenceGroup(label=("A",), ranges=(SentenceRange(0, 1),)),
            SentenceGroup(label=("B",), ranges=(SentenceRange(2, 3),)),
            SentenceGroup(label=("C",), ranges=(SentenceRange(4, 5),)),
        ]
        self.client.call.return_value = "CORRECT"

        self.evaluator.enhance(groups, sentences)

        assert self.client.call.call_count == 2  # one call per boundary

    def test_llm_failure_raises_enhancer_error(self) -> None:
        sentences = [
            _make_sentence(0, _AI_0),
            _make_sentence(1, _ENV_0),
        ]
        groups = [
            SentenceGroup(label=("Technology",), ranges=(SentenceRange(0, 0),)),
            SentenceGroup(label=("Environment",), ranges=(SentenceRange(1, 1),)),
        ]
        self.client.call.side_effect = Exception("Network error")

        with pytest.raises(EnhancerError, match="LLM call failed"):
            self.evaluator.enhance(groups, sentences)

    def test_enhancer_error_passthrough(self) -> None:
        sentences = [
            _make_sentence(0, _AI_0),
            _make_sentence(1, _ENV_0),
        ]
        groups = [
            SentenceGroup(label=("Technology",), ranges=(SentenceRange(0, 0),)),
            SentenceGroup(label=("Environment",), ranges=(SentenceRange(1, 1),)),
        ]
        self.client.call.side_effect = EnhancerError("Custom error")

        with pytest.raises(EnhancerError, match="Custom error"):
            self.evaluator.enhance(groups, sentences)

    def test_prompt_contains_labels_and_sentences(self) -> None:
        sentences = [
            _make_sentence(0, _AI_0),
            _make_sentence(1, _ENV_0),
        ]
        groups = [
            SentenceGroup(label=("Technology", "AI"), ranges=(SentenceRange(0, 0),)),
            SentenceGroup(
                label=("Environment", "Climate"),
                ranges=(SentenceRange(1, 1),),
            ),
        ]
        self.client.call.return_value = "CORRECT"

        self.evaluator.enhance(groups, sentences)

        prompt = self.client.call.call_args[0][0]
        assert "Technology > AI" in prompt
        assert "Environment > Climate" in prompt
        assert _AI_0 in prompt
        assert _ENV_0 in prompt

    def test_custom_context_window(self) -> None:
        sentences = [
            _make_sentence(i, f"Sentence {i} with some content here.") for i in range(8)
        ]
        groups = [
            SentenceGroup(label=("A",), ranges=(SentenceRange(0, 3),)),
            SentenceGroup(label=("B",), ranges=(SentenceRange(4, 7),)),
        ]
        evaluator = BoundaryEvaluator(self.client, context_window=2)
        self.client.call.return_value = "CORRECT"

        evaluator.enhance(groups, sentences)

        prompt = self.client.call.call_args[0][0]
        # With context_window=2, only sentences 2,3 from left and 4,5 appear
        assert "Sentence 2" in prompt
        assert "Sentence 3" in prompt
        assert "Sentence 4" in prompt
        assert "Sentence 5" in prompt
        # Sentences 0,1 should not appear (outside window)
        assert "Sentence 0" not in prompt
        assert "Sentence 1" not in prompt

    def test_custom_temperature_forwarded(self) -> None:
        sentences = [
            _make_sentence(0, _AI_0),
            _make_sentence(1, _ENV_0),
        ]
        groups = [
            SentenceGroup(label=("Technology",), ranges=(SentenceRange(0, 0),)),
            SentenceGroup(label=("Environment",), ranges=(SentenceRange(1, 1),)),
        ]
        evaluator = BoundaryEvaluator(self.client, temperature=0.7)
        self.client.call.return_value = "CORRECT"

        evaluator.enhance(groups, sentences)

        _, kwargs = self.client.call.call_args
        assert kwargs["temperature"] == 0.7

    def test_unknown_response_treated_as_correct(self) -> None:
        sentences = [
            _make_sentence(0, _AI_0),
            _make_sentence(1, _AI_1),
            _make_sentence(2, _ENV_0),
            _make_sentence(3, _ENV_1),
        ]
        groups = [
            SentenceGroup(label=("Technology",), ranges=(SentenceRange(0, 1),)),
            SentenceGroup(label=("Environment",), ranges=(SentenceRange(2, 3),)),
        ]
        self.client.call.return_value = "I am not sure what to do here"

        result = self.evaluator.enhance(groups, sentences)

        assert result[0].ranges == (SentenceRange(0, 1),)
        assert result[1].ranges == (SentenceRange(2, 3),)


class TestParseBoundaryResponse:
    def test_correct(self) -> None:
        assert _parse_boundary_response("CORRECT", 3) == ("correct", 0)

    def test_correct_lowercase(self) -> None:
        assert _parse_boundary_response("correct", 3) == ("correct", 0)

    def test_shift_left(self) -> None:
        assert _parse_boundary_response("SHIFT_LEFT 2", 3) == ("shift_left", 2)

    def test_shift_right(self) -> None:
        assert _parse_boundary_response("SHIFT_RIGHT 1", 3) == ("shift_right", 1)

    def test_shift_left_clamped(self) -> None:
        assert _parse_boundary_response("SHIFT_LEFT 10", 3) == ("shift_left", 3)

    def test_shift_right_clamped(self) -> None:
        assert _parse_boundary_response("SHIFT_RIGHT 99", 2) == ("shift_right", 2)

    def test_shift_left_no_number(self) -> None:
        assert _parse_boundary_response("SHIFT_LEFT", 3) == ("correct", 0)

    def test_unknown_response(self) -> None:
        assert _parse_boundary_response("I don't know", 3) == ("correct", 0)

    def test_empty_response(self) -> None:
        assert _parse_boundary_response("", 3) == ("correct", 0)


class TestBuildBoundaryPrompt:
    def test_prompt_contains_topic_labels(self) -> None:
        prompt = _build_boundary_prompt(
            left_label=("Technology", "AI"),
            left_sentences=[(0, "Sentence about AI.")],
            right_label=("Environment",),
            right_sentences=[(1, "Sentence about climate.")],
        )
        assert "Technology > AI" in prompt
        assert "Environment" in prompt

    def test_prompt_contains_sentences(self) -> None:
        prompt = _build_boundary_prompt(
            left_label=("A",),
            left_sentences=[(0, "Left sentence here.")],
            right_label=("B",),
            right_sentences=[(1, "Right sentence here.")],
        )
        assert "Left sentence here." in prompt
        assert "Right sentence here." in prompt

    def test_empty_context_shows_placeholder(self) -> None:
        prompt = _build_boundary_prompt(
            left_label=("A",),
            left_sentences=[],
            right_label=("B",),
            right_sentences=[],
        )
        assert "(no sentences)" in prompt


class TestGatherBoundaryContext:
    def _make_sentences(self, n: int) -> list[Sentence]:
        return [
            Sentence(index=i, start=i * 20, end=i * 20 + 10, text=f"Sentence {i}.")
            for i in range(n)
        ]

    def _ownership(self, mapping: dict[int, int]) -> dict[int, int]:
        return mapping

    def test_left_direction_gathers_sentences_backwards(self) -> None:
        sentences = self._make_sentences(4)
        # sentences 0,1 belong to group 0; 2,3 to group 1
        ownership = {0: 0, 1: 0, 2: 1, 3: 1}
        result = _gather_boundary_context(sentences, ownership, 0, 1, "left", 3)
        # Should gather sentences 1,0 (backwards from idx=1), returned in order
        assert result == [(0, "Sentence 0."), (1, "Sentence 1.")]

    def test_right_direction_gathers_sentences_forwards(self) -> None:
        sentences = self._make_sentences(4)
        ownership = {0: 0, 1: 0, 2: 1, 3: 1}
        result = _gather_boundary_context(sentences, ownership, 1, 2, "right", 3)
        # Should gather sentences 2,3 (forwards from idx=2)
        assert result == [(2, "Sentence 2."), (3, "Sentence 3.")]

    def test_window_limits_results(self) -> None:
        sentences = self._make_sentences(6)
        ownership = {i: 0 for i in range(6)}
        result = _gather_boundary_context(sentences, ownership, 0, 5, "left", 2)
        assert len(result) == 2
        assert result == [(4, "Sentence 4."), (5, "Sentence 5.")]

    def test_stops_at_different_group(self) -> None:
        sentences = self._make_sentences(4)
        # Group 0 owns 0,1; group 1 owns 2; group 0 owns 3 (non-contiguous)
        ownership = {0: 0, 1: 0, 2: 1, 3: 0}
        result = _gather_boundary_context(sentences, ownership, 0, 1, "left", 5)
        # Walks left from idx=1: ownership[1]==0 ✓, ownership[0]==0 ✓ → both
        assert result == [(0, "Sentence 0."), (1, "Sentence 1.")]

    def test_right_stops_when_ownership_changes(self) -> None:
        sentences = self._make_sentences(4)
        ownership = {0: 0, 1: 1, 2: 0, 3: 1}
        result = _gather_boundary_context(sentences, ownership, 1, 1, "right", 5)
        # From idx=1, ownership[1]==1 ✓, then ownership[2]==0 → stop
        assert result == [(1, "Sentence 1.")]

    def test_empty_when_start_not_owned_by_group(self) -> None:
        sentences = self._make_sentences(3)
        ownership = {0: 0, 1: 0, 2: 1}
        # Ask for group 1 context starting at idx 0 (which belongs to group 0)
        result = _gather_boundary_context(sentences, ownership, 1, 0, "right", 3)
        assert result == []
