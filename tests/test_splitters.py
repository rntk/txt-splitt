"""Tests for sentence splitting."""

import pytest

from txt_splitt.splitters import DenseRegexSentenceSplitter, RegexSentenceSplitter
from txt_splitt.types import Sentence


class TestRegexSentenceSplitter:
    def setup_method(self) -> None:
        self.splitter = RegexSentenceSplitter()

    def test_empty_string(self) -> None:
        assert self.splitter.split("") == []

    def test_whitespace_only(self) -> None:
        assert self.splitter.split("   ") == []

    def test_single_sentence(self) -> None:
        result = self.splitter.split("Hello world.")
        assert len(result) == 1
        assert result[0] == Sentence(index=0, start=0, end=12, text="Hello world.")

    def test_multiple_sentences(self) -> None:
        text = "Hello world. This is a test. Another one."
        result = self.splitter.split(text)
        assert len(result) == 3
        assert result[0].index == 0
        assert result[0].text == "Hello world."
        assert result[1].index == 1
        assert result[1].text == "This is a test."
        assert result[2].index == 2
        assert result[2].text == "Another one."

    def test_zero_based_indices(self) -> None:
        text = "First. Second."
        result = self.splitter.split(text)
        assert result[0].index == 0
        assert result[1].index == 1

    def test_newline_splitting(self) -> None:
        text = "First paragraph.\nSecond paragraph."
        result = self.splitter.split(text)
        assert len(result) == 2
        assert result[0].text == "First paragraph."
        assert result[1].text == "Second paragraph."

    def test_char_offsets_allow_slicing(self) -> None:
        text = "Hello world. This is a test."
        result = self.splitter.split(text)
        for s in result:
            assert text[s.start : s.end] == s.text

    def test_exclamation_mark_boundary(self) -> None:
        text = "Wow! That is great."
        result = self.splitter.split(text)
        assert len(result) == 2
        assert result[0].text == "Wow!"
        assert result[1].text == "That is great."

    def test_question_mark_boundary(self) -> None:
        text = "Is this working? Yes it is."
        result = self.splitter.split(text)
        assert len(result) == 2
        assert result[0].text == "Is this working?"
        assert result[1].text == "Yes it is."

    def test_no_split_on_lowercase_after_period(self) -> None:
        text = "Version 3.14 is great."
        result = self.splitter.split(text)
        assert len(result) == 1

    def test_cyrillic_uppercase_boundary(self) -> None:
        text = "Hello world. Привет мир."
        result = self.splitter.split(text)
        assert len(result) == 2


class TestDenseRegexSentenceSplitter:
    def setup_method(self) -> None:
        self.splitter = DenseRegexSentenceSplitter(anchor_every_words=5)

    def test_empty_string(self) -> None:
        assert self.splitter.split("") == []

    def test_separator_boundaries_split(self) -> None:
        text = "First block | Second block · Third block"
        result = self.splitter.split(text)
        assert len(result) == 3
        assert result[0].text == "First block"
        assert result[1].text == "Second block"
        assert result[2].text == "Third block"

    def test_anchor_splitting_adds_density(self) -> None:
        text = "one two three four five six seven eight nine ten eleven"
        result = self.splitter.split(text)
        assert len(result) == 3
        assert result[0].text == "one two three four five"
        assert result[1].text == "six seven eight nine ten"
        assert result[2].text == "eleven"

    def test_char_offsets_allow_slicing(self) -> None:
        text = "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
        result = self.splitter.split(text)
        for s in result:
            assert text[s.start : s.end] == s.text

    def test_non_positive_anchor_words_rejected(self) -> None:
        with pytest.raises(ValueError, match="anchor_every_words must be positive"):
            DenseRegexSentenceSplitter(anchor_every_words=0)
