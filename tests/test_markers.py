"""Tests for marker strategies."""

from txt_splitt._markers import BracketMarker
from txt_splitt._types import MarkedText, Sentence


class TestBracketMarker:
    def setup_method(self) -> None:
        self.marker = BracketMarker()

    def test_basic_marking(self) -> None:
        text = "Hello world. This is a test."
        sentences = [
            Sentence(index=0, start=0, end=12, text="Hello world."),
            Sentence(index=1, start=13, end=28, text="This is a test."),
        ]
        result = self.marker.mark(text, sentences)
        assert result == MarkedText(
            tagged_text="{0} Hello world.\n{1} This is a test.",
            sentence_count=2,
        )

    def test_empty_sentences_with_text(self) -> None:
        text = "Some text"
        result = self.marker.mark(text, [])
        assert result.sentence_count == 1
        assert result.tagged_text == "{0} Some text"

    def test_empty_sentences_empty_text(self) -> None:
        result = self.marker.mark("", [])
        assert result.sentence_count == 0
        assert result.tagged_text == ""

    def test_sentence_count(self) -> None:
        text = "A. B. C."
        sentences = [
            Sentence(index=0, start=0, end=2, text="A."),
            Sentence(index=1, start=3, end=5, text="B."),
            Sentence(index=2, start=6, end=8, text="C."),
        ]
        result = self.marker.mark(text, sentences)
        assert result.sentence_count == 3

    def test_marker_format(self) -> None:
        text = "First sentence."
        sentences = [
            Sentence(index=0, start=0, end=15, text="First sentence."),
        ]
        result = self.marker.mark(text, sentences)
        assert result.tagged_text.startswith("{0} ")
