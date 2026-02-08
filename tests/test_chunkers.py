"""Unit tests for the MarkedText chunking module."""

import pytest

from txt_splitt.chunkers import SizeBasedChunker
from txt_splitt.types import MarkedText


class TestSizeBasedChunker:
    def test_small_text_returns_single_chunk(self) -> None:
        mt = MarkedText(tagged_text="{0} Hello\n{1} World", sentence_count=2)
        chunker = SizeBasedChunker(max_chars=1000)
        result = chunker.chunk(mt)
        assert result == [mt]

    def test_text_exactly_at_limit_returns_single_chunk(self) -> None:
        text = "{0} A\n{1} B"
        mt = MarkedText(tagged_text=text, sentence_count=2)
        chunker = SizeBasedChunker(max_chars=len(text))
        result = chunker.chunk(mt)
        assert result == [mt]

    def test_splits_into_multiple_chunks(self) -> None:
        lines = [f"{{{i}}} Sentence number {i}." for i in range(10)]
        text = "\n".join(lines)
        chunker = SizeBasedChunker(max_chars=120)
        result = chunker.chunk(MarkedText(tagged_text=text, sentence_count=10))

        assert len(result) >= 2
        reconstructed = "\n".join(chunk.tagged_text for chunk in result)
        assert reconstructed == text

    def test_chunk_sentence_counts_match_lines(self) -> None:
        lines = [f"{{{i}}} Sentence number {i}." for i in range(10)]
        text = "\n".join(lines)
        chunker = SizeBasedChunker(max_chars=120)
        result = chunker.chunk(MarkedText(tagged_text=text, sentence_count=10))

        for chunk in result:
            assert chunk.sentence_count == chunk.tagged_text.count("\n") + 1

    def test_preserves_original_sentence_numbers(self) -> None:
        lines = [f"{{{i}}} Text {i}" for i in range(6)]
        text = "\n".join(lines)
        chunker = SizeBasedChunker(max_chars=40)
        result = chunker.chunk(MarkedText(tagged_text=text, sentence_count=6))

        all_text = "\n".join(c.tagged_text for c in result)
        for i in range(6):
            assert f"{{{i}}}" in all_text

    def test_single_oversized_line_not_split(self) -> None:
        long_line = "{0} " + "x" * 200
        mt = MarkedText(tagged_text=long_line, sentence_count=1)
        chunker = SizeBasedChunker(max_chars=50)
        result = chunker.chunk(mt)
        assert len(result) == 1
        assert result[0].tagged_text == long_line

    def test_oversized_line_among_normal_lines(self) -> None:
        lines = [
            "{0} Short",
            "{1} " + "x" * 200,
            "{2} Also short",
        ]
        text = "\n".join(lines)
        chunker = SizeBasedChunker(max_chars=50)
        result = chunker.chunk(MarkedText(tagged_text=text, sentence_count=3))

        all_text = "\n".join(c.tagged_text for c in result)
        assert all_text == text

    def test_empty_tagged_text(self) -> None:
        mt = MarkedText(tagged_text="", sentence_count=0)
        chunker = SizeBasedChunker(max_chars=100)
        result = chunker.chunk(mt)
        assert result == [mt]

    def test_max_chars_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="max_chars must be positive"):
            SizeBasedChunker(max_chars=0)

    def test_max_chars_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="max_chars must be positive"):
            SizeBasedChunker(max_chars=-10)

    def test_chunk_sentence_counts_sum_to_original(self) -> None:
        lines = [f"{{{i}}} Sentence {i} with some text." for i in range(20)]
        text = "\n".join(lines)
        mt = MarkedText(tagged_text=text, sentence_count=20)
        chunker = SizeBasedChunker(max_chars=200)
        result = chunker.chunk(mt)

        total_sentences = sum(c.sentence_count for c in result)
        assert total_sentences == 20

    def test_default_max_chars(self) -> None:
        chunker = SizeBasedChunker()
        short = MarkedText(tagged_text="{0} Hello", sentence_count=1)
        assert chunker.chunk(short) == [short]
