"""Unit tests for the MarkedText chunking module."""

import re

import pytest

from txt_splitt.sentences.chunkers import OverlapChunker, SizeBasedChunker
from txt_splitt.sentences.types import MarkedText


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


class TestOverlapChunker:
    def test_small_text_returns_single_chunk(self) -> None:
        mt = MarkedText(tagged_text="{0} Hello\n{1} World", sentence_count=2)
        chunker = OverlapChunker(max_chars=1000, overlap_chars=50)
        result = chunker.chunk(mt)
        assert result == [mt]

    def test_max_chars_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="max_chars must be positive"):
            OverlapChunker(max_chars=0, overlap_chars=0)

    def test_overlap_chars_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="overlap_chars must be non-negative"):
            OverlapChunker(max_chars=100, overlap_chars=-1)

    def test_overlap_chars_equals_max_chars_raises(self) -> None:
        msg = "overlap_chars must be less than max_chars"
        with pytest.raises(ValueError, match=msg):
            OverlapChunker(max_chars=100, overlap_chars=100)

    def test_overlap_chars_exceeds_max_chars_raises(self) -> None:
        msg = "overlap_chars must be less than max_chars"
        with pytest.raises(ValueError, match=msg):
            OverlapChunker(max_chars=100, overlap_chars=200)

    def test_first_chunk_has_no_overlap(self) -> None:
        """First chunk should start from the very first line."""
        lines = [f"{{{i}}} Sentence number {i} text." for i in range(10)]
        text = "\n".join(lines)
        chunker = OverlapChunker(max_chars=120, overlap_chars=30)
        result = chunker.chunk(MarkedText(tagged_text=text, sentence_count=10))

        assert result[0].tagged_text.startswith("{0}")

    def test_subsequent_chunks_start_with_overlap(self) -> None:
        """Chunks after the first should begin with lines from previous chunk."""
        lines = [f"{{{i}}} Sentence number {i} here." for i in range(10)]
        text = "\n".join(lines)
        chunker = OverlapChunker(max_chars=120, overlap_chars=30)
        result = chunker.chunk(MarkedText(tagged_text=text, sentence_count=10))

        assert len(result) >= 2
        # Last line(s) of chunk 0 should appear at start of chunk 1.
        chunk0_lines = result[0].tagged_text.split("\n")
        chunk1_lines = result[1].tagged_text.split("\n")
        # The overlap prefix of chunk 1 must be a suffix of chunk 0.
        overlap_count = 0
        for line in chunk1_lines:
            if line in chunk0_lines:
                overlap_count += 1
            else:
                break
        assert overlap_count > 0

    def test_every_chunk_respects_max_chars(self) -> None:
        lines = [f"{{{i}}} Sentence number {i} with more text." for i in range(20)]
        text = "\n".join(lines)
        chunker = OverlapChunker(max_chars=150, overlap_chars=40)
        result = chunker.chunk(MarkedText(tagged_text=text, sentence_count=20))

        for chunk in result:
            # Oversized single lines are the only allowed exception, but they
            # are single lines and thus have no overlap contribution.
            if chunk.sentence_count > 1:
                assert len(chunk.tagged_text) <= 150

    def test_all_original_lines_present(self) -> None:
        lines = [f"{{{i}}} Sentence {i}." for i in range(15)]
        text = "\n".join(lines)
        chunker = OverlapChunker(max_chars=100, overlap_chars=20)
        result = chunker.chunk(MarkedText(tagged_text=text, sentence_count=15))

        all_lines_in_chunks: set[str] = set()
        for chunk in result:
            for line in chunk.tagged_text.split("\n"):
                all_lines_in_chunks.add(line)

        for line in lines:
            assert line in all_lines_in_chunks

    def test_single_oversized_line_not_split(self) -> None:
        long_line = "{0} " + "x" * 200
        mt = MarkedText(tagged_text=long_line, sentence_count=1)
        chunker = OverlapChunker(max_chars=50, overlap_chars=10)
        result = chunker.chunk(mt)
        assert len(result) == 1
        assert result[0].tagged_text == long_line

    def test_overlap_zero_no_duplication(self) -> None:
        """With overlap_chars=0 chunks should be contiguous with no repeated lines."""
        lines = [f"{{{i}}} Sentence number {i}." for i in range(10)]
        text = "\n".join(lines)
        chunker = OverlapChunker(max_chars=120, overlap_chars=0)
        result = chunker.chunk(MarkedText(tagged_text=text, sentence_count=10))

        reconstructed = "\n".join(chunk.tagged_text for chunk in result)
        assert reconstructed == text

    def test_sentence_count_matches_lines(self) -> None:
        lines = [f"{{{i}}} Sentence {i} with words." for i in range(12)]
        text = "\n".join(lines)
        chunker = OverlapChunker(max_chars=120, overlap_chars=30)
        result = chunker.chunk(MarkedText(tagged_text=text, sentence_count=12))

        for chunk in result:
            assert chunk.sentence_count == chunk.tagged_text.count("\n") + 1

    def test_overlap_meets_char_target(self) -> None:
        """Overlap portion should accumulate ≥ overlap_chars characters."""
        lines = [f"{{{i}}} This is sentence number {i} here." for i in range(20)]
        text = "\n".join(lines)
        overlap_target = 60
        chunker = OverlapChunker(max_chars=200, overlap_chars=overlap_target)
        result = chunker.chunk(MarkedText(tagged_text=text, sentence_count=20))

        assert len(result) >= 2
        for idx in range(1, len(result)):
            prev_lines = result[idx - 1].tagged_text.split("\n")
            curr_lines = result[idx].tagged_text.split("\n")
            # Count how many leading lines of curr are also in prev tail.
            overlap_text_parts: list[str] = []
            for line in curr_lines:
                if line in prev_lines:
                    overlap_text_parts.append(line)
                else:
                    break
            if overlap_text_parts:
                overlap_len = (
                    sum(len(p) for p in overlap_text_parts)
                    + len(overlap_text_parts)
                    - 1
                )
                assert overlap_len >= overlap_target

    def test_default_params(self) -> None:
        chunker = OverlapChunker()
        short = MarkedText(tagged_text="{0} Hello", sentence_count=1)
        assert chunker.chunk(short) == [short]

    def test_overlap_start_is_extended_to_marker_boundary(self) -> None:
        """Overlap should not begin with a continuation line without {N} marker."""
        lines = [
            "{0} " + "A" * 20,
            "tail0xxxxxx",
            "{1} B",
            "tail1yyyyyyyyyyyyyyyyyyyy",
            "{2} " + "C" * 20,
        ]
        text = "\n".join(lines)
        chunker = OverlapChunker(max_chars=70, overlap_chars=10)

        result = chunker.chunk(MarkedText(tagged_text=text, sentence_count=3))

        assert len(result) >= 2
        first_line = result[1].tagged_text.split("\n", maxsplit=1)[0]
        assert re.match(r"^\{\d+\}(?:\s|$)", first_line) is not None
