"""Tests for OverlapChunker."""

import pytest

from txt_splitt.sentences.chunkers import OverlapChunker
from txt_splitt.sentences.types import MarkedText


def _make_marked(lines: list[str]) -> MarkedText:
    """Build a MarkedText from a list of tagged lines."""
    text = "\n".join(lines)
    return MarkedText(tagged_text=text, sentence_count=len(lines))


class TestOverlapChunker:
    def test_single_chunk_when_text_fits(self) -> None:
        mt = _make_marked(["{0} Hello world.", "{1} Second sentence."])
        chunker = OverlapChunker(max_chars=200, overlap_chars=0)
        result = chunker.chunk(mt)
        assert len(result) == 1
        assert result[0] is mt

    def test_splits_on_line_boundaries(self) -> None:
        lines = [f"{{{i}}} Sentence number {i}." for i in range(20)]
        mt = _make_marked(lines)
        chunker = OverlapChunker(max_chars=100, overlap_chars=0)
        result = chunker.chunk(mt)
        assert len(result) > 1
        for chunk in result:
            assert len(chunk.tagged_text) <= 100 or chunk.sentence_count == 1

    def test_respects_max_chars(self) -> None:
        lines = [
            f"{{{i}}} Sentence number {i} with some content."
            for i in range(50)
        ]
        mt = _make_marked(lines)
        chunker = OverlapChunker(max_chars=200, overlap_chars=0)
        result = chunker.chunk(mt)
        assert len(result) > 1
        for chunk in result:
            if chunk.sentence_count > 1:
                assert len(chunk.tagged_text) <= 200

    def test_balanced_distribution(self) -> None:
        lines = [f"{{{i}}} Line {i} text." for i in range(30)]
        mt = _make_marked(lines)
        chunker = OverlapChunker(max_chars=150, overlap_chars=0)
        result = chunker.chunk(mt)
        sizes = [len(c.tagged_text) for c in result]
        avg = sum(sizes) / len(sizes)
        for s in sizes:
            assert s >= avg * 0.5, f"Chunk size {s} too small vs avg {avg}"

    def test_overlap_preserved(self) -> None:
        lines = [f"{{{i}}} Sentence {i}." for i in range(20)]
        mt = _make_marked(lines)
        chunker = OverlapChunker(max_chars=120, overlap_chars=30)
        result = chunker.chunk(mt)
        assert len(result) >= 2
        first_lines = result[0].tagged_text.split("\n")
        second_lines = result[1].tagged_text.split("\n")
        assert any(
            line in first_lines for line in second_lines[:3]
        ), "Overlap lines not found in second chunk"

    def test_empty_text(self) -> None:
        mt = MarkedText(tagged_text="", sentence_count=0)
        chunker = OverlapChunker(max_chars=100, overlap_chars=0)
        result = chunker.chunk(mt)
        assert len(result) == 1
        assert result[0].tagged_text == ""

    def test_line_boundaries_only(self) -> None:
        lines = [f"{{{i}}} Word{i}" for i in range(10)]
        mt = _make_marked(lines)
        chunker = OverlapChunker(max_chars=40, overlap_chars=0)
        result = chunker.chunk(mt)
        for chunk in result:
            for line in chunk.tagged_text.split("\n"):
                if line:
                    assert line.startswith("{")

    def test_no_empty_chunks(self) -> None:
        lines = [f"{{{i}}} Text {i}." for i in range(15)]
        mt = _make_marked(lines)
        chunker = OverlapChunker(max_chars=80, overlap_chars=0)
        result = chunker.chunk(mt)
        for chunk in result:
            assert chunk.tagged_text.strip() != ""

    def test_max_chars_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="max_chars must be positive"):
            OverlapChunker(max_chars=0)

    def test_overlap_chars_must_be_non_negative(self) -> None:
        with pytest.raises(
            ValueError, match="overlap_chars must be non-negative"
        ):
            OverlapChunker(overlap_chars=-1)

    def test_overlap_must_be_less_than_max(self) -> None:
        with pytest.raises(
            ValueError, match="overlap_chars must be less than max_chars"
        ):
            OverlapChunker(max_chars=100, overlap_chars=100)
