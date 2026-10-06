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
            assert len(chunk.tagged_text) <= 100

    def test_respects_max_chars(self) -> None:
        lines = [f"{{{i}}} Sentence number {i} with some content." for i in range(50)]
        mt = _make_marked(lines)
        chunker = OverlapChunker(max_chars=200, overlap_chars=0)
        result = chunker.chunk(mt)
        assert len(result) > 1
        for chunk in result:
            assert len(chunk.tagged_text) <= 200

    @pytest.mark.parametrize("overlap_chars", [0, 30, 99])
    def test_oversized_line_preserves_payload_and_marker(
        self, overlap_chars: int
    ) -> None:
        payload = "0123456789" * 53
        mt = _make_marked(["{0} Before.", "{1} " + payload, "{2} After."])
        result = OverlapChunker(max_chars=100, overlap_chars=overlap_chars).chunk(mt)

        assert all(len(chunk.tagged_text) <= 100 for chunk in result)
        fragments: list[str] = []
        for chunk in result:
            lines = chunk.tagged_text.split("\n")
            assert lines[0].startswith("{")
            fragments.extend(line[4:] for line in lines if line.startswith("{1} "))
        # Full-budget fragments cannot be carried as overlap.
        assert "".join(fragments) == payload
        assert result[0].tagged_text.startswith("{0} Before.")
        assert result[-1].tagged_text.endswith("{2} After.")

    def test_oversized_continuation_inherits_marker(self) -> None:
        payload = "x" * 250
        mt = MarkedText(tagged_text="{42} Introduction\n" + payload, sentence_count=1)
        result = OverlapChunker(max_chars=100, overlap_chars=0).chunk(mt)

        assert all(len(chunk.tagged_text) <= 100 for chunk in result)
        assert (
            "".join(
                line[5:]
                for chunk in result
                for line in chunk.tagged_text.split("\n")
                if line.startswith("{42} ") and line != "{42} Introduction"
            )
            == payload
        )

    def test_overlap_leaves_room_for_new_content(self) -> None:
        lines = ["{0} " + "a" * 86, "{1} " + "b" * 86, "{2} End."]
        result = OverlapChunker(max_chars=100, overlap_chars=50).chunk(
            _make_marked(lines)
        )

        assert all(len(chunk.tagged_text) <= 100 for chunk in result)
        assert [chunk.tagged_text for chunk in result] == [
            lines[0],
            "\n".join(lines[1:]),
        ]

    def test_trimmed_overlap_does_not_start_on_continuation(self) -> None:
        lines = ["{0} " + "a" * 66, "continuation", "{1} " + "b" * 66]
        result = OverlapChunker(max_chars=100, overlap_chars=10).chunk(
            _make_marked(lines)
        )

        assert all(len(chunk.tagged_text) <= 100 for chunk in result)
        assert [chunk.tagged_text for chunk in result] == [
            "\n".join(lines[:2]),
            lines[2],
        ]

    @pytest.mark.parametrize("overlap_chars", [0, 10])
    def test_continuation_boundary_without_overlap_stays_bounded(
        self, overlap_chars: int
    ) -> None:
        lines = ["{0} " + "a" * 86, "b" * 70, "{1} End."]
        mt = MarkedText(tagged_text="\n".join(lines), sentence_count=2)
        result = OverlapChunker(max_chars=100, overlap_chars=overlap_chars).chunk(mt)

        # Zero overlap or trimming can expose an unmarked continuation.
        assert [chunk.tagged_text for chunk in result] == [
            lines[0],
            "\n".join(lines[1:]),
        ]
        assert all(len(chunk.tagged_text) <= 100 for chunk in result)
        assert "\n".join(chunk.tagged_text for chunk in result) == mt.tagged_text

    def test_unmarked_oversized_line(self) -> None:
        mt = MarkedText(tagged_text="x" * 250, sentence_count=1)
        result = OverlapChunker(max_chars=100, overlap_chars=0).chunk(mt)
        assert all(len(chunk.tagged_text) <= 100 for chunk in result)
        assert "".join(chunk.tagged_text for chunk in result) == mt.tagged_text

    def test_limit_too_small_for_marker(self) -> None:
        mt = _make_marked(["{123} Long text"])
        with pytest.raises(ValueError, match="sentence marker and text"):
            OverlapChunker(max_chars=6, overlap_chars=0).chunk(mt)

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
        assert any(line in first_lines for line in second_lines[:3]), (
            "Overlap lines not found in second chunk"
        )

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
        with pytest.raises(ValueError, match="overlap_chars must be non-negative"):
            OverlapChunker(overlap_chars=-1)

    def test_overlap_must_be_less_than_max(self) -> None:
        with pytest.raises(
            ValueError, match="overlap_chars must be less than max_chars"
        ):
            OverlapChunker(max_chars=100, overlap_chars=100)
