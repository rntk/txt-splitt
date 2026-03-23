"""Tests for sentence length normalization."""

import pytest

from txt_splitt.sentences.normalizers import (
    NormalizingSplitter,
    _find_split_point,
    _merge_short,
    _reindex,
    _split_long,
)
from txt_splitt.sentences.splitters import SparseRegexSentenceSplitter
from txt_splitt.sentences.types import Sentence

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make(index: int, start: int, text: str) -> Sentence:
    """Shortcut to build a Sentence with correct end offset."""
    return Sentence(index=index, start=start, end=start + len(text), text=text)


def _assert_offsets(sentences: list[Sentence], text: str) -> None:
    """Verify that every sentence's text matches the original slice."""
    for s in sentences:
        assert text[s.start : s.end] == s.text, (
            f"Offset mismatch for index {s.index}: "
            f"text[{s.start}:{s.end}]={text[s.start : s.end]!r} != {s.text!r}"
        )


def _assert_sequential_indices(sentences: list[Sentence]) -> None:
    """Verify indices are 0, 1, 2, …"""
    for i, s in enumerate(sentences):
        assert s.index == i


# ===================================================================
# NormalizingSplitter — construction
# ===================================================================


class TestNormalizingSplitterInit:
    def test_max_must_exceed_min(self) -> None:
        with pytest.raises(ValueError, match="max_length.*must be greater"):
            NormalizingSplitter(
                SparseRegexSentenceSplitter(), min_length=100, max_length=50
            )

    def test_equal_lengths_rejected(self) -> None:
        with pytest.raises(ValueError):
            NormalizingSplitter(
                SparseRegexSentenceSplitter(), min_length=100, max_length=100
            )


# ===================================================================
# Merge tests
# ===================================================================


class TestMergeShort:
    def test_no_short_sentences_unchanged(self) -> None:
        text = (
            "This is a reasonably long sentence one. "
            "This is a reasonably long sentence two."
        )
        sents = [
            _make(0, 0, "This is a reasonably long sentence one."),
            _make(1, 40, "This is a reasonably long sentence two."),
        ]
        result = _merge_short(sents, text, min_length=20)
        assert len(result) == 2

    def test_short_sentence_merged_with_previous(self) -> None:
        text = (
            "This is a normal length sentence here. "
            "OK. Another normal sentence follows."
        )
        sents = SparseRegexSentenceSplitter().split(text)
        result = _merge_short(sents, text, min_length=20)
        # "OK." is short → merges with previous
        assert len(result) == 2
        assert "OK." in result[0].text
        _assert_offsets(result, text)

    def test_first_sentence_short_merges_forward(self) -> None:
        text = "Hi. This is a much longer second sentence here."
        sents = SparseRegexSentenceSplitter().split(text)
        result = _merge_short(sents, text, min_length=20)
        assert len(result) == 1
        assert result[0].text == text
        _assert_offsets(result, text)

    def test_multiple_short_sentences_merge(self) -> None:
        text = "A. B. C. This is a longer sentence for testing."
        sents = SparseRegexSentenceSplitter().split(text)
        result = _merge_short(sents, text, min_length=20)
        # All short ones should merge; at most 2 results
        assert len(result) <= 2
        _assert_offsets(result, text)

    def test_single_sentence_unchanged(self) -> None:
        text = "Short."
        sents = [_make(0, 0, "Short.")]
        result = _merge_short(sents, text, min_length=20)
        assert len(result) == 1
        assert result[0].text == "Short."

    def test_all_short_merge_into_one(self) -> None:
        text = "A. B. C."
        sents = SparseRegexSentenceSplitter().split(text)
        result = _merge_short(sents, text, min_length=20)
        # First is short -> pending. Merges with second.
        # Result still short -> merges with third.
        # Actually: pending "A." merges with "B." (result: "A. B.").
        # Then "C." is short → merges with previous "A. B." → result "A. B. C."
        assert len(result) == 1
        _assert_offsets(result, text)

    def test_empty_list(self) -> None:
        result = _merge_short([], "", min_length=20)
        assert result == []

    def test_merged_text_from_original(self) -> None:
        # Ensure merged text is sliced from original,
        # including inter-sentence whitespace.
        text = "Yes.\n\nThis is the next paragraph with enough length."
        sents = SparseRegexSentenceSplitter().split(text)
        assert len(sents) == 2
        result = _merge_short(sents, text, min_length=20)
        assert len(result) == 1
        # The merged text should include the newlines
        assert "\n\n" in result[0].text
        _assert_offsets(result, text)


# ===================================================================
# Split tests
# ===================================================================


class TestSplitLong:
    def test_no_long_sentences_unchanged(self) -> None:
        text = "Short sentence."
        sents = [_make(0, 0, text)]
        result = _split_long(sents, text, max_length=100)
        assert len(result) == 1

    def test_split_at_semicolon(self) -> None:
        text = "First clause here; second clause here"
        sents = [_make(0, 0, text)]
        result = _split_long(sents, text, max_length=30)
        assert len(result) == 2
        assert result[0].text == "First clause here;"
        _assert_offsets(result, text)

    def test_split_at_comma_conjunction(self) -> None:
        text = (
            "This is the first long clause of the sentence, "
            "and this is the second equally long clause"
        )
        sents = [_make(0, 0, text)]
        result = _split_long(sents, text, max_length=60)
        assert len(result) == 2
        # Split happens after ", and ":
        # the comma+conjunction pattern returns match.end().
        _assert_offsets(result, text)

    def test_split_at_comma(self) -> None:
        text = (
            "No semicolons here, just a comma separating two "
            "reasonably long parts of this text"
        )
        sents = [_make(0, 0, text)]
        result = _split_long(sents, text, max_length=50)
        assert len(result) >= 2
        _assert_offsets(result, text)

    def test_split_at_word_boundary(self) -> None:
        # No punctuation at all — should split at space near midpoint
        text = "word " * 60  # 300 chars, no commas or semicolons
        text = text.strip()
        sents = [_make(0, 0, text)]
        result = _split_long(sents, text, max_length=100)
        assert len(result) >= 2
        _assert_offsets(result, text)

    def test_recursive_split(self) -> None:
        # Very long text should be split multiple times
        text = "; ".join(f"clause number {i} with some padding text" for i in range(10))
        sents = [_make(0, 0, text)]
        result = _split_long(sents, text, max_length=80)
        assert len(result) >= 3
        _assert_offsets(result, text)

    def test_split_preserves_all_content(self) -> None:
        text = "Alpha bravo charlie; delta echo foxtrot; golf hotel india"
        sents = [_make(0, 0, text)]
        result = _split_long(sents, text, max_length=30)
        # All original content should be recoverable from the pieces
        reconstructed = " ".join(s.text for s in result)
        for word in [
            "Alpha",
            "bravo",
            "charlie",
            "delta",
            "echo",
            "foxtrot",
            "golf",
            "hotel",
            "india",
        ]:
            assert word in reconstructed


# ===================================================================
# _find_split_point tests
# ===================================================================


class TestFindSplitPoint:
    def test_prefers_semicolon(self) -> None:
        text = "aaa, bbb; ccc, ddd"
        offset = _find_split_point(text)
        # Should split after semicolon
        assert text[offset - 1] == ";"

    def test_comma_conjunction_over_plain_comma(self) -> None:
        text = "first part, second part, and the third part"
        offset = _find_split_point(text)
        # Should prefer ", and" over plain ","
        assert text[offset:].lstrip().startswith("the third")

    def test_space_fallback(self) -> None:
        text = "no punctuation here just words flowing along"
        offset = _find_split_point(text)
        # Should split at a space
        assert text[offset - 1] == " " or offset == len(text) // 2


# ===================================================================
# Reindex
# ===================================================================


class TestReindex:
    def test_sequential_indices(self) -> None:
        sents = [_make(99, 0, "a"), _make(99, 2, "b"), _make(99, 4, "c")]
        result = _reindex(sents)
        _assert_sequential_indices(result)
        assert result[0].index == 0
        assert result[2].index == 2


# ===================================================================
# Integration: NormalizingSplitter end-to-end
# ===================================================================


class TestNormalizingSplitterIntegration:
    def setup_method(self) -> None:
        self.splitter = NormalizingSplitter(
            SparseRegexSentenceSplitter(), min_length=20, max_length=100
        )

    def test_normal_text_unchanged(self) -> None:
        text = "This is a normal sentence. Another normal sentence here."
        result = self.splitter.split(text)
        assert len(result) == 2
        _assert_offsets(result, text)
        _assert_sequential_indices(result)

    def test_empty_text(self) -> None:
        assert self.splitter.split("") == []

    def test_short_merged_long_split(self) -> None:
        short = "Ok."
        medium = "This is a medium length sentence here."
        # Build a long sentence > 100 chars
        long_sent = (
            "This is a very long sentence that keeps going, and going, "
            "and going, and going until it exceeds the maximum "
            "character limit we set"
        )
        text = f"{short} {medium} {long_sent}."
        result = self.splitter.split(text)
        # "Ok." should merge, long sentence should split
        for s in result:
            assert len(s.text) >= 1  # no empty sentences
        _assert_offsets(result, text)
        _assert_sequential_indices(result)

    def test_protocol_compatible(self) -> None:
        """NormalizingSplitter satisfies SentenceSplitter protocol."""

        splitter = NormalizingSplitter(SparseRegexSentenceSplitter())
        # Structural typing — just verify the method exists and works
        assert hasattr(splitter, "split")
        result = splitter.split("Hello world. This is a test.")
        assert isinstance(result, list)
        assert all(isinstance(s, Sentence) for s in result)

    def test_offsets_accurate_after_both_phases(self) -> None:
        text = (
            "Hi. "
            "This is the first real sentence with enough content to be kept. "
            "A second proper sentence follows here with more text to read; "
            "and then a third clause appears, which is also reasonably long, "
            "and continues further with additional words to push it over the limit. "
            "End."
        )
        splitter = NormalizingSplitter(
            SparseRegexSentenceSplitter(), min_length=10, max_length=80
        )
        result = splitter.split(text)
        _assert_offsets(result, text)
        _assert_sequential_indices(result)

    def test_disabled_normalization(self) -> None:
        """With extreme params, output matches inner splitter."""
        inner = SparseRegexSentenceSplitter()
        wrapper = NormalizingSplitter(inner, min_length=1, max_length=999999)
        text = "Hello world. This is a test. Another one."
        raw = inner.split(text)
        normalized = wrapper.split(text)
        assert len(raw) == len(normalized)
        for r, n in zip(raw, normalized, strict=True):
            assert r.text == n.text
            assert r.start == n.start
            assert r.end == n.end
