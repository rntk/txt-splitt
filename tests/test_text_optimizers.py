"""Tests for text normalization and OptimizingMarker."""

from txt_splitt.markers import BracketMarker
from txt_splitt.text_optimizers import (
    OptimizingMarker,
    collapse_whitespace,
    decode_html_entities,
    is_content_free,
    normalize_for_llm,
    strip_zero_width_chars,
)
from txt_splitt.types import MarkedText, Sentence


class TestDecodeHtmlEntities:
    def test_nbsp(self) -> None:
        assert decode_html_entities("&nbsp;") == "\xa0"

    def test_amp(self) -> None:
        assert decode_html_entities("&amp;") == "&"

    def test_lt_gt(self) -> None:
        assert decode_html_entities("&lt;div&gt;") == "<div>"

    def test_double_encoded_nbsp(self) -> None:
        # &amp;nbsp; -> first pass: &nbsp; -> second pass: \xa0
        assert decode_html_entities("&amp;nbsp;") == "\xa0"

    def test_double_encoded_amp(self) -> None:
        assert decode_html_entities("&amp;amp;") == "&"

    def test_plain_text_unchanged(self) -> None:
        assert decode_html_entities("hello world") == "hello world"

    def test_numeric_entity(self) -> None:
        assert decode_html_entities("&#169;") == "\u00a9"  # copyright sign


class TestStripZeroWidthChars:
    def test_zero_width_non_joiner(self) -> None:
        assert strip_zero_width_chars("a\u200cb") == "ab"

    def test_zero_width_space(self) -> None:
        assert strip_zero_width_chars("a\u200bb") == "ab"

    def test_zero_width_joiner(self) -> None:
        assert strip_zero_width_chars("a\u200db") == "ab"

    def test_bom(self) -> None:
        assert strip_zero_width_chars("\ufeffhello") == "hello"

    def test_word_joiner(self) -> None:
        assert strip_zero_width_chars("a\u2060b") == "ab"

    def test_mixed_invisible_chars(self) -> None:
        result = strip_zero_width_chars("\u200c\u200b\u200d\u200e\u200f")
        assert result == ""

    def test_regular_text_unchanged(self) -> None:
        assert strip_zero_width_chars("hello world") == "hello world"

    def test_invisible_range_2061_to_2064(self) -> None:
        text = "a\u2061\u2062\u2063\u2064b"
        assert strip_zero_width_chars(text) == "ab"


class TestCollapseWhitespace:
    def test_multiple_spaces(self) -> None:
        assert collapse_whitespace("a   b") == "a b"

    def test_nbsp_collapse(self) -> None:
        assert collapse_whitespace("a\xa0\xa0\xa0b") == "a b"

    def test_mixed_horizontal_ws(self) -> None:
        assert collapse_whitespace("a \t \xa0 b") == "a b"

    def test_excess_newlines(self) -> None:
        assert collapse_whitespace("a\n\n\n\nb") == "a\n\nb"

    def test_two_newlines_preserved(self) -> None:
        assert collapse_whitespace("a\n\nb") == "a\n\nb"

    def test_single_newline_preserved(self) -> None:
        assert collapse_whitespace("a\nb") == "a\nb"

    def test_unicode_spaces(self) -> None:
        # EN SPACE, EM SPACE, THIN SPACE
        assert collapse_whitespace("a\u2002\u2003\u2009b") == "a b"


class TestNormalizeForLlm:
    def test_filler_sentence(self) -> None:
        filler = "&nbsp;\u200c&nbsp;\u200c&nbsp;\u200c&nbsp;"
        result = normalize_for_llm(filler)
        assert result == ""

    def test_real_text_preserved(self) -> None:
        result = normalize_for_llm("Hello world, this is a test.")
        assert result == "Hello world, this is a test."

    def test_combined_cleanup(self) -> None:
        text = "&amp;nbsp;\u200c  hello  \u200b  world  "
        result = normalize_for_llm(text)
        # \xa0 from decoded &nbsp; is collapsed to space, then stripped
        assert result == "hello world"

    def test_strips_leading_trailing(self) -> None:
        result = normalize_for_llm("  hello  ")
        assert result == "hello"

    def test_empty_string(self) -> None:
        assert normalize_for_llm("") == ""

    def test_only_entities(self) -> None:
        assert normalize_for_llm("&nbsp;&nbsp;&nbsp;") == ""


class TestIsContentFree:
    def test_filler_is_content_free(self) -> None:
        assert is_content_free("&nbsp;\u200c&nbsp;\u200c&nbsp;") is True

    def test_only_spaces(self) -> None:
        assert is_content_free("     ") is True

    def test_only_zero_width(self) -> None:
        assert is_content_free("\u200b\u200c\u200d") is True

    def test_empty(self) -> None:
        assert is_content_free("") is True

    def test_real_text_has_content(self) -> None:
        assert is_content_free("Hello world") is False

    def test_text_with_entities_has_content(self) -> None:
        assert is_content_free("&amp; hello") is False

    def test_punctuation_only_is_content_free(self) -> None:
        assert is_content_free("...---") is True

    def test_number_has_content(self) -> None:
        assert is_content_free("42") is False


class TestOptimizingMarker:
    def setup_method(self) -> None:
        self.marker = OptimizingMarker(BracketMarker())

    def test_normalizes_sentence_text(self) -> None:
        text = "Hello&nbsp;world. &nbsp;\u200c&nbsp;"
        sentences = [
            Sentence(index=0, start=0, end=16, text="Hello&nbsp;world."),
            Sentence(index=1, start=17, end=32, text="&nbsp;\u200c&nbsp;"),
        ]
        result = self.marker.mark(text, sentences)
        # \xa0 from &nbsp; is collapsed to regular space; filler becomes ""
        # BracketMarker formats as "{N} <text>" so empty text gives "{1} "
        assert result.tagged_text == "{0} Hello world.\n{1} "
        assert result.sentence_count == 2

    def test_preserves_sentence_count(self) -> None:
        text = "A. B. C."
        sentences = [
            Sentence(index=0, start=0, end=2, text="A."),
            Sentence(index=1, start=3, end=5, text="B."),
            Sentence(index=2, start=6, end=8, text="C."),
        ]
        result = self.marker.mark(text, sentences)
        assert result.sentence_count == 3

    def test_preserves_offsets_in_delegation(self) -> None:
        """The inner marker receives sentences with original start/end."""
        captured: list[list[Sentence]] = []

        class CapturingMarker:
            def mark(self, text: str, sentences: list[Sentence]) -> MarkedText:
                captured.append(sentences)
                return MarkedText(tagged_text="", sentence_count=len(sentences))

        marker = OptimizingMarker(CapturingMarker())
        sentences = [
            Sentence(index=0, start=10, end=30, text="&nbsp;hello"),
        ]
        marker.mark("x" * 50, sentences)

        assert len(captured) == 1
        assert captured[0][0].start == 10
        assert captured[0][0].end == 30
        assert captured[0][0].text == "hello"

    def test_satisfies_marker_strategy_protocol(self) -> None:
        """OptimizingMarker has the same mark() signature as MarkerStrategy."""
        marker = OptimizingMarker(BracketMarker())
        assert hasattr(marker, "mark")
        assert callable(marker.mark)

    def test_empty_sentences(self) -> None:
        result = self.marker.mark("some text", [])
        assert result.sentence_count == 1
        assert result.tagged_text == "{0} some text"

    def test_filler_sentence_becomes_minimal(self) -> None:
        """A filler sentence normalizes to empty, costing minimal tokens."""
        filler = "&nbsp;\u200c&nbsp;\u200c&nbsp;\u200c&nbsp;\u200c" * 20
        text = f"Real content. {filler} More content."
        sentences = [
            Sentence(index=0, start=0, end=13, text="Real content."),
            Sentence(index=1, start=14, end=14 + len(filler), text=filler),
            Sentence(
                index=2,
                start=15 + len(filler),
                end=15 + len(filler) + 13,
                text="More content.",
            ),
        ]
        result = self.marker.mark(text, sentences)
        lines = result.tagged_text.split("\n")
        assert lines[0] == "{0} Real content."
        assert lines[1] == "{1} "  # filler normalizes to "", marker adds "{1} "
        assert lines[2] == "{2} More content."
        assert result.sentence_count == 3
