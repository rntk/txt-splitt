"""Tests for HTML tag cleaning and offset mapping."""

from __future__ import annotations

from typing import TypeAlias

import pytest

from txt_splitt.html_cleaners import HTMLParserTagStripCleaner, TagStripCleaner
from txt_splitt.types import OffsetMapping, OffsetSegment

# ---------------------------------------------------------------------------
# Type alias for cleaner instances used in shared mixin tests
# ---------------------------------------------------------------------------
_Cleaner: TypeAlias = TagStripCleaner | HTMLParserTagStripCleaner


class TestTagStripCleaner:
    def setup_method(self) -> None:
        self.cleaner = TagStripCleaner()

    def test_empty_string(self) -> None:
        clean, mapping = self.cleaner.clean("")
        assert clean == ""
        assert mapping.segments == ()
        assert mapping.original_length == 0
        assert mapping.clean_length == 0

    def test_no_tags(self) -> None:
        text = "Hello world"
        clean, mapping = self.cleaner.clean(text)
        assert clean == text
        assert len(mapping.segments) == 1
        assert mapping.segments[0] == OffsetSegment(
            clean_offset=0, original_offset=0, length=11
        )
        assert mapping.original_length == 11
        assert mapping.clean_length == 11

    def test_simple_tag_removal(self) -> None:
        text = "Hello <b>world</b> end"
        clean, mapping = self.cleaner.clean(text)
        assert clean == "Hello world end"
        assert mapping.original_length == len(text)
        assert mapping.clean_length == len(clean)

    def test_self_closing_tag(self) -> None:
        text = "Hello<br/>world"
        clean, mapping = self.cleaner.clean(text)
        assert clean == "Hello world"
        assert mapping.clean_length == 11
        assert mapping.to_original(5) == 5
        assert mapping.to_original(6) == 10

    def test_tag_with_attributes(self) -> None:
        text = 'A <a href="http://x.com">link</a> B'
        clean, mapping = self.cleaner.clean(text)
        assert clean == "A link B"

    def test_tag_with_quoted_angle_brackets(self) -> None:
        text = 'A <div title="a>b">X</div> B'
        clean, mapping = self.cleaner.clean(text)
        assert clean == "A X B"

    def test_multiple_tags(self) -> None:
        text = "<h1>Title</h1><p>Body</p>"
        clean, mapping = self.cleaner.clean(text)
        assert clean == "Title Body"
        assert mapping.clean_length == 10

    def test_adjacent_tags_no_text(self) -> None:
        text = "<br><hr><img/>"
        clean, mapping = self.cleaner.clean(text)
        assert clean == ""
        assert mapping.segments == ()
        assert mapping.clean_length == 0
        assert mapping.original_length == len(text)

    def test_text_entirely_tags(self) -> None:
        text = "<div></div>"
        clean, mapping = self.cleaner.clean(text)
        assert clean == ""
        assert mapping.clean_length == 0

    def test_segment_count_includes_synthetic_spaces(self) -> None:
        text = "A<b>B</b>C"
        clean, mapping = self.cleaner.clean(text)
        assert clean == "A B C"
        assert len(mapping.segments) == 5
        assert mapping.to_original(1) == 1
        assert mapping.to_original(3) == 5

    def test_mapping_lengths_consistent(self) -> None:
        text = "Hello <em>world</em>!"
        clean, mapping = self.cleaner.clean(text)
        total_seg_length = sum(seg.length for seg in mapping.segments)
        assert total_seg_length == mapping.clean_length
        assert mapping.clean_length == len(clean)
        assert mapping.original_length == len(text)


class TestOffsetMapping:
    def _make_mapping(self) -> OffsetMapping:
        """Mapping for 'Hello <b>world</b> end' -> 'Hello world end'."""
        # Original: H e l l o   < b > w o r l  d  <  /  b  >     e  n  d
        #           0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21
        # Clean:    H e l l o   w o r l  d     e  n  d
        #           0 1 2 3 4 5 6 7 8 9  10 11 12 13 14
        return OffsetMapping(
            segments=(
                OffsetSegment(clean_offset=0, original_offset=0, length=6),
                OffsetSegment(clean_offset=6, original_offset=9, length=5),
                OffsetSegment(clean_offset=11, original_offset=18, length=4),
            ),
            original_length=22,
            clean_length=15,
        )

    def test_start_of_first_segment(self) -> None:
        m = self._make_mapping()
        assert m.to_original(0) == 0

    def test_middle_of_first_segment(self) -> None:
        m = self._make_mapping()
        assert m.to_original(3) == 3

    def test_end_of_first_segment_boundary(self) -> None:
        m = self._make_mapping()
        # Position 6 in clean = start of second segment
        assert m.to_original(6) == 9

    def test_middle_of_second_segment(self) -> None:
        m = self._make_mapping()
        # Clean pos 8 = 'r' in "world", orig pos = 9 + (8-6) = 11
        assert m.to_original(8) == 11

    def test_start_of_third_segment(self) -> None:
        m = self._make_mapping()
        assert m.to_original(11) == 18

    def test_at_clean_length_returns_original_length(self) -> None:
        m = self._make_mapping()
        assert m.to_original(15) == 22

    def test_negative_raises(self) -> None:
        m = self._make_mapping()
        with pytest.raises(ValueError, match="non-negative"):
            m.to_original(-1)

    def test_beyond_clean_length_raises(self) -> None:
        m = self._make_mapping()
        with pytest.raises(ValueError, match="exceeds"):
            m.to_original(16)

    def test_empty_segments_raises(self) -> None:
        m = OffsetMapping(segments=(), original_length=10, clean_length=0)
        # clean_pos 0 == clean_length 0 → returns original_length
        assert m.to_original(0) == 10

    def test_empty_segments_nonzero_raises(self) -> None:
        m = OffsetMapping(segments=(), original_length=10, clean_length=5)
        with pytest.raises(ValueError, match="no segments"):
            m.to_original(2)

    def test_roundtrip_every_position(self) -> None:
        """For each char in clean text, to_original should give valid position."""
        original = "AB<b>CD</b>EF"
        cleaner = TagStripCleaner()
        clean, mapping = cleaner.clean(original)
        assert clean == "AB CD EF"

        expected = [0, 1, 2, 5, 6, 7, 11, 12]  # A B _ C D _ E F
        for i, expected_orig in enumerate(expected):
            assert mapping.to_original(i) == expected_orig, f"pos {i}"
        # End position
        assert mapping.to_original(8) == len(original)

    def test_identity_mapping_no_tags(self) -> None:
        """Text without tags should produce identity mapping."""
        text = "Hello world"
        cleaner = TagStripCleaner()
        clean, mapping = cleaner.clean(text)
        for i in range(len(text)):
            assert mapping.to_original(i) == i
        assert mapping.to_original(len(text)) == len(text)


class TestHTMLParserTagStripCleaner:
    def setup_method(self) -> None:
        self.cleaner = HTMLParserTagStripCleaner()

    def test_basic_tag_removal(self) -> None:
        text = "Hello <b>world</b> end"
        clean, mapping = self.cleaner.clean(text)
        assert clean == "Hello world end"
        assert mapping.original_length == len(text)
        assert mapping.clean_length == len(clean)

    def test_preserves_non_tag_angle_brackets(self) -> None:
        text = "Math: 3 < 10 > 2 and heart <3>"
        clean, mapping = self.cleaner.clean(text)
        assert clean == text
        assert mapping.segments == (
            OffsetSegment(clean_offset=0, original_offset=0, length=len(text)),
        )
        assert mapping.to_original(len(text)) == len(text)

    def test_removes_comment_and_declaration(self) -> None:
        text = "<!DOCTYPE html><!--x--><p>A</p>"
        clean, mapping = self.cleaner.clean(text)
        assert clean == "A"
        assert mapping.clean_length == 1


class _StripTagsMixin:
    """Shared strip_tags tests, parameterised via ``_make_cleaner``."""

    def _make_cleaner(
        self, strip_tags: set[str] | None = None
    ) -> TagStripCleaner | HTMLParserTagStripCleaner:
        raise NotImplementedError

    def test_strip_script_content(self) -> None:
        c = self._make_cleaner(strip_tags={"script"})
        text = "<p>Hello</p><script>var x=1;</script><p>World</p>"
        clean, mapping = c.clean(text)
        assert clean == "Hello World"
        assert mapping.original_length == len(text)
        assert mapping.clean_length == len(clean)

    def test_strip_style_content(self) -> None:
        c = self._make_cleaner(strip_tags={"style"})
        text = "<style>.a{color:red}</style><div>Text</div>"
        clean, mapping = c.clean(text)
        assert clean == "Text"

    def test_strip_multiple_tag_types(self) -> None:
        c = self._make_cleaner(strip_tags={"script", "style"})
        text = "<style>css</style><p>Keep</p><script>js</script>"
        clean, mapping = c.clean(text)
        assert clean == "Keep"

    def test_strip_nested_html_in_script(self) -> None:
        c = self._make_cleaner(strip_tags={"script"})
        text = "<script><div>ignored</div></script><p>kept</p>"
        clean, mapping = c.clean(text)
        assert clean == "kept"

    def test_default_strip_tags_none(self) -> None:
        c = self._make_cleaner()
        text = "<script>var x=1;</script>"
        clean, _ = c.clean(text)
        assert clean == "var x=1;"  # content preserved when strip_tags is None

    def test_strip_tag_with_attributes(self) -> None:
        c = self._make_cleaner(strip_tags={"script"})
        text = '<script type="text/javascript">code</script>Text'
        clean, mapping = c.clean(text)
        assert clean == "Text"

    def test_strip_preserves_surrounding_text(self) -> None:
        c = self._make_cleaner(strip_tags={"style"})
        text = "Before<style>css</style>After"
        clean, mapping = c.clean(text)
        assert clean == "Before After"
        total_seg_length = sum(seg.length for seg in mapping.segments)
        assert total_seg_length == mapping.clean_length

    def test_offset_roundtrip_with_strip(self) -> None:
        c = self._make_cleaner(strip_tags={"script"})
        # A  B  <script>ignored</script>  C  D
        # 0  1  2.......................25 26 27
        text = "AB<script>ignored</script>CD"
        clean, mapping = c.clean(text)
        assert clean == "AB CD"
        assert mapping.to_original(0) == 0  # A
        assert mapping.to_original(1) == 1  # B
        assert mapping.to_original(2) == 2  # synthetic separator
        assert mapping.to_original(3) == 26  # C
        assert mapping.to_original(4) == 27  # D
        assert mapping.to_original(5) == 28  # end

    def test_mapping_lengths_consistent_with_strip(self) -> None:
        c = self._make_cleaner(strip_tags={"script", "style"})
        text = "Hello <style>css</style><em>world</em><script>js</script>!"
        clean, mapping = c.clean(text)
        total_seg = sum(s.length for s in mapping.segments)
        assert total_seg == mapping.clean_length
        assert mapping.clean_length == len(clean)
        assert mapping.original_length == len(text)


class TestTagStripCleanerStripTags(_StripTagsMixin):
    def _make_cleaner(self, strip_tags: set[str] | None = None) -> TagStripCleaner:
        return TagStripCleaner(strip_tags=strip_tags)


class TestHTMLParserTagStripCleanerStripTags(_StripTagsMixin):
    def _make_cleaner(
        self, strip_tags: set[str] | None = None
    ) -> HTMLParserTagStripCleaner:
        return HTMLParserTagStripCleaner(strip_tags=strip_tags)

    def test_strip_self_closing_tag_does_not_stick(self) -> None:
        c = self._make_cleaner(strip_tags={"style"})
        text = "<style/>A<b>B</b>"
        clean, _ = c.clean(text)
        assert clean == "A B"

    def test_strip_unclosed_tag_to_end_of_input(self) -> None:
        c = self._make_cleaner(strip_tags={"script"})
        text = "X<script>bad<p>Y</p>"
        clean, _ = c.clean(text)
        assert clean == "X"


# ---------------------------------------------------------------------------
# Shared mixin: nested HTML position-mapping tests
# ---------------------------------------------------------------------------


class _NestedHTMLMixin:
    """Position-mapping tests for multi-level nested HTML.

    Run for both ``TagStripCleaner`` and ``HTMLParserTagStripCleaner`` via the
    concrete subclasses below.
    """

    def _make_cleaner(self) -> _Cleaner:
        raise NotImplementedError

    # -- 3 levels deep, single text node ------------------------------------

    def test_three_level_deep_nesting_single_text(self) -> None:
        """All opening tags and all closing tags are adjacent → each group
        merges into one removed span, leaving a single mapping segment."""
        # <div><p><b> at [0:11], "Hello world" at [11:22], </b></p></div> at [22:36]
        html = "<div><p><b>Hello world</b></p></div>"
        c = self._make_cleaner()
        clean, mapping = c.clean(html)

        assert clean == "Hello world"
        assert len(mapping.segments) == 1
        assert mapping.segments[0] == OffsetSegment(
            clean_offset=0, original_offset=11, length=11
        )
        assert mapping.original_length == 36
        assert mapping.clean_length == 11

        # Every character maps directly into the original text node
        for i in range(11):
            assert mapping.to_original(i) == 11 + i, f"position {i}"
        assert mapping.to_original(11) == 36  # end → original_length

    # -- text nodes interleaved with tags at multiple nesting levels ---------

    def test_text_at_multiple_nesting_levels_no_spaces(self) -> None:
        """Text at the outer and inner div level; no natural whitespace means
        synthetic separators are injected between each text run."""
        # <div> at [0:5], "Outer" at [5:10], <p> at [10:13], "Inner" at [13:18],
        # </p> at [18:22], "End" at [22:25], </div> at [25:31]
        html = "<div>Outer<p>Inner</p>End</div>"
        c = self._make_cleaner()
        clean, mapping = c.clean(html)

        assert clean == "Outer Inner End"
        assert mapping.original_length == len(html)
        assert mapping.clean_length == len(clean)

        # Precomputed expected original positions for each clean position
        expected = [5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 22, 23, 24, 31]
        for i, orig in enumerate(expected):
            assert mapping.to_original(i) == orig, (
                f"clean pos {i}: expected orig {orig}, got {mapping.to_original(i)}"
            )

        # Non-space clean characters must map to the matching character in HTML
        for i in range(len(clean)):
            if clean[i] != " ":
                orig = mapping.to_original(i)
                assert html[orig] == clean[i], (
                    f"char mismatch at clean[{i}]={clean[i]!r}, "
                    f"html[{orig}]={html[orig]!r}"
                )

    # -- adjacent closing + opening tags (paragraph boundary) ---------------

    def test_two_paragraphs_every_position(self) -> None:
        """</p><p> boundary: adjacent tags merge into one removed span;
        a synthetic separator is inserted between the two text runs."""
        # <p> at [0:3], "First." at [3:9], </p><p> merged at [9:16],
        # "Second." at [16:23], </p> at [23:27]
        html = "<p>First.</p><p>Second.</p>"
        c = self._make_cleaner()
        clean, mapping = c.clean(html)

        assert clean == "First. Second."
        assert mapping.original_length == len(html)  # 27
        assert mapping.clean_length == len(clean)  # 14

        # Every position — including synthetic separator at clean[6] → orig[9]
        expected = [3, 4, 5, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21, 22, 27]
        for i, orig in enumerate(expected):
            assert mapping.to_original(i) == orig, (
                f"clean pos {i}: expected {orig}, got {mapping.to_original(i)}"
            )

    # -- multiple self-closing tags -----------------------------------------

    def test_multiple_br_tags_every_position(self) -> None:
        """Two <br/> tags each produce a synthetic separator; text runs 'A',
        'B', 'C' remain individually addressable."""
        # "A" at [0:1], <br/> at [1:6], "B" at [6:7], <br/> at [7:12], "C" at [12:13]
        html = "A<br/>B<br/>C"
        c = self._make_cleaner()
        clean, mapping = c.clean(html)

        assert clean == "A B C"
        assert mapping.original_length == len(html)  # 13
        assert mapping.clean_length == len(clean)  # 5

        expected = [0, 1, 6, 7, 12, 13]
        for i, orig in enumerate(expected):
            assert mapping.to_original(i) == orig, (
                f"clean pos {i}: expected {orig}, got {mapping.to_original(i)}"
            )

    # -- deep nesting followed by sibling text node -------------------------

    def test_deep_nesting_plus_tail_text(self) -> None:
        """Three closing tags are adjacent and merge; the tail text ' tail' is
        separated from the inner text 'Deep' by a synthetic separator."""
        # <div><section><p> merged at [0:17], "Deep" at [17:21],
        # </p></section></div> merged at [21:41], " tail" at [41:46]
        html = "<div><section><p>Deep</p></section></div> tail"
        c = self._make_cleaner()
        clean, mapping = c.clean(html)

        assert clean == "Deep tail"
        assert mapping.original_length == len(html)  # 46
        assert mapping.clean_length == len(clean)  # 9

        # "Deep" → [17:21], synthetic space → 41 (start of " tail" run / end of
        # closing-tags block), " tail" → [41:46]
        expected = [17, 18, 19, 20, 41, 42, 43, 44, 45, 46]
        for i, orig in enumerate(expected):
            assert mapping.to_original(i) == orig, (
                f"clean pos {i}: expected {orig}, got {mapping.to_original(i)}"
            )

    # -- both cleaners must produce identical results -----------------------

    def test_nested_html_both_cleaners_agree(self) -> None:
        """TagStripCleaner and HTMLParserTagStripCleaner must agree on clean
        text and on every position mapping for all nested-HTML cases."""
        cases = [
            "<div><p><b>Hello world</b></p></div>",
            "<div>Outer<p>Inner</p>End</div>",
            "<p>First.</p><p>Second.</p>",
            "A<br/>B<br/>C",
            "<div><section><p>Deep</p></section></div> tail",
        ]
        tag_cleaner = TagStripCleaner()
        html_cleaner = HTMLParserTagStripCleaner()

        for html in cases:
            clean_tag, m_tag = tag_cleaner.clean(html)
            clean_html, m_html = html_cleaner.clean(html)

            assert clean_tag == clean_html, (
                f"clean text mismatch for {html!r}: {clean_tag!r} vs {clean_html!r}"
            )
            for i in range(len(clean_tag) + 1):
                assert m_tag.to_original(i) == m_html.to_original(i), (
                    f"position {i} mismatch for {html!r}: "
                    f"tag={m_tag.to_original(i)}, html={m_html.to_original(i)}"
                )


class TestTagStripCleanerNested(_NestedHTMLMixin):
    def _make_cleaner(self) -> TagStripCleaner:
        return TagStripCleaner()


class TestHTMLParserTagStripCleanerNested(_NestedHTMLMixin):
    def _make_cleaner(self) -> HTMLParserTagStripCleaner:
        return HTMLParserTagStripCleaner()
