"""Tests for HTML tag cleaning and offset mapping."""

import pytest

from txt_splitt.html_cleaners import HTMLParserTagStripCleaner, TagStripCleaner
from txt_splitt.types import OffsetMapping, OffsetSegment


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
        assert clean == "Helloworld"
        assert mapping.clean_length == 10

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
        assert clean == "TitleBody"
        assert mapping.clean_length == 9

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

    def test_segment_count_matches_text_runs(self) -> None:
        text = "A<b>B</b>C"
        clean, mapping = self.cleaner.clean(text)
        assert clean == "ABC"
        assert len(mapping.segments) == 3

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
        assert clean == "ABCDEF"

        expected = [0, 1, 5, 6, 11, 12]  # A B C D E F
        for i, expected_orig in enumerate(expected):
            assert mapping.to_original(i) == expected_orig, f"pos {i}"
        # End position
        assert mapping.to_original(6) == len(original)

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
