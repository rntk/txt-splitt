"""Tests for offset restoration."""

from txt_splitt.html_cleaners import HTMLParserTagStripCleaner, TagStripCleaner
from txt_splitt.sentences.offset_restorers import MappingOffsetRestorer
from txt_splitt.sentences.types import (
    OffsetMapping,
    OffsetSegment,
    Sentence,
    SentenceGroup,
    SentenceRange,
    SplitResult,
)


class TestMappingOffsetRestorer:
    def setup_method(self) -> None:
        self.restorer = MappingOffsetRestorer()

    def test_empty_result(self) -> None:
        result = SplitResult(sentences=(), groups=())
        mapping = OffsetMapping(segments=(), original_length=10, clean_length=0)
        restored = self.restorer.restore(result, mapping)
        assert restored is result  # same object returned

    def test_identity_mapping(self) -> None:
        """No tags: offsets should remain unchanged."""
        mapping = OffsetMapping(
            segments=(OffsetSegment(clean_offset=0, original_offset=0, length=20),),
            original_length=20,
            clean_length=20,
        )
        sentences = (
            Sentence(index=0, start=0, end=10, text="First sent"),
            Sentence(index=1, start=10, end=20, text="Second sen"),
        )
        groups = (
            SentenceGroup(label=("Topic",), ranges=(SentenceRange(start=0, end=1),)),
        )
        result = SplitResult(sentences=sentences, groups=groups)
        restored = self.restorer.restore(result, mapping)

        assert restored.sentences[0].start == 0
        assert restored.sentences[0].end == 10
        assert restored.sentences[1].start == 10
        assert restored.sentences[1].end == 20

    def test_basic_restoration(self) -> None:
        """Offsets should be remapped through the mapping."""
        # Original: "A<b>B</b>C" (len 11), clean: "ABC" (len 3)
        mapping = OffsetMapping(
            segments=(
                OffsetSegment(clean_offset=0, original_offset=0, length=1),
                OffsetSegment(clean_offset=1, original_offset=4, length=1),
                OffsetSegment(clean_offset=2, original_offset=9, length=1),
            ),
            original_length=10,
            clean_length=3,
        )
        sentences = (
            Sentence(index=0, start=0, end=2, text="AB"),
            Sentence(index=1, start=2, end=3, text="C"),
        )
        result = SplitResult(sentences=sentences, groups=())
        restored = self.restorer.restore(result, mapping)

        assert restored.sentences[0].start == 0
        assert restored.sentences[0].end == 9  # maps to start of 3rd segment
        assert restored.sentences[1].start == 9
        assert restored.sentences[1].end == 10  # clean_length -> original_length

    def test_preserves_clean_text(self) -> None:
        mapping = OffsetMapping(
            segments=(
                OffsetSegment(clean_offset=0, original_offset=0, length=5),
                OffsetSegment(clean_offset=5, original_offset=8, length=5),
            ),
            original_length=13,
            clean_length=10,
        )
        sentences = (Sentence(index=0, start=0, end=10, text="HelloWorld"),)
        result = SplitResult(sentences=sentences, groups=())
        restored = self.restorer.restore(result, mapping)
        assert restored.sentences[0].text == "HelloWorld"

    def test_preserves_groups_unchanged(self) -> None:
        mapping = OffsetMapping(
            segments=(OffsetSegment(clean_offset=0, original_offset=0, length=5),),
            original_length=5,
            clean_length=5,
        )
        groups = (
            SentenceGroup(
                label=("A", "B"),
                ranges=(SentenceRange(start=0, end=0),),
            ),
        )
        result = SplitResult(
            sentences=(Sentence(index=0, start=0, end=5, text="Hello"),),
            groups=groups,
        )
        restored = self.restorer.restore(result, mapping)
        assert restored.groups is groups

    def test_preserves_sentence_indices(self) -> None:
        mapping = OffsetMapping(
            segments=(
                OffsetSegment(clean_offset=0, original_offset=0, length=3),
                OffsetSegment(clean_offset=3, original_offset=6, length=3),
            ),
            original_length=9,
            clean_length=6,
        )
        sentences = (
            Sentence(index=0, start=0, end=3, text="ABC"),
            Sentence(index=1, start=3, end=6, text="DEF"),
        )
        result = SplitResult(sentences=sentences, groups=())
        restored = self.restorer.restore(result, mapping)
        assert restored.sentences[0].index == 0
        assert restored.sentences[1].index == 1


# ---------------------------------------------------------------------------
# Integration tests: real cleaner output + MappingOffsetRestorer
# ---------------------------------------------------------------------------


def _make_split_result(
    sentences: list[tuple[int, int, str]],
) -> SplitResult:
    """Build a SplitResult from (start, end, text) triples."""
    return SplitResult(
        sentences=tuple(
            Sentence(index=i, start=s, end=e, text=t)
            for i, (s, e, t) in enumerate(sentences)
        ),
        groups=(),
    )


class TestMappingOffsetRestorerIntegration:
    """End-to-end tests that pipe actual cleaner output into MappingOffsetRestorer."""

    def setup_method(self) -> None:
        self.restorer = MappingOffsetRestorer()

    def _assert_original_contains_text(
        self, original: str, start: int, end: int, text: str
    ) -> None:
        """The original slice [start:end] must contain *text* after stripping tags."""
        import re

        slice_ = original[start:end]
        stripped = re.sub(r"<[^>]+>", "", slice_)
        assert stripped == text, (
            f"original[{start}:{end}]={slice_!r} stripped to {stripped!r}, "
            f"expected {text!r}"
        )

    # -- TagStripCleaner + restorer -----------------------------------------

    def test_restore_sentences_through_paragraph_html_tag_cleaner(self) -> None:
        """Two sentences in adjacent <p> elements; verify that the restored
        original positions bracket exactly the right visible text."""
        html = "<p>First sentence.</p><p>Second sentence.</p>"
        # <p> at [0:3], "First sentence." at [3:18], </p><p> merged at [18:25],
        # "Second sentence." at [25:41], </p> at [41:45]
        cleaner = TagStripCleaner()
        clean, mapping = cleaner.clean(html)
        # clean == "First sentence. Second sentence."

        result = _make_split_result(
            [
                (0, 15, "First sentence."),  # clean[0:15]
                (16, 32, "Second sentence."),  # clean[16:32]
            ]
        )
        restored = self.restorer.restore(result, mapping)

        # Sentence 0: original slice must contain "First sentence."
        s0 = restored.sentences[0]
        self._assert_original_contains_text(html, s0.start, s0.end, "First sentence.")
        assert s0.text == "First sentence."

        # Sentence 1: original slice must contain "Second sentence."
        s1 = restored.sentences[1]
        self._assert_original_contains_text(html, s1.start, s1.end, "Second sentence.")
        assert s1.text == "Second sentence."

    def test_restore_sentences_through_paragraph_html_htmlparser_cleaner(self) -> None:
        """Same as above but with HTMLParserTagStripCleaner."""
        html = "<p>First sentence.</p><p>Second sentence.</p>"
        cleaner = HTMLParserTagStripCleaner()
        clean, mapping = cleaner.clean(html)

        result = _make_split_result(
            [
                (0, 15, "First sentence."),
                (16, 32, "Second sentence."),
            ]
        )
        restored = self.restorer.restore(result, mapping)

        s0 = restored.sentences[0]
        self._assert_original_contains_text(html, s0.start, s0.end, "First sentence.")

        s1 = restored.sentences[1]
        self._assert_original_contains_text(html, s1.start, s1.end, "Second sentence.")

    # -- sentence boundary exactly at synthetic separator -------------------

    def test_restore_sentence_boundary_at_synthetic_separator(self) -> None:
        """When a sentence ends at the synthetic-separator position (the space
        inserted between two tag-separated text runs), the restored *end* must
        map to the start of the removed span — not into the following text."""
        # "First." ends at clean pos 6; synthetic separator is at clean pos 6
        # which maps back to orig pos 9 (start of </p> in original).
        html = "<p>First.</p><p>Second.</p>"
        cleaner = TagStripCleaner()
        clean, mapping = cleaner.clean(html)
        # clean == "First. Second.", synthetic space at clean pos 6 → orig 9

        # A sentence that ends right at the synthetic space
        result = _make_split_result([(0, 6, "First.")])
        restored = self.restorer.restore(result, mapping)

        s = restored.sentences[0]
        assert s.start == 3  # orig pos of 'F' in "First."
        assert s.end == 9  # orig pos of '<' starting </p>
        # The original slice [3:9] must be exactly "First."
        assert html[s.start : s.end] == "First."

    # -- deeply nested HTML: single sentence --------------------------------

    def test_restore_deeply_nested_html(self) -> None:
        """A single sentence extracted from 3-level nested HTML should map
        back to the correct original positions."""
        html = "<div><p><b>Hello world</b></p></div>"
        # Opening tags: [0:11], text "Hello world": [11:22], closing tags: [22:36]
        cleaner = TagStripCleaner()
        clean, mapping = cleaner.clean(html)
        assert clean == "Hello world"

        result = _make_split_result([(0, 11, "Hello world")])
        restored = self.restorer.restore(result, mapping)

        s = restored.sentences[0]
        assert s.start == 11  # first char of text node in original
        assert s.end == 36  # original_length (end maps to end of closing tags)
        assert s.text == "Hello world"
        # The original slice must contain the visible text
        self._assert_original_contains_text(html, s.start, s.end, "Hello world")

    # -- multiple sentences through multi-level nesting --------------------

    def test_restore_multiple_sentences_through_nested_html(self) -> None:
        """Three text nodes at different nesting levels; each becomes a
        sentence and must round-trip back to its correct original position."""
        html = "<div>Outer<p>Inner</p>End</div>"
        # <div> at [0:5], "Outer" at [5:10], <p> at [10:13],
        # "Inner" at [13:18], </p> at [18:22], "End" at [22:25], </div> at [25:31]
        cleaner = TagStripCleaner()
        clean, mapping = cleaner.clean(html)
        # clean == "Outer Inner End"
        # synthetic separator at clean[5] → orig[10], at clean[11] → orig[18]

        result = _make_split_result(
            [
                (0, 5, "Outer"),  # clean[0:5]
                (6, 11, "Inner"),  # clean[6:11]
                (12, 15, "End"),  # clean[12:15]
            ]
        )
        restored = self.restorer.restore(result, mapping)

        s0 = restored.sentences[0]
        assert html[s0.start : s0.end] == "Outer"

        s1 = restored.sentences[1]
        assert html[s1.start : s1.end] == "Inner"

        s2 = restored.sentences[2]
        # The last sentence's end maps to original_length (past the closing </div>),
        # so the slice may include the trailing tag — verify via tag-stripped content.
        self._assert_original_contains_text(html, s2.start, s2.end, "End")
