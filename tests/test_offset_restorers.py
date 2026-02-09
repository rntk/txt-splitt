"""Tests for offset restoration."""

from txt_splitt.offset_restorers import MappingOffsetRestorer
from txt_splitt.types import (
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
