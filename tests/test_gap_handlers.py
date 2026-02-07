"""Tests for gap handling."""

import pytest

from txt_splitt.errors import GapError
from txt_splitt.gap_handlers import StrictGapHandler
from txt_splitt.types import SentenceGroup, SentenceRange


class TestStrictGapHandler:
    def setup_method(self) -> None:
        self.handler = StrictGapHandler()

    def test_full_coverage_passes(
        self, full_coverage_groups: list[SentenceGroup]
    ) -> None:
        result = self.handler.handle(full_coverage_groups, sentence_count=5)
        assert len(result) == 2

    def test_single_group_full_coverage(self) -> None:
        groups = [
            SentenceGroup(
                label=("Technology",),
                ranges=(SentenceRange(start=0, end=4),),
            )
        ]
        result = self.handler.handle(groups, sentence_count=5)
        assert len(result) == 1
        assert result[0].ranges == (SentenceRange(start=0, end=4),)

    def test_gap_at_start_raises(self) -> None:
        groups = [
            SentenceGroup(
                label=("Technology",),
                ranges=(SentenceRange(start=2, end=4),),
            )
        ]
        with pytest.raises(GapError, match="Gap detected"):
            self.handler.handle(groups, sentence_count=5)

    def test_gap_in_middle_raises(self) -> None:
        groups = [
            SentenceGroup(
                label=("Technology",),
                ranges=(SentenceRange(start=0, end=1),),
            ),
            SentenceGroup(
                label=("Science",),
                ranges=(SentenceRange(start=3, end=4),),
            ),
        ]
        with pytest.raises(GapError, match="Gap detected"):
            self.handler.handle(groups, sentence_count=5)

    def test_gap_at_end_raises(self) -> None:
        groups = [
            SentenceGroup(
                label=("Technology",),
                ranges=(SentenceRange(start=0, end=2),),
            )
        ]
        with pytest.raises(GapError, match="Incomplete coverage"):
            self.handler.handle(groups, sentence_count=5)

    def test_overlap_trimmed(self) -> None:
        groups = [
            SentenceGroup(
                label=("Technology",),
                ranges=(SentenceRange(start=0, end=3),),
            ),
            SentenceGroup(
                label=("Science",),
                ranges=(SentenceRange(start=2, end=4),),
            ),
        ]
        result = self.handler.handle(groups, sentence_count=5)
        assert len(result) == 2
        assert result[0].ranges == (SentenceRange(start=0, end=3),)
        assert result[1].ranges == (SentenceRange(start=4, end=4),)

    def test_empty_groups_raises(self) -> None:
        with pytest.raises(GapError, match="No groups"):
            self.handler.handle([], sentence_count=5)

    def test_zero_sentence_count_raises(self) -> None:
        groups = [
            SentenceGroup(
                label=("Technology",),
                ranges=(SentenceRange(start=0, end=0),),
            )
        ]
        with pytest.raises(GapError):
            self.handler.handle(groups, sentence_count=0)

    def test_preserves_group_order(self) -> None:
        groups = [
            SentenceGroup(
                label=("Science",),
                ranges=(SentenceRange(start=3, end=4),),
            ),
            SentenceGroup(
                label=("Technology",),
                ranges=(SentenceRange(start=0, end=2),),
            ),
        ]
        result = self.handler.handle(groups, sentence_count=5)
        # Original order preserved: Science first, Technology second
        assert result[0].label == ("Science",)
        assert result[1].label == ("Technology",)

    def test_completely_overlapping_range_dropped(self) -> None:
        groups = [
            SentenceGroup(
                label=("Technology",),
                ranges=(SentenceRange(start=0, end=4),),
            ),
            SentenceGroup(
                label=("Science",),
                ranges=(SentenceRange(start=1, end=3),),
            ),
        ]
        result = self.handler.handle(groups, sentence_count=5)
        assert len(result) == 1
        assert result[0].label == ("Technology",)
