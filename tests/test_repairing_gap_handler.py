"""Tests for RepairingGapHandler."""

import pytest

from txt_splitt.errors import GapError
from txt_splitt.sentences.gap_handlers import RepairingGapHandler
from txt_splitt.sentences.types import SentenceGroup, SentenceRange


class TestRepairingGapHandler:
    def setup_method(self) -> None:
        self.handler = RepairingGapHandler()

    def test_full_coverage_passes(self) -> None:
        groups = [
            SentenceGroup(
                label=("A",),
                ranges=(SentenceRange(start=0, end=2),),
            ),
            SentenceGroup(
                label=("B",),
                ranges=(SentenceRange(start=3, end=4),),
            ),
        ]
        result = self.handler.handle(groups, sentence_count=5)
        # Gap at 2? No: (0-2) and (3-4) fully cover 0..4.
        # 0,1,2 is 3 sentences and 3,4 is 2 sentences, total 5.
        # Wait, start=0, end=2 is sentences 0, 1, 2.
        # start=3, end=4 is sentences 3, 4.
        # Total coverage 0-4. Correct.
        assert len(result) == 2
        assert result[0].ranges == (SentenceRange(start=0, end=2),)
        assert result[1].ranges == (SentenceRange(start=3, end=4),)

    def test_gap_at_start_repaired(self) -> None:
        groups = [
            SentenceGroup(
                label=("A",),
                ranges=(SentenceRange(start=2, end=4),),
            )
        ]
        result = self.handler.handle(groups, sentence_count=5)
        # 0-1 was missing, should be pulled into A
        assert len(result) == 1
        assert result[0].ranges[0].start == 0
        assert result[0].ranges[0].end == 4

    def test_gap_in_middle_repaired(self) -> None:
        groups = [
            SentenceGroup(
                label=("A",),
                ranges=(SentenceRange(start=0, end=1),),
            ),
            SentenceGroup(
                label=("B",),
                ranges=(SentenceRange(start=3, end=4),),
            ),
        ]
        result = self.handler.handle(groups, sentence_count=5)
        # Sentence 2 was missing, should be pulled into A
        assert len(result) == 2
        assert result[0].ranges == (SentenceRange(start=0, end=2),)
        assert result[1].ranges == (SentenceRange(start=3, end=4),)

    def test_gap_at_end_repaired(self) -> None:
        groups = [
            SentenceGroup(
                label=("A",),
                ranges=(SentenceRange(start=0, end=2),),
            )
        ]
        result = self.handler.handle(groups, sentence_count=5)
        # 3-4 was missing, should be pulled into A
        assert len(result) == 1
        assert result[0].ranges == (SentenceRange(start=0, end=4),)

    def test_overlap_trimmed(self) -> None:
        groups = [
            SentenceGroup(
                label=("A",),
                ranges=(SentenceRange(start=0, end=3),),
            ),
            SentenceGroup(
                label=("B",),
                ranges=(SentenceRange(start=2, end=4),),
            ),
        ]
        result = self.handler.handle(groups, sentence_count=5)
        assert len(result) == 2
        assert result[0].ranges == (SentenceRange(start=0, end=3),)
        assert result[1].ranges == (SentenceRange(start=4, end=4),)

    def test_no_groups_raises(self) -> None:
        with pytest.raises(GapError, match="No groups"):
            self.handler.handle([], sentence_count=5)

    def test_complex_repair(self) -> None:
        # Expected:
        # Gap at beginning (0-1) -> pulls into first range
        # Gap in middle (4-5) -> pulls into preceding range
        # Overlap (6-7) -> trims second range
        # Gap at end (9) -> pulls into last range
        groups = [
            SentenceGroup(
                label=("A",),
                ranges=(SentenceRange(start=2, end=3),),
            ),
            SentenceGroup(
                label=("B",),
                ranges=(SentenceRange(start=6, end=8),),
            ),
            SentenceGroup(
                label=("C",),
                ranges=(SentenceRange(start=7, end=8),),
            ),
        ]
        # Sentence count 10 (indices 0-9)
        result = self.handler.handle(groups, sentence_count=10)

        # A: (2,3) -> was gap at 0,1. Pull A to (0,3).
        # next_expected = 4.
        # B: (6,8) -> gap at 4,5. Pull A to (0,5).
        # B starts at 6, ends at 8. next_expected = 9.
        # C: (7,8) -> overlap. start = max(7, 9) = 9. end = 8.
        # start > end, discard C range.
        # But wait, does B cover the end gap at 9?
        # Actually next_expected=9, max_index=9. next_expected <= max_index is true.
        # last_added was B (since C's range was discarded).
        # Extend B to end (9).

        # Result should be:
        # A: [(0, 5)]
        # B: [(6, 9)]
        # C: discarded (no ranges left)

        assert len(result) == 2
        assert result[0].label == ("A",)
        assert result[0].ranges == (SentenceRange(start=0, end=5),)
        assert result[1].label == ("B",)
        assert result[1].ranges == (SentenceRange(start=6, end=9),)
