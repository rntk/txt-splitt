"""Group-joining implementations."""

from __future__ import annotations

from txt_splitt.types import Sentence, SentenceGroup, SentenceRange


class AdjacentSameTopicJoiner:
    """Merge adjacent groups when they share the same topic label."""

    def join(
        self, groups: list[SentenceGroup], sentences: list[Sentence]
    ) -> list[SentenceGroup]:
        del sentences  # join logic uses only group/range topology
        if not groups:
            return []

        merged: list[SentenceGroup] = [groups[0]]
        for group in groups[1:]:
            prev = merged[-1]
            if prev.label == group.label and _touches_or_overlaps(prev, group):
                merged[-1] = SentenceGroup(
                    label=prev.label,
                    ranges=_merge_ranges(prev.ranges + group.ranges),
                )
                continue
            merged.append(group)
        return merged


def _touches_or_overlaps(left: SentenceGroup, right: SentenceGroup) -> bool:
    if not left.ranges or not right.ranges:
        return False
    left_end = max(r.end for r in left.ranges)
    right_start = min(r.start for r in right.ranges)
    return right_start <= left_end + 1


def _merge_ranges(ranges: tuple[SentenceRange, ...]) -> tuple[SentenceRange, ...]:
    if not ranges:
        return ()
    ordered = sorted(ranges, key=lambda r: (r.start, r.end))
    coalesced: list[SentenceRange] = [ordered[0]]
    for current in ordered[1:]:
        last = coalesced[-1]
        if current.start <= last.end + 1:
            coalesced[-1] = SentenceRange(
                start=last.start,
                end=max(last.end, current.end),
            )
        else:
            coalesced.append(current)
    return tuple(coalesced)
