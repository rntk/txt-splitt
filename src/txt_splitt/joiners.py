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


def join_sentences_by_groups(
    groups: list[SentenceGroup], sentences: list[Sentence]
) -> tuple[list[Sentence], list[SentenceGroup]]:
    """Build sentence output that is already joined by group ranges.

    For each range in each group, creates one joined sentence and remaps group
    ranges to the new sentence indices.
    """
    joined_sentences: list[Sentence] = []
    remapped_groups: list[SentenceGroup] = []

    for group in groups:
        remapped_ranges: list[SentenceRange] = []
        for sentence_range in sorted(group.ranges, key=lambda r: (r.start, r.end)):
            joined_sentence = _join_sentence_range(
                sentence_range, sentences, joined_sentences
            )
            joined_sentences.append(joined_sentence)
            remapped_ranges.append(
                SentenceRange(start=joined_sentence.index, end=joined_sentence.index)
            )
        remapped_groups.append(
            SentenceGroup(label=group.label, ranges=tuple(remapped_ranges))
        )
    return joined_sentences, remapped_groups


def _join_sentence_range(
    sentence_range: SentenceRange,
    sentences: list[Sentence],
    joined_sentences: list[Sentence],
) -> Sentence:
    if sentence_range.start < 0:
        msg = f"sentence range start must be >= 0, got {sentence_range.start}"
        raise ValueError(msg)
    if sentence_range.end < sentence_range.start:
        msg = (
            "sentence range end must be >= start, "
            f"got {sentence_range.start}-{sentence_range.end}"
        )
        raise ValueError(msg)
    if sentence_range.end >= len(sentences):
        msg = (
            "sentence range end exceeds sentence count: "
            f"{sentence_range.end} >= {len(sentences)}"
        )
        raise ValueError(msg)

    selected = sentences[sentence_range.start : sentence_range.end + 1]
    return Sentence(
        index=len(joined_sentences),
        start=selected[0].start,
        end=selected[-1].end,
        text=" ".join(sentence.text for sentence in selected).strip(),
    )


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
