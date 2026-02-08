"""Gap handler implementations."""

from txt_splitt.errors import GapError
from txt_splitt.types import SentenceGroup, SentenceRange


class StrictGapHandler:
    """Validate that sentence groups provide continuous coverage.

    - Trims overlaps by adjusting later ranges
    - Raises GapError on any gap or incomplete coverage
    """

    def handle(
        self, groups: list[SentenceGroup], sentence_count: int
    ) -> list[SentenceGroup]:
        if sentence_count <= 0:
            raise GapError("sentence_count must be positive")

        if not groups:
            raise GapError("No groups provided")

        max_index = sentence_count - 1

        # Flatten all (group_index, range) pairs and sort by start
        flat: list[tuple[int, SentenceRange]] = []
        for gi, group in enumerate(groups):
            for r in group.ranges:
                flat.append((gi, r))
        flat.sort(key=lambda x: (x[1].start, x[1].end))

        # Resolve overlaps by trimming later ranges and verify continuous coverage
        adjusted: dict[int, list[SentenceRange]] = {i: [] for i in range(len(groups))}
        next_expected = 0

        for gi, r in flat:
            if r.end < next_expected:
                # Entirely consumed by previous range
                continue
            start = max(r.start, next_expected)
            if start > r.end:
                continue

            # Check for gap before this range
            if start != next_expected:
                raise GapError(
                    f"Gap detected: sentences {next_expected}-{start - 1} "
                    "are not covered"
                )

            adjusted[gi].append(SentenceRange(start=start, end=r.end))
            next_expected = r.end + 1

        # Check for incomplete coverage at the end
        if next_expected <= max_index:
            raise GapError(
                f"Incomplete coverage: sentences {next_expected}-{max_index} "
                "are not covered"
            )

        # Build result groups, preserving order, dropping empty groups
        result: list[SentenceGroup] = []
        for gi, group in enumerate(groups):
            ranges = adjusted[gi]
            if ranges:
                result.append(
                    SentenceGroup(
                        label=group.label,
                        ranges=tuple(ranges),
                    )
                )

        return result
