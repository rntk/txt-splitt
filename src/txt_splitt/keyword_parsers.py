"""Response parser implementations for the keyword extraction pipeline."""

from __future__ import annotations

import re

from txt_splitt.errors import ParseError

_TOKEN_RE = re.compile(r"(\d+)\s*-\s*(\d+)|(\d+)")


class KeywordIndexParser:
    """Parse LLM responses into keyword index ranges.

    The LLM returns comma-separated marker numbers and optional ranges,
    e.g. ``0, 3, 7-10, 15``.

    - Single index ``5`` → ``(5, 5)``
    - Range ``7-10`` → ``(7, 10)`` (both inclusive, multi-word phrase)

    Validates indices against ``word_count``, deduplicates, and returns
    a sorted list of ``(start_idx, end_idx)`` tuples.
    """

    def parse(self, response: str, word_count: int) -> list[tuple[int, int]]:
        if word_count < 0:
            msg = f"word_count must be non-negative, got {word_count}"
            raise ParseError(msg)

        seen: set[tuple[int, int]] = set()
        result: list[tuple[int, int]] = []

        for match in _TOKEN_RE.finditer(response):
            if match.group(1) is not None:
                # range: N-M
                start = int(match.group(1))
                end = int(match.group(2))
                if start > end:
                    start, end = end, start
            else:
                # single index
                start = int(match.group(3))
                end = start

            # Validate bounds
            if start < 0 or end >= word_count:
                continue

            key = (start, end)
            if key not in seen:
                seen.add(key)
                result.append(key)

        result.sort()
        return result
