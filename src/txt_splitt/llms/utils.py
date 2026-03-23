"""Shared helpers for LLM-backed strategies."""

from __future__ import annotations

import re
from collections import Counter

_WORD_PATTERN = re.compile(r"[A-Za-z0-9#.+-]+")


def looks_repetitive(response: str) -> bool:
    """Heuristically detect looped or highly repetitive LLM output."""
    words = _WORD_PATTERN.findall(response.lower())
    if len(words) < 120:
        return False

    unique_ratio = len(set(words)) / len(words)
    if unique_ratio < 0.18:
        return True

    shingle_size = 8
    if len(words) <= shingle_size:
        return False

    shingles = [
        " ".join(words[i : i + shingle_size])
        for i in range(len(words) - shingle_size + 1)
    ]
    counts = Counter(shingles)
    most_common = counts.most_common(1)
    if not most_common:
        return False

    _, freq = most_common[0]
    return freq >= 12
