"""Sentence length normalization for the splitting pipeline."""

import re

from txt_splitt.protocols import SentenceSplitter
from txt_splitt.types import Sentence

_DEFAULT_MIN_LENGTH = 40
_DEFAULT_MAX_LENGTH = 300

# Split-point patterns (ordered by priority)
_SEMICOLON = re.compile(r";")
_COMMA_CONJUNCTION = re.compile(
    r",\s+(?:and|but|or|so|yet|however|moreover|furthermore|nevertheless)\s",
    re.IGNORECASE,
)
_COMMA = re.compile(r",")


class NormalizingSplitter:
    """Wraps a SentenceSplitter, merging short and splitting long sentences.

    After the inner splitter produces raw sentences, this wrapper:
    1. Merges sentences shorter than *min_length* with an adjacent sentence.
    2. Splits sentences longer than *max_length* at clause boundaries.

    Satisfies the ``SentenceSplitter`` protocol.
    """

    def __init__(
        self,
        inner: SentenceSplitter,
        *,
        min_length: int = _DEFAULT_MIN_LENGTH,
        max_length: int = _DEFAULT_MAX_LENGTH,
    ) -> None:
        if max_length <= min_length:
            raise ValueError(
                f"max_length ({max_length}) must be greater than "
                f"min_length ({min_length})"
            )
        self._inner = inner
        self._min_length = min_length
        self._max_length = max_length

    def split(self, text: str) -> list[Sentence]:
        sentences = self._inner.split(text)
        if not sentences:
            return sentences

        # Phase 1: merge short sentences
        sentences = _merge_short(sentences, text, self._min_length)
        # Phase 2: split long sentences
        sentences = _split_long(sentences, text, self._max_length)
        # Phase 3: re-index
        return _reindex(sentences)


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------


def _merge_short(
    sentences: list[Sentence], text: str, min_length: int
) -> list[Sentence]:
    """Merge sentences shorter than *min_length* with a neighbour.

    Strategy: single left-to-right pass.
    - Short sentence merges with the *previous* sentence.
    - If there is no previous (first sentence), defer merge to the next one.
    """
    if len(sentences) <= 1:
        return sentences

    merged: list[Sentence] = []
    pending_forward: Sentence | None = None  # short first sentence awaiting next

    for sent in sentences:
        if pending_forward is not None:
            # Merge the pending short sentence with the current one
            merged.append(_combine(pending_forward, sent, text))
            pending_forward = None
            continue

        if len(sent.text) < min_length:
            if merged:
                # Merge with previous
                prev = merged[-1]
                merged[-1] = _combine(prev, sent, text)
            else:
                # First sentence is short — wait for next
                pending_forward = sent
        else:
            merged.append(sent)

    # Edge case: pending never got consumed (all sentences short, or only one)
    if pending_forward is not None:
        merged.append(pending_forward)

    return merged


def _combine(a: Sentence, b: Sentence, text: str) -> Sentence:
    """Combine two adjacent sentences into one using the original text span."""
    new_start = a.start
    new_end = b.end
    return Sentence(index=0, start=new_start, end=new_end, text=text[new_start:new_end])


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------


def _split_long(
    sentences: list[Sentence], text: str, max_length: int
) -> list[Sentence]:
    """Split sentences longer than *max_length* at clause boundaries."""
    result: list[Sentence] = []
    for sent in sentences:
        result.extend(_split_single(sent, text, max_length))
    return result


def _split_single(sent: Sentence, text: str, max_length: int) -> list[Sentence]:
    """Recursively split a single sentence until all pieces are within *max_length*."""
    if len(sent.text) <= max_length:
        return [sent]

    split_offset = _find_split_point(sent.text)
    if split_offset <= 0 or split_offset >= len(sent.text):
        return [sent]  # cannot split meaningfully

    # Absolute offset in original text
    abs_split = sent.start + split_offset

    # Trim whitespace between the two pieces
    first_end = abs_split
    while first_end > sent.start and text[first_end - 1].isspace():
        first_end -= 1
    second_start = abs_split
    while second_start < sent.end and text[second_start].isspace():
        second_start += 1

    if first_end <= sent.start or second_start >= sent.end:
        return [sent]  # degenerate split

    first = Sentence(
        index=0, start=sent.start, end=first_end, text=text[sent.start : first_end]
    )
    second = Sentence(
        index=0, start=second_start, end=sent.end, text=text[second_start : sent.end]
    )

    return _split_single(first, text, max_length) + _split_single(
        second, text, max_length
    )


def _find_split_point(sentence_text: str) -> int:
    """Find the best character offset to split *sentence_text* at.

    Returns an offset within *sentence_text* (not the original text).
    Prefers clause boundaries near the midpoint.
    """
    mid = len(sentence_text) // 2

    # Try each pattern tier; pick the match closest to midpoint
    for pattern in (_SEMICOLON, _COMMA_CONJUNCTION, _COMMA):
        matches = list(pattern.finditer(sentence_text))
        if matches:
            best = min(matches, key=lambda m: abs(m.end() - mid))
            return best.end()

    # Fallback: word boundary (space) closest to midpoint
    spaces = [i for i, ch in enumerate(sentence_text) if ch == " "]
    if spaces:
        best_space = min(spaces, key=lambda i: abs(i - mid))
        return best_space + 1  # split after the space

    return mid  # absolute last resort


# ---------------------------------------------------------------------------
# Re-index
# ---------------------------------------------------------------------------


def _reindex(sentences: list[Sentence]) -> list[Sentence]:
    """Assign sequential indices 0, 1, 2, … to sentences."""
    return [
        Sentence(index=i, start=s.start, end=s.end, text=s.text)
        for i, s in enumerate(sentences)
    ]
