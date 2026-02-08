"""Sentence splitting implementations."""

import re

from txt_splitt.types import Sentence

# Compiled regex for sentence boundaries:
# - Punctuation ([.!?]) followed by whitespace and uppercase letter (including Cyrillic)
# - One or more newlines
_SENTENCE_BOUNDARY_PATTERN = re.compile(r"((?<=[.!?])\s+(?=[A-ZА-Я]))|(\n+)")
_DENSE_BOUNDARY_PATTERN = re.compile(r"((?<=[.!?])\s+(?=[A-ZА-Я]))|(\n+)|(\s+[·•|]\s+)")
_WORD_PATTERN = re.compile(r"\S+")


class RegexSentenceSplitter:
    """Split text into sentences using regex boundary detection.

    Splits on:
    - Punctuation ([.!?]) followed by whitespace and an uppercase letter
    - One or more newlines (block boundaries)
    """

    def split(self, text: str) -> list[Sentence]:
        if not text or not text.strip():
            return []

        boundaries = list(_SENTENCE_BOUNDARY_PATTERN.finditer(text))

        result: list[Sentence] = []
        start = 0
        index = 0

        for match in boundaries:
            end = match.start()
            s_start, s_end = _trim_whitespace(text, start, end)
            if s_start < s_end:
                result.append(
                    Sentence(
                        index=index,
                        start=s_start,
                        end=s_end,
                        text=text[s_start:s_end],
                    )
                )
                index += 1
            start = match.end()

        # Handle the last segment
        s_start, s_end = _trim_whitespace(text, start, len(text))
        if s_start < s_end:
            result.append(
                Sentence(
                    index=index,
                    start=s_start,
                    end=s_end,
                    text=text[s_start:s_end],
                )
            )

        return result


class DenseRegexSentenceSplitter:
    """Split text into denser marker units for topic labeling.

    Strategy:
    - Keep regex sentence boundaries from ``RegexSentenceSplitter``.
    - Also split on digest separators like ``·`` and ``|``.
    - Add soft anchors roughly every ``anchor_every_words`` words.
    """

    def __init__(self, *, anchor_every_words: int = 24) -> None:
        if anchor_every_words <= 0:
            raise ValueError("anchor_every_words must be positive")
        self._anchor_every_words = anchor_every_words

    def split(self, text: str) -> list[Sentence]:
        if not text or not text.strip():
            return []

        boundaries = list(_DENSE_BOUNDARY_PATTERN.finditer(text))

        spans: list[tuple[int, int]] = []
        start = 0

        for match in boundaries:
            end = match.start()
            s_start, s_end = _trim_whitespace(text, start, end)
            if s_start < s_end:
                spans.append((s_start, s_end))
            start = match.end()

        # Handle the last segment
        s_start, s_end = _trim_whitespace(text, start, len(text))
        if s_start < s_end:
            spans.append((s_start, s_end))

        anchored_spans: list[tuple[int, int]] = []
        for span_start, span_end in spans:
            anchored_spans.extend(
                _split_span_by_word_anchor(
                    text,
                    span_start,
                    span_end,
                    self._anchor_every_words,
                )
            )

        result: list[Sentence] = []
        for index, (seg_start, seg_end) in enumerate(anchored_spans):
            result.append(
                Sentence(
                    index=index,
                    start=seg_start,
                    end=seg_end,
                    text=text[seg_start:seg_end],
                )
            )

        return result


def _trim_whitespace(text: str, start: int, end: int) -> tuple[int, int]:
    """Trim leading and trailing whitespace from a text span."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _split_span_by_word_anchor(
    text: str, start: int, end: int, anchor_every_words: int
) -> list[tuple[int, int]]:
    """Split a span into smaller spans using periodic word-count anchors."""
    matches = list(_WORD_PATTERN.finditer(text, start, end))
    if len(matches) <= anchor_every_words:
        return [(start, end)]

    cut_points: list[int] = []
    for word_count in range(anchor_every_words, len(matches), anchor_every_words):
        word_end = matches[word_count - 1].end()
        cut = _find_whitespace_cut(text, word_end, end)
        if cut is not None:
            cut_points.append(cut)

    if not cut_points:
        return [(start, end)]

    spans: list[tuple[int, int]] = []
    span_start = start
    for cut in cut_points:
        s_start, s_end = _trim_whitespace(text, span_start, cut)
        if s_start < s_end:
            spans.append((s_start, s_end))
            span_start = cut

    s_start, s_end = _trim_whitespace(text, span_start, end)
    if s_start < s_end:
        spans.append((s_start, s_end))

    return spans if spans else [(start, end)]


def _find_whitespace_cut(text: str, start: int, end: int) -> int | None:
    """Find the nearest whitespace position to cut, preferring forward scan."""
    if start >= end:
        return None

    right = start
    while right < end and not text[right].isspace():
        right += 1
    if right < end:
        return right

    left = start - 1
    while left >= 0 and not text[left].isspace():
        left -= 1
    if left >= 0:
        return left + 1

    return None
