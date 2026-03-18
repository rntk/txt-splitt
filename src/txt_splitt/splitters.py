"""Sentence splitting implementations."""

from __future__ import annotations

import bisect
import html
import re
import unicodedata
from dataclasses import dataclass

from txt_splitt.types import Sentence

# Closing characters that may follow sentence-ending punctuation before whitespace:
# straight/curly quotes, closing parens/brackets, guillemet
_CLOSING = r"\"'\u201D\u2019)\]\u00BB"

_TERMINAL_BOUNDARY_PATTERN = re.compile(
    rf"(?P<punct>[.!?\u2026])(?:[{_CLOSING}])*(?P<gap>\s+)"
)
_SEMICOLON_OR_COLON_BOUNDARY_PATTERN = re.compile(r"(?<=[;:])\s+")
_BLANK_LINE_BOUNDARY_PATTERN = re.compile(r"\n\s*\n+")
_SINGLE_NEWLINE_BOUNDARY_PATTERN = re.compile(r"(?<!\n)\n(?!\n)")
_SEPARATOR_BOUNDARY_PATTERN = re.compile(r"\s+[·•|]\s+")
_OPENING_QUOTES_BRACKETS: frozenset[str] = frozenset("\"'([{<\u201c\u2018\u00ab")
_SENTENCE_START_CATEGORIES: frozenset[str] = frozenset(
    {"Lu", "Lt", "Lo", "So", "Nd", "Nl"}
)
_WORD_PATTERN = re.compile(r"\S+")
_HTML_TAG_PATTERN = re.compile(r"<(?:[^>\"']|\"[^\"]*\"|'[^']*')*>")
_HTML_ENTITY_PATTERN = re.compile(
    r"&(?:#[0-9]+|#x[0-9a-fA-F]+|[A-Za-z][A-Za-z0-9]{1,31});"
)
_LIST_MARKER_PATTERN = re.compile(r"(?:\d{1,3}|[a-zA-Z]|[ivxIVX]{1,5})[.)]")
_KNOWN_ABBREVIATIONS: frozenset[str] = frozenset(
    {
        "Mr.",
        "Mrs.",
        "Ms.",
        "Dr.",
        "Prof.",
        "St.",
        "Jr.",
        "Sr.",
        "vs.",
        "etc.",
        "Inc.",
        "Ltd.",
        "Corp.",
        "Gen.",
        "Gov.",
        "Sgt.",
        "Col.",
        "Capt.",
    }
)
_BRIDGE_PATTERN = re.compile(r"[^\w\s\.!\?\u2026]{1,5}")
_HORIZONTAL_WS_PATTERN = re.compile(r"[ \t\xa0\u2000-\u200a\u202f\u205f\u3000]+")
_EXCESS_NEWLINES_PATTERN = re.compile(r"\n{3,}")
_INVISIBLE_CATEGORIES: frozenset[str] = frozenset(
    {"Cc", "Cf", "Cs", "Co", "Cn", "Zl", "Zp"}
)
_SIGNAL_KIND_CONTENT = "content"
_SIGNAL_KIND_BLANK = "blank"
_SIGNAL_KIND_BRIDGE = "bridge"
_SIGNAL_KIND_LIST_MARKER = "list_marker"

_PRIORITY_TERMINAL = 0
_PRIORITY_SEPARATOR = 1
_PRIORITY_BLANK_LINE = 2


@dataclass(frozen=True, slots=True)
class _BoundaryCandidate:
    start: int
    end: int
    priority: int


@dataclass(slots=True)
class _SignalSpan:
    start: int
    end: int
    kind: str


@dataclass(frozen=True, slots=True)
class _SplitContext:
    tag_starts: tuple[int, ...]
    tag_ends: tuple[int, ...]

    @classmethod
    def from_text(cls, text: str, *, html_aware: bool) -> _SplitContext:
        if not html_aware:
            return cls(tag_starts=(), tag_ends=())

        starts: list[int] = []
        ends: list[int] = []
        for match in _HTML_TAG_PATTERN.finditer(text):
            starts.append(match.start())
            ends.append(match.end())
        return cls(tag_starts=tuple(starts), tag_ends=tuple(ends))

    @property
    def html_aware(self) -> bool:
        return bool(self.tag_starts)

    def pos_inside_tag(self, pos: int) -> bool:
        if not self.tag_starts:
            return False
        idx = bisect.bisect_right(self.tag_starts, pos) - 1
        return idx >= 0 and pos < self.tag_ends[idx]

    def range_overlaps_tag(self, start: int, end: int) -> bool:
        if not self.tag_starts:
            return False
        if self.pos_inside_tag(start):
            return True
        if end > start and self.pos_inside_tag(end - 1):
            return True
        idx = bisect.bisect_left(self.tag_starts, start)
        return idx < len(self.tag_starts) and self.tag_starts[idx] < end

    def boundary_allowed(self, start: int, end: int) -> bool:
        return not self.range_overlaps_tag(start, end)

    def cut_allowed(self, pos: int) -> bool:
        return not self.pos_inside_tag(pos)

    def count_word(self, match: re.Match[str]) -> bool:
        return not self.pos_inside_tag(match.start())


class SparseRegexSentenceSplitter:
    """Split text by natural punctuation boundaries with sparse anchoring.

    Strategy:
    - Split first on high-confidence punctuation/blank-line boundaries.
    - Optional ``html_aware`` mode prevents cuts inside HTML tags.
    - Only if a resulting span is very long, prefer soft boundaries near
      ``anchor_every_words`` words, then fall back to whitespace cuts.
    """

    def __init__(
        self,
        *,
        anchor_every_words: int = 16,
        long_sentence_word_threshold: int = 32,
        min_sentence_words: int = 4,
        html_aware: bool = False,
    ) -> None:
        if anchor_every_words <= 0:
            raise ValueError("anchor_every_words must be positive")
        if long_sentence_word_threshold <= 0:
            raise ValueError("long_sentence_word_threshold must be positive")
        if long_sentence_word_threshold < anchor_every_words:
            raise ValueError(
                "long_sentence_word_threshold must be >= anchor_every_words"
            )
        if min_sentence_words <= 0:
            raise ValueError("min_sentence_words must be positive")
        self._anchor_every_words = anchor_every_words
        self._long_sentence_word_threshold = long_sentence_word_threshold
        self._min_sentence_words = min_sentence_words
        self._html_aware = html_aware

    def split(self, text: str) -> list[Sentence]:
        if not text or not text.strip():
            return []

        context = _SplitContext.from_text(text, html_aware=self._html_aware)
        boundaries = _collect_boundaries(text, context)
        spans = _split_spans(text, boundaries)
        spans = _merge_low_signal_spans(text, spans)
        spans = _anchor_long_spans(
            text,
            spans,
            anchor_every_words=self._anchor_every_words,
            long_sentence_word_threshold=self._long_sentence_word_threshold,
            min_sentence_words=self._min_sentence_words,
            context=context,
        )
        spans = _merge_low_signal_spans(text, spans)
        spans = _merge_short_nonterminal_spans(
            text,
            spans,
            min_sentence_words=self._min_sentence_words,
            target_words=self._anchor_every_words,
            context=context,
        )
        return _spans_to_sentences(text, spans)


def _collect_boundaries(text: str, context: _SplitContext) -> list[tuple[int, int]]:
    candidates: list[_BoundaryCandidate] = []

    for match in _TERMINAL_BOUNDARY_PATTERN.finditer(text):
        gap_start, gap_end = match.span("gap")
        if not _starts_like_sentence_start(text, gap_end):
            continue
        punct_end = match.end("punct")
        if text[punct_end - 1] == "." and _preceded_by_abbreviation(
            text, punct_end - 1
        ):
            continue
        if not context.boundary_allowed(gap_start, gap_end):
            continue
        candidates.append(_BoundaryCandidate(gap_start, gap_end, _PRIORITY_TERMINAL))

    for match in _SEPARATOR_BOUNDARY_PATTERN.finditer(text):
        start, end = match.span()
        if not context.boundary_allowed(start, end):
            continue
        candidates.append(_BoundaryCandidate(start, end, _PRIORITY_SEPARATOR))

    for match in _BLANK_LINE_BOUNDARY_PATTERN.finditer(text):
        start, end = match.span()
        if not context.boundary_allowed(start, end):
            continue
        candidates.append(_BoundaryCandidate(start, end, _PRIORITY_BLANK_LINE))

    for match in _SINGLE_NEWLINE_BOUNDARY_PATTERN.finditer(text):
        start, end = match.span()
        if not _starts_like_sentence_start(text, end):
            continue
        if not context.boundary_allowed(start, end):
            continue
        candidates.append(_BoundaryCandidate(start, end, _PRIORITY_BLANK_LINE))

    return _select_boundaries(candidates)


def _starts_like_sentence_start(text: str, pos: int) -> bool:
    if pos >= len(text):
        return False
    ch = text[pos]
    if ch in _OPENING_QUOTES_BRACKETS:
        return True
    return unicodedata.category(ch) in _SENTENCE_START_CATEGORIES


def _preceded_by_abbreviation(text: str, punct_pos: int) -> bool:
    """Check if the period at punct_pos is part of a known abbreviation."""
    end = punct_pos + 1
    start = punct_pos - 1
    while start >= 0 and not text[start].isspace():
        start -= 1
    start += 1
    return text[start:end] in _KNOWN_ABBREVIATIONS


def _entity_ends_at(text: str, boundary_start: int) -> bool:
    if boundary_start <= 0 or text[boundary_start - 1] != ";":
        return False

    token_start = boundary_start - 1
    while token_start > 0 and not text[token_start - 1].isspace():
        token_start -= 1
    token = text[token_start:boundary_start]
    return _HTML_ENTITY_PATTERN.fullmatch(token) is not None


def _select_boundaries(candidates: list[_BoundaryCandidate]) -> list[tuple[int, int]]:
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            candidate.start,
            candidate.priority,
            -(candidate.end - candidate.start),
        ),
    )

    selected: list[tuple[int, int]] = []
    last_end = -1
    for candidate in ordered:
        if candidate.start < last_end:
            continue
        selected.append((candidate.start, candidate.end))
        last_end = candidate.end
    return selected


def _trim_whitespace(text: str, start: int, end: int) -> tuple[int, int]:
    """Trim leading and trailing whitespace from a text span."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _split_spans(text: str, boundaries: list[tuple[int, int]]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for boundary_start, boundary_end in boundaries:
        span_start, span_end = _trim_whitespace(text, start, boundary_start)
        if span_start < span_end:
            spans.append((span_start, span_end))
        start = boundary_end

    span_start, span_end = _trim_whitespace(text, start, len(text))
    if span_start < span_end:
        spans.append((span_start, span_end))
    return spans


def _normalize_for_signal(text: str) -> str:
    """Normalize text for low-signal detection without altering source spans."""
    decoded = html.unescape(html.unescape(text))
    no_invisible = "".join(
        ch for ch in decoded if unicodedata.category(ch) not in _INVISIBLE_CATEGORIES
    )
    collapsed = _HORIZONTAL_WS_PATTERN.sub(" ", no_invisible)
    collapsed = _EXCESS_NEWLINES_PATTERN.sub("\n\n", collapsed)
    return collapsed.strip()


def _has_letter_or_number(text: str) -> bool:
    """Return True if normalized text contains any alphanumeric character."""
    return any(ch.isalnum() for ch in text)


def _span_signal_kind(span_text: str) -> str:
    """Classify span text into content/blank/bridge/list_marker."""
    normalized = _normalize_for_signal(span_text)
    if not normalized:
        return _SIGNAL_KIND_BLANK
    if _LIST_MARKER_PATTERN.fullmatch(normalized):
        return _SIGNAL_KIND_LIST_MARKER
    if _BRIDGE_PATTERN.fullmatch(normalized):
        return _SIGNAL_KIND_BRIDGE
    if not _has_letter_or_number(normalized):
        return _SIGNAL_KIND_BLANK
    return _SIGNAL_KIND_CONTENT


def _is_soft_gap(gap_text: str) -> bool:
    """Return True when a boundary gap is only whitespace/entity noise."""
    return _normalize_for_signal(gap_text) == ""


def _prefer_attach_to_next(
    text: str,
    prev_end: int,
    span_start: int,
    span_end: int,
    next_start: int,
) -> bool:
    gap_before = text[prev_end:span_start]
    gap_after = text[span_end:next_start]
    soft_before = _is_soft_gap(gap_before)
    soft_after = _is_soft_gap(gap_after)
    if soft_after and not soft_before:
        return True
    if soft_before and not soft_after:
        return False
    return False


def _merge_low_signal_spans(
    text: str, spans: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Merge low-signal spans into neighbouring content spans."""
    classified = [
        _SignalSpan(start=start, end=end, kind=_span_signal_kind(text[start:end]))
        for start, end in spans
        if start < end
    ]
    if not classified:
        return []

    merged: list[list[int]] = []
    index = 0

    def _prepend_to_next(next_span: _SignalSpan, new_start: int) -> None:
        next_span.start = new_start
        next_span.kind = _span_signal_kind(text[next_span.start : next_span.end])

    while index < len(classified):
        span = classified[index]
        if span.kind == _SIGNAL_KIND_CONTENT:
            merged.append([span.start, span.end])
            index += 1
            continue

        prev = merged[-1] if merged else None
        next_span = classified[index + 1] if index + 1 < len(classified) else None

        if span.kind == _SIGNAL_KIND_LIST_MARKER and next_span is not None:
            _prepend_to_next(next_span, span.start)
            index += 1
            continue

        if (
            span.kind == _SIGNAL_KIND_BRIDGE
            and prev is not None
            and next_span is not None
            and next_span.kind == _SIGNAL_KIND_CONTENT
            and _is_soft_gap(text[prev[1] : span.start])
            and _is_soft_gap(text[span.end : next_span.start])
        ):
            prev[1] = next_span.end
            index += 2
            continue

        if prev is None and next_span is not None:
            _prepend_to_next(next_span, span.start)
        elif prev is not None and next_span is not None:
            if _prefer_attach_to_next(
                text, prev[1], span.start, span.end, next_span.start
            ):
                _prepend_to_next(next_span, span.start)
            else:
                prev[1] = span.end
        elif prev is not None:
            prev[1] = span.end
        else:
            merged.append([span.start, span.end])

        index += 1

    return [(start, end) for start, end in merged if start < end]


def _anchor_long_spans(
    text: str,
    spans: list[tuple[int, int]],
    *,
    anchor_every_words: int,
    long_sentence_word_threshold: int,
    min_sentence_words: int,
    context: _SplitContext,
) -> list[tuple[int, int]]:
    split_spans: list[tuple[int, int]] = []
    for start, end in spans:
        split_spans.extend(
            _anchor_span_if_long(
                text,
                start,
                end,
                anchor_every_words=anchor_every_words,
                long_sentence_word_threshold=long_sentence_word_threshold,
                min_sentence_words=min_sentence_words,
                context=context,
            )
        )
    return split_spans


def _anchor_span_if_long(
    text: str,
    start: int,
    end: int,
    *,
    anchor_every_words: int,
    long_sentence_word_threshold: int,
    min_sentence_words: int,
    context: _SplitContext,
) -> list[tuple[int, int]]:
    words = [
        match
        for match in _WORD_PATTERN.finditer(text, start, end)
        if context.count_word(match)
    ]
    if len(words) <= long_sentence_word_threshold:
        return [(start, end)]
    softened = _split_span_by_soft_boundaries(
        text,
        start,
        end,
        target_words=anchor_every_words,
        max_words=long_sentence_word_threshold,
        min_sentence_words=min_sentence_words,
        words=words,
        context=context,
    )
    split_spans: list[tuple[int, int]] = []
    for soft_start, soft_end in softened:
        soft_words = [
            match
            for match in _WORD_PATTERN.finditer(text, soft_start, soft_end)
            if context.count_word(match)
        ]
        if len(soft_words) <= long_sentence_word_threshold:
            split_spans.append((soft_start, soft_end))
            continue
        split_spans.extend(
            _split_span_by_word_target(
                text,
                soft_start,
                soft_end,
                target_words=anchor_every_words,
                min_sentence_words=min_sentence_words,
                words=soft_words,
                context=context,
            )
        )
    return split_spans or [(start, end)]


def _split_span_by_soft_boundaries(
    text: str,
    start: int,
    end: int,
    *,
    target_words: int,
    max_words: int,
    min_sentence_words: int,
    words: list[re.Match[str]],
    context: _SplitContext,
) -> list[tuple[int, int]]:
    if len(words) <= max_words:
        return [(start, end)]

    boundaries = _collect_soft_boundaries(text, start, end, context)
    if not boundaries:
        return [(start, end)]

    spans: list[tuple[int, int]] = []
    span_start = start
    while True:
        span_words = _count_words(text, span_start, end, context)
        if span_words <= max_words:
            split_start, split_end = _trim_whitespace(text, span_start, end)
            if split_start < split_end:
                spans.append((split_start, split_end))
            return spans if spans else [(start, end)]

        boundary = _choose_soft_boundary(
            text,
            span_start,
            end,
            boundaries,
            target_words=target_words,
            min_sentence_words=min_sentence_words,
            context=context,
        )
        if boundary is None:
            split_start, split_end = _trim_whitespace(text, span_start, end)
            if split_start < split_end:
                spans.append((split_start, split_end))
            return spans if spans else [(start, end)]

        boundary_start, boundary_end = boundary
        split_start, split_end = _trim_whitespace(text, span_start, boundary_start)
        if split_start < split_end:
            spans.append((split_start, split_end))
        span_start = boundary_end


def _collect_soft_boundaries(
    text: str, start: int, end: int, context: _SplitContext
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []

    for match in _SEMICOLON_OR_COLON_BOUNDARY_PATTERN.finditer(text, start, end):
        boundary_start, boundary_end = match.span()
        if _entity_ends_at(text, boundary_start):
            continue
        if not context.boundary_allowed(boundary_start, boundary_end):
            continue
        candidates.append((boundary_start, boundary_end))

    for match in _SINGLE_NEWLINE_BOUNDARY_PATTERN.finditer(text, start, end):
        boundary_start, boundary_end = match.span()
        if not context.boundary_allowed(boundary_start, boundary_end):
            continue
        candidates.append((boundary_start, boundary_end))

    return sorted(candidates)


def _choose_soft_boundary(
    text: str,
    start: int,
    end: int,
    boundaries: list[tuple[int, int]],
    *,
    target_words: int,
    min_sentence_words: int,
    context: _SplitContext,
) -> tuple[int, int] | None:
    best_boundary: tuple[int, int] | None = None
    best_score: tuple[int, int] | None = None
    effective_min_words = min(min_sentence_words, target_words)

    for boundary_start, boundary_end in boundaries:
        if boundary_start <= start or boundary_end >= end:
            continue
        left_words = _count_words(text, start, boundary_start, context)
        right_words = _count_words(text, boundary_end, end, context)
        if left_words < effective_min_words or right_words < effective_min_words:
            continue
        score = (abs(left_words - target_words), abs(left_words - right_words))
        if best_score is None or score < best_score:
            best_boundary = (boundary_start, boundary_end)
            best_score = score

    return best_boundary


def _split_span_by_word_target(
    text: str,
    start: int,
    end: int,
    *,
    target_words: int,
    min_sentence_words: int,
    words: list[re.Match[str]],
    context: _SplitContext,
) -> list[tuple[int, int]]:
    if len(words) <= target_words:
        return [(start, end)]

    spans: list[tuple[int, int]] = []
    span_start = start
    word_offset = 0
    effective_min_words = min(min_sentence_words, target_words)

    while word_offset < len(words):
        remaining_words = len(words) - word_offset
        if remaining_words <= target_words + effective_min_words:
            split_start, split_end = _trim_whitespace(text, span_start, end)
            if split_start < split_end:
                spans.append((split_start, split_end))
            break

        words_in_chunk = min(target_words, remaining_words - effective_min_words)
        word_end = words[word_offset + words_in_chunk - 1].end()
        cut = _find_whitespace_cut(text, word_end, end, context)
        if cut is None or cut <= span_start:
            split_start, split_end = _trim_whitespace(text, span_start, end)
            if split_start < split_end:
                spans.append((split_start, split_end))
            break

        split_start, split_end = _trim_whitespace(text, span_start, cut)
        if split_start < split_end:
            spans.append((split_start, split_end))
        span_start = cut
        while word_offset < len(words) and words[word_offset].start() < span_start:
            word_offset += 1

    return spans if spans else [(start, end)]


def _find_whitespace_cut(
    text: str, start: int, end: int, context: _SplitContext
) -> int | None:
    """Find the nearest whitespace position to cut, preferring forward scan."""
    if start >= end:
        return None

    right = start
    while right < end:
        if text[right].isspace() and context.cut_allowed(right):
            return right
        right += 1

    left = start - 1
    while left >= 0:
        if text[left].isspace() and context.cut_allowed(left):
            return left + 1
        left -= 1

    return None


def _count_words(text: str, start: int, end: int, context: _SplitContext) -> int:
    return sum(
        1
        for match in _WORD_PATTERN.finditer(text, start, end)
        if context.count_word(match)
    )


def _has_terminal_ending(span_text: str) -> bool:
    stripped = span_text.rstrip()
    if not stripped:
        return False
    return re.search(rf"[.!?\u2026](?:[{_CLOSING}])*$", stripped) is not None


def _starts_with_list_marker(span_text: str) -> bool:
    normalized = _normalize_for_signal(span_text)
    if not normalized:
        return False
    return re.match(rf"{_LIST_MARKER_PATTERN.pattern}\s", normalized) is not None


def _has_strong_layout_gap(gap_text: str) -> bool:
    return gap_text.count("\n") >= 2 or (
        _SEPARATOR_BOUNDARY_PATTERN.search(gap_text) is not None
    )


def _merge_short_nonterminal_spans(
    text: str,
    spans: list[tuple[int, int]],
    *,
    min_sentence_words: int,
    target_words: int,
    context: _SplitContext,
) -> list[tuple[int, int]]:
    if not spans:
        return []

    pending = list(spans)
    merged: list[list[int]] = []
    index = 0
    while index < len(pending):
        start, end = pending[index]
        word_count = _count_words(text, start, end, context)
        span_text = text[start:end]
        if (
            word_count >= min_sentence_words
            or word_count >= target_words
            or _has_terminal_ending(span_text)
            or _starts_with_list_marker(span_text)
            or (not merged and len(pending) == 1)
        ):
            merged.append([start, end])
            index += 1
            continue

        next_span: tuple[int, int] | None = None
        if merged:
            prev_start, prev_end = merged[-1]
        else:
            prev_start, prev_end = (-1, -1)

        if index + 1 < len(pending):
            next_span = pending[index + 1]

        if next_span is None and merged:
            if _has_strong_layout_gap(text[prev_end:start]):
                merged.append([start, end])
            else:
                merged[-1][1] = end
            index += 1
            continue
        if not merged and next_span is not None:
            if _has_strong_layout_gap(text[end : next_span[0]]):
                merged.append([start, end])
            else:
                pending[index + 1] = (start, next_span[1])
            index += 1
            continue
        if not merged:
            merged.append([start, end])
            index += 1
            continue
        assert next_span is not None
        prev_words = _count_words(text, prev_start, prev_end, context)
        next_words = _count_words(text, next_span[0], next_span[1], context)
        if _has_strong_layout_gap(text[prev_end:start]):
            pending[index + 1] = (start, next_span[1])
            index += 1
            continue
        if _has_strong_layout_gap(text[end : next_span[0]]):
            merged[-1][1] = end
            index += 1
            continue
        attach_prev_score = abs((prev_words + word_count) - target_words)
        attach_next_score = abs((next_words + word_count) - target_words)
        if attach_prev_score <= attach_next_score:
            merged[-1][1] = end
        else:
            pending[index + 1] = (start, next_span[1])
        index += 1

    return [(start, end) for start, end in merged if start < end]


def _spans_to_sentences(text: str, spans: list[tuple[int, int]]) -> list[Sentence]:
    """Build sentence objects from spans using exact source slices."""
    return [
        Sentence(index=index, start=start, end=end, text=text[start:end])
        for index, (start, end) in enumerate(spans)
    ]
