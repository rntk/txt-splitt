"""Sentence splitting implementations."""

from __future__ import annotations

import bisect
import html
import re
import unicodedata
from html.parser import HTMLParser

from txt_splitt.types import Sentence

# Uppercase letter class for sentence boundary lookahead:
# - A-Z: Basic Latin capitals
# - À-Ö, Ø-Þ: Latin Extended capitals (accented, e.g. À É Ñ Ü)
# - А-ЯЁ: Cyrillic capitals (including Ё)
_UPPER = r"A-ZÀ-ÖØ-ÞА-ЯЁ"

# Closing characters that may follow sentence-ending punctuation before whitespace:
# straight/curly quotes, closing parens/brackets, guillemet
_CLOSING = r"\"'\u201D\u2019)\]\u00BB"

# Core sentence boundary fragment:
# matches [.!?…] optionally followed by a closing char, then whitespace, then uppercase
_SENT_BOUNDARY = rf"(?:(?<=[.!?\u2026])|(?<=[.!?\u2026][{_CLOSING}]))\s+(?=[{_UPPER}])"

# Compiled regex for sentence boundaries:
# - Punctuation ([.!?…]) + optional closing char + whitespace + uppercase
# - One or more newlines
_SENTENCE_BOUNDARY_PATTERN = re.compile(rf"({_SENT_BOUNDARY})|(\n+)")
_DENSE_BOUNDARY_PATTERN = re.compile(rf"({_SENT_BOUNDARY})|(\n+)|(\s+[·•|]\s+)")
_SPARSE_BOUNDARY_PATTERN = re.compile(
    rf"({_SENT_BOUNDARY})|((?<=[;:])\s+)|(\n+)|(\s+[·•|]\s+)"
)
_WORD_PATTERN = re.compile(r"\S+")
_HTML_TAG_PATTERN = re.compile(r"<(?:[^>\"']|\"[^\"]*\"|'[^']*')*>")
_HTML_ENTITY_PATTERN = re.compile(
    r"&(?:#[0-9]+|#x[0-9a-fA-F]+|[A-Za-z][A-Za-z0-9]{1,31});"
)
_LIST_MARKER_PATTERN = re.compile(
    r"(?:\d{1,3}|[a-zA-Z]|[ivxIVX]{1,5})[.)]"
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
        cleaned_spans = _split_and_cleanup_spans(text, boundaries)
        return _spans_to_sentences(text, cleaned_spans)


class DenseRegexSentenceSplitter:
    """Split text into denser marker units for topic labeling.

    Strategy:
    - Keep regex sentence boundaries from ``RegexSentenceSplitter``.
    - Also split on digest separators like ``·`` and ``|``.
    - Add soft anchors roughly every ``anchor_every_words`` words.
    - Optional ``html_aware`` mode prevents cuts inside HTML tags.
    """

    def __init__(
        self, *, anchor_every_words: int = 24, html_aware: bool = False
    ) -> None:
        if anchor_every_words <= 0:
            raise ValueError("anchor_every_words must be positive")
        self._anchor_every_words = anchor_every_words
        self._html_aware = html_aware

    def split(self, text: str) -> list[Sentence]:
        if not text or not text.strip():
            return []

        tag_starts: list[int] = []
        tag_ends: list[int] = []
        if self._html_aware:
            tag_starts, tag_ends = _compute_tag_spans(text)

        boundaries = list(_DENSE_BOUNDARY_PATTERN.finditer(text))

        valid_boundaries: list[tuple[int, int]] = []
        for match in boundaries:
            if self._html_aware and _boundary_overlaps_tag(
                match.start(),
                match.end(),
                tag_starts,
                tag_ends,
            ):
                continue
            valid_boundaries.append(match.span())

        spans = _split_and_cleanup_spans(text, valid_boundaries)

        anchored_spans: list[tuple[int, int]] = []
        for span_start, span_end in spans:
            if self._html_aware:
                anchored_spans.extend(
                    _split_span_by_word_anchor_html_aware(
                        text,
                        span_start,
                        span_end,
                        self._anchor_every_words,
                        tag_starts,
                        tag_ends,
                    )
                )
            else:
                anchored_spans.extend(
                    _split_span_by_word_anchor(
                        text,
                        span_start,
                        span_end,
                        self._anchor_every_words,
                    )
                )

        anchored_spans = _cleanup_low_signal_spans(text, anchored_spans)
        return _spans_to_sentences(text, anchored_spans)


class SparseRegexSentenceSplitter:
    """Split text by natural punctuation boundaries with sparse anchoring.

    Strategy:
    - Split first on punctuation/newline boundaries that often indicate
      sentence-like units.
    - Optional ``html_aware`` mode prevents cuts inside HTML tags.
    - Only if a resulting span is very long, split it every
      ``anchor_every_words`` words.
    """

    def __init__(
        self,
        *,
        anchor_every_words: int = 24,
        long_sentence_word_threshold: int = 48,
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
        self._anchor_every_words = anchor_every_words
        self._long_sentence_word_threshold = long_sentence_word_threshold
        self._html_aware = html_aware

    def split(self, text: str) -> list[Sentence]:
        if not text or not text.strip():
            return []

        tag_starts: list[int] = []
        tag_ends: list[int] = []
        if self._html_aware:
            tag_starts, tag_ends = _compute_tag_spans(text)

        boundaries = list(_SPARSE_BOUNDARY_PATTERN.finditer(text))

        valid_boundaries: list[tuple[int, int]] = []
        for match in boundaries:
            if _should_skip_sparse_boundary(text, match):
                continue
            if self._html_aware and _boundary_overlaps_tag(
                match.start(),
                match.end(),
                tag_starts,
                tag_ends,
            ):
                continue
            valid_boundaries.append(match.span())

        spans = _split_and_cleanup_spans(text, valid_boundaries)

        split_spans: list[tuple[int, int]] = []
        for span_start, span_end in spans:
            if self._html_aware:
                split_spans.extend(
                    _split_span_by_word_anchor_if_long_html_aware(
                        text,
                        span_start,
                        span_end,
                        self._anchor_every_words,
                        self._long_sentence_word_threshold,
                        tag_starts,
                        tag_ends,
                    )
                )
            else:
                split_spans.extend(
                    _split_span_by_word_anchor_if_long(
                        text,
                        span_start,
                        span_end,
                        self._anchor_every_words,
                        self._long_sentence_word_threshold,
                    )
                )

        split_spans = _cleanup_low_signal_spans(text, split_spans)
        return _spans_to_sentences(text, split_spans)


def _trim_whitespace(text: str, start: int, end: int) -> tuple[int, int]:
    """Trim leading and trailing whitespace from a text span."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _normalize_for_signal(text: str) -> str:
    """Normalize text for low-signal detection without altering source spans."""
    decoded = html.unescape(html.unescape(text))
    # Filter out characters in invisible/control unicode categories
    no_invisible = "".join(
        ch
        for ch in decoded
        if unicodedata.category(ch) not in _INVISIBLE_CATEGORIES
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


def _attach_to_next(gap_before: str, gap_after: str) -> bool:
    """Choose merge direction for low-signal spans using boundary softness."""
    soft_before = _is_soft_gap(gap_before)
    soft_after = _is_soft_gap(gap_after)
    if soft_after and not soft_before:
        return True
    if soft_before and not soft_after:
        return False
    return False


def _should_skip_sparse_boundary(text: str, match: re.Match[str]) -> bool:
    """Ignore sparse semicolon boundary when semicolon closes an HTML entity."""
    semicolon_or_colon_group = match.group(2)
    if semicolon_or_colon_group is None:
        return False
    boundary_start = match.start()
    if boundary_start <= 0 or text[boundary_start - 1] != ";":
        return False

    token_start = boundary_start - 1
    while token_start > 0 and not text[token_start - 1].isspace():
        token_start -= 1
    token = text[token_start:boundary_start]
    return _HTML_ENTITY_PATTERN.fullmatch(token) is not None


def _split_and_cleanup_spans(
    text: str,
    boundaries: list[re.Match[str]] | list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Split text into spans at boundaries and apply low-signal cleanup."""
    spans: list[tuple[int, int]] = []
    start = 0
    for b in boundaries:
        b_start, b_end = b.span() if isinstance(b, re.Match) else b
        s_start, s_end = _trim_whitespace(text, start, b_start)
        if s_start < s_end:
            spans.append((s_start, s_end))
        start = b_end

    s_start, s_end = _trim_whitespace(text, start, len(text))
    if s_start < s_end:
        spans.append((s_start, s_end))

    return _cleanup_low_signal_spans(text, spans)


def _cleanup_low_signal_spans(
    text: str, spans: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Merge low-signal spans into neighboring content spans.

    This prevents marker-only rows like ``·``, ``&amp;`` and ordinal bullets.
    """
    if not spans:
        return []

    mutable_spans = [[start, end] for start, end in spans]
    cleaned: list[list[int]] = []
    index = 0

    while index < len(mutable_spans):
        start, end = mutable_spans[index]
        kind = _span_signal_kind(text[start:end])
        has_prev = bool(cleaned)
        has_next = index + 1 < len(mutable_spans)

        if kind == _SIGNAL_KIND_CONTENT:
            cleaned.append([start, end])
            index += 1
            continue

        prev_end = cleaned[-1][1] if has_prev else start
        next_start = mutable_spans[index + 1][0] if has_next else end
        gap_before = text[prev_end:start] if has_prev else ""
        gap_after = text[end:next_start] if has_next else ""

        if kind == _SIGNAL_KIND_LIST_MARKER and has_next:
            # Prepend the marker to the next span. If the next span is also
            # low-signal, the marker will chain forward through the while loop.
            mutable_spans[index + 1][0] = start
            index += 1
            continue

        if kind == _SIGNAL_KIND_BRIDGE and has_prev and has_next:
            next_span_start, next_span_end = mutable_spans[index + 1]
            next_kind = _span_signal_kind(text[next_span_start:next_span_end])
            if (
                next_kind == _SIGNAL_KIND_CONTENT
                and _is_soft_gap(gap_before)
                and _is_soft_gap(gap_after)
            ):
                cleaned[-1][1] = next_span_end
                index += 2
                continue

        if has_next and (not has_prev or _attach_to_next(gap_before, gap_after)):
            mutable_spans[index + 1][0] = start
        elif has_prev:
            cleaned[-1][1] = end
        else:
            cleaned.append([start, end])

        index += 1

    return [(start, end) for start, end in cleaned if start < end]


def _spans_to_sentences(text: str, spans: list[tuple[int, int]]) -> list[Sentence]:
    """Build sentence objects from spans using exact source slices."""
    return [
        Sentence(index=index, start=start, end=end, text=text[start:end])
        for index, (start, end) in enumerate(spans)
    ]


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


def _split_span_by_word_anchor_if_long(
    text: str,
    start: int,
    end: int,
    anchor_every_words: int,
    long_sentence_word_threshold: int,
) -> list[tuple[int, int]]:
    """Split by word anchors only when span word-count exceeds threshold."""
    word_count = len(list(_WORD_PATTERN.finditer(text, start, end)))
    if word_count <= long_sentence_word_threshold:
        return [(start, end)]
    return _split_span_by_word_anchor(text, start, end, anchor_every_words)


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


# ---------------------------------------------------------------------------
# HTML-aware helpers
# ---------------------------------------------------------------------------


def _compute_tag_spans(text: str) -> tuple[list[int], list[int]]:
    """Return sorted (starts, ends) lists of all HTML tag spans in text."""
    starts: list[int] = []
    ends: list[int] = []
    for m in _HTML_TAG_PATTERN.finditer(text):
        starts.append(m.start())
        ends.append(m.end())
    return starts, ends


def _pos_inside_tag(pos: int, tag_starts: list[int], tag_ends: list[int]) -> bool:
    """Return True if *pos* falls inside an HTML tag span."""
    idx = bisect.bisect_right(tag_starts, pos) - 1
    return idx >= 0 and pos < tag_ends[idx]


def _boundary_overlaps_tag(
    b_start: int,
    b_end: int,
    tag_starts: list[int],
    tag_ends: list[int],
) -> bool:
    """Return True if the boundary range [b_start, b_end) overlaps any tag."""
    if _pos_inside_tag(b_start, tag_starts, tag_ends):
        return True
    if b_end > b_start and _pos_inside_tag(b_end - 1, tag_starts, tag_ends):
        return True
    idx = bisect.bisect_left(tag_starts, b_start)
    return idx < len(tag_starts) and tag_starts[idx] < b_end


def _find_whitespace_cut_html_aware(
    text: str,
    start: int,
    end: int,
    tag_starts: list[int],
    tag_ends: list[int],
) -> int | None:
    """Find nearest whitespace cut position that is not inside an HTML tag."""
    if start >= end:
        return None

    right = start
    while right < end:
        if text[right].isspace() and not _pos_inside_tag(right, tag_starts, tag_ends):
            return right
        right += 1

    left = start - 1
    while left >= 0:
        if text[left].isspace() and not _pos_inside_tag(left, tag_starts, tag_ends):
            return left + 1
        left -= 1

    return None


def _split_span_by_word_anchor_html_aware(
    text: str,
    start: int,
    end: int,
    anchor_every_words: int,
    tag_starts: list[int],
    tag_ends: list[int],
) -> list[tuple[int, int]]:
    """Split a span using word-count anchors, avoiding cuts inside HTML tags."""
    all_matches = list(_WORD_PATTERN.finditer(text, start, end))
    real_word_matches: list[re.Match[str]] = [
        m for m in all_matches if not _pos_inside_tag(m.start(), tag_starts, tag_ends)
    ]

    if len(real_word_matches) <= anchor_every_words:
        return [(start, end)]

    cut_points: list[int] = []
    for word_count in range(
        anchor_every_words, len(real_word_matches), anchor_every_words
    ):
        word_end = real_word_matches[word_count - 1].end()
        cut = _find_whitespace_cut_html_aware(text, word_end, end, tag_starts, tag_ends)
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


def _split_span_by_word_anchor_if_long_html_aware(
    text: str,
    start: int,
    end: int,
    anchor_every_words: int,
    long_sentence_word_threshold: int,
    tag_starts: list[int],
    tag_ends: list[int],
) -> list[tuple[int, int]]:
    """HTML-aware variant that only anchors when non-tag words exceed threshold."""
    all_matches = list(_WORD_PATTERN.finditer(text, start, end))
    real_word_count = sum(
        1 for m in all_matches if not _pos_inside_tag(m.start(), tag_starts, tag_ends)
    )
    if real_word_count <= long_sentence_word_threshold:
        return [(start, end)]
    return _split_span_by_word_anchor_html_aware(
        text,
        start,
        end,
        anchor_every_words,
        tag_starts,
        tag_ends,
    )


# ---------------------------------------------------------------------------
# HTMLParser-based HTML analysis
# ---------------------------------------------------------------------------

_BLOCK_TAGS: frozenset[str] = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "dd",
        "details",
        "div",
        "dl",
        "dt",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hgroup",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "summary",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "tr",
        "ul",
    }
)

_RAW_CONTENT_TAGS: frozenset[str] = frozenset({"script", "style"})


def _build_line_offsets(text: str) -> list[int]:
    """Build a table mapping 1-based line numbers to character offsets.

    Returns a list where ``offsets[0]`` is the char offset of line 1 (always 0),
    ``offsets[1]`` is the char offset of line 2, etc.
    """
    offsets: list[int] = [0]
    idx = 0
    while True:
        idx = text.find("\n", idx)
        if idx == -1:
            break
        offsets.append(idx + 1)
        idx += 1
    return offsets


class _HtmlAnalyzer(HTMLParser):
    """Parse HTML and collect protected spans and block-element boundaries."""

    def __init__(self, text: str) -> None:
        super().__init__(convert_charrefs=False)
        self._text = text
        self._line_offsets = _build_line_offsets(text)
        self._protected: list[tuple[int, int]] = []
        self._block_boundaries: list[int] = []
        self._in_raw_tag: str = ""
        self._raw_content_start: int = 0

    def analyze(self) -> tuple[list[int], list[int], list[int]]:
        """Feed text and return (protected_starts, protected_ends, block_boundaries)."""
        try:
            self.feed(self._text)
            self.close()
        except Exception:
            self._protected.clear()
            self._block_boundaries.clear()

        self._protected.sort()
        starts = [s for s, _ in self._protected]
        ends = [e for _, e in self._protected]
        block = sorted(set(self._block_boundaries))
        return starts, ends, block

    def _current_offset(self) -> int:
        """Convert current ``getpos()`` to a character offset."""
        line, col = self.getpos()
        return self._line_offsets[line - 1] + col

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        start = self._current_offset()
        tag_text = self.get_starttag_text()
        if tag_text is None:
            return
        end = start + len(tag_text)
        self._protected.append((start, end))
        if tag in _BLOCK_TAGS:
            self._block_boundaries.append(start)
        if tag in _RAW_CONTENT_TAGS:
            self._in_raw_tag = tag
            self._raw_content_start = end

    def handle_endtag(self, tag: str) -> None:
        start = self._current_offset()
        try:
            end = self._text.index(">", start) + 1
        except ValueError:
            return
        self._protected.append((start, end))
        if tag in _BLOCK_TAGS:
            self._block_boundaries.append(end)
        if tag == self._in_raw_tag:
            if self._raw_content_start < start:
                self._protected.append((self._raw_content_start, start))
            self._in_raw_tag = ""

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        start = self._current_offset()
        tag_text = self.get_starttag_text()
        if tag_text is None:
            return
        end = start + len(tag_text)
        self._protected.append((start, end))
        if tag in _BLOCK_TAGS:
            self._block_boundaries.append(start)
            self._block_boundaries.append(end)

    def handle_comment(self, data: str) -> None:
        start = self._current_offset()
        # <!-- data -->  =>  4 + len(data) + 3
        end = start + 4 + len(data) + 3
        self._protected.append((start, end))

    def handle_decl(self, decl: str) -> None:
        start = self._current_offset()
        try:
            end = self._text.index(">", start) + 1
        except ValueError:
            return
        self._protected.append((start, end))

    def handle_pi(self, data: str) -> None:
        start = self._current_offset()
        # <? data >  =>  2 + len(data) + 1
        end = start + 2 + len(data) + 1
        self._protected.append((start, end))

    def handle_unknown_decl(self, data: str) -> None:
        start = self._current_offset()
        try:
            end = self._text.index(">", start) + 1
        except ValueError:
            return
        self._protected.append((start, end))


def _merge_boundaries(
    regex_boundaries: list[tuple[int, int]],
    block_positions: list[int],
) -> list[tuple[int, int]]:
    """Merge regex boundary ranges with block-element boundary positions.

    Block positions become zero-width ``(pos, pos)`` boundaries.  Positions
    already covered by a regex boundary are dropped.
    """
    merged: list[tuple[int, int]] = list(regex_boundaries)
    for pos in block_positions:
        covered = any(b_start <= pos <= b_end for b_start, b_end in regex_boundaries)
        if not covered:
            merged.append((pos, pos))
    merged.sort()
    return merged


class HtmlAwareSentenceSplitter:
    """Split text into sentences using stdlib ``HTMLParser`` for HTML detection.

    Improvements over regex-based HTML handling:

    - Correctly handles HTML comments (``<!-- ... -->``)
    - Masks ``<script>`` and ``<style>`` block contents
    - Optionally uses block-level elements as sentence boundaries
    - More robust against malformed HTML
    """

    def __init__(
        self,
        *,
        anchor_every_words: int = 24,
        block_tags_as_boundaries: bool = True,
    ) -> None:
        if anchor_every_words <= 0:
            raise ValueError("anchor_every_words must be positive")
        self._anchor_every_words = anchor_every_words
        self._block_tags_as_boundaries = block_tags_as_boundaries

    def split(self, text: str) -> list[Sentence]:
        if not text or not text.strip():
            return []

        # Step 1: Analyze HTML structure
        analyzer = _HtmlAnalyzer(text)
        tag_starts, tag_ends, block_boundaries = analyzer.analyze()

        # Step 2: Find regex-based sentence boundaries
        regex_matches = list(_DENSE_BOUNDARY_PATTERN.finditer(text))

        # Step 3: Filter regex boundaries that overlap protected spans
        valid_boundaries: list[tuple[int, int]] = []
        for match in regex_matches:
            if not _boundary_overlaps_tag(
                match.start(), match.end(), tag_starts, tag_ends
            ):
                valid_boundaries.append((match.start(), match.end()))

        # Step 4: Merge block-element boundaries (if enabled)
        if self._block_tags_as_boundaries and block_boundaries:
            all_boundaries = _merge_boundaries(valid_boundaries, block_boundaries)
        else:
            all_boundaries = valid_boundaries

        # Step 5: Split text into spans using merged boundaries
        spans = _split_and_cleanup_spans(text, all_boundaries)

        # Step 6: Apply word-count anchors within each span
        anchored_spans: list[tuple[int, int]] = []
        for span_start, span_end in spans:
            anchored_spans.extend(
                _split_span_by_word_anchor_html_aware(
                    text,
                    span_start,
                    span_end,
                    self._anchor_every_words,
                    tag_starts,
                    tag_ends,
                )
            )

        anchored_spans = _cleanup_low_signal_spans(text, anchored_spans)
        # Step 7: Build Sentence objects
        return _spans_to_sentences(text, anchored_spans)
