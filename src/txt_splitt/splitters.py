"""Sentence splitting implementations."""

from __future__ import annotations

import bisect
import re
from html.parser import HTMLParser

from txt_splitt.types import Sentence

# Compiled regex for sentence boundaries:
# - Punctuation ([.!?]) followed by whitespace and uppercase letter (including Cyrillic)
# - One or more newlines
_SENTENCE_BOUNDARY_PATTERN = re.compile(r"((?<=[.!?])\s+(?=[A-ZА-Я]))|(\n+)")
_DENSE_BOUNDARY_PATTERN = re.compile(r"((?<=[.!?])\s+(?=[A-ZА-Я]))|(\n+)|(\s+[·•|]\s+)")
_WORD_PATTERN = re.compile(r"\S+")
_HTML_TAG_PATTERN = re.compile(r"<(?:[^>\"']|\"[^\"]*\"|'[^']*')*>")


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

        spans: list[tuple[int, int]] = []
        start = 0

        for match in boundaries:
            if self._html_aware and _boundary_overlaps_tag(
                match.start(),
                match.end(),
                tag_starts,
                tag_ends,
            ):
                continue
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
        spans: list[tuple[int, int]] = []
        start = 0
        for b_start, b_end in all_boundaries:
            s_start, s_end = _trim_whitespace(text, start, b_start)
            if s_start < s_end:
                spans.append((s_start, s_end))
            start = b_end

        # Final segment
        s_start, s_end = _trim_whitespace(text, start, len(text))
        if s_start < s_end:
            spans.append((s_start, s_end))

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

        # Step 7: Build Sentence objects
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
