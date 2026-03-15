"""HTML tag cleaning implementations."""

from __future__ import annotations

import re
from html.parser import HTMLParser

from txt_splitt.types import OffsetMapping, OffsetSegment

_HTML_TAG_PATTERN = re.compile(r"""<(?:[^>"']|"[^"]*"|'[^']*')*>""")


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sort and merge overlapping or adjacent spans."""
    if not spans:
        return []

    spans.sort()
    merged: list[tuple[int, int]] = [spans[0]]
    for start, end in spans[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _append_preserved_run(
    clean_parts: list[str],
    segments: list[OffsetSegment],
    clean_offset: int,
    run_text: str,
    original_offset: int,
    separator_offset: int | None,
) -> int:
    """Append a preserved text run and an optional synthetic separator."""
    if not run_text:
        return clean_offset

    if (
        clean_parts
        and separator_offset is not None
        and not clean_parts[-1][-1].isspace()
        and not run_text[0].isspace()
    ):
        clean_parts.append(" ")
        # Synthetic separators stay back-mappable by anchoring them to the
        # start of the removed span between two preserved text runs.
        segments.append(
            OffsetSegment(
                clean_offset=clean_offset,
                original_offset=separator_offset,
                length=1,
            )
        )
        clean_offset += 1

    clean_parts.append(run_text)
    segments.append(
        OffsetSegment(
            clean_offset=clean_offset,
            original_offset=original_offset,
            length=len(run_text),
        )
    )
    return clean_offset + len(run_text)


def _build_clean_text_and_mapping(
    text: str,
    spans: list[tuple[int, int]],
) -> tuple[str, OffsetMapping]:
    """Build clean text and offset mapping from spans to remove."""
    if not spans:
        mapping = OffsetMapping(
            segments=(
                OffsetSegment(clean_offset=0, original_offset=0, length=len(text)),
            ),
            original_length=len(text),
            clean_length=len(text),
        )
        return text, mapping

    merged = _merge_spans(spans)
    segments: list[OffsetSegment] = []
    clean_parts: list[str] = []
    clean_offset = 0
    last_end = 0
    separator_offset: int | None = None

    for span_start, span_end in merged:
        if span_start > last_end:
            clean_offset = _append_preserved_run(
                clean_parts=clean_parts,
                segments=segments,
                clean_offset=clean_offset,
                run_text=text[last_end:span_start],
                original_offset=last_end,
                separator_offset=separator_offset,
            )
        last_end = span_end
        separator_offset = span_start

    if last_end < len(text):
        clean_offset = _append_preserved_run(
            clean_parts=clean_parts,
            segments=segments,
            clean_offset=clean_offset,
            run_text=text[last_end:],
            original_offset=last_end,
            separator_offset=separator_offset,
        )

    clean_text = "".join(clean_parts)
    mapping = OffsetMapping(
        segments=tuple(segments),
        original_length=len(text),
        clean_length=clean_offset,
    )
    return clean_text, mapping


class TagStripCleaner:
    """Strip HTML tags from text, producing clean text and an offset mapping.

    Uses a robust regex pattern (handles quoted attributes) to identify tags.
    Non-tag text segments are preserved with their original offsets recorded
    in the returned ``OffsetMapping``.

    Parameters
    ----------
    strip_tags:
        Optional set of tag names (e.g. ``{"script", "style"}``) whose
        entire content (opening tag, inner text, and closing tag) should
        be removed from the output.  When *None* (the default), only the
        tags themselves are stripped and inner text is kept.
    """

    def __init__(self, strip_tags: set[str] | None = None) -> None:
        self._strip_pattern: re.Pattern[str] | None = None
        if strip_tags:
            tags_alt = "|".join(re.escape(t) for t in sorted(strip_tags))
            self._strip_pattern = re.compile(
                rf"<({tags_alt})\b[^>]*>.*?</\1\s*>",
                re.IGNORECASE | re.DOTALL,
            )

    def clean(self, text: str) -> tuple[str, OffsetMapping]:
        if not text:
            return text, OffsetMapping(segments=(), original_length=0, clean_length=0)

        # Collect all regions to skip (individual tags + strip-tag blocks)
        skip_regions: list[tuple[int, int]] = [
            (m.start(), m.end()) for m in _HTML_TAG_PATTERN.finditer(text)
        ]
        if self._strip_pattern:
            skip_regions.extend(
                (m.start(), m.end()) for m in self._strip_pattern.finditer(text)
            )

        return _build_clean_text_and_mapping(text, skip_regions)


class _TagSpanParser(HTMLParser):
    """Collect start/end offsets for parsed HTML-like tags/declarations.

    Parameters
    ----------
    strip_tags:
        Optional set of lowercase tag names whose entire content should be
        treated as a single span to remove.
    """

    def __init__(self, strip_tags: set[str] | None = None) -> None:
        super().__init__(convert_charrefs=False)
        self.spans: list[tuple[int, int]] = []
        self._strip_tags: set[str] = (
            {t.lower() for t in strip_tags} if strip_tags else set()
        )
        self._strip_start: int | None = None
        self._last_tag: str = ""
        self._source_length: int = 0

    # -- tag-name capture callbacks ------------------------------------------

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._last_tag = tag.lower()

    def handle_endtag(self, tag: str) -> None:
        self._last_tag = tag.lower()

    def feed(self, data: str) -> None:
        self._source_length += len(data)
        super().feed(data)

    def close(self) -> None:
        super().close()
        if self._strip_start is not None:
            self.spans.append((self._strip_start, self._source_length))
            self._strip_start = None

    # -- position-tracking overrides -----------------------------------------

    def parse_starttag(self, i: int) -> int:
        self._last_tag = ""
        end = super().parse_starttag(i)
        if end >= 0:
            starttag_text = self.get_starttag_text()
            is_self_closing = bool(
                starttag_text is not None and starttag_text.rstrip().endswith("/>")
            )
            if self._strip_start is not None:
                pass  # inside a stripped block – ignore inner tags
            elif self._last_tag in self._strip_tags:
                if is_self_closing:
                    self.spans.append((i, end))
                else:
                    self._strip_start = i
            else:
                self.spans.append((i, end))
        return end

    def parse_endtag(self, i: int) -> int:
        self._last_tag = ""
        end = super().parse_endtag(i)
        if end >= 0:
            if self._strip_start is not None:
                if self._last_tag in self._strip_tags:
                    self.spans.append((self._strip_start, end))
                    self._strip_start = None
            else:
                self.spans.append((i, end))
        return end

    def parse_comment(self, i: int, report: bool = True) -> int:
        end = super().parse_comment(i, report=report)
        if end >= 0 and self._strip_start is None:
            self.spans.append((i, end))
        return end

    def parse_pi(self, i: int) -> int:
        end = super().parse_pi(i)
        if end >= 0 and self._strip_start is None:
            self.spans.append((i, end))
        return end

    def parse_html_declaration(self, i: int) -> int:
        end = super().parse_html_declaration(i)
        if end >= 0 and self._strip_start is None:
            self.spans.append((i, end))
        return end

    def parse_bogus_comment(self, i: int, report: bool = True) -> int:
        end = super().parse_bogus_comment(i, report=report)
        if end >= 0 and self._strip_start is None:
            self.spans.append((i, end))
        return end


class HTMLParserTagStripCleaner:
    """Strip HTML constructs using Python's built-in ``html.parser``.

    Parameters
    ----------
    strip_tags:
        Optional set of tag names (e.g. ``{"script", "style"}``) whose
        entire content (opening tag, inner text, and closing tag) should
        be removed from the output.  When *None* (the default), only the
        tags themselves are stripped and inner text is kept.
    """

    def __init__(self, strip_tags: set[str] | None = None) -> None:
        self._strip_tags = strip_tags

    def clean(self, text: str) -> tuple[str, OffsetMapping]:
        if not text:
            return text, OffsetMapping(segments=(), original_length=0, clean_length=0)

        parser = _TagSpanParser(strip_tags=self._strip_tags)
        parser.feed(text)
        parser.close()
        spans = parser.spans
        return _build_clean_text_and_mapping(text, spans)
