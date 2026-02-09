"""HTML tag cleaning implementations."""

from __future__ import annotations

import re

from txt_splitt.types import OffsetMapping, OffsetSegment

_HTML_TAG_PATTERN = re.compile(r"<(?:[^>\"']|\"[^\"]*\"|'[^']*')*>")


class TagStripCleaner:
    """Strip HTML tags from text, producing clean text and an offset mapping.

    Uses a robust regex pattern (handles quoted attributes) to identify tags.
    Non-tag text segments are preserved with their original offsets recorded
    in the returned ``OffsetMapping``.
    """

    def clean(self, text: str) -> tuple[str, OffsetMapping]:
        if not text:
            return text, OffsetMapping(segments=(), original_length=0, clean_length=0)

        segments: list[OffsetSegment] = []
        clean_parts: list[str] = []
        clean_offset = 0
        last_end = 0

        for match in _HTML_TAG_PATTERN.finditer(text):
            tag_start = match.start()
            if tag_start > last_end:
                seg_length = tag_start - last_end
                segments.append(
                    OffsetSegment(
                        clean_offset=clean_offset,
                        original_offset=last_end,
                        length=seg_length,
                    )
                )
                clean_parts.append(text[last_end:tag_start])
                clean_offset += seg_length
            last_end = match.end()

        # Text after the last tag (or all text if no tags found)
        if last_end < len(text):
            seg_length = len(text) - last_end
            segments.append(
                OffsetSegment(
                    clean_offset=clean_offset,
                    original_offset=last_end,
                    length=seg_length,
                )
            )
            clean_parts.append(text[last_end:])
            clean_offset += seg_length

        clean_text = "".join(clean_parts)
        mapping = OffsetMapping(
            segments=tuple(segments),
            original_length=len(text),
            clean_length=len(clean_text),
        )
        return clean_text, mapping
