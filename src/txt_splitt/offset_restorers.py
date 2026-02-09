"""Offset restoration implementations."""

from __future__ import annotations

from txt_splitt.types import OffsetMapping, Sentence, SplitResult


class MappingOffsetRestorer:
    """Remap sentence positions from clean-text to original-text coordinates.

    Each ``Sentence.start`` and ``Sentence.end`` is mapped back to positions in
    the original HTML text using the provided ``OffsetMapping``.  The
    ``Sentence.text`` field is preserved as clean text (without HTML tags).

    After restoration ``original_text[s.start:s.end]`` may include HTML tags
    that were stripped.  The ``s.text`` field contains the clean version.
    """

    def restore(self, result: SplitResult, mapping: OffsetMapping) -> SplitResult:
        if not result.sentences:
            return result

        restored: list[Sentence] = []
        for sent in result.sentences:
            restored.append(
                Sentence(
                    index=sent.index,
                    start=mapping.to_original(sent.start),
                    end=mapping.to_original(sent.end),
                    text=sent.text,
                )
            )

        return SplitResult(
            sentences=tuple(restored),
            groups=result.groups,
        )
