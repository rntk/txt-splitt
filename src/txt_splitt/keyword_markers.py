"""Word marker implementations for the keyword extraction pipeline."""

from __future__ import annotations

from txt_splitt.keyword_types import MarkedWords, Word


class WordBracketMarker:
    """Mark words inline using ``{N} word`` format.

    Output example: ``{0} hello {1} world {2} this {3} is {4} text``

    Markers are global (0-indexed across the entire text).
    """

    def mark(self, text: str, words: list[Word]) -> MarkedWords:
        if not words:
            return MarkedWords(tagged_text="", word_count=0)

        parts: list[str] = []
        for word in words:
            parts.append(f"{{{word.index}}} {word.text}")

        tagged_text = " ".join(parts)
        return MarkedWords(tagged_text=tagged_text, word_count=len(words))
