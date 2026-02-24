"""Word splitter implementations for the keyword extraction pipeline."""

from __future__ import annotations

import re

from txt_splitt.keyword_types import Word

_WORD_RE = re.compile(r"\S+")


class RegexWordSplitter:
    """Split text into words by matching non-whitespace sequences.

    Each ``Word`` carries the original character offsets from the source text.
    """

    def split(self, text: str) -> list[Word]:
        words: list[Word] = []
        for index, match in enumerate(_WORD_RE.finditer(text)):
            words.append(
                Word(
                    index=index,
                    start=match.start(),
                    end=match.end(),
                    text=match.group(),
                )
            )
        return words
