"""Word-aware chunker for the keyword extraction pipeline."""

from __future__ import annotations

import re

from txt_splitt.keyword_types import MarkedWords

_MARKER_RE = re.compile(r"\{(\d+)\}")

_DEFAULT_MAX_CHARS = 12_000
_DEFAULT_OVERLAP_WORDS = 20


class WordOverlapChunker:
    """Split ``MarkedWords`` into chunks with overlapping context.

    Since markers are inline (``{N} word``), this chunker splits at word
    marker boundaries by character budget and adds overlap by repeating
    the last *overlap_words* words from the previous chunk.

    Global marker numbers are preserved in the output chunks.
    """

    def __init__(
        self,
        *,
        max_chars: int = _DEFAULT_MAX_CHARS,
        overlap_words: int = _DEFAULT_OVERLAP_WORDS,
    ) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        if overlap_words < 0:
            raise ValueError("overlap_words must be non-negative")
        self._max_chars = max_chars
        self._overlap_words = overlap_words

    def chunk(self, marked: MarkedWords) -> list[MarkedWords]:
        if len(marked.tagged_text) <= self._max_chars:
            return [marked]

        # Split tagged_text into tokens: each token is "{N} word"
        tokens = _split_tokens(marked.tagged_text)

        if not tokens:
            return [marked]

        chunks: list[MarkedWords] = []
        overlap_tokens: list[str] = []
        i = 0

        while i < len(tokens):
            current_tokens = list(overlap_tokens)
            new_count = 0

            while i < len(tokens):
                token = tokens[i]
                if current_tokens:
                    n = len(current_tokens)
                    if n > 1:
                        cur_len = sum(len(t) for t in current_tokens) + n - 1
                    else:
                        cur_len = len(current_tokens[0])
                    new_len = cur_len + 1 + len(token)
                    if new_len > self._max_chars:
                        break
                current_tokens.append(token)
                i += 1
                new_count += 1

            # Guarantee progress
            if new_count == 0:
                current_tokens.append(tokens[i])
                i += 1
                new_count = 1

            text = " ".join(current_tokens)
            chunks.append(MarkedWords(tagged_text=text, word_count=len(current_tokens)))

            # overlap: last overlap_words from the new tokens of this chunk
            new_tokens = current_tokens[len(current_tokens) - new_count :]
            if self._overlap_words > 0:
                overlap_tokens = new_tokens[-self._overlap_words :]
            else:
                overlap_tokens = []

        return chunks if chunks else [marked]


def _split_tokens(tagged_text: str) -> list[str]:
    """Split inline-marked text into per-word tokens like ``{N} word``."""
    # Find all marker positions
    markers = list(_MARKER_RE.finditer(tagged_text))
    if not markers:
        return []

    tokens: list[str] = []
    for idx, m in enumerate(markers):
        start = m.start()
        end = markers[idx + 1].start() if idx + 1 < len(markers) else len(tagged_text)
        token = tagged_text[start:end].strip()
        if token:
            tokens.append(token)
    return tokens
