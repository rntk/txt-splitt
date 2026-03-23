"""Keyword-oriented protocol definitions."""

from typing import Protocol

from txt_splitt.keywords.types import Keyword, MarkedWords, Word


class WordSplitter(Protocol):
    """Stage 1: Split raw text into words."""

    def split(self, text: str) -> list[Word]: ...


class WordMarkerStrategy(Protocol):
    """Stage 2: Apply markers to words, producing tagged text."""

    def mark(self, text: str, words: list[Word]) -> MarkedWords: ...


class KeywordLLMStrategy(Protocol):
    """Stage 3: Query an LLM with marked words."""

    def query(self, marked: MarkedWords) -> str: ...


class KeywordParser(Protocol):
    """Stage 4: Parse raw LLM response into keyword index ranges."""

    def parse(self, response: str, word_count: int) -> list[tuple[int, int]]: ...


class MarkedWordsChunker(Protocol):
    """Optional: split MarkedWords into smaller chunks for LLM querying."""

    def chunk(self, marked: MarkedWords) -> list[MarkedWords]: ...


class GapHandler(Protocol):
    """Repair large uncovered gaps in keyword coverage."""

    def handle(
        self,
        keywords: list[Keyword],
        words: list[Word],
        text: str,
    ) -> list[Keyword]: ...
