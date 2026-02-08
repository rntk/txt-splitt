"""Protocol definitions for the 5-stage text splitter pipeline."""

from typing import Protocol

from txt_splitt.types import MarkedText, Sentence, SentenceGroup


class LLMCallable(Protocol):
    """Protocol for LLM client callables."""

    def call(self, prompt: str, temperature: float) -> str: ...


class SentenceSplitter(Protocol):
    """Stage 1: Split raw text into sentences."""

    def split(self, text: str) -> list[Sentence]: ...


class MarkerStrategy(Protocol):
    """Stage 2: Apply markers to sentences, producing tagged text."""

    def mark(self, text: str, sentences: list[Sentence]) -> MarkedText: ...


class LLMStrategy(Protocol):
    """Stage 3: Query an LLM with marked text."""

    def query(self, marked_text: MarkedText) -> str: ...


class ResponseParser(Protocol):
    """Stage 4: Parse raw LLM response into sentence groups."""

    def parse(self, response: str, sentence_count: int) -> list[SentenceGroup]: ...


class GapHandler(Protocol):
    """Stage 5: Validate and handle gaps in sentence coverage."""

    def handle(
        self, groups: list[SentenceGroup], sentence_count: int
    ) -> list[SentenceGroup]: ...


class Enhancer(Protocol):
    """Stage 6 (optional): Refine group boundaries for short sentences."""

    def enhance(
        self, groups: list[SentenceGroup], sentences: list[Sentence]
    ) -> list[SentenceGroup]: ...
