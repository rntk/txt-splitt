"""Sentence-oriented protocol definitions."""

from typing import Protocol

from txt_splitt.pipeline import StageResult
from txt_splitt.sentences.types import MarkedText, Sentence, SentenceGroup


class SentenceSplitter(Protocol):
    """Stage 1: Split raw text into sentences."""

    def split(self, text: str) -> list[Sentence]: ...


class MarkerStrategy(Protocol):
    """Stage 2: Apply markers to sentences, producing tagged text."""

    def mark(self, text: str, sentences: list[Sentence]) -> MarkedText: ...


class LLMStrategy(Protocol):
    """Stage 3: Query an LLM with marked text."""

    def query(self, marked_text: MarkedText) -> str: ...


class SchedulableLLMStrategy(Protocol):
    """Stage 3: Emit ordered LLM request batches for marked text."""

    response_format: str

    def plan_query(self, marked_text: MarkedText) -> StageResult[str]: ...


class ResponseParser(Protocol):
    """Stage 4: Parse raw LLM response into sentence groups."""

    def parse(self, response: str, sentence_count: int) -> list[SentenceGroup]: ...


class GapHandler(Protocol):
    """Stage 5: Validate and handle gaps in sentence coverage."""

    def handle(
        self,
        groups: list[SentenceGroup],
        sentence_count: int,
        sentences: list[Sentence] | None = None,
    ) -> list[SentenceGroup]: ...


class TopicExtractor(Protocol):
    """Stage 3a: Extract topic labels from marked text."""

    def extract(self, marked_text: MarkedText) -> list[str]: ...


class RangeAssigner(Protocol):
    """Stage 3b: Assign sentence ranges to given topics."""

    def assign(self, marked_text: MarkedText, topics: list[str]) -> str: ...


class MarkedTextChunker(Protocol):
    """Optional: split MarkedText into smaller chunks for LLM querying."""

    def chunk(self, marked_text: MarkedText) -> list[MarkedText]: ...


class Enhancer(Protocol):
    """Stage 6 (optional): Refine group boundaries for short sentences."""

    def enhance(
        self, groups: list[SentenceGroup], sentences: list[Sentence]
    ) -> list[SentenceGroup]: ...


class SchedulableEnhancer(Protocol):
    """Stage 6 (optional): Emit ordered LLM batches for boundary refinement."""

    stage_name: str

    def plan_process(
        self,
        groups: list[SentenceGroup],
        sentences: list[Sentence],
    ) -> StageResult[list[SentenceGroup]]: ...


class GroupJoiner(Protocol):
    """Stage 7 (optional): Join adjacent groups that belong together."""

    def join(
        self, groups: list[SentenceGroup], sentences: list[Sentence]
    ) -> list[SentenceGroup]: ...
