"""Insight-oriented data types."""

from dataclasses import dataclass

from txt_splitt.sentences.types import Sentence, SentenceRange

__all__ = [
    "Insight",
    "InsightResult",
]


@dataclass(frozen=True, slots=True)
class Insight:
    """A named insight referencing specific sentences."""

    name: str
    ranges: tuple[SentenceRange, ...]


@dataclass(frozen=True, slots=True)
class InsightResult:
    """Final result of the insight extraction pipeline."""

    sentences: tuple[Sentence, ...]
    insights: tuple[Insight, ...]
