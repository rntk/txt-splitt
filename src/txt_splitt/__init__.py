"""txt_splitt - A modular, pipeline-based text splitter."""

from txt_splitt.errors import (
    GapError,
    LLMError,
    MarkerError,
    ParseError,
    SentenceSplitError,
    SplitterError,
)
from txt_splitt.gap_handlers import StrictGapHandler
from txt_splitt.llm import TopicRangeLLM
from txt_splitt.markers import BracketMarker
from txt_splitt.parsers import TopicRangeParser
from txt_splitt.pipeline import Pipeline
from txt_splitt.protocols import (
    GapHandler,
    LLMCallable,
    LLMStrategy,
    MarkerStrategy,
    ResponseParser,
    SentenceSplitter,
)
from txt_splitt.splitters import RegexSentenceSplitter
from txt_splitt.types import (
    MarkedText,
    Sentence,
    SentenceGroup,
    SentenceRange,
    SplitResult,
)

__all__ = [
    # Pipeline
    "Pipeline",
    # Types
    "MarkedText",
    "Sentence",
    "SentenceGroup",
    "SentenceRange",
    "SplitResult",
    # Protocols
    "GapHandler",
    "LLMCallable",
    "LLMStrategy",
    "MarkerStrategy",
    "ResponseParser",
    "SentenceSplitter",
    # Concrete implementations
    "BracketMarker",
    "RegexSentenceSplitter",
    "StrictGapHandler",
    "TopicRangeLLM",
    "TopicRangeParser",
    # Errors
    "GapError",
    "LLMError",
    "MarkerError",
    "ParseError",
    "SentenceSplitError",
    "SplitterError",
]
