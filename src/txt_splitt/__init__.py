"""txt_splitt - A modular, pipeline-based text splitter."""

from txt_splitt._errors import (
    GapError,
    LLMError,
    MarkerError,
    ParseError,
    SentenceSplitError,
    SplitterError,
)
from txt_splitt._gap_handlers import StrictGapHandler
from txt_splitt._llm import LLMCallable, TopicRangeLLM
from txt_splitt._markers import BracketMarker
from txt_splitt._parsers import TopicRangeParser
from txt_splitt._pipeline import Pipeline
from txt_splitt._splitters import RegexSentenceSplitter
from txt_splitt._types import (
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
    # Concrete implementations
    "BracketMarker",
    "LLMCallable",
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
