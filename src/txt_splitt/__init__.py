"""txt_splitt - A modular, pipeline-based text splitter."""

from txt_splitt.chunkers import SizeBasedChunker
from txt_splitt.enhancers import ShortSentenceEnhancer
from txt_splitt.errors import (
    EnhancerError,
    GapError,
    LLMError,
    MarkerError,
    ParseError,
    SentenceSplitError,
    SplitterError,
)
from txt_splitt.gap_handlers import (
    LLMRepairingGapHandler,
    RepairingGapHandler,
    StrictGapHandler,
)
from txt_splitt.llm import TopicRangeLLM
from txt_splitt.markers import BracketMarker
from txt_splitt.normalizers import NormalizingSplitter
from txt_splitt.parsers import TopicRangeParser
from txt_splitt.pipeline import Pipeline
from txt_splitt.protocols import (
    Enhancer,
    GapHandler,
    LLMCallable,
    LLMStrategy,
    MarkedTextChunker,
    MarkerStrategy,
    ResponseParser,
    SentenceSplitter,
)
from txt_splitt.splitters import (
    DenseRegexSentenceSplitter,
    HtmlAwareSentenceSplitter,
    RegexSentenceSplitter,
)
from txt_splitt.tracer import NoOpSpan, NoOpTracer, Span, Tracer, TracingLLMCallable
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
    "Enhancer",
    "GapHandler",
    "LLMCallable",
    "LLMStrategy",
    "MarkedTextChunker",
    "MarkerStrategy",
    "ResponseParser",
    "SentenceSplitter",
    # Concrete implementations
    "BracketMarker",
    "SizeBasedChunker",
    "NormalizingSplitter",
    "DenseRegexSentenceSplitter",
    "HtmlAwareSentenceSplitter",
    "RegexSentenceSplitter",
    "ShortSentenceEnhancer",
    "LLMRepairingGapHandler",
    "RepairingGapHandler",
    "StrictGapHandler",
    "TopicRangeLLM",
    "TopicRangeParser",
    # Tracing
    "NoOpSpan",
    "NoOpTracer",
    "Span",
    "Tracer",
    "TracingLLMCallable",
    # Errors
    "EnhancerError",
    "GapError",
    "LLMError",
    "MarkerError",
    "ParseError",
    "SentenceSplitError",
    "SplitterError",
]
