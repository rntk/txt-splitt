"""txt_splitt - A modular, pipeline-based text splitter."""

from txt_splitt.chunkers import OverlapChunker, SizeBasedChunker
from txt_splitt.enhancers import ShortSentenceEnhancer
from txt_splitt.errors import (
    EnhancerError,
    GapError,
    HtmlCleanError,
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
from txt_splitt.html_cleaners import HTMLParserTagStripCleaner, TagStripCleaner
from txt_splitt.joiners import AdjacentSameTopicJoiner
from txt_splitt.llm import TopicRangeLLM
from txt_splitt.markers import BracketMarker
from txt_splitt.normalizers import NormalizingSplitter
from txt_splitt.offset_restorers import MappingOffsetRestorer
from txt_splitt.parsers import TopicRangeParser
from txt_splitt.pipeline import Pipeline
from txt_splitt.retry import RetryingLLMCallable
from txt_splitt.protocols import (
    Enhancer,
    GapHandler,
    GroupJoiner,
    HtmlCleaner,
    LLMCallable,
    LLMStrategy,
    MarkedTextChunker,
    MarkerStrategy,
    OffsetRestorer,
    ResponseParser,
    SentenceSplitter,
)
from txt_splitt.splitters import (
    DenseRegexSentenceSplitter,
    HtmlAwareSentenceSplitter,
    RegexSentenceSplitter,
    SparseRegexSentenceSplitter,
)
from txt_splitt.tracer import NoOpSpan, NoOpTracer, Span, Tracer, TracingLLMCallable
from txt_splitt.types import (
    MarkedText,
    OffsetMapping,
    OffsetSegment,
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
    "OffsetMapping",
    "OffsetSegment",
    "Sentence",
    "SentenceGroup",
    "SentenceRange",
    "SplitResult",
    # Protocols
    "Enhancer",
    "GapHandler",
    "GroupJoiner",
    "HtmlCleaner",
    "LLMCallable",
    "LLMStrategy",
    "MarkedTextChunker",
    "MarkerStrategy",
    "OffsetRestorer",
    "ResponseParser",
    "SentenceSplitter",
    # Concrete implementations
    "BracketMarker",
    "MappingOffsetRestorer",
    "OverlapChunker",
    "SizeBasedChunker",
    "NormalizingSplitter",
    "HTMLParserTagStripCleaner",
    "TagStripCleaner",
    "AdjacentSameTopicJoiner",
    "DenseRegexSentenceSplitter",
    "HtmlAwareSentenceSplitter",
    "RegexSentenceSplitter",
    "SparseRegexSentenceSplitter",
    "ShortSentenceEnhancer",
    "LLMRepairingGapHandler",
    "RepairingGapHandler",
    "StrictGapHandler",
    "TopicRangeLLM",
    "TopicRangeParser",
    # Retry
    "RetryingLLMCallable",
    # Tracing
    "NoOpSpan",
    "NoOpTracer",
    "Span",
    "Tracer",
    "TracingLLMCallable",
    # Errors
    "EnhancerError",
    "GapError",
    "HtmlCleanError",
    "LLMError",
    "MarkerError",
    "ParseError",
    "SentenceSplitError",
    "SplitterError",
]
