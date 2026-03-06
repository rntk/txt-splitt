"""txt_splitt - A modular, pipeline-based text splitter."""

from txt_splitt.batch_pipeline import BatchPipeline
from txt_splitt.chunkers import OverlapChunker, SizeBasedChunker
from txt_splitt.enhancers import ShortSentenceEnhancer
from txt_splitt.errors import (
    EnhancerError,
    GapError,
    HtmlCleanError,
    KeywordError,
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
from txt_splitt.keyword_chunkers import WordOverlapChunker
from txt_splitt.keyword_gap_validator import KeywordGapValidator
from txt_splitt.keyword_llm import KeywordExtractionLLM
from txt_splitt.keyword_markers import WordBracketMarker
from txt_splitt.keyword_parsers import KeywordIndexParser
from txt_splitt.keyword_pipeline import KeywordPipeline
from txt_splitt.keyword_protocols import (
    KeywordGapValidatorStrategy,
    KeywordLLMStrategy,
    KeywordParser,
    MarkedWordsChunker,
    WordMarkerStrategy,
    WordSplitter,
)
from txt_splitt.keyword_splitters import RegexWordSplitter
from txt_splitt.keyword_types import Keyword, KeywordResult, MarkedWords, Word
from txt_splitt.llm import TopicListLLM, TopicRangeAssignmentLLM, TopicRangeLLM
from txt_splitt.markers import BracketMarker
from txt_splitt.normalizers import NormalizingSplitter
from txt_splitt.offset_restorers import MappingOffsetRestorer
from txt_splitt.parsers import TopicRangeParser
from txt_splitt.pipeline import Pipeline
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
    RangeAssigner,
    ResponseParser,
    SentenceSplitter,
    TopicExtractor,
)
from txt_splitt.retry import RetryingLLMCallable
from txt_splitt.splitters import SparseRegexSentenceSplitter
from txt_splitt.text_optimizers import (
    OptimizingMarker,
    is_content_free,
    normalize_for_llm,
)
from txt_splitt.tracer import (
    NoOpSpan,
    NoOpTracer,
    Span,
    Tracer,
    TracingAsyncLLMCallable,
    TracingLLMCallable,
)
from txt_splitt.types import (
    MarkedText,
    OffsetMapping,
    OffsetSegment,
    PreparedChunk,
    PreparedDocument,
    Sentence,
    SentenceGroup,
    SentenceRange,
    SplitResult,
)

__all__ = [
    # Pipeline
    "BatchPipeline",
    "KeywordPipeline",
    "Pipeline",
    # Types
    "Keyword",
    "KeywordResult",
    "MarkedText",
    "MarkedWords",
    "Word",
    "OffsetMapping",
    "OffsetSegment",
    "PreparedChunk",
    "PreparedDocument",
    "Sentence",
    "SentenceGroup",
    "SentenceRange",
    "SplitResult",
    # Protocols
    "Enhancer",
    "KeywordGapValidatorStrategy",
    "KeywordLLMStrategy",
    "KeywordParser",
    "MarkedWordsChunker",
    "WordMarkerStrategy",
    "WordSplitter",
    "GapHandler",
    "GroupJoiner",
    "HtmlCleaner",
    "LLMCallable",
    "LLMStrategy",
    "MarkedTextChunker",
    "MarkerStrategy",
    "OffsetRestorer",
    "RangeAssigner",
    "ResponseParser",
    "SentenceSplitter",
    "TopicExtractor",
    # Concrete implementations
    "BracketMarker",
    "KeywordExtractionLLM",
    "KeywordGapValidator",
    "KeywordIndexParser",
    "RegexWordSplitter",
    "WordBracketMarker",
    "WordOverlapChunker",
    "MappingOffsetRestorer",
    "OverlapChunker",
    "SizeBasedChunker",
    "NormalizingSplitter",
    "OptimizingMarker",
    "is_content_free",
    "normalize_for_llm",
    "HTMLParserTagStripCleaner",
    "TagStripCleaner",
    "AdjacentSameTopicJoiner",
    "SparseRegexSentenceSplitter",
    "ShortSentenceEnhancer",
    "LLMRepairingGapHandler",
    "RepairingGapHandler",
    "StrictGapHandler",
    "TopicListLLM",
    "TopicRangeAssignmentLLM",
    "TopicRangeLLM",
    "TopicRangeParser",
    # Retry
    "RetryingLLMCallable",
    # Tracing
    "NoOpSpan",
    "NoOpTracer",
    "Span",
    "Tracer",
    "TracingAsyncLLMCallable",
    "TracingLLMCallable",
    # Errors
    "EnhancerError",
    "KeywordError",
    "GapError",
    "HtmlCleanError",
    "LLMError",
    "MarkerError",
    "ParseError",
    "SentenceSplitError",
    "SplitterError",
]
