"""Sentence-oriented pipeline components."""

from txt_splitt.sentences.batch_pipeline import BatchPipeline
from txt_splitt.sentences.boundary_evaluator import BoundaryEvaluator
from txt_splitt.sentences.builders import SentencePipeline, build_pipeline
from txt_splitt.sentences.chunkers import OverlapChunker, SizeBasedChunker
from txt_splitt.sentences.enhancers import ShortSentenceEnhancer
from txt_splitt.sentences.gap_handlers import (
    LLMRepairingGapHandler,
    RepairingGapHandler,
    StrictGapHandler,
)
from txt_splitt.sentences.joiners import (
    AdjacentSameTopicJoiner,
    join_sentences_by_groups,
)
from txt_splitt.sentences.llm import (
    HierarchicalTopicRangeLLM,
    TopicRangeLLM,
)
from txt_splitt.sentences.markers import BracketMarker
from txt_splitt.sentences.normalizers import NormalizingSplitter
from txt_splitt.sentences.offset_restorers import MappingOffsetRestorer
from txt_splitt.sentences.parsers import TopicRangeParser
from txt_splitt.sentences.protocols import (
    Enhancer,
    GapHandler,
    GroupJoiner,
    LLMStrategy,
    MarkedTextChunker,
    MarkerStrategy,
    RangeAssigner,
    ResponseParser,
    SentenceSplitter,
    TopicExtractor,
)
from txt_splitt.sentences.splitters import SparseRegexSentenceSplitter
from txt_splitt.sentences.text_optimizers import (
    OptimizingMarker,
    is_content_free,
    normalize_for_llm,
)
from txt_splitt.sentences.types import (
    MarkedText,
    PreparedChunk,
    PreparedDocument,
    Sentence,
    SentenceGroup,
    SentenceRange,
    SplitResult,
    _indices_to_ranges,
)

__all__ = [
    "AdjacentSameTopicJoiner",
    "BatchPipeline",
    "BoundaryEvaluator",
    "BracketMarker",
    "Enhancer",
    "GapHandler",
    "GroupJoiner",
    "HierarchicalTopicRangeLLM",
    "LLMRepairingGapHandler",
    "LLMStrategy",
    "MarkedText",
    "MarkedTextChunker",
    "MappingOffsetRestorer",
    "MarkerStrategy",
    "NormalizingSplitter",
    "OptimizingMarker",
    "OverlapChunker",
    "PreparedChunk",
    "PreparedDocument",
    "RangeAssigner",
    "RepairingGapHandler",
    "ResponseParser",
    "Sentence",
    "SentencePipeline",
    "SentenceGroup",
    "SentenceRange",
    "SentenceSplitter",
    "ShortSentenceEnhancer",
    "SizeBasedChunker",
    "SparseRegexSentenceSplitter",
    "SplitResult",
    "StrictGapHandler",
    "TopicExtractor",
    "TopicRangeLLM",
    "TopicRangeParser",
    "_indices_to_ranges",
    "build_pipeline",
    "is_content_free",
    "join_sentences_by_groups",
    "normalize_for_llm",
]
