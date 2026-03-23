"""Keyword pipeline builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from txt_splitt.keywords.offset_restorers import MappingOffsetRestorer
from txt_splitt.keywords.results import build_keywords, build_result
from txt_splitt.pipeline import Pipeline, PipelineState, PostProcessor, StateBuilder

if TYPE_CHECKING:
    from txt_splitt.keywords.protocols import (
        GapHandler,
        KeywordLLMStrategy,
        KeywordParser,
        WordMarkerStrategy,
        WordSplitter,
    )
    from txt_splitt.keywords.types import Keyword, KeywordResult, MarkedWords, Word
    from txt_splitt.protocols import HtmlCleaner, OffsetRestorer
    from txt_splitt.tracer import Tracer


@dataclass(frozen=True, slots=True)
class _KeywordStateBuilder(
    StateBuilder["Word", list[tuple[int, int]], list["Keyword"]]
):
    def build(
        self,
        parsed: list[tuple[int, int]],
        items: list["Word"],
        text: str,
    ) -> PipelineState["Word", list["Keyword"]]:
        keywords = build_keywords(parsed, items, text)
        return PipelineState(items=list(items), value=keywords)


@dataclass(frozen=True, slots=True)
class _KeywordGapHandlerProcessor(PostProcessor["Word", list["Keyword"]]):
    gap_handler: GapHandler
    stage_name: str = "gap_handler"

    def process(
        self,
        state: PipelineState["Word", list["Keyword"]],
        *,
        text: str,
        item_count: int,
    ) -> PipelineState["Word", list["Keyword"]]:
        del item_count
        keywords = self.gap_handler.handle(state.value, state.items, text)
        return PipelineState(items=list(state.items), value=keywords)


@dataclass(frozen=True, slots=True)
class _KeywordResultFactory:
    def create(
        self,
        state: PipelineState["Word", list["Keyword"]],
    ) -> "KeywordResult":
        return build_result(state.value, state.items)


def build_pipeline(
    *,
    splitter: WordSplitter,
    marker: WordMarkerStrategy,
    llm: KeywordLLMStrategy,
    parser: KeywordParser,
    gap_handler: GapHandler | None = None,
    html_cleaner: HtmlCleaner | None = None,
    offset_restorer: OffsetRestorer["KeywordResult"] | None = None,
    tracer: Tracer | None = None,
) -> Pipeline[
    "Word",
    "MarkedWords",
    list[tuple[int, int]],
    list["Keyword"],
    "KeywordResult",
]:
    if html_cleaner is not None and offset_restorer is None:
        offset_restorer = MappingOffsetRestorer()

    processors: tuple[PostProcessor["Word", list["Keyword"]], ...]
    if gap_handler is None:
        processors = ()
    else:
        processors = (_KeywordGapHandlerProcessor(gap_handler=gap_handler),)

    return Pipeline(
        splitter=splitter,
        marker=marker,
        query=llm,
        parser=parser,
        state_builder=_KeywordStateBuilder(),
        postprocessors=processors,
        result_factory=_KeywordResultFactory(),
        count_resolver=_resolve_count,
        html_cleaner=html_cleaner,
        offset_restorer=offset_restorer,
        tracer=tracer,
    )


def _resolve_count(marked: object, items: list["Word"]) -> int:
    word_count = getattr(marked, "word_count", None)
    return word_count if isinstance(word_count, int) else len(items)
