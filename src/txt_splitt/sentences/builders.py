"""Sentence pipeline builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence, cast

from txt_splitt.pipeline import (
    CompletedStage,
    PendingStage,
    Pipeline,
    PipelineSession,
    PipelineState,
    PostProcessor,
    StageResult,
)
from txt_splitt.sentences.joiners import join_sentences_by_groups
from txt_splitt.sentences.types import MarkedText, Sentence, SentenceGroup, SplitResult
from txt_splitt.tracer import NoOpTracer

if TYPE_CHECKING:
    from txt_splitt.protocols import HtmlCleaner, OffsetRestorer
    from txt_splitt.sentences.protocols import (
        Enhancer,
        GapHandler,
        GroupJoiner,
        MarkerStrategy,
        RangeAssigner,
        ResponseParser,
        SchedulableLLMStrategy,
        SentenceSplitter,
        TopicExtractor,
    )
    from txt_splitt.tracer import Tracer


@dataclass(frozen=True, slots=True)
class _SentenceResultFactory:
    def create(
        self,
        state: PipelineState[Sentence, list[SentenceGroup]],
    ) -> SplitResult:
        return SplitResult(
            sentences=tuple(state.items),
            groups=tuple(state.value),
        )


def _wrap_groups_result(
    stage_result: StageResult[list[SentenceGroup]],
    items: list[Sentence],
) -> StageResult[PipelineState[Sentence, list[SentenceGroup]]]:
    """Wrap a StageResult of groups into a StageResult of PipelineState."""
    if isinstance(stage_result, CompletedStage):
        return CompletedStage(
            PipelineState(items=list(items), value=stage_result.value)
        )
    return PendingStage(
        requests=stage_result.requests,
        resume=lambda responses: _wrap_groups_result(
            stage_result.resume(responses),
            items,
        ),
    )


@dataclass(frozen=True, slots=True)
class _GapHandlerProcessor(PostProcessor[Sentence, list[SentenceGroup]]):
    gap_handler: GapHandler
    stage_name: str = "gap_handler"

    def process(
        self,
        state: PipelineState[Sentence, list[SentenceGroup]],
        *,
        text: str,
        item_count: int,
    ) -> PipelineState[Sentence, list[SentenceGroup]]:
        del text
        groups = self.gap_handler.handle(
            state.value,
            item_count,
            sentences=state.items,
        )
        return PipelineState(items=list(state.items), value=groups)

    def plan_process(
        self,
        state: PipelineState[Sentence, list[SentenceGroup]],
        *,
        text: str,
        item_count: int,
    ) -> StageResult[PipelineState[Sentence, list[SentenceGroup]]]:
        del text
        plan_handle = getattr(self.gap_handler, "plan_handle", None)
        if not callable(plan_handle):
            return CompletedStage(self.process(state, text="", item_count=item_count))
        stage_result = cast(
            StageResult[list[SentenceGroup]],
            plan_handle(
                state.value,
                item_count,
                sentences=state.items,
            ),
        )
        if isinstance(stage_result, CompletedStage):
            return CompletedStage(
                PipelineState(items=list(state.items), value=stage_result.value)
            )
        return PendingStage(
            requests=stage_result.requests,
            resume=lambda responses: _wrap_groups_result(
                stage_result.resume(responses),
                state.items,
            ),
        )


@dataclass(frozen=True, slots=True)
class _EnhancerProcessor(PostProcessor[Sentence, list[SentenceGroup]]):
    enhancer: Enhancer
    stage_name: str = "enhance"

    def process(
        self,
        state: PipelineState[Sentence, list[SentenceGroup]],
        *,
        text: str,
        item_count: int,
    ) -> PipelineState[Sentence, list[SentenceGroup]]:
        del text, item_count
        groups = self.enhancer.enhance(state.value, state.items)
        return PipelineState(items=list(state.items), value=groups)

    def plan_process(
        self,
        state: PipelineState[Sentence, list[SentenceGroup]],
        *,
        text: str,
        item_count: int,
    ) -> StageResult[PipelineState[Sentence, list[SentenceGroup]]]:
        del text, item_count
        plan_enhance = getattr(self.enhancer, "plan_enhance", None)
        if not callable(plan_enhance):
            return CompletedStage(self.process(state, text="", item_count=0))
        stage_result = cast(
            StageResult[list[SentenceGroup]],
            plan_enhance(state.value, state.items),
        )
        if isinstance(stage_result, CompletedStage):
            return CompletedStage(
                PipelineState(items=list(state.items), value=stage_result.value)
            )
        return PendingStage(
            requests=stage_result.requests,
            resume=lambda responses: _wrap_groups_result(
                stage_result.resume(responses),
                state.items,
            ),
        )


@dataclass(frozen=True, slots=True)
class _JoinerProcessor(PostProcessor[Sentence, list[SentenceGroup]]):
    joiner: GroupJoiner
    stage_name: str = "join"

    def process(
        self,
        state: PipelineState[Sentence, list[SentenceGroup]],
        *,
        text: str,
        item_count: int,
    ) -> PipelineState[Sentence, list[SentenceGroup]]:
        del text, item_count
        groups = self.joiner.join(state.value, state.items)
        sentences, groups = join_sentences_by_groups(groups, state.items)
        return PipelineState(items=sentences, value=groups)


class _TwoStageQuery:
    def __init__(
        self,
        *,
        topic_extractor: TopicExtractor,
        range_assigner: RangeAssigner,
        tracer: Tracer | NoOpTracer | None = None,
    ) -> None:
        self._topic_extractor = topic_extractor
        self._range_assigner = range_assigner
        self._tracer = tracer if tracer is not None else NoOpTracer()

    @property
    def response_format(self) -> str:
        response_format = getattr(self._range_assigner, "response_format", "text")
        return response_format if isinstance(response_format, str) else "text"

    def plan_query(self, marked_text: MarkedText) -> StageResult[str]:
        with self._tracer.span("topic_extract") as span:
            topics = self._topic_extractor.extract(marked_text)
            span.attributes["topic_count"] = len(topics)
        with self._tracer.span("range_assign") as span:
            response = self._range_assigner.assign(marked_text, topics)
            span.attributes["response_length"] = len(response)
        return CompletedStage(response)


@dataclass(frozen=True, slots=True)
class SentencePipeline:
    """Sentence-specific wrapper around the generic pipeline."""

    _pipeline: Pipeline[
        Sentence,
        MarkedText,
        list[SentenceGroup],
        list[SentenceGroup],
        SplitResult,
    ]
    _llm: SchedulableLLMStrategy | None
    _topic_extractor: TopicExtractor | None
    _range_assigner: RangeAssigner | None
    _gap_handler: GapHandler
    _enhancers: tuple[Enhancer, ...]
    _tracer: Tracer | NoOpTracer

    def start(self, text: str) -> PipelineSession[SplitResult]:
        return self._pipeline.start(text)

    def run(self, text: str) -> SplitResult:
        return self._pipeline.run(text)


def build_pipeline(
    *,
    splitter: SentenceSplitter,
    marker: MarkerStrategy,
    parser: ResponseParser,
    gap_handler: GapHandler,
    llm: SchedulableLLMStrategy | None = None,
    topic_extractor: TopicExtractor | None = None,
    range_assigner: RangeAssigner | None = None,
    enhancers: Sequence[Enhancer] | None = None,
    joiner: GroupJoiner | None = None,
    html_cleaner: HtmlCleaner | None = None,
    offset_restorer: OffsetRestorer[SplitResult] | None = None,
    tracer: Tracer | None = None,
) -> SentencePipeline:
    has_single = llm is not None
    has_two_stage = topic_extractor is not None or range_assigner is not None
    if has_single and has_two_stage:
        msg = (
            "Cannot provide both 'llm' and 'topic_extractor'/'range_assigner'; "
            "choose one path"
        )
        raise ValueError(msg)
    if not has_single and not has_two_stage:
        msg = "Must provide either 'llm' or both 'topic_extractor' and 'range_assigner'"
        raise ValueError(msg)
    if has_two_stage and (topic_extractor is None or range_assigner is None):
        msg = (
            "Both 'topic_extractor' and 'range_assigner' must be provided "
            "for two-stage mode"
        )
        raise ValueError(msg)

    resolved_tracer = tracer if tracer is not None else NoOpTracer()

    query = (
        llm
        if llm is not None
        else _TwoStageQuery(
            topic_extractor=topic_extractor,  # type: ignore[arg-type]
            range_assigner=range_assigner,  # type: ignore[arg-type]
            tracer=resolved_tracer,
        )
    )

    processors: list[PostProcessor[Sentence, list[SentenceGroup]]] = [
        _GapHandlerProcessor(gap_handler=gap_handler)
    ]
    for enhancer in enhancers or ():
        processors.append(_EnhancerProcessor(enhancer=enhancer))
    if joiner is not None:
        processors.append(_JoinerProcessor(joiner=joiner))

    pipeline: Pipeline[
        Sentence,
        MarkedText,
        list[SentenceGroup],
        list[SentenceGroup],
        SplitResult,
    ] = Pipeline(
        splitter=splitter,
        marker=marker,
        query=query,
        parser=parser,
        result_factory=_SentenceResultFactory(),
        postprocessors=tuple(processors),
        count_resolver=_resolve_count,
        html_cleaner=html_cleaner,
        offset_restorer=offset_restorer,
        tracer=tracer,
    )
    return SentencePipeline(
        _pipeline=pipeline,
        _llm=llm,
        _topic_extractor=topic_extractor,
        _range_assigner=range_assigner,
        _gap_handler=gap_handler,
        _enhancers=tuple(enhancers or ()),
        _tracer=resolved_tracer,
    )


def _resolve_count(marked: object, items: list[Sentence]) -> int:
    sentence_count = getattr(marked, "sentence_count", None)
    return sentence_count if isinstance(sentence_count, int) else len(items)
