"""Generic pipeline orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Generic, Protocol, TypeVar, cast, final

from txt_splitt.protocols import HtmlCleaner, LLMRequest, LLMResponse, OffsetRestorer
from txt_splitt.tracer import NoOpTracer
from txt_splitt.types import OffsetMapping

if TYPE_CHECKING:
    from txt_splitt.tracer import Tracer

TItem = TypeVar("TItem")
TMarked = TypeVar("TMarked")
TParsed = TypeVar("TParsed")
TValue = TypeVar("TValue")
TResult = TypeVar("TResult")
TMarkedOut = TypeVar("TMarkedOut", covariant=True)
TMarkedIn = TypeVar("TMarkedIn", contravariant=True)
TParsedOut = TypeVar("TParsedOut", covariant=True)
TParsedIn = TypeVar("TParsedIn", contravariant=True)
TResultOut = TypeVar("TResultOut", covariant=True)
TStageValue = TypeVar("TStageValue")


class Splitter(Protocol[TItem]):
    """Split raw text into domain items."""

    def split(self, text: str) -> list[TItem]: ...


class Marker(Protocol[TItem, TMarkedOut]):
    """Mark items for query-time addressing."""

    def mark(self, text: str, items: list[TItem]) -> TMarkedOut: ...


class Query(Protocol[TMarkedIn]):
    """Produce a raw model response from marked input."""

    def query(self, marked: TMarkedIn) -> str: ...


class SchedulableQuery(Protocol[TMarkedIn]):
    """Produce ordered LLM request batches for marked input."""

    def plan_query(self, marked: TMarkedIn) -> StageResult[str]: ...


class Parser(Protocol[TParsedOut]):
    """Parse a raw model response into structured data."""

    def parse(self, response: str, item_count: int) -> TParsedOut: ...


@dataclass(slots=True)
class PipelineState(Generic[TItem, TValue]):
    """Mutable pipeline state shared across postprocessors."""

    items: list[TItem]
    value: TValue


class StateBuilder(Protocol[TItem, TParsedIn, TValue]):
    """Build pipeline state from parsed data and split items."""

    def build(
        self,
        parsed: TParsedIn,
        items: list[TItem],
        text: str,
    ) -> PipelineState[TItem, TValue]: ...


class PostProcessor(Protocol[TItem, TValue]):
    """Transform pipeline state after parsing."""

    stage_name: str

    def process(
        self,
        state: PipelineState[TItem, TValue],
        *,
        text: str,
        item_count: int,
    ) -> PipelineState[TItem, TValue]: ...


class SchedulablePostProcessor(Protocol[TItem, TValue]):
    """Transform pipeline state through one or more ordered LLM batches."""

    stage_name: str

    def plan_process(
        self,
        state: PipelineState[TItem, TValue],
        *,
        text: str,
        item_count: int,
    ) -> StageResult[PipelineState[TItem, TValue]]: ...


class ResultFactory(Protocol[TItem, TValue, TResultOut]):
    """Assemble the final result from pipeline state."""

    def create(self, state: PipelineState[TItem, TValue]) -> TResultOut: ...


@dataclass(frozen=True, slots=True)
class _DefaultStateBuilder(Generic[TItem, TParsed]):
    def build(
        self,
        parsed: TParsed,
        items: list[TItem],
        text: str,
    ) -> PipelineState[TItem, TParsed]:
        del text
        return PipelineState(items=list(items), value=parsed)


@dataclass(frozen=True, slots=True)
class CompletedStage(Generic[TStageValue]):
    """Completed stage value."""

    value: TStageValue


@dataclass(frozen=True, slots=True)
class PendingStage(Generic[TStageValue]):
    """Pending stage that must be resumed with ordered responses."""

    requests: tuple[LLMRequest, ...]
    resume: Callable[[list[LLMResponse]], StageResult[TStageValue]]


StageResult = CompletedStage[TStageValue] | PendingStage[TStageValue]


@final
class PipelineSession(Generic[TResult]):
    """Mutable run session for staged pipeline execution."""

    def __init__(
        self,
        *,
        requests: tuple[LLMRequest, ...] | None = None,
        resume: Callable[[list[LLMResponse]], SessionStep[TResult]] | None = None,
        result: TResult | None = None,
    ) -> None:
        self._requests = requests
        self._resume = resume
        self._result = result

    def pending_requests(self) -> tuple[LLMRequest, ...]:
        if self._requests is None:
            return ()
        return self._requests

    def submit_responses(self, responses: list[LLMResponse]) -> None:
        if self._requests is None or self._resume is None:
            msg = "pipeline session is not waiting for responses"
            raise RuntimeError(msg)
        if len(responses) != len(self._requests):
            msg = (
                "response count does not match pending request count: "
                f"expected {len(self._requests)}, got {len(responses)}"
            )
            raise ValueError(msg)
        step = self._resume(list(responses))
        self._apply_step(step)

    def is_complete(self) -> bool:
        return self._result is not None

    def result(self) -> TResult:
        if self._result is None:
            msg = "pipeline session is not complete"
            raise RuntimeError(msg)
        return self._result

    def _apply_step(self, step: SessionStep[TResult]) -> None:
        if isinstance(step, _SessionPending):
            self._requests = step.requests
            self._resume = step.resume
            self._result = None
            return
        self._requests = None
        self._resume = None
        self._result = step.result


@dataclass(frozen=True, slots=True)
class _SessionPending(Generic[TResult]):
    requests: tuple[LLMRequest, ...]
    resume: Callable[[list[LLMResponse]], SessionStep[TResult]]


@dataclass(frozen=True, slots=True)
class _SessionComplete(Generic[TResult]):
    result: TResult


SessionStep = _SessionPending[TResult] | _SessionComplete[TResult]


@final
class Pipeline(Generic[TItem, TMarked, TParsed, TValue, TResult]):
    """Generic pipeline for sentence and keyword flows."""

    def __init__(
        self,
        *,
        splitter: Splitter[TItem],
        marker: Marker[TItem, TMarked],
        query: Query[TMarked] | SchedulableQuery[TMarked],
        parser: Parser[TParsed],
        result_factory: ResultFactory[TItem, TValue, TResult],
        state_builder: StateBuilder[TItem, TParsed, TValue] | None = None,
        postprocessors: tuple[
            PostProcessor[TItem, TValue] | SchedulablePostProcessor[TItem, TValue], ...
        ]
        | None = None,
        count_resolver: Callable[[TMarked, list[TItem]], int] | None = None,
        html_cleaner: HtmlCleaner | None = None,
        offset_restorer: OffsetRestorer[TResult] | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        if (html_cleaner is None) != (offset_restorer is None):
            msg = (
                "html_cleaner and offset_restorer must both be provided or both be None"
            )
            raise ValueError(msg)
        _validate_stage_compatibility(query, parser)
        self._splitter = splitter
        self._marker = marker
        self._query = query
        self._parser = parser
        self._result_factory = result_factory
        self._has_custom_state_builder = state_builder is not None
        self._state_builder: StateBuilder[TItem, TParsed, TValue] = (
            state_builder
            if state_builder is not None
            else cast(
                StateBuilder[TItem, TParsed, TValue],
                _DefaultStateBuilder(),
            )
        )
        self._postprocessors = postprocessors if postprocessors is not None else ()
        self._count_resolver = (
            count_resolver
            if count_resolver is not None
            else lambda marked, items: len(items)
        )
        self._html_cleaner = html_cleaner
        self._offset_restorer = offset_restorer
        self._tracer = tracer if tracer is not None else NoOpTracer()

    def start(self, text: str) -> PipelineSession[TResult]:
        """Prepare a run session and emit the first request batch if needed."""
        with self._tracer.span("pipeline.start", input_length=len(text)):
            step = self._start_step(text)
        session = PipelineSession[TResult]()
        session._apply_step(step)
        return session

    def run(self, text: str) -> TResult:
        """Run the full pipeline when no deferred LLM execution is needed."""
        with self._tracer.span("pipeline.run", input_length=len(text)):
            session = PipelineSession[TResult]()
            session._apply_step(self._start_step(text))
        if session.pending_requests():
            msg = (
                "run() cannot execute deferred batches; "
                "use start() and drive the session"
            )
            raise RuntimeError(msg)
        return session.result()

    async def run_async(self, text: str) -> TResult:
        """Run the full pipeline asynchronously for query stages with query_async()."""
        query_async = getattr(self._query, "query_async", None)
        if query_async is None:
            msg = "run_async requires a query stage with query_async() support"
            raise RuntimeError(msg)
        with self._tracer.span("pipeline.run_async", input_length=len(text)):
            prepared_text, mapping = self._prepare_text(text)
            items, marked, item_count = self._split_and_mark(prepared_text)
            with self._tracer.span("llm.query") as span:
                response = await query_async(marked)
                span.attributes["response_length"] = len(response)
            state = self._parse_and_build(
                response=response,
                items=items,
                prepared_text=prepared_text,
                item_count=item_count,
            )
            for processor in self._postprocessors:
                process = getattr(processor, "process", None)
                if not callable(process):
                    msg = (
                        f"processor {type(processor).__name__!r} does not implement "
                        "process(); run_async() only supports PostProcessor types"
                    )
                    raise RuntimeError(msg)

                sync_processor = cast(PostProcessor[TItem, TValue], processor)
                state = sync_processor.process(
                    state,
                    text=prepared_text,
                    item_count=item_count,
                )
            result = self._result_factory.create(state)
            return self._restore_offsets(result, mapping)

    def _start_step(self, text: str) -> SessionStep[TResult]:
        prepared_text, mapping = self._prepare_text(text)
        items, marked, item_count = self._split_and_mark(prepared_text)
        return self._resolve_query_step(
            query_step=self._plan_query(marked),
            items=items,
            marked=marked,
            prepared_text=prepared_text,
            item_count=item_count,
            mapping=mapping,
        )

    def _prepare_text(self, text: str) -> tuple[str, OffsetMapping | None]:
        mapping: OffsetMapping | None = None
        prepared_text = text
        if self._html_cleaner is not None:
            with self._tracer.span("html_clean") as span:
                prepared_text, mapping = self._html_cleaner.clean(text)
                span.attributes["clean_length"] = len(prepared_text)
        return prepared_text, mapping

    def _split_and_mark(self, text: str) -> tuple[list[TItem], TMarked, int]:
        with self._tracer.span("split") as span:
            items = self._splitter.split(text)
            span.attributes["item_count"] = len(items)
            span.attributes["sentence_count"] = len(items)
            span.attributes["word_count"] = len(items)

        with self._tracer.span("mark") as span:
            marked = self._marker.mark(text, items)
            tagged_text = getattr(marked, "tagged_text", None)
            if isinstance(tagged_text, str):
                span.attributes["tagged_text_length"] = len(tagged_text)

        item_count = self._count_resolver(marked, items)
        return items, marked, item_count

    def _plan_query(self, marked: TMarked) -> StageResult[str]:
        plan_query = getattr(self._query, "plan_query", None)
        if callable(plan_query):
            return cast(StageResult[str], plan_query(marked))
        query = cast(Query[TMarked], self._query)
        with self._tracer.span("llm.query") as span:
            response = query.query(marked)
            span.attributes["response_length"] = len(response)
            return CompletedStage(response)

    def _resolve_query_step(
        self,
        *,
        query_step: StageResult[str],
        items: list[TItem],
        marked: TMarked,
        prepared_text: str,
        item_count: int,
        mapping: OffsetMapping | None,
    ) -> SessionStep[TResult]:
        if isinstance(query_step, PendingStage):
            return _SessionPending(
                requests=query_step.requests,
                resume=lambda responses: self._resume_query(
                    query_step=query_step,
                    responses=responses,
                    items=items,
                    marked=marked,
                    prepared_text=prepared_text,
                    item_count=item_count,
                    mapping=mapping,
                ),
            )
        return self._complete_after_response(
            response=query_step.value,
            items=items,
            marked=marked,
            prepared_text=prepared_text,
            item_count=item_count,
            mapping=mapping,
        )

    def _resume_query(
        self,
        *,
        query_step: PendingStage[str],
        responses: list[LLMResponse],
        items: list[TItem],
        marked: TMarked,
        prepared_text: str,
        item_count: int,
        mapping: OffsetMapping | None,
    ) -> SessionStep[TResult]:
        with self._tracer.span("llm.query") as span:
            next_step = query_step.resume(responses)
            if isinstance(next_step, PendingStage):
                span.attributes["batch_size"] = len(query_step.requests)
                return self._resolve_query_step(
                    query_step=next_step,
                    items=items,
                    marked=marked,
                    prepared_text=prepared_text,
                    item_count=item_count,
                    mapping=mapping,
                )
            span.attributes["response_length"] = len(next_step.value)
            return self._complete_after_response(
                response=next_step.value,
                items=items,
                marked=marked,
                prepared_text=prepared_text,
                item_count=item_count,
                mapping=mapping,
            )

    def _complete_after_response(
        self,
        *,
        response: str,
        items: list[TItem],
        marked: TMarked,
        prepared_text: str,
        item_count: int,
        mapping: OffsetMapping | None,
    ) -> SessionStep[TResult]:
        state = self._parse_and_build(
            response=response,
            items=items,
            prepared_text=prepared_text,
            item_count=item_count,
        )
        return self._resolve_postprocessors(
            state=state,
            prepared_text=prepared_text,
            item_count=item_count,
            mapping=mapping,
            index=0,
        )

    def _parse_and_build(
        self,
        *,
        response: str,
        items: list[TItem],
        prepared_text: str,
        item_count: int,
    ) -> PipelineState[TItem, TValue]:
        with self._tracer.span("parse") as span:
            parsed = self._parser.parse(response, item_count)
            _maybe_record_len(span.attributes, "parsed_count", parsed)
            _maybe_record_len(span.attributes, "group_count", parsed)

        if self._has_custom_state_builder:
            with self._tracer.span("build") as span:
                state = self._state_builder.build(parsed, items, prepared_text)
                span.attributes["item_count"] = len(state.items)
                _maybe_record_len(span.attributes, "value_count", state.value)
                return state
        return self._state_builder.build(parsed, items, prepared_text)

    def _resolve_postprocessors(
        self,
        *,
        state: PipelineState[TItem, TValue],
        prepared_text: str,
        item_count: int,
        mapping: OffsetMapping | None,
        index: int,
    ) -> SessionStep[TResult]:
        if index >= len(self._postprocessors):
            result = self._result_factory.create(state)
            return _SessionComplete(self._restore_offsets(result, mapping))

        processor = self._postprocessors[index]
        stage_name = getattr(processor, "stage_name", "postprocess")
        plan_process = getattr(processor, "plan_process", None)

        if callable(plan_process):
            pending_processor: PendingStage[PipelineState[TItem, TValue]] | None = None
            completed_state: PipelineState[TItem, TValue] | None = None
            with self._tracer.span(
                stage_name,
                processor=type(processor).__name__,
            ) as span:
                processor_step = cast(
                    StageResult[PipelineState[TItem, TValue]],
                    plan_process(
                        state,
                        text=prepared_text,
                        item_count=item_count,
                    ),
                )
                if isinstance(processor_step, CompletedStage):
                    completed_state = processor_step.value
                    span.attributes["item_count"] = len(completed_state.items)
                    _maybe_record_len(
                        span.attributes,
                        "value_count",
                        completed_state.value,
                    )
                else:
                    pending_processor = processor_step
                    span.attributes["batch_size"] = len(processor_step.requests)
            if completed_state is not None:
                return self._resolve_postprocessors(
                    state=completed_state,
                    prepared_text=prepared_text,
                    item_count=item_count,
                    mapping=mapping,
                    index=index + 1,
                )
            assert pending_processor is not None
            return _SessionPending(
                requests=pending_processor.requests,
                resume=lambda responses: self._resume_postprocessor(
                    processor_step=pending_processor,
                    prepared_text=prepared_text,
                    item_count=item_count,
                    mapping=mapping,
                    index=index,
                    responses=responses,
                ),
            )

        sync_processor = cast(PostProcessor[TItem, TValue], processor)
        with self._tracer.span(stage_name, processor=type(processor).__name__) as span:
            next_state = sync_processor.process(
                state,
                text=prepared_text,
                item_count=item_count,
            )
            span.attributes["item_count"] = len(next_state.items)
            _maybe_record_len(span.attributes, "value_count", next_state.value)
        return self._resolve_postprocessors(
            state=next_state,
            prepared_text=prepared_text,
            item_count=item_count,
            mapping=mapping,
            index=index + 1,
        )

    def _resume_postprocessor(
        self,
        *,
        processor_step: PendingStage[PipelineState[TItem, TValue]],
        prepared_text: str,
        item_count: int,
        mapping: OffsetMapping | None,
        index: int,
        responses: list[LLMResponse],
    ) -> SessionStep[TResult]:
        processor_result = processor_step.resume(responses)
        if isinstance(processor_result, PendingStage):
            return _SessionPending(
                requests=processor_result.requests,
                resume=lambda next_responses: self._resume_postprocessor(
                    processor_step=processor_result,
                    prepared_text=prepared_text,
                    item_count=item_count,
                    mapping=mapping,
                    index=index,
                    responses=next_responses,
                ),
            )
        return self._resolve_postprocessors(
            state=processor_result.value,
            prepared_text=prepared_text,
            item_count=item_count,
            mapping=mapping,
            index=index + 1,
        )

    def _restore_offsets(
        self,
        result: TResult,
        mapping: OffsetMapping | None,
    ) -> TResult:
        if self._offset_restorer is None or mapping is None:
            return result
        with self._tracer.span("offset_restore") as span:
            restored = self._offset_restorer.restore(result, mapping)
            sentence_like = getattr(restored, "sentences", None)
            word_like = getattr(restored, "words", None)
            if sentence_like is not None:
                span.attributes["item_count"] = len(sentence_like)
            elif word_like is not None:
                span.attributes["item_count"] = len(word_like)
            return restored


def _maybe_record_len(attributes: dict[str, object], key: str, value: object) -> None:
    try:
        attributes[key] = len(value)  # type: ignore[arg-type]
    except TypeError:
        return


def _validate_stage_compatibility(query: object, parser: object) -> None:
    query_format_obj = getattr(query, "response_format", "text")
    parser_formats_obj = getattr(
        parser,
        "supported_response_formats",
        frozenset({"text"}),
    )

    query_format = query_format_obj if isinstance(query_format_obj, str) else "text"
    if isinstance(parser_formats_obj, (set, frozenset)):
        parser_formats = {item for item in parser_formats_obj if isinstance(item, str)}
    else:
        parser_formats = {"text"}

    if query_format not in parser_formats:
        msg = (
            "Incompatible llm/parser response formats: "
            f"llm outputs {query_format!r}, parser supports {sorted(parser_formats)!r}"
        )
        raise ValueError(msg)
