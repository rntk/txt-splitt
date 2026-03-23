"""Generic pipeline orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Generic, Protocol, TypeVar, cast, final

from txt_splitt.protocols import HtmlCleaner, OffsetRestorer
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


class Splitter(Protocol[TItem]):
    """Split raw text into domain items."""

    def split(self, text: str) -> list[TItem]: ...


class Marker(Protocol[TItem, TMarkedOut]):
    """Mark items for query-time addressing."""

    def mark(self, text: str, items: list[TItem]) -> TMarkedOut: ...


class Query(Protocol[TMarkedIn]):
    """Produce a raw model response from marked input."""

    def query(self, marked: TMarkedIn) -> str: ...


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


@final
class Pipeline(Generic[TItem, TMarked, TParsed, TValue, TResult]):
    """Generic pipeline for sentence and keyword flows."""

    def __init__(
        self,
        *,
        splitter: Splitter[TItem],
        marker: Marker[TItem, TMarked],
        query: Query[TMarked],
        parser: Parser[TParsed],
        result_factory: ResultFactory[TItem, TValue, TResult],
        state_builder: StateBuilder[TItem, TParsed, TValue] | None = None,
        postprocessors: tuple[PostProcessor[TItem, TValue], ...] | None = None,
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

    def run(self, text: str) -> TResult:
        """Run the full pipeline synchronously."""
        with self._tracer.span("pipeline.run", input_length=len(text)):
            prepared_text, mapping = self._prepare_text(text)
            items, marked, item_count = self._split_and_mark(prepared_text)

            with self._tracer.span("llm.query") as span:
                response = self._query.query(marked)
                span.attributes["response_length"] = len(response)

            state = self._parse_build_and_process(
                response=response,
                items=items,
                marked=marked,
                prepared_text=prepared_text,
                item_count=item_count,
            )

            result = self._result_factory.create(state)
            return self._restore_offsets(result, mapping)

    async def run_async(self, text: str) -> TResult:
        """Run the pipeline asynchronously when the query stage supports it."""
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

            state = self._parse_build_and_process(
                response=response,
                items=items,
                marked=marked,
                prepared_text=prepared_text,
                item_count=item_count,
            )

            result = self._result_factory.create(state)
            return self._restore_offsets(result, mapping)

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

    def _parse_build_and_process(
        self,
        *,
        response: str,
        items: list[TItem],
        marked: TMarked,
        prepared_text: str,
        item_count: int,
    ) -> PipelineState[TItem, TValue]:
        del marked
        with self._tracer.span("parse") as span:
            parsed = self._parser.parse(response, item_count)
            _maybe_record_len(span.attributes, "parsed_count", parsed)
            _maybe_record_len(span.attributes, "group_count", parsed)

        if self._has_custom_state_builder:
            with self._tracer.span("build") as span:
                state = self._state_builder.build(parsed, items, prepared_text)
                span.attributes["item_count"] = len(state.items)
                _maybe_record_len(span.attributes, "value_count", state.value)
        else:
            state = self._state_builder.build(parsed, items, prepared_text)

        for processor in self._postprocessors:
            stage_name = getattr(processor, "stage_name", "postprocess")
            with self._tracer.span(
                stage_name,
                processor=type(processor).__name__,
            ) as span:
                state = processor.process(
                    state,
                    text=prepared_text,
                    item_count=item_count,
                )
                span.attributes["item_count"] = len(state.items)
                _maybe_record_len(span.attributes, "value_count", state.value)

        return state

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
