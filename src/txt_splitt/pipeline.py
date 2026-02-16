"""Pipeline orchestrator for text splitting."""

from __future__ import annotations

from typing import TYPE_CHECKING, final

from txt_splitt.joiners import join_sentences_by_groups
from txt_splitt.protocols import (
    Enhancer,
    GapHandler,
    GroupJoiner,
    HtmlCleaner,
    LLMStrategy,
    MarkerStrategy,
    OffsetRestorer,
    RangeAssigner,
    ResponseParser,
    SentenceSplitter,
    TopicExtractor,
)
from txt_splitt.tracer import NoOpTracer
from txt_splitt.types import SplitResult

if TYPE_CHECKING:
    from txt_splitt.tracer import Tracer


@final
class Pipeline:
    """Orchestrates the text splitting pipeline."""

    def __init__(
        self,
        *,
        splitter: SentenceSplitter,
        marker: MarkerStrategy,
        llm: LLMStrategy | None = None,
        topic_extractor: TopicExtractor | None = None,
        range_assigner: RangeAssigner | None = None,
        parser: ResponseParser,
        gap_handler: GapHandler,
        enhancer: Enhancer | None = None,
        joiner: GroupJoiner | None = None,
        html_cleaner: HtmlCleaner | None = None,
        offset_restorer: OffsetRestorer | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        if (html_cleaner is None) != (offset_restorer is None):
            msg = (
                "html_cleaner and offset_restorer must both be provided or both be None"
            )
            raise ValueError(msg)
        has_single = llm is not None
        has_two_stage = topic_extractor is not None or range_assigner is not None
        if has_single and has_two_stage:
            msg = (
                "Cannot provide both 'llm' and 'topic_extractor'/'range_assigner'; "
                "choose one path"
            )
            raise ValueError(msg)
        if not has_single and not has_two_stage:
            msg = (
                "Must provide either 'llm' or both "
                "'topic_extractor' and 'range_assigner'"
            )
            raise ValueError(msg)
        if has_two_stage and (topic_extractor is None or range_assigner is None):
            msg = (
                "Both 'topic_extractor' and 'range_assigner' must be provided "
                "for two-stage mode"
            )
            raise ValueError(msg)
        if llm is not None:
            _validate_stage_compatibility(llm, parser)
        if range_assigner is not None:
            _validate_stage_compatibility(range_assigner, parser)
        self._splitter = splitter
        self._marker = marker
        self._llm = llm
        self._topic_extractor = topic_extractor
        self._range_assigner = range_assigner
        self._parser = parser
        self._gap_handler = gap_handler
        self._enhancer = enhancer
        self._joiner = joiner
        self._html_cleaner = html_cleaner
        self._offset_restorer = offset_restorer
        self._tracer = tracer if tracer is not None else NoOpTracer()

    def run(self, text: str) -> SplitResult:
        """Run the full pipeline on input text.

        Exceptions from any stage propagate directly to the caller.
        """
        with self._tracer.span("pipeline.run", input_length=len(text)):
            # Stage 0 (optional): Clean HTML tags
            mapping = None
            if self._html_cleaner is not None:
                with self._tracer.span("html_clean") as s:
                    text, mapping = self._html_cleaner.clean(text)
                    s.attributes["clean_length"] = len(text)

            # Stage 1: Split into sentences
            with self._tracer.span("split") as s:
                sentences = self._splitter.split(text)
                s.attributes["sentence_count"] = len(sentences)

            # Stage 2: Apply markers
            with self._tracer.span("mark") as s:
                marked = self._marker.mark(text, sentences)
                s.attributes["tagged_text_length"] = len(marked.tagged_text)

            # Stage 3: Query LLM (single-stage or two-stage)
            if self._llm is not None:
                with self._tracer.span("llm.query") as s:
                    response = self._llm.query(marked)
                    s.attributes["response_length"] = len(response)
            else:
                assert self._topic_extractor is not None
                assert self._range_assigner is not None
                with self._tracer.span("topic_extract") as s:
                    topics = self._topic_extractor.extract(marked)
                    s.attributes["topic_count"] = len(topics)
                with self._tracer.span("range_assign") as s:
                    response = self._range_assigner.assign(marked, topics)
                    s.attributes["response_length"] = len(response)

            # Stage 4: Parse response
            with self._tracer.span("parse") as s:
                groups = self._parser.parse(response, marked.sentence_count)
                s.attributes["group_count"] = len(groups)

            # Stage 5: Handle gaps
            with self._tracer.span("gap_handler") as s:
                groups = self._gap_handler.handle(
                    groups, marked.sentence_count, sentences=sentences
                )
                s.attributes["group_count"] = len(groups)

            # Stage 6 (optional): Enhance boundaries
            if self._enhancer is not None:
                with self._tracer.span("enhance") as s:
                    groups = self._enhancer.enhance(groups, sentences)
                    s.attributes["group_count"] = len(groups)

            # Stage 7 (optional): Join adjacent groups
            if self._joiner is not None:
                with self._tracer.span("join") as s:
                    groups = self._joiner.join(groups, sentences)
                    sentences, groups = join_sentences_by_groups(groups, sentences)
                    s.attributes["sentence_count"] = len(sentences)
                    s.attributes["group_count"] = len(groups)

            result = SplitResult(sentences=tuple(sentences), groups=tuple(groups))

            # Stage 8 (optional): Restore original-text offsets
            if self._offset_restorer is not None and mapping is not None:
                with self._tracer.span("offset_restore") as s:
                    result = self._offset_restorer.restore(result, mapping)
                    s.attributes["sentence_count"] = len(result.sentences)

            return result

    async def run_async(self, text: str) -> SplitResult:
        """Run the pipeline with async range assignment.

        Only works when using two-stage mode with an async range_assigner.
        """
        if self._range_assigner is None:
            msg = "run_async requires two-stage mode with range_assigner"
            raise RuntimeError(msg)
        if not hasattr(self._range_assigner, "assign_async"):
            msg = "range_assigner must support assign_async for async mode"
            raise RuntimeError(msg)

        with self._tracer.span("pipeline.run_async", input_length=len(text)):
            # Stage 0 (optional): Clean HTML tags
            mapping = None
            if self._html_cleaner is not None:
                with self._tracer.span("html_clean") as s:
                    text, mapping = self._html_cleaner.clean(text)
                    s.attributes["clean_length"] = len(text)

            # Stage 1: Split into sentences
            with self._tracer.span("split") as s:
                sentences = self._splitter.split(text)
                s.attributes["sentence_count"] = len(sentences)

            # Stage 2: Apply markers
            with self._tracer.span("mark") as s:
                marked = self._marker.mark(text, sentences)
                s.attributes["tagged_text_length"] = len(marked.tagged_text)

            # Stage 3: Two-stage LLM (async)
            with self._tracer.span("topic_extract") as s:
                topics = self._topic_extractor.extract(marked)  # type: ignore[union-attr]
                s.attributes["topic_count"] = len(topics)
            with self._tracer.span("range_assign_async") as s:
                response = await self._range_assigner.assign_async(marked, topics)  # type: ignore[attr-defined]
                s.attributes["response_length"] = len(response)

            # Stage 4: Parse response
            with self._tracer.span("parse") as s:
                groups = self._parser.parse(response, marked.sentence_count)
                s.attributes["group_count"] = len(groups)

            # Stage 5: Handle gaps
            with self._tracer.span("gap_handler") as s:
                groups = self._gap_handler.handle(
                    groups, marked.sentence_count, sentences=sentences
                )
                s.attributes["group_count"] = len(groups)

            # Stage 6 (optional): Enhance boundaries
            if self._enhancer is not None:
                with self._tracer.span("enhance") as s:
                    groups = self._enhancer.enhance(groups, sentences)
                    s.attributes["group_count"] = len(groups)

            # Stage 7 (optional): Join adjacent groups
            if self._joiner is not None:
                with self._tracer.span("join") as s:
                    groups = self._joiner.join(groups, sentences)
                    sentences, groups = join_sentences_by_groups(groups, sentences)
                    s.attributes["sentence_count"] = len(sentences)
                    s.attributes["group_count"] = len(groups)

            result = SplitResult(sentences=tuple(sentences), groups=tuple(groups))

            # Stage 8 (optional): Restore original-text offsets
            if self._offset_restorer is not None and mapping is not None:
                with self._tracer.span("offset_restore") as s:
                    result = self._offset_restorer.restore(result, mapping)
                    s.attributes["sentence_count"] = len(result.sentences)

            return result


def _validate_stage_compatibility(
    llm: LLMStrategy | RangeAssigner, parser: ResponseParser
) -> None:
    llm_format_obj = getattr(llm, "response_format", "text")
    parser_formats_obj = getattr(
        parser, "supported_response_formats", frozenset({"text"})
    )

    llm_format = llm_format_obj if isinstance(llm_format_obj, str) else "text"
    if isinstance(parser_formats_obj, (set, frozenset)):
        parser_formats = {item for item in parser_formats_obj if isinstance(item, str)}
    else:
        parser_formats = {"text"}

    if llm_format not in parser_formats:
        msg = (
            "Incompatible llm/parser response formats: "
            f"llm outputs {llm_format!r}, parser supports {sorted(parser_formats)!r}"
        )
        raise ValueError(msg)
