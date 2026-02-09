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
    ResponseParser,
    SentenceSplitter,
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
        llm: LLMStrategy,
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
        self._splitter = splitter
        self._marker = marker
        self._llm = llm
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

            # Stage 3: Query LLM
            with self._tracer.span("llm.query") as s:
                response = self._llm.query(marked)
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
