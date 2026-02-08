"""Pipeline orchestrator for text splitting."""

from __future__ import annotations

from typing import TYPE_CHECKING, final

from txt_splitt.protocols import (
    Enhancer,
    GapHandler,
    LLMStrategy,
    MarkerStrategy,
    ResponseParser,
    SentenceSplitter,
)
from txt_splitt.tracer import NoOpTracer
from txt_splitt.types import SplitResult

if TYPE_CHECKING:
    from txt_splitt.tracer import Tracer


@final
class Pipeline:
    """Orchestrates the 5-stage text splitting pipeline."""

    def __init__(
        self,
        *,
        splitter: SentenceSplitter,
        marker: MarkerStrategy,
        llm: LLMStrategy,
        parser: ResponseParser,
        gap_handler: GapHandler,
        enhancer: Enhancer | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self._splitter = splitter
        self._marker = marker
        self._llm = llm
        self._parser = parser
        self._gap_handler = gap_handler
        self._enhancer = enhancer
        self._tracer = tracer if tracer is not None else NoOpTracer()

    def run(self, text: str) -> SplitResult:
        """Run the full pipeline on input text.

        Exceptions from any stage propagate directly to the caller.
        """
        with self._tracer.span("pipeline.run", input_length=len(text)):
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

            return SplitResult(sentences=tuple(sentences), groups=tuple(groups))
