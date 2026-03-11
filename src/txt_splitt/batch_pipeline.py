"""Two-stage pipeline for externally orchestrated LLM batching."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Sequence, final

from txt_splitt.joiners import join_sentences_by_groups
from txt_splitt.protocols import (
    Enhancer,
    GapHandler,
    GroupJoiner,
    HtmlCleaner,
    MarkedTextChunker,
    MarkerStrategy,
    OffsetRestorer,
    ResponseParser,
    SentenceSplitter,
)
from txt_splitt.tracer import NoOpTracer
from txt_splitt.types import (
    MarkedText,
    PreparedChunk,
    PreparedDocument,
    SplitResult,
)

if TYPE_CHECKING:
    from txt_splitt.tracer import Tracer

_MARKER_PATTERN = re.compile(r"\{(\d+)\}")


@final
class BatchPipeline:
    """Prepare/finalize pipeline for one document.

    The caller is responsible for sending ``PreparedDocument.chunks`` to an LLM,
    collecting the responses, and supplying one merged response string to
    :meth:`finalize`.
    """

    def __init__(
        self,
        *,
        splitter: SentenceSplitter,
        marker: MarkerStrategy,
        parser: ResponseParser,
        gap_handler: GapHandler,
        chunker: MarkedTextChunker | None = None,
        enhancers: Sequence[Enhancer] | None = None,
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
        self._parser = parser
        self._gap_handler = gap_handler
        self._chunker = chunker
        self._enhancers = tuple(enhancers) if enhancers else ()
        self._joiner = joiner
        self._html_cleaner = html_cleaner
        self._offset_restorer = offset_restorer
        self._tracer = tracer if tracer is not None else NoOpTracer()

    def prepare(self, text: str) -> PreparedDocument:
        """Run all pre-LLM stages and return chunk-ready document state."""
        with self._tracer.span("batch_pipeline.prepare", input_length=len(text)):
            prepared_text = text
            mapping = None
            if self._html_cleaner is not None:
                with self._tracer.span("html_clean") as s:
                    prepared_text, mapping = self._html_cleaner.clean(text)
                    s.attributes["clean_length"] = len(prepared_text)

            with self._tracer.span("split") as s:
                sentences = self._splitter.split(prepared_text)
                s.attributes["sentence_count"] = len(sentences)

            with self._tracer.span("mark") as s:
                marked = self._marker.mark(prepared_text, sentences)
                s.attributes["tagged_text_length"] = len(marked.tagged_text)

            with self._tracer.span("chunk") as s:
                marked_chunks = (
                    self._chunker.chunk(marked)
                    if self._chunker is not None
                    else [marked]
                )
                chunks = tuple(
                    _build_prepared_chunk(chunk_id=idx, chunk=chunk)
                    for idx, chunk in enumerate(marked_chunks)
                )
                s.attributes["chunk_count"] = len(chunks)

            return PreparedDocument(
                original_text=text,
                prepared_text=prepared_text,
                sentences=tuple(sentences),
                marked_text=marked,
                chunks=chunks,
                offset_mapping=mapping,
            )

    def finalize(self, prepared: PreparedDocument, llm_response: str) -> SplitResult:
        """Run parser and post-LLM stages for a prepared document."""
        with self._tracer.span(
            "batch_pipeline.finalize",
            sentence_count=len(prepared.sentences),
            response_length=len(llm_response),
        ):
            with self._tracer.span("parse") as s:
                groups = self._parser.parse(llm_response, len(prepared.sentences))
                s.attributes["group_count"] = len(groups)

            sentences = list(prepared.sentences)

            with self._tracer.span("gap_handler") as s:
                groups = self._gap_handler.handle(
                    groups, len(sentences), sentences=sentences
                )
                s.attributes["group_count"] = len(groups)

            for enhancer in self._enhancers:
                with self._tracer.span(
                    "enhance", enhancer=type(enhancer).__name__
                ) as s:
                    groups = enhancer.enhance(groups, sentences)
                    s.attributes["group_count"] = len(groups)

            if self._joiner is not None:
                with self._tracer.span("join") as s:
                    groups = self._joiner.join(groups, sentences)
                    sentences, groups = join_sentences_by_groups(groups, sentences)
                    s.attributes["sentence_count"] = len(sentences)
                    s.attributes["group_count"] = len(groups)

            result = SplitResult(sentences=tuple(sentences), groups=tuple(groups))

            if (
                self._offset_restorer is not None
                and prepared.offset_mapping is not None
            ):
                with self._tracer.span("offset_restore") as s:
                    result = self._offset_restorer.restore(
                        result, prepared.offset_mapping
                    )
                    s.attributes["sentence_count"] = len(result.sentences)

            return result


def _build_prepared_chunk(*, chunk_id: int, chunk: MarkedText) -> PreparedChunk:
    """Build PreparedChunk metadata from a MarkedText-like object."""
    tagged_text = chunk.tagged_text
    sentence_count = chunk.sentence_count
    marker_values = [
        int(match.group(1)) for match in _MARKER_PATTERN.finditer(tagged_text)
    ]
    marker_start = marker_values[0] if marker_values else None
    marker_end = marker_values[-1] if marker_values else None
    return PreparedChunk(
        chunk_id=chunk_id,
        tagged_text=tagged_text,
        sentence_count=sentence_count,
        marker_start=marker_start,
        marker_end=marker_end,
    )
