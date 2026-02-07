"""Pipeline orchestrator for text splitting."""

from typing import final

from txt_splitt._protocols import (
    GapHandler,
    LLMStrategy,
    MarkerStrategy,
    ResponseParser,
    SentenceSplitter,
)
from txt_splitt._types import SplitResult


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
    ) -> None:
        self._splitter = splitter
        self._marker = marker
        self._llm = llm
        self._parser = parser
        self._gap_handler = gap_handler

    def run(self, text: str) -> SplitResult:
        """Run the full pipeline on input text.

        Exceptions from any stage propagate directly to the caller.
        """
        # Stage 1: Split into sentences
        sentences = self._splitter.split(text)

        # Stage 2: Apply markers
        marked = self._marker.mark(text, sentences)

        # Stage 3: Query LLM
        response = self._llm.query(marked)

        # Stage 4: Parse response
        groups = self._parser.parse(response, marked.sentence_count)

        # Stage 5: Handle gaps
        groups = self._gap_handler.handle(groups, marked.sentence_count)

        return SplitResult(
            sentences=tuple(sentences),
            groups=tuple(groups),
        )
