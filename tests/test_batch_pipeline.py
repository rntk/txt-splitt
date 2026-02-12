"""Tests for the two-stage BatchPipeline."""

import pytest

from txt_splitt.batch_pipeline import BatchPipeline
from txt_splitt.pipeline import Pipeline
from txt_splitt.types import (
    MarkedText,
    OffsetMapping,
    OffsetSegment,
    PreparedDocument,
    Sentence,
    SentenceGroup,
    SentenceRange,
    SplitResult,
)


class StubSplitter:
    def __init__(self, sentences: list[Sentence]) -> None:
        self._sentences = sentences

    def split(self, text: str) -> list[Sentence]:
        return self._sentences


class StubMarker:
    def __init__(self, marked_text: MarkedText) -> None:
        self._marked_text = marked_text

    def mark(self, text: str, sentences: list[Sentence]) -> MarkedText:
        return self._marked_text


class StubLLM:
    def __init__(self, response: str) -> None:
        self._response = response

    def query(self, marked_text: MarkedText) -> str:
        return self._response


class StubParser:
    def __init__(self, groups: list[SentenceGroup]) -> None:
        self._groups = groups

    def parse(self, response: str, sentence_count: int) -> list[SentenceGroup]:
        return self._groups


class RecordingParser:
    def __init__(self, groups: list[SentenceGroup]) -> None:
        self._groups = groups
        self.seen_response: str | None = None
        self.seen_sentence_count: int | None = None

    def parse(self, response: str, sentence_count: int) -> list[SentenceGroup]:
        self.seen_response = response
        self.seen_sentence_count = sentence_count
        return self._groups


class StubGapHandler:
    def __init__(self, groups: list[SentenceGroup]) -> None:
        self._groups = groups

    def handle(
        self,
        groups: list[SentenceGroup],
        sentence_count: int,
        sentences: list[Sentence] | None = None,
    ) -> list[SentenceGroup]:
        return self._groups


class RecordingGapHandler:
    def __init__(self, groups: list[SentenceGroup]) -> None:
        self._groups = groups
        self.seen_sentence_count: int | None = None
        self.seen_sentences: list[Sentence] | None = None

    def handle(
        self,
        groups: list[SentenceGroup],
        sentence_count: int,
        sentences: list[Sentence] | None = None,
    ) -> list[SentenceGroup]:
        self.seen_sentence_count = sentence_count
        self.seen_sentences = sentences
        return self._groups


class StubChunker:
    def __init__(self, chunks: list[MarkedText]) -> None:
        self._chunks = chunks
        self.seen_marked_text: MarkedText | None = None

    def chunk(self, marked_text: MarkedText) -> list[MarkedText]:
        self.seen_marked_text = marked_text
        return self._chunks


class StubHtmlCleaner:
    def __init__(self, clean_text: str, mapping: OffsetMapping) -> None:
        self._clean_text = clean_text
        self._mapping = mapping

    def clean(self, text: str) -> tuple[str, OffsetMapping]:
        return self._clean_text, self._mapping


class RecordingOffsetRestorer:
    def __init__(self, restored: SplitResult) -> None:
        self._restored = restored
        self.seen_result: SplitResult | None = None
        self.seen_mapping: OffsetMapping | None = None

    def restore(self, result: SplitResult, mapping: OffsetMapping) -> SplitResult:
        self.seen_result = result
        self.seen_mapping = mapping
        return self._restored


def _make_sentences(n: int) -> list[Sentence]:
    return [
        Sentence(index=i, start=i * 10, end=i * 10 + 5, text=f"Sent {i}.")
        for i in range(n)
    ]


def _make_groups() -> list[SentenceGroup]:
    return [
        SentenceGroup(
            label=("Technology", "AI"),
            ranges=(SentenceRange(start=0, end=2),),
        )
    ]


def test_prepare_without_chunker_produces_single_chunk() -> None:
    sentences = _make_sentences(2)
    marked = MarkedText(tagged_text="{0} A\n{1} B", sentence_count=2)
    pipeline = BatchPipeline(
        splitter=StubSplitter(sentences),
        marker=StubMarker(marked),
        parser=StubParser(_make_groups()),
        gap_handler=StubGapHandler(_make_groups()),
    )

    prepared = pipeline.prepare("Text")

    assert isinstance(prepared, PreparedDocument)
    assert prepared.original_text == "Text"
    assert prepared.prepared_text == "Text"
    assert prepared.marked_text == marked
    assert len(prepared.chunks) == 1
    assert prepared.chunks[0].chunk_id == 0
    assert prepared.chunks[0].marker_start == 0
    assert prepared.chunks[0].marker_end == 1


def test_prepare_with_chunker_builds_chunk_metadata() -> None:
    sentences = _make_sentences(5)
    marked = MarkedText(
        tagged_text="{0} A\n{1} B\n{2} C\n{3} D\n{4} E",
        sentence_count=5,
    )
    chunker = StubChunker(
        [
            MarkedText(tagged_text="{0} A\n{1} B\n{2} C", sentence_count=3),
            MarkedText(tagged_text="{2} C\n{3} D\n{4} E", sentence_count=3),
        ]
    )
    pipeline = BatchPipeline(
        splitter=StubSplitter(sentences),
        marker=StubMarker(marked),
        parser=StubParser(_make_groups()),
        gap_handler=StubGapHandler(_make_groups()),
        chunker=chunker,
    )

    prepared = pipeline.prepare("Text")

    assert chunker.seen_marked_text == marked
    assert len(prepared.chunks) == 2
    assert prepared.chunks[0].marker_start == 0
    assert prepared.chunks[0].marker_end == 2
    assert prepared.chunks[1].marker_start == 2
    assert prepared.chunks[1].marker_end == 4


def test_finalize_starts_at_parser_and_uses_prepared_sentences() -> None:
    sentences = _make_sentences(3)
    groups = _make_groups()
    parser = RecordingParser(groups)
    gap_handler = RecordingGapHandler(groups)
    pipeline = BatchPipeline(
        splitter=StubSplitter(sentences),
        marker=StubMarker(MarkedText(tagged_text="{0} A", sentence_count=1)),
        parser=parser,
        gap_handler=gap_handler,
    )
    prepared = PreparedDocument(
        original_text="Original",
        prepared_text="Original",
        sentences=tuple(sentences),
        marked_text=MarkedText(tagged_text="{0} A\n{1} B\n{2} C", sentence_count=3),
        chunks=(),
        offset_mapping=None,
    )

    result = pipeline.finalize(prepared, "Technology>AI: 0-2")

    assert parser.seen_response == "Technology>AI: 0-2"
    assert parser.seen_sentence_count == 3
    assert gap_handler.seen_sentence_count == 3
    assert gap_handler.seen_sentences == sentences
    assert len(result.groups) == 1


def test_prepare_finalize_matches_pipeline_behavior() -> None:
    sentences = _make_sentences(3)
    groups = _make_groups()
    marked = MarkedText(tagged_text="{0} A\n{1} B\n{2} C", sentence_count=3)
    llm_response = "Technology>AI: 0-2"

    single = Pipeline(
        splitter=StubSplitter(sentences),
        marker=StubMarker(marked),
        llm=StubLLM(llm_response),
        parser=StubParser(groups),
        gap_handler=StubGapHandler(groups),
    )
    batch = BatchPipeline(
        splitter=StubSplitter(sentences),
        marker=StubMarker(marked),
        parser=StubParser(groups),
        gap_handler=StubGapHandler(groups),
    )

    single_result = single.run("Text")
    prepared = batch.prepare("Text")
    batch_result = batch.finalize(prepared, llm_response)

    assert single_result == batch_result


def test_finalize_restores_offsets_when_mapping_present() -> None:
    mapping = OffsetMapping(
        segments=(OffsetSegment(clean_offset=0, original_offset=0, length=4),),
        original_length=4,
        clean_length=4,
    )
    sentences = _make_sentences(1)
    groups = [
        SentenceGroup(label=("A",), ranges=(SentenceRange(start=0, end=0),)),
    ]
    restored = SplitResult(sentences=tuple(sentences), groups=tuple(groups))
    restorer = RecordingOffsetRestorer(restored)
    pipeline = BatchPipeline(
        splitter=StubSplitter(sentences),
        marker=StubMarker(MarkedText(tagged_text="{0} A", sentence_count=1)),
        parser=StubParser(groups),
        gap_handler=StubGapHandler(groups),
        html_cleaner=StubHtmlCleaner("Text", mapping),
        offset_restorer=restorer,
    )

    prepared = pipeline.prepare("<p>Text</p>")
    result = pipeline.finalize(prepared, "A: 0")

    assert restorer.seen_mapping == mapping
    assert result == restored


def test_init_requires_html_cleaner_and_offset_restorer_pair() -> None:
    with pytest.raises(
        ValueError,
        match="html_cleaner and offset_restorer must both be provided or both be None",
    ):
        BatchPipeline(
            splitter=StubSplitter(_make_sentences(1)),
            marker=StubMarker(MarkedText(tagged_text="{0} A", sentence_count=1)),
            parser=StubParser(
                [SentenceGroup(label=("A",), ranges=(SentenceRange(0, 0),))]
            ),
            gap_handler=StubGapHandler(
                [SentenceGroup(label=("A",), ranges=(SentenceRange(0, 0),))]
            ),
            html_cleaner=StubHtmlCleaner(
                "Text",
                OffsetMapping(
                    segments=(
                        OffsetSegment(
                            clean_offset=0,
                            original_offset=0,
                            length=1,
                        ),
                    ),
                    original_length=1,
                    clean_length=1,
                ),
            ),
        )
