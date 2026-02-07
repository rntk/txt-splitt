"""Integration tests for the pipeline with stubs."""

import pytest

from txt_splitt.errors import GapError, LLMError, ParseError
from txt_splitt.pipeline import Pipeline
from txt_splitt.types import (
    MarkedText,
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
    def __init__(self, result: MarkedText) -> None:
        self._result = result

    def mark(self, text: str, sentences: list[Sentence]) -> MarkedText:
        return self._result


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


class StubGapHandler:
    def __init__(self, groups: list[SentenceGroup]) -> None:
        self._groups = groups

    def handle(
        self, groups: list[SentenceGroup], sentence_count: int
    ) -> list[SentenceGroup]:
        return self._groups


class FailingLLM:
    def query(self, marked_text: MarkedText) -> str:
        raise LLMError("LLM unavailable")


class FailingParser:
    def parse(self, response: str, sentence_count: int) -> list[SentenceGroup]:
        raise ParseError("Cannot parse")


class FailingGapHandler:
    def handle(
        self, groups: list[SentenceGroup], sentence_count: int
    ) -> list[SentenceGroup]:
        raise GapError("Gap found")


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
        ),
    ]


class TestPipeline:
    def test_successful_run(self) -> None:
        sentences = _make_sentences(3)
        groups = _make_groups()

        pipeline = Pipeline(
            splitter=StubSplitter(sentences),
            marker=StubMarker(MarkedText(tagged_text="...", sentence_count=3)),
            llm=StubLLM("Technology>AI: 0-2"),
            parser=StubParser(groups),
            gap_handler=StubGapHandler(groups),
        )
        result = pipeline.run("Some text")

        assert isinstance(result, SplitResult)
        assert len(result.sentences) == 3
        assert len(result.groups) == 1
        assert result.groups[0].label == ("Technology", "AI")

    def test_result_is_immutable_tuples(self) -> None:
        sentences = _make_sentences(2)
        groups = _make_groups()

        pipeline = Pipeline(
            splitter=StubSplitter(sentences),
            marker=StubMarker(MarkedText(tagged_text="...", sentence_count=2)),
            llm=StubLLM("..."),
            parser=StubParser(groups),
            gap_handler=StubGapHandler(groups),
        )
        result = pipeline.run("text")

        assert isinstance(result.sentences, tuple)
        assert isinstance(result.groups, tuple)

    def test_llm_error_propagates(self) -> None:
        sentences = _make_sentences(3)

        pipeline = Pipeline(
            splitter=StubSplitter(sentences),
            marker=StubMarker(MarkedText(tagged_text="...", sentence_count=3)),
            llm=FailingLLM(),
            parser=StubParser([]),
            gap_handler=StubGapHandler([]),
        )
        with pytest.raises(LLMError):
            pipeline.run("text")

    def test_parse_error_propagates(self) -> None:
        sentences = _make_sentences(3)

        pipeline = Pipeline(
            splitter=StubSplitter(sentences),
            marker=StubMarker(MarkedText(tagged_text="...", sentence_count=3)),
            llm=StubLLM("response"),
            parser=FailingParser(),
            gap_handler=StubGapHandler([]),
        )
        with pytest.raises(ParseError):
            pipeline.run("text")

    def test_gap_error_propagates(self) -> None:
        sentences = _make_sentences(3)
        groups = _make_groups()

        pipeline = Pipeline(
            splitter=StubSplitter(sentences),
            marker=StubMarker(MarkedText(tagged_text="...", sentence_count=3)),
            llm=StubLLM("response"),
            parser=StubParser(groups),
            gap_handler=FailingGapHandler(),
        )
        with pytest.raises(GapError):
            pipeline.run("text")

    def test_end_to_end_with_real_stages(self) -> None:
        """Integration test using real splitter, marker, parser, gap_handler."""
        from txt_splitt.gap_handlers import StrictGapHandler
        from txt_splitt.markers import BracketMarker
        from txt_splitt.parsers import TopicRangeParser
        from txt_splitt.splitters import RegexSentenceSplitter

        text = "AI is growing fast. Climate change is real."

        llm_response = "Technology>AI: 0\nScience>Climate: 1"

        pipeline = Pipeline(
            splitter=RegexSentenceSplitter(),
            marker=BracketMarker(),
            llm=StubLLM(llm_response),
            parser=TopicRangeParser(),
            gap_handler=StrictGapHandler(),
        )
        result = pipeline.run(text)

        assert len(result.sentences) == 2
        assert len(result.groups) == 2
        assert result.sentences[0].text == "AI is growing fast."
        assert result.sentences[1].text == "Climate change is real."
        assert result.groups[0].label == ("Technology", "AI")
        assert result.groups[1].label == ("Science", "Climate")
