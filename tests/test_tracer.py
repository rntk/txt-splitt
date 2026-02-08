"""Tests for the tracer module."""

import time

import pytest

from txt_splitt.pipeline import Pipeline
from txt_splitt.tracer import Span, Tracer, TracingLLMCallable
from txt_splitt.types import (
    MarkedText,
    Sentence,
    SentenceGroup,
    SentenceRange,
    SplitResult,
)


class TestSpan:
    def test_duration_ms_none_end(self) -> None:
        s = Span(name="x", start_time=1.0)
        assert s.duration_ms == 0.0

    def test_duration_ms(self) -> None:
        s = Span(name="x", start_time=1.0, end_time=1.5)
        assert s.duration_ms == pytest.approx(500.0)

    def test_children_default_empty(self) -> None:
        s = Span(name="x", start_time=0.0)
        assert s.children == []

    def test_attributes_default_empty(self) -> None:
        s = Span(name="x", start_time=0.0)
        assert s.attributes == {}


class TestNoOpTracer:
    def test_span_does_nothing(self) -> None:
        from txt_splitt.tracer import NoOpTracer

        tracer = NoOpTracer()
        with tracer.span("operation", key="value") as s:
            s.attributes["extra"] = "added"
        # NoOpTracer doesn't collect anything
        assert tracer.format() == ""

    def test_nested_spans(self) -> None:
        from txt_splitt.tracer import NoOpTracer

        tracer = NoOpTracer()
        with tracer.span("parent"):
            with tracer.span("child"):
                pass
        # NoOpTracer doesn't collect anything
        assert tracer.format() == ""


class TestTracer:
    def test_empty_tracer(self) -> None:
        tracer = Tracer()
        assert tracer.spans == []
        assert tracer.format() == ""

    def test_single_span(self) -> None:
        tracer = Tracer()
        with tracer.span("my_op"):
            pass
        assert len(tracer.spans) == 1
        assert tracer.spans[0].name == "my_op"
        assert tracer.spans[0].end_time is not None
        assert tracer.spans[0].duration_ms >= 0

    def test_span_attributes(self) -> None:
        tracer = Tracer()
        with tracer.span("op", key="value", count=42) as s:
            s.attributes["extra"] = "added"
        assert tracer.spans[0].attributes["key"] == "value"
        assert tracer.spans[0].attributes["count"] == 42
        assert tracer.spans[0].attributes["extra"] == "added"

    def test_nested_spans(self) -> None:
        tracer = Tracer()
        with tracer.span("parent"):
            with tracer.span("child1"):
                pass
            with tracer.span("child2"):
                pass
        assert len(tracer.spans) == 1
        parent = tracer.spans[0]
        assert parent.name == "parent"
        assert len(parent.children) == 2
        assert parent.children[0].name == "child1"
        assert parent.children[1].name == "child2"

    def test_deeply_nested(self) -> None:
        tracer = Tracer()
        with tracer.span("a"):
            with tracer.span("b"):
                with tracer.span("c"):
                    pass
        c = tracer.spans[0].children[0].children[0]
        assert c.name == "c"

    def test_multiple_root_spans(self) -> None:
        tracer = Tracer()
        with tracer.span("first"):
            pass
        with tracer.span("second"):
            pass
        assert len(tracer.spans) == 2

    def test_span_closed_on_exception(self) -> None:
        tracer = Tracer()
        with pytest.raises(ValueError):
            with tracer.span("failing"):
                raise ValueError("boom")
        assert tracer.spans[0].end_time is not None

    def test_format_output(self) -> None:
        tracer = Tracer()
        with tracer.span("parent", x=1):
            with tracer.span("child", y="hello"):
                pass
        output = tracer.format()
        assert "[TRACE] parent" in output
        assert "x: 1" in output
        assert "  [TRACE] child" in output
        assert "y: hello" in output

    def test_format_prints_full_long_values(self) -> None:
        tracer = Tracer()
        long_value = "A" * 500
        with tracer.span("op", big=long_value):
            pass
        output = tracer.format()
        assert f"big: {long_value}" in output


class StubLLM:
    def __init__(self, response: str) -> None:
        self._response = response

    def call(self, prompt: str, temperature: float) -> str:
        return self._response


class TestTracingLLMCallable:
    def test_records_call(self) -> None:
        tracer = Tracer()
        inner = StubLLM("the response")
        traced = TracingLLMCallable(inner, tracer)

        result = traced.call("the prompt", temperature=0.5)

        assert result == "the response"
        assert len(tracer.spans) == 1
        s = tracer.spans[0]
        assert s.name == "llm.call"
        assert s.attributes["prompt"] == "the prompt"
        assert s.attributes["response"] == "the response"
        assert s.attributes["prompt_length"] == 10
        assert s.attributes["response_length"] == 12
        assert s.attributes["temperature"] == 0.5

    def test_nested_under_parent_span(self) -> None:
        tracer = Tracer()
        inner = StubLLM("resp")
        traced = TracingLLMCallable(inner, tracer)

        with tracer.span("outer"):
            traced.call("prompt", temperature=0.0)

        assert len(tracer.spans) == 1
        assert tracer.spans[0].name == "outer"
        assert len(tracer.spans[0].children) == 1
        assert tracer.spans[0].children[0].name == "llm.call"

    def test_propagates_exception(self) -> None:
        tracer = Tracer()

        class FailingLLM:
            def call(self, prompt: str, temperature: float) -> str:
                raise RuntimeError("LLM down")

        traced = TracingLLMCallable(FailingLLM(), tracer)

        with pytest.raises(RuntimeError, match="LLM down"):
            traced.call("prompt", temperature=0.0)

        # Span should still be recorded (closed by context manager)
        assert len(tracer.spans) == 1
        assert tracer.spans[0].end_time is not None


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


class StubLLMStrategy:
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
        self,
        groups: list[SentenceGroup],
        sentence_count: int,
        sentences: list[Sentence] | None = None,
    ) -> list[SentenceGroup]:
        return self._groups


class TestPipelineTracing:
    def _make_sentences(self, n: int) -> list[Sentence]:
        return [
            Sentence(index=i, start=i * 10, end=i * 10 + 5, text=f"Sent {i}.")
            for i in range(n)
        ]

    def _make_groups(self) -> list[SentenceGroup]:
        return [
            SentenceGroup(
                label=("Tech",), ranges=(SentenceRange(start=0, end=2),)
            )
        ]

    def test_pipeline_without_tracer(self) -> None:
        sentences = self._make_sentences(3)
        groups = self._make_groups()
        pipeline = Pipeline(
            splitter=StubSplitter(sentences),
            marker=StubMarker(MarkedText(tagged_text="...", sentence_count=3)),
            llm=StubLLMStrategy("Tech: 0-2"),
            parser=StubParser(groups),
            gap_handler=StubGapHandler(groups),
        )
        result = pipeline.run("text")
        assert len(result.sentences) == 3

    def test_pipeline_with_none_tracer_uses_noop(self) -> None:
        from txt_splitt.tracer import NoOpTracer

        sentences = self._make_sentences(3)
        groups = self._make_groups()
        pipeline = Pipeline(
            splitter=StubSplitter(sentences),
            marker=StubMarker(MarkedText(tagged_text="...", sentence_count=3)),
            llm=StubLLMStrategy("Tech: 0-2"),
            parser=StubParser(groups),
            gap_handler=StubGapHandler(groups),
            tracer=None,  # Explicitly pass None
        )
        # Pipeline should work fine with NoOpTracer internally
        result = pipeline.run("text")
        assert len(result.sentences) == 3
        # Verify NoOpTracer was used (it's an implementation detail, but we can check)
        assert isinstance(pipeline._tracer, NoOpTracer)

    def test_pipeline_with_tracer_records_stages(self) -> None:
        tracer = Tracer()
        sentences = self._make_sentences(3)
        groups = self._make_groups()
        pipeline = Pipeline(
            splitter=StubSplitter(sentences),
            marker=StubMarker(MarkedText(tagged_text="tagged", sentence_count=3)),
            llm=StubLLMStrategy("Tech: 0-2"),
            parser=StubParser(groups),
            gap_handler=StubGapHandler(groups),
            tracer=tracer,
        )
        result = pipeline.run("some text")

        assert len(result.sentences) == 3

        # Should have one root span: pipeline.run
        assert len(tracer.spans) == 1
        root = tracer.spans[0]
        assert root.name == "pipeline.run"
        assert root.attributes["input_length"] == 9

        # 5 child stages (no enhancer)
        children = root.children
        assert len(children) == 5
        assert children[0].name == "split"
        assert children[0].attributes["sentence_count"] == 3
        assert children[1].name == "mark"
        assert children[1].attributes["tagged_text_length"] == 6
        assert children[2].name == "llm.query"
        assert children[3].name == "parse"
        assert children[3].attributes["group_count"] == 1
        assert children[4].name == "gap_handler"

    def test_pipeline_with_tracer_and_enhancer(self) -> None:
        tracer = Tracer()
        sentences = self._make_sentences(3)
        groups = self._make_groups()

        class StubEnhancer:
            def enhance(
                self, groups: list[SentenceGroup], sentences: list[Sentence]
            ) -> list[SentenceGroup]:
                return groups

        pipeline = Pipeline(
            splitter=StubSplitter(sentences),
            marker=StubMarker(MarkedText(tagged_text="...", sentence_count=3)),
            llm=StubLLMStrategy("..."),
            parser=StubParser(groups),
            gap_handler=StubGapHandler(groups),
            enhancer=StubEnhancer(),
            tracer=tracer,
        )
        pipeline.run("text")

        root = tracer.spans[0]
        assert len(root.children) == 6
        assert root.children[5].name == "enhance"

    def test_format_contains_all_stages(self) -> None:
        tracer = Tracer()
        sentences = self._make_sentences(3)
        groups = self._make_groups()
        pipeline = Pipeline(
            splitter=StubSplitter(sentences),
            marker=StubMarker(MarkedText(tagged_text="...", sentence_count=3)),
            llm=StubLLMStrategy("Tech: 0-2"),
            parser=StubParser(groups),
            gap_handler=StubGapHandler(groups),
            tracer=tracer,
        )
        pipeline.run("text")

        output = tracer.format()
        assert "pipeline.run" in output
        assert "split" in output
        assert "mark" in output
        assert "llm.query" in output
        assert "parse" in output
        assert "gap_handler" in output
