"""Tests for pipeline Prometheus metrics."""

import pytest
from prometheus_client import CollectorRegistry

from txt_splitt.metrics import NoOpMetrics, PipelineMetrics
from txt_splitt.pipeline import Pipeline
from txt_splitt.types import (
    MarkedText,
    Sentence,
    SentenceGroup,
    SentenceRange,
    SplitResult,
)

# ---- helpers / stubs ----


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
        self,
        groups: list[SentenceGroup],
        sentence_count: int,
        sentences: list[Sentence] | None = None,
    ) -> list[SentenceGroup]:
        return self._groups


class FailingSplitter:
    def split(self, text: str) -> list[Sentence]:
        raise RuntimeError("boom")


# ---- helpers to read metrics ----


def _get_histogram_count(
    registry: CollectorRegistry, name: str, labels: dict[str, str]
) -> float:
    """Return the _count value for a histogram with given labels."""
    for metric in registry.collect():
        if metric.name == name:
            for sample in metric.samples:
                if sample.name == f"{name}_count" and all(
                    sample.labels.get(k) == v for k, v in labels.items()
                ):
                    return float(sample.value)
    return 0.0


def _get_counter_value(
    registry: CollectorRegistry, name: str, labels: dict[str, str]
) -> float:
    """Return the value of a counter with given labels."""
    for metric in registry.collect():
        if metric.name == name:
            for sample in metric.samples:
                if sample.name == f"{name}_total" and all(
                    sample.labels.get(k) == v for k, v in labels.items()
                ):
                    return float(sample.value)
    return 0.0


# ---- PipelineMetrics unit tests ----


class TestPipelineMetrics:
    def test_stage_success_records_duration(self) -> None:
        registry = CollectorRegistry()
        metrics = PipelineMetrics(registry=registry)

        with metrics.stage("my_pipe", "split"):
            pass  # simulate work

        count = _get_histogram_count(
            registry,
            "pipeline_stage_duration_seconds",
            {"pipeline": "my_pipe", "stage": "split"},
        )
        assert count == 1.0

    def test_stage_success_no_failure_count(self) -> None:
        registry = CollectorRegistry()
        metrics = PipelineMetrics(registry=registry)

        with metrics.stage("my_pipe", "split"):
            pass

        failures = _get_counter_value(
            registry,
            "pipeline_stage_failures",
            {"pipeline": "my_pipe", "stage": "split"},
        )
        assert failures == 0.0

    def test_stage_failure_records_duration_and_increments_counter(self) -> None:
        registry = CollectorRegistry()
        metrics = PipelineMetrics(registry=registry)

        with (
            pytest.raises(ValueError, match="bad"),
            metrics.stage("pipe1", "llm_query"),
        ):
            raise ValueError("bad")

        count = _get_histogram_count(
            registry,
            "pipeline_stage_duration_seconds",
            {"pipeline": "pipe1", "stage": "llm_query"},
        )
        assert count == 1.0

        failures = _get_counter_value(
            registry,
            "pipeline_stage_failures",
            {"pipeline": "pipe1", "stage": "llm_query"},
        )
        assert failures == 1.0

    def test_multiple_calls_accumulate(self) -> None:
        registry = CollectorRegistry()
        metrics = PipelineMetrics(registry=registry)

        for _ in range(3):
            with metrics.stage("p", "parse"):
                pass

        count = _get_histogram_count(
            registry,
            "pipeline_stage_duration_seconds",
            {"pipeline": "p", "stage": "parse"},
        )
        assert count == 3.0


# ---- NoOpMetrics unit tests ----


class TestNoOpMetrics:
    def test_stage_context_manager_is_noop(self) -> None:
        metrics = NoOpMetrics()
        with metrics.stage("any", "stage"):
            pass  # should not raise

    def test_stage_does_not_swallow_exceptions(self) -> None:
        metrics = NoOpMetrics()
        with pytest.raises(RuntimeError, match="oops"), metrics.stage("any", "stage"):
            raise RuntimeError("oops")


# ---- Pipeline integration tests ----


class TestPipelineMetricsIntegration:
    def test_metrics_recorded_on_successful_run(self) -> None:
        registry = CollectorRegistry()
        metrics = PipelineMetrics(registry=registry)
        sentences = _make_sentences(3)
        groups = _make_groups()

        pipeline = Pipeline(
            splitter=StubSplitter(sentences),
            marker=StubMarker(MarkedText(tagged_text="...", sentence_count=3)),
            llm=StubLLM("Technology>AI: 0-2"),
            parser=StubParser(groups),
            gap_handler=StubGapHandler(groups),
            name="test_pipe",
            metrics=metrics,
        )
        pipeline.run("Some text")

        for stage in ["split", "mark", "llm_query", "parse", "gap_handler"]:
            count = _get_histogram_count(
                registry,
                "pipeline_stage_duration_seconds",
                {"pipeline": "test_pipe", "stage": stage},
            )
            assert count == 1.0, f"Expected 1 observation for stage {stage}"

    def test_metrics_recorded_on_failure(self) -> None:
        registry = CollectorRegistry()
        metrics = PipelineMetrics(registry=registry)

        pipeline = Pipeline(
            splitter=FailingSplitter(),
            marker=StubMarker(MarkedText(tagged_text="...", sentence_count=0)),
            llm=StubLLM("response"),
            parser=StubParser([]),
            gap_handler=StubGapHandler([]),
            name="fail_pipe",
            metrics=metrics,
        )

        with pytest.raises(RuntimeError):
            pipeline.run("text")

        failures = _get_counter_value(
            registry,
            "pipeline_stage_failures",
            {"pipeline": "fail_pipe", "stage": "split"},
        )
        assert failures == 1.0

    def test_pipeline_works_without_metrics(self) -> None:
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

    def test_pipeline_works_with_metrics_none(self) -> None:
        sentences = _make_sentences(3)
        groups = _make_groups()

        pipeline = Pipeline(
            splitter=StubSplitter(sentences),
            marker=StubMarker(MarkedText(tagged_text="...", sentence_count=3)),
            llm=StubLLM("Technology>AI: 0-2"),
            parser=StubParser(groups),
            gap_handler=StubGapHandler(groups),
            metrics=None,
        )
        result = pipeline.run("Some text")

        assert isinstance(result, SplitResult)
        assert len(result.sentences) == 3
