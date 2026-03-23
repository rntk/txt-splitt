from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

from split_text import build_cache_store, create_pipeline, wrap_async_llm, wrap_sync_llm
from txt_splitt import (
    CachingAsyncLLMCallable,
    CachingLLMCallable,
    SQLiteLLMCacheStore,
    Tracer,
    TracingAsyncLLMCallable,
    TracingLLMCallable,
)
from txt_splitt.sentences import (
    BoundaryEvaluator,
    LLMRepairingGapHandler,
    ShortSentenceEnhancer,
    TopicListLLM,
    TopicRangeAssignmentLLM,
    TopicRangeLLM,
)


class StubLLM:
    def call(self, prompt: str, temperature: float) -> str:
        return "ok"


class AsyncStubLLM:
    async def call(self, prompt: str, temperature: float) -> str:
        return "ok"


def _make_args(**overrides: object) -> SimpleNamespace:
    defaults: dict[str, object] = {
        "model": "demo-model",
        "cache_db": None,
        "cache_nonzero_temperature": False,
        "anchor_words": 12,
        "long_sentence_threshold": 24,
        "min_sentence_words": 4,
        "short_sentence_min_length": 20,
        "boundary_context_window": 3,
        "boundary_max_shift": 2,
        "temperature": 0.0,
        "single_stage": False,
        "max_chunk_chars": 84_000,
        "max_concurrent": 10,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_build_cache_store_returns_sqlite_store(tmp_path: Path) -> None:
    store = build_cache_store(_make_args(cache_db=str(tmp_path / "cache.sqlite")))

    assert isinstance(store, SQLiteLLMCacheStore)


def test_wrap_sync_llm_applies_cache_before_tracing(tmp_path: Path) -> None:
    args = _make_args(
        cache_db=str(tmp_path / "cache.sqlite"),
        cache_nonzero_temperature=True,
    )
    tracer = Tracer()
    wrapped = wrap_sync_llm(
        StubLLM(),
        namespace="topic-range",
        args=args,
        tracer=tracer,
        cache_store=build_cache_store(args),
    )

    assert isinstance(wrapped, TracingLLMCallable)
    assert isinstance(wrapped._inner, CachingLLMCallable)
    assert wrapped._inner._namespace == "topic-range"
    assert wrapped._inner._model_id == "demo-model"
    assert wrapped._inner._cache_nonzero_temperature is True


def test_wrap_async_llm_applies_cache_before_tracing(tmp_path: Path) -> None:
    args = _make_args(cache_db=str(tmp_path / "cache.sqlite"))
    tracer = Tracer()
    wrapped = wrap_async_llm(
        AsyncStubLLM(),
        namespace="topic-range-assignment",
        args=args,
        tracer=tracer,
        cache_store=build_cache_store(args),
    )

    assert isinstance(wrapped, TracingAsyncLLMCallable)
    assert isinstance(wrapped._inner, CachingAsyncLLMCallable)
    assert wrapped._inner._namespace == "topic-range-assignment"


def test_create_pipeline_assigns_distinct_cache_namespaces(tmp_path: Path) -> None:
    args = _make_args(cache_db=str(tmp_path / "cache.sqlite"))
    pipeline = create_pipeline(
        args,
        Path("input.txt"),
        StubLLM(),
        AsyncStubLLM(),
        Tracer(),
        build_cache_store(args),
    )

    assert isinstance(pipeline._topic_extractor, TopicListLLM)
    assert isinstance(pipeline._topic_extractor._client, TracingLLMCallable)
    topic_list_client = cast(
        CachingLLMCallable, pipeline._topic_extractor._client._inner
    )
    assert topic_list_client._namespace == "topic-list"

    assert isinstance(pipeline._range_assigner, TopicRangeAssignmentLLM)
    assert isinstance(pipeline._range_assigner._client, TracingAsyncLLMCallable)
    range_assigner_client = cast(
        CachingAsyncLLMCallable, pipeline._range_assigner._client._inner
    )
    assert range_assigner_client._namespace == "topic-range-assignment"

    assert isinstance(pipeline._gap_handler, LLMRepairingGapHandler)
    assert isinstance(pipeline._gap_handler._client, TracingLLMCallable)
    gap_repair_client = cast(CachingLLMCallable, pipeline._gap_handler._client._inner)
    assert gap_repair_client._namespace == "gap-repair"

    assert len(pipeline._enhancers) == 2
    first_enhancer = pipeline._enhancers[0]
    second_enhancer = pipeline._enhancers[1]
    assert isinstance(first_enhancer, ShortSentenceEnhancer)
    assert isinstance(first_enhancer._client, TracingLLMCallable)
    short_sentence_client = cast(CachingLLMCallable, first_enhancer._client._inner)
    assert short_sentence_client._namespace == "short-sentence-enhancer"
    assert isinstance(second_enhancer, BoundaryEvaluator)
    assert isinstance(second_enhancer._client, TracingLLMCallable)
    boundary_client = cast(CachingLLMCallable, second_enhancer._client._inner)
    assert boundary_client._namespace == "boundary-evaluator"


def test_create_pipeline_uses_topic_range_namespace_in_single_stage(
    tmp_path: Path,
) -> None:
    args = _make_args(
        cache_db=str(tmp_path / "cache.sqlite"),
        short_sentence_min_length=0,
        boundary_max_shift=0,
        single_stage=True,
    )
    pipeline = create_pipeline(
        args,
        Path("input.txt"),
        StubLLM(),
        tracer=Tracer(),
        cache_store=build_cache_store(args),
    )

    assert isinstance(pipeline._llm, TopicRangeLLM)
    assert isinstance(pipeline._llm._client, TracingLLMCallable)
    topic_range_client = cast(CachingLLMCallable, pipeline._llm._client._inner)
    assert topic_range_client._namespace == "topic-range"
