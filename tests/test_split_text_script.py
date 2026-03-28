from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from split_text import (
    _namespace_for_request,
    build_cache_store,
    create_pipeline,
    wrap_async_llm,
    wrap_sync_llm,
)
from txt_splitt import (
    CachingAsyncLLMCallable,
    CachingLLMCallable,
    LLMRequest,
    SQLiteLLMCacheStore,
    Tracer,
    TracingAsyncLLMCallable,
    TracingLLMCallable,
)
from txt_splitt.sentences import (
    HierarchicalTopicRangeLLM,
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


def test_create_pipeline_uses_single_stage_only(tmp_path: Path) -> None:
    """Two-stage pipeline is no longer supported - only single-stage works."""
    args = _make_args(
        cache_db=str(tmp_path / "cache.sqlite"),
        short_sentence_min_length=0,
        boundary_max_shift=0,
        single_stage=True,
    )
    pipeline = create_pipeline(
        args,
        Path("input.txt"),
        tracer=Tracer(),
        cache_store=build_cache_store(args),
    )

    assert isinstance(pipeline._llm, HierarchicalTopicRangeLLM)


def test_create_pipeline_two_stage_raises_not_implemented(tmp_path: Path) -> None:
    """Two-stage pipeline (single_stage=False) raises NotImplementedError."""
    args = _make_args(
        cache_db=str(tmp_path / "cache.sqlite"),
        single_stage=False,
    )
    with pytest.raises(NotImplementedError, match="Two-stage pipeline"):
        create_pipeline(
            args,
            Path("input.txt"),
            tracer=Tracer(),
            cache_store=build_cache_store(args),
        )


def test_namespace_for_request_returns_explicit_namespace() -> None:
    request = LLMRequest(
        prompt="prompt",
        temperature=0.0,
        metadata={"namespace": "gap-repair"},
    )

    assert _namespace_for_request(request) == "gap-repair"


def test_namespace_for_request_raises_when_missing() -> None:
    request = LLMRequest(
        prompt="prompt",
        temperature=0.0,
        stage_name="topic_range.coarse",
    )

    with pytest.raises(ValueError, match="missing a valid namespace"):
        _namespace_for_request(request)
