"""Tests for LLM cache wrappers and stores."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from txt_splitt.cache import (
    AsyncLLMCacheStore,
    CacheEntry,
    CachingAsyncLLMCallable,
    CachingLLMCallable,
    MemoryLLMCacheStore,
    SQLiteLLMCacheStore,
)
from txt_splitt.sentences.gap_handlers import LLMRepairingGapHandler
from txt_splitt.sentences.llm import HierarchicalTopicRangeLLM
from txt_splitt.sentences.types import (
    MarkedText,
    Sentence,
    SentenceGroup,
    SentenceRange,
)
from txt_splitt.tracer import Tracer, TracingLLMCallable


class AsyncMemoryStore(AsyncLLMCacheStore):
    def __init__(self) -> None:
        self._entries: dict[str, CacheEntry] = {}

    async def get(self, key: str) -> CacheEntry | None:
        return self._entries.get(key)

    async def set(self, entry: CacheEntry) -> None:
        self._entries[entry.key] = entry


class AsyncStubLLM:
    def __init__(self, response: str) -> None:
        self._response = response
        self.calls: list[tuple[str, float]] = []

    async def call(self, prompt: str, temperature: float) -> str:
        self.calls.append((prompt, temperature))
        return self._response


def _make_sentences(n: int) -> list[Sentence]:
    return [
        Sentence(index=i, start=i * 10, end=i * 10 + 8, text=f"Sentence {i}.")
        for i in range(n)
    ]


class TestCachingLLMCallable:
    def test_cache_hit_reuses_response(self) -> None:
        inner = MagicMock()
        inner.call.return_value = "cached-response"
        wrapper = CachingLLMCallable(
            inner,
            MemoryLLMCacheStore(),
            namespace="topic-range",
            model_id="demo-model",
            prompt_version="v1",
        )

        first = wrapper.call("prompt", 0.0)
        second = wrapper.call("prompt", 0.0)

        assert first == "cached-response"
        assert second == "cached-response"
        inner.call.assert_called_once_with("prompt", 0.0)

    def test_cache_key_separates_namespace_and_version(self) -> None:
        inner = MagicMock()
        inner.call.side_effect = ["one", "two", "three"]
        store = MemoryLLMCacheStore()
        a = CachingLLMCallable(
            inner,
            store,
            namespace="topic-range",
            model_id="demo-model",
            prompt_version="v1",
        )
        b = CachingLLMCallable(
            inner,
            store,
            namespace="gap-repair",
            model_id="demo-model",
            prompt_version="v1",
        )
        c = CachingLLMCallable(
            inner,
            store,
            namespace="topic-range",
            model_id="demo-model",
            prompt_version="v2",
        )

        assert a.call("prompt", 0.0) == "one"
        assert b.call("prompt", 0.0) == "two"
        assert c.call("prompt", 0.0) == "three"
        assert inner.call.call_count == 3

    def test_nonzero_temperature_bypasses_cache_by_default(self) -> None:
        inner = MagicMock()
        inner.call.side_effect = ["first", "second"]
        wrapper = CachingLLMCallable(
            inner,
            MemoryLLMCacheStore(),
            namespace="topic-range",
        )

        assert wrapper.call("prompt", 0.2) == "first"
        assert wrapper.call("prompt", 0.2) == "second"
        assert inner.call.call_count == 2

    def test_nonzero_temperature_can_be_cached_when_enabled(self) -> None:
        inner = MagicMock()
        inner.call.return_value = "same"
        wrapper = CachingLLMCallable(
            inner,
            MemoryLLMCacheStore(),
            namespace="topic-range",
            cache_nonzero_temperature=True,
        )

        assert wrapper.call("prompt", 0.2) == "same"
        assert wrapper.call("prompt", 0.2) == "same"
        inner.call.assert_called_once_with("prompt", 0.2)

    def test_invalid_namespace_raises(self) -> None:
        inner = MagicMock()
        with pytest.raises(ValueError, match="namespace must be non-empty"):
            CachingLLMCallable(inner, MemoryLLMCacheStore(), namespace="   ")

    def test_tracing_records_cache_metadata(self) -> None:
        tracer = Tracer()
        inner = MagicMock()
        inner.call.return_value = "resp"
        cached = CachingLLMCallable(
            inner,
            MemoryLLMCacheStore(),
            namespace="topic-range",
            model_id="demo-model",
            prompt_version="v1",
        )
        traced = TracingLLMCallable(cached, tracer)

        traced.call("prompt", 0.0)
        traced.call("prompt", 0.0)

        first = tracer.spans[0]
        second = tracer.spans[1]
        assert first.attributes["cache_hit"] is False
        assert second.attributes["cache_hit"] is True
        assert first.attributes["cache_namespace"] == "topic-range"
        assert second.attributes["cache_backend"] == "MemoryLLMCacheStore"


class TestCachingAsyncLLMCallable:
    def test_async_store_and_client_are_supported(self) -> None:
        async def run_test() -> None:
            inner = AsyncStubLLM("async-response")
            wrapper = CachingAsyncLLMCallable(
                inner,
                AsyncMemoryStore(),
                namespace="topic-range",
                model_id="demo-model",
                prompt_version="v1",
            )

            first = await wrapper.call("prompt", 0.0)
            second = await wrapper.call("prompt", 0.0)

            assert first == "async-response"
            assert second == "async-response"
            assert inner.calls == [("prompt", 0.0)]

        asyncio.run(run_test())

    def test_sync_store_works_with_async_wrapper(self) -> None:
        async def run_test() -> None:
            inner = AsyncStubLLM("async-response")
            wrapper = CachingAsyncLLMCallable(
                inner,
                MemoryLLMCacheStore(),
                namespace="topic-range",
            )

            await wrapper.call("prompt", 0.0)
            await wrapper.call("prompt", 0.0)

            assert inner.calls == [("prompt", 0.0)]

        asyncio.run(run_test())


class TestSQLiteLLMCacheStore:
    def test_persists_across_instances(self, tmp_path: Path) -> None:
        db_path = tmp_path / "llm-cache.sqlite"
        first_store = SQLiteLLMCacheStore(db_path)
        inner = MagicMock()
        inner.call.return_value = "sqlite-response"
        first_wrapper = CachingLLMCallable(
            inner,
            first_store,
            namespace="topic-range",
            model_id="demo-model",
            prompt_version="v1",
        )

        assert first_wrapper.call("prompt", 0.0) == "sqlite-response"
        assert inner.call.call_count == 1

        second_store = SQLiteLLMCacheStore(db_path)
        second_inner = MagicMock()
        second_wrapper = CachingLLMCallable(
            second_inner,
            second_store,
            namespace="topic-range",
            model_id="demo-model",
            prompt_version="v1",
        )

        assert second_wrapper.call("prompt", 0.0) == "sqlite-response"
        second_inner.call.assert_not_called()


class TestCacheIntegration:
    def test_topic_range_llm_uses_cached_client(self) -> None:
        inner = MagicMock()
        inner.call.return_value = "Topic: 0-1"
        client = CachingLLMCallable(
            inner,
            MemoryLLMCacheStore(),
            namespace="topic-range",
            prompt_version="v1",
        )
        llm = HierarchicalTopicRangeLLM(client)
        marked_text = MarkedText(tagged_text="{0} A", sentence_count=1)

        assert llm.query(marked_text) == "Topic: 0-1"
        assert llm.query(marked_text) == "Topic: 0-1"
        assert inner.call.call_count == 2

    def test_gap_handler_uses_cached_client(self) -> None:
        inner = MagicMock()
        inner.call.return_value = "PREVIOUS"
        client = CachingLLMCallable(
            inner,
            MemoryLLMCacheStore(),
            namespace="gap-repair",
            prompt_version="v1",
        )
        handler = LLMRepairingGapHandler(client)
        groups = [
            SentenceGroup(label=("A",), ranges=(SentenceRange(0, 0),)),
            SentenceGroup(label=("B",), ranges=(SentenceRange(2, 2),)),
        ]
        sentences = _make_sentences(3)

        first = handler.handle(groups, sentence_count=3, sentences=sentences)
        second = handler.handle(groups, sentence_count=3, sentences=sentences)

        assert first[0].ranges == (SentenceRange(0, 1),)
        assert second[0].ranges == (SentenceRange(0, 1),)
        inner.call.assert_called_once()
