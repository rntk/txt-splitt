"""Cache wrappers and storage backends for LLM calls."""

from __future__ import annotations

import hashlib
import inspect
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from txt_splitt.protocols import AsyncLLMCallable, LLMCallable
from txt_splitt.tracer import add_span_attribute

_CACHE_KEY_VERSION = 1


@dataclass(frozen=True)
class CacheEntry:
    """A cached LLM response plus metadata used for debugging."""

    key: str
    response: str
    created_at: float
    namespace: str
    model_id: str | None
    prompt_version: str | None
    temperature: float


class LLMCacheStore(Protocol):
    """Storage backend for synchronous cache access."""

    def get(self, key: str) -> CacheEntry | None: ...

    def set(self, entry: CacheEntry) -> None: ...


class AsyncLLMCacheStore(Protocol):
    """Storage backend for asynchronous cache access."""

    async def get(self, key: str) -> CacheEntry | None: ...

    async def set(self, entry: CacheEntry) -> None: ...


class MemoryLLMCacheStore:
    """Simple in-process cache store."""

    def __init__(self) -> None:
        self._entries: dict[str, CacheEntry] = {}

    def get(self, key: str) -> CacheEntry | None:
        return self._entries.get(key)

    def set(self, entry: CacheEntry) -> None:
        self._entries[entry.key] = entry


class SQLiteLLMCacheStore:
    """SQLite-backed persistent cache store."""

    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_cache (
                    key TEXT PRIMARY KEY,
                    response TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    namespace TEXT NOT NULL,
                    model_id TEXT,
                    prompt_version TEXT,
                    temperature REAL NOT NULL
                )
                """
            )
            self._conn.commit()

    def get(self, key: str) -> CacheEntry | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT key, response, created_at, namespace, model_id,
                       prompt_version, temperature
                FROM llm_cache
                WHERE key = ?
                """,
                (key,),
            ).fetchone()

        if row is None:
            return None

        return CacheEntry(
            key=str(row["key"]),
            response=str(row["response"]),
            created_at=float(row["created_at"]),
            namespace=str(row["namespace"]),
            model_id=_optional_str(row["model_id"]),
            prompt_version=_optional_str(row["prompt_version"]),
            temperature=float(row["temperature"]),
        )

    def set(self, entry: CacheEntry) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO llm_cache (
                    key, response, created_at, namespace, model_id,
                    prompt_version, temperature
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.key,
                    entry.response,
                    entry.created_at,
                    entry.namespace,
                    entry.model_id,
                    entry.prompt_version,
                    entry.temperature,
                ),
            )
            self._conn.commit()


class CachingLLMCallable:
    """Wrap an LLM client with response caching."""

    def __init__(
        self,
        inner: LLMCallable,
        store: LLMCacheStore,
        *,
        namespace: str,
        model_id: str | None = None,
        prompt_version: str | None = None,
        cache_nonzero_temperature: bool = False,
    ) -> None:
        if not namespace.strip():
            raise ValueError("namespace must be non-empty")
        self._inner = inner
        self._store = store
        self._namespace = namespace
        self._model_id = model_id
        self._prompt_version = prompt_version
        self._cache_nonzero_temperature = cache_nonzero_temperature

    def call(self, prompt: str, temperature: float) -> str:
        if not self._should_cache(temperature):
            self._annotate_cache_event(
                hit=False,
                cache_key=None,
                bypass_reason="nonzero_temperature",
            )
            return self._inner.call(prompt, temperature)

        cache_key = _build_cache_key(
            namespace=self._namespace,
            model_id=self._model_id,
            prompt_version=self._prompt_version,
            prompt=prompt,
            temperature=temperature,
        )
        entry = self._store.get(cache_key)
        if entry is not None:
            self._annotate_cache_event(hit=True, cache_key=cache_key)
            return entry.response

        response = self._inner.call(prompt, temperature)
        self._store.set(
            CacheEntry(
                key=cache_key,
                response=response,
                created_at=time.time(),
                namespace=self._namespace,
                model_id=self._model_id,
                prompt_version=self._prompt_version,
                temperature=temperature,
            )
        )
        self._annotate_cache_event(hit=False, cache_key=cache_key)
        return response

    def _should_cache(self, temperature: float) -> bool:
        return self._cache_nonzero_temperature or temperature == 0.0

    def _annotate_cache_event(
        self,
        *,
        hit: bool,
        cache_key: str | None,
        bypass_reason: str | None = None,
    ) -> None:
        add_span_attribute("cache_backend", type(self._store).__name__)
        add_span_attribute("cache_namespace", self._namespace)
        add_span_attribute("cache_hit", hit)
        if cache_key is not None:
            add_span_attribute("cache_key", cache_key)
        if bypass_reason is not None:
            add_span_attribute("cache_bypass_reason", bypass_reason)


class CachingAsyncLLMCallable:
    """Wrap an async or sync LLM client with response caching."""

    def __init__(
        self,
        inner: AsyncLLMCallable | LLMCallable,
        store: AsyncLLMCacheStore | LLMCacheStore,
        *,
        namespace: str,
        model_id: str | None = None,
        prompt_version: str | None = None,
        cache_nonzero_temperature: bool = False,
    ) -> None:
        if not namespace.strip():
            raise ValueError("namespace must be non-empty")
        self._inner = inner
        self._store = store
        self._namespace = namespace
        self._model_id = model_id
        self._prompt_version = prompt_version
        self._cache_nonzero_temperature = cache_nonzero_temperature
        self._is_async_client = inspect.iscoroutinefunction(inner.call)
        self._is_async_store = inspect.iscoroutinefunction(store.get)

    async def call(self, prompt: str, temperature: float) -> str:
        if not self._should_cache(temperature):
            self._annotate_cache_event(
                hit=False,
                cache_key=None,
                bypass_reason="nonzero_temperature",
            )
            return await self._call_inner(prompt, temperature)

        cache_key = _build_cache_key(
            namespace=self._namespace,
            model_id=self._model_id,
            prompt_version=self._prompt_version,
            prompt=prompt,
            temperature=temperature,
        )
        entry = await self._get_entry(cache_key)
        if entry is not None:
            self._annotate_cache_event(hit=True, cache_key=cache_key)
            return entry.response

        response = await self._call_inner(prompt, temperature)
        await self._set_entry(
            CacheEntry(
                key=cache_key,
                response=response,
                created_at=time.time(),
                namespace=self._namespace,
                model_id=self._model_id,
                prompt_version=self._prompt_version,
                temperature=temperature,
            )
        )
        self._annotate_cache_event(hit=False, cache_key=cache_key)
        return response

    async def _call_inner(self, prompt: str, temperature: float) -> str:
        if self._is_async_client:
            async_inner = cast(AsyncLLMCallable, self._inner)
            return await async_inner.call(prompt, temperature)
        sync_inner = cast(LLMCallable, self._inner)
        return sync_inner.call(prompt, temperature)

    async def _get_entry(self, key: str) -> CacheEntry | None:
        if self._is_async_store:
            async_store = cast(AsyncLLMCacheStore, self._store)
            return await async_store.get(key)
        sync_store = cast(LLMCacheStore, self._store)
        return sync_store.get(key)

    async def _set_entry(self, entry: CacheEntry) -> None:
        if self._is_async_store:
            async_store = cast(AsyncLLMCacheStore, self._store)
            await async_store.set(entry)
            return
        sync_store = cast(LLMCacheStore, self._store)
        sync_store.set(entry)

    def _should_cache(self, temperature: float) -> bool:
        return self._cache_nonzero_temperature or temperature == 0.0

    def _annotate_cache_event(
        self,
        *,
        hit: bool,
        cache_key: str | None,
        bypass_reason: str | None = None,
    ) -> None:
        add_span_attribute("cache_backend", type(self._store).__name__)
        add_span_attribute("cache_namespace", self._namespace)
        add_span_attribute("cache_hit", hit)
        if cache_key is not None:
            add_span_attribute("cache_key", cache_key)
        if bypass_reason is not None:
            add_span_attribute("cache_bypass_reason", bypass_reason)


def _build_cache_key(
    *,
    namespace: str,
    model_id: str | None,
    prompt_version: str | None,
    prompt: str,
    temperature: float,
) -> str:
    payload = json.dumps(
        {
            "version": _CACHE_KEY_VERSION,
            "namespace": namespace,
            "model_id": model_id,
            "prompt_version": prompt_version,
            "prompt": prompt,
            "temperature": temperature,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
