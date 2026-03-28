"""Shared protocol definitions for the generic pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, TypeVar

from txt_splitt.types import OffsetMapping

TResult = TypeVar("TResult")


class LLMCallable(Protocol):
    """Protocol for sync LLM client callables."""

    def call(self, prompt: str, temperature: float) -> str: ...


class AsyncLLMCallable(Protocol):
    """Protocol for async LLM client callables."""

    async def call(self, prompt: str, temperature: float) -> str: ...


@dataclass(frozen=True, slots=True)
class LLMRequest:
    """Raw LLM work item emitted by a pipeline session."""

    prompt: str
    temperature: float
    response_format: str = "text"
    stage_name: str = "llm"
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Raw LLM response returned to a pipeline session."""

    content: str


class HtmlCleaner(Protocol):
    """Strip HTML tags while returning an offset mapping."""

    def clean(self, text: str) -> tuple[str, OffsetMapping]: ...


class OffsetRestorer(Protocol[TResult]):
    """Restore original offsets on a final result object."""

    def restore(self, result: TResult, mapping: OffsetMapping) -> TResult: ...
