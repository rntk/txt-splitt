"""Retry wrapper for LLM calls."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from txt_splitt.errors import LLMError

if TYPE_CHECKING:
    from txt_splitt.protocols import LLMCallable


class RetryPolicy(Protocol):
    """Protocol for LLM retry policies.

    Called after each failed attempt. Return ``(new_prompt, new_temperature)``
    to retry with (optionally modified) parameters, or ``None`` to stop
    retrying and re-raise the error.

    ``attempt`` is 0-indexed: ``0`` means the first retry (after the initial
    call failed), ``1`` means the second retry, and so on.
    """

    def next(
        self,
        attempt: int,
        prompt: str,
        temperature: float,
        error: Exception,
    ) -> tuple[str, float] | None: ...


@dataclass
class RetryConfig:
    """Simple declarative retry policy.

    Retries up to ``max_attempts`` additional times after the initial failure.
    On each retry, the prompt and temperature may be updated.

    Args:
        max_attempts: Maximum number of retries (not counting the initial attempt).
        temperature_schedule: Temperatures to use on retry attempts 0, 1, 2 …
            If shorter than ``max_attempts``, the original temperature is
            reused once the list is exhausted.
        prompt_modifier: ``(prompt, attempt) -> new_prompt`` applied on each
            retry.  ``attempt`` is 0-indexed (0 = first retry).

    Example::

        # Raise temperature on each retry and append a hint to the prompt
        policy = RetryConfig(
            max_attempts=3,
            temperature_schedule=[0.2, 0.5, 0.8],
            prompt_modifier=lambda p, _: p + "\\nBe concise.",
        )
    """

    max_attempts: int = 3
    temperature_schedule: list[float] | None = None
    prompt_modifier: Callable[[str, int], str] | None = None

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            msg = f"max_attempts must be >= 1, got {self.max_attempts}"
            raise ValueError(msg)

    def next(
        self,
        attempt: int,
        prompt: str,
        temperature: float,
        error: Exception,
    ) -> tuple[str, float] | None:
        if attempt >= self.max_attempts:
            return None
        new_temp = (
            self.temperature_schedule[attempt]
            if self.temperature_schedule and attempt < len(self.temperature_schedule)
            else temperature
        )
        new_temp = max(0.0, min(2.0, new_temp))
        new_prompt = (
            self.prompt_modifier(prompt, attempt)
            if self.prompt_modifier is not None
            else prompt
        )
        return new_prompt, new_temp


def execute_with_retry(
    call: Callable[[str, float], str],
    prompt: str,
    temperature: float,
    policy: RetryPolicy | None,
) -> str:
    """Call ``call(prompt, temperature)``, retrying on :exc:`LLMError` via *policy*.

    If *policy* is ``None`` or ``policy.next()`` returns ``None``, the error
    is re-raised immediately without retrying.
    """
    attempt = 0
    while True:
        try:
            return call(prompt, temperature)
        except LLMError as exc:
            if policy is None:
                raise
            nxt = policy.next(attempt, prompt, temperature, exc)
            if nxt is None:
                raise
            prompt, temperature = nxt
            attempt += 1


class RetryingLLMCallable:
    """Wraps an LLM client with retry logic and exponential backoff."""

    def __init__(
        self,
        client: LLMCallable,
        *,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
    ) -> None:
        if max_retries < 0:
            msg = f"max_retries must be >= 0, got {max_retries}"
            raise ValueError(msg)
        if backoff_factor < 0:
            msg = f"backoff_factor must be >= 0, got {backoff_factor}"
            raise ValueError(msg)
        self._client = client
        self._max_retries = max_retries
        self._backoff_factor = backoff_factor

    def call(self, prompt: str, temperature: float) -> str:
        """Call the wrapped client with retry logic."""
        last_exception: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                return self._client.call(prompt, temperature)
            except Exception as e:
                last_exception = e
                if attempt < self._max_retries:
                    wait_time = self._backoff_factor * (2**attempt)
                    time.sleep(wait_time)

        if last_exception is not None:
            raise last_exception
        msg = "Retry logic failed without exception"
        raise RuntimeError(msg)
