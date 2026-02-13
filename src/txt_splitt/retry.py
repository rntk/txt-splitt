"""Retry wrapper for LLM calls."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from txt_splitt.protocols import LLMCallable


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
