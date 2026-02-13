"""Tests for retry wrapper."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from txt_splitt.retry import RetryingLLMCallable


class TestRetryingLLMCallable:
    def test_successful_call_no_retry(self) -> None:
        client = Mock()
        client.call.return_value = "success"
        wrapper = RetryingLLMCallable(client, max_retries=3)

        result = wrapper.call("prompt", 0.5)

        assert result == "success"
        assert client.call.call_count == 1

    def test_retries_on_failure_then_succeeds(self) -> None:
        client = Mock()
        client.call.side_effect = [
            RuntimeError("fail1"),
            RuntimeError("fail2"),
            "success",
        ]
        wrapper = RetryingLLMCallable(client, max_retries=3, backoff_factor=0.01)

        result = wrapper.call("prompt", 0.5)

        assert result == "success"
        assert client.call.call_count == 3

    def test_exhausts_retries_and_raises(self) -> None:
        client = Mock()
        client.call.side_effect = RuntimeError("persistent failure")
        wrapper = RetryingLLMCallable(client, max_retries=2, backoff_factor=0.01)

        with pytest.raises(RuntimeError, match="persistent failure"):
            wrapper.call("prompt", 0.5)

        assert client.call.call_count == 3  # initial + 2 retries

    def test_zero_retries_fails_immediately(self) -> None:
        client = Mock()
        client.call.side_effect = RuntimeError("fail")
        wrapper = RetryingLLMCallable(client, max_retries=0)

        with pytest.raises(RuntimeError, match="fail"):
            wrapper.call("prompt", 0.5)

        assert client.call.call_count == 1

    def test_invalid_max_retries_raises(self) -> None:
        client = Mock()
        with pytest.raises(ValueError, match="max_retries must be >= 0"):
            RetryingLLMCallable(client, max_retries=-1)

    def test_invalid_backoff_factor_raises(self) -> None:
        client = Mock()
        with pytest.raises(ValueError, match="backoff_factor must be >= 0"):
            RetryingLLMCallable(client, backoff_factor=-1.0)

    def test_passes_arguments_correctly(self) -> None:
        client = Mock()
        client.call.return_value = "result"
        wrapper = RetryingLLMCallable(client)

        wrapper.call("test prompt", 0.7)

        client.call.assert_called_once_with("test prompt", 0.7)
