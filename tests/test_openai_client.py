"""Unit tests for OpenAI SDK client wrapper."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import pytest

from txt_splitt.llms.openai import OpenAIClient


class _FakePart:
    def __init__(self, part_type: str, text: str | None) -> None:
        self.type = part_type
        self.text = text


class _FakeOutputItem:
    def __init__(self, parts: list[_FakePart]) -> None:
        self.content = parts


class _FakeResponse:
    def __init__(
        self,
        *,
        output_text: str | None = None,
        output: list[_FakeOutputItem] | None = None,
    ) -> None:
        self.output_text = output_text
        self.output = output if output is not None else []


class _FakeResponses:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.last_kwargs: dict[str, Any] | None = None
        self.calls: list[dict[str, Any]] = []
        self.raise_once: Exception | None = None

    def create(self, **kwargs: Any) -> _FakeResponse:
        self.last_kwargs = kwargs
        self.calls.append(kwargs)
        if self.raise_once is not None:
            err = self.raise_once
            self.raise_once = None
            raise err
        return self._response


class _FakeOpenAI:
    def __init__(self, responses: _FakeResponses, **kwargs: Any) -> None:
        self.responses = responses
        self.init_kwargs = kwargs


class _FakeOpenAIFactory:
    def __init__(self, response: _FakeResponse) -> None:
        self.responses = _FakeResponses(response)
        self.instances: list[_FakeOpenAI] = []

    def __call__(self, **kwargs: Any) -> _FakeOpenAI:
        instance = _FakeOpenAI(self.responses, **kwargs)
        self.instances.append(instance)
        return instance


def test_call_uses_responses_api_and_forwards_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _FakeOpenAIFactory(_FakeResponse(output_text="Technology>AI: 0-2"))
    fake_module = SimpleNamespace(OpenAI=factory)
    monkeypatch.setattr(importlib, "import_module", lambda _: fake_module)

    client = OpenAIClient(model="gpt-5-mini", api_key="key")
    result = client.call("prompt text", temperature=0.2)

    assert result == "Technology>AI: 0-2"
    assert factory.instances[0].init_kwargs["api_key"] == "key"
    assert factory.responses.last_kwargs == {
        "model": "gpt-5-mini",
        "input": "prompt text",
        "temperature": 0.2,
    }


def test_call_forwards_reasoning_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = _FakeOpenAIFactory(_FakeResponse(output_text="ok"))
    fake_module = SimpleNamespace(OpenAI=factory)
    monkeypatch.setattr(importlib, "import_module", lambda _: fake_module)

    client = OpenAIClient(reasoning_effort="high")
    client.call("prompt")

    assert factory.responses.last_kwargs is not None
    assert factory.responses.last_kwargs["reasoning"] == {"effort": "high"}


def test_call_retries_without_temperature_for_unsupported_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _FakeOpenAIFactory(_FakeResponse(output_text="ok"))
    factory.responses.raise_once = Exception(
        "Error code: 400 - {'error': {'message': "
        "\"Unsupported parameter: 'temperature' is not supported with this model.\"}}"
    )
    fake_module = SimpleNamespace(OpenAI=factory)
    monkeypatch.setattr(importlib, "import_module", lambda _: fake_module)

    client = OpenAIClient(model="gpt-5-nano")
    result = client.call("prompt", temperature=0.2)

    assert result == "ok"
    assert len(factory.responses.calls) == 2
    assert "temperature" in factory.responses.calls[0]
    assert "temperature" not in factory.responses.calls[1]


def test_call_uses_first_user_message_from_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _FakeOpenAIFactory(_FakeResponse(output_text="ok"))
    fake_module = SimpleNamespace(OpenAI=factory)
    monkeypatch.setattr(importlib, "import_module", lambda _: fake_module)

    client = OpenAIClient()
    result = client.call(["first prompt", "second prompt"])

    assert result == "ok"
    assert factory.responses.last_kwargs is not None
    assert factory.responses.last_kwargs["input"] == "first prompt"


def test_call_falls_back_to_output_content(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _FakeResponse(
        output_text=None,
        output=[
            _FakeOutputItem([_FakePart("output_text", "part1 "), _FakePart("x", "x")]),
            _FakeOutputItem([_FakePart("output_text", "part2")]),
        ],
    )
    factory = _FakeOpenAIFactory(response)
    fake_module = SimpleNamespace(OpenAI=factory)
    monkeypatch.setattr(importlib, "import_module", lambda _: fake_module)

    client = OpenAIClient()
    assert client.call("prompt") == "part1 part2"


def test_call_removes_think_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = _FakeOpenAIFactory(_FakeResponse(output_text="<think>hidden</think>ok"))
    fake_module = SimpleNamespace(OpenAI=factory)
    monkeypatch.setattr(importlib, "import_module", lambda _: fake_module)

    client = OpenAIClient()
    assert client.call("prompt") == "ok"


def test_call_empty_user_messages_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = _FakeOpenAIFactory(_FakeResponse(output_text="unused"))
    fake_module = SimpleNamespace(OpenAI=factory)
    monkeypatch.setattr(importlib, "import_module", lambda _: fake_module)

    client = OpenAIClient()
    with pytest.raises(ValueError, match="user_msgs cannot be empty"):
        client.call([])


def test_call_missing_output_text_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = _FakeOpenAIFactory(_FakeResponse(output_text=None, output=[]))
    fake_module = SimpleNamespace(OpenAI=factory)
    monkeypatch.setattr(importlib, "import_module", lambda _: fake_module)

    client = OpenAIClient()
    with pytest.raises(ValueError, match="did not include output text"):
        client.call("prompt")


def test_missing_openai_dependency_raises_helpful_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(_: str) -> Any:
        raise ModuleNotFoundError("No module named 'openai'")

    monkeypatch.setattr(importlib, "import_module", _raise)

    with pytest.raises(ImportError, match="pip install openai"):
        OpenAIClient()
