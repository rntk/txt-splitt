"""Unit tests for OpenAI SDK client wrapper."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import pytest

from txt_splitt.llms.openai import OpenAIClient


class _FakeMessage:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str | None) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str | None) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, content: str | None) -> None:
        self._content = content
        self.last_kwargs: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> _FakeResponse:
        self.last_kwargs = kwargs
        return _FakeResponse(self._content)


class _FakeOpenAI:
    def __init__(self, completions: _FakeCompletions, **kwargs: Any) -> None:
        self.chat = SimpleNamespace(completions=completions)
        self.init_kwargs = kwargs


class _FakeOpenAIFactory:
    def __init__(self, content: str | None) -> None:
        self.completions = _FakeCompletions(content)
        self.instances: list[_FakeOpenAI] = []

    def __call__(self, **kwargs: Any) -> _FakeOpenAI:
        instance = _FakeOpenAI(self.completions, **kwargs)
        self.instances.append(instance)
        return instance


def test_call_uses_openai_sdk_and_forwards_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _FakeOpenAIFactory("Technology>AI: 0-2")
    fake_module = SimpleNamespace(OpenAI=factory)

    monkeypatch.setattr(importlib, "import_module", lambda _: fake_module)

    client = OpenAIClient(model="gpt-4o-mini", api_key="key")
    result = client.call("prompt text", temperature=0.2)

    assert result == "Technology>AI: 0-2"
    assert factory.instances[0].init_kwargs["api_key"] == "key"
    assert factory.completions.last_kwargs == {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "prompt text"}],
        "temperature": 0.2,
    }


def test_call_uses_first_user_message_from_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _FakeOpenAIFactory("ok")
    fake_module = SimpleNamespace(OpenAI=factory)
    monkeypatch.setattr(importlib, "import_module", lambda _: fake_module)

    client = OpenAIClient()
    result = client.call(["first prompt", "second prompt"])

    assert result == "ok"
    assert factory.completions.last_kwargs is not None
    assert factory.completions.last_kwargs["messages"][0]["content"] == "first prompt"


def test_call_removes_think_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = _FakeOpenAIFactory("<think>hidden</think>visible")
    fake_module = SimpleNamespace(OpenAI=factory)
    monkeypatch.setattr(importlib, "import_module", lambda _: fake_module)

    client = OpenAIClient()
    assert client.call("prompt") == "visible"


def test_call_empty_user_messages_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = _FakeOpenAIFactory("unused")
    fake_module = SimpleNamespace(OpenAI=factory)
    monkeypatch.setattr(importlib, "import_module", lambda _: fake_module)

    client = OpenAIClient()
    with pytest.raises(ValueError, match="user_msgs cannot be empty"):
        client.call([])


def test_call_missing_content_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = _FakeOpenAIFactory(None)
    fake_module = SimpleNamespace(OpenAI=factory)
    monkeypatch.setattr(importlib, "import_module", lambda _: fake_module)

    client = OpenAIClient()
    with pytest.raises(ValueError, match="did not include message content"):
        client.call("prompt")


def test_missing_openai_dependency_raises_helpful_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(_: str) -> Any:
        raise ModuleNotFoundError("No module named 'openai'")

    monkeypatch.setattr(importlib, "import_module", _raise)

    with pytest.raises(ImportError, match="pip install openai"):
        OpenAIClient()
