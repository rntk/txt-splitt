from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from txt_splitt.llms.llamacpp import AsyncLLamaCPP, LLamaCPP
from txt_splitt.tracer import Tracer, TracingLLMCallable


class _FakeResponse:
    def __init__(self, body: Mapping[str, object], status: int = 200) -> None:
        self.status = status
        self.reason = "OK"
        self._body = dict(body)

    def read(self) -> bytes:
        return json.dumps(self._body).encode("utf-8")


class _FakeConnection:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.payload: dict[str, Any] = {}

    def request(
        self, method: str, path: str, body: str, headers: dict[str, str]
    ) -> None:
        self.payload = json.loads(body)

    def getresponse(self) -> _FakeResponse:
        return self._response


class _LLamaCPPAdapter:
    def __init__(self, client: LLamaCPP) -> None:
        self._client = client

    def call(self, prompt: str, temperature: float) -> str:
        return self._client.call([prompt], temperature=temperature)


def test_sync_llamacpp_think_tag_reasoning_is_added_to_trace() -> None:
    client = LLamaCPP("http://localhost:8080")
    body = {
        "choices": [
            {"message": {"content": "<think>step 1\nstep 2</think>\nFinal answer"}}
        ]
    }
    client.get_connection = lambda: _FakeConnection(_FakeResponse(body))  # type: ignore[assignment,return-value]

    tracer = Tracer()
    wrapped = TracingLLMCallable(_LLamaCPPAdapter(client), tracer)
    response = wrapped.call("prompt", temperature=0.0)

    assert response == "Final answer"
    trace = tracer.format()
    assert "reasoning:" in trace
    assert "step 1" in trace
    assert "step 2" in trace


def test_async_llamacpp_sync_path_extracts_reasoning_content() -> None:
    client = AsyncLLamaCPP("http://localhost:8080")
    body = {
        "choices": [
            {
                "message": {
                    "content": "Final answer only",
                    "reasoning_content": "hidden chain-of-thought",
                }
            }
        ]
    }
    client._get_connection = lambda: _FakeConnection(_FakeResponse(body))  # type: ignore[assignment,return-value]

    response = client._sync_call(["prompt"], temperature=0.0)

    assert response == "Final answer only"


@pytest.mark.parametrize("use_async", [False, True])
def test_llamacpp_sends_requested_model(use_async: bool) -> None:
    connection = _FakeConnection(
        _FakeResponse({"choices": [{"message": {"content": "result"}}]})
    )
    if use_async:
        client = AsyncLLamaCPP("http://localhost:8080", model="candidate-model")
        client._get_connection = lambda: connection  # type: ignore[assignment,return-value]
        client._sync_call(["prompt"], temperature=0.0)
    else:
        sync_client = LLamaCPP("http://localhost:8080", model="candidate-model")
        sync_client.get_connection = lambda: connection  # type: ignore[assignment,return-value]
        sync_client.call(["prompt"])
    assert connection.payload["model"] == "candidate-model"


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("status", [400, 401, 429, 500])
def test_llamacpp_http_failure_raises(use_async: bool, status: int) -> None:
    connection = _FakeConnection(_FakeResponse({"error": "unavailable"}, status))
    error_type = ValueError if status == 400 else RuntimeError
    with pytest.raises(error_type, match=str(status)):
        if use_async:
            client = AsyncLLamaCPP("http://localhost:8080")
            client._get_connection = lambda: connection  # type: ignore[assignment,return-value]
            client._sync_call(["prompt"], temperature=0.0)
        else:
            sync_client = LLamaCPP("http://localhost:8080")
            sync_client.get_connection = lambda: connection  # type: ignore[assignment,return-value]
            sync_client.call(["prompt"])
