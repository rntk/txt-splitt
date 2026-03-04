from __future__ import annotations

import json
from collections.abc import Mapping

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

    def request(
        self, method: str, path: str, body: str, headers: dict[str, str]
    ) -> None:
        del method, path, body, headers

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
