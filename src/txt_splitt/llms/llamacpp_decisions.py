"""Client for the llama.cpp decision ("System One") API.

Port of ``example/core/llm/decisionClient.js``. A decision request carries a
``state`` and named ``questions``; the server answers each question with a
choice and a probability per option instead of free text.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from http.client import HTTPConnection, HTTPException, HTTPSConnection
from typing import Any
from urllib.parse import urlparse

from txt_splitt.errors import LLMError

# 529 is the TypeSafe API's "overloaded" status.
TRANSIENT_STATUSES = frozenset({429, 502, 503, 504, 529})
DECISION_MAX_ATTEMPTS = 3


class DecisionHTTPError(LLMError):
    """Non-200 response from the decision server."""

    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"llama.cpp decision API returned HTTP {status}: {body}")
        self.status = status
        self.body = body


def choice(instructions: str, criteria: dict[str, Any]) -> dict[str, Any]:
    """Build a choice question from named option descriptions."""
    if not criteria or any(not name for name in criteria):
        msg = "Choice requires nonempty option names"
        raise ValueError(msg)
    return {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}


class LLamaCPPDecisions:
    """Sync decision client with bounded retries on transient failures."""

    def __init__(
        self,
        host: str,
        *,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        max_attempts: int = DECISION_MAX_ATTEMPTS,
    ) -> None:
        if timeout <= 0:
            msg = "timeout must be a positive number of seconds"
            raise ValueError(msg)
        if max_attempts < 1:
            msg = "max_attempts must be >= 1"
            raise ValueError(msg)
        parsed = urlparse(host.strip())
        if not parsed.netloc:
            msg = f"Invalid decision host: {host!r}"
            raise ValueError(msg)
        self._netloc = parsed.netloc
        self._is_https = parsed.scheme.lower() == "https"
        path = parsed.path.rstrip("/")
        self._base_path = path if path.endswith("/v1") else f"{path}/v1"
        self._model = model
        self._api_key = api_key
        self._timeout = timeout
        self._max_attempts = max_attempts

    def decide(
        self, state: Any, questions: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """Answer named questions; the full server response is returned."""
        if not questions:
            msg = "questions must be a nonempty mapping"
            raise ValueError(msg)
        payload: dict[str, Any] = {"state": state, "questions": questions}
        if self._model:
            payload["model"] = self._model
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = self._post("systemone", payload)
            except (DecisionHTTPError, OSError, HTTPException) as exc:
                transient = not isinstance(exc, DecisionHTTPError) or (
                    exc.status in TRANSIENT_STATUSES
                )
                if not transient or attempt >= self._max_attempts:
                    raise
                time.sleep(
                    min(60.0, 0.5 * 2 ** (attempt - 1) * (0.5 + random.random()))
                )
                continue
            if not isinstance(response.get("answers"), dict):
                raise LLMError("Decision server response has no 'answers' object")
            return response
        raise AssertionError("unreachable")

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        connection: HTTPConnection = (
            HTTPSConnection(self._netloc, timeout=self._timeout)
            if self._is_https
            else HTTPConnection(self._netloc, timeout=self._timeout)
        )
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            connection.request(
                "POST",
                f"{self._base_path}/{endpoint}",
                json.dumps(payload),
                headers,
            )
            response = connection.getresponse()
            body = response.read().decode("utf-8", errors="replace")
        finally:
            connection.close()
        if response.status != 200:
            raise DecisionHTTPError(response.status, body)
        try:
            result = json.loads(body)
        except json.JSONDecodeError as exc:
            raise LLMError("Decision server returned invalid JSON") from exc
        if not isinstance(result, dict):
            raise LLMError("Decision server response must be a JSON object")
        return result


class AsyncLLamaCPPDecisions:
    """Async wrapper running the sync decision client in a worker thread."""

    def __init__(self, client: LLamaCPPDecisions) -> None:
        self._client = client

    async def decide(
        self, state: Any, questions: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        return await asyncio.to_thread(self._client.decide, state, questions)
