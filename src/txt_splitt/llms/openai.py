"""OpenAI SDK-backed LLM client."""

from __future__ import annotations

import importlib
import re
from typing import Any, Literal

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]


class OpenAIClient:
    """Minimal OpenAI SDK wrapper compatible with txt_splitt LLMCallable."""

    ALLOWED_MODELS = ["default"]

    def __init__(
        self,
        *,
        model: str = "gpt-5-nano",
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> None:
        self._model = model
        self._reasoning_effort = reasoning_effort
        openai_module = _import_openai_module()

        client_kwargs: dict[str, object] = {}
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        if organization is not None:
            client_kwargs["organization"] = organization

        self._client: Any = openai_module.OpenAI(**client_kwargs)

    def call(
        self,
        user_msgs: str | list[str],
        temperature: float = 0.0,
    ) -> str:
        prompt = _extract_prompt(user_msgs)
        payload: dict[str, object] = {
            "model": self._model,
            "input": prompt,
            "temperature": temperature,
        }
        if self._reasoning_effort is not None:
            payload["reasoning"] = {"effort": self._reasoning_effort}

        try:
            response = self._client.responses.create(**payload)
        except Exception as exc:
            # Some models (e.g., gpt-5-nano) reject `temperature`.
            if _is_unsupported_temperature_error(exc):
                payload_without_temperature = dict(payload)
                payload_without_temperature.pop("temperature", None)
                response = self._client.responses.create(**payload_without_temperature)
            else:
                raise
        content = _extract_output_text(response)

        # Remove <think></think> tags and their content
        return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)


def _extract_prompt(user_msgs: str | list[str]) -> str:
    if isinstance(user_msgs, str):
        return user_msgs

    if not user_msgs:
        raise ValueError("user_msgs cannot be empty")

    return user_msgs[0]


def _import_openai_module() -> Any:
    try:
        return importlib.import_module("openai")
    except ModuleNotFoundError as exc:
        raise ImportError(
            "openai package is required for OpenAIClient. Install it with "
            "`pip install openai`."
        ) from exc


def _extract_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text:
        return output_text

    output_items = getattr(response, "output", None)
    if isinstance(output_items, list):
        text_parts: list[str] = []
        for item in output_items:
            contents = getattr(item, "content", None)
            if isinstance(contents, list):
                for part in contents:
                    if getattr(part, "type", None) == "output_text":
                        text = getattr(part, "text", None)
                        if isinstance(text, str) and text:
                            text_parts.append(text)
        if text_parts:
            return "".join(text_parts)

    raise ValueError("OpenAI response did not include output text")


def _is_unsupported_temperature_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "unsupported parameter" in msg and "temperature" in msg
