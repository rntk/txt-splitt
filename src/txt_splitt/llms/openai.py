"""OpenAI SDK-backed LLM client."""

from __future__ import annotations

import importlib
import re
from typing import Any


class OpenAIClient:
    """Minimal OpenAI SDK wrapper compatible with txt_splitt LLMCallable."""

    ALLOWED_MODELS = ["default"]

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
    ) -> None:
        self._model = model
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
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )

        choices = getattr(response, "choices", None)
        if not choices:
            raise ValueError("OpenAI response did not include any choices")

        content = choices[0].message.content
        if content is None:
            raise ValueError("OpenAI response did not include message content")

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
