import asyncio
import json
import logging
import re
from http.client import HTTPConnection, HTTPSConnection
from typing import Any, List, Union
from urllib.parse import urlparse

from txt_splitt.tracer import add_span_attribute

_THINK_TAG_RE = re.compile(
    r"<think\b[^>]*>(.*?)</think>", flags=re.DOTALL | re.IGNORECASE
)


def _extract_reasoning_and_content(response: dict[str, Any]) -> tuple[str, str]:
    """Extract reasoning text and cleaned content from an OpenAI-style response."""
    choices = response.get("choices")
    first_choice = choices[0] if isinstance(choices, list) and choices else {}
    message = first_choice.get("message") if isinstance(first_choice, dict) else {}
    if not isinstance(message, dict):
        message = {}

    raw_content = message.get("content")
    content = raw_content if isinstance(raw_content, str) else ""

    reasoning_parts: list[str] = []
    for key in ("reasoning_content", "reasoning"):
        value = message.get(key)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                reasoning_parts.append(stripped)

    for think_match in _THINK_TAG_RE.findall(content):
        stripped = think_match.strip()
        if stripped:
            reasoning_parts.append(stripped)

    reasoning = "\n\n".join(reasoning_parts).strip()
    cleaned_content = _THINK_TAG_RE.sub("", content).strip()
    return reasoning, cleaned_content


class LLamaCPP:
    ALLOWED_MODELS = ["default"]

    def __init__(self, host: str, model: str = "default"):
        u = urlparse(host)
        self.__host = u.netloc
        self.__is_https = u.scheme.lower() == "https"
        self.__model = model

    def call(
        self,
        user_msgs: List[str],
        temperature: float = 0.0,
    ) -> str:
        conn = self.get_connection()
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": user_msgs[0]}],
            "temperature": temperature,
            "cache_prompt": True,
        }
        body = json.dumps(payload)
        headers = {"Content-type": "application/json"}
        conn.request("POST", "/v1/chat/completions", body, headers)
        res = conn.getresponse()
        resp_body = res.read()
        resp_body_text = resp_body.decode("utf-8", errors="replace")
        # logging.info("server response: %s", resp_body)
        if res.status != 200:
            err_msg = f"{res.status} - {res.reason} - {resp_body_text}"
            logging.error(err_msg)
            # Raise exception for 400 status (request too large)
            if res.status == 400:
                raise ValueError(f"Request too large (400): {err_msg}")
            return err_msg
        resp = json.loads(resp_body)

        reasoning, content = _extract_reasoning_and_content(resp)
        if reasoning:
            add_span_attribute("reasoning", reasoning)
        return content

    def get_connection(self) -> Union[HTTPConnection, HTTPSConnection]:
        if self.__is_https:
            return HTTPSConnection(self.__host)
        else:
            return HTTPConnection(self.__host)


class AsyncLLamaCPP:
    """Async version of LLamaCPP client."""

    ALLOWED_MODELS = ["default"]

    def __init__(self, host: str, model: str = "default"):
        u = urlparse(host)
        self.__host = u.netloc
        self.__is_https = u.scheme.lower() == "https"
        self.__model = model

    async def call(
        self,
        user_msgs: List[str],
        temperature: float = 0.0,
    ) -> str:
        return await asyncio.to_thread(self._sync_call, user_msgs, temperature)

    def _sync_call(self, user_msgs: List[str], temperature: float) -> str:
        conn = self._get_connection()
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": user_msgs[0]}],
            "temperature": temperature,
            "cache_prompt": True,
        }
        body = json.dumps(payload)
        headers = {"Content-type": "application/json"}
        conn.request("POST", "/v1/chat/completions", body, headers)
        res = conn.getresponse()
        resp_body = res.read()
        resp_body_text = resp_body.decode("utf-8", errors="replace")
        if res.status != 200:
            err_msg = f"{res.status} - {res.reason} - {resp_body_text}"
            logging.error(err_msg)
            if res.status == 400:
                raise ValueError(f"Request too large (400): {err_msg}")
            return err_msg
        resp = json.loads(resp_body)

        reasoning, content = _extract_reasoning_and_content(resp)
        if reasoning:
            logging.info(f"Extracted reasoning ({len(reasoning)} chars)")
            add_span_attribute("reasoning", reasoning)
        else:
            logging.debug("No reasoning content found in LLM response")
        return content

    def _get_connection(self) -> Union[HTTPConnection, HTTPSConnection]:
        if self.__is_https:
            return HTTPSConnection(self.__host)
        else:
            return HTTPConnection(self.__host)
