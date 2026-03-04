import asyncio
import json
import logging
import re
from http.client import HTTPConnection, HTTPSConnection
from typing import List, Union
from urllib.parse import urlparse

from txt_splitt.tracer import add_span_attribute


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

        full_content = resp["choices"][0]["message"]["content"]

        # Extract <think></think> tags and their content for tracing
        reasoning_match = re.search(r"<think>(.*?)</think>", full_content, flags=re.DOTALL)
        if reasoning_match:
            add_span_attribute("reasoning", reasoning_match.group(1).strip())

        # Remove <think></think> tags and their content from the final answer
        content = re.sub(r"<think>.*?</think>", "", full_content, flags=re.DOTALL).strip()
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
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._sync_call, user_msgs, temperature
        )

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

        full_content = resp["choices"][0]["message"]["content"]

        # Extract <think></think> tags and their content for tracing
        reasoning_match = re.search(r"<think>(.*?)</think>", full_content, flags=re.DOTALL)
        if reasoning_match:
            add_span_attribute("reasoning", reasoning_match.group(1).strip())

        content = re.sub(r"<think>.*?</think>", "", full_content, flags=re.DOTALL).strip()
        return content

    def _get_connection(self) -> Union[HTTPConnection, HTTPSConnection]:
        if self.__is_https:
            return HTTPSConnection(self.__host)
        else:
            return HTTPConnection(self.__host)
