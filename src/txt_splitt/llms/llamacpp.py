import json
import logging
import re
from http.client import HTTPConnection, HTTPSConnection
from typing import List, Union
from urllib.parse import urlparse


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

        content = resp["choices"][0]["message"]["content"]
        # Remove <think></think> tags and their content
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        return content

    def get_connection(self) -> Union[HTTPConnection, HTTPSConnection]:
        if self.__is_https:
            return HTTPSConnection(self.__host)
        else:
            return HTTPConnection(self.__host)
