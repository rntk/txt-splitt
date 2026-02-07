"""Sentence splitting implementations."""

import re

from txt_splitt.types import Sentence


class RegexSentenceSplitter:
    """Split text into sentences using regex boundary detection.

    Splits on:
    - Punctuation ([.!?]) followed by whitespace and an uppercase letter
    - One or more newlines (block boundaries)
    """

    def split(self, text: str) -> list[Sentence]:
        if not text or not text.strip():
            return []

        boundaries = list(re.finditer(r"((?<=[.!?])\s+(?=[A-ZА-Я]))|(\n+)", text))

        result: list[Sentence] = []
        start = 0
        index = 0

        for match in boundaries:
            end = match.start()
            s_start, s_end = _trim_whitespace(text, start, end)
            if s_start < s_end:
                result.append(
                    Sentence(
                        index=index,
                        start=s_start,
                        end=s_end,
                        text=text[s_start:s_end],
                    )
                )
                index += 1
            start = match.end()

        # Handle the last segment
        s_start, s_end = _trim_whitespace(text, start, len(text))
        if s_start < s_end:
            result.append(
                Sentence(
                    index=index,
                    start=s_start,
                    end=s_end,
                    text=text[s_start:s_end],
                )
            )

        return result


def _trim_whitespace(text: str, start: int, end: int) -> tuple[int, int]:
    """Trim leading and trailing whitespace from a text span."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end
