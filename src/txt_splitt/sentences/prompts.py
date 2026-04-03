"""Default prompt builders for sentence-oriented LLM stages."""

from txt_splitt.sentences.llm import (
    _build_topic_ranges_json_prompt,
    _build_topic_ranges_prompt,
)

build_topic_ranges_prompt = _build_topic_ranges_prompt
build_topic_ranges_json_prompt = _build_topic_ranges_json_prompt
