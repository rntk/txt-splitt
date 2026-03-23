"""Default prompt builders for sentence-oriented LLM stages."""

from txt_splitt.sentences.llm import (
    _build_single_topic_range_json_prompt,
    _build_single_topic_range_prompt,
    _build_topic_list_prompt,
    _build_topic_ranges_json_prompt,
    _build_topic_ranges_prompt,
)

build_single_topic_range_json_prompt = _build_single_topic_range_json_prompt
build_single_topic_range_prompt = _build_single_topic_range_prompt
build_topic_list_prompt = _build_topic_list_prompt
build_topic_ranges_json_prompt = _build_topic_ranges_json_prompt
build_topic_ranges_prompt = _build_topic_ranges_prompt
