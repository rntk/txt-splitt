"""Default prompt builders for keyword extraction."""

from txt_splitt.keywords.llm import _PROMPT_TEMPLATE


def build_keyword_prompt(tagged_text: str, max_keywords: int) -> str:
    """Build the default keyword extraction prompt."""
    return _PROMPT_TEMPLATE.format(max_keywords=max_keywords, text=tagged_text)
