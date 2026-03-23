"""Keyword offset restorers."""

from txt_splitt.keywords.results import build_result, remap_keywords, remap_words
from txt_splitt.keywords.types import KeywordResult
from txt_splitt.types import OffsetMapping


class MappingOffsetRestorer:
    """Restore clean-text keyword offsets back to original text offsets."""

    def restore(self, result: KeywordResult, mapping: OffsetMapping) -> KeywordResult:
        return build_result(
            remap_keywords(list(result.keywords), mapping),
            remap_words(list(result.words), mapping),
        )
