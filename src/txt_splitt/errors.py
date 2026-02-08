"""Exception hierarchy for the text splitter pipeline."""


class SplitterError(Exception):
    """Base exception for all splitter errors."""


class SentenceSplitError(SplitterError):
    """Raised when sentence splitting (stage 1) fails."""


class MarkerError(SplitterError):
    """Raised when marker formatting (stage 2) fails."""


class LLMError(SplitterError):
    """Raised when the LLM query (stage 3) fails."""


class ParseError(SplitterError):
    """Raised when response parsing (stage 4) fails."""


class GapError(SplitterError):
    """Raised when gap validation (stage 5) fails."""


class EnhancerError(SplitterError):
    """Raised when the enhancer stage (stage 6) fails."""
