"""Text normalization utilities for reducing LLM token waste.

After HTML tag stripping, text may still contain HTML entities (``&nbsp;``),
invisible Unicode characters (zero-width joiners, etc.), and excessive
whitespace.  The helpers here clean that up *at the marker level* so the
LLM receives only meaningful content while sentence offsets remain intact.
"""

from __future__ import annotations

import html
import re
import unicodedata

from txt_splitt.protocols import MarkerStrategy
from txt_splitt.types import MarkedText, Sentence

# ---------------------------------------------------------------------------
# Invisible / zero-width Unicode characters to strip
# ---------------------------------------------------------------------------

_ZERO_WIDTH_CHARS: frozenset[str] = frozenset(
    {
        "\u200b",  # ZERO WIDTH SPACE
        "\u200c",  # ZERO WIDTH NON-JOINER
        "\u200d",  # ZERO WIDTH JOINER
        "\u200e",  # LEFT-TO-RIGHT MARK
        "\u200f",  # RIGHT-TO-LEFT MARK
        "\u2060",  # WORD JOINER
        "\u2061",  # FUNCTION APPLICATION
        "\u2062",  # INVISIBLE TIMES
        "\u2063",  # INVISIBLE SEPARATOR
        "\u2064",  # INVISIBLE PLUS
        "\ufeff",  # ZERO WIDTH NO-BREAK SPACE / BOM
    }
)

# ---------------------------------------------------------------------------
# Whitespace-collapse regex
# ---------------------------------------------------------------------------

# Horizontal whitespace (regular space, tab, NBSP, and various Unicode spaces)
_HORIZONTAL_WS = re.compile(r"[ \t\xa0\u2000-\u200a\u202f\u205f\u3000]+")

# Three or more consecutive newlines (with optional surrounding spaces)
_EXCESS_NEWLINES = re.compile(r"\n{3,}")

# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def decode_html_entities(text: str) -> str:
    """Decode HTML entities, handling double-encoded cases.

    Applies ``html.unescape`` up to two passes so that sequences like
    ``&amp;nbsp;`` (which first decode to ``&nbsp;``) are fully resolved.
    """
    first = html.unescape(text)
    second = html.unescape(first)
    return second


def strip_zero_width_chars(text: str) -> str:
    """Remove zero-width and invisible Unicode characters."""
    return "".join(ch for ch in text if ch not in _ZERO_WIDTH_CHARS)


def collapse_whitespace(text: str) -> str:
    """Collapse runs of horizontal whitespace and excess blank lines.

    * Multiple spaces / NBSP / tabs -> single space
    * Three or more consecutive newlines -> two newlines
    """
    text = _HORIZONTAL_WS.sub(" ", text)
    text = _EXCESS_NEWLINES.sub("\n\n", text)
    return text


def normalize_for_llm(text: str) -> str:
    """Full normalization pipeline: entities -> zero-width -> whitespace.

    Returns the stripped result ready for LLM consumption.
    """
    text = decode_html_entities(text)
    text = strip_zero_width_chars(text)
    text = collapse_whitespace(text)
    return text.strip()


def is_content_free(text: str) -> bool:
    """Return *True* if *text* carries no alphanumeric content after normalization.

    Useful for detecting filler sentences that consist entirely of whitespace,
    HTML entities, and invisible characters.
    """
    normalized = normalize_for_llm(text)
    return not any(unicodedata.category(ch)[0] in ("L", "N") for ch in normalized)


# ---------------------------------------------------------------------------
# Decorator marker
# ---------------------------------------------------------------------------


class OptimizingMarker:
    """Wraps a ``MarkerStrategy``, normalizing sentence text before marking.

    The original ``Sentence`` offsets (``start`` / ``end``) are preserved;
    only the ``text`` field — which feeds ``MarkedText.tagged_text`` — is
    cleaned.  This means the LLM sees compact, meaningful text while
    offset-based operations (e.g. restoring original positions) are
    unaffected.

    Satisfies the ``MarkerStrategy`` protocol.

    Usage::

        marker = OptimizingMarker(BracketMarker())
        marked = marker.mark(text, sentences)
    """

    def __init__(self, inner: MarkerStrategy) -> None:
        self._inner = inner

    def mark(self, text: str, sentences: list[Sentence]) -> MarkedText:
        optimized = [
            Sentence(
                index=s.index,
                start=s.start,
                end=s.end,
                text=normalize_for_llm(s.text),
            )
            for s in sentences
        ]
        return self._inner.mark(text, optimized)
