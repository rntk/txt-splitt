"""Marker strategy implementations."""

from txt_splitt._types import MarkedText, Sentence


class BracketMarker:
    """Format sentences with {N} bracket markers."""

    def mark(self, text: str, sentences: list[Sentence]) -> MarkedText:
        rows: list[str] = [text[s.start : s.end] for s in sentences]

        if not rows and text.strip():
            rows.append(text)

        formatted = [f"{{{i}}} {row}" for i, row in enumerate(rows)]

        return MarkedText(
            tagged_text="\n".join(formatted),
            sentence_count=len(rows),
        )
