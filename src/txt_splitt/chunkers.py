"""Chunking strategies for splitting MarkedText into smaller pieces."""

from txt_splitt.types import MarkedText

_DEFAULT_MAX_CHARS = 12_000


class SizeBasedChunker:
    """Split MarkedText into chunks where each chunk's tagged_text
    does not exceed *max_chars*.

    Splits along line boundaries only.  Original sentence numbers
    (embedded in ``{N}`` markers) are preserved.  A single line that
    exceeds *max_chars* is kept as its own chunk (never split mid-line).
    """

    def __init__(self, *, max_chars: int = _DEFAULT_MAX_CHARS) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        self._max_chars = max_chars

    def chunk(self, marked_text: MarkedText) -> list[MarkedText]:
        tagged_text = marked_text.tagged_text

        if len(tagged_text) <= self._max_chars:
            return [marked_text]

        lines = tagged_text.split("\n")
        chunks: list[MarkedText] = []
        current_lines: list[str] = []
        current_chars = 0

        for line in lines:
            line_len = len(line)
            added_chars = line_len + (1 if current_lines else 0)

            if current_lines and current_chars + added_chars > self._max_chars:
                chunks.append(
                    MarkedText(
                        tagged_text="\n".join(current_lines),
                        sentence_count=len(current_lines),
                    )
                )
                current_lines = [line]
                current_chars = line_len
            else:
                current_lines.append(line)
                current_chars += added_chars

        if current_lines:
            chunks.append(
                MarkedText(
                    tagged_text="\n".join(current_lines),
                    sentence_count=len(current_lines),
                )
            )

        return chunks if chunks else [marked_text]
