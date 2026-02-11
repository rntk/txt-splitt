"""Chunking strategies for splitting MarkedText into smaller pieces."""

from txt_splitt.types import MarkedText

_DEFAULT_MAX_CHARS = 12_000
_DEFAULT_OVERLAP_CHARS = 500


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


class OverlapChunker:
    """Split MarkedText into chunks with overlapping context.

    Like :class:`SizeBasedChunker`, each chunk's ``tagged_text`` will not
    exceed *max_chars* and splits happen on line boundaries only.

    Additionally, each chunk (after the first) is prefixed with lines
    carried over from the *end* of the previous chunk.  The amount of
    overlap is controlled by *overlap_chars*: lines are collected from
    the tail of the previous chunk's **new** content until their combined
    character length reaches at least *overlap_chars*.  The overlap text
    counts toward the *max_chars* budget of the receiving chunk.
    """

    def __init__(
        self,
        *,
        max_chars: int = _DEFAULT_MAX_CHARS,
        overlap_chars: int = _DEFAULT_OVERLAP_CHARS,
    ) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        if overlap_chars < 0:
            raise ValueError("overlap_chars must be non-negative")
        if overlap_chars >= max_chars:
            raise ValueError("overlap_chars must be less than max_chars")
        self._max_chars = max_chars
        self._overlap_chars = overlap_chars

    def chunk(self, marked_text: MarkedText) -> list[MarkedText]:
        tagged_text = marked_text.tagged_text

        if len(tagged_text) <= self._max_chars:
            return [marked_text]

        lines = tagged_text.split("\n")
        chunks: list[MarkedText] = []
        overlap_lines: list[str] = []
        i = 0  # index of the next unprocessed line

        while i < len(lines):
            # Start a chunk with overlap from the previous iteration.
            current_lines = list(overlap_lines)
            current_chars = self._text_len(current_lines)

            # Add new lines until the budget is exhausted.
            new_count = 0
            while i < len(lines):
                line = lines[i]
                added = len(line) + (1 if current_lines else 0)
                if current_lines and current_chars + added > self._max_chars:
                    break
                current_lines.append(line)
                current_chars += added
                i += 1
                new_count += 1

            # Guarantee progress: if no new line fit, force-add one.
            if new_count == 0:
                current_lines.append(lines[i])
                i += 1
                new_count = 1

            chunks.append(
                MarkedText(
                    tagged_text="\n".join(current_lines),
                    sentence_count=len(current_lines),
                )
            )

            # Determine overlap lines for the next chunk by walking
            # backwards through this chunk's new lines.
            new_lines = current_lines[len(current_lines) - new_count :]
            overlap_lines = self._select_overlap(new_lines)

        return chunks if chunks else [marked_text]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _text_len(lines: list[str]) -> int:
        """Return the character length of *lines* joined by newlines."""
        if not lines:
            return 0
        return sum(len(ln) for ln in lines) + len(lines) - 1

    def _select_overlap(self, new_lines: list[str]) -> list[str]:
        """Pick trailing lines from *new_lines* totalling ≥ overlap_chars."""
        if self._overlap_chars == 0:
            return []
        selected: list[str] = []
        acc = 0
        for line in reversed(new_lines):
            acc += len(line) + (1 if selected else 0)
            selected.append(line)
            if acc >= self._overlap_chars:
                break
        selected.reverse()
        return selected
