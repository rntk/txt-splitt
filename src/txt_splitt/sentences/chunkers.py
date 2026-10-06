"""Chunking strategies for splitting MarkedText into smaller pieces."""

import math
import re

from txt_splitt.sentences.types import MarkedText

_DEFAULT_MAX_CHARS = 12_000
_DEFAULT_OVERLAP_CHARS = 500
_MARKER_LINE_RE = re.compile(r"^\{\d+\}(?:\s|$)")
_MARKER_PREFIX_RE = re.compile(r"^(\{\d+\}(?:\s|$))")


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------


def _text_len(lines: list[str]) -> int:
    """Return the character length of *lines* joined by newlines."""
    if not lines:
        return 0
    return sum(len(ln) for ln in lines) + len(lines) - 1


def _is_marker_line(line: str) -> bool:
    """Return True when *line* starts with a sentence number marker."""
    return _MARKER_LINE_RE.match(line) is not None


def _split_oversized_lines(lines: list[str], max_chars: int) -> list[str]:
    """Bound long lines, repeating their marker on each payload fragment.

    Continuation lines inherit the most recent marker when fragmented.
    Payload characters are preserved, including unbroken encoded data.
    """
    bounded: list[str] = []
    prefix = ""
    for line in lines:
        match = _MARKER_PREFIX_RE.match(line)
        if match is not None:
            prefix = match.group(1)
        if len(line) <= max_chars:
            bounded.append(line)
            continue
        payload = line[len(prefix) :] if match is not None else line
        payload_budget = max_chars - len(prefix)
        if payload_budget <= 0:
            raise ValueError("max_chars must leave room for a sentence marker and text")
        bounded.extend(
            prefix + payload[start : start + payload_budget]
            for start in range(0, len(payload), payload_budget)
        )
    return bounded


def _select_overlap(new_lines: list[str], overlap_chars: int) -> list[str]:
    """Pick trailing lines from *new_lines* totalling >= *overlap_chars*.

    If the overlap would start on a non-marker continuation line,
    extend it backwards to include the corresponding marker line when
    that marker is present in *new_lines*.
    """
    if overlap_chars == 0:
        return []
    selected: list[str] = []
    acc = 0
    first_idx = len(new_lines)
    for line in reversed(new_lines):
        acc += len(line) + (1 if selected else 0)
        selected.append(line)
        first_idx -= 1
        if acc >= overlap_chars:
            break
    selected.reverse()

    while selected and not _is_marker_line(selected[0]) and first_idx > 0:
        first_idx -= 1
        selected.insert(0, new_lines[first_idx])

    return selected


class OverlapChunker:
    """Split MarkedText into balanced chunks with overlapping context.

    Each chunk's ``tagged_text`` will not exceed *max_chars* and splits
    happen on line boundaries where possible. Oversized lines are split
    into fragments with the same sentence marker. The chunker estimates the optimal
    number of chunks up-front and distributes lines so that all chunks
    are approximately the same size.

    Each chunk (after the first) is prefixed with lines carried over
    from the *end* of the previous chunk.  The amount of overlap is
    controlled by *overlap_chars*.  The overlap text counts toward the
    *max_chars* budget of the receiving chunk.
    Overlap is reduced or omitted when necessary to fit new content.

    Chunk boundaries can fall on continuation lines, so chunks are not
    guaranteed to start with a sentence marker. Marker repetition applies
    to oversized-line fragments; overlap selection includes the preceding
    marker only when it is available in the previous chunk's new content.
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

        lines = _split_oversized_lines(tagged_text.split("\n"), self._max_chars)
        total_len = _text_len(lines)

        # Budget available for *new* content per chunk (overlap eats into
        # max_chars, so the content portion is the remainder). The first
        # chunk has no overlap, so it can hold max_chars of new content;
        # every subsequent chunk can hold content_budget of new content.
        content_budget = self._max_chars - self._overlap_chars
        if content_budget <= 0:
            content_budget = 1  # degenerate but safe

        # Minimum chunks a greedy packing would produce. We already
        # returned above when the whole text fits in one chunk.
        num_chunks = 1 + math.ceil((total_len - self._max_chars) / content_budget)

        # Balance by equal tagged_text size across chunks. Summed over N
        # chunks, tagged_text totals total_len + (N-1)*overlap_chars
        # (overlap is counted once per chunk after the first).
        target_size = (total_len + (num_chunks - 1) * self._overlap_chars) / num_chunks

        chunks: list[MarkedText] = []
        overlap_lines: list[str] = []
        i = 0
        chunks_remaining = num_chunks

        while i < len(lines):
            # Whole-line overlap can exceed the requested overlap budget.
            # Drop leading sentences until the next new line fits. When
            # trimming, skip continuation lines after each removed marker.
            overlap_size = _text_len(overlap_lines)
            overlap_start = 0
            while (
                overlap_start < len(overlap_lines)
                and overlap_size + 1 + len(lines[i]) > self._max_chars
            ):
                overlap_size -= len(overlap_lines[overlap_start]) + 1
                overlap_start += 1
                while overlap_start < len(overlap_lines) and not _is_marker_line(
                    overlap_lines[overlap_start]
                ):
                    overlap_size -= len(overlap_lines[overlap_start]) + 1
                    overlap_start += 1
            overlap_lines = overlap_lines[overlap_start:]
            current_lines = list(overlap_lines)
            current_chars = _text_len(current_lines)
            new_count = 0
            chunks_remaining -= 1

            while i < len(lines):
                line = lines[i]
                added = len(line) + (1 if current_lines else 0)

                # Hard cap: never exceed max_chars.
                if current_lines and current_chars + added > self._max_chars:
                    break

                current_lines.append(line)
                current_chars += added
                i += 1
                new_count += 1

                # Soft target: close the chunk once its tagged_text
                # reaches the target, but only if more chunks remain.
                if (
                    chunks_remaining > 0
                    and current_chars >= target_size
                    and i < len(lines)
                ):
                    break

            chunks.append(
                MarkedText(
                    tagged_text="\n".join(current_lines),
                    sentence_count=len(current_lines),
                )
            )

            new_lines = current_lines[len(current_lines) - new_count :]
            overlap_lines = _select_overlap(new_lines, self._overlap_chars)

        return chunks if chunks else [marked_text]
