"""Enhancer implementations for refining group boundaries."""

import logging

from txt_splitt.errors import EnhancerError
from txt_splitt.protocols import LLMCallable
from txt_splitt.types import Sentence, SentenceGroup, SentenceRange

_CONTEXT_SIZE = 3


class ShortSentenceEnhancer:
    """Reassign short sentences at group boundaries using LLM judgment.

    For each boundary between adjacent groups, checks if the bordering
    sentence(s) are below a character-length threshold. If so, queries
    the LLM to decide whether the sentence belongs with the previous
    topic or the next topic, and adjusts ranges accordingly.
    """

    def __init__(
        self,
        client: LLMCallable,
        *,
        min_length: int = 40,
        temperature: float = 0.0,
    ) -> None:
        self._client = client
        self._min_length = min_length
        self._temperature = temperature

    def enhance(
        self,
        groups: list[SentenceGroup],
        sentences: list[Sentence],
    ) -> list[SentenceGroup]:
        sentence_count = len(sentences)
        if sentence_count <= 1 or len(groups) <= 1:
            return groups

        # Step 1: Build ownership map (sentence_index -> group_index)
        ownership: dict[int, int] = {}
        for gi, group in enumerate(groups):
            for r in group.ranges:
                for si in range(r.start, r.end + 1):
                    ownership[si] = gi

        # Step 2: Find boundary candidates
        candidates: list[tuple[int, int, int]] = []  # (sent_idx, from_group, to_group)
        for i in range(sentence_count - 1):
            if ownership[i] != ownership[i + 1]:
                gi_a = ownership[i]
                gi_b = ownership[i + 1]
                if len(sentences[i].text) < self._min_length:
                    candidates.append((i, gi_a, gi_b))
                if len(sentences[i + 1].text) < self._min_length:
                    candidates.append((i + 1, gi_b, gi_a))

        # Step 3: Query LLM for each candidate
        for sent_idx, from_group, to_group in candidates:
            if ownership[sent_idx] != from_group:
                continue  # already reassigned by a prior candidate

            # Determine which is previous and which is next
            if sent_idx > 0 and ownership.get(sent_idx - 1) == to_group:
                prev_gi, next_gi = to_group, from_group
            else:
                prev_gi, next_gi = from_group, to_group

            prev_context = _gather_context(sentences, ownership, prev_gi, sent_idx, -1)
            next_context = _gather_context(sentences, ownership, next_gi, sent_idx, 1)

            prompt = _build_reassignment_prompt(
                sentence_text=sentences[sent_idx].text,
                prev_label=groups[prev_gi].label,
                prev_context=prev_context,
                next_label=groups[next_gi].label,
                next_context=next_context,
            )

            try:
                response = self._client.call(prompt, temperature=self._temperature)
            except EnhancerError:
                raise
            except Exception as e:
                raise EnhancerError(f"LLM call failed during enhancement: {e}") from e

            decision = _parse_reassignment_response(response)
            if decision == "previous":
                ownership[sent_idx] = prev_gi
            elif decision == "next":
                ownership[sent_idx] = next_gi
            # else: None — ambiguous, keep original assignment

        # Step 4: Reconstruct groups from ownership map
        group_sentences: dict[int, list[int]] = {i: [] for i in range(len(groups))}
        for si in range(sentence_count):
            group_sentences[ownership[si]].append(si)

        result: list[SentenceGroup] = []
        for gi, group in enumerate(groups):
            indices = group_sentences[gi]
            if not indices:
                continue
            ranges = _indices_to_ranges(indices)
            result.append(SentenceGroup(label=group.label, ranges=tuple(ranges)))

        return result


def _gather_context(
    sentences: list[Sentence],
    ownership: dict[int, int],
    group_idx: int,
    exclude_idx: int,
    direction: int,
) -> list[str]:
    """Gather up to _CONTEXT_SIZE sentences from a group near exclude_idx."""
    context: list[str] = []
    step = direction
    idx = exclude_idx + step
    while 0 <= idx < len(sentences) and len(context) < _CONTEXT_SIZE:
        if ownership.get(idx) == group_idx:
            context.append(sentences[idx].text)
        elif ownership.get(idx) != group_idx and context:
            break  # moved past this group's territory
        idx += step
    if direction < 0:
        context.reverse()
    return context


def _build_reassignment_prompt(
    sentence_text: str,
    prev_label: tuple[str, ...],
    prev_context: list[str],
    next_label: tuple[str, ...],
    next_context: list[str],
) -> str:
    prev_topic = " > ".join(prev_label)
    next_topic = " > ".join(next_label)
    prev_block = (
        "\n".join(f"  - {s}" for s in prev_context)
        if prev_context
        else "  (no other sentences)"
    )
    next_block = (
        "\n".join(f"  - {s}" for s in next_context)
        if next_context
        else "  (no other sentences)"
    )

    return (
        "You are deciding which topic a short sentence belongs to.\n"
        "\n"
        "The sentence in question:\n"
        f'  "{sentence_text}"\n'
        "\n"
        f"Option A - Previous topic ({prev_topic}):\n"
        f"{prev_block}\n"
        "\n"
        f"Option B - Next topic ({next_topic}):\n"
        f"{next_block}\n"
        "\n"
        "Does the sentence belong to the PREVIOUS topic or the NEXT topic?\n"
        "Reply with exactly one word: PREVIOUS or NEXT"
    )


def _parse_reassignment_response(response: str) -> str | None:
    """Parse LLM response into 'previous' or 'next', or None if ambiguous."""
    cleaned = response.strip().upper()
    has_previous = "PREVIOUS" in cleaned
    has_next = "NEXT" in cleaned
    if has_previous and not has_next:
        return "previous"
    if has_next and not has_previous:
        return "next"
    return None


def _indices_to_ranges(indices: list[int]) -> list[SentenceRange]:
    """Convert sorted sentence indices into minimal contiguous ranges."""
    if not indices:
        return []
    ranges: list[SentenceRange] = []
    start = indices[0]
    end = indices[0]
    for idx in indices[1:]:
        if idx == end + 1:
            end = idx
        else:
            ranges.append(SentenceRange(start=start, end=end))
            start = idx
            end = idx
    ranges.append(SentenceRange(start=start, end=end))
    return ranges
