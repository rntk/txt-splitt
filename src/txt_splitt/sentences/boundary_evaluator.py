"""BoundaryEvaluator enhancer: shift group boundaries using LLM judgment."""

from txt_splitt.errors import EnhancerError
from txt_splitt.pipeline import CompletedStage, PendingStage, StageResult
from txt_splitt.protocols import LLMCallable, LLMRequest, LLMResponse
from txt_splitt.sentences.types import Sentence, SentenceGroup, _indices_to_ranges


class BoundaryEvaluator:
    """Evaluate every boundary between adjacent groups and shift sentences.

    For each boundary between consecutive groups, gathers context sentences
    from each side and asks the LLM whether the boundary is correct or should
    shift left/right by up to *max_shift* sentences.
    """

    def __init__(
        self,
        client: LLMCallable | None = None,
        *,
        context_window: int = 3,
        max_shift: int = 2,
        temperature: float = 0.0,
    ) -> None:
        self._client = client
        self._context_window = context_window
        self._max_shift = max_shift
        self._temperature = temperature

    def enhance(
        self,
        groups: list[SentenceGroup],
        sentences: list[Sentence],
    ) -> list[SentenceGroup]:
        if self._client is None:
            msg = "BoundaryEvaluator.enhance() requires a configured client"
            raise EnhancerError(msg)
        sentence_count = len(sentences)
        if sentence_count <= 1 or len(groups) <= 1:
            return groups

        # Build ownership map (sentence_index → group_index)
        ownership: dict[int, int] = {}
        for gi, group in enumerate(groups):
            for r in group.ranges:
                for si in range(r.start, r.end + 1):
                    ownership[si] = gi

        # Find boundary positions (index i where ownership[i] != ownership[i+1])
        boundaries: list[int] = []
        for i in range(sentence_count - 1):
            if ownership[i] != ownership[i + 1]:
                boundaries.append(i)

        for boundary_idx in boundaries:
            left_gi = ownership[boundary_idx]
            right_gi = ownership[boundary_idx + 1]

            left_sentences = _gather_boundary_context(
                sentences,
                ownership,
                left_gi,
                boundary_idx,
                "left",
                self._context_window,
            )
            right_sentences = _gather_boundary_context(
                sentences,
                ownership,
                right_gi,
                boundary_idx + 1,
                "right",
                self._context_window,
            )

            prompt = _build_boundary_prompt(
                left_label=groups[left_gi].label,
                left_sentences=left_sentences,
                right_label=groups[right_gi].label,
                right_sentences=right_sentences,
            )

            try:
                response = self._client.call(prompt, temperature=self._temperature)
            except EnhancerError:
                raise
            except Exception as e:
                raise EnhancerError(f"LLM call failed during enhancement: {e}") from e

            direction, shift = _parse_boundary_response(response, self._max_shift)

            if direction == "shift_left" and shift > 0:
                # Move 'shift' sentences from right group to left group
                # (sentences just after current boundary move left)
                count = 0
                idx = boundary_idx + 1
                while idx < sentence_count and count < shift:
                    if ownership[idx] == right_gi:
                        ownership[idx] = left_gi
                        count += 1
                        idx += 1
                    else:
                        break
            elif direction == "shift_right" and shift > 0:
                # Move 'shift' sentences from left group to right group
                # (sentences just before current boundary move right)
                count = 0
                idx = boundary_idx
                while idx >= 0 and count < shift:
                    if ownership[idx] == left_gi:
                        ownership[idx] = right_gi
                        count += 1
                        idx -= 1
                    else:
                        break

        # Reconstruct groups from ownership map
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

    def plan_enhance(
        self,
        groups: list[SentenceGroup],
        sentences: list[Sentence],
    ) -> StageResult[list[SentenceGroup]]:
        sentence_count = len(sentences)
        if sentence_count <= 1 or len(groups) <= 1:
            return CompletedStage(groups)

        ownership: dict[int, int] = {}
        for gi, group in enumerate(groups):
            for r in group.ranges:
                for si in range(r.start, r.end + 1):
                    ownership[si] = gi

        boundaries: list[tuple[int, int, int, str]] = []
        for i in range(sentence_count - 1):
            if ownership[i] != ownership[i + 1]:
                left_gi = ownership[i]
                right_gi = ownership[i + 1]
                boundaries.append(
                    (
                        i,
                        left_gi,
                        right_gi,
                        _build_boundary_prompt(
                            left_label=groups[left_gi].label,
                            left_sentences=_gather_boundary_context(
                                sentences,
                                ownership,
                                left_gi,
                                i,
                                "left",
                                self._context_window,
                            ),
                            right_label=groups[right_gi].label,
                            right_sentences=_gather_boundary_context(
                                sentences,
                                ownership,
                                right_gi,
                                i + 1,
                                "right",
                                self._context_window,
                            ),
                        ),
                    )
                )

        if not boundaries:
            return CompletedStage(groups)

        requests = tuple(
            LLMRequest(
                prompt=prompt,
                temperature=self._temperature,
                stage_name="enhancer.boundary_evaluator",
                metadata={"namespace": "boundary-evaluator"},
            )
            for _, _, _, prompt in boundaries
        )
        return PendingStage(
            requests=requests,
            resume=lambda responses: CompletedStage(
                _apply_boundary_responses(
                    groups=groups,
                    sentences=sentences,
                    ownership=dict(ownership),
                    boundaries=boundaries,
                    responses=responses,
                    max_shift=self._max_shift,
                )
            ),
        )


def _gather_boundary_context(
    sentences: list[Sentence],
    ownership: dict[int, int],
    group_idx: int,
    start_idx: int,
    direction: str,
    window: int,
) -> list[tuple[int, str]]:
    """Gather up to *window* sentences belonging to *group_idx* near *start_idx*.

    direction="left"  → walk backwards from start_idx (inclusive)
    direction="right" → walk forwards from start_idx (inclusive)
    Returns list of (sentence_index, text) in document order.
    """
    result: list[tuple[int, str]] = []
    if direction == "left":
        idx = start_idx
        while idx >= 0 and len(result) < window:
            if ownership.get(idx) == group_idx:
                result.append((idx, sentences[idx].text))
                idx -= 1
            else:
                break
        result.reverse()
    else:  # right
        idx = start_idx
        while idx < len(sentences) and len(result) < window:
            if ownership.get(idx) == group_idx:
                result.append((idx, sentences[idx].text))
                idx += 1
            else:
                break
    return result


def _build_boundary_prompt(
    left_label: tuple[str, ...],
    left_sentences: list[tuple[int, str]],
    right_label: tuple[str, ...],
    right_sentences: list[tuple[int, str]],
) -> str:
    left_topic = " > ".join(left_label)
    right_topic = " > ".join(right_label)

    left_block = (
        "\n".join(f"  - {text}" for _, text in left_sentences)
        if left_sentences
        else "  (no sentences)"
    )
    right_block = (
        "\n".join(f"  - {text}" for _, text in right_sentences)
        if right_sentences
        else "  (no sentences)"
    )

    return (
        "You are evaluating the boundary between two topic groups.\n"
        "\n"
        f"Left group ({left_topic}) — last sentences:\n"
        f"{left_block}\n"
        "\n"
        f"Right group ({right_topic}) — first sentences:\n"
        f"{right_block}\n"
        "\n"
        "Is the boundary correct, or should it shift?\n"
        "Reply with exactly one of:\n"
        "  CORRECT\n"
        "  SHIFT_LEFT N   (move N sentences from the right group into the left group)\n"
        "  SHIFT_RIGHT N  (move N sentences from the left group into the right group)\n"
        "where N is a positive integer."
    )


def _parse_boundary_response(response: str, max_shift: int) -> tuple[str, int]:
    """Parse LLM response into (direction, n).

    Returns:
        ("correct", 0)         — boundary is fine
        ("shift_left", N)      — move N sentences rightward into left group
        ("shift_right", N)     — move N sentences leftward into right group
    """
    cleaned = response.strip().upper()
    if cleaned.startswith("CORRECT"):
        return ("correct", 0)
    for prefix, direction in (
        ("SHIFT_LEFT", "shift_left"),
        ("SHIFT_RIGHT", "shift_right"),
    ):
        if cleaned.startswith(prefix):
            rest = cleaned[len(prefix) :].strip()
            parts = rest.split()
            if parts and parts[0].isdigit():
                n = min(int(parts[0]), max_shift)
                return (direction, n)
            # No valid number → treat as correct
            return ("correct", 0)
    return ("correct", 0)


def _apply_boundary_responses(
    *,
    groups: list[SentenceGroup],
    sentences: list[Sentence],
    ownership: dict[int, int],
    boundaries: list[tuple[int, int, int, str]],
    responses: list[LLMResponse],
    max_shift: int,
) -> list[SentenceGroup]:
    sentence_count = len(sentences)
    for (boundary_idx, left_gi, right_gi, _), response in zip(
        boundaries, responses, strict=True
    ):
        direction, shift = _parse_boundary_response(response.content, max_shift)
        if direction == "shift_left" and shift > 0:
            count = 0
            idx = boundary_idx + 1
            while idx < sentence_count and count < shift:
                if ownership[idx] == right_gi:
                    ownership[idx] = left_gi
                    count += 1
                    idx += 1
                else:
                    break
        elif direction == "shift_right" and shift > 0:
            count = 0
            idx = boundary_idx
            while idx >= 0 and count < shift:
                if ownership[idx] == left_gi:
                    ownership[idx] = right_gi
                    count += 1
                    idx -= 1
                else:
                    break

    group_sentences: dict[int, list[int]] = {i: [] for i in range(len(groups))}
    for si, owner in ownership.items():
        group_sentences[owner].append(si)

    result: list[SentenceGroup] = []
    for gi, group in enumerate(groups):
        indices = group_sentences[gi]
        if indices:
            result.append(
                SentenceGroup(
                    label=group.label,
                    ranges=tuple(_indices_to_ranges(indices)),
                )
            )
    return result
