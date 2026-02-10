"""Gap handler implementations."""

from __future__ import annotations

from typing import TypeAlias

from txt_splitt.errors import GapError
from txt_splitt.protocols import LLMCallable
from txt_splitt.tracer import NoOpSpan, NoOpTracer, Span, Tracer
from txt_splitt.types import Sentence, SentenceGroup, SentenceRange

_CONTEXT_SIZE = 3
_DEFAULT_NEW_TOPIC = ("Uncategorized",)

OwnerId: TypeAlias = int | str
TracingSpan: TypeAlias = Span | NoOpSpan
GapTracer: TypeAlias = Tracer | NoOpTracer


class StrictGapHandler:
    """Validate that sentence groups provide continuous coverage.

    - Trims overlaps by adjusting later ranges
    - Raises GapError on any gap or incomplete coverage
    """

    def handle(
        self,
        groups: list[SentenceGroup],
        sentence_count: int,
        sentences: list[Sentence] | None = None,
    ) -> list[SentenceGroup]:
        if sentence_count <= 0:
            raise GapError("sentence_count must be positive")

        if not groups:
            raise GapError("No groups provided")

        max_index = sentence_count - 1

        # Flatten all (group_index, range) pairs and sort by start
        flat: list[tuple[int, SentenceRange]] = []
        for gi, group in enumerate(groups):
            for r in group.ranges:
                flat.append((gi, r))
        flat.sort(key=lambda x: (x[1].start, x[1].end))

        # Resolve overlaps by trimming later ranges and verify continuous coverage
        adjusted: dict[int, list[SentenceRange]] = {i: [] for i in range(len(groups))}
        next_expected = 0

        for gi, r in flat:
            if r.end < next_expected:
                # Entirely consumed by previous range
                continue
            start = max(r.start, next_expected)
            if start > r.end:
                continue

            # Check for gap before this range
            if start != next_expected:
                raise GapError(
                    f"Gap detected: sentences {next_expected}-{start - 1} "
                    "are not covered"
                )

            adjusted[gi].append(SentenceRange(start=start, end=r.end))
            next_expected = r.end + 1

        # Check for incomplete coverage at the end
        if next_expected <= max_index:
            raise GapError(
                f"Incomplete coverage: sentences {next_expected}-{max_index} "
                "are not covered"
            )

        # Build result groups, preserving order, dropping empty groups
        result: list[SentenceGroup] = []
        for gi, group in enumerate(groups):
            ranges = adjusted[gi]
            if ranges:
                result.append(
                    SentenceGroup(
                        label=group.label,
                        ranges=tuple(ranges),
                    )
                )

        return result


class RepairingGapHandler:
    """Repair sentence groups to provide continuous coverage.

    - Trims overlaps by adjusting later ranges
    - Fills gaps by extending adjacent ranges
    - Ensures the entire range [0, sentence_count-1] is covered
    """

    def handle(
        self,
        groups: list[SentenceGroup],
        sentence_count: int,
        sentences: list[Sentence] | None = None,
    ) -> list[SentenceGroup]:
        if sentence_count <= 0:
            raise GapError("sentence_count must be positive")

        if not groups:
            raise GapError("No groups provided")

        max_index = sentence_count - 1

        # Flatten all (group_index, range) pairs and sort by start
        flat: list[tuple[int, SentenceRange]] = []
        for gi, group in enumerate(groups):
            for r in group.ranges:
                flat.append((gi, r))
        flat.sort(key=lambda x: (x[1].start, x[1].end))

        # Adjusted ranges per group
        adjusted: dict[int, list[SentenceRange]] = {i: [] for i in range(len(groups))}

        # Track where we are in global sentence space
        next_expected = 0

        # Keep track of the last added range so we can extend it
        last_added: tuple[int, int] | None = None  # (gi, index_in_adjusted[gi])

        for gi, r in flat:
            if r.end < next_expected:
                # Entirely consumed by previous range
                continue

            start = max(r.start, next_expected)
            if start > r.end:
                continue

            # Gap detected
            if start > next_expected:
                if last_added is None:
                    # Gap at the very beginning: pull the first range back
                    start = 0
                else:
                    # Gap between ranges: extend the previous range forward
                    l_gi, l_idx = last_added
                    prev_r = adjusted[l_gi][l_idx]
                    adjusted[l_gi][l_idx] = SentenceRange(
                        start=prev_r.start, end=start - 1
                    )

            adjusted[gi].append(SentenceRange(start=start, end=r.end))
            last_added = (gi, len(adjusted[gi]) - 1)
            next_expected = r.end + 1

        # Final gap at the end
        if next_expected <= max_index:
            if last_added:
                l_gi, l_idx = last_added
                prev_r = adjusted[l_gi][l_idx]
                adjusted[l_gi][l_idx] = SentenceRange(start=prev_r.start, end=max_index)
            else:
                # This should be covered by "if not groups" but just in case
                raise GapError("Unable to cover end gap - no groups found")

        # Build result groups, preserving order, dropping empty groups
        result: list[SentenceGroup] = []
        for gi, group in enumerate(groups):
            ranges = adjusted[gi]
            if ranges:
                result.append(
                    SentenceGroup(
                        label=group.label,
                        ranges=tuple(ranges),
                    )
                )

        return result


class LLMRepairingGapHandler:
    """Repair gaps by asking an LLM where each missing sentence belongs.

    For each uncovered sentence between two neighboring groups, asks the LLM
    whether it belongs to the PREVIOUS group, the NEXT group, or a NEW group.
    Overlaps are still trimmed by keeping earlier coverage first.
    """

    def __init__(
        self,
        client: LLMCallable,
        *,
        temperature: float = 0.0,
        tracer: Tracer | None = None,
    ) -> None:
        self._client = client
        self._temperature = temperature
        self._tracer: GapTracer = tracer if tracer is not None else NoOpTracer()

    def handle(
        self,
        groups: list[SentenceGroup],
        sentence_count: int,
        sentences: list[Sentence] | None = None,
    ) -> list[SentenceGroup]:
        with self._tracer.span(
            "gap_handler.llm_repair",
            sentence_count=sentence_count,
            input_group_count=len(groups),
        ) as handler_span:
            return self._handle(groups, sentence_count, sentences, handler_span)

    def _handle(
        self,
        groups: list[SentenceGroup],
        sentence_count: int,
        sentences: list[Sentence] | None,
        handler_span: TracingSpan,
    ) -> list[SentenceGroup]:
        if sentence_count <= 0:
            raise GapError("sentence_count must be positive")

        if not groups:
            raise GapError("No groups provided")

        if sentences is None:
            raise GapError("LLMRepairingGapHandler requires sentences context")
        if len(sentences) != sentence_count:
            raise GapError("sentences length must match sentence_count")

        max_index = sentence_count - 1

        flat: list[tuple[int, SentenceRange]] = []
        for gi, group in enumerate(groups):
            for r in group.ranges:
                flat.append((gi, r))
        flat.sort(key=lambda x: (x[1].start, x[1].end))

        ownership: dict[int, OwnerId] = {}
        next_expected = 0
        last_owner: int | None = None
        gaps: list[tuple[int, int, int | None, int | None]] = []

        for gi, r in flat:
            if r.end < next_expected:
                continue

            start = max(r.start, next_expected)
            if start > r.end:
                continue

            if start > next_expected:
                gaps.append((next_expected, start - 1, last_owner, gi))

            for si in range(start, r.end + 1):
                ownership[si] = gi

            last_owner = gi
            next_expected = r.end + 1

        if next_expected <= max_index:
            gaps.append((next_expected, max_index, last_owner, None))

        handler_span.attributes["gap_count"] = len(gaps)
        handler_span.attributes["gaps"] = [
            {
                "start": gap_start,
                "end": gap_end,
                "prev_owner": _owner_debug_label(prev_owner, groups),
                "next_owner": _owner_debug_label(next_owner, groups),
            }
            for gap_start, gap_end, prev_owner, next_owner in gaps
        ]

        new_group_labels: dict[str, tuple[str, ...]] = {}
        new_group_order: list[str] = []
        new_group_by_label: dict[tuple[str, ...], str] = {}

        for gap_start, gap_end, prev_owner, next_owner in gaps:
            with self._tracer.span(
                "gap_handler.llm_repair.gap",
                gap_start=gap_start,
                gap_end=gap_end,
                prev_owner=_owner_debug_label(prev_owner, groups),
                next_owner=_owner_debug_label(next_owner, groups),
            ):
                for sent_idx in range(gap_start, gap_end + 1):
                    owner = _resolve_gap_sentence_owner(
                        tracer=self._tracer,
                        client=self._client,
                        temperature=self._temperature,
                        sentences=sentences,
                        ownership=ownership,
                        sentence_index=sent_idx,
                        groups=groups,
                        prev_owner=prev_owner,
                        next_owner=next_owner,
                    )

                    if isinstance(owner, tuple):
                        label = owner
                        if label not in new_group_by_label:
                            new_owner_id = f"new-{len(new_group_order)}"
                            new_group_by_label[label] = new_owner_id
                            new_group_labels[new_owner_id] = label
                            new_group_order.append(new_owner_id)
                        ownership[sent_idx] = new_group_by_label[label]
                    else:
                        ownership[sent_idx] = owner

        for si in range(sentence_count):
            if si not in ownership:
                raise GapError(f"Unable to assign sentence {si}")

        existing_indices: dict[int, list[int]] = {i: [] for i in range(len(groups))}
        new_indices: dict[str, list[int]] = {gid: [] for gid in new_group_order}
        for si in range(sentence_count):
            owner_id = ownership[si]
            if isinstance(owner_id, int):
                existing_indices[owner_id].append(si)
            else:
                new_indices[owner_id].append(si)

        result: list[SentenceGroup] = []
        for gi, group in enumerate(groups):
            indices = existing_indices[gi]
            if indices:
                result.append(
                    SentenceGroup(
                        label=group.label, ranges=tuple(_indices_to_ranges(indices))
                    )
                )

        for gid in new_group_order:
            indices = new_indices[gid]
            if not indices:
                continue
            result.append(
                SentenceGroup(
                    label=new_group_labels[gid],
                    ranges=tuple(_indices_to_ranges(indices)),
                )
            )

        handler_span.attributes["output_group_count"] = len(result)
        handler_span.attributes["new_group_count"] = len(new_group_order)
        handler_span.attributes["new_groups"] = [
            {
                "id": gid,
                "label": " > ".join(new_group_labels[gid]),
                "size": len(new_indices[gid]),
            }
            for gid in new_group_order
        ]
        return result


def _resolve_gap_sentence_owner(
    *,
    tracer: GapTracer,
    client: LLMCallable,
    temperature: float,
    sentences: list[Sentence],
    ownership: dict[int, OwnerId],
    sentence_index: int,
    groups: list[SentenceGroup],
    prev_owner: int | None,
    next_owner: int | None,
) -> int | tuple[str, ...]:
    with tracer.span(
        "gap_handler.llm_repair.resolve_sentence",
        sentence_index=sentence_index,
        prev_owner=_owner_debug_label(prev_owner, groups),
        next_owner=_owner_debug_label(next_owner, groups),
    ) as span:
        if prev_owner is None and next_owner is None:
            raise GapError("Unable to resolve gap: no neighboring groups")
        if prev_owner is None:
            if next_owner is None:
                raise GapError("Unable to resolve gap: missing next owner")
            span.attributes["resolved_owner"] = _owner_debug_label(next_owner, groups)
            span.attributes["decision"] = "next_no_prev_neighbor"
            return next_owner
        if next_owner is None:
            span.attributes["resolved_owner"] = _owner_debug_label(prev_owner, groups)
            span.attributes["decision"] = "previous_no_next_neighbor"
            return prev_owner

        prev_context = _gather_context(
            sentences, ownership, prev_owner, sentence_index, -1
        )
        next_context = _gather_context(
            sentences, ownership, next_owner, sentence_index, 1
        )

        prompt = _build_gap_prompt(
            sentence_text=sentences[sentence_index].text,
            prev_label=groups[prev_owner].label,
            prev_context=prev_context,
            next_label=groups[next_owner].label,
            next_context=next_context,
        )
        span.attributes["sentence_text"] = sentences[sentence_index].text
        span.attributes["prev_label"] = " > ".join(groups[prev_owner].label)
        span.attributes["next_label"] = " > ".join(groups[next_owner].label)
        span.attributes["prev_context"] = prev_context
        span.attributes["next_context"] = next_context
        span.attributes["prompt"] = prompt

        try:
            response = client.call(prompt, temperature=temperature)
        except GapError:
            raise
        except Exception as e:
            span.attributes["error"] = str(e)
            raise GapError(f"LLM call failed during gap repair: {e}") from e

        span.attributes["response"] = response
        decision, label = _parse_gap_response(response)
        span.attributes["parsed_decision"] = decision
        span.attributes["parsed_label"] = " > ".join(label) if label else None
        if decision == "previous":
            span.attributes["resolved_owner"] = _owner_debug_label(prev_owner, groups)
            return prev_owner
        if decision == "next":
            span.attributes["resolved_owner"] = _owner_debug_label(next_owner, groups)
            return next_owner
        if label is not None:
            span.attributes["resolved_owner"] = f"NEW ({' > '.join(label)})"
            return label

        span.attributes["fallback"] = "previous"
        span.attributes["resolved_owner"] = _owner_debug_label(prev_owner, groups)
        return prev_owner


def _gather_context(
    sentences: list[Sentence],
    ownership: dict[int, OwnerId],
    owner: OwnerId,
    anchor_idx: int,
    direction: int,
) -> list[str]:
    context: list[str] = []
    idx = anchor_idx + direction

    while 0 <= idx < len(sentences) and len(context) < _CONTEXT_SIZE:
        if ownership.get(idx) == owner:
            context.append(sentences[idx].text)
        elif context:
            break
        idx += direction

    if direction < 0:
        context.reverse()
    return context


def _build_gap_prompt(
    *,
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
        "You are resolving a sentence gap between two neighboring topic groups.\n"
        "\n"
        "Gap sentence:\n"
        f'  "{sentence_text}"\n'
        "\n"
        f"Option A - Previous topic ({prev_topic}):\n"
        f"{prev_block}\n"
        "\n"
        f"Option B - Next topic ({next_topic}):\n"
        f"{next_block}\n"
        "\n"
        "Decide where this sentence belongs.\n"
        "Allowed answers:\n"
        "PREVIOUS\n"
        "NEXT\n"
        "NEW: Level1 > Level2 > Topic\n"
        "Reply using exactly one allowed answer."
    )


def _parse_gap_response(response: str) -> tuple[str, tuple[str, ...] | None]:
    cleaned = response.strip()
    upper = cleaned.upper()

    if upper.startswith("PREVIOUS"):
        return ("previous", None)
    if upper.startswith("NEXT"):
        return ("next", None)
    if upper.startswith("NEW"):
        _, _, topic_raw = cleaned.partition(":")
        topic_text = topic_raw.strip()
        if not topic_text:
            return ("new", _DEFAULT_NEW_TOPIC)
        label = tuple(part.strip() for part in topic_text.split(">") if part.strip())
        if not label:
            return ("new", _DEFAULT_NEW_TOPIC)
        return ("new", label)

    has_previous = "PREVIOUS" in upper
    has_next = "NEXT" in upper
    if has_previous and not has_next:
        return ("previous", None)
    if has_next and not has_previous:
        return ("next", None)
    return ("unknown", None)


def _indices_to_ranges(indices: list[int]) -> list[SentenceRange]:
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


def _owner_debug_label(owner: int | None, groups: list[SentenceGroup]) -> str:
    if owner is None:
        return "None"
    return f"{owner} ({' > '.join(groups[owner].label)})"
