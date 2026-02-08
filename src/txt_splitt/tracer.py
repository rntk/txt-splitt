"""Simple tracing for pipeline debugging."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

from txt_splitt.protocols import LLMCallable


@dataclass
class Span:
    """A single traced operation."""

    name: str
    start_time: float
    end_time: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    children: list[Span] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000


class Tracer:
    """Collects a tree of spans for later inspection.

    Usage::

        tracer = Tracer()
        with tracer.span("my_op", key="value") as s:
            result = do_work()
            s.attributes["result_len"] = len(result)
        print(tracer.format())
    """

    def __init__(self) -> None:
        self._root_spans: list[Span] = []
        self._stack: list[Span] = []

    @property
    def spans(self) -> list[Span]:
        """Return root-level spans."""
        return list(self._root_spans)

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Generator[Span, None, None]:
        """Create a timed span. Nests automatically via internal stack."""
        s = Span(
            name=name,
            start_time=time.monotonic(),
            attributes=dict(attributes),
        )
        if self._stack:
            self._stack[-1].children.append(s)
        else:
            self._root_spans.append(s)
        self._stack.append(s)
        try:
            yield s
        finally:
            s.end_time = time.monotonic()
            self._stack.pop()

    def format(self) -> str:
        """Format all collected spans as an indented text tree."""
        lines: list[str] = []
        for span in self._root_spans:
            _format_span(span, lines, indent=0)
        return "\n".join(lines)


def _format_span(span: Span, lines: list[str], indent: int) -> None:
    prefix = "  " * indent
    dur = f"{span.duration_ms:.1f}ms"
    lines.append(f"{prefix}[TRACE] {span.name} ({dur})")
    for key, value in span.attributes.items():
        lines.append(f"{prefix}  {key}: {value}")
    for child in span.children:
        _format_span(child, lines, indent + 1)


@dataclass
class NoOpSpan:
    """A no-op span that does nothing."""

    attributes: dict[str, Any] = field(default_factory=dict)


class NoOpTracer:
    """A no-op tracer that performs no tracing operations.

    Used as a default when tracing is disabled to avoid conditional logic.
    """

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Generator[NoOpSpan, None, None]:
        """Create a no-op span that does nothing."""
        yield NoOpSpan()

    def format(self) -> str:
        """Return empty string (no traces collected)."""
        return ""


class TracingLLMCallable:
    """Wrapper that records every LLM call as a tracer span."""

    def __init__(self, inner: LLMCallable, tracer: Tracer) -> None:
        self._inner = inner
        self._tracer = tracer

    def call(self, prompt: str, temperature: float) -> str:
        with self._tracer.span(
            "llm.call",
            prompt_length=len(prompt),
            temperature=temperature,
        ) as s:
            response = self._inner.call(prompt, temperature)
            s.attributes["response_length"] = len(response)
            s.attributes["prompt"] = prompt
            s.attributes["response"] = response
            return response
