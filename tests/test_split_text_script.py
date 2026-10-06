from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from split_text import (
    _namespace_for_request,
    build_cache_store,
    create_pipeline,
    generate_html_report,
    wrap_async_llm,
    wrap_sync_llm,
)
from txt_splitt import (
    CachingAsyncLLMCallable,
    CachingLLMCallable,
    LLMRequest,
    SQLiteLLMCacheStore,
    Tracer,
    TracingAsyncLLMCallable,
    TracingLLMCallable,
)
from txt_splitt.sentences import (
    TopicRangeLLM,
)


class StubLLM:
    def call(self, prompt: str, temperature: float) -> str:
        return "ok"


class AsyncStubLLM:
    async def call(self, prompt: str, temperature: float) -> str:
        return "ok"


def _make_args(**overrides: object) -> SimpleNamespace:
    defaults: dict[str, object] = {
        "model": "demo-model",
        "cache_db": None,
        "cache_nonzero_temperature": False,
        "anchor_words": 12,
        "long_sentence_threshold": 24,
        "min_sentence_words": 4,
        "short_sentence_min_length": 20,
        "boundary_context_window": 3,
        "boundary_max_shift": 2,
        "temperature": 0.0,
        "max_chunk_chars": 84_000,
        "max_concurrent": 10,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_build_cache_store_returns_sqlite_store(tmp_path: Path) -> None:
    store = build_cache_store(_make_args(cache_db=str(tmp_path / "cache.sqlite")))

    assert isinstance(store, SQLiteLLMCacheStore)


def test_wrap_sync_llm_applies_cache_before_tracing(tmp_path: Path) -> None:
    args = _make_args(
        cache_db=str(tmp_path / "cache.sqlite"),
        cache_nonzero_temperature=True,
    )
    tracer = Tracer()
    wrapped = wrap_sync_llm(
        StubLLM(),
        namespace="topic-range",
        args=args,
        tracer=tracer,
        cache_store=build_cache_store(args),
    )

    assert isinstance(wrapped, TracingLLMCallable)
    assert isinstance(wrapped._inner, CachingLLMCallable)
    assert wrapped._inner._namespace == "topic-range"
    assert wrapped._inner._model_id == "demo-model"
    assert wrapped._inner._cache_nonzero_temperature is True


def test_wrap_async_llm_applies_cache_before_tracing(tmp_path: Path) -> None:
    args = _make_args(cache_db=str(tmp_path / "cache.sqlite"))
    tracer = Tracer()
    wrapped = wrap_async_llm(
        AsyncStubLLM(),
        namespace="topic-range-assignment",
        args=args,
        tracer=tracer,
        cache_store=build_cache_store(args),
    )

    assert isinstance(wrapped, TracingAsyncLLMCallable)
    assert isinstance(wrapped._inner, CachingAsyncLLMCallable)
    assert wrapped._inner._namespace == "topic-range-assignment"


def test_create_pipeline_uses_topic_range_llm(tmp_path: Path) -> None:
    args = _make_args(
        cache_db=str(tmp_path / "cache.sqlite"),
        short_sentence_min_length=0,
        boundary_max_shift=0,
    )
    pipeline = create_pipeline(
        args,
        Path("input.txt"),
        tracer=Tracer(),
        cache_store=build_cache_store(args),
    )

    assert isinstance(pipeline._llm, TopicRangeLLM)


def test_html_audio_payload_is_bounded_in_planned_requests() -> None:
    args = _make_args(short_sentence_min_length=0, boundary_max_shift=0)
    pipeline = create_pipeline(args, Path("audio.txt.html"))
    payload = "UklGRtTgBQBXQVZF" * 32_000
    text = (
        "<p>Send the audio to the realtime API.</p>"
        "<pre>{ &quot;type&quot;: &quot;input_audio_buffer.append&quot;, "
        f"&quot;audio&quot;: &quot;{payload}&quot; }}</pre>"
        "<p>Read the response from the server.</p>"
    )

    session = pipeline.start(text)
    requests = session.pending_requests()

    assert len(requests) > 1
    contents = [
        request.prompt.split("<content>\n", 1)[1].rsplit("\n</content>", 1)[0]
        for request in requests
    ]
    assert all(len(content) <= args.max_chunk_chars for content in contents)
    assert all(len(request.prompt) < 100_096 for request in requests)
    # The huge payload retains its original sentence ID in every fragment.
    audio_lines = [
        line for content in contents for line in content.split("\n") if "UklGR" in line
    ]
    markers = {line.split(" ", 1)[0] for line in audio_lines}
    assert len(audio_lines) > 1
    assert len(markers) == 1


def test_namespace_for_request_returns_explicit_namespace() -> None:
    request = LLMRequest(
        prompt="prompt",
        temperature=0.0,
        metadata={"namespace": "gap-repair"},
    )

    assert _namespace_for_request(request) == "gap-repair"


def test_namespace_for_request_raises_when_missing() -> None:
    request = LLMRequest(
        prompt="prompt",
        temperature=0.0,
        stage_name="topic_range.single_stage",
    )

    with pytest.raises(ValueError, match="missing a valid namespace"):
        _namespace_for_request(request)


def test_generate_html_report_includes_total_execution_time() -> None:
    result = SimpleNamespace(
        sentences=[
            SimpleNamespace(index=0, start=0, end=12, text="Hello world."),
        ],
        groups=[
            SimpleNamespace(
                label=("topic",),
                ranges=[SimpleNamespace(start=0, end=0)],
            )
        ],
    )

    report = generate_html_report(
        result,
        "Hello world.",
        Path("input.txt"),
        trace_output="trace",
        execution_time_seconds=1.23456,
    )

    assert "Total execution time:" in report
    assert "1.235 seconds" in report
