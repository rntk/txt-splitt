#!/usr/bin/env python3
"""Split text with the llama.cpp decision API and label topics with an LLM.

The decision model judges each sentence gap ("continue" vs "split") and the
completion LLM names the resulting ranges. Port of the decision-API splitter in
example/core/pipeline/decisionTopicSplit.js.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

# Add src to path so we can import txt_splitt when running from root
sys.path.append(str(Path(__file__).parent / "src"))

from split_text import (
    AsyncLLamaCPPAdapter,
    LLamaCPPAdapter,
    build_cache_store,
    execute_session_async,
    execute_session_sync,
    generate_html_report,
    wrap_async_llm,
    wrap_sync_llm,
)
from txt_splitt import (
    AsyncLLMCallable,
    HtmlCleaner,
    LLMCallable,
    SQLiteLLMCacheStore,
    Tracer,
)
from txt_splitt.html_cleaners import HTMLParserTagStripCleaner
from txt_splitt.llms.llamacpp import AsyncLLamaCPP, LLamaCPP
from txt_splitt.llms.llamacpp_decisions import (
    AsyncLLamaCPPDecisions,
    LLamaCPPDecisions,
)
from txt_splitt.sentences import (
    AdjacentSameTopicJoiner,
    DecisionMarker,
    DecisionTopicRangeLLM,
    MappingOffsetRestorer,
    RepairingGapHandler,
    SentencePipeline,
    SparseRegexSentenceSplitter,
    TopicRangeParser,
    build_pipeline,
)
from txt_splitt.sentences.decision_split import (
    DECISION_BATCH_SIZE,
    DECISION_CONTEXT_SENTENCES,
    DECISION_NAMESPACE,
    DECISION_SPLIT_THRESHOLD,
    LABEL_NAMESPACE,
)


class DecisionAdapter:
    """Expose a decision client as an ``LLMCallable``.

    The prompt is the JSON ``{"state", "questions"}`` payload produced by
    ``DecisionTopicRangeLLM``; the result is the raw JSON server response. As a
    plain string-to-string call it works with the cache and tracing wrappers.
    """

    def __init__(self, client: LLamaCPPDecisions) -> None:
        self._client = client

    def call(self, prompt: str, temperature: float) -> str:
        del temperature  # Decisions return probabilities; there is no sampling.
        payload = json.loads(prompt)
        return json.dumps(self._client.decide(payload["state"], payload["questions"]))


class AsyncDecisionAdapter:
    """Async counterpart of ``DecisionAdapter``."""

    def __init__(self, client: AsyncLLamaCPPDecisions) -> None:
        self._client = client

    async def call(self, prompt: str, temperature: float) -> str:
        del temperature
        payload = json.loads(prompt)
        result = await self._client.decide(payload["state"], payload["questions"])
        return json.dumps(result)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Split text into topics: the llama.cpp decision API places "
            "boundaries, the LLM names topics"
        )
    )
    parser.add_argument("input_file", help="Input text file to split")
    parser.add_argument(
        "--host",
        required=True,
        help="LLamaCPP completion server for topic labels (e.g. http://localhost:8080)",
    )
    parser.add_argument(
        "--model", default="default", help="Labeling model name (default: default)"
    )
    parser.add_argument(
        "--decision-host",
        help="Decision API server (default: same as --host)",
    )
    parser.add_argument(
        "--decision-model",
        help="Decision model name; omitted from requests when not set",
    )
    parser.add_argument(
        "--decision-api-key-env",
        default="DECISION_API_KEY",
        help=(
            "Environment variable holding an optional decision API bearer "
            "token (default: DECISION_API_KEY)"
        ),
    )
    parser.add_argument(
        "--decision-timeout",
        type=float,
        default=60.0,
        help="Decision request timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DECISION_SPLIT_THRESHOLD,
        help=(
            "Split when P(split) >= threshold; lower gives more, smaller topics "
            f"(default: {DECISION_SPLIT_THRESHOLD})"
        ),
    )
    parser.add_argument(
        "--decision-batch-size",
        type=int,
        default=DECISION_BATCH_SIZE,
        help=(
            f"Boundary questions per decision request (default: {DECISION_BATCH_SIZE})"
        ),
    )
    parser.add_argument(
        "--decision-context-sentences",
        type=int,
        default=DECISION_CONTEXT_SENTENCES,
        help=(
            "Extra sentences of context on each side of a batch "
            f"(default: {DECISION_CONTEXT_SENTENCES})"
        ),
    )
    parser.add_argument(
        "--decision-context-window",
        type=int,
        default=0,
        help=(
            "Decision model context window in tokens; batches are shrunk to "
            "fit. 0 disables the check (default: 0)"
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Labeling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--anchor-words",
        type=int,
        default=12,
        help="Preferred sentence size for long-span splitting (default: 12)",
    )
    parser.add_argument(
        "--long-sentence-threshold",
        type=int,
        default=24,
        help="Only shorten spans longer than N words (default: 24)",
    )
    parser.add_argument(
        "--min-sentence-words",
        type=int,
        default=4,
        help="Avoid creating non-terminal spans shorter than N words (default: 4)",
    )
    parser.add_argument(
        "--max-chunk-chars",
        type=int,
        default=84000,
        help="Max section characters per labeling request (default: 84000)",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Enable tracing and print the trace after the run",
    )
    parser.add_argument(
        "--async",
        dest="use_async",
        action="store_true",
        help="Use async clients with concurrent request limiting",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Max concurrent requests for async mode (default: 10)",
    )
    parser.add_argument(
        "--cache-db",
        help=(
            "SQLite database path for persistent response caching of both "
            "decision and label requests. Disabled when omitted."
        ),
    )
    parser.add_argument(
        "--cache-nonzero-temperature",
        action="store_true",
        help="Also cache label requests with non-zero temperature",
    )
    return parser


def create_pipeline(
    args: Any,
    input_path: Path,
    tracer: Tracer | None = None,
) -> SentencePipeline:
    """Create the decision-split pipeline; clients are injected at execution."""
    html_cleaner: HtmlCleaner | None = None
    offset_restorer = None
    if input_path.suffix.lower() in {".html", ".htm"}:
        html_cleaner = HTMLParserTagStripCleaner(strip_tags={"style", "script"})
        offset_restorer = MappingOffsetRestorer()

    return build_pipeline(
        splitter=SparseRegexSentenceSplitter(
            anchor_every_words=args.anchor_words,
            long_sentence_word_threshold=args.long_sentence_threshold,
            min_sentence_words=args.min_sentence_words,
        ),
        marker=DecisionMarker(),
        llm=DecisionTopicRangeLLM(
            threshold=args.threshold,
            batch_size=args.decision_batch_size,
            context_sentences=args.decision_context_sentences,
            context_window_tokens=args.decision_context_window or None,
            label_max_chars=args.max_chunk_chars,
            label_temperature=args.temperature,
        ),
        parser=TopicRangeParser(),
        # Decision ranges cover every sentence; this only guards edge cases.
        gap_handler=RepairingGapHandler(),
        joiner=AdjacentSameTopicJoiner(),
        html_cleaner=html_cleaner,
        offset_restorer=offset_restorer,
        tracer=tracer,
        merge_similar_topics=True,
    )


def _build_decision_client(args: Any) -> LLamaCPPDecisions:
    return LLamaCPPDecisions(
        args.decision_host or args.host,
        model=args.decision_model,
        api_key=os.environ.get(args.decision_api_key_env) or None,
        timeout=args.decision_timeout,
    )


def build_sync_clients(
    args: Any,
    *,
    tracer: Tracer | None,
    cache_store: SQLiteLLMCacheStore | None,
) -> dict[str, LLMCallable]:
    decision_args = argparse.Namespace(**{**vars(args), "model": _decision_id(args)})
    return {
        DECISION_NAMESPACE: wrap_sync_llm(
            DecisionAdapter(_build_decision_client(args)),
            namespace=DECISION_NAMESPACE,
            args=decision_args,
            tracer=tracer,
            cache_store=cache_store,
        ),
        LABEL_NAMESPACE: wrap_sync_llm(
            LLamaCPPAdapter(LLamaCPP(host=args.host, model=args.model)),
            namespace=LABEL_NAMESPACE,
            args=args,
            tracer=tracer,
            cache_store=cache_store,
        ),
    }


def build_async_clients(
    args: Any,
    *,
    tracer: Tracer | None,
    cache_store: SQLiteLLMCacheStore | None,
) -> dict[str, AsyncLLMCallable]:
    decision_args = argparse.Namespace(**{**vars(args), "model": _decision_id(args)})
    return {
        DECISION_NAMESPACE: wrap_async_llm(
            AsyncDecisionAdapter(AsyncLLamaCPPDecisions(_build_decision_client(args))),
            namespace=DECISION_NAMESPACE,
            args=decision_args,
            tracer=tracer,
            cache_store=cache_store,
        ),
        LABEL_NAMESPACE: wrap_async_llm(
            AsyncLLamaCPPAdapter(AsyncLLamaCPP(host=args.host, model=args.model)),
            namespace=LABEL_NAMESPACE,
            args=args,
            tracer=tracer,
            cache_store=cache_store,
        ),
    }


def _decision_id(args: Any) -> str:
    """Cache identity for decision responses: server plus model."""
    return f"{args.decision_host or args.host}|{args.decision_model or 'default'}"


def _write_report(
    result: Any,
    text: str,
    input_path: Path,
    trace_output: str | None,
    execution_time_seconds: float | None,
) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(f"{input_path.stem}_decisions_report_{timestamp}.html")
    output_file.write_text(
        generate_html_report(
            result, text, input_path, trace_output, execution_time_seconds
        ),
        encoding="utf-8",
    )
    print(f"Results saved to '{output_file}'")
    print(f"  - Sentences: {len(result.sentences)}")
    print(f"  - Groups: {len(result.groups)}")


def _read_input(args: Any) -> tuple[Path, str]:
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{args.input_file}' not found", file=sys.stderr)
        sys.exit(1)
    return input_path, input_path.read_text(encoding="utf-8")


def run_sync(args: Any) -> None:
    input_path, text = _read_input(args)
    tracer: Tracer | None = Tracer() if args.trace else None
    pipeline = create_pipeline(args, input_path, tracer=tracer)
    clients = build_sync_clients(
        args, tracer=tracer, cache_store=build_cache_store(args)
    )

    print(f"Processing '{args.input_file}' (decision split)...")
    trace_output: str | None = None
    start_time = perf_counter()
    try:
        result = execute_session_sync(pipeline, text, clients)
    except Exception as e:
        print(f"Error processing text: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        execution_time_seconds = perf_counter() - start_time
        if tracer:
            trace_output = tracer.format()
            print(trace_output, file=sys.stderr)
    _write_report(result, text, input_path, trace_output, execution_time_seconds)


async def run_async(args: Any) -> None:
    input_path, text = _read_input(args)
    tracer: Tracer | None = Tracer() if args.trace else None
    pipeline = create_pipeline(args, input_path, tracer=tracer)
    clients = build_async_clients(
        args, tracer=tracer, cache_store=build_cache_store(args)
    )

    print(
        f"Processing '{args.input_file}' (decision split, async, "
        f"max {args.max_concurrent} concurrent)..."
    )
    trace_output: str | None = None
    start_time = perf_counter()
    try:
        result = await execute_session_async(
            pipeline, text, clients, max_concurrent=args.max_concurrent
        )
    except Exception as e:
        print(f"Error processing text: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        execution_time_seconds = perf_counter() - start_time
        if tracer:
            trace_output = tracer.format()
            print(trace_output, file=sys.stderr)
    _write_report(result, text, input_path, trace_output, execution_time_seconds)


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.use_async:
        asyncio.run(run_async(args))
    else:
        run_sync(args)


if __name__ == "__main__":
    main()
