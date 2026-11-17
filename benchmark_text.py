#!/usr/bin/env python3
"""Benchmark text splitting pipelines and write portable reports.

``--splitter llm`` (default) runs the completion-only pipeline from
``split_text.py``; ``--splitter decisions`` runs the decision-API pipeline from
``split_text_decisions.py`` (decision model places boundaries, LLM names topics).
"""

import argparse
import asyncio
import hashlib
import json
import math
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

# Keep the script runnable from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).parent / "src"))

import split_text  # noqa: E402
import split_text_decisions  # noqa: E402
from txt_splitt.benchmark_metrics import measure_result  # noqa: E402
from txt_splitt.benchmark_report import render_report  # noqa: E402
from txt_splitt.llms.llamacpp import AsyncLLamaCPP, LLamaCPP  # noqa: E402
from txt_splitt.llms.llamacpp_decisions import (  # noqa: E402
    AsyncLLamaCPPDecisions,
    LLamaCPPDecisions,
)
from txt_splitt.protocols import AsyncLLMCallable, LLMCallable  # noqa: E402
from txt_splitt.sentences.decision_split import (  # noqa: E402
    DECISION_BATCH_SIZE,
    DECISION_CONTEXT_SENTENCES,
    DECISION_NAMESPACE,
    DECISION_SPLIT_THRESHOLD,
    LABEL_NAMESPACE,
)

NAMESPACES = (
    "topic-range",
    "gap-repair",
    "short-sentence-enhancer",
    "boundary-evaluator",
)
SPLITTERS = ("llm", "decisions")
# Settings shared by both splitters, then the ones only one of them uses. Only
# the keys of the selected splitter go into the artifact config, so the report
# flags a real configuration difference, not unused defaults.
COMMON_CONFIG_KEYS = (
    "splitter",
    "temperature",
    "anchor_words",
    "long_sentence_threshold",
    "min_sentence_words",
    "max_chunk_chars",
    "use_async",
    "max_concurrent",
    "repeats",
    "warmup",
)
LLM_CONFIG_KEYS = (
    "short_sentence_min_length",
    "boundary_context_window",
    "boundary_max_shift",
)
DECISION_CONFIG_KEYS = (
    "threshold",
    "decision_batch_size",
    "decision_context_sentences",
    "decision_context_window",
    "decision_timeout",
)
EXTENSIONS = {".txt", ".md", ".html", ".htm"}


class CallStats:
    """Collect timings safely when the synchronous executor uses threads."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.data: dict[str, Any] = {
            "count": 0,
            "elapsed_seconds": 0.0,
            "failures": 0,
            "namespaces": {},
        }

    def record(self, namespace: str, elapsed: float, failed: bool) -> None:
        with self.lock:
            entry = self.data["namespaces"].setdefault(
                namespace, {"count": 0, "elapsed_seconds": 0.0, "failures": 0}
            )
            for item in (self.data, entry):
                item["count"] += 1
                item["elapsed_seconds"] += elapsed
                item["failures"] += int(failed)


class TimedSync:
    def __init__(self, client: LLMCallable, namespace: str, stats: CallStats) -> None:
        self.client, self.namespace, self.stats = client, namespace, stats

    def call(self, prompt: str, temperature: float) -> str:
        start = perf_counter()
        failed = True
        try:
            result = self.client.call(prompt, temperature)
            failed = False
            return result
        finally:
            self.stats.record(self.namespace, perf_counter() - start, failed)


class TimedAsync:
    def __init__(
        self, client: AsyncLLMCallable, namespace: str, stats: CallStats
    ) -> None:
        self.client, self.namespace, self.stats = client, namespace, stats

    async def call(self, prompt: str, temperature: float) -> str:
        start = perf_counter()
        failed = True
        try:
            result = await self.client.call(prompt, temperature)
            failed = False
            return result
        finally:
            self.stats.record(self.namespace, perf_counter() - start, failed)


def discover(inputs: list[str]) -> list[Path]:
    """Resolve and deduplicate files, independent of input order."""
    files: set[Path] = set()
    for item in inputs:
        path = Path(item)
        if not path.exists():
            raise ValueError(f"Input does not exist: {path}")
        if path.is_dir():
            files.update(
                p.resolve()
                for p in path.rglob("*")
                if p.is_file() and p.suffix.lower() in EXTENSIONS
            )
        elif path.is_file() and path.suffix.lower() in EXTENSIONS:
            files.add(path.resolve())
        else:
            raise ValueError(f"Unsupported input: {path}")
    return sorted(files)


def parser_for_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", help="Files or directories to benchmark")
    parser.add_argument("--host", help="LLamaCPP server URL")
    parser.add_argument("--model", default="default")
    parser.add_argument("--label", default="benchmark")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results"))
    parser.add_argument(
        "--compare", action="extend", nargs="+", default=[], metavar="JSON"
    )
    parser.add_argument("--report-only", nargs="+", metavar="JSON")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--anchor-words", type=int, default=12)
    parser.add_argument("--long-sentence-threshold", type=int, default=24)
    parser.add_argument("--min-sentence-words", type=int, default=4)
    parser.add_argument("--max-chunk-chars", type=int, default=84000)
    parser.add_argument("--short-sentence-min-length", type=int, default=0)
    parser.add_argument("--boundary-context-window", type=int, default=3)
    parser.add_argument("--boundary-max-shift", type=int, default=0)
    parser.add_argument("--async", dest="use_async", action="store_true")
    parser.add_argument("--max-concurrent", type=int, default=10)
    decisions = parser.add_argument_group(
        "decision splitter", "Used only with --splitter decisions"
    )
    decisions.add_argument(
        "--splitter",
        choices=SPLITTERS,
        default="llm",
        help="llm: completion-only pipeline; decisions: decision API places "
        "boundaries and the LLM names topics (default: llm)",
    )
    decisions.add_argument(
        "--decision-host", help="Decision API server (default: same as --host)"
    )
    decisions.add_argument(
        "--decision-model", help="Decision model name; omitted when not set"
    )
    decisions.add_argument(
        "--decision-api-key-env",
        default="DECISION_API_KEY",
        help="Environment variable holding an optional bearer token",
    )
    decisions.add_argument("--decision-timeout", type=float, default=60.0)
    decisions.add_argument(
        "--threshold",
        type=float,
        default=DECISION_SPLIT_THRESHOLD,
        help="Split when P(split) >= threshold",
    )
    decisions.add_argument(
        "--decision-batch-size", type=int, default=DECISION_BATCH_SIZE
    )
    decisions.add_argument(
        "--decision-context-sentences", type=int, default=DECISION_CONTEXT_SENTENCES
    )
    decisions.add_argument(
        "--decision-context-window",
        type=int,
        default=0,
        help="Decision model context window in tokens; 0 disables the check",
    )
    return parser


def load_payload(path: str) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError(f"Unsupported benchmark artifact: {path}")
    return data


def output_paths(output_dir: Path) -> tuple[Path, Path]:
    """Create the output directory and pick one JSON/HTML pair for the whole run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    base = output_dir / f"benchmark_{stamp}"
    return base.with_suffix(".json"), base.with_suffix(".html")


def write_atomic(path: Path, content: str) -> None:
    """Replace the file in one step so an interrupt never leaves it truncated."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def save_reports(
    payload: dict[str, Any],
    comparisons: list[dict[str, Any]],
    json_path: Path,
    html_path: Path,
) -> None:
    write_atomic(json_path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    write_atomic(html_path, render_report(payload, comparisons))


def decision_client(args: argparse.Namespace) -> LLamaCPPDecisions:
    return LLamaCPPDecisions(
        args.decision_host or args.host,
        model=args.decision_model,
        api_key=os.environ.get(args.decision_api_key_env) or None,
        timeout=args.decision_timeout,
    )


def sync_clients(args: argparse.Namespace, stats: CallStats) -> dict[str, LLMCallable]:
    """Timed clients keyed by the namespaces the selected pipeline requests."""
    llm = split_text.LLamaCPPAdapter(LLamaCPP(host=args.host, model=args.model))
    if args.splitter == "decisions":
        decider = split_text_decisions.DecisionAdapter(decision_client(args))
        return {
            DECISION_NAMESPACE: TimedSync(decider, DECISION_NAMESPACE, stats),
            LABEL_NAMESPACE: TimedSync(llm, LABEL_NAMESPACE, stats),
        }
    return {n: TimedSync(llm, n, stats) for n in NAMESPACES}


def async_clients(
    args: argparse.Namespace, stats: CallStats
) -> dict[str, AsyncLLMCallable]:
    llm = split_text.AsyncLLamaCPPAdapter(
        AsyncLLamaCPP(host=args.host, model=args.model)
    )
    if args.splitter == "decisions":
        decider = split_text_decisions.AsyncDecisionAdapter(
            AsyncLLamaCPPDecisions(decision_client(args))
        )
        return {
            DECISION_NAMESPACE: TimedAsync(decider, DECISION_NAMESPACE, stats),
            LABEL_NAMESPACE: TimedAsync(llm, LABEL_NAMESPACE, stats),
        }
    return {n: TimedAsync(llm, n, stats) for n in NAMESPACES}


def execute_one(args: argparse.Namespace, path: Path, text: str) -> dict[str, Any]:
    stats = CallStats()
    start = perf_counter()
    metrics = None
    error = None
    elapsed = None
    try:
        if args.splitter == "decisions":
            pipeline = split_text_decisions.create_pipeline(args, path)
        else:
            pipeline = split_text.create_pipeline(args, path)
        if args.use_async:
            result = asyncio.run(
                split_text.execute_session_async(
                    pipeline,
                    text,
                    async_clients(args, stats),
                    max_concurrent=args.max_concurrent,
                )
            )
        else:
            result = split_text.execute_session_sync(
                pipeline, text, sync_clients(args, stats)
            )
        elapsed = perf_counter() - start
        metrics = measure_result(result)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {
        "status": "error" if error else "ok",
        "elapsed_seconds": elapsed if elapsed is not None else perf_counter() - start,
        "error": error,
        "metrics": metrics,
        "llm": stats.data,
    }


def main(argv: list[str] | None = None) -> int:
    parser = parser_for_cli()
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.warmup < 0 or args.max_concurrent < 1:
        parser.error(
            "repeats and max-concurrent must be positive; warmup must be nonnegative"
        )
    if not math.isfinite(args.temperature):
        parser.error("temperature must be finite")
    if any(
        getattr(args, key) < 1
        for key in (
            "anchor_words",
            "long_sentence_threshold",
            "min_sentence_words",
            "max_chunk_chars",
            "boundary_context_window",
        )
    ):
        parser.error("sentence, chunk and context sizes must be positive")
    if args.short_sentence_min_length < 0 or args.boundary_max_shift < 0:
        parser.error("enhancer settings must be nonnegative")
    if args.splitter == "decisions":
        if args.short_sentence_min_length or args.boundary_max_shift:
            parser.error(
                "--short-sentence-min-length and --boundary-max-shift are not "
                "supported with --splitter decisions"
            )
        if not 0 <= args.threshold <= 1:
            parser.error("threshold must be between 0 and 1")
        if args.decision_batch_size < 1:
            parser.error("decision-batch-size must be positive")
        if args.decision_context_sentences < 0 or args.decision_context_window < 0:
            parser.error("decision context settings must be nonnegative")
        if not (math.isfinite(args.decision_timeout) and args.decision_timeout > 0):
            parser.error("decision-timeout must be positive")
        if args.max_chunk_chars < 16:
            parser.error("max-chunk-chars must be at least 16 for label requests")
    standalone_compare = (
        not args.report_only
        and not args.inputs
        and not args.host
        and bool(args.compare)
    )
    if standalone_compare and len(args.compare) < 2:
        parser.error("--compare without inputs requires at least two JSON files")
    if (
        not args.report_only
        and not standalone_compare
        and (not args.host or not args.inputs)
    ):
        parser.error("--host and at least one input are required for a benchmark")
    if args.report_only and args.inputs:
        parser.error("inputs cannot be used with --report-only")
    interrupted = False
    try:
        report_files = args.report_only or (
            list(args.compare) if standalone_compare else None
        )
        compare_files = [] if standalone_compare else args.compare
        comparisons = [load_payload(p) for p in compare_files]
        if report_files:
            previous = [load_payload(p) for p in report_files]
            payload = previous[0]
            comparisons = previous[1:] + comparisons
            json_path, html_path = output_paths(args.output_dir)
        else:
            paths = discover(args.inputs)
            if not paths:
                raise ValueError("No supported input files found")
            decisions = args.splitter == "decisions"
            config_keys = COMMON_CONFIG_KEYS + (
                DECISION_CONFIG_KEYS if decisions else LLM_CONFIG_KEYS
            )
            config = {key: getattr(args, key) for key in config_keys}
            config["cache_enabled"] = False
            payload = {
                "schema_version": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "label": args.label,
                "model": args.model,
                "host": args.host,
                "splitter": args.splitter,
                "decision_host": (args.decision_host or args.host)
                if decisions
                else None,
                "decision_model": args.decision_model if decisions else None,
                "config": config,
                "complete": False,
                "documents": [],
                "runs": [],
                "warmups": [],
            }
            json_path, html_path = output_paths(args.output_dir)
            try:
                for path in paths:
                    text = ""
                    digest = None
                    read_error = None
                    try:
                        raw = path.read_bytes()
                        digest = hashlib.sha256(raw).hexdigest()
                        text = raw.decode("utf-8")
                    except (OSError, UnicodeError) as exc:
                        read_error = f"{type(exc).__name__}: {exc}"
                    doc_id = str(path)
                    payload["documents"].append(
                        {
                            "id": doc_id,
                            "path": str(path),
                            "sha256": digest,
                            "input_chars": len(text) if read_error is None else None,
                        }
                    )
                    for kind, count in (
                        ("warmups", args.warmup),
                        ("runs", args.repeats),
                    ):
                        for repeat in range(1, count + 1):
                            print(f"{kind}: {path} ({repeat}/{count})", flush=True)
                            if read_error is None:
                                run = execute_one(args, path, text)
                            else:
                                run = {
                                    "status": "error",
                                    "elapsed_seconds": 0.0,
                                    "error": read_error,
                                    "metrics": None,
                                    "llm": CallStats().data,
                                }
                            run.update(document_id=doc_id, repeat=repeat)
                            payload[kind].append(run)
                    # Persist after every file so the benchmark can be stopped
                    # at any time and still leave usable results behind.
                    save_reports(payload, comparisons, json_path, html_path)
                    print(f"Saved progress: {json_path}", flush=True)
            except KeyboardInterrupt:
                interrupted = True
                print("Interrupted; saving results collected so far.", file=sys.stderr)
            payload["complete"] = not interrupted
        save_reports(payload, comparisons, json_path, html_path)
        print(f"JSON: {json_path}\nHTML: {html_path}")
        if interrupted:
            return 130
        return int(
            any(
                r.get("status") != "ok"
                for r in payload.get("runs", []) + payload.get("warmups", [])
            )
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
