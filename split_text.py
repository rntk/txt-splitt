#!/usr/bin/env python3
"""Simple script to split text using LLamaCPP client and txt_splitt pipeline."""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

# Add src to path so we can import txt_splitt when running from root
sys.path.append(str(Path(__file__).parent / "src"))

from txt_splitt import (
    AdjacentSameTopicJoiner,
    BracketMarker,
    LLMCallable,
    LLMRepairingGapHandler,
    MappingOffsetRestorer,
    OptimizingMarker,
    OverlapChunker,
    Pipeline,
    RetryingLLMCallable,
    SparseRegexSentenceSplitter,
    HTMLParserTagStripCleaner,
    TopicListLLM,
    TopicRangeAssignmentLLM,
    TopicRangeLLM,
    TopicRangeParser,
    Tracer,
    TracingAsyncLLMCallable,
    TracingLLMCallable,
)
from txt_splitt.llms.llamacpp import AsyncLLamaCPP, LLamaCPP


class LLamaCPPAdapter:
    """Adapter to make LLamaCPP compatible with LLMCallable protocol."""

    def __init__(self, client: LLamaCPP) -> None:
        self._client = client

    def call(self, prompt: str, temperature: float) -> str:
        """Call the LLM with a prompt and temperature."""
        return self._client.call([prompt], temperature=temperature)  # type: ignore[no-any-return]


class AsyncLLamaCPPAdapter:
    """Adapter to make AsyncLLamaCPP compatible with AsyncLLMCallable protocol."""

    def __init__(self, client: AsyncLLamaCPP) -> None:
        self._client = client

    async def call(self, prompt: str, temperature: float) -> str:
        """Call the LLM with a prompt and temperature."""
        return await self._client.call([prompt], temperature=temperature)  # type: ignore[no-any-return]


def result_to_dict(result: Any) -> dict[str, Any]:
    """Convert SplitResult to a dictionary."""
    return {
        "sentences": [
            {
                "index": s.index,
                "start": s.start,
                "end": s.end,
                "text": s.text,
            }
            for s in result.sentences
        ],
        "groups": [
            {
                "label": list(g.label),
                "ranges": [{"start": r.start, "end": r.end} for r in g.ranges],
            }
            for g in result.groups
        ],
    }


def generate_html_report(
    result: Any,
    source_text: str,
    input_file: Path,
    trace_output: str | None = None,
) -> str:
    """Generate HTML report content from split result."""
    data = result_to_dict(result)
    json_payload = json.dumps(data, indent=2)
    sentences: list[dict[str, Any]] = data.get("sentences", [])
    groups: list[dict[str, Any]] = data.get("groups", [])

    sentence_map: dict[int, dict[str, Any]] = {
        int(sentence["index"]): sentence for sentence in sentences
    }

    css_lines = [
        "        body {",
        "            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',",
        "                Roboto, Helvetica, Arial, sans-serif;",
        "            line-height: 1.6;",
        "            color: #333;",
        "            max-width: 1000px;",
        "            margin: 0 auto;",
        "            padding: 20px;",
        "            background-color: #f4f7f6;",
        "        }",
        "        h1 { color: #2c3e50; text-align: center; }",
        "        .group {",
        "            background: #fff;",
        "            border-radius: 8px;",
        "            box-shadow: 0 2px 4px rgba(0,0,0,0.1);",
        "            margin-bottom: 30px;",
        "            overflow: hidden;",
        "            border: 1px solid #e0e0e0;",
        "        }",
        "        .group-header {",
        "            background: #3498db;",
        "            color: #fff;",
        "            padding: 15px 20px;",
        "            margin: 0;",
        "            font-size: 1.25rem;",
        "        }",
        "        .group-labels {",
        "            display: flex;",
        "            flex-wrap: wrap;",
        "            gap: 8px;",
        "            margin-top: 5px;",
        "        }",
        "        .label-tag {",
        "            background: rgba(255,255,255,0.2);",
        "            padding: 2px 8px;",
        "            border-radius: 4px;",
        "            font-size: 0.85rem;",
        "            font-weight: normal;",
        "        }",
        "        .sentences-list { padding: 0; margin: 0; list-style: none; }",
        "        .sentence-item {",
        "            padding: 15px 20px;",
        "            border-bottom: 1px solid #eee;",
        "            transition: background-color 0.2s;",
        "        }",
        "        .sentence-item:last-child { border-bottom: none; }",
        "        .sentence-item:hover { background-color: #f9f9f9; }",
        "        .sentence-index {",
        "            font-weight: bold;",
        "            color: #7f8c8d;",
        "            margin-right: 10px;",
        "            font-size: 0.9rem;",
        "        }",
        "        .sentence-text { display: inline; }",
        "        .sentence-tabs { margin-top: 10px; }",
        "        .tab-buttons {",
        "            display: inline-flex;",
        "            gap: 6px;",
        "            border-bottom: 1px solid #e0e0e0;",
        "            margin-bottom: 8px;",
        "        }",
        "        .tab-button {",
        "            background: transparent;",
        "            border: none;",
        "            padding: 6px 10px;",
        "            cursor: pointer;",
        "            font-size: 0.85rem;",
        "            color: #2c3e50;",
        "            border-bottom: 2px solid transparent;",
        "        }",
        "        .tab-button.active {",
        "            border-bottom-color: #3498db;",
        "            font-weight: 600;",
        "        }",
        "        .tab-panel { display: none; }",
        "        .tab-panel.active { display: block; }",
        "        .original-file {",
        "            background: #fff;",
        "            border-radius: 8px;",
        "            box-shadow: 0 2px 4px rgba(0,0,0,0.1);",
        "            margin-bottom: 30px;",
        "            padding: 20px;",
        "            border: 1px solid #e0e0e0;",
        "        }",
        "        .json-block {",
        "            background: #fff;",
        "            border-radius: 8px;",
        "            box-shadow: 0 2px 4px rgba(0,0,0,0.1);",
        "            margin-bottom: 30px;",
        "            padding: 20px;",
        "            border: 1px solid #e0e0e0;",
        "        }",
        "        .json-block h2 { margin-top: 0; color: #2c3e50; }",
        "        .original-file h2 { margin-top: 0; color: #2c3e50; }",
        "        .report-tabs { margin-top: 10px; }",
        "        .report-tab-buttons {",
        "            display: inline-flex;",
        "            gap: 6px;",
        "            border-bottom: 1px solid #e0e0e0;",
        "            margin-bottom: 8px;",
        "        }",
        "        .report-tab-button {",
        "            background: transparent;",
        "            border: none;",
        "            padding: 6px 10px;",
        "            cursor: pointer;",
        "            font-size: 0.85rem;",
        "            color: #2c3e50;",
        "            border-bottom: 2px solid transparent;",
        "        }",
        "        .report-tab-button.active {",
        "            border-bottom-color: #3498db;",
        "            font-weight: 600;",
        "        }",
        "        .report-tab-panel { display: none; }",
        "        .report-tab-panel.active { display: block; }",
        "        pre {",
        "            background: #f8f9fa;",
        "            padding: 15px;",
        "            border-radius: 4px;",
        "            border: 1px solid #e9ecef;",
        "            overflow-x: auto;",
        "            white-space: pre-wrap;",
        "            word-wrap: break-word;",
        "        }",
    ]

    html_content = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "    <meta charset='UTF-8'>",
        "    <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        "    <title>Sentence Grouping Report</title>",
        "    <style>",
        *css_lines,
        "    </style>",
        "</head>",
        "<body>",
        "    <h1>Sentence Grouping Report</h1>",
        "    <div class='original-file'>",
        f"        <h2>Original File: {escape(input_file.name)}</h2>",
        f"        <pre><code>{escape(source_text)}</code></pre>",
        "    </div>",
        "    <div class='json-block'>",
        "        <h2>Diagnostics</h2>",
        "        <div class='report-tabs'>",
        "            <div class='report-tab-buttons'>",
        "                <button class='report-tab-button active' "
        "data-tab='json'>Pipeline JSON</button>",
        "                <button class='report-tab-button' "
        "data-tab='trace'>Trace</button>",
        "            </div>",
        "            <div class='report-tab-panel active' data-tab-panel='json'>",
        f"                <pre><code>{escape(json_payload)}</code></pre>",
        "            </div>",
        "            <div class='report-tab-panel' data-tab-panel='trace'>",
        "                <pre><code>"
        f"{escape(trace_output if trace_output is not None else 'Trace disabled')}"
        "</code></pre>",
        "            </div>",
        "        </div>",
        "    </div>",
    ]

    for group in groups:
        labels: list[str] = [str(label) for label in group.get("label", [])]
        label_text = " > ".join(labels)

        html_content.append("    <div class='group'>")
        html_content.append("        <div class='group-header'>")
        html_content.append(f"            <div>{escape(label_text)}</div>")
        html_content.append("            <div class='group-labels'>")
        for label in labels:
            html_content.append(
                f"                <span class='label-tag'>{escape(label)}</span>"
            )
        html_content.append("            </div>")
        html_content.append("        </div>")
        html_content.append("        <ul class='sentences-list'>")

        for range_item in group.get("ranges", []):
            start = int(range_item["start"])
            end = int(range_item["end"])
            for sentence_index in range(start, end + 1):
                if sentence_index in sentence_map:
                    sentence = sentence_map[sentence_index]
                    original_slice = source_text[
                        int(sentence["start"]) : int(sentence["end"])
                    ]
                    sentence_text = escape(str(sentence["text"]))
                    html_content.append("            <li class='sentence-item'>")
                    html_content.append(
                        "                <span class='sentence-index'>"
                        f"#{int(sentence['index'])}</span>"
                    )
                    html_content.append("                <div class='sentence-tabs'>")
                    html_content.append("                    <div class='tab-buttons'>")
                    html_content.append(
                        "                        <button class='tab-button active' "
                        "data-tab='result'>Result</button>"
                    )
                    html_content.append(
                        "                        <button class='tab-button' "
                        "data-tab='original'>Original</button>"
                    )
                    html_content.append("                    </div>")
                    html_content.append(
                        "                    <div class='tab-panel active' "
                        "data-tab-panel='result'>"
                        f"<span class='sentence-text'>{sentence_text}</span>"
                        "</div>"
                    )
                    html_content.append(
                        "                    <div class='tab-panel' "
                        "data-tab-panel='original'>"
                        f"<span class='sentence-text'>{escape(original_slice)}</span>"
                        "</div>"
                    )
                    html_content.append("                </div>")
                    html_content.append("            </li>")

        html_content.append("        </ul>")
        html_content.append("    </div>")

    html_content.extend(
        [
            "    <script>",
            "        document.addEventListener('click', (event) => {",
            "            const button = event.target.closest('.tab-button');",
            "            if (!button) {",
            "                return;",
            "            }",
            "            const tabName = button.getAttribute('data-tab');",
            "            const tabs = button.closest('.sentence-tabs');",
            "            if (!tabs || !tabName) {",
            "                return;",
            "            }",
            "            tabs.querySelectorAll('.tab-button').forEach((btn) => {",
            "                btn.classList.toggle('active', btn === button);",
            "            });",
            "            tabs.querySelectorAll('.tab-panel').forEach((panel) => {",
            "                panel.classList.toggle(",
            "                    'active',",
            "                    panel.getAttribute('data-tab-panel') === tabName",
            "                );",
            "            });",
            "        });",
            "        document.addEventListener('click', (event) => {",
            "            const button = event.target.closest('.report-tab-button');",
            "            if (!button) {",
            "                return;",
            "            }",
            "            const tabName = button.getAttribute('data-tab');",
            "            const tabs = button.closest('.report-tabs');",
            "            if (!tabs || !tabName) {",
            "                return;",
            "            }",
            "            tabs.querySelectorAll('.report-tab-button')"
            ".forEach((btn) => {",
            "                btn.classList.toggle('active', btn === button);",
            "            });",
            "            tabs.querySelectorAll('.report-tab-panel')"
            ".forEach((panel) => {",
            "                panel.classList.toggle(",
            "                    'active',",
            "                    panel.getAttribute('data-tab-panel') === tabName",
            "                );",
            "            });",
            "        });",
            "    </script>",
            "</body>",
            "</html>",
        ]
    )
    return "\n".join(html_content)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split text using LLamaCPP and txt_splitt pipeline"
    )
    parser.add_argument("input_file", help="Input text file to split")
    parser.add_argument(
        "--host",
        required=True,
        help="LLamaCPP server host (e.g., http://localhost:8080)",
    )
    parser.add_argument(
        "--model", default="default", help="Model name (default: default)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Temperature (default: 0.0)"
    )
    parser.add_argument(
        "--anchor-words",
        type=int,
        default=5,
        help="Add a marker anchor roughly every N words (default: 5)",
    )
    parser.add_argument(
        "--max-chunk-chars",
        type=int,
        default=84000,
        help="Max characters per LLM chunk (default: 84000)",
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
        help="Use async LLM client with concurrent request limiting",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Max concurrent requests for async mode (default: 10)",
    )
    parser.add_argument(
        "--single-stage",
        action="store_true",
        help="Use single-stage LLM split instead of two-stage",
    )
    args = parser.parse_args()

    if args.use_async:
        asyncio.run(run_async(args))
    else:
        run_sync(args)


def create_pipeline(
    args: Any,
    input_path: Path,
    sync_llm: LLMCallable,
    async_llm: Any = None,
    tracer: Tracer | None = None,
) -> Pipeline:
    """Create pipeline with appropriate configuration."""
    splitter = SparseRegexSentenceSplitter(anchor_every_words=args.anchor_words)
    html_cleaner = None
    offset_restorer = None
    if input_path.suffix.lower() in {".html", ".htm"}:
        html_cleaner = HTMLParserTagStripCleaner(strip_tags={"style", "script"})
        offset_restorer = MappingOffsetRestorer()

    if args.single_stage:
        return Pipeline(
            splitter=splitter,
            marker=OptimizingMarker(BracketMarker()),
            llm=TopicRangeLLM(
                client=sync_llm,
                temperature=args.temperature,
                chunker=OverlapChunker(max_chars=args.max_chunk_chars),
            ),
            parser=TopicRangeParser(),
            gap_handler=LLMRepairingGapHandler(
                sync_llm, temperature=args.temperature, tracer=tracer
            ),
            joiner=AdjacentSameTopicJoiner(),
            html_cleaner=html_cleaner,
            offset_restorer=offset_restorer,
            tracer=tracer,
        )
    else:
        return Pipeline(
            splitter=splitter,
            marker=OptimizingMarker(BracketMarker()),
            topic_extractor=TopicListLLM(
                client=sync_llm,
                temperature=args.temperature,
                chunker=OverlapChunker(max_chars=args.max_chunk_chars),
                tracer=tracer,
            ),
            range_assigner=TopicRangeAssignmentLLM(
                client=async_llm or sync_llm,
                temperature=args.temperature,
                chunker=OverlapChunker(max_chars=args.max_chunk_chars),
                max_concurrent_requests=args.max_concurrent if async_llm else None,
                tracer=tracer,
            ),
            parser=TopicRangeParser(),
            gap_handler=LLMRepairingGapHandler(
                sync_llm, temperature=args.temperature, tracer=tracer
            ),
            joiner=AdjacentSameTopicJoiner(),
            html_cleaner=html_cleaner,
            offset_restorer=offset_restorer,
            tracer=tracer,
        )


def run_sync(args: Any) -> None:
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{args.input_file}' not found", file=sys.stderr)
        sys.exit(1)

    text = input_path.read_text(encoding="utf-8")

    llm_client = LLamaCPP(host=args.host, model=args.model)
    llm_adapter = LLamaCPPAdapter(llm_client)
    llm_with_retry: LLMCallable = RetryingLLMCallable(
        llm_adapter, max_retries=3, backoff_factor=1.0
    )

    tracer: Tracer | None = Tracer() if args.trace else None
    llm_callable: LLMCallable = llm_with_retry
    if tracer is not None:
        llm_callable = TracingLLMCallable(llm_with_retry, tracer)

    pipeline = create_pipeline(args, input_path, llm_callable, tracer=tracer)

    print(f"Processing '{args.input_file}'...")
    trace_output: str | None = None
    try:
        result = pipeline.run(text)
    except Exception as e:
        print(f"Error processing text: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if tracer:
            trace_output = tracer.format()
            print(trace_output, file=sys.stderr)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(f"{input_path.stem}_report_{timestamp}.html")
    report_html = generate_html_report(result, text, input_path, trace_output)
    output_file.write_text(report_html, encoding="utf-8")

    print(f"Results saved to '{output_file}'")
    print(f"  - Sentences: {len(result.sentences)}")
    print(f"  - Groups: {len(result.groups)}")


async def run_async(args: Any) -> None:
    """Run pipeline with async LLM client."""
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{args.input_file}' not found", file=sys.stderr)
        sys.exit(1)

    text = input_path.read_text(encoding="utf-8")

    sync_llm_client = LLamaCPP(host=args.host, model=args.model)
    sync_llm_adapter = LLamaCPPAdapter(sync_llm_client)
    async_llm_client = AsyncLLamaCPP(host=args.host, model=args.model)
    async_llm_adapter = AsyncLLamaCPPAdapter(async_llm_client)

    tracer: Tracer | None = Tracer() if args.trace else None
    sync_llm_callable: LLMCallable = RetryingLLMCallable(
        sync_llm_adapter, max_retries=3, backoff_factor=1.0
    )
    async_llm_callable = async_llm_adapter
    if tracer is not None:
        sync_llm_callable = TracingLLMCallable(sync_llm_callable, tracer)
        async_llm_callable = TracingAsyncLLMCallable(async_llm_adapter, tracer)

    pipeline = create_pipeline(
        args, input_path, sync_llm_callable, async_llm_callable, tracer
    )

    print(
        f"Processing '{args.input_file}' (async mode, "
        f"max {args.max_concurrent} concurrent)..."
    )
    trace_output: str | None = None
    try:
        result = await pipeline.run_async(text)
    except Exception as e:
        print(f"Error processing text: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if tracer:
            trace_output = tracer.format()
            print(trace_output, file=sys.stderr)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(f"{input_path.stem}_report_{timestamp}.html")
    report_html = generate_html_report(result, text, input_path, trace_output)
    output_file.write_text(report_html, encoding="utf-8")

    print(f"Results saved to '{output_file}'")
    print(f"  - Sentences: {len(result.sentences)}")
    print(f"  - Groups: {len(result.groups)}")


if __name__ == "__main__":
    main()
