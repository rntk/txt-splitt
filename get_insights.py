#!/usr/bin/env python3
"""Extract key insights from text using LLamaCPP and the txt_splitt pipeline."""

import argparse
import json
import sys
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

# Add src to path so we can import txt_splitt when running from root
sys.path.append(str(Path(__file__).parent / "src"))

from txt_splitt import (
    CachingLLMCallable,
    LLMCallable,
    SQLiteLLMCacheStore,
    Tracer,
    TracingLLMCallable,
)
from txt_splitt.errors import LLMError
from txt_splitt.html_cleaners import HTMLParserTagStripCleaner
from txt_splitt.insights import InsightParser, InsightResult, build_insight_llm
from txt_splitt.llms.llamacpp import LLamaCPP
from txt_splitt.pipeline import PendingStage
from txt_splitt.protocols import LLMResponse
from txt_splitt.sentences import (
    BracketMarker,
    OptimizingMarker,
    OverlapChunker,
    SparseRegexSentenceSplitter,
)


def _drive_llm(
    llm: Any,
    marked_text: Any,
    client: LLMCallable,
) -> str:
    """Drive plan_query() by executing requests against a client."""
    stage: Any = llm.plan_query(marked_text)
    while isinstance(stage, PendingStage):
        responses = []
        for request in stage.requests:
            try:
                content = client.call(request.prompt, request.temperature)
            except LLMError:
                raise
            except Exception as e:
                raise LLMError(f"LLM call failed: {e}") from e
            responses.append(LLMResponse(content=str(content)))
        stage = stage.resume(responses)
    result: str = stage.value
    return result


class LLamaCPPAdapter:
    """Adapter to make LLamaCPP compatible with LLMCallable protocol."""

    def __init__(self, client: LLamaCPP) -> None:
        self._client = client

    def call(self, prompt: str, temperature: float) -> str:
        return str(self._client.call([prompt], temperature=temperature))


def build_cache_store(args: Any) -> SQLiteLLMCacheStore | None:
    cache_db = getattr(args, "cache_db", None)
    if not cache_db:
        return None
    return SQLiteLLMCacheStore(cache_db)


def wrap_llm(
    llm: LLMCallable,
    *,
    namespace: str,
    args: Any,
    tracer: Tracer | None = None,
    cache_store: SQLiteLLMCacheStore | None = None,
) -> LLMCallable:
    wrapped: LLMCallable = llm
    if cache_store is not None:
        wrapped = CachingLLMCallable(
            wrapped,
            cache_store,
            namespace=namespace,
            model_id=str(args.model),
            prompt_version="get_insights_v1",
            cache_nonzero_temperature=bool(args.cache_nonzero_temperature),
        )
    if tracer is not None:
        wrapped = TracingLLMCallable(wrapped, tracer)
    return wrapped


def result_to_dict(result: InsightResult) -> dict[str, Any]:
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
        "insights": [
            {
                "name": insight.name,
                "ranges": [{"start": r.start, "end": r.end} for r in insight.ranges],
            }
            for insight in result.insights
        ],
    }


def generate_html_report(
    result: InsightResult,
    source_text: str,
    input_file: Path,
    trace_output: str | None = None,
) -> str:
    data = result_to_dict(result)
    json_payload = json.dumps(data, indent=2)
    sentences: list[dict[str, Any]] = data.get("sentences", [])
    insights: list[dict[str, Any]] = data.get("insights", [])

    sentence_map: dict[int, dict[str, Any]] = {int(s["index"]): s for s in sentences}

    css_lines = [
        "        body {",
        "            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',",
        "                Roboto, Helvetica, Arial, sans-serif;",
        "            line-height: 1.6;",
        "            color: #333;",
        "            max-width: 1000px;",
        "            margin: 0 auto;",
        "            padding: 20px;",
        "            background-color: #fdf6ec;",
        "        }",
        "        h1 { color: #2c3e50; text-align: center; }",
        "        .insight {",
        "            background: #fff;",
        "            border-radius: 8px;",
        "            box-shadow: 0 2px 4px rgba(0,0,0,0.1);",
        "            margin-bottom: 30px;",
        "            overflow: hidden;",
        "            border: 1px solid #e0e0e0;",
        "        }",
        "        .insight-header {",
        "            background: #e67e22;",
        "            color: #fff;",
        "            padding: 15px 20px;",
        "            margin: 0;",
        "            font-size: 1.15rem;",
        "            font-weight: 600;",
        "        }",
        "        .sentences-list { padding: 0; margin: 0; list-style: none; }",
        "        .sentence-item {",
        "            padding: 15px 20px;",
        "            border-bottom: 1px solid #eee;",
        "            transition: background-color 0.2s;",
        "        }",
        "        .sentence-item:last-child { border-bottom: none; }",
        "        .sentence-item:hover { background-color: #fef9f4; }",
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
        "            border-bottom-color: #e67e22;",
        "            font-weight: 600;",
        "        }",
        "        .tab-panel { display: none; }",
        "        .tab-panel.active { display: block; }",
        "        .json-block {",
        "            background: #fff;",
        "            border-radius: 8px;",
        "            box-shadow: 0 2px 4px rgba(0,0,0,0.1);",
        "            margin-bottom: 30px;",
        "            padding: 20px;",
        "            border: 1px solid #e0e0e0;",
        "        }",
        "        .json-block h2 { margin-top: 0; color: #2c3e50; }",
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
        "            border-bottom-color: #e67e22;",
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
        "    <title>Insights Report</title>",
        "    <style>",
        *css_lines,
        "    </style>",
        "</head>",
        "<body>",
        f"    <h1>Insights Report &mdash; {escape(input_file.name)}</h1>",
        f"    <p style='text-align:center;color:#7f8c8d'>"
        f"{len(insights)} insights from {len(sentences)} sentences</p>",
        "    <div class='json-block'>",
        "        <h2>Diagnostics</h2>",
        "        <div class='report-tabs'>",
        "            <div class='report-tab-buttons'>",
        "                <button class='report-tab-button active' "
        "data-tab='json'>Pipeline JSON</button>",
        "                <button class='report-tab-button' "
        "data-tab='trace'>Trace</button>",
        "                <button class='report-tab-button' "
        f"data-tab='original'>Original File: {escape(input_file.name)}</button>",
        "            </div>",
        "            <div class='report-tab-panel active' data-tab-panel='json'>",
        f"                <pre><code>{escape(json_payload)}</code></pre>",
        "            </div>",
        "            <div class='report-tab-panel' data-tab-panel='trace'>",
        "                <pre><code>"
        f"{escape(trace_output if trace_output is not None else 'Trace disabled')}"
        "</code></pre>",
        "            </div>",
        "            <div class='report-tab-panel' data-tab-panel='original'>",
        f"                <pre><code>{escape(source_text)}</code></pre>",
        "            </div>",
        "        </div>",
        "    </div>",
    ]

    for insight_dict in insights:
        name = str(insight_dict.get("name", ""))
        html_content.append("    <div class='insight'>")
        html_content.append(f"        <div class='insight-header'>{escape(name)}</div>")
        html_content.append("        <ul class='sentences-list'>")

        for range_item in insight_dict.get("ranges", []):
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


def run(args: Any) -> None:
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{args.input_file}' not found", file=sys.stderr)
        sys.exit(1)

    raw_text = input_path.read_text(encoding="utf-8")

    # Strip HTML tags if needed; work with plain text for the pipeline
    if input_path.suffix.lower() in {".html", ".htm"}:
        cleaner = HTMLParserTagStripCleaner(strip_tags={"style", "script"})
        cleaned_text, _ = cleaner.clean(raw_text)
    else:
        cleaned_text = raw_text

    # Split and mark
    splitter = SparseRegexSentenceSplitter(
        anchor_every_words=args.anchor_words,
        long_sentence_word_threshold=args.long_sentence_threshold,
        min_sentence_words=args.min_sentence_words,
    )
    marker = OptimizingMarker(BracketMarker())
    sentences_list = splitter.split(cleaned_text)
    marked_text = marker.mark(cleaned_text, sentences_list)

    # Set up LLM client
    llm_client = LLamaCPP(host=args.host, model=args.model)
    llm_adapter = LLamaCPPAdapter(llm_client)

    tracer: Tracer | None = Tracer() if args.trace else None
    cache_store = build_cache_store(args)

    wrapped_llm = wrap_llm(
        llm_adapter,
        namespace="insights",
        args=args,
        tracer=tracer,
        cache_store=cache_store,
    )

    insight_llm = build_insight_llm(
        temperature=args.temperature,
        chunker=OverlapChunker(max_chars=args.max_chunk_chars),
    )
    parser = InsightParser()

    print(f"Processing '{args.input_file}'...")
    trace_output: str | None = None
    try:
        raw_response = _drive_llm(insight_llm, marked_text, wrapped_llm)
        insights_list = parser.parse(raw_response, marked_text.sentence_count)
    except Exception as e:
        print(f"Error processing text: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if tracer:
            trace_output = tracer.format()
            print(trace_output, file=sys.stderr)

    result = InsightResult(
        sentences=tuple(sentences_list),
        insights=tuple(insights_list),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(f"{input_path.stem}_insights_{timestamp}.html")
    report_html = generate_html_report(result, cleaned_text, input_path, trace_output)
    output_file.write_text(report_html, encoding="utf-8")

    print(f"Results saved to '{output_file}'")
    print(f"  - Sentences: {len(result.sentences)}")
    print(f"  - Insights: {len(result.insights)}")
    for insight in result.insights:
        ranges_str = ", ".join(f"{r.start}-{r.end}" for r in insight.ranges)
        print(f"    • {insight.name} [{ranges_str}]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract key insights from text using LLamaCPP and txt_splitt"
    )
    parser.add_argument("input_file", help="Input text or HTML file")
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
        help="Max characters per LLM chunk (default: 84000)",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Enable tracing and print the trace after the run",
    )
    parser.add_argument(
        "--cache-db",
        help=(
            "SQLite database path for persistent LLM response caching. "
            "Disabled when omitted."
        ),
    )
    parser.add_argument(
        "--cache-nonzero-temperature",
        action="store_true",
        help="Also cache requests with non-zero temperature",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
