#!/usr/bin/env python3
"""Script to extract keywords from text using LLamaCPP client and KeywordPipeline."""

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
    HTMLParserTagStripCleaner,
    KeywordExtractionLLM,
    KeywordGapValidator,
    KeywordIndexParser,
    KeywordPipeline,
    LLMCallable,
    RegexWordSplitter,
    RetryingLLMCallable,
    TracingLLMCallable,
    WordBracketMarker,
    WordOverlapChunker,
)
from txt_splitt.llms.llamacpp import LLamaCPP
from txt_splitt.tracer import Tracer


class LLamaCPPAdapter:
    """Adapter to make LLamaCPP compatible with LLMCallable protocol."""

    def __init__(self, client: LLamaCPP) -> None:
        self._client = client

    def call(self, prompt: str, temperature: float) -> str:
        """Call the LLM with a prompt and temperature."""
        return self._client.call([prompt], temperature=temperature)  # type: ignore[no-any-return]


def result_to_dict(result: Any) -> dict[str, Any]:
    """Convert KeywordResult to a dictionary."""
    return {
        "keywords": [
            {
                "text": kw.text,
                "start": kw.start,
                "end": kw.end,
            }
            for kw in result.keywords
        ],
        "words": [
            {
                "index": w.index,
                "start": w.start,
                "end": w.end,
                "text": w.text,
            }
            for w in result.words
        ],
    }


def generate_html_report(
    result: Any,
    source_text: str,
    input_file: Path,
    trace_output: str | None = None,
) -> str:
    """Generate HTML report content from keyword result."""
    data = result_to_dict(result)
    json_payload = json.dumps(data, indent=2)
    keywords: list[dict[str, Any]] = data.get("keywords", [])

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
        "        .section {",
        "            background: #fff;",
        "            border-radius: 8px;",
        "            box-shadow: 0 2px 4px rgba(0,0,0,0.1);",
        "            margin-bottom: 30px;",
        "            padding: 20px;",
        "            border: 1px solid #e0e0e0;",
        "        }",
        "        .section h2 { margin-top: 0; color: #2c3e50; }",
        "        .highlighted-text {",
        "            line-height: 2.2;",
        "            font-size: 1rem;",
        "            white-space: pre-wrap;",
        "            word-wrap: break-word;",
        "        }",
        "        mark {",
        "            background-color: #ffe082;",
        "            border-radius: 3px;",
        "            padding: 1px 3px;",
        "            font-weight: 600;",
        "        }",
        "        .keyword-list {",
        "            display: flex;",
        "            flex-wrap: wrap;",
        "            gap: 8px;",
        "            margin-top: 10px;",
        "        }",
        "        .keyword-tag {",
        "            background: #3498db;",
        "            color: #fff;",
        "            padding: 4px 12px;",
        "            border-radius: 20px;",
        "            font-size: 0.9rem;",
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

    # Build highlighted text by inserting <mark> tags around keyword spans
    highlighted = _highlight_text(source_text, keywords)

    # Keyword tags list
    keyword_tags = "".join(
        f"<span class='keyword-tag'>{escape(kw['text'])}</span>" for kw in keywords
    )

    html_content = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "    <meta charset='UTF-8'>",
        "    <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        "    <title>Keyword Extraction Report</title>",
        "    <style>",
        *css_lines,
        "    </style>",
        "</head>",
        "<body>",
        "    <h1>Keyword Extraction Report</h1>",
        "    <div class='section'>",
        f"        <h2>File: {escape(input_file.name)}</h2>",
        f"        <p>{len(keywords)} keyword(s) extracted</p>",
        "        <div class='keyword-list'>",
        f"            {keyword_tags}",
        "        </div>",
        "    </div>",
        "    <div class='section'>",
        "        <h2>Highlighted Text</h2>",
        f"        <div class='highlighted-text'>{highlighted}</div>",
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
        "    <script>",
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
        "            tabs.querySelectorAll('.report-tab-button').forEach((btn) => {",
        "                btn.classList.toggle('active', btn === button);",
        "            });",
        "            tabs.querySelectorAll('.report-tab-panel').forEach((panel) => {",
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

    return "\n".join(html_content)


def _highlight_text(text: str, keywords: list[dict[str, Any]]) -> str:
    """Insert <mark> tags around keyword spans in text, sorted by start offset."""
    # Sort by start, then by descending end (so overlapping ranges handled stably)
    spans = sorted(
        [(int(kw["start"]), int(kw["end"])) for kw in keywords],
        key=lambda x: x[0],
    )

    # Merge overlapping spans
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    parts: list[str] = []
    pos = 0
    for start, end in merged:
        if pos < start:
            parts.append(escape(text[pos:start]))
        parts.append(f"<mark>{escape(text[start:end])}</mark>")
        pos = end
    if pos < len(text):
        parts.append(escape(text[pos:]))

    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract keywords from text using LLamaCPP and KeywordPipeline"
    )
    parser.add_argument("input_file", help="Input text file to process")
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
        "--max-chunk-chars",
        type=int,
        default=12000,
        help="Max characters per LLM chunk (default: 12000)",
    )
    parser.add_argument(
        "--overlap-words",
        type=int,
        default=20,
        help="Overlap words between chunks (default: 20)",
    )
    parser.add_argument(
        "--max-keywords",
        type=int,
        default=50,
        help="Max keywords to extract (default: 50)",
    )
    parser.add_argument(
        "--min-gap-words",
        type=int,
        default=0,
        help=(
            "Enable gap validator: re-query LLM for text regions with >= N "
            "consecutive uncovered words. 0 = disabled (default: 0)"
        ),
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Print JSON result to stdout instead of generating HTML report",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Print pipeline trace to stderr after processing",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{args.input_file}' not found", file=sys.stderr)
        sys.exit(1)

    text = input_path.read_text(encoding="utf-8")

    tracer: Tracer | None = Tracer() if args.trace else None

    llm_client = LLamaCPP(host=args.host, model=args.model)
    llm_adapter = LLamaCPPAdapter(llm_client)
    llm_callable: LLMCallable = RetryingLLMCallable(
        llm_adapter, max_retries=3, backoff_factor=1.0
    )
    if tracer is not None:
        llm_callable = TracingLLMCallable(llm_callable, tracer)

    html_cleaner = None
    if input_path.suffix.lower() in {".html", ".htm"}:
        html_cleaner = HTMLParserTagStripCleaner(strip_tags={"style", "script"})

    gap_validator = None
    if args.min_gap_words > 0:
        gap_validator = KeywordGapValidator(
            word_splitter=RegexWordSplitter(),
            marker=WordBracketMarker(),
            llm=KeywordExtractionLLM(
                llm_callable,
                chunker=WordOverlapChunker(
                    max_chars=args.max_chunk_chars,
                    overlap_words=args.overlap_words,
                ),
                temperature=args.temperature,
                max_keywords=args.max_keywords,
            ),
            parser=KeywordIndexParser(),
            min_gap_words=args.min_gap_words,
            tracer=tracer,
        )

    pipeline = KeywordPipeline(
        word_splitter=RegexWordSplitter(),
        marker=WordBracketMarker(),
        llm=KeywordExtractionLLM(
            llm_callable,
            chunker=WordOverlapChunker(
                max_chars=args.max_chunk_chars,
                overlap_words=args.overlap_words,
            ),
            temperature=args.temperature,
            max_keywords=args.max_keywords,
        ),
        parser=KeywordIndexParser(),
        html_cleaner=html_cleaner,
        gap_validator=gap_validator,
        tracer=tracer,
    )

    print(f"Processing '{args.input_file}'...", file=sys.stderr)
    trace_output: str | None = None
    try:
        result = pipeline.run(text)
    except Exception as e:
        print(f"Error processing text: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if tracer is not None:
            trace_output = tracer.format()
            print("\n--- Trace ---", file=sys.stderr)
            print(trace_output, file=sys.stderr)

    if args.json_output:
        print(json.dumps(result_to_dict(result), indent=2))
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(f"{input_path.stem}_keywords_{timestamp}.html")
        report_html = generate_html_report(result, text, input_path, trace_output)
        output_file.write_text(report_html, encoding="utf-8")
        print(f"Results saved to '{output_file}'")

    print(f"  - Words:    {len(result.words)}", file=sys.stderr)
    print(f"  - Keywords: {len(result.keywords)}", file=sys.stderr)
    for kw in result.keywords:
        print(f"    [{kw.start}:{kw.end}] {kw.text!r}", file=sys.stderr)


if __name__ == "__main__":
    main()
