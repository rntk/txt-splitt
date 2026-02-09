#!/usr/bin/env python3
"""Simple script to split text using LLamaCPP client and txt_splitt pipeline."""

import argparse
import sys
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

# Add src to path so we can import txt_splitt when running from root
sys.path.append(str(Path(__file__).parent / "src"))

from txt_splitt import (
    BracketMarker,
    DenseRegexSentenceSplitter,
    LLMCallable,
    LLMRepairingGapHandler,
    NormalizingSplitter,
    Pipeline,
    TopicRangeLLM,
    TopicRangeParser,
    Tracer,
    TracingLLMCallable,
)
from txt_splitt.llms.llamacpp import LLamaCPP


class LLamaCPPAdapter:
    """Adapter to make LLamaCPP compatible with LLMCallable protocol."""

    def __init__(self, client: LLamaCPP) -> None:
        self._client = client

    def call(self, prompt: str, temperature: float) -> str:
        """Call the LLM with a prompt and temperature."""
        return self._client.call([prompt], temperature=temperature)


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


def generate_html_report(result: Any, source_text: str, input_file: Path) -> str:
    """Generate HTML report content from split result."""
    data = result_to_dict(result)
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
        "        .original-file {",
        "            background: #fff;",
        "            border-radius: 8px;",
        "            box-shadow: 0 2px 4px rgba(0,0,0,0.1);",
        "            margin-bottom: 30px;",
        "            padding: 20px;",
        "            border: 1px solid #e0e0e0;",
        "        }",
        "        .original-file h2 { margin-top: 0; color: #2c3e50; }",
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
                    html_content.append("            <li class='sentence-item'>")
                    html_content.append(
                        "                <span class='sentence-index'>"
                        f"#{int(sentence['index'])}</span>"
                    )
                    html_content.append(
                        "                <span class='sentence-text'>"
                        f"{escape(str(sentence['text']))}</span>"
                    )
                    html_content.append("            </li>")

        html_content.append("        </ul>")
        html_content.append("    </div>")

    html_content.extend(["</body>", "</html>"])
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
        help="Add a marker anchor roughly every N words (default: 22)",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Enable tracing and print the trace after the run",
    )
    args = parser.parse_args()

    # Read input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{args.input_file}' not found", file=sys.stderr)
        sys.exit(1)

    text = input_path.read_text(encoding="utf-8")

    # Create LLM client and adapter
    llm_client = LLamaCPP(host=args.host, model=args.model)
    llm_adapter = LLamaCPPAdapter(llm_client)

    # Set up tracing if requested
    tracer: Tracer | None = Tracer() if args.trace else None
    llm_callable: LLMCallable = llm_adapter
    if tracer is not None:
        llm_callable = TracingLLMCallable(llm_adapter, tracer)

    # Create pipeline
    pipeline = Pipeline(
        splitter=NormalizingSplitter(
            DenseRegexSentenceSplitter(anchor_every_words=args.anchor_words),
            min_length=20,
            max_length=260,
        ),
        marker=BracketMarker(),
        llm=TopicRangeLLM(llm_callable, temperature=args.temperature),
        parser=TopicRangeParser(),
        gap_handler=LLMRepairingGapHandler(
            llm_callable, temperature=args.temperature, tracer=tracer
        ),
        tracer=tracer,
    )

    # Run pipeline
    print(f"Processing '{args.input_file}'...")
    try:
        result = pipeline.run(text)
    except Exception as e:
        print(f"Error processing text: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if tracer:
            print(tracer.format(), file=sys.stderr)

    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_stem = input_path.stem
    output_file = Path(f"{input_stem}_report_{timestamp}.html")

    # Generate and save report
    report_html = generate_html_report(result, text, input_path)
    output_file.write_text(report_html, encoding="utf-8")

    print(f"Results saved to '{output_file}'")
    print(f"  - Sentences: {len(result.sentences)}")
    print(f"  - Groups: {len(result.groups)}")


if __name__ == "__main__":
    main()
