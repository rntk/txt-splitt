#!/usr/bin/env python3
"""Simple script to split text using LLamaCPP client and txt_splitt pipeline."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add src to path so we can import txt_splitt when running from root
sys.path.append(str(Path(__file__).parent / "src"))

from txt_splitt import (
    BracketMarker,
    Pipeline,
    RegexSentenceSplitter,
    StrictGapHandler,
    TopicRangeLLM,
    TopicRangeParser,
)
from txt_splitt.llms.llamacpp import LLamaCPP


class LLamaCPPAdapter:
    """Adapter to make LLamaCPP compatible with LLMCallable protocol."""

    def __init__(self, client: LLamaCPP) -> None:
        self._client = client

    def call(self, prompt: str, temperature: float) -> str:
        """Call the LLM with a prompt and temperature."""
        print("===" * 10)
        print(f"DEBUG: LLM Prompt:\n{prompt}")
        print("===" * 10)
        response = self._client.call([prompt], temperature=temperature)
        print("===" * 10)
        print(f"DEBUG: LLM Response:\n{response}")
        print("===" * 10)
        return response


def result_to_dict(result: Any) -> dict:
    """Convert SplitResult to a dictionary for JSON serialization."""
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

    # Create pipeline
    pipeline = Pipeline(
        splitter=RegexSentenceSplitter(),
        marker=BracketMarker(),
        llm=TopicRangeLLM(llm_adapter, temperature=args.temperature),
        parser=TopicRangeParser(),
        gap_handler=StrictGapHandler(),
    )

    # Run pipeline
    print(f"Processing '{args.input_file}'...")
    try:
        result = pipeline.run(text)
    except Exception as e:
        print(f"Error processing text: {e}", file=sys.stderr)
        sys.exit(1)

    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_stem = input_path.stem
    output_file = f"{input_stem}_{timestamp}.json"

    # Convert result to dict and save
    output_data = result_to_dict(result)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"✓ Results saved to '{output_file}'")
    print(f"  - Sentences: {len(result.sentences)}")
    print(f"  - Groups: {len(result.groups)}")


if __name__ == "__main__":
    main()
