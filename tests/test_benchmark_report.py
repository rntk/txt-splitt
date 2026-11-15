"""Offline report coverage, including failure-only data and unsafe labels."""

from copy import deepcopy
from typing import Any

from txt_splitt.benchmark_report import render_report


def _payload() -> dict[str, Any]:
    return {
        "label": "<script>alert('x')</script>",
        "model": "local-model",
        "config": {"cache_enabled": False},
        "documents": [{"id": "doc", "sha256": "abc", "input_chars": 100}],
        "runs": [
            {
                "document_id": "doc",
                "repeat": i,
                "status": "ok",
                "elapsed_seconds": duration,
                "metrics": {
                    "topic_count": 3,
                    "range_count": 4,
                    "coverage_rate": 0.9,
                    "position_bins": [{"topic_count": j} for j in range(10)],
                    "topic_char_size": {"min": 2, "median": 5, "mean": 6, "max": 9},
                    "hierarchy_child_count": {
                        "min": 0,
                        "median": 1,
                        "mean": 2,
                        "max": 4,
                    },
                    "topics": [{"label": ["<img src=x onerror=alert(1)>"]}],
                },
                "llm": {"count": 2, "elapsed_seconds": 8, "failures": 0},
            }
            for i, duration in enumerate([1, 3], 1)
        ],
    }


def test_report_statistics_charts_and_safe_offline_markup() -> None:
    html = render_report(_payload(), [])
    assert "<script>" not in html
    assert "<img" not in html
    assert "&lt;script&gt;" in html
    assert "&lt;img" in html
    assert "https://" not in html
    assert "src=" not in html.replace("&lt;img src=x", "")
    assert "<td>2</td><td>2.9</td><td>1</td>" in html
    assert "<td>66.667</td>" in html
    assert html.count("<meter ") == 39
    assert "90–100%" in html
    assert "Immediate children" in html
    assert "Mean coverage fraction" in html
    assert "doc: mean topic count" in html
    assert "<td>3</td><td>4</td><td>6</td>" in html
    assert "semantic quality" in html
    assert "splitter-defined" in html
    assert "cumulative call time" in html
    assert "server-side prompt caching" in html


def test_report_comparison_warnings_and_failure_details() -> None:
    payload = _payload()
    comparison = deepcopy(payload)
    comparison["config"] = {"cache_enabled": False, "temperature": 1}
    comparison["documents"][0]["sha256"] = "changed"
    comparison["warmups"] = [
        {"document_id": "doc", "status": "error", "error": "warmup bad"}
    ]
    comparison["runs"] = [
        {
            "document_id": "doc",
            "repeat": 1,
            "status": "error",
            "elapsed_seconds": 99,
            "error": "<bad failure>",
            "metrics": None,
        }
    ]
    html = render_report(payload, [comparison])
    assert "Corpus mismatch" in html
    assert "Warmup failure" in html
    assert "warmup bad" in html
    assert "Configuration mismatch" in html
    assert "&lt;bad failure&gt;" in html
    assert "<td>—</td>" in html
    assert "99 s" in html
    assert "<td>99</td><td>99</td>" not in html


def test_empty_report_is_complete_document() -> None:
    html = render_report({}, [])
    assert html.startswith("<!doctype html>")
    assert html.endswith("</html>")
    assert "No observations" in html
