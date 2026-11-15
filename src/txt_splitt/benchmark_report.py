"""Dependency-free, offline HTML presentation of benchmark observations."""

from __future__ import annotations

import json
import math
import statistics
from html import escape
from typing import Any


def _text(value: Any) -> str:
    return escape(str(value), quote=True)


def _number(value: Any) -> str:
    if not isinstance(value, (float, int)) or not math.isfinite(value):
        return "—"
    return f"{value:,.3f}".rstrip("0").rstrip(".")


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low = math.floor(index)
    high = math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def _stats(values: list[float]) -> tuple[str, str, str]:
    if not values:
        return "—", "—", "—"
    return (
        _number(_percentile(values, 0.5)),
        _number(_percentile(values, 0.95)),
        _number(statistics.pstdev(values)),
    )


def _table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th scope='col'>{_text(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    if not rows:
        body = f"<tr><td colspan='{len(headers)}'>No observations</td></tr>"
    return (
        f"<div class='scroll'><table><thead><tr>{head}</tr></thead>"
        f"<tbody>{body}</tbody></table></div>"
    )


def _bars(title: str, labels: list[str], values: list[float], unit: str) -> str:
    maximum = max(values, default=0) or 1
    rows = "".join(
        f"<li><span>{_text(label)}</span><meter min='0' max='{maximum}' "
        f"value='{value}' aria-label='{_text(label)}: {_number(value)} "
        f"{_text(unit)}'></meter><strong>{_number(value)} {_text(unit)}</strong></li>"
        for label, value in zip(labels, values, strict=True)
    )
    return (
        f"<figure><figcaption>{_text(title)}</figcaption>"
        f"<ul class='bars'>{rows}</ul></figure>"
    )


def _runs(payload: dict[str, Any], document_id: Any = None) -> list[dict[str, Any]]:
    return [
        run
        for run in payload.get("runs", [])
        if document_id is None or run.get("document_id") == document_id
    ]


def _latencies(runs: list[dict[str, Any]]) -> list[float]:
    return [
        float(run["elapsed_seconds"])
        for run in runs
        if run.get("status") == "ok"
        and isinstance(run.get("elapsed_seconds"), (int, float))
    ]


def _identity(payload: dict[str, Any]) -> str:
    return (
        f"{payload.get('label', 'Benchmark')} · {payload.get('model', 'unknown model')}"
    )


def _warnings(payloads: list[dict[str, Any]]) -> str:
    warnings: list[str] = []
    baseline = payloads[0]
    corpus = sorted(
        (d.get("id", ""), d.get("sha256", "")) for d in baseline.get("documents", [])
    )
    for other in payloads[1:]:
        other_corpus = sorted(
            (d.get("id", ""), d.get("sha256", "")) for d in other.get("documents", [])
        )
        if corpus != other_corpus:
            warnings.append(
                f"Corpus mismatch: {_identity(other)}. Inputs differ; timing and "
                "structure are not directly comparable."
            )
        if baseline.get("config") != other.get("config"):
            warnings.append(
                f"Configuration mismatch: {_identity(other)}. "
                "Review configuration before comparing."
            )
    return "".join(f"<p class='warning'>{_text(warning)}</p>" for warning in warnings)


def _structure_charts(metrics: dict[str, Any]) -> str:
    charts: list[str] = []
    bins = metrics.get("position_bins", [])
    if bins:
        charts.append(
            _bars(
                "Topic density by sentence position (10 equal-position bins)",
                [f"{i * 10}–{(i + 1) * 10}%" for i in range(len(bins))],
                [float(item.get("topic_count", 0)) for item in bins],
                "topics",
            )
        )
    for key, title, unit in [
        ("topic_sentence_size", "Topic sizes", "sentences"),
        ("topic_char_size", "Topic sizes", "chars"),
        ("range_sentence_size", "Range sizes", "sentences"),
        ("range_char_size", "Range sizes", "chars"),
        (
            "hierarchy_child_count",
            "Immediate children per hierarchy node (subtopics)",
            "children",
        ),
    ]:
        summary = metrics.get(key, {})
        if summary:
            labels = ["min", "median", "mean", "max"]
            charts.append(
                _bars(title, labels, [float(summary.get(k) or 0) for k in labels], unit)
            )
    return "".join(charts)


def _metrics_detail(metrics: dict[str, Any]) -> str:
    """Show all metrics, preserving schema additions without hiding observations."""
    rows = [
        [_text(key), _text(json.dumps(value, ensure_ascii=False, sort_keys=True))]
        for key, value in metrics.items()
    ]
    return _table(["Structural metric", "Value"], rows)


def _document_comparisons(payloads: list[dict[str, Any]]) -> str:
    charts: list[str] = []
    identifiers = list(
        dict.fromkeys(d.get("id") for p in payloads for d in p.get("documents", []))
    )
    for identifier in identifiers:
        labels: list[str] = []
        times: list[float] = []
        topic_labels: list[str] = []
        topics: list[float] = []
        for payload in payloads:
            runs = _runs(payload, identifier)
            durations = _latencies(runs)
            if durations:
                labels.append(_identity(payload))
                times.append(statistics.median(durations))
            counts = [
                float(r["metrics"]["topic_count"])
                for r in runs
                if r.get("status") == "ok"
                and isinstance(
                    (r.get("metrics") or {}).get("topic_count"), (int, float)
                )
            ]
            if counts:
                topic_labels.append(_identity(payload))
                topics.append(statistics.mean(counts))
        charts.append(_bars(f"{identifier}: median latency", labels, times, "s"))
        charts.append(
            _bars(f"{identifier}: mean topic count", topic_labels, topics, "topics")
        )
    return "".join(charts)


def render_report(payload: dict[str, Any], comparisons: list[dict[str, Any]]) -> str:
    """Render current and prior results without scripts, remote assets, or raw HTML."""
    payloads = [payload, *comparisons]
    summary: list[list[str]] = []
    latency_labels: list[str] = []
    latency_values: list[float] = []
    sections: list[str] = []
    for index, result in enumerate(payloads):
        identity = _identity(result)
        runs = _runs(result)
        latencies = _latencies(runs)
        p50, p95, spread = _stats(latencies)
        failures = sum(run.get("status") != "ok" for run in runs)
        summary.append(
            [
                f"<a href='#result-{index}'>{_text(identity)}</a>",
                str(len(runs)),
                str(failures),
                _number(100 * failures / len(runs)) if runs else "—",
                p50,
                p95,
                spread,
            ]
        )
        if latencies:
            latency_labels.append(identity)
            latency_values.append(_percentile(latencies, 0.5))
        documents: list[list[str]] = []
        details: list[str] = []
        for document in result.get("documents", []):
            document_id = document.get("id")
            doc_runs = _runs(result, document_id)
            doc_times = _latencies(doc_runs)
            median, tail, variability = _stats(doc_times)
            throughput = [
                float(document.get("input_chars", 0)) / duration
                for duration in doc_times
                if duration > 0
            ]
            structure: list[str] = []
            for key in [
                "topic_count",
                "range_count",
                "topic_char_size",
                "topic_sentence_size",
                "hierarchy_child_count",
                "coverage_rate",
            ]:
                values: list[float] = []
                for observation in doc_runs:
                    if observation.get("status") != "ok":
                        continue
                    metric = (observation.get("metrics") or {}).get(key)
                    if isinstance(metric, dict):
                        metric = metric.get("mean")
                    if isinstance(metric, (int, float)):
                        values.append(float(metric))
                structure.append(_number(statistics.mean(values)) if values else "—")
            documents.append(
                [
                    _text(document_id),
                    _text(document.get("input_chars", "—")),
                    str(len(doc_runs)),
                    str(sum(r.get("status") != "ok" for r in doc_runs)),
                    median,
                    tail,
                    variability,
                    _number(statistics.median(throughput)) if throughput else "—",
                    *structure,
                ]
            )
            for run in doc_runs:
                llm = run.get("llm") or {}
                details.append(
                    f"<details><summary>{_text(document_id)} · repeat "
                    f"{_text(run.get('repeat'))} · {_text(run.get('status'))} · "
                    f"{_number(run.get('elapsed_seconds'))} s</summary>"
                    + _table(
                        [
                            "Wall time (s)",
                            "LLM calls",
                            "LLM cumulative time (s)",
                            "LLM failures",
                        ],
                        [
                            [
                                _number(run.get("elapsed_seconds")),
                                _number(llm.get("count")),
                                _number(llm.get("elapsed_seconds")),
                                _number(llm.get("failures")),
                            ]
                        ],
                    )
                    + (
                        f"<p class='warning'>{_text(run['error'])}</p>"
                        if run.get("error")
                        else ""
                    )
                    + _structure_charts(run.get("metrics") or {})
                    + _metrics_detail(run.get("metrics") or {})
                    + "</details>"
                )
        sections.append(
            f"<section id='result-{index}'><h2>{_text(identity)}</h2>"
            f"<p>Created: {_text(result.get('created_at', 'unknown'))} · "
            f"Host: {_text(result.get('host', 'unknown'))}</p>"
            + _table(
                [
                    "Document",
                    "Input chars",
                    "Runs",
                    "Failures",
                    "p50 (s)",
                    "p95 (s)",
                    "Std dev (s)",
                    "Median chars/s",
                    "Mean topics",
                    "Mean ranges",
                    "Mean topic chars",
                    "Mean topic sentences",
                    "Mean immediate children",
                    "Mean coverage fraction",
                ],
                documents,
            )
            + "<p>Structure columns are means across successful runs; "
            "size and child columns average each run’s mean.</p>"
            + "".join(
                f"<p class='warning'>Warmup failure · "
                f"{_text(w.get('document_id'))}: {_text(w.get('error'))}</p>"
                for w in result.get("warmups", [])
                if w.get("status") != "ok"
            )
            + "<h3>Run details</h3>"
            + "".join(details)
            + "<details><summary>Configuration and corpus</summary><pre>"
            + _text(
                json.dumps(
                    {
                        "config": result.get("config"),
                        "documents": result.get("documents"),
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            + "</pre></details></section>"
        )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Topic splitter benchmark</title><style>"
        "body{font:16px system-ui,sans-serif;max-width:1200px;margin:2rem "
        "auto;padding:0 1rem;color:#172338;background:#f6f8fc}"
        "h1,h2,h3{line-height:1.2}section,figure,details{background:white;"
        "border:1px solid #ccd5e3;border-radius:8px;padding:1rem;margin:1r"
        "em 0}"
        "table{border-collapse:collapse;width:100%;font-size:.9rem}th,td{t"
        "ext-align:left;padding:.65rem;border-bottom:1px solid #dce2ec;ver"
        "tical-align:top}"
        "th{background:#eaf0fa}.scroll{overflow:auto}td{overflow-wrap:anyw"
        "here}pre{white-space:pre-wrap;overflow-wrap:anywhere}"
        ".warning{background:#fff0cb;padding:1rem;border-left:4px solid #9"
        "b6500}.bars{padding:0;list-style:none}.bars li{display:grid;grid-"
        "template-columns:minmax(8rem,1fr) 2fr minmax(5rem,auto);gap:1rem;"
        "margin:.8rem 0;align-items:center}"
        "meter{width:100%;height:1.5rem}figcaption,summary{font-weight:650"
        "}summary{cursor:pointer}a{color:#1456a0}"
        "@media(max-width:600px){.bars li{grid-template-columns:1fr}body{m"
        "argin:1rem auto}}"
        "</style></head><body><main><h1>Topic splitter benchmark</h1>"
        "<p>Descriptive structure and performance measurements. These do n"
        "ot measure semantic quality.</p>"
        "<p>Application cache is disabled for benchmark runs; server-side "
        "prompt caching may still exist. "
        "Sentence units are splitter-defined. LLM time is cumulative call "
        "time, not wall-clock latency, "
        "and overlapping calls can exceed wall time. Warmups are excluded."
        " Latency percentiles, "
        "population standard deviation, and throughput use successful meas"
        "ured runs only; p95 uses linear interpolation.</p>"
        + _warnings(payloads)
        + _table(
            [
                "Run / model",
                "Attempts",
                "Failures",
                "Failure rate (%)",
                "p50 (s)",
                "p95 (s)",
                "Std dev (s)",
            ],
            summary,
        )
        + _bars(
            "Median successful document latency", latency_labels, latency_values, "s"
        )
        + _document_comparisons(payloads)
        + "".join(sections)
        + "</main></body></html>"
    )
