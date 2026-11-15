"""Exercise benchmark orchestration without a running model server."""

import json
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest

import benchmark_text


def test_discovery_deduplicates_and_sorts(tmp_path: Path) -> None:
    (tmp_path / "b.MD").write_text("b")
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "skip.pdf").write_text("x")
    assert benchmark_text.discover([str(tmp_path), str(tmp_path / "a.txt")]) == [
        (tmp_path / "a.txt").resolve(),
        (tmp_path / "b.MD").resolve(),
    ]


def test_runs_continue_on_failure_and_report_only(
    tmp_path: Path, monkeypatch: Any
) -> None:
    source = tmp_path / "input.txt"
    source.write_text("Hello.", encoding="utf-8")
    out = tmp_path / "out"
    calls: list[int] = []

    def fake_run(args: Any, path: Path, text: str) -> dict[str, Any]:
        calls.append(1)
        failed = len(calls) == 2
        return {
            "status": "error" if failed else "ok",
            "elapsed_seconds": 0.1,
            "error": "failure" if failed else None,
            "metrics": None,
            "llm": {"count": 0, "elapsed_seconds": 0, "failures": 0, "namespaces": {}},
        }

    monkeypatch.setattr(benchmark_text, "execute_one", fake_run)
    assert (
        benchmark_text.main(
            [
                str(source),
                "--host",
                "http://localhost:8080",
                "--repeats",
                "2",
                "--warmup",
                "1",
                "--output-dir",
                str(out),
            ]
        )
        == 1
    )
    artifact = next(out.glob("*.json"))
    data = json.loads(artifact.read_text())
    assert len(data["warmups"]) == 1
    assert [run["status"] for run in data["runs"]] == ["error", "ok"]
    assert data["config"]["cache_enabled"] is False
    assert len(data["documents"][0]["sha256"]) == 64
    assert next(out.glob("*.html")).exists()
    assert (
        benchmark_text.main(
            ["--report-only", str(artifact), "--output-dir", str(tmp_path / "report")]
        )
        == 1
    )
    assert len(calls) == 3


def test_timed_client_counts_failed_calls() -> None:
    class FailingClient:
        def call(self, prompt: str, temperature: float) -> str:
            raise RuntimeError("failure")

    stats = benchmark_text.CallStats()
    client = benchmark_text.TimedSync(FailingClient(), "topic-range", stats)
    with suppress(RuntimeError):
        client.call("prompt", 0.0)
    assert stats.data["count"] == 1
    assert stats.data["failures"] == 1
    assert stats.data["namespaces"]["topic-range"]["failures"] == 1


def test_help_runs_without_pythonpath() -> None:
    import os
    import subprocess
    import sys

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "benchmark_text.py", "--help"],
        cwd=Path(__file__).resolve().parent.parent,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_invalid_utf8_is_recorded_and_other_documents_run(
    tmp_path: Path, monkeypatch: Any
) -> None:
    (tmp_path / "bad.txt").write_bytes(b"\xff")
    (tmp_path / "good.txt").write_text("ok")
    monkeypatch.setattr(
        benchmark_text,
        "execute_one",
        lambda *args: {
            "status": "ok",
            "elapsed_seconds": 0.1,
            "error": None,
            "metrics": None,
            "llm": {"count": 0, "elapsed_seconds": 0, "failures": 0, "namespaces": {}},
        },
    )
    output = tmp_path / "out"
    assert (
        benchmark_text.main(
            [
                str(tmp_path),
                "--host",
                "http://localhost",
                "--repeats",
                "1",
                "--output-dir",
                str(output),
            ]
        )
        == 1
    )
    data = json.loads(next(output.glob("*.json")).read_text())
    assert [run["status"] for run in data["runs"]] == ["error", "ok"]
    assert data["documents"][0]["input_chars"] is None


def _ok_run() -> dict[str, Any]:
    return {
        "status": "ok",
        "elapsed_seconds": 0.1,
        "error": None,
        "metrics": None,
        "llm": {"count": 0, "elapsed_seconds": 0, "failures": 0, "namespaces": {}},
    }


def test_results_are_saved_after_every_file(tmp_path: Path, monkeypatch: Any) -> None:
    inputs = tmp_path / "in"
    inputs.mkdir()
    for name in ("a.txt", "b.txt", "c.txt"):
        (inputs / name).write_text(name)
    out = tmp_path / "out"
    snapshots: list[list[str]] = []

    def fake_run(args: Any, path: Path, text: str) -> dict[str, Any]:
        files = list(out.glob("*.json"))
        if files:
            data = json.loads(files[0].read_text())
            assert data["complete"] is False
            snapshots.append([run["document_id"] for run in data["runs"]])
        return _ok_run()

    monkeypatch.setattr(benchmark_text, "execute_one", fake_run)
    argv = [str(inputs), "--host", "http://x", "--repeats", "1"]
    assert benchmark_text.main([*argv, "--output-dir", str(out)]) == 0
    # Before b and c ran, the file already held the previous documents' runs.
    assert [len(s) for s in snapshots] == [1, 2]
    assert len(list(out.glob("*.json"))) == 1
    assert not list(out.glob("*.tmp"))
    assert json.loads(next(out.glob("*.json")).read_text())["complete"] is True


def test_interrupt_keeps_finished_files(tmp_path: Path, monkeypatch: Any) -> None:
    inputs = tmp_path / "in"
    inputs.mkdir()
    for name in ("a.txt", "b.txt"):
        (inputs / name).write_text(name)
    out = tmp_path / "out"
    calls: list[int] = []

    def fake_run(args: Any, path: Path, text: str) -> dict[str, Any]:
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return _ok_run()

    monkeypatch.setattr(benchmark_text, "execute_one", fake_run)
    code = benchmark_text.main(
        [str(inputs), "--host", "http://x", "--repeats", "1", "--output-dir", str(out)]
    )
    assert code == 130
    data = json.loads(next(out.glob("*.json")).read_text())
    assert data["complete"] is False
    assert len(data["runs"]) == 1
    assert next(out.glob("*.html")).exists()


def _write_benchmark_artifact(path: Path, label: str, model: str) -> Path:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at": "2026-11-15T00:00:00+00:00",
        "label": label,
        "model": model,
        "host": "http://localhost:8080",
        "config": {"cache_enabled": False},
        "complete": True,
        "documents": [{"id": "doc.html", "sha256": "abc", "input_chars": 100}],
        "runs": [
            {
                "document_id": "doc.html",
                "repeat": 1,
                "status": "ok",
                "elapsed_seconds": 1.0,
                "error": None,
                "metrics": {"topic_count": 2},
                "llm": {"count": 1, "elapsed_seconds": 0.5, "failures": 0},
            }
        ],
        "warmups": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_compare_two_files_without_running_benchmark(tmp_path: Path) -> None:
    file1 = _write_benchmark_artifact(tmp_path / "a.json", "run-alpha", "model-alpha")
    file2 = _write_benchmark_artifact(tmp_path / "b.json", "run-beta", "model-beta")

    for index, argv_compare in enumerate(
        [
            ["--compare", str(file1), str(file2)],
            ["--compare", str(file1), "--compare", str(file2)],
        ]
    ):
        out_dir = tmp_path / f"out_{index}"
        assert benchmark_text.main([*argv_compare, "--output-dir", str(out_dir)]) == 0
        json_file = next(out_dir.glob("*.json"))
        html_file = next(out_dir.glob("*.html"))
        assert json.loads(json_file.read_text(encoding="utf-8"))["label"] == "run-alpha"
        html = html_file.read_text(encoding="utf-8")
        assert "run-alpha · model-alpha" in html
        assert "run-beta · model-beta" in html


def test_compare_single_file_without_inputs_errors(tmp_path: Path) -> None:
    file1 = _write_benchmark_artifact(tmp_path / "a.json", "run-alpha", "model-alpha")
    with pytest.raises(SystemExit) as exc_info:
        benchmark_text.main(["--compare", str(file1)])
    assert exc_info.value.code == 2
