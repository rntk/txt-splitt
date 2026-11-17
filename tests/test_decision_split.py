from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from split_text import execute_session_sync
from split_text_decisions import DecisionAdapter, create_pipeline
from txt_splitt.errors import LLMError, ParseError
from txt_splitt.llms.llamacpp_decisions import DecisionHTTPError, LLamaCPPDecisions
from txt_splitt.pipeline import CompletedStage, PendingStage, StageResult
from txt_splitt.protocols import LLMRequest, LLMResponse
from txt_splitt.sentences import (
    DecisionMarker,
    DecisionTopicRangeLLM,
    RepairingGapHandler,
    Sentence,
    SentenceRange,
    TopicRangeParser,
    build_pipeline,
)
from txt_splitt.sentences.decision_split import (
    DECISION_NAMESPACE,
    LABEL_NAMESPACE,
    batch_ranges_for_labels,
    boundaries_to_ranges,
    fit_text_to_chars,
    parse_topic_labels,
    read_split_probability,
)

TEXT = (
    "Apple released a phone. It has a fast chip.\n\n"
    "The Fed raised rates. Markets fell after that."
)


def _sentences(text: str) -> list[Sentence]:
    parts = ["Apple released a phone.", "It has a fast chip."]
    parts += ["The Fed raised rates.", "Markets fell after that."]
    sentences: list[Sentence] = []
    cursor = 0
    for index, part in enumerate(parts):
        start = text.index(part, cursor)
        sentences.append(Sentence(index, start, start + len(part), part))
        cursor = start + len(part)
    return sentences


def _answer(split: float) -> dict[str, Any]:
    return {
        "choice": "split" if split >= 0.5 else "continue",
        "probabilities": {"split": split, "continue": 1 - split},
    }


def _decision_responder(
    split_by_id: dict[int, float],
) -> Callable[[LLMRequest], str]:
    def respond(request: LLMRequest) -> str:
        questions = json.loads(request.prompt)["questions"]
        return json.dumps(
            {"answers": {qid: _answer(split_by_id[int(qid[1:])]) for qid in questions}}
        )

    return respond


def _drive(
    stage: StageResult[str],
    decide: Callable[[LLMRequest], str],
    label: Callable[[LLMRequest], str],
) -> tuple[str, list[LLMRequest]]:
    seen: list[LLMRequest] = []
    while isinstance(stage, PendingStage):
        seen.extend(stage.requests)
        stage = stage.resume(
            [
                LLMResponse(
                    content=decide(r)
                    if r.metadata["namespace"] == DECISION_NAMESPACE
                    else label(r)
                )
                for r in stage.requests
            ]
        )
    assert isinstance(stage, CompletedStage)
    return stage.value, seen


def _label_all(request: LLMRequest) -> str:
    count = request.prompt.count("\n[")
    return "\n".join(f"{i}: Topic>Section {i}" for i in range(1, count + 1))


# -- helpers ----------------------------------------------------------------


def test_fit_text_to_chars_keeps_head_and_tail() -> None:
    assert fit_text_to_chars("abcdefghij", 20) == "abcdefghij"
    assert fit_text_to_chars("abcdefghij", 5) == "ab…ij"
    assert fit_text_to_chars("abcdefghij", 2) == "a…"
    assert fit_text_to_chars("abcdefghij", 1) == "a"


def test_boundaries_to_ranges() -> None:
    assert boundaries_to_ranges(0, []) == []
    assert boundaries_to_ranges(4, [False, True, False]) == [
        SentenceRange(0, 1),
        SentenceRange(2, 3),
    ]
    with pytest.raises(ValueError):
        boundaries_to_ranges(3, [True])


def test_read_split_probability_validates_answer() -> None:
    assert read_split_probability({"b2": _answer(0.75)}, "b2") == 0.75
    bad = {"choice": "split", "probabilities": {"split": 0.9, "continue": 0.9}}
    with pytest.raises(LLMError):
        read_split_probability({"b2": bad}, "b2")
    with pytest.raises(LLMError):
        read_split_probability({}, "b2")


def test_parse_topic_labels_handles_formats_and_missing() -> None:
    response = "1: Tech>Apple: Phone\n**[3]** - Business>Fed\nnoise"
    assert parse_topic_labels(response, 3) == [
        ["Tech", "Apple Phone"],
        None,
        ["Business", "Fed"],
    ]


def test_batch_ranges_for_labels_respects_char_budget() -> None:
    texts = ["x" * 30] * 4
    ranges = [SentenceRange(i, i) for i in range(4)]
    batches = batch_ranges_for_labels(ranges, texts, max_chars=80)
    assert [len(b.ranges) for b in batches] == [2, 2]
    assert all(len(b.text) <= 80 for b in batches)


# -- strategy ---------------------------------------------------------------


def test_plan_query_splits_on_threshold_and_labels_ranges() -> None:
    marked = DecisionMarker().mark(TEXT, _sentences(TEXT))
    strategy = DecisionTopicRangeLLM(threshold=0.5, batch_size=2)
    value, seen = _drive(
        strategy.plan_query(marked),
        _decision_responder({2: 0.1, 3: 0.9, 4: 0.2}),
        _label_all,
    )

    decision_requests = [
        r for r in seen if r.metadata["namespace"] == DECISION_NAMESPACE
    ]
    assert len(decision_requests) == 2  # 3 gaps in batches of 2
    first = json.loads(decision_requests[0].prompt)
    assert list(first["questions"]) == ["b2", "b3"]
    assert [s["paragraph_break_before"] for s in first["state"]["sentences"]] == [
        False,
        False,
        True,
        False,
    ]
    assert strategy.last_probabilities == [0.1, 0.9, 0.2]
    assert value == "Topic>Section 1: 0-1\nTopic>Section 2: 2-3"


def test_missing_labels_are_reasked_alone() -> None:
    marked = DecisionMarker().mark(TEXT, _sentences(TEXT))
    calls: list[str] = []

    def label(request: LLMRequest) -> str:
        calls.append(request.prompt)
        return "1: Tech>Phone" if len(calls) == 1 else "1: Business>Fed"

    value, _ = _drive(
        DecisionTopicRangeLLM().plan_query(marked),
        _decision_responder({2: 0.0, 3: 1.0, 4: 0.0}),
        label,
    )
    assert value == "Tech>Phone: 0-1\nBusiness>Fed: 2-3"
    assert "The Fed raised rates." in calls[1]
    assert "Apple released" not in calls[1]


def test_labels_fail_after_retries() -> None:
    marked = DecisionMarker().mark(TEXT, _sentences(TEXT))
    with pytest.raises(ParseError):
        _drive(
            DecisionTopicRangeLLM(label_max_retries=1).plan_query(marked),
            _decision_responder({2: 0.0, 3: 0.0, 4: 0.0}),
            lambda _: "nothing useful",
        )


def test_context_window_shrinks_batches() -> None:
    marked = DecisionMarker().mark(TEXT, _sentences(TEXT))
    stage = DecisionTopicRangeLLM(batch_size=8, context_window_tokens=2000).plan_query(
        marked
    )
    assert isinstance(stage, PendingStage)
    assert len(stage.requests) > 1
    for request in stage.requests:
        payload = json.loads(request.prompt)
        assert len(payload["questions"]) >= 1


def test_single_sentence_skips_decisions() -> None:
    text = "Only one sentence here."
    marked = DecisionMarker().mark(text, [Sentence(0, 0, len(text), text)])
    value, seen = _drive(
        DecisionTopicRangeLLM().plan_query(marked),
        _decision_responder({}),
        lambda _: "1: Misc>Single",
    )
    assert value == "Misc>Single: 0-0"
    assert all(r.metadata["namespace"] == LABEL_NAMESPACE for r in seen)


def test_plan_query_requires_decision_marker() -> None:
    from txt_splitt.sentences import MarkedText

    with pytest.raises(TypeError):
        DecisionTopicRangeLLM().plan_query(MarkedText("", 0))


def test_full_pipeline_merges_returning_topic() -> None:
    text = "Phones are new. Rates went up. Phones sell well."
    pipeline = build_pipeline(
        splitter=_FixedSplitter(),
        marker=DecisionMarker(),
        llm=DecisionTopicRangeLLM(),
        parser=TopicRangeParser(),
        gap_handler=RepairingGapHandler(),
    )
    session = pipeline.start(text)
    decide = _decision_responder({2: 0.9, 3: 0.9})
    while not session.is_complete():
        responses = []
        for request in session.pending_requests():
            if request.metadata["namespace"] == DECISION_NAMESPACE:
                responses.append(LLMResponse(decide(request)))
            else:
                responses.append(
                    LLMResponse("1: Tech>Phones\n2: Business>Rates\n3: Tech>Phones")
                )
        session.submit_responses(responses)
    groups = session.result().groups
    assert [(g.label, g.ranges) for g in groups] == [
        (("Tech", "Phones"), (SentenceRange(0, 0), SentenceRange(2, 2))),
        (("Business", "Rates"), (SentenceRange(1, 1),)),
    ]


class _FixedSplitter:
    def split(self, text: str) -> list[Sentence]:
        sentences: list[Sentence] = []
        start = 0
        for index, part in enumerate(text.split(". ")):
            piece = part if part.endswith(".") else f"{part}."
            begin = text.index(piece, start)
            sentences.append(Sentence(index, begin, begin + len(piece), piece))
            start = begin + len(piece)
        return sentences


# -- script and client ------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    statuses: list[int] = []
    bodies: list[dict[str, Any]] = []

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        type(self).bodies.append({"path": self.path, **body})
        status = type(self).statuses.pop(0) if type(self).statuses else 200
        if status != 200:
            payload = b"busy"
        else:
            payload = json.dumps(
                {
                    "answers": {
                        qid: _answer(1.0 if qid == "b3" else 0.0)
                        for qid in body["questions"]
                    }
                }
            ).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:
        del format, args


@pytest.fixture
def decision_server() -> Iterator[str]:
    _Handler.statuses = []
    _Handler.bodies = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_decision_client_posts_and_retries_transient(decision_server: str) -> None:
    _Handler.statuses = [503]
    client = LLamaCPPDecisions(decision_server, model="sys1")
    result = client.decide({"x": 1}, {"b2": {"type": "choice"}})
    assert "b2" in result["answers"]
    assert [b["path"] for b in _Handler.bodies] == ["/v1/systemone"] * 2
    assert _Handler.bodies[-1]["model"] == "sys1"


def test_decision_client_raises_on_client_error(decision_server: str) -> None:
    _Handler.statuses = [400]
    with pytest.raises(DecisionHTTPError) as info:
        LLamaCPPDecisions(decision_server).decide({}, {"b2": {"type": "choice"}})
    assert info.value.status == 400
    assert len(_Handler.bodies) == 1


def test_script_pipeline_end_to_end(decision_server: str, tmp_path: Path) -> None:
    args = SimpleNamespace(
        anchor_words=12,
        long_sentence_threshold=24,
        min_sentence_words=1,
        threshold=0.5,
        decision_batch_size=8,
        decision_context_sentences=2,
        decision_context_window=0,
        max_chunk_chars=84_000,
        temperature=0.0,
    )
    pipeline = create_pipeline(args, tmp_path / "input.txt")
    clients = {
        DECISION_NAMESPACE: DecisionAdapter(LLamaCPPDecisions(decision_server)),
        LABEL_NAMESPACE: SimpleNamespace(
            call=lambda prompt, temperature: "1: Tech>Phone\n2: Business>Fed"
        ),
    }
    result = execute_session_sync(pipeline, TEXT, clients)  # type: ignore[arg-type]
    assert [g.label for g in result.groups] == [("Tech", "Phone"), ("Business", "Fed")]
