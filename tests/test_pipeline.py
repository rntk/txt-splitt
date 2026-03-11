"""Integration tests for the pipeline with stubs."""

import pytest

from txt_splitt.errors import (
    EnhancerError,
    GapError,
    LLMError,
    ParseError,
    SentenceSplitError,
)
from txt_splitt.pipeline import Pipeline
from txt_splitt.types import (
    MarkedText,
    OffsetMapping,
    OffsetSegment,
    Sentence,
    SentenceGroup,
    SentenceRange,
    SplitResult,
)


class StubSplitter:
    def __init__(self, sentences: list[Sentence]) -> None:
        self._sentences = sentences

    def split(self, text: str) -> list[Sentence]:
        return self._sentences


class StubMarker:
    def __init__(self, result: MarkedText) -> None:
        self._result = result

    def mark(self, text: str, sentences: list[Sentence]) -> MarkedText:
        return self._result


class StubLLM:
    def __init__(self, response: str) -> None:
        self._response = response

    def query(self, marked_text: MarkedText) -> str:
        return self._response


class JsonStubLLM(StubLLM):
    response_format = "json"


class StubParser:
    def __init__(self, groups: list[SentenceGroup]) -> None:
        self._groups = groups

    def parse(self, response: str, sentence_count: int) -> list[SentenceGroup]:
        return self._groups


class JsonStubParser(StubParser):
    supported_response_formats = frozenset({"json"})


class AutoStubParser(StubParser):
    supported_response_formats = frozenset({"text", "json"})


class StubGapHandler:
    def __init__(self, groups: list[SentenceGroup]) -> None:
        self._groups = groups

    def handle(
        self,
        groups: list[SentenceGroup],
        sentence_count: int,
        sentences: list[Sentence] | None = None,
    ) -> list[SentenceGroup]:
        return self._groups


class FailingSplitter:
    def split(self, text: str) -> list[Sentence]:
        raise SentenceSplitError("Splitter failed")


class FailingLLM:
    def query(self, marked_text: MarkedText) -> str:
        raise LLMError("LLM unavailable")


class FailingParser:
    def parse(self, response: str, sentence_count: int) -> list[SentenceGroup]:
        raise ParseError("Cannot parse")


class FailingGapHandler:
    def handle(
        self,
        groups: list[SentenceGroup],
        sentence_count: int,
        sentences: list[Sentence] | None = None,
    ) -> list[SentenceGroup]:
        raise GapError("Gap found")


class RecordingParser:
    def __init__(self, groups: list[SentenceGroup]) -> None:
        self._groups = groups
        self.seen_response: str | None = None
        self.seen_sentence_count: int | None = None

    def parse(self, response: str, sentence_count: int) -> list[SentenceGroup]:
        self.seen_response = response
        self.seen_sentence_count = sentence_count
        return self._groups


class RecordingGapHandler:
    def __init__(self, groups: list[SentenceGroup]) -> None:
        self._groups = groups
        self.seen_groups: list[SentenceGroup] | None = None
        self.seen_sentence_count: int | None = None
        self.seen_sentences: list[Sentence] | None = None

    def handle(
        self,
        groups: list[SentenceGroup],
        sentence_count: int,
        sentences: list[Sentence] | None = None,
    ) -> list[SentenceGroup]:
        self.seen_groups = groups
        self.seen_sentence_count = sentence_count
        self.seen_sentences = sentences
        return self._groups


class StubEnhancer:
    def __init__(self, groups: list[SentenceGroup]) -> None:
        self._groups = groups

    def enhance(
        self, groups: list[SentenceGroup], sentences: list[Sentence]
    ) -> list[SentenceGroup]:
        return self._groups


class RecordingEnhancer:
    def __init__(self, groups: list[SentenceGroup]) -> None:
        self._groups = groups
        self.seen_groups: list[SentenceGroup] | None = None
        self.seen_sentences: list[Sentence] | None = None

    def enhance(
        self, groups: list[SentenceGroup], sentences: list[Sentence]
    ) -> list[SentenceGroup]:
        self.seen_groups = groups
        self.seen_sentences = sentences
        return self._groups


class FailingEnhancer:
    def enhance(
        self, groups: list[SentenceGroup], sentences: list[Sentence]
    ) -> list[SentenceGroup]:
        raise EnhancerError("Enhancement failed")


class RecordingJoiner:
    def __init__(self, groups: list[SentenceGroup]) -> None:
        self._groups = groups
        self.seen_groups: list[SentenceGroup] | None = None
        self.seen_sentences: list[Sentence] | None = None

    def join(
        self, groups: list[SentenceGroup], sentences: list[Sentence]
    ) -> list[SentenceGroup]:
        self.seen_groups = groups
        self.seen_sentences = sentences
        return self._groups


class FailingJoiner:
    def join(
        self, groups: list[SentenceGroup], sentences: list[Sentence]
    ) -> list[SentenceGroup]:
        raise RuntimeError("Join failed")


class RecordingMarker:
    def __init__(self, result: MarkedText) -> None:
        self._result = result
        self.seen_text: str | None = None
        self.seen_sentences: list[Sentence] | None = None

    def mark(self, text: str, sentences: list[Sentence]) -> MarkedText:
        self.seen_text = text
        self.seen_sentences = sentences
        return self._result


def _make_sentences(n: int) -> list[Sentence]:
    return [
        Sentence(index=i, start=i * 10, end=i * 10 + 5, text=f"Sent {i}.")
        for i in range(n)
    ]


def _make_groups() -> list[SentenceGroup]:
    return [
        SentenceGroup(
            label=("Technology", "AI"),
            ranges=(SentenceRange(start=0, end=2),),
        ),
    ]


class TestPipeline:
    def test_successful_run(self) -> None:
        sentences = _make_sentences(3)
        groups = _make_groups()

        pipeline = Pipeline(
            splitter=StubSplitter(sentences),
            marker=StubMarker(MarkedText(tagged_text="...", sentence_count=3)),
            llm=StubLLM("Technology>AI: 0-2"),
            parser=StubParser(groups),
            gap_handler=StubGapHandler(groups),
        )
        result = pipeline.run("Some text")

        assert isinstance(result, SplitResult)
        assert len(result.sentences) == 3
        assert len(result.groups) == 1
        assert result.groups[0].label == ("Technology", "AI")

    def test_result_is_immutable_tuples(self) -> None:
        sentences = _make_sentences(2)
        groups = _make_groups()

        pipeline = Pipeline(
            splitter=StubSplitter(sentences),
            marker=StubMarker(MarkedText(tagged_text="...", sentence_count=2)),
            llm=StubLLM("..."),
            parser=StubParser(groups),
            gap_handler=StubGapHandler(groups),
        )
        result = pipeline.run("text")

        assert isinstance(result.sentences, tuple)
        assert isinstance(result.groups, tuple)

    def test_constructor_rejects_incompatible_llm_and_parser_formats(self) -> None:
        with pytest.raises(
            ValueError, match="Incompatible llm/parser response formats"
        ):
            Pipeline(
                splitter=StubSplitter(_make_sentences(1)),
                marker=StubMarker(MarkedText(tagged_text="...", sentence_count=1)),
                llm=JsonStubLLM('{"topics":[]}'),
                parser=StubParser(_make_groups()),
                gap_handler=StubGapHandler(_make_groups()),
            )

    def test_constructor_accepts_json_llm_with_json_parser(self) -> None:
        Pipeline(
            splitter=StubSplitter(_make_sentences(1)),
            marker=StubMarker(MarkedText(tagged_text="...", sentence_count=1)),
            llm=JsonStubLLM('{"topics":[]}'),
            parser=JsonStubParser(_make_groups()),
            gap_handler=StubGapHandler(_make_groups()),
        )

    def test_constructor_accepts_json_llm_with_auto_parser(self) -> None:
        Pipeline(
            splitter=StubSplitter(_make_sentences(1)),
            marker=StubMarker(MarkedText(tagged_text="...", sentence_count=1)),
            llm=JsonStubLLM('{"topics":[]}'),
            parser=AutoStubParser(_make_groups()),
            gap_handler=StubGapHandler(_make_groups()),
        )

    def test_splitter_error_propagates(self) -> None:
        pipeline = Pipeline(
            splitter=FailingSplitter(),
            marker=StubMarker(MarkedText(tagged_text="...", sentence_count=0)),
            llm=StubLLM("response"),
            parser=StubParser([]),
            gap_handler=StubGapHandler([]),
        )
        with pytest.raises(SentenceSplitError):
            pipeline.run("text")

    def test_llm_error_propagates(self) -> None:
        sentences = _make_sentences(3)

        pipeline = Pipeline(
            splitter=StubSplitter(sentences),
            marker=StubMarker(MarkedText(tagged_text="...", sentence_count=3)),
            llm=FailingLLM(),
            parser=StubParser([]),
            gap_handler=StubGapHandler([]),
        )
        with pytest.raises(LLMError):
            pipeline.run("text")

    def test_parse_error_propagates(self) -> None:
        sentences = _make_sentences(3)

        pipeline = Pipeline(
            splitter=StubSplitter(sentences),
            marker=StubMarker(MarkedText(tagged_text="...", sentence_count=3)),
            llm=StubLLM("response"),
            parser=FailingParser(),
            gap_handler=StubGapHandler([]),
        )
        with pytest.raises(ParseError):
            pipeline.run("text")

    def test_gap_error_propagates(self) -> None:
        sentences = _make_sentences(3)
        groups = _make_groups()

        pipeline = Pipeline(
            splitter=StubSplitter(sentences),
            marker=StubMarker(MarkedText(tagged_text="...", sentence_count=3)),
            llm=StubLLM("response"),
            parser=StubParser(groups),
            gap_handler=FailingGapHandler(),
        )
        with pytest.raises(GapError):
            pipeline.run("text")

    def test_end_to_end_with_real_stages(self) -> None:
        """Integration test using real splitter, marker, parser, gap_handler."""
        from txt_splitt.gap_handlers import StrictGapHandler
        from txt_splitt.markers import BracketMarker
        from txt_splitt.parsers import TopicRangeParser
        from txt_splitt.splitters import SparseRegexSentenceSplitter

        text = "AI is growing fast. Climate change is real."

        llm_response = "Technology>AI: 0\nScience>Climate: 1"

        pipeline = Pipeline(
            splitter=SparseRegexSentenceSplitter(),
            marker=BracketMarker(),
            llm=StubLLM(llm_response),
            parser=TopicRangeParser(),
            gap_handler=StrictGapHandler(),
        )
        result = pipeline.run(text)

        assert len(result.sentences) == 2
        assert len(result.groups) == 2
        assert result.sentences[0].text == "AI is growing fast."
        assert result.sentences[1].text == "Climate change is real."
        assert result.groups[0].label == ("Technology", "AI")
        assert result.groups[1].label == ("Science", "Climate")

    def test_parser_and_gap_handler_use_marker_sentence_count(self) -> None:
        sentences = _make_sentences(3)
        groups = _make_groups()
        parser = RecordingParser(groups)
        gap_handler = RecordingGapHandler(groups)

        pipeline = Pipeline(
            splitter=StubSplitter(sentences),
            marker=StubMarker(MarkedText(tagged_text="...", sentence_count=7)),
            llm=StubLLM("Technology>AI: 0-6"),
            parser=parser,
            gap_handler=gap_handler,
        )

        pipeline.run("Some text")

        assert parser.seen_response == "Technology>AI: 0-6"
        assert parser.seen_sentence_count == 7
        assert gap_handler.seen_groups == groups
        assert gap_handler.seen_sentence_count == 7
        assert gap_handler.seen_sentences == sentences

    def test_marker_receives_original_text_and_splitter_output(self) -> None:
        text = "Alpha. Beta."
        sentences = _make_sentences(2)
        groups = _make_groups()
        marker = RecordingMarker(MarkedText(tagged_text="...", sentence_count=2))

        pipeline = Pipeline(
            splitter=StubSplitter(sentences),
            marker=marker,
            llm=StubLLM("Technology>AI: 0-1"),
            parser=StubParser(groups),
            gap_handler=StubGapHandler(groups),
        )

        pipeline.run(text)

        assert marker.seen_text == text
        assert marker.seen_sentences == sentences

    def test_pipeline_without_enhancer_works_unchanged(self) -> None:
        sentences = _make_sentences(3)
        groups = _make_groups()

        pipeline = Pipeline(
            splitter=StubSplitter(sentences),
            marker=StubMarker(MarkedText(tagged_text="...", sentence_count=3)),
            llm=StubLLM("Technology>AI: 0-2"),
            parser=StubParser(groups),
            gap_handler=StubGapHandler(groups),
        )
        result = pipeline.run("Some text")

        assert len(result.groups) == 1
        assert result.groups[0].label == ("Technology", "AI")

    def test_pipeline_with_enhancer_none_works(self) -> None:
        sentences = _make_sentences(3)
        groups = _make_groups()

        pipeline = Pipeline(
            splitter=StubSplitter(sentences),
            marker=StubMarker(MarkedText(tagged_text="...", sentence_count=3)),
            llm=StubLLM("..."),
            parser=StubParser(groups),
            gap_handler=StubGapHandler(groups),
            enhancers=None,
        )
        result = pipeline.run("text")

        assert len(result.groups) == 1

    def test_pipeline_with_enhancer_calls_enhance(self) -> None:
        sentences = _make_sentences(3)
        gap_groups = _make_groups()
        enhanced_groups = [
            SentenceGroup(
                label=("Science",),
                ranges=(SentenceRange(start=0, end=2),),
            ),
        ]
        enhancer = RecordingEnhancer(enhanced_groups)

        pipeline = Pipeline(
            splitter=StubSplitter(sentences),
            marker=StubMarker(MarkedText(tagged_text="...", sentence_count=3)),
            llm=StubLLM("..."),
            parser=StubParser(gap_groups),
            gap_handler=StubGapHandler(gap_groups),
            enhancers=[enhancer],
        )
        result = pipeline.run("text")

        assert enhancer.seen_groups == gap_groups
        assert enhancer.seen_sentences == sentences
        # Pipeline returns the enhancer's output
        assert result.groups[0].label == ("Science",)

    def test_multiple_enhancers_run_sequentially(self) -> None:
        sentences = _make_sentences(3)
        gap_groups = _make_groups()
        first_output = [
            SentenceGroup(label=("First",), ranges=(SentenceRange(start=0, end=1),)),
        ]
        second_output = [
            SentenceGroup(label=("Second",), ranges=(SentenceRange(start=0, end=2),)),
        ]
        first = RecordingEnhancer(first_output)
        second = RecordingEnhancer(second_output)

        pipeline = Pipeline(
            splitter=StubSplitter(sentences),
            marker=StubMarker(MarkedText(tagged_text="...", sentence_count=3)),
            llm=StubLLM("..."),
            parser=StubParser(gap_groups),
            gap_handler=StubGapHandler(gap_groups),
            enhancers=[first, second],
        )
        result = pipeline.run("text")

        # First enhancer receives gap handler output
        assert first.seen_groups == gap_groups
        # Second enhancer receives first enhancer's output
        assert second.seen_groups == first_output
        # Pipeline returns second enhancer's output
        assert result.groups[0].label == ("Second",)

    def test_multiple_enhancers_run_sequentially_async(self) -> None:
        import asyncio

        sentences = _make_sentences(3)
        gap_groups = _make_groups()
        first_output = [
            SentenceGroup(label=("First",), ranges=(SentenceRange(start=0, end=1),)),
        ]
        second_output = [
            SentenceGroup(label=("Second",), ranges=(SentenceRange(start=0, end=2),)),
        ]
        first = RecordingEnhancer(first_output)
        second = RecordingEnhancer(second_output)

        class AsyncStubRangeAssigner:
            async def assign_async(
                self, marked_text: MarkedText, topics: list[str]
            ) -> str:
                return "..."

        pipeline = Pipeline(
            splitter=StubSplitter(sentences),
            marker=StubMarker(MarkedText(tagged_text="...", sentence_count=3)),
            topic_extractor=StubTopicExtractor(["Technology>AI"]),
            range_assigner=AsyncStubRangeAssigner(),
            parser=StubParser(gap_groups),
            gap_handler=StubGapHandler(gap_groups),
            enhancers=[first, second],
        )
        result = asyncio.run(pipeline.run_async("text"))

        assert first.seen_groups == gap_groups
        assert second.seen_groups == first_output
        assert result.groups[0].label == ("Second",)

    def test_enhancer_error_propagates(self) -> None:
        sentences = _make_sentences(3)
        groups = _make_groups()

        pipeline = Pipeline(
            splitter=StubSplitter(sentences),
            marker=StubMarker(MarkedText(tagged_text="...", sentence_count=3)),
            llm=StubLLM("..."),
            parser=StubParser(groups),
            gap_handler=StubGapHandler(groups),
            enhancers=[FailingEnhancer()],
        )
        with pytest.raises(EnhancerError, match="Enhancement failed"):
            pipeline.run("text")

    def test_pipeline_with_joiner_none_works(self) -> None:
        sentences = _make_sentences(3)
        groups = _make_groups()

        pipeline = Pipeline(
            splitter=StubSplitter(sentences),
            marker=StubMarker(MarkedText(tagged_text="...", sentence_count=3)),
            llm=StubLLM("..."),
            parser=StubParser(groups),
            gap_handler=StubGapHandler(groups),
            joiner=None,
        )
        result = pipeline.run("text")

        assert len(result.groups) == 1

    def test_pipeline_with_joiner_calls_join(self) -> None:
        sentences = _make_sentences(3)
        gap_groups = _make_groups()
        joined_groups = [
            SentenceGroup(
                label=("Joined",),
                ranges=(SentenceRange(start=0, end=2),),
            ),
        ]
        joiner = RecordingJoiner(joined_groups)

        pipeline = Pipeline(
            splitter=StubSplitter(sentences),
            marker=StubMarker(MarkedText(tagged_text="...", sentence_count=3)),
            llm=StubLLM("..."),
            parser=StubParser(gap_groups),
            gap_handler=StubGapHandler(gap_groups),
            joiner=joiner,
        )
        result = pipeline.run("text")

        assert joiner.seen_groups == gap_groups
        assert joiner.seen_sentences == sentences
        assert result.groups[0].label == ("Joined",)
        assert len(result.sentences) == 1
        assert result.sentences[0].text == "Sent 0. Sent 1. Sent 2."
        assert result.groups[0].ranges == (SentenceRange(start=0, end=0),)

    def test_pipeline_runs_joiner_after_enhancer(self) -> None:
        sentences = _make_sentences(3)
        gap_groups = [
            SentenceGroup(
                label=("A",),
                ranges=(SentenceRange(start=0, end=1),),
            ),
        ]
        enhanced_groups = [
            SentenceGroup(
                label=("B",),
                ranges=(SentenceRange(start=0, end=2),),
            ),
        ]
        joiner = RecordingJoiner(enhanced_groups)

        pipeline = Pipeline(
            splitter=StubSplitter(sentences),
            marker=StubMarker(MarkedText(tagged_text="...", sentence_count=3)),
            llm=StubLLM("..."),
            parser=StubParser(gap_groups),
            gap_handler=StubGapHandler(gap_groups),
            enhancers=[StubEnhancer(enhanced_groups)],
            joiner=joiner,
        )
        pipeline.run("text")

        assert joiner.seen_groups == enhanced_groups

    def test_joiner_error_propagates(self) -> None:
        sentences = _make_sentences(3)
        groups = _make_groups()

        pipeline = Pipeline(
            splitter=StubSplitter(sentences),
            marker=StubMarker(MarkedText(tagged_text="...", sentence_count=3)),
            llm=StubLLM("..."),
            parser=StubParser(groups),
            gap_handler=StubGapHandler(groups),
            joiner=FailingJoiner(),
        )
        with pytest.raises(RuntimeError, match="Join failed"):
            pipeline.run("text")


class StubHtmlCleaner:
    def __init__(self, clean_text: str, mapping: OffsetMapping) -> None:
        self._clean_text = clean_text
        self._mapping = mapping
        self.seen_text: str | None = None

    def clean(self, text: str) -> tuple[str, OffsetMapping]:
        self.seen_text = text
        return self._clean_text, self._mapping


class StubOffsetRestorer:
    def __init__(self) -> None:
        self.seen_result: SplitResult | None = None
        self.seen_mapping: OffsetMapping | None = None

    def restore(self, result: SplitResult, mapping: OffsetMapping) -> SplitResult:
        self.seen_result = result
        self.seen_mapping = mapping
        return result


class RecordingSplitter:
    def __init__(self, sentences: list[Sentence]) -> None:
        self._sentences = sentences
        self.seen_text: str | None = None

    def split(self, text: str) -> list[Sentence]:
        self.seen_text = text
        return self._sentences


class StubTopicExtractor:
    def __init__(self, topics: list[str]) -> None:
        self._topics = topics

    def extract(self, marked_text: MarkedText) -> list[str]:
        return self._topics


class StubRangeAssigner:
    def __init__(self, response: str) -> None:
        self._response = response

    def assign(self, marked_text: MarkedText, topics: list[str]) -> str:
        return self._response


class JsonStubRangeAssigner(StubRangeAssigner):
    response_format = "json"


class FailingTopicExtractor:
    def extract(self, marked_text: MarkedText) -> list[str]:
        raise LLMError("Extraction failed")


class FailingRangeAssigner:
    def assign(self, marked_text: MarkedText, topics: list[str]) -> str:
        raise LLMError("Assignment failed")


class RecordingRangeAssigner:
    def __init__(self, response: str) -> None:
        self._response = response
        self.seen_topics: list[str] | None = None

    def assign(self, marked_text: MarkedText, topics: list[str]) -> str:
        self.seen_topics = topics
        return self._response


class TestTwoStagePipeline:
    def test_successful_run(self) -> None:
        sentences = _make_sentences(3)
        groups = _make_groups()

        pipeline = Pipeline(
            splitter=StubSplitter(sentences),
            marker=StubMarker(MarkedText(tagged_text="...", sentence_count=3)),
            topic_extractor=StubTopicExtractor(["Technology>AI", "Science>Climate"]),
            range_assigner=StubRangeAssigner("Technology>AI: 0-2"),
            parser=StubParser(groups),
            gap_handler=StubGapHandler(groups),
        )
        result = pipeline.run("Some text")

        assert isinstance(result, SplitResult)
        assert len(result.sentences) == 3
        assert len(result.groups) == 1
        assert result.groups[0].label == ("Technology", "AI")

    def test_extractor_output_passed_to_assigner(self) -> None:
        sentences = _make_sentences(3)
        groups = _make_groups()
        topics = ["Technology>AI>GPT-4", "Sport>Football>England"]
        assigner = RecordingRangeAssigner("Technology>AI>GPT-4: 0-2")

        pipeline = Pipeline(
            splitter=StubSplitter(sentences),
            marker=StubMarker(MarkedText(tagged_text="...", sentence_count=3)),
            topic_extractor=StubTopicExtractor(topics),
            range_assigner=assigner,
            parser=StubParser(groups),
            gap_handler=StubGapHandler(groups),
        )
        pipeline.run("text")

        assert assigner.seen_topics == topics

    def test_rejects_llm_with_topic_extractor(self) -> None:
        with pytest.raises(ValueError, match="Cannot provide both"):
            Pipeline(
                splitter=StubSplitter(_make_sentences(1)),
                marker=StubMarker(MarkedText(tagged_text=".", sentence_count=1)),
                llm=StubLLM("x"),
                topic_extractor=StubTopicExtractor(["Technology>AI"]),
                range_assigner=StubRangeAssigner("Technology>AI: 0"),
                parser=StubParser(_make_groups()),
                gap_handler=StubGapHandler(_make_groups()),
            )

    def test_rejects_llm_with_range_assigner(self) -> None:
        with pytest.raises(ValueError, match="Cannot provide both"):
            Pipeline(
                splitter=StubSplitter(_make_sentences(1)),
                marker=StubMarker(MarkedText(tagged_text=".", sentence_count=1)),
                llm=StubLLM("x"),
                range_assigner=StubRangeAssigner("Technology>AI: 0"),
                parser=StubParser(_make_groups()),
                gap_handler=StubGapHandler(_make_groups()),
            )

    def test_rejects_no_llm_path(self) -> None:
        with pytest.raises(ValueError, match="Must provide either"):
            Pipeline(
                splitter=StubSplitter(_make_sentences(1)),
                marker=StubMarker(MarkedText(tagged_text=".", sentence_count=1)),
                parser=StubParser(_make_groups()),
                gap_handler=StubGapHandler(_make_groups()),
            )

    def test_rejects_extractor_without_assigner(self) -> None:
        with pytest.raises(ValueError, match="Both .* must be provided"):
            Pipeline(
                splitter=StubSplitter(_make_sentences(1)),
                marker=StubMarker(MarkedText(tagged_text=".", sentence_count=1)),
                topic_extractor=StubTopicExtractor(["Technology>AI"]),
                parser=StubParser(_make_groups()),
                gap_handler=StubGapHandler(_make_groups()),
            )

    def test_rejects_assigner_without_extractor(self) -> None:
        with pytest.raises(ValueError, match="Both .* must be provided"):
            Pipeline(
                splitter=StubSplitter(_make_sentences(1)),
                marker=StubMarker(MarkedText(tagged_text=".", sentence_count=1)),
                range_assigner=StubRangeAssigner("Technology>AI: 0"),
                parser=StubParser(_make_groups()),
                gap_handler=StubGapHandler(_make_groups()),
            )

    def test_extractor_error_propagates(self) -> None:
        sentences = _make_sentences(3)

        pipeline = Pipeline(
            splitter=StubSplitter(sentences),
            marker=StubMarker(MarkedText(tagged_text="...", sentence_count=3)),
            topic_extractor=FailingTopicExtractor(),
            range_assigner=StubRangeAssigner("x"),
            parser=StubParser([]),
            gap_handler=StubGapHandler([]),
        )
        with pytest.raises(LLMError, match="Extraction failed"):
            pipeline.run("text")

    def test_assigner_error_propagates(self) -> None:
        sentences = _make_sentences(3)

        pipeline = Pipeline(
            splitter=StubSplitter(sentences),
            marker=StubMarker(MarkedText(tagged_text="...", sentence_count=3)),
            topic_extractor=StubTopicExtractor(["Technology>AI"]),
            range_assigner=FailingRangeAssigner(),
            parser=StubParser([]),
            gap_handler=StubGapHandler([]),
        )
        with pytest.raises(LLMError, match="Assignment failed"):
            pipeline.run("text")

    def test_json_assigner_with_json_parser(self) -> None:
        Pipeline(
            splitter=StubSplitter(_make_sentences(1)),
            marker=StubMarker(MarkedText(tagged_text=".", sentence_count=1)),
            topic_extractor=StubTopicExtractor(["Technology>AI"]),
            range_assigner=JsonStubRangeAssigner('{"topics":[]}'),
            parser=JsonStubParser(_make_groups()),
            gap_handler=StubGapHandler(_make_groups()),
        )

    def test_json_assigner_incompatible_with_text_parser(self) -> None:
        with pytest.raises(
            ValueError, match="Incompatible llm/parser response formats"
        ):
            Pipeline(
                splitter=StubSplitter(_make_sentences(1)),
                marker=StubMarker(MarkedText(tagged_text=".", sentence_count=1)),
                topic_extractor=StubTopicExtractor(["Technology>AI"]),
                range_assigner=JsonStubRangeAssigner('{"topics":[]}'),
                parser=StubParser(_make_groups()),
                gap_handler=StubGapHandler(_make_groups()),
            )

    def test_end_to_end_with_real_stages(self) -> None:
        from txt_splitt.gap_handlers import StrictGapHandler
        from txt_splitt.markers import BracketMarker
        from txt_splitt.parsers import TopicRangeParser
        from txt_splitt.splitters import SparseRegexSentenceSplitter

        text = "AI is growing fast. Climate change is real."

        pipeline = Pipeline(
            splitter=SparseRegexSentenceSplitter(),
            marker=BracketMarker(),
            topic_extractor=StubTopicExtractor(["Technology>AI", "Science>Climate"]),
            range_assigner=StubRangeAssigner("Technology>AI: 0\nScience>Climate: 1"),
            parser=TopicRangeParser(),
            gap_handler=StrictGapHandler(),
        )
        result = pipeline.run(text)

        assert len(result.sentences) == 2
        assert len(result.groups) == 2
        assert result.groups[0].label == ("Technology", "AI")
        assert result.groups[1].label == ("Science", "Climate")


class TestPipelineHtmlCleaning:
    def _make_mapping(self) -> OffsetMapping:
        return OffsetMapping(
            segments=(OffsetSegment(clean_offset=0, original_offset=0, length=10),),
            original_length=20,
            clean_length=10,
        )

    def test_validation_cleaner_without_restorer_raises(self) -> None:
        with pytest.raises(ValueError, match="both be provided"):
            Pipeline(
                splitter=StubSplitter(_make_sentences(1)),
                marker=StubMarker(MarkedText(tagged_text=".", sentence_count=1)),
                llm=StubLLM("x"),
                parser=StubParser([]),
                gap_handler=StubGapHandler([]),
                html_cleaner=StubHtmlCleaner("x", self._make_mapping()),
            )

    def test_validation_restorer_without_cleaner_raises(self) -> None:
        with pytest.raises(ValueError, match="both be provided"):
            Pipeline(
                splitter=StubSplitter(_make_sentences(1)),
                marker=StubMarker(MarkedText(tagged_text=".", sentence_count=1)),
                llm=StubLLM("x"),
                parser=StubParser([]),
                gap_handler=StubGapHandler([]),
                offset_restorer=StubOffsetRestorer(),
            )

    def test_validation_both_provided_ok(self) -> None:
        Pipeline(
            splitter=StubSplitter(_make_sentences(1)),
            marker=StubMarker(MarkedText(tagged_text=".", sentence_count=1)),
            llm=StubLLM("x"),
            parser=StubParser(_make_groups()),
            gap_handler=StubGapHandler(_make_groups()),
            html_cleaner=StubHtmlCleaner("x", self._make_mapping()),
            offset_restorer=StubOffsetRestorer(),
        )

    def test_splitter_receives_clean_text(self) -> None:
        clean_text = "Hello world"
        mapping = OffsetMapping(
            segments=(OffsetSegment(clean_offset=0, original_offset=0, length=11),),
            original_length=20,
            clean_length=11,
        )
        sentences = [Sentence(index=0, start=0, end=11, text=clean_text)]
        groups = _make_groups()
        splitter = RecordingSplitter(sentences)

        pipeline = Pipeline(
            splitter=splitter,
            marker=StubMarker(MarkedText(tagged_text=".", sentence_count=1)),
            llm=StubLLM("Technology>AI: 0"),
            parser=StubParser(groups),
            gap_handler=StubGapHandler(groups),
            html_cleaner=StubHtmlCleaner(clean_text, mapping),
            offset_restorer=StubOffsetRestorer(),
        )
        pipeline.run("Hello <b>world</b>!!")

        assert splitter.seen_text == clean_text

    def test_marker_receives_clean_text(self) -> None:
        clean_text = "Hello world"
        mapping = OffsetMapping(
            segments=(OffsetSegment(clean_offset=0, original_offset=0, length=11),),
            original_length=20,
            clean_length=11,
        )
        sentences = [Sentence(index=0, start=0, end=11, text=clean_text)]
        groups = _make_groups()
        marker = RecordingMarker(MarkedText(tagged_text=".", sentence_count=1))

        pipeline = Pipeline(
            splitter=StubSplitter(sentences),
            marker=marker,
            llm=StubLLM("Technology>AI: 0"),
            parser=StubParser(groups),
            gap_handler=StubGapHandler(groups),
            html_cleaner=StubHtmlCleaner(clean_text, mapping),
            offset_restorer=StubOffsetRestorer(),
        )
        pipeline.run("Hello <b>world</b>!!")

        assert marker.seen_text == clean_text

    def test_restorer_called_with_result_and_mapping(self) -> None:
        mapping = self._make_mapping()
        sentences = _make_sentences(1)
        groups = _make_groups()
        restorer = StubOffsetRestorer()

        pipeline = Pipeline(
            splitter=StubSplitter(sentences),
            marker=StubMarker(MarkedText(tagged_text=".", sentence_count=1)),
            llm=StubLLM("Technology>AI: 0"),
            parser=StubParser(groups),
            gap_handler=StubGapHandler(groups),
            html_cleaner=StubHtmlCleaner("clean", mapping),
            offset_restorer=restorer,
        )
        pipeline.run("original <b>text</b>")

        assert restorer.seen_result is not None
        assert restorer.seen_mapping is mapping

    def test_end_to_end_with_real_cleaner_and_restorer(self) -> None:
        from txt_splitt.gap_handlers import StrictGapHandler
        from txt_splitt.html_cleaners import TagStripCleaner
        from txt_splitt.markers import BracketMarker
        from txt_splitt.offset_restorers import MappingOffsetRestorer
        from txt_splitt.parsers import TopicRangeParser
        from txt_splitt.splitters import SparseRegexSentenceSplitter

        original = "<p>AI is growing fast.</p> <p>Climate change is real.</p>"

        llm_response = "Technology>AI: 0\nScience>Climate: 1"

        pipeline = Pipeline(
            splitter=SparseRegexSentenceSplitter(),
            marker=BracketMarker(),
            llm=StubLLM(llm_response),
            parser=TopicRangeParser(),
            gap_handler=StrictGapHandler(),
            html_cleaner=TagStripCleaner(),
            offset_restorer=MappingOffsetRestorer(),
        )
        result = pipeline.run(original)

        assert len(result.sentences) == 2
        assert result.sentences[0].text == "AI is growing fast."
        assert result.sentences[1].text == "Climate change is real."
        # Offsets should point into the original HTML text
        assert result.sentences[0].start < result.sentences[0].end
        assert result.sentences[1].start < result.sentences[1].end
        # The start of first sentence should skip the <p> tag
        assert result.sentences[0].start == 3  # after "<p>"
        assert result.groups[0].label == ("Technology", "AI")
        assert result.groups[1].label == ("Science", "Climate")

    def test_pipeline_without_html_cleaning_unchanged(self) -> None:
        sentences = _make_sentences(3)
        groups = _make_groups()

        pipeline = Pipeline(
            splitter=StubSplitter(sentences),
            marker=StubMarker(MarkedText(tagged_text="...", sentence_count=3)),
            llm=StubLLM("Technology>AI: 0-2"),
            parser=StubParser(groups),
            gap_handler=StubGapHandler(groups),
        )
        result = pipeline.run("Some text")

        assert len(result.sentences) == 3
        assert result.groups[0].label == ("Technology", "AI")
