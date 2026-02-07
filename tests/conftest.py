"""Shared fixtures for text splitter tests."""

import pytest

from txt_splitt.types import Sentence, SentenceGroup, SentenceRange


@pytest.fixture
def simple_text() -> str:
    return "Hello world. This is a test. Another sentence here."


@pytest.fixture
def multiline_text() -> str:
    return "First paragraph.\nSecond paragraph.\nThird paragraph."


@pytest.fixture
def simple_sentences() -> list[Sentence]:
    return [
        Sentence(index=0, start=0, end=12, text="Hello world."),
        Sentence(index=1, start=13, end=27, text="This is a test."),
        Sentence(index=2, start=28, end=51, text="Another sentence here."),
    ]


@pytest.fixture
def full_coverage_groups() -> list[SentenceGroup]:
    """Groups that cover sentences 0-4 completely."""
    return [
        SentenceGroup(
            label=("Technology", "AI"),
            ranges=(SentenceRange(start=0, end=2),),
        ),
        SentenceGroup(
            label=("Science", "Climate"),
            ranges=(SentenceRange(start=3, end=4),),
        ),
    ]
