"""Tests for sparse sentence splitting."""

import pytest

from txt_splitt.splitters import SparseRegexSentenceSplitter
from txt_splitt.types import Sentence


class TestSparseRegexSentenceSplitter:
    def setup_method(self) -> None:
        self.splitter = SparseRegexSentenceSplitter(
            anchor_every_words=5, long_sentence_word_threshold=10
        )

    def test_empty_string(self) -> None:
        assert self.splitter.split("") == []

    def test_whitespace_only(self) -> None:
        assert self.splitter.split("   ") == []

    def test_splits_on_natural_punctuation_boundaries(self) -> None:
        text = "Lead: details here; then more. Finally done."
        result = self.splitter.split(text)
        assert len(result) == 4
        assert result[0] == Sentence(index=0, start=0, end=5, text="Lead:")
        assert result[1].text == "details here;"
        assert result[2].text == "then more."
        assert result[3].text == "Finally done."

    def test_does_not_anchor_short_sentences(self) -> None:
        text = "one two three four five six seven eight nine ten"
        result = self.splitter.split(text)
        assert len(result) == 1
        assert result[0].text == text

    def test_anchors_only_when_sentence_is_very_long(self) -> None:
        text = (
            "one two three four five six seven eight nine ten "
            "eleven twelve thirteen fourteen"
        )
        result = self.splitter.split(text)
        assert len(result) == 3
        assert result[0].text == "one two three four five"
        assert result[1].text == "six seven eight nine ten"
        assert result[2].text == "eleven twelve thirteen fourteen"

    def test_char_offsets_allow_slicing(self) -> None:
        text = "A: one two three four five six seven eight nine ten eleven twelve"
        result = self.splitter.split(text)
        for s in result:
            assert text[s.start : s.end] == s.text

    def test_invalid_params_rejected(self) -> None:
        with pytest.raises(ValueError, match="anchor_every_words must be positive"):
            SparseRegexSentenceSplitter(anchor_every_words=0)
        with pytest.raises(
            ValueError, match="long_sentence_word_threshold must be positive"
        ):
            SparseRegexSentenceSplitter(long_sentence_word_threshold=0)
        with pytest.raises(
            ValueError,
            match="long_sentence_word_threshold must be >= anchor_every_words",
        ):
            SparseRegexSentenceSplitter(
                anchor_every_words=10, long_sentence_word_threshold=5
            )


class TestSparseRegexSentenceSplitterHtmlAware:
    def test_avoids_splitting_on_punctuation_inside_html_tag(self) -> None:
        splitter = SparseRegexSentenceSplitter(
            anchor_every_words=5, long_sentence_word_threshold=10, html_aware=True
        )
        text = (
            '<div title="Lead: details; more.">'
            "alpha beta gamma delta epsilon zeta"
            "</div>"
        )
        result = splitter.split(text)
        assert len(result) == 1
        assert result[0].text == text

    def test_anchors_only_for_long_visible_text(self) -> None:
        splitter = SparseRegexSentenceSplitter(
            anchor_every_words=3, long_sentence_word_threshold=6, html_aware=True
        )
        text = 'w1 w2 w3 <span title="x: y; z."> w4 w5 w6 w7 w8'
        result = splitter.split(text)
        assert len(result) == 3
        for s in result:
            assert text[s.start : s.end] == s.text


class TestLowSignalCleanup:
    def _splitter(self) -> SparseRegexSentenceSplitter:
        return SparseRegexSentenceSplitter(anchor_every_words=24)

    def test_bridge_token_not_emitted_as_standalone_sentence(self) -> None:
        text = "By\nNathan Bomey\n·\nFeb 09, 2026"
        rows = [s.text.strip() for s in self._splitter().split(text)]
        assert "·" not in rows

    def test_ordinal_marker_merged_with_following_heading(self) -> None:
        text = "Intro\n\n2.\nStripe talking $140B valuation"
        rows = [s.text.strip() for s in self._splitter().split(text)]
        assert "2." not in rows
        assert any(
            row.startswith("2.") and "Stripe talking $140B valuation" in row
            for row in rows
        )

    def test_nbsp_entity_not_standalone_sentence(self) -> None:
        text = "Alpha\n&nbsp;\nBeta"
        rows = [s.text.strip() for s in self._splitter().split(text)]
        assert "&nbsp;" not in rows


class TestSparseEntitySemicolonBoundary:
    def test_html_entity_semicolon_does_not_split_sentence(self) -> None:
        splitter = SparseRegexSentenceSplitter(anchor_every_words=24)
        text = (
            "mass-marketed by companies, including Hims &amp; Hers. "
            "Driving the news: None of those actions were enough."
        )
        result = splitter.split(text)
        rows = [s.text for s in result]
        assert "&amp;" not in [row.strip() for row in rows]
        assert rows[0] == "mass-marketed by companies, including Hims &amp; Hers."
        assert rows[1] == "Driving the news:"


class TestClosingQuoteBoundary:
    def test_curly_closing_quote_sparse_splitter(self) -> None:
        splitter = SparseRegexSentenceSplitter(anchor_every_words=24)
        text = "Lambo demand is \u201cclose to zero.\u201d Amazon opened a campus."
        result = splitter.split(text)
        assert len(result) == 2
        assert result[0].text == "Lambo demand is \u201cclose to zero.\u201d"
        assert result[1].text == "Amazon opened a campus."

    def test_straight_closing_quote(self) -> None:
        splitter = SparseRegexSentenceSplitter(anchor_every_words=24)
        text = 'He said "done." Next step follows.'
        result = splitter.split(text)
        assert len(result) == 2
        assert result[1].text == "Next step follows."

    def test_closing_paren(self) -> None:
        splitter = SparseRegexSentenceSplitter(anchor_every_words=24)
        text = "The deal closed (finally.) Next steps were taken."
        result = splitter.split(text)
        assert len(result) == 2
        assert result[1].text == "Next steps were taken."

    def test_accented_uppercase_lookahead(self) -> None:
        splitter = SparseRegexSentenceSplitter(anchor_every_words=24)
        text = "C\u2019est fini. \u00c9videmment il a raison."
        result = splitter.split(text)
        assert len(result) == 2
        assert result[1].text == "\u00c9videmment il a raison."

    def test_ellipsis_boundary(self) -> None:
        splitter = SparseRegexSentenceSplitter(anchor_every_words=24)
        text = "He trailed off\u2026 Then she spoke."
        result = splitter.split(text)
        assert len(result) == 2
        assert result[0].text == "He trailed off\u2026"
        assert result[1].text == "Then she spoke."


class TestEmDashNotSplitting:
    def test_em_dash_parenthetical_not_split(self) -> None:
        splitter = SparseRegexSentenceSplitter(anchor_every_words=24)
        text = "Amazon opened a campus \u2014 its biggest \u2014 for employees."
        result = splitter.split(text)
        assert len(result) == 1
        assert result[0].text == text

    def test_en_dash_parenthetical_not_split(self) -> None:
        splitter = SparseRegexSentenceSplitter(anchor_every_words=24)
        text = "The campus \u2013 its second-largest \u2013 opened last year."
        result = splitter.split(text)
        assert len(result) == 1
        assert result[0].text == text

    def test_lamborghini_amazon_regression(self) -> None:
        splitter = SparseRegexSentenceSplitter(anchor_every_words=5)
        text = (
            "Lambo demand is \u201cclose to zero.\u201d "
            "Amazon opened a 1.1M\u2011square\u2011foot campus in north Bengaluru "
            "\u2014 its second-largest office in Asia \u2014 "
            "designed to house more than 7K employees."
        )
        result = splitter.split(text)
        assert len(result) >= 2
        assert result[0].text == "Lambo demand is \u201cclose to zero.\u201d"
        amazon_text = " ".join(s.text for s in result[1:])
        assert "\u2014" in amazon_text
