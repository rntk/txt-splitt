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

    def test_splits_on_high_confidence_boundaries(self) -> None:
        text = "Lead: details here; then more. Finally done."
        result = self.splitter.split(text)
        assert len(result) == 2
        assert result[0] == Sentence(
            index=0, start=0, end=30, text="Lead: details here; then more."
        )
        assert result[1].text == "Finally done."

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
        assert len(result) == 2
        assert result[0].text == "one two three four five"
        assert (
            result[1].text == "six seven eight nine ten eleven twelve thirteen fourteen"
        )

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
        with pytest.raises(ValueError, match="min_sentence_words must be positive"):
            SparseRegexSentenceSplitter(min_sentence_words=0)


class TestSparseRegexSentenceSplitterHtmlAware:
    def test_matches_plain_mode_when_text_has_no_html(self) -> None:
        plain = SparseRegexSentenceSplitter(
            anchor_every_words=3, long_sentence_word_threshold=6
        )
        html_aware = SparseRegexSentenceSplitter(
            anchor_every_words=3, long_sentence_word_threshold=6, html_aware=True
        )
        text = "Alpha beta gamma delta epsilon zeta eta theta iota"

        plain_result = plain.split(text)
        html_result = html_aware.split(text)

        assert [sentence.text for sentence in plain_result] == [
            sentence.text for sentence in html_result
        ]

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
        assert len(result) == 2
        for s in result:
            assert text[s.start : s.end] == s.text

    def test_html_aware_anchor_spans_preserve_exact_source_slices(self) -> None:
        splitter = SparseRegexSentenceSplitter(
            anchor_every_words=2, long_sentence_word_threshold=3, html_aware=True
        )
        text = '<p>w1 w2 <span title="ignore."> w3 w4 w5</span> w6</p>'

        result = splitter.split(text)

        assert len(result) >= 2
        for sentence in result:
            assert text[sentence.start : sentence.end] == sentence.text


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

    def test_list_marker_does_not_back_attach_to_previous_content(self) -> None:
        text = "Alpha\n\n2.\n&nbsp;\nBeta"
        rows = [s.text.strip() for s in self._splitter().split(text)]
        assert rows == ["Alpha", "2.\n&nbsp;\nBeta"]

    def test_list_marker_with_bridge_does_not_back_attach(self) -> None:
        text = "Alpha\n\n2.\n·"
        rows = [s.text.strip() for s in self._splitter().split(text)]
        assert rows == ["Alpha", "2.\n·"]


class TestSoftBoundarySizing:
    def test_semicolon_and_colon_split_only_when_span_is_long(self) -> None:
        splitter = SparseRegexSentenceSplitter(
            anchor_every_words=6,
            long_sentence_word_threshold=10,
            min_sentence_words=3,
        )
        text = (
            "Overview: alpha beta gamma delta epsilon zeta; "
            "eta theta iota kappa lambda mu."
        )
        rows = [s.text for s in splitter.split(text)]
        assert rows == [
            "Overview: alpha beta gamma delta epsilon zeta;",
            "eta theta iota kappa lambda mu.",
        ]

    def test_single_newline_is_soft_boundary(self) -> None:
        splitter = SparseRegexSentenceSplitter(
            anchor_every_words=5,
            long_sentence_word_threshold=8,
            min_sentence_words=3,
        )
        text = "alpha beta gamma delta\nepsilon zeta eta theta iota kappa lambda"
        rows = [s.text for s in splitter.split(text)]
        assert rows == [
            "alpha beta gamma delta",
            "epsilon zeta eta theta iota kappa lambda",
        ]


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
        assert rows[1] == "Driving the news: None of those actions were enough."


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

    def test_multiple_closing_chars(self) -> None:
        splitter = SparseRegexSentenceSplitter(anchor_every_words=24)
        text = 'He said "done.") Next thing.'
        result = splitter.split(text)
        assert len(result) == 2


class TestAbbreviationHandling:
    def _splitter(self) -> SparseRegexSentenceSplitter:
        return SparseRegexSentenceSplitter(anchor_every_words=24)

    def test_dr_does_not_split(self) -> None:
        text = "Dr. Smith went home. He was tired."
        result = self._splitter().split(text)
        assert len(result) == 2
        assert result[0].text.startswith("Dr.")

    def test_mr_does_not_split(self) -> None:
        text = "Mr. Jones arrived. The meeting began."
        result = self._splitter().split(text)
        assert len(result) == 2
        assert result[0].text.startswith("Mr.")

    def test_etc_mid_sentence_no_split(self) -> None:
        # etc. inside a clause — the comma after makes it clearly mid-sentence
        text = "We accept cash, checks, etc. and other payment types."
        result = self._splitter().split(text)
        assert len(result) == 1

    def test_etc_sentence_final_splits(self) -> None:
        # etc. at the end of a clause — a genuine boundary must still be recognised
        text = "Items include hats, coats, etc. The list is long."
        result = self._splitter().split(text)
        assert len(result) == 2

    def test_real_period_still_splits(self) -> None:
        text = "The doctor arrived. She examined the patient."
        result = self._splitter().split(text)
        assert len(result) == 2



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


class TestEmojiAndUnicodeSentenceStart:
    def test_emoji_after_period_bird(self) -> None:
        splitter = SparseRegexSentenceSplitter(anchor_every_words=24)
        text = "The rain stopped at noon. \U0001f989 A new day began."
        result = splitter.split(text)
        assert len(result) == 2
        assert result[0].text == "The rain stopped at noon."
        assert result[1].text == "\U0001f989 A new day began."

    def test_emoji_after_period_wolf(self) -> None:
        splitter = SparseRegexSentenceSplitter(anchor_every_words=24)
        text = "She closed the book. \U0001f43a The night was quiet."
        result = splitter.split(text)
        assert len(result) == 2
        assert result[0].text == "She closed the book."
        assert result[1].text == "\U0001f43a The night was quiet."

    def test_emoji_starts_sentence_directly(self) -> None:
        splitter = SparseRegexSentenceSplitter(anchor_every_words=24)
        text = "The park was empty. \U0001f436Dogs love open spaces."
        result = splitter.split(text)
        assert len(result) == 2
        assert result[1].text == "\U0001f436Dogs love open spaces."

    def test_cjk_sentence_start(self) -> None:
        # CJK text has no spaces so counts as 1 word; use min_sentence_words=1
        # to prevent the short-span merge from re-joining the split.
        splitter = SparseRegexSentenceSplitter(
            anchor_every_words=24, min_sentence_words=1
        )
        text = "The meeting ended. \u4eca\u65e5\u306f\u6674\u308c\u3067\u3059\u3002"
        result = splitter.split(text)
        assert len(result) == 2
        assert result[1].text == "\u4eca\u65e5\u306f\u6674\u308c\u3067\u3059\u3002"

    def test_greek_uppercase_sentence_start(self) -> None:
        splitter = SparseRegexSentenceSplitter(anchor_every_words=24)
        # "Αυτό είναι." — Greek uppercase start
        text = "The session ended. \u0391\u03c5\u03c4\u03cc."
        result = splitter.split(text)
        assert len(result) == 2
        assert result[1].text.startswith("\u0391")

    def test_arabic_sentence_start(self) -> None:
        splitter = SparseRegexSentenceSplitter(anchor_every_words=24)
        # "مرحباً بالعالم." (Hello world.)
        text = "The session ended. \u0645\u0631\u062d\u0628\u0627\u064b."
        result = splitter.split(text)
        assert len(result) == 2
        assert result[1].text.startswith("\u0645")
