"""Tests for sentence splitting."""

import pytest

from txt_splitt.splitters import (
    DenseRegexSentenceSplitter,
    HtmlAwareSentenceSplitter,
    RegexSentenceSplitter,
    SparseRegexSentenceSplitter,
)
from txt_splitt.types import Sentence


class TestRegexSentenceSplitter:
    def setup_method(self) -> None:
        self.splitter = RegexSentenceSplitter()

    def test_empty_string(self) -> None:
        assert self.splitter.split("") == []

    def test_whitespace_only(self) -> None:
        assert self.splitter.split("   ") == []

    def test_single_sentence(self) -> None:
        result = self.splitter.split("Hello world.")
        assert len(result) == 1
        assert result[0] == Sentence(index=0, start=0, end=12, text="Hello world.")

    def test_multiple_sentences(self) -> None:
        text = "Hello world. This is a test. Another one."
        result = self.splitter.split(text)
        assert len(result) == 3
        assert result[0].index == 0
        assert result[0].text == "Hello world."
        assert result[1].index == 1
        assert result[1].text == "This is a test."
        assert result[2].index == 2
        assert result[2].text == "Another one."

    def test_zero_based_indices(self) -> None:
        text = "First. Second."
        result = self.splitter.split(text)
        assert result[0].index == 0
        assert result[1].index == 1

    def test_newline_splitting(self) -> None:
        text = "First paragraph.\nSecond paragraph."
        result = self.splitter.split(text)
        assert len(result) == 2
        assert result[0].text == "First paragraph."
        assert result[1].text == "Second paragraph."

    def test_char_offsets_allow_slicing(self) -> None:
        text = "Hello world. This is a test."
        result = self.splitter.split(text)
        for s in result:
            assert text[s.start : s.end] == s.text

    def test_exclamation_mark_boundary(self) -> None:
        text = "Wow! That is great."
        result = self.splitter.split(text)
        assert len(result) == 2
        assert result[0].text == "Wow!"
        assert result[1].text == "That is great."

    def test_question_mark_boundary(self) -> None:
        text = "Is this working? Yes it is."
        result = self.splitter.split(text)
        assert len(result) == 2
        assert result[0].text == "Is this working?"
        assert result[1].text == "Yes it is."

    def test_no_split_on_lowercase_after_period(self) -> None:
        text = "Version 3.14 is great."
        result = self.splitter.split(text)
        assert len(result) == 1

    def test_cyrillic_uppercase_boundary(self) -> None:
        text = "Hello world. Привет мир."
        result = self.splitter.split(text)
        assert len(result) == 2


class TestDenseRegexSentenceSplitter:
    def setup_method(self) -> None:
        self.splitter = DenseRegexSentenceSplitter(anchor_every_words=5)

    def test_empty_string(self) -> None:
        assert self.splitter.split("") == []

    def test_separator_boundaries_split(self) -> None:
        text = "First block | Second block · Third block"
        result = self.splitter.split(text)
        assert len(result) == 3
        assert result[0].text == "First block"
        assert result[1].text == "Second block"
        assert result[2].text == "Third block"

    def test_anchor_splitting_adds_density(self) -> None:
        text = "one two three four five six seven eight nine ten eleven"
        result = self.splitter.split(text)
        assert len(result) == 3
        assert result[0].text == "one two three four five"
        assert result[1].text == "six seven eight nine ten"
        assert result[2].text == "eleven"

    def test_char_offsets_allow_slicing(self) -> None:
        text = "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
        result = self.splitter.split(text)
        for s in result:
            assert text[s.start : s.end] == s.text

    def test_non_positive_anchor_words_rejected(self) -> None:
        with pytest.raises(ValueError, match="anchor_every_words must be positive"):
            DenseRegexSentenceSplitter(anchor_every_words=0)


class TestDenseRegexSentenceSplitterHtmlAware:
    def setup_method(self) -> None:
        self.splitter = DenseRegexSentenceSplitter(
            anchor_every_words=5, html_aware=True
        )

    # -- Baseline parity (no HTML) --

    def test_empty_string(self) -> None:
        assert self.splitter.split("") == []

    def test_whitespace_only(self) -> None:
        assert self.splitter.split("   ") == []

    def test_separator_boundaries_split(self) -> None:
        text = "First block | Second block · Third block"
        result = self.splitter.split(text)
        assert len(result) == 3
        assert result[0].text == "First block"
        assert result[1].text == "Second block"
        assert result[2].text == "Third block"

    def test_anchor_splitting_adds_density(self) -> None:
        text = "one two three four five six seven eight nine ten eleven"
        result = self.splitter.split(text)
        assert len(result) == 3
        assert result[0].text == "one two three four five"
        assert result[1].text == "six seven eight nine ten"
        assert result[2].text == "eleven"

    def test_char_offsets_allow_slicing(self) -> None:
        text = "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
        result = self.splitter.split(text)
        for s in result:
            assert text[s.start : s.end] == s.text

    def test_non_positive_anchor_words_rejected(self) -> None:
        with pytest.raises(ValueError, match="anchor_every_words must be positive"):
            DenseRegexSentenceSplitter(anchor_every_words=0, html_aware=True)

    def test_no_html_matches_base_splitter(self) -> None:
        text = "First sentence. Second sentence here today! Third sentence."
        base = DenseRegexSentenceSplitter(anchor_every_words=5)
        html_aware = DenseRegexSentenceSplitter(anchor_every_words=5, html_aware=True)
        assert base.split(text) == html_aware.split(text)

    # -- HTML tag preservation --

    def test_tag_with_complex_attributes(self) -> None:
        tag = '<div class="example" onclick="something()" data-attr=\'{"bla": "bla"}\'>'
        text = f"Before text. {tag} After text here."
        result = self.splitter.split(text)
        for s in result:
            assert text[s.start : s.end] == s.text
            if tag[:10] in s.text:
                assert tag in s.text

    def test_tag_with_newline_in_attributes(self) -> None:
        text = '<div\nclass="test"> content here </div>'
        result = self.splitter.split(text)
        found_partial = any(
            s.text.strip() == "<div" or s.text.strip().startswith("class=")
            for s in result
        )
        assert not found_partial

    def test_self_closing_tags(self) -> None:
        text = 'Hello <br/> world <img src="pic.jpg" alt="A B C"/> end'
        result = self.splitter.split(text)
        for s in result:
            assert text[s.start : s.end] == s.text

    def test_sentence_boundary_inside_attribute_value(self) -> None:
        text = '<div title="Hello. World"> content </div>'
        result = self.splitter.split(text)
        assert any('<div title="Hello. World">' in s.text for s in result)

    # -- Word anchor + HTML --

    def test_anchor_split_avoids_tag_interior(self) -> None:
        words_before = "one two three"
        tag = '<div class="big red" data-x="hello world">'
        words_after = "four five six seven eight nine ten"
        text = f"{words_before} {tag} {words_after}"
        result = self.splitter.split(text)
        tag_start = text.index(tag)
        tag_end = tag_start + len(tag)
        for s in result:
            assert text[s.start : s.end] == s.text
            assert not (tag_start < s.start < tag_end), (
                f"Sentence starts inside tag: {s}"
            )
            assert not (tag_start < s.end < tag_end), f"Sentence ends inside tag: {s}"

    def test_word_counting_skips_tag_tokens(self) -> None:
        splitter = DenseRegexSentenceSplitter(anchor_every_words=3, html_aware=True)
        text = 'w1 w2 w3 <span class="a b c"> w4 w5 w6'
        result = splitter.split(text)
        assert len(result) == 2
        assert result[0].text.startswith("w1")
        for s in result:
            assert text[s.start : s.end] == s.text

    # -- Edge cases --

    def test_text_entirely_a_tag(self) -> None:
        text = '<div class="example">'
        result = self.splitter.split(text)
        assert len(result) == 1
        assert result[0].text == text

    def test_adjacent_tags(self) -> None:
        text = '<span class="a b"></span><div class="c d"> text </div>'
        result = self.splitter.split(text)
        for s in result:
            assert text[s.start : s.end] == s.text


class TestSparseRegexSentenceSplitter:
    def setup_method(self) -> None:
        self.splitter = SparseRegexSentenceSplitter(
            anchor_every_words=5, long_sentence_word_threshold=10
        )

    def test_empty_string(self) -> None:
        assert self.splitter.split("") == []

    def test_splits_on_natural_punctuation_boundaries(self) -> None:
        text = "Lead: details here; then more. Finally done."
        result = self.splitter.split(text)
        assert len(result) == 4
        assert result[0].text == "Lead:"
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


class TestHtmlAwareSentenceSplitter:
    def setup_method(self) -> None:
        self.splitter = HtmlAwareSentenceSplitter(anchor_every_words=5)

    # -- Baseline parity (no HTML) --

    def test_empty_string(self) -> None:
        assert self.splitter.split("") == []

    def test_whitespace_only(self) -> None:
        assert self.splitter.split("   ") == []

    def test_no_html_plain_text(self) -> None:
        """Plain text without HTML should produce same results as dense splitter."""
        text = "one two three four five six seven eight nine ten eleven"
        base = DenseRegexSentenceSplitter(anchor_every_words=5, html_aware=True)
        result = self.splitter.split(text)
        base_result = base.split(text)
        assert [s.text for s in result] == [s.text for s in base_result]

    def test_char_offsets_allow_slicing(self) -> None:
        text = "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
        result = self.splitter.split(text)
        for s in result:
            assert text[s.start : s.end] == s.text

    def test_non_positive_anchor_words_rejected(self) -> None:
        with pytest.raises(ValueError, match="anchor_every_words must be positive"):
            HtmlAwareSentenceSplitter(anchor_every_words=0)

    def test_separator_boundaries_split(self) -> None:
        text = "First block | Second block · Third block"
        splitter = HtmlAwareSentenceSplitter(
            anchor_every_words=24, block_tags_as_boundaries=False
        )
        result = splitter.split(text)
        assert len(result) == 3
        assert result[0].text == "First block"
        assert result[1].text == "Second block"
        assert result[2].text == "Third block"

    # -- HTML comment protection --

    def test_comment_with_sentence_boundary(self) -> None:
        text = "Before. <!-- This is. A comment. --> After here now."
        splitter = HtmlAwareSentenceSplitter(
            anchor_every_words=24, block_tags_as_boundaries=False
        )
        result = splitter.split(text)
        # The period inside the comment should NOT cause a split
        found_comment_fragment = any(
            s.text.strip() == "A comment." or s.text.strip() == "This is."
            for s in result
        )
        assert not found_comment_fragment
        for s in result:
            assert text[s.start : s.end] == s.text

    def test_multiline_comment(self) -> None:
        text = "Start.\n<!-- multi\nline\ncomment -->\nEnd."
        result = self.splitter.split(text)
        # Comment should not be split by its internal newlines
        found_partial = any(s.text.strip() == "line" for s in result)
        assert not found_partial
        for s in result:
            assert text[s.start : s.end] == s.text

    # -- Script/style content protection --

    def test_script_content_protected(self) -> None:
        text = '<script>if (x < 2) { alert("Hello. World!"); }</script> After text.'
        splitter = HtmlAwareSentenceSplitter(
            anchor_every_words=24, block_tags_as_boundaries=False
        )
        result = splitter.split(text)
        # "Hello. World!" inside script should not cause a split
        found_hello = any(s.text.strip() == "Hello." for s in result)
        assert not found_hello
        for s in result:
            assert text[s.start : s.end] == s.text

    def test_style_content_protected(self) -> None:
        text = '<style>.cls { content: "Hello. World"; }</style> After text.'
        splitter = HtmlAwareSentenceSplitter(
            anchor_every_words=24, block_tags_as_boundaries=False
        )
        result = splitter.split(text)
        found_hello = any(s.text.strip() == "Hello." for s in result)
        assert not found_hello
        for s in result:
            assert text[s.start : s.end] == s.text

    # -- Block-level boundaries --

    def test_block_tag_creates_boundary(self) -> None:
        text = "First sentence.<div>Second sentence.</div>Third sentence."
        result = self.splitter.split(text)
        texts = [s.text for s in result]
        assert len(texts) >= 2
        # Should split around the <div> boundaries
        assert any("First" in t for t in texts)
        assert any("Third" in t for t in texts)
        for s in result:
            assert text[s.start : s.end] == s.text

    def test_block_boundaries_disabled(self) -> None:
        splitter = HtmlAwareSentenceSplitter(
            anchor_every_words=24, block_tags_as_boundaries=False
        )
        text = "First sentence.<div>Second sentence.</div>Third sentence."
        result_no_block = splitter.split(text)
        result_with_block = self.splitter.split(text)
        # With block boundaries disabled, fewer splits are expected
        assert len(result_no_block) <= len(result_with_block)
        for s in result_no_block:
            assert text[s.start : s.end] == s.text

    def test_paragraph_tags_as_boundaries(self) -> None:
        text = "<p>First paragraph.</p><p>Second paragraph.</p>"
        result = self.splitter.split(text)
        texts = [s.text for s in result]
        assert len(texts) >= 2
        assert any("First" in t for t in texts)
        assert any("Second" in t for t in texts)
        for s in result:
            assert text[s.start : s.end] == s.text

    def test_heading_tag_as_boundary(self) -> None:
        text = "<h1>Title</h1>Body text here."
        result = self.splitter.split(text)
        texts = [s.text for s in result]
        assert len(texts) >= 2
        assert any("Title" in t for t in texts)
        assert any("Body" in t for t in texts)
        for s in result:
            assert text[s.start : s.end] == s.text

    def test_br_creates_boundary(self) -> None:
        text = "First line.<br>Second line."
        result = self.splitter.split(text)
        texts = [s.text for s in result]
        assert any("First" in t for t in texts)
        assert any("Second" in t for t in texts)
        for s in result:
            assert text[s.start : s.end] == s.text

    # -- Tag protection (parity with regex HTML-aware tests) --

    def test_tag_with_complex_attributes(self) -> None:
        tag = '<div class="example" onclick="something()" data-attr=\'{"bla": "bla"}\'>'
        text = f"Before text. {tag} After text here."
        result = self.splitter.split(text)
        for s in result:
            assert text[s.start : s.end] == s.text
            if tag[:10] in s.text:
                assert tag in s.text

    def test_self_closing_tags(self) -> None:
        text = 'Hello <br/> world <img src="pic.jpg" alt="A B C"/> end'
        result = self.splitter.split(text)
        for s in result:
            assert text[s.start : s.end] == s.text

    def test_sentence_boundary_inside_attribute(self) -> None:
        splitter = HtmlAwareSentenceSplitter(
            anchor_every_words=24, block_tags_as_boundaries=False
        )
        text = '<span title="Hello. World"> content </span>'
        result = splitter.split(text)
        assert any('<span title="Hello. World">' in s.text for s in result)

    def test_anchor_split_avoids_tag_interior(self) -> None:
        words_before = "one two three"
        tag = '<span class="big red" data-x="hello world">'
        words_after = "four five six seven eight nine ten"
        text = f"{words_before} {tag} {words_after}"
        splitter = HtmlAwareSentenceSplitter(
            anchor_every_words=5, block_tags_as_boundaries=False
        )
        result = splitter.split(text)
        tag_start = text.index(tag)
        tag_end = tag_start + len(tag)
        for s in result:
            assert text[s.start : s.end] == s.text
            assert not (tag_start < s.start < tag_end), (
                f"Sentence starts inside tag: {s}"
            )
            assert not (tag_start < s.end < tag_end), f"Sentence ends inside tag: {s}"

    def test_word_counting_skips_tag_tokens(self) -> None:
        splitter = HtmlAwareSentenceSplitter(
            anchor_every_words=3, block_tags_as_boundaries=False
        )
        text = 'w1 w2 w3 <span class="a b c"> w4 w5 w6'
        result = splitter.split(text)
        assert len(result) == 2
        assert result[0].text.startswith("w1")
        for s in result:
            assert text[s.start : s.end] == s.text

    # -- Declaration / PI --

    def test_doctype_protected(self) -> None:
        text = "<!DOCTYPE html><html><body>Hello.</body></html>"
        result = self.splitter.split(text)
        # Should not crash; doctype is protected
        for s in result:
            assert text[s.start : s.end] == s.text

    # -- Malformed HTML resilience --

    def test_stray_angle_bracket(self) -> None:
        text = "3 < 5 is true. Hello there."
        result = self.splitter.split(text)
        # Should not crash
        assert len(result) >= 1
        for s in result:
            assert text[s.start : s.end] == s.text

    def test_template_syntax(self) -> None:
        text = "Hello {{ name }}. World here."
        result = self.splitter.split(text)
        assert len(result) >= 1
        for s in result:
            assert text[s.start : s.end] == s.text

    def test_unclosed_tag(self) -> None:
        text = "Hello <div world"
        result = self.splitter.split(text)
        assert len(result) >= 1
        for s in result:
            assert text[s.start : s.end] == s.text

    # -- Edge cases --

    def test_text_entirely_a_tag(self) -> None:
        text = '<div class="example">'
        result = self.splitter.split(text)
        assert len(result) == 1
        assert result[0].text == text

    def test_adjacent_block_tags(self) -> None:
        text = "<div></div><p></p>"
        result = self.splitter.split(text)
        for s in result:
            assert text[s.start : s.end] == s.text

    def test_mixed_block_and_inline(self) -> None:
        text = "Text <em>emphasis</em> more. <div>Block content.</div>"
        result = self.splitter.split(text)
        texts = [s.text for s in result]
        # The <div> should create a boundary
        assert any("Block" in t for t in texts)
        for s in result:
            assert text[s.start : s.end] == s.text


class TestClosingQuoteBoundary:
    """Closing quotes after sentence-ending punctuation must not block splits."""

    def test_curly_closing_quote_regex_splitter(self) -> None:
        splitter = RegexSentenceSplitter()
        text = "He said \u201chello.\u201d She replied."
        result = splitter.split(text)
        assert len(result) == 2
        assert result[0].text == "He said \u201chello.\u201d"
        assert result[1].text == "She replied."

    def test_curly_closing_quote_sparse_splitter(self) -> None:
        splitter = SparseRegexSentenceSplitter(anchor_every_words=24)
        text = 'Lambo demand is \u201cclose to zero.\u201d Amazon opened a campus.'
        result = splitter.split(text)
        assert len(result) == 2
        assert result[0].text == 'Lambo demand is \u201cclose to zero.\u201d'
        assert result[1].text == "Amazon opened a campus."

    def test_curly_closing_quote_dense_splitter(self) -> None:
        splitter = DenseRegexSentenceSplitter(anchor_every_words=24)
        text = "First sentence.\u201d Second sentence."
        result = splitter.split(text)
        assert len(result) == 2

    def test_straight_closing_quote(self) -> None:
        splitter = RegexSentenceSplitter()
        text = 'He said "done." Next step follows.'
        result = splitter.split(text)
        assert len(result) == 2
        assert result[1].text == "Next step follows."

    def test_closing_paren(self) -> None:
        splitter = RegexSentenceSplitter()
        text = "The deal closed (finally.) Next steps were taken."
        result = splitter.split(text)
        assert len(result) == 2
        assert result[1].text == "Next steps were taken."

    def test_accented_uppercase_lookahead(self) -> None:
        splitter = RegexSentenceSplitter()
        # É is Latin Extended capital — should trigger sentence boundary
        text = "C\u2019est fini. \u00c9videmment il a raison."
        result = splitter.split(text)
        assert len(result) == 2
        assert result[1].text == "\u00c9videmment il a raison."

    def test_ellipsis_boundary(self) -> None:
        splitter = RegexSentenceSplitter()
        text = "He trailed off\u2026 Then she spoke."
        result = splitter.split(text)
        assert len(result) == 2
        assert result[0].text == "He trailed off\u2026"
        assert result[1].text == "Then she spoke."


class TestEmDashNotSplitting:
    """Em-dash should NOT create sentence boundaries in SparseRegexSentenceSplitter."""

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
        """Regression: closing curly quote after period must split sentences."""
        splitter = SparseRegexSentenceSplitter(anchor_every_words=5)
        text = (
            "Lambo demand is \u201cclose to zero.\u201d "
            "Amazon opened a 1.1M\u2011square\u2011foot campus in north Bengaluru "
            "\u2014 its second-largest office in Asia \u2014 "
            "designed to house more than 7K employees."
        )
        result = splitter.split(text)
        # Must split at the sentence boundary after the closing curly quote
        assert len(result) >= 2
        assert result[0].text == 'Lambo demand is \u201cclose to zero.\u201d'
        # The Amazon sentence must NOT be split at the em-dashes
        amazon_text = " ".join(s.text for s in result[1:])
        assert "\u2014" in amazon_text
