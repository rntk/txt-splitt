from txt_splitt.sentences.llm import _extract_lines_by_range
from txt_splitt.sentences.types import SentenceRange


def test_extract_lines_with_multi_line_marker():
    tagged = "{0} Line 1\nMore of line 1\n{1} Line 2"
    ranges = [SentenceRange(start=0, end=0)]
    result = _extract_lines_by_range(tagged, ranges)
    # Current behavior: returns only "{0} Line 1"
    # Expected behavior: returns "{0} Line 1\nMore of line 1"
    print(f"Result: {result!r}")
    assert result == "{0} Line 1\nMore of line 1"


if __name__ == "__main__":
    try:
        test_extract_lines_with_multi_line_marker()
        print("Test PASSED")
    except AssertionError:
        print("Test FAILED")
    except Exception as e:
        print(f"An error occurred: {e}")
