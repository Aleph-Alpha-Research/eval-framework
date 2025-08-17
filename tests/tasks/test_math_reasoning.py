import pytest

from eval_framework.tasks.benchmarks.math_reasoning import MATH
from tests.utils import DatasetPatcher


@pytest.fixture()
def math_reasoning() -> MATH:
    with DatasetPatcher(MATH) as patched_task:
        return patched_task


@pytest.mark.parametrize(
    "string, start_index, expected",
    [
        # Basic cases
        ("{abc}", 0, 4),
        ("{hello world}", 0, 12),
        ("{x+y}", 0, 4),
        ("{a{b}c}", 0, 6),  # Nested once
        ("{a{b{c}d}e}", 0, 10),  # Nested twice
        ("{{{}}}", 0, 5),  # Deeply nested
        ("{a}{b}", 0, 2),
        ("{x+y} {a*b}", 0, 4),
        ("{outer {inner} content}", 0, 22),
        ("{}", 0, 1),  # Only one pair
        ("{abc", 0, -1),  # No closing bracket
        ("xyz {abc} def", 4, 8),
        ("text {inside} more text", 5, 12),
        ("{a{b}c}", 2, 4),  # Finds closing for second "{"
        ("{a{b{c}d}e}", 2, 8),  # Finds closing for second "{"
        pytest.param("{" + "a" * 10000 + "}", 0, 10001, id="large_content_inside_brackets-0-10001"),
    ],
)
def test_find_closing_bracket(math_reasoning: MATH, string: str, start_index: int, expected: int) -> None:
    assert math_reasoning._find_closing_bracket(string, start_index) == expected


def test_find_closing_bracket_exceptions(math_reasoning: MATH) -> None:
    """Test cases that should raise ValueError."""
    with pytest.raises(ValueError):
        math_reasoning._find_closing_bracket("{}", -1)  # Negative index

    with pytest.raises(ValueError):
        math_reasoning._find_closing_bracket("{}", 10)  # Out-of-bounds index

    with pytest.raises(ValueError):
        math_reasoning._find_closing_bracket("abc", 0)  # No '{' at index 0

    with pytest.raises(ValueError):
        math_reasoning._find_closing_bracket("", 0)  # Empty string with non-existent index


@pytest.mark.parametrize(
    "string, expected",
    [
        (r"Some \text{example} text", ("Some ", "example", " text")),
        (r"\text{hello}", ("", "hello", "")),
        (r"\text{first} \text{second}", ("", "first", r" \text{second}")),
        (r"Start \text{one} middle \text{two} end", ("Start ", "one", r" middle \text{two} end")),
        (r"Text \text{nested {inside} here} end", ("Text ", "nested {inside} here", " end")),
        (r"\text{a{b}c}", ("", "a{b}c", "")),
        ("No text command here", ("No text command here", "", "")),  # No `\text{}` present
        (r"\text{}", ("", "", "")),  # Empty content inside `\text{}`
        (r"Before \text{missing end", ("Before ", "missing end", "")),  # Missing closing `}`
        (r"\text{unclosed", ("", "unclosed", "")),  # Missing closing `}`
        (r"Invalid \text content", (r"Invalid \text content", "", "")),
        (r"Incorrect usage: \text} here", (r"Incorrect usage: \text} here", "", "")),
        (r"Escaped \text{\textbf{bold} word}", ("Escaped ", r"\textbf{bold} word", "")),
        pytest.param(r"\text{" + "a" * 10000 + "}", ("", "a" * 10000, ""), id="large_content_inside_text_command"),
        ("", ("", "", "")),
    ],
)
def test_split_text_command(math_reasoning: MATH, string: str, expected: tuple[str, str, str]) -> None:
    assert math_reasoning._split_text_command(string) == expected


@pytest.mark.parametrize(
    "string, search, expected",
    [
        (r"Some \text{example} text", r"\text{", ("Some ", "example", " text")),
        (r"Start \emph{italic} end", r"\emph{", ("Start ", "italic", " end")),
        (r"\emph{only}", r"\emph{", ("", "only", "")),
        (r"Multiple \emph{first} \emph{second}", r"\emph{", ("Multiple ", "first", r" \emph{second}")),
        (r"Start \emph{italic} end", r"\text{", (r"Start \emph{italic} end", "", "")),  # Different search term
        ("normal text", "", ("normal text", "", "")),  # No LaTeX command found
    ],
)
def test_split_text_command_with_search(
    math_reasoning: MATH, string: str, search: str, expected: tuple[str, str, str]
) -> None:
    assert math_reasoning._split_text_command(string, search) == expected
