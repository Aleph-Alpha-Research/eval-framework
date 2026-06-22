from unittest.mock import patch

import pytest

from eval_framework.tasks.benchmarks.math_reasoning import MATH, MATHMinervaBPB
from template_formatting.formatter import BaseFormatter, ConcatFormatter, Llama3Formatter, Message, Role
from tests.tests_eval_framework.tasks.benchmarks.utils import (
    ExpectedPrompt,
    assert_offline_oneshot_prompt,
    assert_offline_zeroshot_prompt,
    get_task_names_for_module,
    run_formatter_hash_test,
)
from tests.tests_eval_framework.utils import DatasetPatcher

_NUM_FEWSHOT = {
    "GSM8KReasoning": 0,
    "MATHMinervaBPB": 0,
    "MATHMinervaEvalHarness": 0,
    "MATH500Minerva": 0,
    "MATHMinerva_OLMES": 4,
}


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


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter])
@pytest.mark.parametrize("task_name", get_task_names_for_module("math_reasoning"))
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls, num_fewshot=_NUM_FEWSHOT.get(task_name, 1))


# ---------------------------------------------------------------------------
# Offline prompt tests for MATHMinervaBPB (patched HF load; production task API otherwise)
# ---------------------------------------------------------------------------

_SUBJECT = "algebra"

_CUE = "Solution:"
_EVAL_ROW: dict[str, str] = {
    "problem": "What is 2 + 3?",
    "solution": "2 + 3 = 5. So the answer is $\\boxed{5}$.",
}

_FEWSHOT_ROW: dict[str, str] = {
    "problem": "What is 1 + 1?",
    "solution": "1 + 1 = 2. Final Answer: The final answer is $2$. I hope it is correct.",
}

# _get_instruction_text no longer embeds the cue; it becomes its own ASSISTANT message.
_EVAL_PROBLEM = "Problem:\nWhat is 2 + 3?\n\n"
_FEWSHOT_PROBLEM = f"Problem:\n{_FEWSHOT_ROW['problem']}\n\n"
_FEWSHOT_ANSWER = f"{_CUE} {_FEWSHOT_ROW['solution']}"

# _get_ground_truth returns _get_choices()[0] — no leading space
_GROUND_TRUTH = "2 + 3 = 5. So the answer is $\\boxed{5}$.\nFinal Answer: The final answer is 5. I hope it is correct."
# BPBStyle.get_possible_completions adds a leading space
_EVAL_COMPLETION = " " + _GROUND_TRUTH
_COMPLETIONS = [_EVAL_COMPLETION]

# Concat strings are identical to before; structure just moves the cue to an ASSISTANT turn.
_EXPECTED_CONCAT_0SHOT = f"{_EVAL_PROBLEM}{_CUE}"
_EXPECTED_CONCAT_1SHOT = f"{_FEWSHOT_PROBLEM}{_FEWSHOT_ANSWER}\n\n{_EVAL_PROBLEM}{_CUE}"


_ZEROSHOT = ExpectedPrompt(
    messages=[
        Message(role=Role.USER, content=_EVAL_PROBLEM),
        Message(role=Role.ASSISTANT, content=_CUE),
    ],
    concat=_EXPECTED_CONCAT_0SHOT,
    ground_truth=_GROUND_TRUTH,
    completions=_COMPLETIONS,
)

_ONESHOT = ExpectedPrompt(
    messages=[
        Message(role=Role.USER, content=_FEWSHOT_PROBLEM),
        Message(role=Role.ASSISTANT, content=_FEWSHOT_ANSWER),
        Message(role=Role.USER, content=_EVAL_PROBLEM),
        Message(role=Role.ASSISTANT, content=_CUE),
    ],
    concat=_EXPECTED_CONCAT_1SHOT,
    ground_truth=_GROUND_TRUTH,
    completions=_COMPLETIONS,
)


def test_mathminervabpb_offline_prompt_formatting_zeroshot() -> None:

    # Patch _sample_fewshot_examples to return empty list for zeroshot
    def mock_zero_fewshot_examples(self, item):
        return []

    with patch.object(MATHMinervaBPB, "_sample_fewshot_examples", mock_zero_fewshot_examples):
        assert_offline_zeroshot_prompt(
            MATHMinervaBPB,
            eval_row=_EVAL_ROW,
            subjects=[_SUBJECT],
            expected=_ZEROSHOT,
        )


def test_mathminervabpb_offline_prompt_formatting_oneshot() -> None:

    def mock_one_fewshot_example(self, item):
        return [_FEWSHOT_ROW]

    with patch.object(MATHMinervaBPB, "_sample_fewshot_examples", mock_one_fewshot_example):
        assert_offline_oneshot_prompt(
            MATHMinervaBPB, eval_row=_EVAL_ROW, fewshot_row=_FEWSHOT_ROW, subjects=[_SUBJECT], expected=_ONESHOT
        )
