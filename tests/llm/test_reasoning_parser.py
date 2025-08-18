# ruff: noqa: E501
from dataclasses import asdict

import pytest

from template_formatting.formatter import (
    Llama3Formatter,
    ReasoningFormatter,
    ReasoningTemplate,
)


@pytest.fixture()
def formatter() -> ReasoningFormatter:
    formatter = ReasoningFormatter(Llama3Formatter)
    formatter.template.end_of_text = "<|end_of_text|>"
    return formatter


@pytest.fixture()
def template(formatter: ReasoningFormatter) -> ReasoningTemplate:
    return formatter.template


@pytest.mark.parametrize(
    "input_text_template, expected_output",
    [
        (
            "{begin_thought_id}i am thinking{end_thought_id}{begin_thought_id}extra{end_thought_id}{begin_solution_id}i know the solution{end_solution_id}{begin_answer_id}\\boxed[answer]{end_answer_id}{end_of_text}",  # noqa: E501
            "error",
        ),  # duplicate token
        (
            "{begin_thought_id}i am thinking{end_thought_id}{begin_answer_id}\\boxed[answer]{end_answer_id}{end_of_text}",  # noqa: E501
            "error",
        ),  # missing token
        (
            "{begin_thought_id}i am thinking{end_thought_id}{begin_answer_id}\\boxed[answer]{end_answer_id}{begin_solution_id}i know the solution{end_solution_id}{end_of_text}",  # noqa: E501
            "error",
        ),  # wrong order
        ("{begin_thought_id}i am thinking{end_of_text}", "not_finished_thinking"),  # still thinking
        (
            "{begin_thought_id}i am thinking{end_thought_id}{begin_solution_id}i know the solution{end_of_text}",
            "incomplete",
        ),  # incomplete
        (
            "{begin_thought_id}i am thinking{end_thought_id}{begin_solution_id}i know the solution{end_solution_id}{begin_answer_id}\\boxed[answer]{end_answer_id}{end_of_text}",  # noqa: E501
            "valid",
        ),  # complete and valid
    ],
)
def test_validate_output(
    formatter: ReasoningFormatter, template: ReasoningTemplate, input_text_template: str, expected_output: str
) -> None:
    format_dict = asdict(template)
    input_text = input_text_template.format(**format_dict)
    validation = formatter._validate_output(input_text)
    assert validation[0] == expected_output


@pytest.mark.parametrize(
    "formatted_text, thought_only, expected_output",
    [
        (
            "{begin_thought_id}i am thinking{end_thought_id}{begin_solution_id}i know the solution{end_solution_id}{begin_answer_id}\\boxed[answer]{end_answer_id}{end_of_text}",  # noqa: E501
            False,  # complete output
            {
                "thought": "i am thinking",
                "solution": "i know the solution",
                "answer": "\\boxed[answer]",
            },
        ),
        (
            "{begin_thought_id}i am thinking{end_thought_id}{end_of_text}",
            True,  # thought only
            {
                "thought": "i am thinking",
                "solution": "",
                "answer": "",
            },
        ),
        (
            "{begin_thought_id}i am thinking{end_thought_id}{begin_solution_id}i know the solution{end_solution_id}{end_of_text}",  # noqa: E501
            True,  # incomplete output
            {
                "thought": "i am thinking",
                "solution": "",
                "answer": "",
            },
        ),
    ],
)
def test_parse_output_of_valid_input(
    formatter: ReasoningFormatter,
    template: ReasoningTemplate,
    formatted_text: str,
    thought_only: bool,
    expected_output: dict[str, str],
) -> None:
    format_dict = asdict(template)
    input_text = formatted_text.format(**format_dict)
    output = formatter._parse_output(input_text, thought_only)
    assert output == expected_output


@pytest.mark.parametrize(
    "formatted_text, expected_error",
    [
        (
            "{begin_thought_id}i am thinking{end_thought_id}{begin_answer_id}\\boxed[answer]{end_answer_id}{end_of_text}",  # noqa: E501
            "Parsing failed: Output format does not match expected structure.",
        ),
        (
            "{begin_thought_id}i am thinking{end_thought_id}{begin_answer_id}\\boxed[answer]{end_answer_id}{begin_solution_id}i know the solution{end_solution_id}{end_of_text}",  # noqa: E501
            "Parsing failed: Output format does not match expected structure.",
        ),
    ],
)
def test_parse_output_of_invalid_input(
    formatter: ReasoningFormatter, template: ReasoningTemplate, formatted_text: str, expected_error: str
) -> None:
    format_dict = asdict(template)
    input_text = formatted_text.format(**format_dict)
    with pytest.raises(ValueError) as e:
        formatter._parse_output(input_text, thought_only=False)
    assert str(e.value) == expected_error


@pytest.mark.parametrize(
    "input_text_template, expected_output",
    [
        (  # duplicate token
            "{begin_thought_id}i am thinking{end_thought_id}{begin_thought_id}extra{end_thought_id}{begin_solution_id}i know the solution{end_solution_id}{begin_answer_id}\\boxed[answer]{end_answer_id}{end_of_text}",  # noqa: E501
            ({}, "Duplicate tokens detected: '{begin_thought_id}' appears 2 times."),
        ),
        (  # missing token
            "{begin_thought_id}i am thinking{end_thought_id}{begin_answer_id}\\boxed[answer]{end_answer_id}{end_of_text}",  # noqa: E501
            ({}, "Missing token: Expected '{begin_solution_id}' but found '{begin_answer_id}' instead."),
        ),
        (  # wrong order
            "{begin_thought_id}i am thinking{end_thought_id}{begin_answer_id}\\boxed[answer]{end_answer_id}{begin_solution_id}i know the solution{end_solution_id}{end_of_text}",  # noqa: E501
            ({}, "Incorrect token order: '{begin_answer_id}' appears before expected."),
        ),
        (  # still thinking
            "{begin_thought_id}i am thinking{end_of_text}",
            (
                {
                    "thought": "i am thinking",
                    "solution": "",
                    "answer": "",
                },
                None,
            ),
        ),
        (  # incomplete
            "{begin_thought_id}i am thinking{end_thought_id}{begin_solution_id}i know the solution{end_of_text}",
            (
                {
                    "thought": "i am thinking",
                    "solution": "",
                    "answer": "",
                },
                None,
            ),
        ),
        (  # complete and valid
            "{begin_thought_id}i am thinking{end_thought_id}{begin_solution_id}i know the solution{end_solution_id}{begin_answer_id}\\boxed[answer]{end_answer_id}{end_of_text}",  # noqa: E501
            (
                {
                    "thought": "i am thinking",
                    "solution": "i know the solution",
                    "answer": "\\boxed[answer]",
                },
                None,
            ),
        ),
    ],
)
def test_parse(
    formatter: ReasoningFormatter,
    template: ReasoningTemplate,
    input_text_template: str,
    expected_output: tuple[dict[str, str], str],
) -> None:
    format_dict = asdict(template)
    input_text = input_text_template.format(**format_dict)
    expected_dict, error_msg = expected_output
    expected_error = None if error_msg is None else ValueError(error_msg.format(**format_dict))
    output_dict, error = formatter.parse(input_text)
    print(output_dict)
    print(error)
    print(expected_error)
    assert output_dict == expected_dict
    assert isinstance(error, type(expected_error))
    assert str(error) == str(expected_error)
