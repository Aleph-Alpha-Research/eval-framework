# ruff: noqa: E501
import importlib.util

import pytest

from eval_framework.llm.models import Smollm135MInstruct
from template_formatting.formatter import (
    BaseFormatter,
    ConcatFormatter,
    Llama3Formatter,
    Message,
    Property,
    ReasoningFormatter,
    Role,
    get_formatter,
)

package_exists = importlib.util.find_spec("transformers") is not None

# no tests requiring a GPU runner are contained here -> no additional pytest GPU markers


@pytest.fixture()
def concat_formatter() -> BaseFormatter:
    return ConcatFormatter()


@pytest.fixture()
def llama3_formatter() -> BaseFormatter:
    return Llama3Formatter()


@pytest.fixture()
def hf_formatter() -> BaseFormatter:
    return Smollm135MInstruct.DEFAULT_FORMATTER()


@pytest.fixture()
def llama3_reasoning_formatter() -> BaseFormatter:
    llama3_reasoning_formatter = ReasoningFormatter(Llama3Formatter)
    llama3_reasoning_formatter.template.end_of_text = "<|end_of_text|>"
    return llama3_reasoning_formatter


def test_concat_formatter(concat_formatter: BaseFormatter) -> None:
    messages = [
        Message(role=Role.SYSTEM, content="You are a helpful AI assistant for travel tips and recommendations"),
        Message(role=Role.USER, content="What is France's capital?\n"),  # new line has to be handled on task level
        Message(role=Role.ASSISTANT, content="Bonjour! The capital of France is Paris!"),
        Message(role=Role.USER, content="Great, thanks!"),
    ]

    formatted_conversation = concat_formatter.format(messages, output_mode="string")
    expected_output = (
        "You are a helpful AI assistant for travel tips and recommendations\n\n"
        "What is France's capital?\n"
        "Bonjour! The capital of France is Paris!\n\n"
        "Great, thanks!"
    )

    assert formatted_conversation == expected_output


@pytest.mark.skipif(
    not package_exists,
    reason="`transformers` package is not installed, HFFormatter will not be available.",
)
def test_llama3_formatter_with_system_and_assistant_simple(
    llama3_formatter: BaseFormatter, hf_formatter: BaseFormatter
) -> None:
    conversation = [
        Message(role=Role.SYSTEM, content="You are a helpful AI assistant for travel tips and recommendations"),
        Message(role=Role.USER, content="What is France's capital?"),
        Message(role=Role.ASSISTANT, content="Bonjour! The capital of France is Paris!"),
    ]

    formatted_conversation = llama3_formatter.format(conversation, output_mode="string")
    expected_output = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a helpful AI assistant for travel tips and recommendations<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        "What is France's capital?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        "Bonjour! The capital of France is Paris!"
    )

    assert formatted_conversation == expected_output

    hf_formatted_conversation = hf_formatter.format(conversation, output_mode="string")
    assert hf_formatted_conversation == expected_output


@pytest.mark.skipif(
    not package_exists,
    reason="`transformers` package is not installed, HFFormatter will not be available.",
)
def test_llama3_formatter_with_system_and_assistant(
    llama3_formatter: BaseFormatter, hf_formatter: BaseFormatter
) -> None:
    conversation = [
        Message(role=Role.SYSTEM, content="You are a helpful AI assistant for travel tips and recommendations"),
        Message(role=Role.USER, content="What is France's capital?"),
        Message(role=Role.ASSISTANT, content="Bonjour! The capital of France is Paris!"),
        Message(role=Role.USER, content="Great, thanks!"),
    ]

    formatted_conversation = llama3_formatter.format(conversation, output_mode="string")
    expected_output = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a helpful AI assistant for travel tips and recommendations<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        "What is France's capital?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        "Bonjour! The capital of France is Paris!<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        "Great, thanks!<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    assert formatted_conversation == expected_output

    hf_formatted_conversation = hf_formatter.format(conversation, output_mode="string")
    assert hf_formatted_conversation == expected_output


@pytest.mark.skipif(
    not package_exists,
    reason="`transformers` package is not installed, HFFormatter will not be available.",
)
def test_llama3_formatter_without_system_and_assistant(
    llama3_formatter: BaseFormatter, hf_formatter: BaseFormatter
) -> None:
    conversation = [
        Message(role=Role.USER, content="What is France's capital?"),
    ]

    formatted_conversation = llama3_formatter.format(conversation, output_mode="string")
    expected_output = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        "What is France's capital?<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    assert formatted_conversation == expected_output

    hf_formatted_conversation = hf_formatter.format(conversation, output_mode="string")
    assert hf_formatted_conversation == expected_output


@pytest.mark.skipif(
    not package_exists,
    reason="`transformers` package is not installed, HFFormatter will not be available.",
)
def test_llama3_formatter_without_system_multiple_rounds(
    llama3_formatter: BaseFormatter, hf_formatter: BaseFormatter
) -> None:
    conversation = [
        Message(role=Role.USER, content="What is France's capital?"),
        Message(role=Role.ASSISTANT, content="Bonjour! The capital of France is Paris!"),
        Message(role=Role.USER, content="What can I do there?"),
        Message(
            role=Role.ASSISTANT,
            content=(
                "Paris offers many attractions and activities. "
                "Some popular things to do include visiting the Eiffel Tower, "
                "exploring the Louvre Museum, taking a river cruise along the Seine, "
                "and strolling through charming neighborhoods like Montmartre."
            ),
        ),
        Message(role=Role.USER, content="What else?"),
    ]

    formatted_conversation = llama3_formatter.format(conversation, output_mode="string")
    expected_output = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        "What is France's capital?<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        "Bonjour! The capital of France is Paris!<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        "What can I do there?<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        "Paris offers many attractions and activities. Some popular things to do "
        "include visiting the Eiffel Tower, exploring the Louvre Museum, taking a river "
        "cruise along the Seine, and strolling through charming neighborhoods like Montmartre.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        "What else?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    assert formatted_conversation == expected_output

    hf_formatted_conversation = hf_formatter.format(conversation, output_mode="string")
    assert hf_formatted_conversation == expected_output


@pytest.mark.skipif(
    not package_exists,
    reason="`transformers` package is not installed, HFFormatter will not be available.",
)
def test_llama3_formatter_with_prefilling(llama3_formatter: BaseFormatter, hf_formatter: BaseFormatter) -> None:
    conversation = [
        Message(role=Role.USER, content="How many helicopters can a human eat in one sitting?"),
        Message(role=Role.ASSISTANT, content="A human can"),  # aka "cue"
    ]

    formatted_conversation = llama3_formatter.format(conversation, output_mode="string")
    expected_output = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        "How many helicopters can a human eat in one sitting?<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        "A human can"
    )
    assert formatted_conversation == expected_output

    hf_formatted_conversation = hf_formatter.format(conversation, output_mode="string")
    assert hf_formatted_conversation == expected_output


@pytest.mark.skipif(
    not package_exists,
    reason="`transformers` package is not installed, HFFormatter will not be available.",
)
def test_stripping_of_whitespace(llama3_formatter: BaseFormatter, hf_formatter: BaseFormatter) -> None:
    conversation = [
        Message(role=Role.USER, content="  What is the capital of France?  "),
        Message(role=Role.ASSISTANT, content="  The capital of France is  "),  #
    ]

    formatted_conversation = llama3_formatter.format(conversation, output_mode="string")
    expected_output = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        "What is the capital of France?<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        "The capital of France is"
    )
    assert formatted_conversation == expected_output

    hf_formatted_conversation = hf_formatter.format(conversation, output_mode="string")
    assert hf_formatted_conversation == expected_output


@pytest.mark.parametrize(
    "model_name, expected_formatter",
    [
        pytest.param("llama-3", Llama3Formatter, id="llama-3"),
        pytest.param("llama-3-base", Llama3Formatter, id="llama-3-base"),
        pytest.param("llama-3-large", Llama3Formatter, id="llama-3-large"),
        pytest.param("my-llama-3-model", Llama3Formatter, id="custom-llama-3-model"),
        pytest.param("gpt2", ConcatFormatter, id="gpt2"),
        pytest.param("bert", ConcatFormatter, id="bert"),
        pytest.param("roberta", ConcatFormatter, id="roberta"),
        pytest.param("distilbert", ConcatFormatter, id="distilbert"),
        pytest.param("custom-model", ConcatFormatter, id="custom-non-llama3-model"),
        pytest.param("", ConcatFormatter, id="empty-model-name"),
    ],
)
def test_get_formatter(model_name: str, expected_formatter: type[BaseFormatter]) -> None:
    formatter = get_formatter(model_name)
    assert isinstance(formatter, expected_formatter)


# ReasoningFormatter tests


def test_reasoning_formatter_with_system_and_user(llama3_reasoning_formatter: BaseFormatter) -> None:
    conversation = [
        Message(role=Role.SYSTEM, content="You are a helpful AI assistant for travel tips and recommendations"),
        Message(role=Role.USER, content="What is France's capital?"),
    ]

    formatted_conversation = llama3_reasoning_formatter.format(conversation, output_mode="string")
    expected_output = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a helpful AI assistant for travel tips and recommendations<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        "What is France's capital?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        "<|begin_of_thought|>"
    )

    assert formatted_conversation == expected_output


def test_reasoning_formatter_with_user(llama3_reasoning_formatter: BaseFormatter) -> None:
    conversation = [
        Message(role=Role.USER, content="What is France's capital?"),
    ]

    formatted_conversation = llama3_reasoning_formatter.format(conversation, output_mode="string")
    expected_output = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        "What is France's capital?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        "<|begin_of_thought|>"
    )

    assert formatted_conversation == expected_output


def test_reasoning_formatter_with_system_user_and_thought(llama3_reasoning_formatter: BaseFormatter) -> None:
    conversation = [
        Message(role=Role.SYSTEM, content="You are a helpful AI assistant for travel tips and recommendations"),
        Message(role=Role.USER, content="What is France's capital?"),
        Message(role=Role.ASSISTANT, property=Property.THOUGHT, content="Bonjour! Let me think about this..."),
    ]

    formatted_conversation = llama3_reasoning_formatter.format(conversation, output_mode="string")
    expected_output = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a helpful AI assistant for travel tips and recommendations<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        "What is France's capital?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        "<|begin_of_thought|>Bonjour! Let me think about this...<|end_of_thought|>"
        "<|begin_of_solution|>"
    )
    assert formatted_conversation == expected_output


def test_reasoning_formatter_with_system_user_thought_and_solution(llama3_reasoning_formatter: BaseFormatter) -> None:
    conversation = [
        Message(role=Role.SYSTEM, content="You are a helpful AI assistant for travel tips and recommendations"),
        Message(role=Role.USER, content="What is France's capital?"),
        Message(role=Role.ASSISTANT, property=Property.THOUGHT, content="Bonjour! Let me think about this..."),
        Message(role=Role.ASSISTANT, property=Property.SOLUTION, content="Merci! The capital of France is Paris!"),
    ]

    formatted_conversation = llama3_reasoning_formatter.format(conversation, output_mode="string")
    expected_output = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a helpful AI assistant for travel tips and recommendations<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        "What is France's capital?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        "<|begin_of_thought|>Bonjour! Let me think about this..."
        "<|end_of_thought|><|begin_of_solution|>Merci! The capital of France is Paris!"
        "<|begin_of_answer|>"
    )

    assert formatted_conversation == expected_output


def test_reasoning_formatter_with_system_user_thought_solution_and_answer(
    llama3_reasoning_formatter: BaseFormatter,
) -> None:
    conversation = [
        Message(role=Role.SYSTEM, content="You are a helpful AI assistant for travel tips and recommendations"),
        Message(role=Role.USER, content="What is France's capital?"),
        Message(role=Role.ASSISTANT, property=Property.THOUGHT, content="Bonjour! Let me think about this..."),
        Message(role=Role.ASSISTANT, property=Property.SOLUTION, content="Merci! The capital of France is Paris!"),
        Message(role=Role.ASSISTANT, property=Property.ANSWER, content="\\boxed{Paris}"),
    ]

    formatted_conversation = llama3_reasoning_formatter.format(conversation, output_mode="string")
    expected_output = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a helpful AI assistant for travel tips and recommendations<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        "What is France's capital?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        "<|begin_of_thought|>Bonjour! Let me think about this...<|end_of_thought|>"
        "<|begin_of_solution|>Merci! The capital of France is Paris!"
        "<|begin_of_answer|>\\boxed{Paris}<|end_of_answer|><|end_of_solution|><|eot_id|><|end_of_text|>"
    )

    assert formatted_conversation == expected_output


def test_reasoning_formatter_parse_wrong_order() -> None:
    base_formatter = Llama3Formatter
    rf = ReasoningFormatter(base_formatter)
    rt = rf.template
    output_str = (
        rt.begin_thought_id
        + "thought"
        + rt.begin_solution_id
        + "solution"  # Wrong: begin_solution_id comes before end_thought_id.
        + rt.end_thought_id
        + rt.end_solution_id
        + rt.begin_answer_id
        + "answer"
        + rt.end_answer_id
        + rt.end_of_text
    )
    parsed, error = rf.parse(output_str)
    assert error is not None
    with pytest.raises(ValueError):
        raise error


def test_reasoning_formatter_parse_incomplete() -> None:
    base_formatter = Llama3Formatter
    rf = ReasoningFormatter(base_formatter)
    rt = rf.template

    output_str = rt.begin_thought_id + "only thought" + rt.end_thought_id
    parsed, error = rf.parse(output_str)
    assert error is None
    assert parsed["thought"] == "only thought"
    assert parsed.get("solution", "") == ""
    assert parsed.get("answer", "") == ""


def test_reasoning_formatter_parse_duplicate_tokens() -> None:
    base_formatter = Llama3Formatter
    rf = ReasoningFormatter(base_formatter)
    rt = rf.template

    output_str = (
        rt.begin_thought_id
        + "thought"
        + rt.begin_thought_id
        + "duplicate"
        + rt.end_thought_id
        + rt.begin_solution_id
        + "solution"
        + rt.end_solution_id
        + rt.begin_answer_id
        + "answer"
        + rt.end_answer_id
        + rt.end_of_text
    )
    parsed, error = rf.parse(output_str)
    assert error is not None
    with pytest.raises(ValueError):
        raise error
