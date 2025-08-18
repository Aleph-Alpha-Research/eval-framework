# ruff: noqa: E501
import importlib.util

import pytest

from template_formatting.formatter import (
    BaseFormatter,
    ConcatFormatter,
    HFFormatter,
    Llama3Formatter,
    Message,
    Property,
    ReasoningFormatter,
    Role,
)

package_exists = importlib.util.find_spec("transformers") is not None


@pytest.fixture()
def concat_formatter() -> BaseFormatter:
    return ConcatFormatter()


@pytest.fixture()
def llama3_formatter() -> BaseFormatter:
    return Llama3Formatter()


@pytest.fixture()
def hf_formatter() -> BaseFormatter:
    return HFFormatter("meta-llama/Meta-Llama-3-8B-Instruct")


def test_get_grouped_messages_same_property() -> None:
    defaults = {"content": "dummy", "has_loss": False, "type": "text"}

    messages = [
        Message(role=Role.USER, property=None, **defaults),
        Message(role=Role.ASSISTANT, property=None, **defaults),
        Message(role=Role.ASSISTANT, property=None, **defaults),
    ]

    grouped_messages = BaseFormatter._get_grouped_messages(messages)

    assert grouped_messages == [
        [Message(role=Role.USER, property=None, **defaults)],
        [Message(role=Role.ASSISTANT, property=None, **defaults)],
        [Message(role=Role.ASSISTANT, property=None, **defaults)],
    ]


def test_get_grouped_messages_different_property() -> None:
    defaults = {"content": "dummy", "has_loss": False, "type": "text"}

    messages = [
        Message(role=Role.USER, property=None, **defaults),
        Message(role=Role.ASSISTANT, property=None, **defaults),
        Message(role=Role.ASSISTANT, property=Property.ANSWER, **defaults),
    ]

    grouped_messages = BaseFormatter._get_grouped_messages(messages)

    assert grouped_messages == [
        [Message(role=Role.USER, property=None, **defaults)],
        [
            Message(role=Role.ASSISTANT, property=None, **defaults),
            Message(role=Role.ASSISTANT, property=Property.ANSWER, **defaults),
        ],
    ]


def test_base_verify_messages() -> None:
    defaults = {"content": "dummy", "has_loss": False, "type": "text"}

    messages = [
        Message(role=Role.USER, property=None, **defaults),
        Message(role=Role.ASSISTANT, property=None, **defaults),
    ]

    # Does not raise an assertion error.
    BaseFormatter._verify_messages(messages)


def test_base_verify_messages_raises_exception() -> None:
    defaults = {"content": "dummy", "has_loss": False, "type": "text"}

    messages = [
        Message(role=Role.USER, property=None, **defaults),
        Message(role=Role.ASSISTANT, property=None, **defaults),
        Message(role=Role.ASSISTANT, property=None, **defaults),
    ]

    with pytest.raises(AssertionError):
        BaseFormatter._verify_messages(messages)


def test_reasoning_verify_messages() -> None:
    defaults = {"content": "dummy", "has_loss": False, "type": "text"}

    messages = [
        Message(role=Role.USER, property=None, **defaults),
        Message(role=Role.ASSISTANT, property=Property.THOUGHT, **defaults),
        Message(role=Role.ASSISTANT, property=Property.SOLUTION, **defaults),
        Message(role=Role.ASSISTANT, property=Property.ANSWER, **defaults),
    ]

    # Does not raise an assertion error.
    ReasoningFormatter._verify_messages(messages)


def test_reasoning_verify_messages_raises_exception() -> None:
    defaults = {"content": "dummy", "has_loss": False, "type": "text"}

    messages = [
        Message(role=Role.USER, property=None, **defaults),
        Message(role=Role.ASSISTANT, property=None, **defaults),
        Message(role=Role.ASSISTANT, property=Property.ANSWER, **defaults),
    ]

    with pytest.raises(AssertionError):
        ReasoningFormatter._verify_messages(messages)


## Assert that formatting is in line with HF Formatter


@pytest.mark.skipif(
    not package_exists,
    reason="`transformers` package is not installed, HFFormatter will not be available.",
)
def test_llama3_formatter_with_system_and_assistant_simple(
    llama3_formatter: BaseFormatter, hf_formatter: BaseFormatter
) -> None:
    conversation = [
        Message(
            role=Role.SYSTEM,
            content="You are a helpful AI assistant for travel tips and recommendations",
            has_loss=False,
            type="text",
        ),
        Message(role=Role.USER, content="What is France's capital?", has_loss=False, type="text"),
        Message(role=Role.ASSISTANT, content="Bonjour! The capital of France is Paris!", has_loss=True, type="text"),
    ]

    formatted_conversation = llama3_formatter.format(conversation, output_mode="list")

    expected_contents = [
        (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n\n"
            "You are a helpful AI assistant for travel tips and recommendations<|eot_id|>"
        ),
        (
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "What is France's capital?<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
        "Bonjour! The capital of France is Paris!<|eot_id|>",
    ]

    for formatted_message, expected in zip(formatted_conversation, expected_contents):
        assert formatted_message.content == expected

    # stringify the list
    formatted_conversation_str = "".join(elm.content for elm in formatted_conversation)

    expected_output_str = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a helpful AI assistant for travel tips and recommendations<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"  # noqa: E501
        "What is France's capital?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        "Bonjour! The capital of France is Paris!<|eot_id|>"
    )

    assert formatted_conversation_str == expected_output_str

    hf_formatted_conversation = hf_formatter.format(conversation, output_mode="list")
    assert hf_formatted_conversation == formatted_conversation_str


@pytest.mark.skipif(
    not package_exists,
    reason="`transformers` package is not installed, HFFormatter will not be available.",
)
def test_llama3_formatter_without_system_multiple_rounds_list(
    llama3_formatter: BaseFormatter, hf_formatter: BaseFormatter
) -> None:
    conversation = [
        Message(role=Role.USER, content="What is France's capital?", has_loss=False, type="text"),
        Message(role=Role.ASSISTANT, content="Bonjour! The capital of France is Paris!", has_loss=True, type="text"),
        Message(role=Role.USER, content="What can I do there?", has_loss=False, type="text"),
        Message(
            role=Role.ASSISTANT,
            content=(
                "Paris offers many attractions and activities. "
                "Some popular things to do include visiting the Eiffel Tower, "
                "exploring the Louvre Museum, taking a river cruise along the Seine, "
                "and strolling through charming neighborhoods like Montmartre."
            ),
            has_loss=False,
            type="text",
        ),
        Message(role=Role.USER, content="What else?", has_loss=False, type="text"),
    ]

    original_conversation = conversation.copy()

    formatted_conversation = llama3_formatter.format(conversation, output_mode="list")

    expected_contents = [
        (
            "<|begin_of_text|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "What is France's capital?<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
        "Bonjour! The capital of France is Paris!<|eot_id|>",
        (
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "What can I do there?<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
        (
            "Paris offers many attractions and activities. Some popular things to do include visiting the Eiffel Tower, "
            "exploring the Louvre Museum, taking a river cruise along the Seine, and strolling through charming neighborhoods like Montmartre.<|eot_id|>"
        ),
        ("<|start_header_id|>user<|end_header_id|>\n\nWhat else?<|eot_id|>"),
    ]

    for formatted_message, expected in zip(formatted_conversation, expected_contents):
        assert formatted_message.content == expected

    # stringify the list
    formatted_conversation_str = "".join(elm.content for elm in formatted_conversation)

    expected_output_str = (
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
        "What else?<|eot_id|>"
    )
    assert formatted_conversation_str == expected_output_str

    hf_formatted_conversation = hf_formatter.format(original_conversation, output_mode="list")
    assert hf_formatted_conversation == expected_output_str
