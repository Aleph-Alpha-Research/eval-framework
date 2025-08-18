# ruff: noqa: E501
import importlib.util
from collections.abc import Callable
from functools import partial

import pytest

from template_formatting.formatter import (
    BaseFormatter,
    Llama3Formatter,
    Message,
    Qwen3Formatter,
    Qwen3ReasoningFormatter,
    Role,
)

package_exists = importlib.util.find_spec("transformers") is not None

if package_exists:
    from transformers import AutoTokenizer

# no tests requiring a GPU runner are contained here -> no additional pytest GPU markers


@pytest.fixture()
def llama3_formatter() -> BaseFormatter:
    return Llama3Formatter()


@pytest.fixture()
def qwen3_formatter() -> BaseFormatter:
    return Qwen3Formatter()


@pytest.fixture()
def qwen3_formatter_no_thinking() -> BaseFormatter:
    return Qwen3ReasoningFormatter(disable_thinking=True)


@pytest.fixture()
def qwen3_reasoning_formatter() -> BaseFormatter:
    return Qwen3ReasoningFormatter()


@pytest.mark.skipif(
    not package_exists,
    reason="`transformers` package is not installed, HFFormatter will not be available.",
)
@pytest.fixture()
def qwen3_hf_formatter() -> Callable:
    # For more details, see https://huggingface.co/Qwen/Qwen3-8B/blob/main/tokenizer_config.json
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    return partial(tokenizer.apply_chat_template, tokenize=False, add_generation_prompt=True, enable_thinking=False)


@pytest.mark.skipif(
    not package_exists,
    reason="`transformers` package is not installed, HFFormatter will not be available.",
)
@pytest.fixture()
def qwen3_reasoning_hf_formatter() -> Callable:
    # For more details, see https://huggingface.co/Qwen/Qwen3-8B/blob/main/tokenizer_config.json
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

    def apply_chat_template_with_thinking(messages: list[dict]) -> str:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        # hf allows the model to decide whether to think or not
        if messages[-1]["role"] == "user":
            return text + "<think>\n"
        elif messages[-1]["role"] == "assistant":
            text = text + "<think>\n"
        return text

    return apply_chat_template_with_thinking


@pytest.mark.skipif(
    not package_exists,
    reason="`transformers` package is not installed, HFFormatter will not be available.",
)
@pytest.mark.parametrize(
    "messages, test_name",
    [
        pytest.param(
            [
                Message(role=Role.SYSTEM, content="You are a helpful AI assistant for travel tips and recommendations"),
                Message(
                    role=Role.USER, content="What is France's capital?\n"
                ),  # new line has to be handled on task level
                Message(role=Role.ASSISTANT, content="Bonjour! The capital of France is Paris!"),
                Message(role=Role.USER, content="Great, thanks!"),
            ],
            "system_user_assistant_user",
            id="system_user_assistant_user",
        ),
        pytest.param(
            [
                Message(role=Role.SYSTEM, content="You are a helpful AI assistant for travel tips and recommendations"),
                Message(role=Role.USER, content="What is France's capital?"),
                Message(role=Role.ASSISTANT, content="Bonjour! The capital of France is Paris!"),
            ],
            "system_user_assistant_simple",
            id="system_user_assistant_simple",
        ),
        pytest.param(
            [
                Message(role=Role.USER, content="What is France's capital?"),
            ],
            "user_only",
            id="user_only",
        ),
        pytest.param(
            [
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
            ],
            "multiple_rounds_no_system",
            id="multiple_rounds_no_system",
        ),
        pytest.param(
            [
                Message(role=Role.USER, content="How many helicopters can a human eat in one sitting?"),
                Message(role=Role.ASSISTANT, content="A human can"),  # aka "cue"
            ],
            "prefilling_scenario",
            id="prefilling_scenario",
        ),
        pytest.param(
            [
                Message(role=Role.USER, content="  What is the capital of France?  "),
                Message(role=Role.ASSISTANT, content="  The capital of France is  "),
            ],
            "whitespace_handling",
            id="whitespace_handling",
        ),
    ],
)
def test_qwen3_formatter(
    qwen3_formatter: BaseFormatter, qwen3_hf_formatter: Callable, messages: list[Message], test_name: str
) -> None:
    hf_messages = [{"role": msg.role.value, "content": msg.content} for msg in messages if msg.role is not None]
    expected_hf_output = qwen3_hf_formatter(hf_messages)
    formatted_conversation = qwen3_formatter.format(messages, output_mode="string")
    assert formatted_conversation == expected_hf_output


@pytest.mark.skipif(
    not package_exists,
    reason="`transformers` package is not installed, HFFormatter will not be available.",
)
@pytest.mark.parametrize(
    "messages, test_name",
    [
        pytest.param(
            [
                Message(role=Role.SYSTEM, content="You are a helpful AI assistant for travel tips and recommendations"),
                Message(
                    role=Role.USER, content="What is France's capital?\n"
                ),  # new line has to be handled on task level
                Message(role=Role.ASSISTANT, content="Bonjour! The capital of France is Paris!"),
                Message(role=Role.USER, content="Great, thanks!"),
            ],
            "system_user_assistant_user",
            id="system_user_assistant_user",
        ),
        pytest.param(
            [
                Message(role=Role.SYSTEM, content="You are a helpful AI assistant for travel tips and recommendations"),
                Message(role=Role.USER, content="What is France's capital?"),
                Message(role=Role.ASSISTANT, content="Bonjour! The capital of France is Paris!"),
            ],
            "system_user_assistant_simple",
            id="system_user_assistant_simple",
        ),
        pytest.param(
            [
                Message(role=Role.USER, content="What is France's capital?"),
            ],
            "user_only",
            id="user_only",
        ),
        pytest.param(
            [
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
            ],
            "multiple_rounds_no_system",
            id="multiple_rounds_no_system",
        ),
        pytest.param(
            [
                Message(role=Role.USER, content="How many helicopters can a human eat in one sitting?"),
                Message(role=Role.ASSISTANT, content="A human can"),  # aka "cue"
            ],
            "prefilling_scenario",
            id="prefilling_scenario",
        ),
        pytest.param(
            [
                Message(role=Role.USER, content="  What is the capital of France?  "),
                Message(role=Role.ASSISTANT, content="  The capital of France is  "),
            ],
            "whitespace_handling",
            id="whitespace_handling",
        ),
    ],
)
def test_qwen3_no_thinking_formatter(
    qwen3_formatter_no_thinking: BaseFormatter, qwen3_hf_formatter: Callable, messages: list[Message], test_name: str
) -> None:
    hf_messages = [{"role": msg.role.value, "content": msg.content} for msg in messages if msg.role is not None]
    expected_hf_output = qwen3_hf_formatter(hf_messages)
    formatted_conversation = qwen3_formatter_no_thinking.format(messages, output_mode="string")
    assert formatted_conversation == expected_hf_output


@pytest.mark.skipif(
    not package_exists,
    reason="`transformers` package is not installed, HFFormatter will not be available.",
)
@pytest.mark.parametrize(
    "messages, test_name",
    [
        pytest.param(
            [
                Message(role=Role.SYSTEM, content="You are a helpful AI assistant for travel tips and recommendations"),
                Message(
                    role=Role.USER, content="What is France's capital?\n"
                ),  # new line has to be handled on task level
                Message(role=Role.ASSISTANT, content="Bonjour! The capital of France is Paris!"),
                Message(role=Role.USER, content="Great, thanks!"),
            ],
            "system_user_assistant_user",
            id="system_user_assistant_user",
        ),
        pytest.param(
            [
                Message(role=Role.SYSTEM, content="You are a helpful AI assistant for travel tips and recommendations"),
                Message(role=Role.USER, content="What is France's capital?"),
                Message(role=Role.ASSISTANT, content="Bonjour! The capital of France is Paris!"),
            ],
            "system_user_assistant_simple",
            id="system_user_assistant_simple",
        ),
        pytest.param(
            [
                Message(role=Role.USER, content="What is France's capital?"),
            ],
            "user_only",
            id="user_only",
        ),
        pytest.param(
            [
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
            ],
            "multiple_rounds_no_system",
            id="multiple_rounds_no_system",
        ),
        pytest.param(
            [
                Message(role=Role.USER, content="How many helicopters can a human eat in one sitting?"),
                Message(role=Role.ASSISTANT, content="A human can"),  # aka "cue"
            ],
            "prefilling_scenario",
            id="prefilling_scenario",
        ),
        pytest.param(
            [
                Message(role=Role.USER, content="  What is the capital of France?  "),
                Message(role=Role.ASSISTANT, content="  The capital of France is  "),
            ],
            "whitespace_handling",
            id="whitespace_handling",
        ),
    ],
)
def test_qwen3_reasoning_formatter(
    qwen3_reasoning_formatter: BaseFormatter,
    qwen3_reasoning_hf_formatter: Callable,
    messages: list[Message],
    test_name: str,
) -> None:
    hf_messages = [{"role": msg.role.value, "content": msg.content} for msg in messages if msg.role is not None]
    expected_hf_output = qwen3_reasoning_hf_formatter(hf_messages)
    formatted_conversation = qwen3_reasoning_formatter.format(messages, output_mode="string")
    print(formatted_conversation)
    print(expected_hf_output)
    assert formatted_conversation == expected_hf_output
