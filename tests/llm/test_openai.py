import pytest

from eval_framework.llm.openai import (
    Deepseek_chat,
    Deepseek_chat_with_formatter,
    OpenAI_davinci_002,
    OpenAI_gpt_4o_mini,
    OpenAI_gpt_4o_mini_with_ConcatFormatter,
)
from eval_framework.shared.types import RawCompletion, RawLoglikelihood
from eval_framework.tasks.base import Sample
from template_formatting.formatter import Message, Role

# NOTE: an API key in your .env file is required to run these API tests


@pytest.mark.external_api
# @pytest.mark.xfail(strict=False, reason="External API models are flaky or not required to always pass.")
# some of the responses are not deterministic and may cause an expected failure in the test (eg. deepseek-reasoner)
@pytest.mark.parametrize(
    "model_cls,expected_completion,max_tokens",
    [
        (OpenAI_gpt_4o_mini, "The night sky", 10),  # test chat completions
        (OpenAI_gpt_4o_mini_with_ConcatFormatter, " The night sky", 10),  # test formatted completions
        (Deepseek_chat, "The night sky appears", 10),  # test chat completions
        # (Deepseek_reasoner, "That's an excellent question", 300),  # test chat completions (needs enough tokens for reasoning)  # noqa: E501
        (
            Deepseek_chat_with_formatter,
            " The color of the night sky is primarily black",
            10,
        ),  # using formatter rather than chat templates
    ],
)
def test_openai_completions(model_cls, expected_completion, max_tokens) -> None:
    model = model_cls()

    messages: list[Message] = [
        Message(role=Role.USER, content="Question: What color is the night sky?\n"),
        Message(role=Role.ASSISTANT, content="Answer:"),
    ]

    generation_results: list[RawCompletion] = model.generate_from_messages(
        messages=[messages, messages], stop_sequences=["\n"], max_tokens=max_tokens, temperature=0
    )

    assert len(generation_results) == 2
    for generation_result in generation_results:
        assert expected_completion in generation_result.completion


@pytest.mark.external_api
# @pytest.mark.xfail(strict=False, reason="External API models are flaky or not required to always pass.")
@pytest.mark.parametrize(
    "model_cls",
    [
        OpenAI_davinci_002,
    ],
)
def test_openai_loglikelihoods(model_cls) -> None:
    model = model_cls()

    messages: list[Message] = [
        Message(role=Role.USER, content="Question: What color is the night sky?\n"),
        Message(role=Role.ASSISTANT, content="Answer:"),
    ]

    list_of_samples = [
        Sample(
            id=0,
            subject="no_subject",
            messages=messages,
            ground_truth="black",
            possible_completions=[" red", " blue", " black", " white"],
            context=None,
        ),
        Sample(
            id=0,
            subject="no_subject",
            messages=messages,
            ground_truth="foo",
            possible_completions=[" foo", " bar"],
            context=None,
        ),
    ]

    results: list[RawLoglikelihood] = model.logprobs(list_of_samples)

    assert len(results) == 2
    assert set(results[0].loglikelihoods.keys()) == {" red", " blue", " black", " white"}
    assert set(results[0].loglikelihoods_sequence_positions.keys()) == {" red", " blue", " black", " white"}
    assert set(results[1].loglikelihoods.keys()) == {" foo", " bar"}
    assert set(results[1].loglikelihoods_sequence_positions.keys()) == {" foo", " bar"}
