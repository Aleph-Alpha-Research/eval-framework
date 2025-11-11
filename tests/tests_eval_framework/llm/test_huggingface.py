from typing import Any
from unittest.mock import MagicMock, Mock

import pytest
import torch
from pytest_mock import MockerFixture

from eval_framework.llm.huggingface import HFLLM, SmolLM135M, StopSequenceCriteria
from eval_framework.shared.types import PromptTooLongException, RawCompletion, RawLoglikelihood
from eval_framework.tasks.base import Sample
from template_formatting.formatter import (
    ConcatFormatter,
    HFFormatter,
    IdentityFormatter,
    Message,
    Role,
)
from tests.tests_eval_framework.llm.test_base import LLM_INIT_FORMATTER_PARAMS, LLM_INIT_SOURCE_PARAMS


@pytest.mark.gpu
def test_hf_llm() -> None:
    """Test a Hugging Face model with logprobs and completions."""

    model = SmolLM135M()

    messages: list[Message] = [
        Message(role=Role.USER, content="Question: What color is the night sky?\n"),
        Message(role=Role.ASSISTANT, content="Answer:"),
    ]

    # -- TEST LOGPROBS --
    list_of_samples = [
        Sample(
            id=0,
            subject="no_subject",
            messages=messages,
            ground_truth="black",
            possible_completions=["red", "blue", "black", "white"],
            context=None,
        ),
        Sample(
            id=0,
            subject="no_subject",
            messages=messages,
            ground_truth="foo",
            possible_completions=["foo", "bar"],
            context=None,
        ),
    ]

    results: list[RawLoglikelihood] = model.logprobs(list_of_samples)

    assert len(results) == 2
    assert set(results[0].loglikelihoods.keys()) == {"red", "blue", "black", "white"}
    assert set(results[0].loglikelihoods_sequence_positions.keys()) == {"red", "blue", "black", "white"}
    assert set(results[1].loglikelihoods.keys()) == {"foo", "bar"}
    assert set(results[1].loglikelihoods_sequence_positions.keys()) == {"foo", "bar"}

    # -- TEST COMPLETIONS --
    generation_results: list[RawCompletion] = model.generate_from_messages(
        messages=[messages, messages], stop_sequences=["\n"], max_tokens=4, temperature=0
    )

    assert len(generation_results) == 2
    for generation_result in generation_results:
        assert " The sky is blue" in generation_result.completion


@pytest.mark.gpu
def test_error_on_overly_long_prompt() -> None:
    """Test that the model raises an error when the prompt is too long."""

    model = SmolLM135M()

    # given a too long log-likelihood task ...
    sample = Sample(
        id=0,
        subject="no_subject",
        messages=[
            Message(role=Role.USER, content="Question: What color is the night sky?"),
            Message(role=Role.ASSISTANT, content="Answer:"),
        ],
        ground_truth="black",
        possible_completions=["red", "blue", "black", "white" * 10000],
        context=None,
    )

    lresults: list[RawLoglikelihood] = model.logprobs([sample])

    # ... the loglikelihoods should be empty and the error should be stored
    assert len(lresults) == 1
    assert not lresults[0].loglikelihoods
    assert not lresults[0].loglikelihoods_sequence_positions
    assert (
        lresults[0].raw_loglikelihood_error is not None
        and lresults[0].raw_loglikelihood_error.error_class == PromptTooLongException.__name__
    )

    # given a too long log-likelihood task ...
    msg = (Message(role=Role.USER, content="text" * 10000),)
    cresults: list[RawCompletion] = model.generate_from_messages(messages=[msg], max_tokens=4)

    # ... the completion should be empty and the error should be stored
    assert len(cresults) == 1
    assert cresults[0].completion == ""
    assert (
        cresults[0].raw_completion_error is not None
        and cresults[0].raw_completion_error.error_class == PromptTooLongException.__name__
    )


@pytest.mark.parametrize("stop_sequences", [[], ["stop", "end"]])
def test_stop_sequence_criteria(stop_sequences: list[str]) -> None:
    """Test StopSequenceCriteria with empty and non-empty stop sequences."""
    # Mock tokenizer
    mock_tokenizer = Mock()
    mock_tokenizer.decode.return_value = "decoded text end"

    criteria = StopSequenceCriteria(
        stop_sequences=stop_sequences,
        tokenizer=mock_tokenizer,
        prompt_token_count=2,
    )

    if not stop_sequences:
        input_ids = torch.LongTensor([[1, 2, 3, 4]])
        scores = torch.FloatTensor([[0.1, 0.2, 0.3, 0.4]])
        assert not criteria(input_ids, scores), "Criteria should return False when no stop sequences are provided."
    else:
        input_ids = torch.LongTensor([list(range(16))])
        scores = torch.FloatTensor([[0.1] * 16])
        assert criteria(input_ids, scores), "Text contains stop sequence, criteria should return True."


@pytest.mark.gpu
def test_resource_cleanup() -> None:
    class Qwen8B(HFLLM):
        LLM_NAME = "Qwen/Qwen3-8B"

    try:
        formatter = HFFormatter("Qwen/Qwen3-8B", chat_template_kwargs={"enable_thinking": True})
        generator_model = Qwen8B(formatter=formatter)
        generator_model.generate_from_messages(
            messages=[[Message(role=Role.USER, content="What is capital of Germany ?")]],
            max_tokens=100,
            temperature=0.0,
        )
        del generator_model
        judge_model = Qwen8B(formatter=formatter)
        judge_model.generate_from_messages(
            messages=[
                [
                    Message(
                        role=Role.USER,
                        content=(
                            "Rank the following responses between 0 and 1"
                            "based on clarity : \nResponse 1: {response1}"
                            "\nResponse 2: {response2}"
                        ).format(response1="This is an ambiguous answer", response2="This is a clear answer"),
                    )
                ]
            ],
            max_tokens=100,
            temperature=0.0,
        )
        del judge_model
    except Exception as e:
        pytest.fail(f"{e.__class__.__name__} : {e}")


@pytest.mark.parametrize("kwargs, expected_model, expected_name", LLM_INIT_SOURCE_PARAMS)
def test_hfllm_init_source(mocker: MockerFixture, kwargs: Any, expected_model: str, expected_name: str) -> None:
    """Test that VLLMModel initializes correctly with different checkpoint source arguments."""

    mocker.patch("eval_framework.llm.huggingface.AutoTokenizer.from_pretrained")
    HF_patch = mocker.patch("eval_framework.llm.huggingface.AutoModelForCausalLM.from_pretrained")

    mock_wandb_fs = MagicMock()
    mock_wandb_fs.__enter__().find_hf_checkpoint_root_from_path_list.return_value = "/download"
    mocker.patch("eval_framework.utils.file_ops.WandbFs", return_value=mock_wandb_fs)

    # Test with a typical subclass
    class MyModel(HFLLM):
        LLM_NAME = "org/model"
        DEFAULT_FORMATTER = ConcatFormatter

    model = MyModel(**kwargs)
    assert HF_patch.call_args[0][0] == expected_model
    assert model.name == expected_name
    assert model.LLM_NAME == expected_model

    # Test with the base class
    if not kwargs or list(kwargs.keys()) == ["checkpoint_name"]:  # no checkpoint source -> error
        with pytest.raises(ValueError):
            HFLLM(**kwargs)
    else:
        base_model = HFLLM(**kwargs, formatter=ConcatFormatter())
        assert HF_patch.call_args[0][0] == expected_model
        assert base_model.name == expected_name.replace("MyModel", "HFLLM")
        assert base_model.LLM_NAME == expected_model


def test_hfllm_init_source_multiple_args() -> None:
    """Test that providing multiple checkpoint source arguments raises an error."""
    with pytest.raises(ValueError):
        HFLLM(checkpoint_path="/ckpt/m", model_name="org/other")
    with pytest.raises(ValueError):
        HFLLM(checkpoint_path="/ckpt/m", artifact_name="art:v0")
    with pytest.raises(ValueError):
        HFLLM(model_name="org/other", artifact_name="art:v0")


@pytest.mark.parametrize("kwargs, expected_formatter_cls", LLM_INIT_FORMATTER_PARAMS)
def test_hfllm_init_formatter(mocker: MockerFixture, kwargs: Any, expected_formatter_cls: type) -> None:
    tokenizer_mock = mocker.patch("eval_framework.llm.huggingface.AutoTokenizer.from_pretrained")
    mocker.patch("eval_framework.llm.huggingface.AutoModelForCausalLM.from_pretrained")

    # Test with a typical subclass
    class MyModel(HFLLM):
        LLM_NAME = "org/model"
        DEFAULT_FORMATTER = ConcatFormatter

    model = MyModel(**kwargs)
    assert isinstance(model._formatter, expected_formatter_cls)

    # Test with the base class
    if len(kwargs) <= 1:  # no formatter -> error
        tokenizer_mock.return_value.chat_template = None
        with pytest.raises(ValueError):
            HFLLM(**kwargs)
    else:
        base_model = HFLLM(**kwargs)
        assert isinstance(base_model._formatter, expected_formatter_cls)


def test_hfllm_init_formatter_multiple_args() -> None:
    """Test that providing multiple formatter arguments raises an error."""
    with pytest.raises(ValueError):
        HFLLM(formatter=IdentityFormatter(), formatter_name="Llama3Formatter")
    with pytest.raises(ValueError):
        HFLLM(formatter=IdentityFormatter(), formatter_kwargs=dict(hf_llm_name="HuggingFaceTB/SmolLM-135M"))


@pytest.mark.gpu
def test_max_tokens_generation() -> None:
    model = SmolLM135M(bytes_per_token=4.0)

    messages: list[Message] = [
        Message(role=Role.USER, content="Tell me a long story.\n"),
        Message(role=Role.ASSISTANT, content="Once upon a time,"),
    ]

    generation_results: list[RawCompletion] = model.generate_from_messages(
        messages=[messages], max_tokens=10, temperature=0
    )

    assert len(generation_results) == 1
    generated_num_tokens = generation_results[0].completion_sequence_positions
    assert generated_num_tokens == 10

    byte_level_model = SmolLM135M(bytes_per_token=1.0)

    byte_level_model_messages: list[Message] = [
        Message(role=Role.USER, content="Tell me a long story.\n"),
        Message(role=Role.ASSISTANT, content="Once upon a time,"),
    ]

    byte_level_model_generation_results: list[RawCompletion] = byte_level_model.generate_from_messages(
        messages=[byte_level_model_messages], max_tokens=10, temperature=0
    )

    assert len(byte_level_model_generation_results) == 1
    byte_level_model_generated_num_tokens = byte_level_model_generation_results[0].completion_sequence_positions
    assert byte_level_model_generated_num_tokens == 40
