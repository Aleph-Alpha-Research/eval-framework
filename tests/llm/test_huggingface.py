from unittest.mock import Mock

import pytest
import torch

from eval_framework.llm.huggingface import SmolLM135M, StopSequenceCriteria
from eval_framework.shared.types import PromptTooLongException, RawCompletion, RawLoglikelihood
from eval_framework.tasks.base import Sample
from template_formatting.formatter import Message, Role


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
