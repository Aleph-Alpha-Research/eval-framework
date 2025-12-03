import json
from typing import Any, NamedTuple, cast
from unittest import mock

import pytest

import eval_framework.llm.aleph_alpha as aleph_alpha
from eval_framework.llm.aleph_alpha import AlephAlphaAPIModel, Llama31_8B_Instruct_API
from eval_framework.shared.types import PromptTooLongException, RawCompletion, RawLoglikelihood
from eval_framework.tasks.base import Sample
from template_formatting.formatter import Message, Role


# Mocking because of the fluff in the actual implementation
class MockCompletionResult(NamedTuple):
    completion_tokens: list[str]
    log_probs: list[dict[str, float]] | None


class MockCompletionResponse(NamedTuple):
    completions: list[MockCompletionResult]


@pytest.mark.external_api
@pytest.mark.xfail(strict=False, reason="External API models are flaky or not required to always pass.")
def test_aleph_alpha() -> None:
    model = Llama31_8B_Instruct_API(max_retries=0)

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
            ground_truth=" black",
            possible_completions=[" red", " blue", " black", " white"],
            context=None,
        ),
        Sample(
            id=0,
            subject="no_subject",
            messages=messages,
            ground_truth=" foo",
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

    # -- TEST COMPLETIONS --
    generation_results: list[RawCompletion] = model.generate_from_messages(
        messages=[messages, messages], stop_sequences=["\n"], max_tokens=4, temperature=0
    )

    assert len(generation_results) == 2
    for generation_result in generation_results:
        assert " The night sky is" in generation_result.completion


@pytest.mark.external_api
@mock.patch.object(aleph_alpha.AsyncClient, "complete", new_callable=mock.AsyncMock)
@mock.patch.object(aleph_alpha.AsyncClient, "evaluate", new_callable=mock.AsyncMock)
def test_error_on_overly_long_prompt(mock_complete: mock.AsyncMock, mock_evaluate: mock.AsyncMock) -> None:
    # Let's mock the API since redis wrapper doesn't cache errors and inference scheduler can return busy state.
    mock_error = RuntimeError(400, json.dumps({"error": "xyz", "code": "PROMPT_TOO_LONG"}))
    mock_complete.side_effect = mock_evaluate.side_effect = mock_error

    model = Llama31_8B_Instruct_API(max_retries=0)

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


@pytest.mark.external_api
def test_max_tokens_generation() -> None:
    model = Llama31_8B_Instruct_API(max_retries=0, bytes_per_token=4.0)

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

    byte_level_model = Llama31_8B_Instruct_API(max_retries=0, bytes_per_token=1.0)

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


def test_count_prompt_tokens_perfect_match() -> None:
    # Given
    tokens = ["Hello", " ", "world"]
    prompt = "Hello world"

    # When
    count = AlephAlphaAPIModel._count_prompt_tokens_from_sequence(tokens, prompt)  # type: ignore

    # Then
    assert count == 3


def test_count_prompt_tokens_prefix_match() -> None:
    # Given
    tokens = ["Hello", " ", "world", "!"]
    prompt = "Hello world"

    # When
    count = AlephAlphaAPIModel._count_prompt_tokens_from_sequence(tokens, prompt)  # type: ignore

    # Then
    assert count == 3


def test_count_prompt_tokens_empty_prompt() -> None:
    # Given
    tokens = ["Hello", " ", "world"]
    prompt = ""

    # When
    count = AlephAlphaAPIModel._count_prompt_tokens_from_sequence(tokens, prompt)  # type: ignore

    # Then
    assert count == 0


def test_count_prompt_tokens_misaligned_tokens() -> None:
    # Given
    tokens = ["He", "llo", " world"]
    prompt = "Hello world"

    # When
    count = AlephAlphaAPIModel._count_prompt_tokens_from_sequence(tokens, prompt)  # type: ignore

    # Then
    assert count == 3


def test_count_prompt_tokens_failure() -> None:
    # Given
    tokens = ["Hello", " universe"]
    prompt = "Hello world"

    # When / Then
    with pytest.raises(ValueError, match="Unable to align completion tokens."):
        AlephAlphaAPIModel._count_prompt_tokens_from_sequence(tokens, prompt)  # type: ignore


def test_extract_choice_logprob_success() -> None:
    # Given
    prompt = "The sky is"
    choice = " blue"
    tokens = ["The", " sky", " is", " blue"]
    log_probs = [
        {"The": -0.1},
        {" sky": -0.2},
        {" is": -0.3},
        {" blue": -0.9},
    ]
    completion_result = MockCompletionResult(completion_tokens=tokens, log_probs=log_probs)
    response = MockCompletionResponse(completions=[completion_result])

    # When
    logprob, choice_tokens, prompt_tokens = AlephAlphaAPIModel._extract_choice_logprob_from_completion(  # type: ignore
        prompt, choice, cast(Any, response)
    )

    # Then
    assert prompt_tokens == 3
    assert choice_tokens == 1
    assert logprob == pytest.approx(-0.9)  # type: ignore


def test_extract_choice_logprob_empty_response() -> None:
    # Given
    prompt = "A"
    choice = "B"
    response = MockCompletionResponse(completions=[])

    # When / Then
    with pytest.raises(ValueError, match="Completion response did not contain any choices"):
        AlephAlphaAPIModel._extract_choice_logprob_from_completion(prompt, choice, cast(Any, response))  # type: ignore


def test_extract_choice_logprob_token_mismatch() -> None:
    # Given
    prompt = "A"
    choice = "B"
    completion_result = MockCompletionResult(completion_tokens=["A", "B"], log_probs=[{"A": -0.1}])
    response = MockCompletionResponse(completions=[completion_result])

    # When / Then
    with pytest.raises(ValueError, match="Mismatch between completion tokens"):
        AlephAlphaAPIModel._extract_choice_logprob_from_completion(prompt, choice, cast(Any, response))  # type: ignore


def test_extract_choice_logprob_text_mismatch() -> None:
    # Given
    prompt = "A"
    choice = "B"
    completion_result = MockCompletionResult(completion_tokens=["A", "C"], log_probs=[{"A": -0.1}, {"C": -0.2}])
    response = MockCompletionResponse(completions=[completion_result])

    # When / Then
    with pytest.raises(ValueError, match="Completion tokens differed"):
        AlephAlphaAPIModel._extract_choice_logprob_from_completion(prompt, choice, cast(Any, response))  # type: ignore


test_extract_choice_logprob_success()
