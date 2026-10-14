import math
from unittest.mock import MagicMock

import pytest
from openai import omit
from openai.types.chat.completion_create_params import ResponseFormat
from pytest_mock import MockerFixture

from eval_framework.llm.openai import (
    OpenAIEmbeddingModel,
    OpenAIModel,
    _choice_start,
)
from eval_framework.tasks.base import Sample
from eval_framework.utils.helpers import pairwise_cosine_similarity
from template_formatting.formatter import ConcatFormatter, Message, Role

# NOTE: an API key in your .env file is required to run these API tests


@pytest.mark.external_api
def test_openai_embedding_model():
    model = OpenAIEmbeddingModel()
    messages = [
        [Message(role=Role.USER, content="This is a test input for embedding generation.")],
        [Message(role=Role.USER, content="Different message to test embedding generation.")],
    ]

    embeddings = model.generate_embeddings(messages)
    assert len(embeddings) == len(messages)

    cosine_sims = pairwise_cosine_similarity(embeddings, embeddings)
    # assert self-simiarlity is 1
    for i in range(len(embeddings)):
        assert abs(cosine_sims[i][i] - 1.0) < 1e-5
    # assert different embeddings are less similar than self-similarity
    assert cosine_sims[0][1] < 1.0
    assert cosine_sims[1][0] < 1.0


def _mock_async_client(mocker: MockerFixture) -> MagicMock:
    mocker.patch("eval_framework.llm.openai.OpenAI")
    mock_client = mocker.MagicMock()
    mocker.patch("eval_framework.llm.openai.AsyncOpenAI", return_value=mock_client)
    return mock_client


def _make_chat_response(
    mocker: MockerFixture,
    content: str = "test",
    reasoning: str | None = None,
    finish_reason: str = "stop",
) -> object:
    response = mocker.MagicMock()
    response.choices[0].message.content = content
    response.choices[0].message.reasoning = reasoning
    response.choices[0].message.reasoning_content = None
    response.choices[0].finish_reason = finish_reason
    response.usage.prompt_tokens = 5
    return response


def test_generate_from_messages_keeps_reasoning_and_finish_reason(mocker: MockerFixture) -> None:
    # Given a chat response whose think block was split off by the server
    mock_client = _mock_async_client(mocker)
    mock_client.chat.completions.create = mocker.AsyncMock(
        return_value=_make_chat_response(mocker, content="42", reasoning="let me think", finish_reason="length")
    )
    model = OpenAIModel(model_name="gpt-4o-mini-2024-07-18")

    # When
    [result] = model.generate_from_messages([[Message(role=Role.USER, content="Hello")]])

    # Then the completion stays answer-only and the rest is kept alongside it
    assert result.completion == "42"
    assert result.reasoning == "let me think"
    assert result.finish_reason == "length"


def test_openai_chat_api_top_p_generate_from_messages(mocker: MockerFixture) -> None:
    mock_client = _mock_async_client(mocker)
    mock_client.chat.completions.create = mocker.AsyncMock(return_value=_make_chat_response(mocker))

    model = OpenAIModel(model_name="gpt-4o-mini-2024-07-18", top_p=0.85)
    _MESSAGES = [[Message(role=Role.USER, content="Hello")]]

    model.generate_from_messages(_MESSAGES)
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["top_p"] == 0.85

    model.generate_from_messages(_MESSAGES, top_p=0.75)
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["top_p"] == 0.75


def test_openai_chat_api_forwards_response_format(mocker: MockerFixture) -> None:
    mock_client = _mock_async_client(mocker)
    mock_client.chat.completions.create = mocker.AsyncMock(return_value=_make_chat_response(mocker))
    _MESSAGES = [[Message(role=Role.USER, content="Hello")]]

    OpenAIModel(model_name="gpt-4o-mini-2024-07-18").generate_from_messages(_MESSAGES)
    assert mock_client.chat.completions.create.call_args.kwargs["response_format"] is omit

    response_format: ResponseFormat = {
        "type": "json_schema",
        "json_schema": {"name": "verdict", "schema": {"type": "object"}},
    }
    OpenAIModel(model_name="gpt-4o-mini-2024-07-18", response_format=response_format).generate_from_messages(_MESSAGES)
    assert mock_client.chat.completions.create.call_args.kwargs["response_format"] == response_format


def test_generate_from_messages_validates_temperature_and_top_p(mocker: MockerFixture) -> None:
    _mock_async_client(mocker)
    model = OpenAIModel(model_name="gpt-4o-mini-2024-07-18")
    with pytest.raises(AssertionError, match="[Tt]emperature"):
        model.generate_from_messages([], temperature=3.0)
    with pytest.raises(AssertionError, match="[Tt]emperature"):
        model.generate_from_messages([], temperature=-0.5)
    with pytest.raises(AssertionError, match="top_p"):
        model.generate_from_messages([], top_p=1.5)
    with pytest.raises(AssertionError, match="top_p"):
        model.generate_from_messages([], top_p=-1.0)


def test_generate_from_messages_preserves_message_roles(mocker: MockerFixture) -> None:
    mock_client = _mock_async_client(mocker)
    mock_client.chat.completions.create = mocker.AsyncMock(return_value=_make_chat_response(mocker))

    model = OpenAIModel(model_name="gpt-4o-mini-2024-07-18")
    messages = [
        [
            Message(role=Role.SYSTEM, content="system prompt"),
            Message(role=Role.USER, content="user prompt"),
            Message(role=Role.ASSISTANT, content="assistant prompt"),
        ]
    ]

    model.generate_from_messages(messages)

    sent_messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
    assert [m["role"] for m in sent_messages] == ["system", "user", "assistant"]
    assert [m["content"] for m in sent_messages] == ["system prompt", "user prompt", "assistant prompt"]


def test_generate_from_messages_omits_unset_sampling_arguments(mocker: MockerFixture) -> None:
    # Given no top_p, max_tokens or stop sequences
    mock_client = _mock_async_client(mocker)
    mock_client.chat.completions.create = mocker.AsyncMock(return_value=_make_chat_response(mocker))
    model = OpenAIModel(model_name="gpt-4o-mini-2024-07-18")

    # When
    model.generate_from_messages([[Message(role=Role.USER, content="Hello")]], stop_sequences=[])

    # Then the request leaves them to the server's defaults instead of sending nulls or an empty list
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["top_p"] is omit
    assert call_kwargs["max_tokens"] is omit
    assert call_kwargs["stop"] is omit


def _make_completion_choice(mocker: MockerFixture, index: int, text: str, finish_reason: str = "stop") -> MagicMock:
    choice = mocker.MagicMock()
    choice.index = index
    choice.text = text
    choice.finish_reason = finish_reason
    return choice


def test_generate_from_messages_with_formatter_batches_completions(mocker: MockerFixture) -> None:
    # Given a formatter and a server that returns the batch's choices out of order
    mock_client = _mock_async_client(mocker)
    response = mocker.MagicMock()
    response.choices = [
        _make_completion_choice(mocker, index=1, text="second", finish_reason="length"),
        _make_completion_choice(mocker, index=0, text="first"),
    ]
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 4
    mock_client.completions.create = mocker.AsyncMock(return_value=response)
    model = OpenAIModel(model_name="some-vllm-model", formatter=ConcatFormatter())
    messages = [
        [Message(role=Role.USER, content="one")],
        [Message(role=Role.USER, content="two")],
    ]

    # When
    results = model.generate_from_messages(messages, max_tokens=8, stop_sequences=["END"])

    # Then all prompts travel in one request and come back in prompt order
    mock_client.completions.create.assert_called_once()
    call_kwargs = mock_client.completions.create.call_args.kwargs
    assert call_kwargs["prompt"] == [ConcatFormatter().format(m, output_mode="string") for m in messages]
    assert call_kwargs["max_tokens"] == 8
    assert call_kwargs["stop"] == ["END"]
    assert [r.completion for r in results] == ["first", "second"]
    assert [r.finish_reason for r in results] == ["stop", "length"]
    assert [r.prompt_num_tokens for r in results] == [5, 5]
    assert [r.completion_num_tokens for r in results] == [2, 2]


def test_generate_from_messages_with_formatter_skips_request_for_no_messages(mocker: MockerFixture) -> None:
    mock_client = _mock_async_client(mocker)
    mock_client.completions.create = mocker.AsyncMock()
    model = OpenAIModel(model_name="some-vllm-model", formatter=ConcatFormatter())

    assert model.generate_from_messages([]) == []
    mock_client.completions.create.assert_not_called()


def _make_logprobs_choice(
    mocker: MockerFixture, index: int, tokens: list[str], token_logprobs: list[float | None]
) -> MagicMock:
    choice = mocker.MagicMock()
    choice.index = index
    choice.logprobs.tokens = tokens
    choice.logprobs.token_logprobs = token_logprobs
    offsets, pos = [], 0
    for token in tokens:
        offsets.append(pos)
        pos += len(token)
    choice.logprobs.text_offset = offsets
    return choice


def _sample(messages: list[Message], choices: list[str]) -> Sample:
    return Sample(id=0, subject="s", messages=messages, possible_completions=choices, ground_truth=choices[0])


def test_logprobs_requests_echoed_logprobs_for_all_choices_of_a_sample(mocker: MockerFixture) -> None:
    # Given a sample with two choices and no formatter configured
    mock_client = _mock_async_client(mocker)
    response = mocker.MagicMock()
    response.choices = [
        _make_logprobs_choice(mocker, 0, ["Q", ":", " 4"], [None, -0.5, -1.0]),
        _make_logprobs_choice(mocker, 1, ["Q", ":", " 5"], [None, -0.5, -3.0]),
    ]
    mock_client.completions.create = mocker.AsyncMock(return_value=response)
    model = OpenAIModel(model_name="some-vllm-model")
    sample = _sample([Message(role=Role.USER, content="Q:")], [" 4", " 5"])

    # When
    [result] = model.logprobs([sample])

    # Then one echo request carries prompt plus choice for every choice, built with ConcatFormatter
    mock_client.completions.create.assert_called_once()
    call_kwargs = mock_client.completions.create.call_args.kwargs
    assert call_kwargs["prompt"] == ["Q: 4", "Q: 5"]
    assert call_kwargs["echo"] is True
    assert call_kwargs["max_tokens"] == 0
    assert call_kwargs["logprobs"] == 1
    assert result.prompt == "Q:"
    assert result.prompt_num_tokens == 2
    assert result.loglikelihoods == {" 4": -1.0, " 5": -3.0}
    assert result.loglikelihoods_num_tokens == {" 4": 1, " 5": 1}
    assert result.loglikelihoods_per_token[" 4"].byte_lens == [2]
    assert result.loglikelihoods_per_token[" 4"].bits == pytest.approx([1.0 / math.log(2)])
    assert result.raw_loglikelihood_error is None


def test_logprobs_scores_choices_by_text_offset_without_a_local_tokenizer(mocker: MockerFixture) -> None:
    # Given a server whose tokenization merges the prompt's last character with the choice's first
    mock_client = _mock_async_client(mocker)
    response = mocker.MagicMock()
    response.choices = [_make_logprobs_choice(mocker, 0, ["ab", "cd", "e"], [None, -2.0, -4.0])]
    mock_client.completions.create = mocker.AsyncMock(return_value=response)
    model = OpenAIModel(model_name="some-vllm-model", formatter=ConcatFormatter())
    sample = _sample([Message(role=Role.USER, content="abc")], ["de"])

    # When
    [result] = model.logprobs([sample])

    # Then the straddling token is scored as part of the choice
    assert result.loglikelihoods == {"de": -6.0}
    assert result.loglikelihoods_num_tokens == {"de": 2}
    assert result.prompt_num_tokens == 1


def test_logprobs_captures_request_errors_per_sample(mocker: MockerFixture) -> None:
    mock_client = _mock_async_client(mocker)
    mock_client.completions.create = mocker.AsyncMock(side_effect=RuntimeError("boom"))
    model = OpenAIModel(model_name="some-vllm-model")

    [result] = model.logprobs([_sample([Message(role=Role.USER, content="Q:")], [" 4"])])

    assert result.loglikelihoods == {}
    assert result.prompt_num_tokens is None
    assert result.raw_loglikelihood_error is not None
    assert result.raw_loglikelihood_error.error_class == "RuntimeError"


def test_choice_start_handles_boundary_cases() -> None:
    assert _choice_start(["ab", "cd"], [0, 2], prompt_len=2) == 1
    assert _choice_start(["ab", "cd"], [0, 2], prompt_len=1) == 0
    assert _choice_start(["ab", "cd"], [0, 2], prompt_len=4) == 2
    assert _choice_start([], [], prompt_len=0) == 0
