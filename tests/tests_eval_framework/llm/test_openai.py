import httpx
import pytest
from openai import AuthenticationError, RateLimitError, omit
from openai.types.chat.completion_create_params import ResponseFormat
from pytest_mock import MockerFixture

from eval_framework.contract import Sample
from eval_framework.llm.openai import (
    EmptyCompletionError,
    OpenAIEmbeddingModel,
    OpenAIModel,
)
from eval_framework.llm.retry import LLMUnavailable, RetryPolicy
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
    mock_client = mocker.MagicMock()
    mocker.patch("eval_framework.llm.openai.OpenAI", return_value=mock_client)
    mock_client.chat.completions.create.return_value = _make_chat_response(
        mocker, content="42", reasoning="let me think", finish_reason="length"
    )
    model = OpenAIModel(model_name="gpt-4o-mini-2024-07-18")

    # When
    [result] = model.generate_from_messages([[Message(role=Role.USER, content="Hello")]])

    # Then the completion stays answer-only and the rest is kept alongside it
    assert result.completion == "42"
    assert result.reasoning == "let me think"
    assert result.finish_reason == "length"


def test_generate_from_messages_raises_with_the_error_body_when_response_has_no_choices(
    mocker: MockerFixture,
) -> None:
    # Given a 200 response carrying an error body instead of choices, as OpenRouter sends
    mock_client = mocker.MagicMock()
    mocker.patch("eval_framework.llm.openai.OpenAI", return_value=mock_client)
    response = mocker.MagicMock(choices=None, model_extra={"error": {"code": 502, "message": "upstream failed"}})
    mock_client.chat.completions.create.return_value = response
    model = OpenAIModel(model_name="gpt-4o-mini-2024-07-18")

    # When / Then
    with pytest.raises(EmptyCompletionError, match="upstream failed"):
        model.generate_from_messages([[Message(role=Role.USER, content="Hello")]])


def test_openai_chat_api_top_p_generate_from_messages(mocker: MockerFixture) -> None:
    mock_client = mocker.MagicMock()
    mocker.patch("eval_framework.llm.openai.OpenAI", return_value=mock_client)
    mock_client.chat.completions.create.return_value = _make_chat_response(mocker)

    model = OpenAIModel(model_name="gpt-4o-mini-2024-07-18", top_p=0.85)
    _MESSAGES = [[Message(role=Role.USER, content="Hello")]]

    model.generate_from_messages(_MESSAGES)
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["top_p"] == 0.85

    model.generate_from_messages(_MESSAGES, top_p=0.75)
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["top_p"] == 0.75


def test_openai_chat_api_forwards_response_format(mocker: MockerFixture) -> None:
    mock_client = mocker.MagicMock()
    mocker.patch("eval_framework.llm.openai.OpenAI", return_value=mock_client)
    mock_client.chat.completions.create.return_value = _make_chat_response(mocker)
    _MESSAGES = [[Message(role=Role.USER, content="Hello")]]

    OpenAIModel(model_name="gpt-4o-mini-2024-07-18").generate_from_messages(_MESSAGES)
    assert mock_client.chat.completions.create.call_args.kwargs["response_format"] is omit

    response_format: ResponseFormat = {
        "type": "json_schema",
        "json_schema": {"name": "verdict", "schema": {"type": "object"}},
    }
    OpenAIModel(model_name="gpt-4o-mini-2024-07-18").generate_from_messages(_MESSAGES, response_format=response_format)
    assert mock_client.chat.completions.create.call_args.kwargs["response_format"] == response_format


def test_generate_from_messages_validates_temperature_and_top_p(mocker: MockerFixture) -> None:
    mocker.patch("eval_framework.llm.openai.OpenAI")
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
    mock_client = mocker.MagicMock()
    mocker.patch("eval_framework.llm.openai.OpenAI", return_value=mock_client)
    mock_client.chat.completions.create.return_value = _make_chat_response(mocker)

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


def _position(token: str, logprob: float) -> dict[int, dict[str, float | str]]:
    return {hash(token): {"logprob": logprob, "decoded_token": token}}


def test_logprobs_sums_choice_tokens_from_vllm_prompt_logprobs(mocker: MockerFixture) -> None:
    # Given a vLLM response for "Q: hi" + " yes" where " yes" is split into two tokens
    mock_client = mocker.MagicMock()
    mocker.patch("eval_framework.llm.openai.OpenAI", return_value=mock_client)
    prompt_logprobs = [None, _position("Q", -9.0), _position(":", -8.0), _position(" ye", -1.0), _position("s", -0.5)]
    mock_client.completions.create.return_value.choices = [
        mocker.MagicMock(model_extra={"prompt_logprobs": prompt_logprobs})
    ]
    model = OpenAIModel(model_name="some-vllm-model", formatter=ConcatFormatter())
    sample = Sample(
        id=0,
        subject="s",
        messages=[Message(role=Role.USER, content="Q:")],
        ground_truth=" yes",
        possible_completions=[" yes"],
    )

    # When
    [result] = model.logprobs([sample])

    # Then only the choice tokens are summed and the rest counts as prompt
    assert result.loglikelihoods == {" yes": -1.5}
    assert result.loglikelihoods_num_tokens == {" yes": 2}
    assert result.prompt_num_tokens == 3
    assert mock_client.completions.create.call_args.kwargs["extra_body"] == {"prompt_logprobs": 0}


def test_logprobs_raises_when_server_returns_no_prompt_logprobs(mocker: MockerFixture) -> None:
    # Given a server that does not return prompt_logprobs
    mock_client = mocker.MagicMock()
    mocker.patch("eval_framework.llm.openai.OpenAI", return_value=mock_client)
    mock_client.completions.create.return_value.choices = [mocker.MagicMock(model_extra={})]
    model = OpenAIModel(model_name="some-model", formatter=ConcatFormatter())
    sample = Sample(
        id=0,
        subject="s",
        messages=[Message(role=Role.USER, content="Q:")],
        ground_truth="a",
        possible_completions=["a"],
    )

    # When / Then
    with pytest.raises(AssertionError, match="prompt_logprobs"):
        model.logprobs([sample])


def test_logprobs_includes_token_straddling_prompt_and_choice(mocker: MockerFixture) -> None:
    # Given a prompt ending in a space, so the tokenizer merges it into the choice token " A"
    mock_client = mocker.MagicMock()
    mocker.patch("eval_framework.llm.openai.OpenAI", return_value=mock_client)
    prompt_logprobs = [None, _position("Answer", -9.0), _position(":", -8.0), _position(" A", -0.25)]
    mock_client.completions.create.return_value.choices = [
        mocker.MagicMock(model_extra={"prompt_logprobs": prompt_logprobs})
    ]
    model = OpenAIModel(model_name="some-vllm-model", formatter=ConcatFormatter())
    sample = Sample(
        id=0,
        subject="s",
        messages=[Message(role=Role.USER, content="Answer: ")],
        ground_truth="A",
        possible_completions=["A"],
    )

    # When
    [result] = model.logprobs([sample])

    # Then the merged token is scored rather than silently dropped
    assert result.loglikelihoods == {"A": -0.25}
    assert result.loglikelihoods_num_tokens == {"A": 1}
    assert result.prompt_num_tokens == 3


def _rate_limit_error() -> RateLimitError:
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    return RateLimitError("throttled", response=httpx.Response(429, request=request), body=None)


def test_retrying_openai_model_recovers_from_a_rate_limit(mocker: MockerFixture) -> None:
    # Given a client that is throttled once before answering
    mock_client = mocker.MagicMock()
    mocker.patch("eval_framework.llm.openai.OpenAI", return_value=mock_client)
    mocker.patch("time.sleep")
    mock_client.chat.completions.create.side_effect = [_rate_limit_error(), _make_chat_response(mocker, content="42")]
    model = OpenAIModel(model_name="gpt-4o-mini-2024-07-18").with_retry()

    # When
    results = model.generate_from_messages([[Message(role=Role.USER, content="hi")]])

    # Then the answer comes from the second request
    assert results[0].completion == "42"
    assert mock_client.chat.completions.create.call_count == 2


def test_retrying_openai_model_recovers_from_a_choiceless_body(mocker: MockerFixture) -> None:
    # Given a 200 body without choices, as OpenRouter returns for a provider error
    mock_client = mocker.MagicMock()
    mocker.patch("eval_framework.llm.openai.OpenAI", return_value=mock_client)
    mocker.patch("time.sleep")
    choiceless = mocker.MagicMock()
    choiceless.choices = None
    mock_client.chat.completions.create.side_effect = [choiceless, _make_chat_response(mocker, content="42")]
    model = OpenAIModel(model_name="gpt-4o-mini-2024-07-18").with_retry()

    # When
    results = model.generate_from_messages([[Message(role=Role.USER, content="hi")]])

    # Then
    assert results[0].completion == "42"
    assert mock_client.chat.completions.create.call_count == 2


def test_retrying_openai_model_gives_up_on_a_throttled_key_after_its_budget(mocker: MockerFixture) -> None:
    # Given a client that is throttled on every attempt
    mock_client = mocker.MagicMock()
    mocker.patch("eval_framework.llm.openai.OpenAI", return_value=mock_client)
    mocker.patch("time.sleep")
    mock_client.chat.completions.create.side_effect = _rate_limit_error()
    model = OpenAIModel(model_name="gpt-4o-mini-2024-07-18").with_retry(RetryPolicy(attempts=2))

    # When / Then the retries are spent before the model is given up on, with the throttle as cause
    with pytest.raises(LLMUnavailable) as excinfo:
        model.generate_from_messages([[Message(role=Role.USER, content="hi")]])
    assert isinstance(excinfo.value.__cause__, RateLimitError)
    assert mock_client.chat.completions.create.call_count == 2


def test_retrying_openai_model_raises_a_rejected_key_at_once(mocker: MockerFixture) -> None:
    # Given a client whose key is rejected
    mock_client = mocker.MagicMock()
    mocker.patch("eval_framework.llm.openai.OpenAI", return_value=mock_client)
    sleep = mocker.patch("time.sleep")
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    mock_client.chat.completions.create.side_effect = AuthenticationError(
        "bad key", response=httpx.Response(401, request=request), body=None
    )
    model = OpenAIModel(model_name="gpt-4o-mini-2024-07-18").with_retry()

    # When / Then the misconfiguration surfaces on the first attempt, since every row would fail the same way
    with pytest.raises(AuthenticationError):
        model.generate_from_messages([[Message(role=Role.USER, content="hi")]])
    assert mock_client.chat.completions.create.call_count == 1
    sleep.assert_not_called()
