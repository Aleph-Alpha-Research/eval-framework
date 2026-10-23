import time
from collections.abc import Sequence

import pytest
from openai.types.chat.completion_create_params import ResponseFormat

from eval_framework.contract import Sample
from eval_framework.llm.base import BaseLLM
from eval_framework.llm.retry import LLMUnavailable, RetryingLLM, RetryPolicy
from eval_framework.shared.types import RawCompletion, RawLoglikelihood
from template_formatting.formatter import Message, Role

MESSAGES: list[Sequence[Message]] = [[Message(role=Role.USER, content="judge this")]]
SAMPLE = Sample(id=0, subject="s", messages=list(MESSAGES[0]), ground_truth=None, possible_completions=["a"])


def _answer(completion: str = "an answer") -> list[RawCompletion]:
    return [RawCompletion(prompt="prompt", completion=completion, prompt_num_tokens=None, completion_num_tokens=None)]


class ScriptedLLM(BaseLLM):
    """Answers or raises each queued outcome in turn, from both generation and logprob calls."""

    def __init__(self, *outcomes: Exception | list[RawCompletion] | list[RawLoglikelihood]) -> None:
        self._outcomes = list(outcomes)
        self.calls = 0

    @property
    def name(self) -> str:
        return "scripted"

    def _next(self):  # type: ignore[no-untyped-def]
        self.calls += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def generate_from_messages(
        self,
        messages: list[Sequence[Message]],
        stop_sequences: list[str] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        response_format: ResponseFormat | None = None,
    ) -> list[RawCompletion]:
        return self._next()

    def logprobs(self, samples: list[Sample]) -> list[RawLoglikelihood]:
        return self._next()


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    return sleeps


def test_with_retry_wraps_the_model_once() -> None:
    # Given
    llm = ScriptedLLM()

    # When
    retrying = llm.with_retry()

    # Then the wrapper keeps the model's name and wrapping it again is a no-op
    assert isinstance(retrying, RetryingLLM)
    assert retrying.name == "scripted"
    assert retrying.with_retry() is retrying


def test_retry_returns_the_first_successful_answer() -> None:
    # Given a model that answers straight away
    llm = ScriptedLLM(_answer())

    # When
    raw_completion = llm.with_retry().generate_from_messages(MESSAGES)[0]

    # Then
    assert raw_completion.completion == "an answer"
    assert llm.calls == 1


def test_retry_repeats_any_failure_by_default() -> None:
    # Given the first call fails in some way the model does not rule out
    llm = ScriptedLLM(IndexError("list index out of range"), _answer())

    # When
    raw_completion = llm.with_retry().generate_from_messages(MESSAGES)[0]

    # Then the answer comes from the second call
    assert raw_completion.completion == "an answer"
    assert llm.calls == 2


def test_retry_covers_logprobs() -> None:
    # Given a model whose first logprob call fails
    loglikelihoods = [
        RawLoglikelihood(prompt="p", prompt_num_tokens=1, loglikelihoods={"a": 0.0}, loglikelihoods_num_tokens={"a": 1})
    ]
    llm = ScriptedLLM(ConnectionResetError("reset"), loglikelihoods)

    # When
    result = llm.with_retry().logprobs([SAMPLE])

    # Then
    assert result == loglikelihoods
    assert llm.calls == 2


def test_retry_gives_up_as_unavailable_after_the_configured_attempts() -> None:
    # Given a model that never answers
    llm = ScriptedLLM(*[IndexError("list index out of range")] * 3)

    # When / Then the call is given up after exactly the configured attempts, the last failure as cause
    with pytest.raises(LLMUnavailable, match="scripted failed 3 times; last failure: IndexError") as excinfo:
        llm.with_retry(RetryPolicy(attempts=3)).generate_from_messages(MESSAGES)
    assert llm.calls == 3
    assert isinstance(excinfo.value.__cause__, IndexError)


def test_retry_does_not_repeat_what_the_model_rules_out() -> None:
    # Given a model that rules out retrying a rejected key, since every row would fail the same way
    class PickyLLM(ScriptedLLM):
        def is_retryable(self, exc: Exception) -> bool:
            return not isinstance(exc, PermissionError)

    llm = PickyLLM(PermissionError("bad key"))

    # When / Then the error surfaces as is after a single call, through the wrapper's own hook as well
    with pytest.raises(PermissionError):
        llm.with_retry().generate_from_messages(MESSAGES)
    assert llm.calls == 1
    assert llm.with_retry().is_retryable(PermissionError()) is False


def test_retry_keeps_the_fallback_from_samples_to_messages() -> None:
    # Given a model without generate_from_samples, as most are
    llm = ScriptedLLM(_answer())

    # When the generic entry point is used through the wrapper
    raw_completion = llm.with_retry().generate([SAMPLE])[0]

    # Then NotImplementedError still falls through to generate_from_messages instead of being retried
    assert raw_completion.completion == "an answer"
    assert llm.calls == 1


def test_retry_backs_off_exponentially_up_to_the_cap(no_sleep: list[float]) -> None:
    # Given a model that fails four times before answering
    llm = ScriptedLLM(*[TypeError("x")] * 4, _answer())

    # When
    llm.with_retry(RetryPolicy(attempts=5, backoff_s=8.0, max_backoff_s=60.0)).generate_from_messages(MESSAGES)

    # Then the pause doubles from the base, is capped, and none follows the success
    assert no_sleep == [8.0, 16.0, 32.0, 60.0]


def test_default_budget_outlasts_a_minutes_long_throttle() -> None:
    # A budget of a few seconds was burned through by a throttled key in seconds per row.
    policy = RetryPolicy()
    pauses = [policy.pause_before(attempt) for attempt in range(2, policy.attempts + 1)]
    assert policy.attempts >= 5
    assert sum(pauses) >= 90.0
    assert max(pauses) == policy.max_backoff_s == 60.0
