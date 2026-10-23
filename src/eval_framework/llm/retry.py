"""Repeating an LLM's transport calls while its provider fails transiently."""

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

from openai.types.chat.completion_create_params import ResponseFormat

from eval_framework.contract import Sample
from eval_framework.llm.base import BaseLLM
from eval_framework.shared.types import RawCompletion, RawLoglikelihood
from template_formatting.formatter import Message

logger = logging.getLogger(__name__)

T = TypeVar("T")


class LLMUnavailable(Exception):
    """The model kept failing transiently until the retries ran out, the last failure as ``__cause__``."""


@dataclass(frozen=True)
class RetryPolicy:
    """How often and how patiently one call is repeated before the model is given up on.

    The pause before a retry doubles from ``backoff_s`` and is capped at ``max_backoff_s``.
    """

    # 8, 16, 32, 60 s: about two minutes in all. A throttled OpenRouter key stays throttled for
    # minutes while its rejections return in milliseconds, so a budget of a few seconds burned
    # through every remaining row of a run within a minute of the throttle setting in.
    attempts: int = 5
    backoff_s: float = 8.0
    max_backoff_s: float = 60.0

    def pause_before(self, attempt: int) -> float:
        return min(self.backoff_s * 2 ** (attempt - 2), self.max_backoff_s)


class RetryingLLM(BaseLLM):
    """A model whose generation and logprob calls are repeated while its ``is_retryable`` allows."""

    def __init__(self, inner: BaseLLM, policy: RetryPolicy) -> None:
        self._inner = inner
        self._policy = policy

    @property
    def name(self) -> str:
        return self._inner.name

    def is_retryable(self, exc: Exception) -> bool:
        return self._inner.is_retryable(exc)

    def with_retry(self, policy: RetryPolicy | None = None) -> BaseLLM:
        return self

    def generate_from_messages(
        self,
        messages: list[Sequence[Message]],
        stop_sequences: list[str] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        response_format: ResponseFormat | None = None,
    ) -> list[RawCompletion]:
        return self._repeat(
            lambda: self._inner.generate_from_messages(
                messages, stop_sequences, max_tokens, temperature, top_p, response_format
            )
        )

    def generate_from_samples(
        self,
        samples: list[Sample],
        stop_sequences: list[str] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        response_format: ResponseFormat | None = None,
    ) -> list[RawCompletion]:
        return self._repeat(
            lambda: self._inner.generate_from_samples(
                samples, stop_sequences, max_tokens, temperature, top_p, response_format
            )
        )

    def logprobs(self, samples: list[Sample]) -> list[RawLoglikelihood]:
        return self._repeat(lambda: self._inner.logprobs(samples))

    def post_process_completion(self, completion: str, sample: Sample) -> str:
        return self._inner.post_process_completion(completion, sample)

    def _repeat(self, call: Callable[[], T]) -> T:
        for attempt in range(1, self._policy.attempts + 1):
            try:
                return call()
            except Exception as exc:
                if not self._inner.is_retryable(exc):
                    raise
                if attempt == self._policy.attempts:
                    raise LLMUnavailable(
                        f"{self.name} failed {attempt} times; last failure: {exc.__class__.__name__}: {exc}"
                    ) from exc
                pause_s = self._policy.pause_before(attempt + 1)
                logger.warning(
                    "%s call failed (attempt %d/%d): %s: %s; retrying in %.0f s",
                    self.name,
                    attempt,
                    self._policy.attempts,
                    exc.__class__.__name__,
                    exc,
                    pause_s,
                )
                time.sleep(pause_s)
        raise AssertionError("unreachable: the retry loop returns or raises")
