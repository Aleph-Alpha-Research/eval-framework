import asyncio
import json
import logging
import math
import os
import random
import re
import time
import traceback
from collections.abc import Callable, Sequence

import aiohttp
from aleph_alpha_client import (
    AsyncClient,
    BusyError,
    Client,
    CompletionRequest,
    CompletionResponse,
    Prompt,
)
from aleph_alpha_client.prompt import Text
from dotenv import load_dotenv

from eval_framework.llm.base import BaseLLM
from eval_framework.shared.types import Error, PromptTooLongException, RawCompletion, RawLoglikelihood
from eval_framework.tasks.base import Sample
from eval_framework.tasks.utils import raise_errors
from template_formatting.formatter import BaseFormatter, Llama3Formatter, Message

load_dotenv()

logger = logging.getLogger(__name__)


def safe_json_loads(s: str) -> dict[str, str]:
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return {}


class AlephAlphaAPIModel(BaseLLM):
    LLM_NAME: str
    DEFAULT_FORMATTER: Callable[[], BaseFormatter] | None = None
    BYTES_PER_TOKEN: float = 4.0  # rule of thumb according to https://platform.openai.com/tokenizer

    def __init__(
        self,
        formatter: BaseFormatter | None = None,
        checkpoint_name: str | None = None,
        temperature: float | None = None,
        # Please see README.md for tips if adapting the following parameters.
        max_retries: int = 100,
        max_async_concurrent_requests: int = 32,
        request_timeout_seconds: int = 30 * 60 + 5,
        queue_full_timeout_seconds: int = 30 * 60 + 5,
        bytes_per_token: float | None = None,
        token: str = os.getenv("AA_TOKEN", "dummy"),
        base_url: str = os.getenv("AA_INFERENCE_ENDPOINT", "dummy_endpoint"),
    ) -> None:
        self._formatter: BaseFormatter
        if formatter is None:
            if self.DEFAULT_FORMATTER is None:
                raise ValueError("Either formatter or default formatter must be specified")
            self._formatter = self.DEFAULT_FORMATTER()
        else:
            self._formatter = formatter
        self._llm_name = checkpoint_name or self.LLM_NAME
        self._temperature = temperature if temperature is not None else 0.0
        self.max_async_concurrent_requests = max_async_concurrent_requests
        self.max_retries = max_retries
        self.request_timeout_seconds = request_timeout_seconds
        self.queue_full_timeout_seconds = queue_full_timeout_seconds
        self.token = token
        self.base_url = base_url
        self._validate_model_availability(base_url, token)
        # set bytes_per_token_scalar for non-standard models
        if bytes_per_token is not None and bytes_per_token <= 0:
            raise ValueError("bytes_per_token must be positive")
        self.bytes_per_token_scalar = (
            4.0 / bytes_per_token if bytes_per_token is not None else 4.0 / self.BYTES_PER_TOKEN
        )

    def _validate_model_availability(self, base_url: str, token: str) -> None:
        """
        Validate that the model name is available by making a test request.
        """
        try:
            # 'Client' object does not support the context manager protocol
            client = Client(
                host=base_url,
                token=token,
            )

            request = CompletionRequest(
                prompt=Prompt.from_text(""),
                maximum_tokens=1,
            )
            client.complete(request, model=self._llm_name)
            logger.info(f"Model '{self._llm_name}' available and loaded.")
        except Exception as e:
            raise RuntimeError(f"Model '{self._llm_name}' is not available: {e}")

    async def _request_with_backoff(
        self, client: AsyncClient, request: CompletionRequest, id: int
    ) -> CompletionResponse:
        """
        Query Aleph-Alpha API with complete. Retry with back-off until it responds.
        """
        num_attempts = 0
        start_time: float | None = None

        while True:
            try:
                return await client.complete(request, model=self._llm_name)

            except (TimeoutError, BusyError, RuntimeError, aiohttp.ClientError) as e:
                status_code: str = safe_json_loads(e.args[1]).get("code", "") if len(e.args) >= 2 else ""
                str_e = str(e)
                if status_code == "QUEUE_FULL":
                    # Worker not available or missed a heartbeat (inference longer than scheduler's
                    # API_MODEL_AVAILABLE_TIMEOUT_DURATION_MILLIS) or the scheduler is overloaded.
                    if start_time is None:
                        start_time = time.time()
                    elapsed = time.time() - start_time
                    if elapsed <= self.queue_full_timeout_seconds:
                        logger.info(
                            f"Request {id}: {status_code or str_e[:256]} - retrying: attempt"
                            f" {num_attempts}/{self.max_retries}, elapsed {elapsed:.1f} sec"
                        )
                        # don't count as retry (request returns immediately, so just wait a bit not to DoS the server)
                        await asyncio.sleep(random.randint(5, 30))
                        continue

                elif (
                    status_code == "TIMEOUT_TASK"
                    or isinstance(e, TimeoutError)
                    or "502 Bad Gateway" in str_e
                    or "504 Gateway Time-out" in str_e
                    or isinstance(e, aiohttp.ClientError)
                ):
                    # client timeout, either because task too long in a queue or inference too long
                    # (scheduler's API_CLIENT_TIMEOUT_DURATION_MILLIS). Retrying for the "inference too long"
                    # case makes no sense but we unfortunately don't know which case has happened.
                    num_attempts += 1
                    start_time = None
                    if num_attempts < self.max_retries:
                        logger.info(f"Request {id}: TIMEOUT_TASK - retrying: attempt {num_attempts}/{self.max_retries}")
                        await asyncio.sleep(random.randint(5, 30))
                        continue

                raise e

    def _error_from_exception(self, e: Exception) -> Error:
        """Convert an exception to an Error object."""
        if len(e.args) >= 2:
            status_code: str = safe_json_loads(e.args[1]).get("code", "")
            if status_code == "PROMPT_TOO_LONG":
                return Error(
                    error_class=PromptTooLongException.__name__,
                    message="Prompt exceeded context size.",
                    traceback=traceback.format_exc(),
                )
            else:
                return Error(
                    error_class=status_code or e.__class__.__name__, message=str(e), traceback=traceback.format_exc()
                )
        else:
            return Error(error_class=e.__class__.__name__, message=str(e), traceback=traceback.format_exc())

    async def _process_request_with_client(
        self,
        client: AsyncClient,
        request: CompletionRequest,
        id: int,
    ) -> tuple[CompletionRequest, CompletionResponse | Error]:
        """Process a single request, returning the request and either a response or error."""
        try:
            response = await self._request_with_backoff(client=client, request=request, id=id)
            logger.info(f"Request {id}: Success")
            return (request, response)
        except Exception as e:
            if raise_errors():
                raise e
            logger.info(f"Request {id}: Failure: {str(e)[:256]}")
            return (request, self._error_from_exception(e))

    async def _process_requests(
        self,
        requests: list[CompletionRequest],
    ) -> list[tuple[CompletionRequest, CompletionResponse | Error]]:
        """Process multiple requests concurrently, returning request/response pairs."""
        async with AsyncClient(
            host=self.base_url,
            nice=True,
            request_timeout_seconds=self.request_timeout_seconds,
            token=self.token,
            total_retries=0,  # we have a custom retry policy in _request_with_backoff()
            limit=self.max_async_concurrent_requests,
        ) as client:
            tasks = (
                self._process_request_with_client(
                    client,
                    request,
                    i,
                )
                for i, request in enumerate(requests)
            )
            responses = await asyncio.gather(*tasks)  # guarantees order of responses
        return list(responses)

    def _response_to_raw_completion(
        self, request: CompletionRequest, response: CompletionResponse | Error
    ) -> RawCompletion:
        """Convert a request/response pair to a RawCompletion."""
        assert isinstance(request.prompt.items[0], Text)
        prompt = request.prompt.items[0].text

        if isinstance(response, Error):
            return RawCompletion(
                prompt=prompt,
                prompt_sequence_positions=None,
                completion="",
                completion_sequence_positions=0,
                raw_completion_error=response,
            )

        assert len(response.completions) == 1
        completion = response.completions[0].completion or ""
        prompt_sequence_positions: int | None = None
        completion_sequence_positions: int | None = None

        # Support workaround in api-worker-transformer's scaling generator to return the correct number of tokens.
        # These are part of the completion string; those in CompletionResponse are invalid in this case.
        m = re.match(r"\uf8c9(\d+),(\d+)\uf8c9(.*)", completion, re.DOTALL)
        if m is not None:
            num_input_tokens, num_completion_tokens, completion = m.groups()
            prompt_sequence_positions = int(num_input_tokens)
            completion_sequence_positions = int(num_completion_tokens)
        else:
            prompt_sequence_positions = response.num_tokens_prompt_total if response else None
            completion_sequence_positions = response.num_tokens_generated if response else None

        return RawCompletion(
            prompt=prompt,
            prompt_sequence_positions=prompt_sequence_positions,
            completion=completion,
            completion_sequence_positions=completion_sequence_positions,
        )

    def generate_from_messages(
        self,
        messages: list[Sequence[Message]],
        stop_sequences: list[str] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> list[RawCompletion]:
        effective_temperature = temperature if temperature is not None else self._temperature

        requests: list[CompletionRequest] = []

        # Adjust max tokens based on bytes_per_token_scalar so that non-standard models generate full responses
        scaled_max_tokens = math.ceil(max_tokens * self.bytes_per_token_scalar) if max_tokens is not None else None

        for single_messages in messages:
            requests.append(
                CompletionRequest(
                    prompt=Prompt.from_text(self._formatter.format(single_messages, output_mode="string")),
                    maximum_tokens=scaled_max_tokens,
                    stop_sequences=stop_sequences,
                    temperature=effective_temperature,
                )
            )

        responses = asyncio.run(self._process_requests(requests))
        return [self._response_to_raw_completion(req, resp) for req, resp in responses]

    def logprobs(self, samples: list[Sample]) -> list[RawLoglikelihood]:
        prompts: list[str] = []
        completion_requests: list[CompletionRequest] = []

        for sample in samples:
            prompt: str = self._formatter.format(sample.messages, output_mode="string") if sample.messages else ""
            prompts.append(prompt)
            for choice in sample.possible_completions or []:
                completion_requests.append(
                    CompletionRequest(
                        prompt=Prompt.from_text(prompt + choice),
                        maximum_tokens=0,
                        temperature=0.0,
                        log_probs=0,
                        echo=True,
                        tokens=True,
                    )
                )

        completion_responses: list[tuple[CompletionRequest, CompletionResponse | Error]] = []
        if completion_requests:
            completion_responses = asyncio.run(self._process_requests(completion_requests))
        completion_iter = iter(completion_responses)

        results: list[RawLoglikelihood] = []
        for sample_idx, (sample, prompt) in enumerate(zip(samples, prompts, strict=True)):
            choices_log_probs: dict[str, float] = {}
            choices_sequence_positions: dict[str, int] = {}
            prompt_sequence_positions: int | None = 0
            number_of_initial_choices_tokens: int | None = None
            error: Error | None = None

            for choice in sample.possible_completions or []:
                request, response = next(completion_iter)
                assert isinstance(request, CompletionRequest)
                if error is not None:
                    continue

                if isinstance(response, Error):
                    error = response
                    prompt_sequence_positions = None
                    choices_log_probs = {}
                    choices_sequence_positions = {}
                else:
                    try:
                        logprob, choice_token_count = self._extract_choice_logprob_from_completion(
                            prompt=prompt,
                            choice=choice,
                            response=response,
                        )
                        choices_log_probs[choice] = logprob
                        choices_sequence_positions[choice] = choice_token_count
                        if number_of_initial_choices_tokens is None:
                            number_of_initial_choices_tokens = choice_token_count

                        self._check_choices_token_count(
                            sample_idx, choice_token_count, number_of_initial_choices_tokens
                        )

                    except Exception as exc:
                        if raise_errors():
                            raise
                        error = Error(
                            error_class=exc.__class__.__name__,
                            message=str(exc),
                            traceback=traceback.format_exc(),
                        )
                        prompt_sequence_positions = None
                        choices_log_probs = {}
                        choices_sequence_positions = {}

            results.append(
                RawLoglikelihood(
                    prompt=prompt,
                    prompt_sequence_positions=prompt_sequence_positions,
                    loglikelihoods=choices_log_probs,
                    loglikelihoods_sequence_positions=choices_sequence_positions,
                    raw_loglikelihood_error=error,
                )
            )

        return results

    @staticmethod
    def _check_choices_token_count(
        sample_idx: int, choice_token_count: int, number_of_initial_choices_tokens: int | None
    ) -> None:
        if number_of_initial_choices_tokens is not None:
            if choice_token_count != number_of_initial_choices_tokens:
                logger.warning(
                    "Choice token count differed between choices for sample %s (%s vs %s). Using latest value.",
                    sample_idx,
                    choice_token_count,
                    number_of_initial_choices_tokens,
                )

    @staticmethod
    def _extract_choice_logprob_from_completion(
        prompt: str, choice: str, response: CompletionResponse
    ) -> tuple[float, int]:
        if not response.completions:
            raise ValueError("Completion response did not contain any choices.")
        completion_result = response.completions[0]
        if completion_result.log_probs is None:
            raise ValueError("Completion result did not include log_probs.")
        if completion_result.completion_tokens is None:
            raise ValueError("Completion result did not include completion_tokens.")

        tokens = list(completion_result.completion_tokens)
        log_prob_entries = list(completion_result.log_probs)

        if len(tokens) != len(log_prob_entries):
            raise ValueError("Mismatch between completion tokens and log_prob entries.")

        combined_text = "".join(tokens)
        expected_text = prompt + choice
        if combined_text != expected_text:
            raise ValueError("Completion tokens differed from prompt + choice text.")

        prompt_token_count = AlephAlphaAPIModel._count_prompt_tokens_from_sequence(tokens, prompt)
        choice_token_count = len(tokens) - prompt_token_count
        if choice_token_count < 0:
            raise ValueError("Choice token count computed as negative.")

        total_logprob = 0.0
        for entry in log_prob_entries[prompt_token_count:]:
            assert isinstance(entry, dict)
            if len(entry) != 1:
                raise ValueError("Log_probs entry did not contain exactly one key-value pair.")
            _, value = entry.popitem()
            assert isinstance(value, float)
            total_logprob += value

        return total_logprob, choice_token_count

    @staticmethod
    def _count_prompt_tokens_from_sequence(tokens: Sequence[str], prompt: str) -> int:
        if not prompt:
            return 0
        current_text = ""
        for idx, token in enumerate(tokens):
            current_text += token
            if current_text == prompt:
                return idx + 1
            if len(current_text) > len(prompt):
                break
        raise ValueError("Unable to align completion tokens with prompt text.")


class Llama31_8B_Instruct_API(AlephAlphaAPIModel):
    LLM_NAME = "llama-3.1-8b-instruct"
    DEFAULT_FORMATTER = Llama3Formatter
