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
    EvaluationRequest,
    EvaluationResponse,
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


def safe_json_loads(s: str) -> dict:
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
        # Please see README.md for tips if adapting the following parameters.
        max_retries: int = 100,
        max_async_concurrent_requests: int = 32,
        request_timeout_seconds: int = 30 * 60 + 5,
        queue_full_timeout_seconds: int = 30 * 60 + 5,
        bytes_per_token: int | None = None,
    ) -> None:
        self._formatter: BaseFormatter
        if formatter is None:
            if self.DEFAULT_FORMATTER is None:
                raise ValueError("Either formatter or default formatter must be specified")
            self._formatter = self.DEFAULT_FORMATTER()
        else:
            self._formatter = formatter
        self._llm_name = checkpoint_name or self.LLM_NAME
        self.max_async_concurrent_requests = max_async_concurrent_requests
        self.max_retries = max_retries
        self.request_timeout_seconds = request_timeout_seconds
        self.queue_full_timeout_seconds = queue_full_timeout_seconds
        self._validate_model_availability()
        # set bytes_per_token_scalar for non-standard models
        self.bytes_per_token_scalar = (
            4.0 / bytes_per_token if bytes_per_token is not None else 4.0 / self.BYTES_PER_TOKEN
        )

    def _validate_model_availability(self) -> None:
        """
        Validate that the model name is available by making a test request.
        """
        try:
            # 'Client' object does not support the context manager protocol
            client = Client(
                host=os.getenv("AA_INFERENCE_ENDPOINT", "dummy_endpoint"),
                token=os.getenv("AA_TOKEN", "dummy"),
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
        self, client: AsyncClient, request: CompletionRequest | EvaluationRequest, id: int
    ) -> CompletionResponse | EvaluationResponse:
        """
        Query Aleph-Alpha API with complete. Retry with back-off until it responds.
        """
        num_attempts = 0
        start_time: float | None = None

        while True:
            try:
                if isinstance(request, CompletionRequest):
                    return await client.complete(request, model=self._llm_name)
                elif isinstance(request, EvaluationRequest):
                    return await client.evaluate(request, model=self._llm_name)
                else:
                    raise ValueError(f"Unsupported request type: {type(request)}")

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

    async def _process_request_with_client(
        self,
        client: AsyncClient,
        semaphore: asyncio.Semaphore,
        request: CompletionRequest | EvaluationRequest,
        id: int,
    ) -> RawCompletion | tuple[EvaluationRequest, EvaluationResponse | Error]:
        async with semaphore:
            try:
                response = await self._request_with_backoff(client=client, request=request, id=id)
                logger.info(f"Request {id}: Success")
            except Exception as e:
                if raise_errors():
                    raise e
                logger.info(f"Request {id}: Failure: {str(e)[:256]}")
                if len(e.args) >= 2:
                    status_code: str = safe_json_loads(e.args[1]).get("code", "")
                    if status_code == "PROMPT_TOO_LONG":
                        error = Error(
                            error_class=PromptTooLongException.__name__,
                            message="Prompt exceeded context size.",
                            traceback=traceback.format_exc(),
                        )
                    else:
                        error = Error(error_class=status_code, message=str(e), traceback=traceback.format_exc())
                else:
                    error = Error(error_class=e.__class__.__name__, message=str(e), traceback=traceback.format_exc())

                if isinstance(request, CompletionRequest):
                    assert isinstance(request.prompt.items[0], Text)
                    return RawCompletion(
                        prompt=request.prompt.items[0].text,
                        prompt_sequence_positions=None,
                        completion="",
                        completion_sequence_positions=0,
                        raw_completion_error=error,
                    )
                else:
                    return (request, error)

        # Completion responses can directly be converted to RawCompletion
        if isinstance(request, CompletionRequest):
            assert isinstance(request.prompt.items[0], Text) and isinstance(response, CompletionResponse)
            assert len(response.completions) == 1
            prompt = request.prompt.items[0].text
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

        # Evaluation responses must be assembled from individual choice requests later
        else:
            assert isinstance(response, EvaluationResponse)
            return (request, response)

    async def _process_requests(
        self, requests: list[CompletionRequest] | list[EvaluationRequest]
    ) -> list[RawCompletion | tuple[EvaluationRequest, EvaluationResponse | Error]]:
        semaphore = asyncio.Semaphore(self.max_async_concurrent_requests)
        async with AsyncClient(
            host=os.getenv("AA_INFERENCE_ENDPOINT", "dummy_endpoint"),
            nice=True,
            request_timeout_seconds=self.request_timeout_seconds,
            token=os.getenv("AA_TOKEN", "dummy"),
            total_retries=0,  # we have a custom retry policy in _request_with_backoff()
        ) as client:
            tasks = (
                self._process_request_with_client(client, semaphore, request, i)
                for i, request in enumerate[CompletionRequest | EvaluationRequest](requests)
            )
            responses = await asyncio.gather(*tasks)  # guarantees order of responses
        return responses

    def generate_from_messages(
        self,
        messages: list[Sequence[Message]],
        stop_sequences: list[str] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> list[RawCompletion]:
        if temperature is None:
            effective_temperature = 0.0  # Current default, TODO: refactor to use model's default
            logger.info(
                f"Using default temperature value: {effective_temperature} as no custom temperature value was provided"
            )
        else:
            effective_temperature = temperature

        requests = []

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
        return responses  # type: ignore

    def logprobs(self, samples: list[Sample]) -> list[RawLoglikelihood]:
        samples_prompt: list[str] = []
        evaluation_requests: list[EvaluationRequest] = []
        results: list[RawLoglikelihood] = []

        # evaluate all choices independently in flattened list
        for sample in samples:
            prompt: str = self._formatter.format(sample.messages, output_mode="string") if sample.messages else ""
            samples_prompt.append(prompt)
            for choice in sample.possible_completions or []:
                evaluation_requests.append(
                    EvaluationRequest(prompt=Prompt.from_text(prompt), completion_expected=choice)
                )

        evaluation_responses = asyncio.run(self._process_requests(evaluation_requests))
        evaluation_responses_iter = iter(evaluation_responses)

        # assemble choices to RawLoglikelihood from a flattened list for all possible choice replies
        for sample, prompt in zip(samples, samples_prompt, strict=True):
            choices_log_probs: dict[str, float] = {}
            choices_sequence_positions: dict[str, int] = {}
            prompt_sequence_positions: int | None = 0
            error = None

            for choice in sample.possible_completions or []:
                request, response = next(evaluation_responses_iter)
                if error is not None:
                    continue
                if isinstance(response, Error):  # failure for one choice leads to failure of the whole sample
                    error = response
                    prompt_sequence_positions = None
                    choices_log_probs = {}
                    choices_sequence_positions = {}
                else:
                    assert isinstance(request, EvaluationRequest) and isinstance(response, EvaluationResponse)
                    assert isinstance(request.prompt.items[0], Text)
                    assert prompt == request.prompt.items[0].text, f"{prompt} != {request.prompt.items[0].text}"
                    assert choice == request.completion_expected, f"{choice} != {request.completion_expected}"
                    prompt_sequence_positions = response.num_tokens_prompt_total - response.result["token_count"]
                    choices_log_probs[choice] = response.result["log_probability"]
                    choices_sequence_positions[choice] = response.result["token_count"]

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


class Llama31_8B_Instruct_API(AlephAlphaAPIModel):
    LLM_NAME = "llama-3.1-8b-instruct"
    DEFAULT_FORMATTER = Llama3Formatter
