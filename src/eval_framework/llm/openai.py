import concurrent.futures
import logging
import math
import os
from collections.abc import Callable, Sequence
from typing import Any

import tiktoken
from openai import OpenAI, omit
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.completion_create_params import ResponseFormat
from transformers import PreTrainedTokenizerBase

from eval_framework.contract import Sample
from eval_framework.llm.base import BaseLLM
from eval_framework.shared.types import (
    ConcatCompression,
    RawCompletion,
    RawLoglikelihood,
)
from template_formatting.formatter import BaseFormatter, Message, Role

logger = logging.getLogger(__name__)


def _to_chat_completion_message(message: Message) -> ChatCompletionMessageParam:
    match message.role:
        case Role.SYSTEM:
            return ChatCompletionSystemMessageParam(role="system", content=message.content)
        case Role.USER:
            return ChatCompletionUserMessageParam(role="user", content=message.content)
        case Role.ASSISTANT:
            return ChatCompletionAssistantMessageParam(role="assistant", content=message.content)
        case None:
            raise ValueError(
                "Cannot send a Message without a role through the chat completion API; "
                "the legacy roleless format is only supported for fine-tuning."
            )


def _choice_loglikelihood(prompt_logprobs: Sequence[Any], choice: str) -> tuple[float, int]:
    """Sum the logprobs of the trailing tokens covering `choice`, and count them.

    A token straddling the prompt/choice boundary is included.
    """
    total = 0.0
    num_tokens = 0
    num_chars = 0
    for position in reversed(prompt_logprobs):
        if num_chars >= len(choice) or position is None:
            break
        # A single alternative per position, since the request asks for `prompt_logprobs=0`.
        (token,) = position.values()
        num_chars += len(token["decoded_token"])
        total += token["logprob"]
        num_tokens += 1
    return total, num_tokens


class OpenAIModel(BaseLLM):
    """
    LLM wrapper for OpenAI API providing text/chat completions and log-probability evaluation output.
    """

    LLM_NAME: str | None = None
    DEFAULT_FORMATTER: Callable[[], BaseFormatter] | None = None
    BYTES_PER_TOKEN: float = 4.0  # rule of thumb according to https://platform.openai.com/tokenizer

    def __init__(
        self,
        model_name: str | None = None,
        formatter: BaseFormatter | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        api_key: str | None = None,
        organization: str | None = None,
        base_url: str | None = None,
        bytes_per_token: float | None = None,
    ) -> None:
        """
        Initialize the OpenAIModel.

        Args:
            model_name: OpenAI model name (e.g., "gpt-4o", "gpt-3.5-turbo"). If None, uses LLM_NAME class attribute.
            formatter: Optional message formatter.
            temperature: Sampling temperature used when not passed to generate methods (from 0.0 to 2.0).
            top_p: Nucleus sampling probability mass (from 0.0 to 1.0). If None, the API default is used.
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env variable).
            organization: Optional OpenAI organization ID.
            base_url: Optional API base URL for Azure or alternate endpoints.
            bytes_per_token: Optional custom bytes per token scalar for non-standard models.
        """
        assert model_name is not None or self.LLM_NAME is not None, "A model name must be specified."
        self._model_name = model_name if model_name else self.LLM_NAME
        logger.info(f"Instantiating OpenAIModel with name: {self._model_name}")

        self._formatter = formatter or (self.DEFAULT_FORMATTER() if self.DEFAULT_FORMATTER is not None else None)
        self._temperature = temperature if temperature is not None else 0.0
        assert 0.0 <= self._temperature <= 2.0, "Temperature must be between 0.0 and 2.0"

        if top_p is not None:
            assert 0.0 <= top_p <= 1.0, "top_p must be between 0.0 and 1.0"
        self._top_p = top_p

        self._client = OpenAI(
            api_key=api_key if api_key is not None else os.getenv("OPENAI_API_KEY", ""),
            organization=organization,
            base_url=base_url,
        )

        self._encoder: tiktoken.Encoding | PreTrainedTokenizerBase | None = self._get_encoder_or_none()

        # set bytes_per_token_scalar for non-standard models
        if bytes_per_token is not None and bytes_per_token <= 0:
            raise ValueError("bytes_per_token must be positive")
        self.bytes_per_token_scalar = (
            4.0 / bytes_per_token if bytes_per_token is not None else 4.0 / self.BYTES_PER_TOKEN
        )

    def _get_encoder_or_none(self) -> tiktoken.Encoding | None:
        assert self._model_name is not None
        try:
            return tiktoken.encoding_for_model(self._model_name)
        except KeyError:
            logger.info(
                "tiktoken could not map model_name=%r. Disabling token counting for this model.",
                self._model_name,
            )
            return None
        except Exception as e:
            logger.warning(
                "Failed to initialize tiktoken encoder for model_name=%r (%s). Disabling token counting.",
                self._model_name,
                e.__class__.__name__,
            )
            return None

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens for the given text using the encoder.

        Args:
            text: Input string.

        Returns:
            Number of tokens.
        """
        if self._encoder is None:
            raise RuntimeError("Token counting is not available (no encoder configured).")
        return len(self._encoder.encode(text))

    def generate_from_messages(
        self,
        messages: list[Sequence[Message]],
        stop_sequences: list[str] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        response_format: ResponseFormat | None = None,
    ) -> list[RawCompletion]:
        """
        Generate completions for a list of message sequences concurrently.

        Uses text completion API when a formatter is configured, otherwise uses chat completion API.

        Args:
            messages: Sequence of messages.
            stop_sequences: Optional list of stop sequences.
            max_tokens: Optional maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability mass (0.0 to 1.0). Overrides instance default if provided.
            response_format: Optional OpenAI `response_format` payload for structured output, e.g.
                `{"type": "json_schema", "json_schema": {...}}`. Only used by the chat completion API.

        Returns:
            List of RawCompletion objects containing prompts and completions.
        """

        effective_temperature = temperature if temperature is not None else self._temperature
        assert 0.0 <= effective_temperature <= 2.0, "Temperature must be between 0.0 and 2.0"

        effective_top_p = top_p if top_p is not None else self._top_p
        if effective_top_p is not None:
            assert 0.0 <= effective_top_p <= 1.0, "top_p must be between 0.0 and 1.0"

        def _process_one(single_messages: Sequence[Message]) -> RawCompletion:
            # Adjust max tokens based on bytes_per_token_scalar so that non-standard models generate full responses
            scaled_max_tokens = math.ceil(max_tokens * self.bytes_per_token_scalar) if max_tokens is not None else None

            if self._formatter is not None:
                # Use formatter and text completion API
                prompt = self._formatter.format(single_messages, output_mode="string")
                # documentation: https://platform.openai.com/docs/api-reference/completions/create
                assert self._model_name is not None
                response = self._client.completions.create(
                    model=self._model_name,
                    prompt=prompt,
                    temperature=effective_temperature,
                    top_p=effective_top_p,
                    max_tokens=scaled_max_tokens,
                    stop=stop_sequences,
                )
                completion = response.choices[0].text
                usage = getattr(response, "usage", None)
                prompt_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
                completion_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
                return RawCompletion(
                    prompt=prompt,
                    prompt_num_tokens=(
                        prompt_tokens
                        if prompt_tokens is not None
                        else (self._count_tokens(prompt) if self._encoder is not None else None)
                    ),
                    concat_compression=(
                        ConcatCompression.calculate(
                            single_messages,
                            count_tokens=self._count_tokens,
                            completion=completion,
                        )
                        if self._encoder is not None
                        else None
                    ),
                    completion=completion,
                    completion_num_tokens=(
                        completion_tokens
                        if completion_tokens is not None
                        else (self._count_tokens(completion) if self._encoder is not None else None)
                    ),
                    finish_reason=response.choices[0].finish_reason,
                )

            else:
                # Use chat completion API
                chat_messages = [_to_chat_completion_message(m) for m in single_messages]
                assert self._model_name is not None
                chat_response = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=chat_messages,
                    temperature=effective_temperature,
                    top_p=effective_top_p,
                    max_tokens=scaled_max_tokens,
                    stop=stop_sequences,
                    response_format=response_format if response_format is not None else omit,
                )
                prompt = "\n".join([f"{m.get('role', '')}: {m.get('content', '')}" for m in chat_messages])
                prompt_tokens = getattr(chat_response.usage, "prompt_tokens", None)
                completion_tokens = getattr(chat_response.usage, "completion_tokens", None)
                completion_details = getattr(chat_response.usage, "completion_tokens_details", None)
                reasoning_tokens = getattr(completion_details, "reasoning_tokens", None)
                choice = chat_response.choices[0]
                completion = choice.message.content or ""
                # vLLM's reasoning parser returns the think block as `reasoning`; older
                # versions named the field `reasoning_content`.
                reasoning = getattr(choice.message, "reasoning", None) or getattr(
                    choice.message, "reasoning_content", None
                )
                return RawCompletion(
                    prompt=prompt,
                    prompt_num_tokens=prompt_tokens,
                    concat_compression=(
                        ConcatCompression.calculate(
                            single_messages,
                            count_tokens=self._count_tokens,
                            completion=completion,
                        )
                        if self._encoder is not None
                        else None
                    ),
                    completion=completion,
                    completion_num_tokens=(
                        completion_tokens
                        if completion_tokens is not None
                        else (self._count_tokens(completion) if self._encoder is not None else None)
                    ),
                    reasoning_num_tokens=reasoning_tokens,
                    reasoning=reasoning,
                    finish_reason=choice.finish_reason,
                )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(_process_one, messages))
        return results

    def logprobs(self, samples: list[Sample]) -> list[RawLoglikelihood]:
        """
        Compute total log-probabilities for possible completions given each sample's prompt.

        Note:
            Requires a vLLM server, which returns `prompt_logprobs` on the completions API.
            All prompt and choice pairs are sent in a single batched request.
        """
        assert self._model_name is not None
        assert self._formatter is not None, "Log-probs require a formatter to create text prompts."

        prompts: list[str] = []
        prompt_num_tokens: list[int | None] = []
        sample_loglikelihoods: list[dict[str, float]] = []
        sample_num_tokens: list[dict[str, int]] = []

        batched_full_texts: list[str] = []
        batched_index: list[tuple[int, str]] = []

        for sample_idx, sample in enumerate(samples):
            prompt = self._formatter.format(sample.messages, output_mode="string") if sample.messages else ""
            prompts.append(prompt)
            prompt_num_tokens.append(None)
            sample_loglikelihoods.append({})
            sample_num_tokens.append({})

            for choice in sample.possible_completions or []:
                batched_full_texts.append(prompt + choice)
                batched_index.append((sample_idx, choice))

        if batched_full_texts:
            response = self._client.completions.create(
                model=self._model_name,
                prompt=batched_full_texts,
                # vLLM always returns `prompt_logprobs`; `echo` would only add a second copy.
                echo=False,
                # One token we throw away, because `max_tokens=0` is only allowed with `echo`.
                max_tokens=1,
                temperature=0,
                # One entry per prompt token holding the token actually there plus N alternatives above it.
                # N=0 is all we read. It is a vLLM extension, so it travels in `extra_body`.
                extra_body={"prompt_logprobs": 0},
            )
            assert len(response.choices) == len(batched_full_texts), (
                f"Response size mismatch: expected {len(batched_full_texts)} choices, got {len(response.choices)}."
            )

            for choice_obj, (sample_idx, choice_text) in zip(response.choices, batched_index, strict=True):
                prompt_logprobs = (choice_obj.model_extra or {}).get("prompt_logprobs")
                assert prompt_logprobs is not None, (
                    "`prompt_logprobs` not returned. Please check that the model runs on vLLM."
                )
                loglikelihood, num_choice_tokens = _choice_loglikelihood(prompt_logprobs, choice_text)
                prompt_num_tokens[sample_idx] = len(prompt_logprobs) - num_choice_tokens
                sample_loglikelihoods[sample_idx][choice_text] = loglikelihood
                sample_num_tokens[sample_idx][choice_text] = num_choice_tokens

        return [
            RawLoglikelihood(
                prompt=prompts[idx],
                prompt_num_tokens=prompt_num_tokens[idx],
                loglikelihoods=sample_loglikelihoods[idx],
                loglikelihoods_num_tokens=sample_num_tokens[idx],
            )
            for idx in range(len(samples))
        ]

    def __del__(self) -> None:
        if hasattr(self, "_client"):
            self._client.close()


class OpenAIEmbeddingModel(BaseLLM):
    def __init__(
        self,
        model_name: str = "text-embedding-3-large",
        formatter: BaseFormatter | None = None,
        api_key: str | None = None,
        organization: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """Initialize OpenAI API client.
        Args:
            model_name: Name of the OpenAI model to use (e.g., "text-embedding-3-large")
            formatter: Optional message formatter
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env variable)
            organization: Optional organization ID
            base_url: Optional API base URL for Azure or other endpoints
        """
        if formatter is not None:
            raise ValueError("Formatter is not supported for embedding model.")
        self._model_name = model_name
        logger.info(f"Using {model_name} as embedding model")
        self._client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
            organization=organization,
            base_url=base_url,
        )

    def generate_from_messages(
        self,
        messages: list[Sequence[Message]],
        stop_sequences: list[str] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        response_format: ResponseFormat | None = None,
    ) -> list[RawCompletion]:
        raise NotImplementedError(
            "Embedding model does not support generate_from_messages. Use generate_embeddings instead."
        )

    def generate_embeddings(
        self,
        messages: list[Sequence[Message]],
    ) -> list[list[float]]:
        embeddings = []
        for single_messages in messages:
            prompt = "".join([m.content for m in single_messages])
            response = self._client.embeddings.create(model=self._model_name, input=[prompt])
            embedding = response.data[0].embedding
            embeddings.append(embedding)
        return embeddings

    def logprobs(self, samples: list[Sample]) -> list[RawLoglikelihood]:
        raise NotImplementedError("Embedding model cannot return logprobs.")

    def __del__(self) -> None:
        if hasattr(self, "_client"):
            self._client.close()
            try:
                self._client.close()
            except Exception:
                pass
