import json
import logging
import math
import os
from collections.abc import Callable, Sequence
from typing import Any

import tiktoken  # OpenAI's official tokenizer library
from openai import OpenAI

from eval_framework.llm.base import BaseLLM
from eval_framework.shared.types import ConcatCompression, RawCompletion, RawLoglikelihood
from eval_framework.tasks.base import Sample
from template_formatting.formatter import BaseFormatter, Message, Role

logger = logging.getLogger(__name__)


class OpenAIModel(BaseLLM):
    DEFAULT_FORMATTER: Callable[[], BaseFormatter] | None = None
    BYTES_PER_TOKEN: float = 4.0  # rule of thumb according to https://platform.openai.com/tokenizer

    def __init__(
        self,
        model_name: str = "gpt-4o",
        formatter: BaseFormatter | None = None,
        temperature: float | None = None,
        api_key: str | None = None,
        organization: str | None = None,
        base_url: str | None = None,
        bytes_per_token: int | None = None,
    ) -> None:
        """Initialize OpenAI API client.
        Args:
            model_name: Name of the OpenAI model to use (e.g., "gpt-4", "gpt-3.5-turbo")
            formatter: Optional message formatter
            temperature: Sampling temperature (0.0 to 2.0)
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env variable)
            organization: Optional organization ID
            base_url: Optional API base URL for Azure or other endpoints
        """
        self._model_name = model_name
        logger.info(f"Using {model_name} as a judge")
        self._formatter = formatter or self.DEFAULT_FORMATTER() if self.DEFAULT_FORMATTER is not None else None
        self._temperature = temperature
        # Initialize OpenAI client
        self._client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
            organization=organization,
            base_url=base_url,
        )

        # Initialize tiktoken tokenizer for the model
        self._encoding = tiktoken.encoding_for_model(self._model_name)
        # set bytes_per_token_scalar for non-standard models
        self.bytes_per_token_scalar = (
            4.0 / bytes_per_token if bytes_per_token is not None else 4.0 / self.BYTES_PER_TOKEN
        )

    def _count_tokens(self, text: str) -> int:
        """Helper method to count tokens using tiktoken."""
        return len(self._encoding.encode(text))

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
        """Generate completion from messages.
        Args:
            messages: Sequence of messages
            stop_sequences: Optional list of stop sequences
            max_tokens: Optional maximum number of tokens to generate
        Returns:
            Tuple of (prompt, completion)
        """
        results = []
        for single_messages in messages:
            # Adjust max tokens based on bytes_per_token_scalar so that non-standard models generate full responses
            scaled_max_tokens = math.ceil(max_tokens * self.bytes_per_token_scalar) if max_tokens is not None else None

            if self._formatter is not None:
                # Use formatter for text completion API
                prompt = self._formatter.format(single_messages, output_mode="string")
                response = self._client.completions.create(
                    model=self._model_name,
                    prompt=prompt,
                    temperature=effective_temperature,
                    max_tokens=scaled_max_tokens,
                    stop=stop_sequences,
                )

                prompt_sequence_positions: int | None = self._count_tokens(prompt)
                completion = response.choices[0].text
                completion_sequence_positions = self._count_tokens(completion)

                results.append(
                    RawCompletion(
                        prompt=prompt,
                        prompt_sequence_positions=prompt_sequence_positions,
                        concat_compression=ConcatCompression.calculate(
                            single_messages, count_tokens=self._count_tokens, completion=completion
                        ),
                        completion=completion,
                        completion_sequence_positions=completion_sequence_positions,
                    )
                )
            else:
                # Use chat completion API
                from openai.types.chat import ChatCompletionAssistantMessageParam, ChatCompletionUserMessageParam

                chat_messages = [
                    (
                        ChatCompletionUserMessageParam(role="user", content=m.content)
                        if m.role is not None and m.role.value.lower() == "user"
                        else ChatCompletionAssistantMessageParam(role="assistant", content=m.content)
                    )
                    for m in single_messages
                ]

                chat_response = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=chat_messages,
                    temperature=effective_temperature,
                    max_tokens=scaled_max_tokens,
                    stop=stop_sequences,
                )

                # Reconstruct the prompt (since OpenAI API does not return it)
                prompt = "\n".join([f"{m['role']}: {m['content']}" for m in chat_messages])

                prompt_sequence_positions = (
                    chat_response.usage.prompt_tokens if chat_response.usage else None
                )  # OpenAI API gives token count
                completion = (
                    chat_response.choices[0].message.content if chat_response.choices[0].message.content else ""
                )
                completion_sequence_positions = self._count_tokens(completion)

                results.append(
                    RawCompletion(
                        prompt=prompt,
                        prompt_sequence_positions=prompt_sequence_positions,
                        concat_compression=ConcatCompression.calculate(
                            single_messages, count_tokens=self._count_tokens, completion=completion
                        ),
                        completion=completion,
                        completion_sequence_positions=completion_sequence_positions,
                    )
                )
        return results

    def logprobs(self, samples: list[Sample]) -> list[RawLoglikelihood]:
        """Get log probabilities for possible completions.
        Args:
            samples: list of Sample containing possible completions
        Returns:
            list of Tuple of (prompt, dict of completion->logprob)
        Raises:
            NotImplementedError: Logprobs not yet implemented
        """
        raise NotImplementedError("Logprobs not yet implemented for OpenAI API")

    def generate_structured_output(
        self,
        messages: list[Sequence[Message]],
        stop_sequences: list[str] | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> Any:
        """Generate structured output (e.g. JSON) from messages.
        This implementation ensures the model returns valid JSON.
        Args:
            messages: list of Sequence of messages
            stop_sequences: Optional stop sequences
            max_tokens: Optional max tokens
        Returns:
            Parsed JSON response
        """
        completions = []
        list_json_messages: list[Sequence[Message]] = []
        for single_messages in messages:
            # Add system message to encourage JSON output
            json_messages = list(single_messages)
            if not any(m.role == Role.SYSTEM for m in single_messages):
                json_messages.insert(
                    0,
                    Message(
                        role=Role.SYSTEM, content="You are a helpful assistant that always responds with valid JSON."
                    ),
                )
            list_json_messages.append(json_messages)

        # Adjust max tokens based on bytes_per_token_scalar so that non-standard models generate full responses
        scaled_max_tokens = math.ceil(max_tokens * self.bytes_per_token_scalar) if max_tokens is not None else None

        # Generate completion
        completions = self.generate_from_messages(
            messages=list_json_messages, stop_sequences=stop_sequences, max_tokens=scaled_max_tokens
        )
        responses = []
        for completion in completions:
            try:
                # Parse JSON responses
                responses.append(json.loads(completion.completion))
            except json.JSONDecodeError as e:
                logger.info(f"Warning: Failed to parse JSON response: {e}")
                logger.info(f"Raw response: {completion.completion}")
                raise
        return responses

    def __del__(self) -> None:
        if hasattr(self, "_client"):
            self._client.close()
