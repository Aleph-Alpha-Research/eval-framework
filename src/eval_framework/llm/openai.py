import concurrent.futures
import logging
import os
import traceback
from collections.abc import Callable, Sequence
from functools import partial

import tiktoken
from openai import OpenAI
from openai.types.chat import ChatCompletionAssistantMessageParam, ChatCompletionUserMessageParam
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from eval_framework.llm.base import BaseLLM
from eval_framework.shared.types import ConcatCompression, Error, RawCompletion, RawLoglikelihood
from eval_framework.tasks.base import Sample
from template_formatting.formatter import BaseFormatter, ConcatFormatter, HFFormatter, Message

logger = logging.getLogger(__name__)


class OpenAIModel(BaseLLM):
    """
    LLM wrapper for OpenAI API providing text/chat completions and log-probability evaluation output.
    """

    DEFAULT_FORMATTER: Callable[[], BaseFormatter] | None = None
    API_KEY = os.getenv("OPENAI_API_KEY", "")

    def __init__(
        self,
        model_name: str = "gpt-4o",
        formatter: BaseFormatter | None = None,
        temperature: float | None = None,
        api_key: str | None = None,
        organization: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """
        Initialize the OpenAIModel.

        Args:
            model_name: OpenAI model name (e.g., "gpt-4o", "gpt-3.5-turbo").
            formatter: Optional message formatter.
            temperature: Sampling temperature used when not passed to generate methods (from 0.0 to 2.0).
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env variable).
            organization: Optional OpenAI organization ID.
            base_url: Optional API base URL for Azure or alternate endpoints.
        """
        self._model_name = model_name
        logger.info(f"Instantiating OpenAIModel with name: {self._model_name}")

        self._formatter = formatter or (self.DEFAULT_FORMATTER() if self.DEFAULT_FORMATTER is not None else None)
        self._temperature = temperature if temperature is not None else 0.0
        assert 0.0 <= self._temperature <= 2.0, "Temperature must be between 0.0 and 2.0"

        self._client = OpenAI(
            api_key=api_key or self.API_KEY,
            organization=organization,
            base_url=base_url,
        )

        # Initialize tokenizer for the model
        self._encoder = self._get_encoder()

    def _get_encoder(self) -> tiktoken.Encoding:
        return tiktoken.encoding_for_model(self._model_name)

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens for the given text using the encoder.

        Args:
            text: Input string.

        Returns:
            Number of tokens.
        """
        return len(self._encoder.encode(text))

    def generate_from_messages(
        self,
        messages: list[Sequence[Message]],
        stop_sequences: list[str] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> list[RawCompletion]:
        """
        Generate completions for a list of message sequences concurrently.

        Uses text completion API when a formatter is configured, otherwise uses chat completion API.

        Args:
            messages: Sequence of messages.
            stop_sequences: Optional list of stop sequences.
            max_tokens: Optional maximum number of tokens to generate.
            temperature: Sampling temperature.

        Returns:
            List of RawCompletion objects containing prompts and completions.
        """
        effective_temperature = temperature if temperature is not None else self._temperature
        assert 0.0 <= effective_temperature <= 2.0, "Temperature must be between 0.0 and 2.0"

        def _process_one(single_messages):
            if self._formatter is not None:
                # Use formatter and text completion API
                prompt = self._formatter.format(single_messages, output_mode="string")
                # documentation: https://platform.openai.com/docs/api-reference/completions/create
                response = self._client.completions.create(
                    model=self._model_name,
                    prompt=prompt,
                    temperature=effective_temperature,
                    max_tokens=max_tokens,
                    stop=stop_sequences,
                )
                completion = response.choices[0].text
                return RawCompletion(
                    prompt=prompt,
                    prompt_sequence_positions=self._count_tokens(prompt),
                    concat_compression=ConcatCompression.calculate(
                        single_messages, count_tokens=self._count_tokens, completion=completion
                    ),
                    completion=completion,
                    completion_sequence_positions=self._count_tokens(completion),
                )
            else:
                # Use chat completion API
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
                    max_tokens=max_tokens,
                    stop=stop_sequences,
                )
                prompt = "\n".join([f"{m.get('role', '')}: {m.get('content', '')}" for m in chat_messages])
                prompt_tokens = getattr(chat_response.usage, "prompt_tokens", None)
                completion = chat_response.choices[0].message.content or ""
                return RawCompletion(
                    prompt=prompt,
                    prompt_sequence_positions=prompt_tokens,
                    concat_compression=ConcatCompression.calculate(
                        single_messages, count_tokens=self._count_tokens, completion=completion
                    ),
                    completion=completion,
                    completion_sequence_positions=self._count_tokens(completion),
                )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(_process_one, messages))
        return results

    # def generate_from_messages_sync(
    #     self,
    #     messages: list[Sequence[Message]],
    #     stop_sequences: list[str] | None = None,
    #     max_tokens: int | None = None,
    #     temperature: float | None = None,
    # ) -> list[RawCompletion]:
    #     """
    #     Generate completions for a list of message sequences.

    #     Uses text completion API when a formatter is configured, otherwise uses chat completion API.

    #     Args:
    #         messages: Sequence of messages.
    #         stop_sequences: Optional list of stop sequences.
    #         max_tokens: Optional maximum number of tokens to generate.
    #         temperature: Sampling temperature.

    #     Returns:
    #         List of RawCompletion objects containing prompts and completions.
    #     """
    #     effective_temperature = temperature if temperature is not None else self._temperature
    #     assert 0.0 <= effective_temperature <= 2.0, "Temperature must be between 0.0 and 2.0"

    #     results: list[RawCompletion] = []

    #     for single_messages in messages:
    #         if self._formatter is not None:
    #             # Use formatter and text completion API
    #             prompt = self._formatter.format(single_messages, output_mode="string")
    #             # documentation: https://platform.openai.com/docs/api-reference/completions/create
    #             response = self._client.completions.create(
    #                 model=self._model_name,
    #                 prompt=prompt,
    #                 temperature=effective_temperature,
    #                 max_tokens=max_tokens,
    #                 stop=stop_sequences,
    #             )
    #             completion = response.choices[0].text
    #             results.append(
    #                 RawCompletion(
    #                     prompt=prompt,
    #                     prompt_sequence_positions=self._count_tokens(prompt),
    #                     concat_compression=ConcatCompression.calculate(
    #                         single_messages, count_tokens=self._count_tokens, completion=completion
    #                     ),
    #                     completion=completion,
    #                     completion_sequence_positions=self._count_tokens(completion),
    #                 )
    #             )
    #         else:
    #             # Take one chat at a time
    #             chat_messages = [
    #                 (
    #                     ChatCompletionUserMessageParam(role="user", content=m.content)
    #                     if m.role is not None and m.role.value.lower() == "user"
    #                     else ChatCompletionAssistantMessageParam(role="assistant", content=m.content)
    #                 )
    #                 for m in single_messages
    #             ]
    #             chat_response = self._client.chat.completions.create(
    #                 model=self._model_name,
    #                 messages=chat_messages,
    #                 temperature=effective_temperature,
    #                 max_tokens=max_tokens,
    #                 stop=stop_sequences,
    #             )
    #             prompt = "\n".join([f"{m.get('role', '')}: {m.get('content', '')}" for m in chat_messages])
    #             prompt_tokens = getattr(chat_response.usage, "prompt_tokens", None)
    #             completion = chat_response.choices[0].message.content or ""
    #             results.append(
    #                 RawCompletion(
    #                     prompt=prompt,
    #                     prompt_sequence_positions=prompt_tokens,
    #                     concat_compression=ConcatCompression.calculate(
    #                         single_messages, count_tokens=self._count_tokens, completion=completion
    #                     ),
    #                     completion=completion,
    #                     completion_sequence_positions=self._count_tokens(completion),
    #                 )
    #             )
    #     return results

    def logprobs(self, samples: list[Sample]) -> list[RawLoglikelihood]:
        """
        Compute total log-probabilities for possible completions given each sample's prompt.

        Args:
            samples: List of Sample objects, each with prompt messages and possible completions.

        Returns:
            List of RawLoglikelihood objects mapping each prompt and completion to its log-probability.

        Note:
            Uses the OpenAI completions API with echo=True; chat logprobs are not supported.
        """
        assert self._model_name in ["babbage-002", "davinci-002"], (
            "Log-probs for prompt tokens are not supported for a limited set of models."
        )
        # apparently OpenAI stopped providing logprobs of prompt tokens, see discussion in:
        # https://github.com/EleutherAI/lm-evaluation-harness/issues/1196

        assert self._formatter is not None, "Log-probs require a formatter to create text prompts."
        results: list[RawLoglikelihood] = []
        for sample in samples:
            prompt = self._formatter.format(sample.messages, output_mode="string") if sample.messages else ""
            choices_log_probs: dict[str, float] = {}
            choices_sequence_positions: dict[str, int] = {}
            prompt_sequence_positions: int | None = self._count_tokens(prompt)
            error: Error | None = None

            for choice in sample.possible_completions or []:
                if error is not None:
                    continue

                # Tokenize prompt and completion
                prompt_tokens = self._encoder.encode(prompt)
                completion_tokens = self._encoder.encode(choice)
                full_text = prompt + choice

                try:
                    response = self._client.completions.create(
                        model=self._model_name,
                        prompt=full_text,
                        echo=True,
                        max_tokens=0,
                        logprobs=1,
                        temperature=0,
                    )

                    choice_obj = response.choices[0]
                    if not hasattr(choice_obj, "logprobs") or choice_obj.logprobs is None:
                        raise ValueError("Logprobs not returned in response.")

                    all_tokens = getattr(choice_obj.logprobs, "tokens", None)
                    all_logprobs = getattr(choice_obj.logprobs, "token_logprobs", None)

                    if all_tokens is None or all_logprobs is None:
                        raise ValueError("Logprobs response missing expected 'tokens' or 'token_logprobs' fields.")

                    if len(all_tokens) != len(prompt_tokens) + len(completion_tokens):
                        raise ValueError(
                            f"Token count mismatch: tokens in response ({len(all_tokens)}) != prompt+completion "
                            f"({len(prompt_tokens) + len(completion_tokens)})"
                        )

                    # Sum logprobs for the completion portion
                    choices_log_probs[choice] = sum(all_logprobs[len(prompt_tokens) :])
                    choices_sequence_positions[choice] = len(completion_tokens)

                except Exception as e:
                    error = Error(error_class=e.__class__.__name__, message=str(e), traceback=traceback.format_exc())
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

    def __del__(self) -> None:
        if hasattr(self, "_client"):
            try:
                self._client.close()
            except Exception:
                pass


class OpenAI_gpt_4o_mini(OpenAIModel):
    LLM_NAME = "gpt-4o-mini-2024-07-18"

    def __init__(
        self,
        formatter: BaseFormatter | None = None,
        temperature: float | None = None,
        api_key: str | None = None,
        organization: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__(
            model_name=self.LLM_NAME,
            formatter=formatter,
            temperature=temperature,
            api_key=api_key,
            organization=organization,
            base_url=base_url,
        )


class OpenAI_davinci_002(OpenAIModel):
    LLM_NAME = "davinci-002"
    DEFAULT_FORMATTER = ConcatFormatter

    def __init__(
        self,
        formatter: BaseFormatter | None = None,
        temperature: float | None = None,
        api_key: str | None = None,
        organization: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__(
            model_name=self.LLM_NAME,
            formatter=formatter,
            temperature=temperature,
            api_key=api_key,
            organization=organization,
            base_url=base_url,
        )


class OpenAI_gpt_4o_mini_with_ConcatFormatter(OpenAI_gpt_4o_mini):
    DEFAULT_FORMATTER = ConcatFormatter


class Deepseek_chat(OpenAIModel):
    # Deepseek API: https://api-docs.deepseek.com/quick_start/pricing
    LLM_NAME = "deepseek-chat"  # DeepSeek-V3.2-Exp (Non-thinking Mode)
    API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

    def __init__(
        self,
        formatter: BaseFormatter | None = None,
        temperature: float | None = None,
        api_key: str | None = None,
        organization: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__(
            model_name=self.LLM_NAME,
            formatter=formatter,
            temperature=temperature,
            api_key=api_key,
            organization=organization,
            base_url="https://api.deepseek.com",
        )

    def _get_encoder(self) -> Tokenizer:
        return AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3.2-Exp")

    def _count_tokens(self, text: str) -> int:
        return len(self._encoder.encode(text))


class Deepseek_chat_with_formatter(OpenAIModel):
    LLM_NAME = "deepseek-chat"  # DeepSeek-V3.2-Exp (Non-thinking Mode)
    API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    DEFAULT_FORMATTER = partial(HFFormatter, "deepseek-ai/DeepSeek-V3.2-Exp")
    """
        <｜begin▁of▁sentence｜><｜User｜>Question: What color is the night sky?
        <｜Assistant｜></think>Answer:
    """

    def __init__(
        self,
        formatter: BaseFormatter | None = None,
        temperature: float | None = None,
        api_key: str | None = None,
        organization: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__(
            model_name=self.LLM_NAME,
            formatter=formatter,
            temperature=temperature,
            api_key=api_key,
            organization=organization,
            base_url="https://api.deepseek.com/beta",
        )

    def _get_encoder(self) -> Tokenizer:
        return AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3.2-Exp")

    def _count_tokens(self, text: str) -> int:
        return len(self._encoder.encode(text))


class Deepseek_reasoner(Deepseek_chat):
    LLM_NAME = "deepseek-reasoner"  # DeepSeek-V3.2-Exp (Thinking Mode)
    # multi-round conversations for reasoning model documented here:
    # https://api-docs.deepseek.com/guides/reasoning_model#api-example
    # does not support completion API
