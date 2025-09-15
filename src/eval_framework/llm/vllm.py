import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal, Protocol, cast, override

import torch
import wandb
from vllm import LLM, SamplingParams
from vllm.inputs.data import TokensPrompt
from vllm.outputs import RequestOutput
from vllm.transformers_utils.tokenizer import get_tokenizer

from eval_framework.llm.base import BaseLLM
from eval_framework.shared.types import (
    ConcatCompression,
    Error,
    PromptTooLongException,
    RawCompletion,
    RawLoglikelihood,
)
from eval_framework.tasks.base import Sample
from eval_framework.tasks.utils import raise_errors
from eval_framework.utils.constants import RED, RESET
from template_formatting.formatter import BaseFormatter, HFFormatter, Message

logger = logging.getLogger(__name__)


@dataclass
class TokenizedContainer:
    """
    Container object to store tokens and formatted prompt
    """

    tokens: list[int]
    text: str


class VLLMTokenizerAPI[prompt_type: (list[Message], str)](ABC):
    """
    Protocol for tokenizer interface that defines required methods.
    Needed for type checking because of the vllm tokenizer.
    """

    @abstractmethod
    def encode_formatted_struct(self, struct: prompt_type) -> TokenizedContainer:
        """Encode prompt to token IDs."""
        pass

    @abstractmethod
    def encode_plain_text(self, text: str) -> TokenizedContainer:
        pass

    @property
    def chat_template(self) -> str | None:
        return None


class HFTokenizerProtocol(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode text to token IDs."""
        ...

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs to text."""
        ...

    @property
    def chat_template(self) -> str | None:
        """Chat template for the tokenizer."""
        ...


class VLLMTokenizer(VLLMTokenizerAPI[str]):
    def __init__(self, target_mdl: str | Path) -> None:
        self.tokenizer = cast(HFTokenizerProtocol, get_tokenizer(target_mdl))

    def _encode_text(self, text: str) -> TokenizedContainer:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return TokenizedContainer(tokens=tokens, text=text)

    def encode_formatted_struct(self, struct: str) -> TokenizedContainer:
        return self._encode_text(text=struct)

    def encode_plain_text(self, text: str) -> TokenizedContainer:
        return self._encode_text(text=text)

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)

    @override
    @property
    def chat_template(self) -> str | None:
        return self.tokenizer.chat_template


class VLLMModel(BaseLLM):
    LLM_NAME: str
    DEFAULT_FORMATTER: Callable[[], BaseFormatter] | None = None
    SEQ_LENGTH: int | None = None

    def __init__(
        self,
        formatter: BaseFormatter | None = None,
        max_model_len: int | None = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        batch_size: int = 1,
        checkpoint_path: str | Path | None = None,
        checkpoint_name: str | None = None,
        sampling_params: SamplingParams | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # Store the max_model_len for later use
        self._max_model_len = max_model_len
        self.checkpoint_name = checkpoint_name
        self.checkpoint_path = checkpoint_path

        model_args = {
            "model": str(self.checkpoint_path)
            if self.checkpoint_path
            else self.LLM_NAME,  # for some reason vllm needs a str here even though it can take an os.pathlike
            "max_model_len": max_model_len or self.SEQ_LENGTH,
            "max_num_seqs": batch_size,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            **kwargs,
        }

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.batch_size = batch_size
        self._tokenizer: None | VLLMTokenizerAPI = None

        self.model = LLM(**model_args, device=device)

        self.sampling_params: SamplingParams = self._process_sampling_params(sampling_params)

        logger.info(f"{RED}[ Model initialized --------------------- {RESET}{self.LLM_NAME} {RED}]{RESET}")
        self._set_formatter(formatter)

    def _process_sampling_params(self, sampling_params: SamplingParams | dict[str, Any] | None) -> SamplingParams:
        processed_sampling_params: SamplingParams | None = None
        if isinstance(sampling_params, dict):
            processed_sampling_params = SamplingParams(**sampling_params)
            logger.info(f"Converted sampling_params dict to SamplingParams: {processed_sampling_params}")
        elif sampling_params is not None:
            processed_sampling_params = sampling_params
        else:
            processed_sampling_params = self.model.get_default_sampling_params()

        return processed_sampling_params

    def _set_formatter(self, formatter: BaseFormatter | None = None) -> None:
        if formatter is not None:
            self._formatter = formatter
        elif self.DEFAULT_FORMATTER is not None:
            self._formatter = self.DEFAULT_FORMATTER()
        elif self.tokenizer.chat_template is not None:
            self._formatter = HFFormatter(self.checkpoint_path or self.LLM_NAME)
        else:
            raise ValueError("No formatter specified and no default formatter available.")

        logger.info(
            f"{RED}[ Using default formatter --------------------- {RESET}{self._formatter.__class__.__name__} {RED}]{RESET}"  # noqa: E501
        )

    @property
    def tokenizer(self) -> VLLMTokenizerAPI:
        if self._tokenizer is None:
            self._tokenizer = VLLMTokenizer(target_mdl=self.checkpoint_path or self.LLM_NAME)
        return self._tokenizer

    def count_tokens(self, text: str, /) -> int:
        return len(self.tokenizer.encode_plain_text(text).tokens)

    @property
    def formatter_output_mode(self) -> Literal["string", "list"]:
        return "string"

    @property
    def name(self) -> str:
        if self.checkpoint_name:
            return f"{self.__class__.__name__}_checkpoint_{self.checkpoint_name}"
        return self.__class__.__name__

    def build_redis_key_from_prompt_objs(
        self, prompt_objs: list[TokenizedContainer], sampling_params: SamplingParams
    ) -> Any:
        """
        Build a redis key from a list of prompt objects and sampling parameters.
        TokenizedContainers are not serializable so we just pass the tokens and sampling params.
        """
        return ([obj.tokens for obj in prompt_objs], sampling_params)

    def generate_from_messages(
        self,
        messages: list[Sequence[Message]],
        stop_sequences: list[str] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> list[RawCompletion]:
        raw_completions: list[RawCompletion | None] = [None] * len(messages)
        prompt_objs = []
        prompt_info = []

        sampling_params = self._resolve_sampling_params(self.sampling_params, max_tokens, stop_sequences, temperature)

        for i, single_messages in enumerate(messages):
            output_mode = self.formatter_output_mode
            prompt: str | list[Message] = self._formatter.format(single_messages, output_mode=output_mode)
            prompt_obj: TokenizedContainer = self.tokenizer.encode_formatted_struct(prompt)
            prompt_token_count = len(prompt_obj.tokens)

            max_tokens_to_generate = self.max_seq_length - prompt_token_count

            if max_tokens is not None:
                max_tokens_to_generate = min(max_tokens_to_generate, max_tokens)

            if max_tokens_to_generate < 1:
                if raise_errors():
                    raise PromptTooLongException("Prompt exceeded context size.")

                raw_completions[i] = RawCompletion(
                    prompt=prompt_obj.text,
                    prompt_sequence_positions=prompt_token_count,
                    completion="",
                    completion_sequence_positions=0,
                    raw_completion_error=Error(
                        error_class=PromptTooLongException.__name__,
                        message="Prompt exceeded context size.",
                        traceback="",
                    ),
                )
                continue

            prompt_objs.append(prompt_obj)
            prompt_info.append((i, single_messages))

        if prompt_objs:
            model_outputs = self._model_generate(prompt_objs=prompt_objs, sampling_params=sampling_params)

            for (original_index, single_messages), prompt_obj, output in zip(prompt_info, prompt_objs, model_outputs):
                raw_completions[original_index] = RawCompletion(
                    prompt=prompt_obj.text,
                    prompt_sequence_positions=len(output.prompt_token_ids) if output.prompt_token_ids else 0,
                    concat_compression=ConcatCompression.calculate(
                        single_messages, count_tokens=self.count_tokens, completion=output.outputs[0].text
                    ),
                    completion=output.outputs[0].text,
                    completion_sequence_positions=len(output.outputs[0].token_ids)
                    if output.outputs[0].token_ids
                    else 0,
                    raw_completion_error=None,
                )

        # Ensure all positions are filled (should never be None at this point)
        return cast(list[RawCompletion], raw_completions)

    @staticmethod
    def _resolve_sampling_params(
        sampling_params: SamplingParams,
        max_tokens: int | None,
        stop_sequences: list[str] | None,
        temperature: float | None,
    ) -> SamplingParams:
        sampling_params.max_tokens = max_tokens
        sampling_params.stop = stop_sequences
        if temperature is not None:
            logger.warning(
                f"Overriding sampling params temperature {sampling_params.temperature} with custom value {temperature}"
            )
            sampling_params.temperature = temperature
        else:
            logger.info(
                f"Using sampling params temperature value: {sampling_params.temperature} "
                f"as no custom temperature value was provided"
            )
        return sampling_params

    def _model_generate(
        self,
        prompt_objs: list[TokenizedContainer],
        sampling_params: SamplingParams,
    ) -> list[RequestOutput]:
        vllm_token_prompt = [TokensPrompt(prompt_token_ids=prompt_obj.tokens) for prompt_obj in prompt_objs]
        outputs = self.model.generate(vllm_token_prompt, sampling_params)

        return outputs

    def logprobs(self, samples: list[Sample]) -> list[RawLoglikelihood]:
        """Batched version of logprobs for improved performance."""
        results: list[RawLoglikelihood | None] = [None] * len(samples)

        # Collect all prompt-choice combinations
        batch_data = []
        sample_choice_indices = []  # Maps batch index back to (sample_index, choice)

        for sample_idx, sample in enumerate(samples):
            output_mode = self.formatter_output_mode
            prompt: str | list[Message] = self._formatter.format(sample.messages, output_mode=output_mode)
            prompt_obj: TokenizedContainer = self.tokenizer.encode_formatted_struct(prompt)

            choices_log_probs: dict[str, float] = {}
            choices_log_probs_sequence_positions: dict[str, int] = {}
            error: Error | None = None
            valid_choices = []

            for choice in sample.possible_completions or []:
                choice_obj: TokenizedContainer = self.tokenizer.encode_plain_text(choice)
                total_tokens_count = len(prompt_obj.tokens + choice_obj.tokens)

                if total_tokens_count > self.max_seq_length:
                    if raise_errors():
                        raise PromptTooLongException("Prompt exceeded context size.")
                    choices_log_probs = {}
                    choices_log_probs_sequence_positions = {}
                    error = Error(
                        error_class=PromptTooLongException.__name__,
                        message="Prompt and choice exceeded context size.",
                        traceback="",
                    )
                    break
                else:
                    batch_data.append((prompt_obj, choice_obj))
                    sample_choice_indices.append((sample_idx, choice))
                    valid_choices.append(choice)
                    choices_log_probs_sequence_positions[choice] = len(choice_obj.tokens)

            # If we had an error, store the result immediately
            if error is not None:
                results[sample_idx] = RawLoglikelihood(
                    prompt=prompt_obj.text,
                    prompt_sequence_positions=len(prompt_obj.tokens),
                    loglikelihoods=choices_log_probs,
                    loglikelihoods_sequence_positions=choices_log_probs_sequence_positions,
                    raw_loglikelihood_error=error,
                )
            else:
                results[sample_idx] = RawLoglikelihood(
                    prompt=prompt_obj.text,
                    prompt_sequence_positions=len(prompt_obj.tokens),
                    loglikelihoods=choices_log_probs,
                    loglikelihoods_sequence_positions=choices_log_probs_sequence_positions,
                    raw_loglikelihood_error=None,
                    concat_compression=ConcatCompression.calculate(
                        sample.messages, count_tokens=self.count_tokens, choices=valid_choices
                    ),
                )

        # Process batch if we have valid data
        if batch_data:
            batch_logprobs = self._model_log_probs(batch_data)

            # Distribute results back to samples
            for batch_idx, logprob in enumerate(batch_logprobs):
                sample_idx, choice = sample_choice_indices[batch_idx]
                result = results[sample_idx]
                if result is not None:
                    result.loglikelihoods[choice] = logprob

        return cast(list[RawLoglikelihood], results)

    def _model_log_probs(self, batch_data: list[tuple[TokenizedContainer, TokenizedContainer]]) -> list[float]:
        """Batched version of _model_log_probs for processing multiple prompt-choice pairs at once."""
        sampling_params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            prompt_logprobs=1,
            detokenize=False,
        )

        vllm_token_prompts = [
            TokensPrompt(prompt_token_ids=prompt_obj.tokens + choice_obj.tokens)
            for prompt_obj, choice_obj in batch_data
        ]

        try:
            outputs = self.model.generate(vllm_token_prompts, sampling_params)
        except Exception as e:
            raise e

        results = []
        for i, (prompt_obj, choice_obj) in enumerate(batch_data):
            output = outputs[i]
            assert output.prompt_logprobs is not None

            choice_logprobs = output.prompt_logprobs[-len(choice_obj.tokens) :]
            total_logprob = 0.0

            # VLLM guarantees the actual token's logprob is included in the output
            for j, token_id in enumerate(choice_obj.tokens):
                logprob_obj = choice_logprobs[j]
                assert logprob_obj is not None, f"logprob_obj is None: {logprob_obj}"
                logprob_value = getattr(logprob_obj[token_id], "logprob")
                assert logprob_value is not None, f"logprob_value is None: {logprob_value}"
                total_logprob += logprob_value

            results.append(total_logprob)

        return results

    @property
    def max_seq_length(self) -> int:
        """
        Returns the maximum sequence length for this model.
        Priority order:
        1. max_model_len parameter passed to __init__
        2. SEQ_LENGTH class attribute
        3. Model's actual max_model_len from config
        4. Default fallback of 2048
        """
        if self._max_model_len is not None:
            return self._max_model_len

        if self.SEQ_LENGTH is not None:
            return self.SEQ_LENGTH

        if hasattr(self.model, "llm_engine") and hasattr(self.model.llm_engine, "model_config"):
            return self.model.llm_engine.model_config.max_model_len

        return 2048

    @property
    def seq_length(self) -> int | None:
        """
        Kept for backward compatibility.
        """
        return self.max_seq_length


class _VLLM_from_wandb_registry(VLLMModel):
    """
    A class to create VLLM instances from registered models in Wandb registry.
    Downloads the model artifacts from Wandb and creates a local VLLM instance.
    """

    def __init__(
        self,
        artifact_name: str,
        version: str = "latest",
        formatter: str = "",
        formatter_identifier: str = "",
        **kwargs: Any,
    ) -> None:
        """
        Initialize VLLM from a Wandb registered model artifact.

        Args:
            artifact_name: Name of the artifact in the Wandb registry
            version: Version of the artifact to download (default: "latest")
            formatter: Type of formatter to use (default: "")
            **kwargs: Additional arguments passed to VLLMModel
        """
        print(f"{RED}[ Loading registered model from Wandb for VLLM: {artifact_name}:{version} ]{RESET}")
        self.artifact_used = False

        selected_formatter = self.get_formatter(formatter, formatter_identifier)

        # Remove download_path from kwargs
        download_path = kwargs.pop("download_path", None)
        local_artifact_path = self.download_wandb_artifact(
            artifact_name, version, user_supplied_download_path=download_path
        )
        # Set LLM_NAME to local path which VLLM can use directly
        self.LLM_NAME = str(local_artifact_path)
        super().__init__(
            formatter=selected_formatter,
            checkpoint_path=local_artifact_path,
            checkpoint_name=f"{artifact_name}/{version}",
            **kwargs,
        )

        print(f"{RED}[ VLLM Model initialized ----------------- {RESET}")
        print(f"{artifact_name}:{version} {RED}]{RESET}")
        print(f"{RED}[ Formatter: {formatter} ]{RESET}")

    def use_artifact(self) -> None:
        if self.artifact_used is False:
            wandb.use_artifact(self.artifact)
            self.artifact_used = True

    def generate_from_messages(
        self,
        messages: list[Sequence[Message]],
        stop_sequences: list[str] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> list[RawCompletion]:
        # use artifact should only be called once per run
        self.use_artifact()
        return super().generate_from_messages(messages, stop_sequences, max_tokens, temperature)

    def logprobs(self, samples: list[Sample]) -> list[RawLoglikelihood]:
        self.use_artifact()
        return super().logprobs(samples)


class Qwen3_0_6B_VLLM(VLLMModel):
    LLM_NAME = "Qwen/Qwen3-0.6B"
    DEFAULT_FORMATTER = partial(HFFormatter, LLM_NAME, chat_template_kwargs={"enable_thinking": True})


class Qwen3_0_6B_VLLM_No_Thinking(VLLMModel):
    LLM_NAME = "Qwen/Qwen3-0.6B"
    DEFAULT_FORMATTER = partial(HFFormatter, LLM_NAME, chat_template_kwargs={"enable_thinking": False})
