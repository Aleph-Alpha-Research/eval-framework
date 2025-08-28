import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any, Generic, List, Literal, Optional, Protocol, Sequence, TypeVar, cast, override
import boto3

import torch
from vllm import LLM, SamplingParams
from vllm.inputs.data import TokensPrompt
from vllm.outputs import RequestOutput
from vllm.transformers_utils.tokenizer import get_tokenizer

from eval_framework.constants import RED, RESET
from eval_framework.llm.base import BaseLLM
from eval_framework.shared.types import Error, PromptTooLongException, RawCompletion, RawLoglikelihood
from eval_framework.tasks.base import Sample
from eval_framework.tasks.utils import raise_errors, redis_cache
from template_formatting.formatter import BaseFormatter, HFFormatter, Message
from template_formatting.mistral_formatter import MistralSerializer

logger = logging.getLogger(__name__)


@dataclass
class TokenizedContainer:
    """
    Container object to store tokens and formatted prompt
    """

    tokens: list[int]
    text: str


prompt_type = TypeVar("prompt_type", list[Message], str)


class VLLMTokenizerAPI(ABC, Generic[prompt_type]):
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


class MistralAdapter(VLLMTokenizerAPI[list[Message]]):
    def __init__(self, target_mdl: str) -> None:
        self.serializer = MistralSerializer(llm_target=target_mdl)
        self.tokenizer = self.serializer.get_tokenizer()

    def encode_formatted_struct(self, struct: list[Message]) -> TokenizedContainer:
        mistral_msg_lst = self.serializer.convert_from_aa(msg_lst=struct)
        mistral_request = self.serializer.build_mistral_request(mistral_msg_lst=mistral_msg_lst)
        mistral_tokenized_obj = self.tokenizer.encode_instruct(mistral_request)
        return TokenizedContainer(tokens=mistral_tokenized_obj.tokens, text=mistral_tokenized_obj.text)

    def encode_plain_text(self, text: str) -> TokenizedContainer:
        choice_tokens = self.tokenizer.tokenizer.encode(text, False, False)
        return TokenizedContainer(tokens=choice_tokens, text=text)


class VLLMTokenizer(VLLMTokenizerAPI[str]):
    def __init__(self, target_mdl: str) -> None:
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
    DEFAULT_FORMATTER: BaseFormatter | None = None
    SEQ_LENGTH: int | None = None

    def __init__(
        self,
        formatter: BaseFormatter | None = None,
        max_model_len: int | None = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        batch_size: int = 1,
        checkpoint_path: str | None = None,
        checkpoint_name: str | None = None,
        sampling_params: SamplingParams | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # Store the max_model_len for later use
        self._max_model_len = max_model_len
        self.checkpoint_name = checkpoint_name
        self.checkpoint_path = checkpoint_path

        model_args = {
            "model": self.checkpoint_path or self.LLM_NAME,
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
            self._formatter = self.DEFAULT_FORMATTER
        elif self.tokenizer.chat_template is not None:
            self._formatter = HFFormatter(self.LLM_NAME)
        else:
            raise ValueError("No formatter specified and no default formatter available.")

        logger.info(
            f"{RED}[ Using default formatter --------------------- {RESET}{self._formatter.__class__.__name__} {RED}]{RESET}"  # noqa: E501
        )

    @property
    def tokenizer(self) -> VLLMTokenizerAPI:
        if self._tokenizer is None:
            self._tokenizer = VLLMTokenizer(target_mdl=self.LLM_NAME)
        return self._tokenizer

    def _get_formatter_output_mode(self) -> Literal["string", "list"]:
        """Determine the correct output mode for the formatter based on tokenizer type."""
        if isinstance(self.tokenizer, MistralAdapter):
            return "list"
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
        valid_indices = []

        sampling_params = self._resolve_sampling_params(self.sampling_params, max_tokens, stop_sequences, temperature)

        for i, single_messages in enumerate(messages):
            output_mode = self._get_formatter_output_mode()
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
            valid_indices.append(i)

        if prompt_objs:
            model_outputs = self._model_generate(
                prompt_objs=prompt_objs,
                sampling_params=sampling_params,
            )

            for j, output in enumerate(model_outputs):
                original_index = valid_indices[j]
                raw_completions[original_index] = RawCompletion(
                    prompt=prompt_objs[j].text,
                    prompt_sequence_positions=len(output.prompt_token_ids) if output.prompt_token_ids else 0,
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
            output_mode = self._get_formatter_output_mode()
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

    @redis_cache(version_id="v10")
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


class MistralVLLM(VLLMModel):
    def __init__(
        self,
        formatter: BaseFormatter | None = None,
        max_model_len: int | None = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        batch_size: int = 1,
        checkpoint_path: str | None = None,
        checkpoint_name: str | None = None,
        sampling_params: SamplingParams | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        model_args = {"tokenizer_mode": "mistral", "config_format": "mistral", "load_format": "mistral"}
        super().__init__(
            formatter,
            max_model_len,
            tensor_parallel_size,
            gpu_memory_utilization,
            batch_size,
            checkpoint_path,
            checkpoint_name,
            sampling_params,
            **{**model_args, **kwargs},
        )

    @override
    @property
    def tokenizer(self) -> VLLMTokenizerAPI:
        if self._tokenizer is None:
            self._tokenizer = MistralAdapter(target_mdl=self.LLM_NAME)
        return self._tokenizer

# initial registered checkpoint loading class
# does the following
# from the model_path, loads checkpoint from s3 in a tmpdir
# handles verl checkpoints

class RegisteredCheckpointLoader(VLLMModel):
    def __init__(self, registered_model_name: str, version: str="latest") -> None:
        self.registered_model_name = registered_model_name
        self.version = version
    
    def download_from_s3(self, bucket, object, file):
        # get bucket structure in s3
        s3_client = boto3.client("s3")
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=object)
        if "Contents" not in response:
            print(f"No objects found for {object} in bucket {bucket}")
            return

        
       

    def load_checkpoint(self) -> None:
        pass

    def retrieve_reference(self) -> None:
        pass
    
