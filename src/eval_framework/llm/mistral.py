import math

from functools import partial
from pathlib import Path
from typing import Any, Literal, override

from vllm import SamplingParams

from eval_framework.llm.vllm import TokenizedContainer, VLLMModel, VLLMTokenizerAPI
from template_formatting.formatter import BaseFormatter, Message
from template_formatting.mistral_formatter import MagistralFormatter, MistralSerializer

__all__ = [
    "MistralAdapter",
    "MistralVLLM",
]


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


class MistralVLLM(VLLMModel):
    def __init__(
        self,
        # Model source (3 options: file path, HuggingFace model name, Wandb artifact name):
        checkpoint_path: str | Path | None = None,
        model_name: str | None = None,
        artifact_name: str | None = None,
        # Formatter (2 options):
        formatter: BaseFormatter | None = None,
        formatter_name: str | None = None,
        formatter_kwargs: dict[str, Any] | None = None,
        # Explicit name for the `name` property:
        checkpoint_name: str | None = None,
        # VLLM parameters (not complete):
        max_model_len: int | None = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        batch_size: int = 1,
        sampling_params: SamplingParams | dict[str, Any] | None = None,
        bytes_per_token: int | None = None,
        **kwargs: Any,
    ) -> None:
        model_args = {"tokenizer_mode": "mistral", "config_format": "mistral", "load_format": "mistral"}
        super().__init__(
            checkpoint_path=checkpoint_path,
            model_name=model_name,
            artifact_name=artifact_name,
            formatter=formatter,
            formatter_name=formatter_name,
            formatter_kwargs=formatter_kwargs,
            checkpoint_name=checkpoint_name,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            batch_size=batch_size,
            sampling_params=sampling_params,
            bytes_per_token_scalar=bytes_per_token_scalar,
            **{**model_args, **kwargs},
        )

    @override
    @property
    def tokenizer(self) -> VLLMTokenizerAPI:
        if self._tokenizer is None:
            self._tokenizer = MistralAdapter(target_mdl=self.LLM_NAME)
        return self._tokenizer

    @property
    def formatter_output_mode(self) -> Literal["string", "list"]:
        """Determine the correct output mode for the formatter based on tokenizer type."""
        return "list"


class MagistralVLLM(MistralVLLM):
    LLM_NAME = "mistralai/Magistral-Small-2506"
    DEFAULT_FORMATTER = partial(MagistralFormatter, "mistralai/Magistral-Small-2506")
