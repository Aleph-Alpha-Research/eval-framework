from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_framework.constants import RED, RESET
from eval_framework.llm.aleph_alpha_api_llm import AlephAlphaAPIModel
from eval_framework.llm.base import BaseLLM
from eval_framework.llm.huggingface_llm import HFLLM
from eval_framework.llm.vllm_models import MistralVLLM, VLLMModel
from template_formatting.formatter import (
    ConcatFormatter,
    HFFormatter,
    Llama3Formatter,
)
from template_formatting.mistral_formatter import MagistralFormatter


class Bert(HFLLM):
    LLM_NAME = "google-bert/bert-base-uncased"
    DEFAULT_FORMATTER = ConcatFormatter()


class Pythia410m(HFLLM):
    LLM_NAME = "EleutherAI/pythia-410m"
    DEFAULT_FORMATTER = ConcatFormatter()


class SmolLM135M(HFLLM):
    LLM_NAME = "HuggingFaceTB/SmolLM-135M"
    DEFAULT_FORMATTER = ConcatFormatter()


class Smollm135MInstruct(HFLLM):
    LLM_NAME = "HuggingFaceTB/SmolLM-135M-Instruct"
    DEFAULT_FORMATTER = ConcatFormatter()


class SmolLM_1_7B_Instruct(HFLLM):
    LLM_NAME = "HuggingFaceTB/SmolLM-1.7B-Instruct"
    DEFAULT_FORMATTER = ConcatFormatter()


class Qwen3_0_6B_VLLM(VLLMModel):
    LLM_NAME = "Qwen/Qwen3-0.6B"
    DEFAULT_FORMATTER = HFFormatter(LLM_NAME, chat_template_kwargs={"enable_thinking": True})


class Qwen3_0_6B_VLLM_No_Thinking(VLLMModel):
    LLM_NAME = "Qwen/Qwen3-0.6B"
    DEFAULT_FORMATTER = HFFormatter(LLM_NAME, chat_template_kwargs={"enable_thinking": False})


class Qwen3_1_7B_VLLM_No_Thinking(VLLMModel):
    LLM_NAME = "Qwen/Qwen3-1.7B"
    DEFAULT_FORMATTER = HFFormatter(LLM_NAME, chat_template_kwargs={"enable_thinking": False})


class Qwen3_8B_VLLM_No_Thinking(VLLMModel):
    LLM_NAME = "Qwen/Qwen3-8B"
    DEFAULT_FORMATTER = HFFormatter(LLM_NAME, chat_template_kwargs={"enable_thinking": False})


class Qwen3_4B_VLLM_No_Thinking(VLLMModel):
    LLM_NAME = "Qwen/Qwen3-4B"
    DEFAULT_FORMATTER = HFFormatter(LLM_NAME, chat_template_kwargs={"enable_thinking": False})


class Qwen3_14B_VLLM_No_Thinking(VLLMModel):
    LLM_NAME = "Qwen/Qwen3-14B"
    DEFAULT_FORMATTER = HFFormatter(LLM_NAME, chat_template_kwargs={"enable_thinking": False})


class Qwen3_32B_VLLM_No_Thinking(VLLMModel):
    LLM_NAME = "Qwen/Qwen3-32B"
    DEFAULT_FORMATTER = HFFormatter(LLM_NAME, chat_template_kwargs={"enable_thinking": False})


class Qwen3_30B_A3B_VLLM_No_Thinking(VLLMModel):
    LLM_NAME = "Qwen/Qwen3-30B-A3B"
    DEFAULT_FORMATTER = HFFormatter(LLM_NAME, chat_template_kwargs={"enable_thinking": False})


class Qwen3_235B_A22B_VLLM_No_Thinking(VLLMModel):
    LLM_NAME = "Qwen/Qwen3-235B-A22B"
    DEFAULT_FORMATTER = HFFormatter(LLM_NAME, chat_template_kwargs={"enable_thinking": False})


class Qwen3_4B_VLLM_Reasoning(VLLMModel):
    LLM_NAME = "Qwen/Qwen3-4B"
    DEFAULT_FORMATTER = HFFormatter(LLM_NAME, chat_template_kwargs={"enable_thinking": True})


class Qwen3_8B_VLLM_Reasoning(VLLMModel):
    LLM_NAME = "Qwen/Qwen3-8B"
    DEFAULT_FORMATTER = HFFormatter(LLM_NAME, chat_template_kwargs={"enable_thinking": True})


class Qwen3_14B_VLLM_Reasoning(VLLMModel):
    LLM_NAME = "Qwen/Qwen3-14B"
    DEFAULT_FORMATTER = HFFormatter(LLM_NAME, chat_template_kwargs={"enable_thinking": True})


class Qwen3_32B_VLLM_Reasoning(VLLMModel):
    LLM_NAME = "Qwen/Qwen3-32B"
    DEFAULT_FORMATTER = HFFormatter(LLM_NAME, chat_template_kwargs={"enable_thinking": True})


class Qwen3_30B_A3B_VLLM_Reasoning(VLLMModel):
    LLM_NAME = "Qwen/Qwen3-30B-A3B"
    DEFAULT_FORMATTER = HFFormatter(LLM_NAME, chat_template_kwargs={"enable_thinking": True})


class Qwen3_0_6B(HFLLM):
    LLM_NAME = "Qwen/Qwen3-0.6B"
    DEFAULT_FORMATTER = HFFormatter(LLM_NAME, chat_template_kwargs={"enable_thinking": True})


class Qwen3_0_6B_No_Thinking(HFLLM):
    LLM_NAME = "Qwen/Qwen3-0.6B"
    DEFAULT_FORMATTER = HFFormatter(LLM_NAME, chat_template_kwargs={"enable_thinking": False})


class Qwen3_1_7B_No_Thinking(HFLLM):
    LLM_NAME = "Qwen/Qwen3-1.7B"
    DEFAULT_FORMATTER = HFFormatter(LLM_NAME, chat_template_kwargs={"enable_thinking": False})


class Qwen3_8B_No_Thinking(HFLLM):
    LLM_NAME = "Qwen/Qwen3-8B"
    DEFAULT_FORMATTER = HFFormatter(LLM_NAME, chat_template_kwargs={"enable_thinking": False})


class Qwen3_4B_No_Thinking(HFLLM):
    LLM_NAME = "Qwen/Qwen3-4B"
    DEFAULT_FORMATTER = HFFormatter(LLM_NAME, chat_template_kwargs={"enable_thinking": False})


class Qwen3_14B_No_Thinking(HFLLM):
    LLM_NAME = "Qwen/Qwen3-14B"
    DEFAULT_FORMATTER = HFFormatter(LLM_NAME, chat_template_kwargs={"enable_thinking": False})


class Qwen3_32B_No_Thinking(HFLLM):
    LLM_NAME = "Qwen/Qwen3-32B"
    DEFAULT_FORMATTER = HFFormatter(LLM_NAME, chat_template_kwargs={"enable_thinking": False})


class Qwen3_30B_A3B_No_Thinking(HFLLM):
    LLM_NAME = "Qwen/Qwen3-30B-A3B"
    DEFAULT_FORMATTER = HFFormatter(LLM_NAME, chat_template_kwargs={"enable_thinking": False})


class Phi3Mini4kInstruct(HFLLM):
    LLM_NAME = "microsoft/Phi-3-mini-4k-instruct"
    DEFAULT_FORMATTER = ConcatFormatter()


class Qwen1_5B(HFLLM):
    LLM_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    DEFAULT_FORMATTER = ConcatFormatter()


class MagistralVLLM(MistralVLLM):
    LLM_NAME = "mistralai/Magistral-Small-2506"
    DEFAULT_FORMATTER = MagistralFormatter("mistralai/Magistral-Small-2506")


class Viking_7b_API(AlephAlphaAPIModel):
    LLM_NAME = "viking-7b"
    DEFAULT_FORMATTER = ConcatFormatter()


class Poro_34bChat_API(AlephAlphaAPIModel):
    LLM_NAME = "poro-34b-chat"
    DEFAULT_FORMATTER = HFFormatter("LumiOpen/Poro-34B-chat")


class Pharia1_7B_Control_API(AlephAlphaAPIModel):
    LLM_NAME = "pharia-1-llm-7b-control"
    DEFAULT_FORMATTER = Llama3Formatter()


class Llama31_8B_HF(HFLLM):
    LLM_NAME = "meta-llama/Meta-Llama-3.1-8B"
    DEFAULT_FORMATTER = ConcatFormatter()


class Llama31_8B_API(AlephAlphaAPIModel):
    LLM_NAME = "llama-3.1-8b"
    DEFAULT_FORMATTER = ConcatFormatter()


class Llama31_8B_Instruct_API(AlephAlphaAPIModel):
    LLM_NAME = "llama-3.1-8b-instruct"
    DEFAULT_FORMATTER = Llama3Formatter()


class Llama31_70B_API(AlephAlphaAPIModel):
    LLM_NAME = "llama-3.1-70b"
    DEFAULT_FORMATTER = ConcatFormatter()


class Llama31_70B_Instruct_API(AlephAlphaAPIModel):
    LLM_NAME = "llama-3.1-70b-instruct"
    DEFAULT_FORMATTER = Llama3Formatter()


class Llama33_70B_Instruct_API(AlephAlphaAPIModel):
    LLM_NAME = "llama-3.3-70b-instruct"
    DEFAULT_FORMATTER = Llama3Formatter()


class Llama31_405B_Instruct_API(AlephAlphaAPIModel):
    LLM_NAME = "llama-3.1-405b-instruct-fp8"
    DEFAULT_FORMATTER = Llama3Formatter()


class Llama31_8B_Tulu_3_8B_SFT(AlephAlphaAPIModel):
    LLM_NAME = "tulu-3-8b-sft"
    DEFAULT_FORMATTER = HFFormatter("allenai/Llama-3.1-Tulu-3-8B-SFT")


class Llama31_8B_Tulu_3_8B(AlephAlphaAPIModel):
    LLM_NAME = "tulu-3-8b"
    DEFAULT_FORMATTER = HFFormatter("allenai/Llama-3.1-Tulu-3-8B")


class HFLLM_from_name(HFLLM):
    """
    A generic class to create HFLLM instances from a given model name.
    """

    def __init__(self, model_name: str | None = None, formatter: str = "Llama3Formatter", **kwargs: Any) -> None:
        if model_name is None:
            raise ValueError("model_name is required")

        self.LLM_NAME = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.LLM_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(self.LLM_NAME, device_map="auto")

        # Lazy formatter initialization - only create the one we need
        selected_formatter = self.get_formatter(formatter, model_name)

        print(f"{RED}[ Model initialized --------------------- {RESET}{self.LLM_NAME} {RED}]{RESET}")
        print(f"{RED}[ Formatter: {formatter} ]{RESET}")
        self._set_formatter(selected_formatter)


class HFLLM_from_wandb_registry(HFLLM):
    """
    A class to create HFLLM instances from registered models in Wandb registry.
    Downloads the model artifacts from Wandb and creates a local HFLLM instance.
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
        Initialize HFLLM from a Wandb registered model artifact.

        Args:
            artifact_name: Name of the artifact in the Wandb registry
            version: Version of the artifact to download (default: "latest")
            formatter: Type of formatter to use (default: "")
            **kwargs: Additional arguments passed to the parent class
        """
        print(f"{RED}[ Loading registered model from Wandb: {artifact_name}:{version} ]{RESET}")

        with self.download_wandb_artifact(artifact_name, version) as local_artifact_path:
            self.LLM_NAME = local_artifact_path
            self.artifact_name = artifact_name
            self.artifact_version = version
            selected_formatter = self.get_formatter(formatter, formatter_identifier)
            super().__init__(formatter=selected_formatter)

        print(f"{RED}[ Model initialized --------------------- {RESET}")
        print(f"{self.artifact_name}:{self.artifact_version} {RED}]{RESET}")
        print(f"{RED}[ Formatter: {formatter} ]{RESET}")


class VLLM_from_wandb_registry(VLLMModel):
    """
    A class to create VLLM instances from registered models in Wandb registry.
    Downloads the model artifacts from Wandb and creates a local VLLM instance.
    """

    LLM_NAME = ""

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

        self.artifact_name = artifact_name
        self.artifact_version = version
        selected_formatter = self.get_formatter(formatter, formatter_identifier)

        with self.download_wandb_artifact(artifact_name, version) as local_artifact_path:
            self.LLM_NAME = local_artifact_path

            super().__init__(formatter=selected_formatter, checkpoint_path=local_artifact_path, **kwargs)

        print(f"{RED}[ VLLM Model initialized ----------------- {RESET}")
        print(f"{self.artifact_name}:{self.artifact_version} {RED}]{RESET}")
        print(f"{RED}[ Formatter: {formatter} ]{RESET}")


class RegistryModel(BaseLLM):
    """
    This class pulls a model from the registry and uses one of two user-defined backends.
    Supports both HFLLM and VLLM inference backends.

    This class allows any registered model to be defined in config files with:

    llm_class: RegistryModel
    llm_args:
      artifact_name: "my-model-artifact"
      version: "v1"
      formatter: "ConcatFormatter"
      backend: "hfllm"  # or "vllm" (default: "hfllm")
      # Additional VLLM-specific args when backend="vllm":
      tensor_parallel_size: 1
      gpu_memory_utilization: 0.9
      batch_size: 1
    """

    def __init__(
        self, artifact_name: str, version: str = "latest", formatter: str = "", backend: str = "hfllm", **kwargs: Any
    ) -> None:
        """
        Initialize registry model

        Args:
            artifact_name: Name of the artifact in the Wandb registry
            version: Version of the artifact to download (default: "latest")
            formatter: Type of formatter to use (default: "")
            backend: Inference backend to use - "hfllm" or "vllm" (default: "hfllm")
            **kwargs: Additional arguments passed to the underlying model class
        """
        backend = backend.lower()

        if backend == "vllm":
            print(f"{RED}[ Creating VLLM backend for registry model ]{RESET}")
            self._model = VLLM_from_wandb_registry(
                artifact_name=artifact_name, version=version, formatter=formatter, **kwargs
            )

        elif backend == "hfllm":
            print(f"{RED}[ Creating HFLLM backend for registry model ]{RESET}")
            self._model = HFLLM_from_wandb_registry(
                artifact_name=artifact_name, version=version, formatter=formatter, **kwargs
            )

        else:
            raise ValueError(f"Unsupported backend: {backend}. Supported backends: 'hfllm', 'vllm'")

    def generate_from_messages(self, messages, stop_sequences=None, max_tokens=None, temperature=None):
        return self._model.generate_from_messages(
            messages=messages, stop_sequences=stop_sequences, max_tokens=max_tokens, temperature=temperature
        )

    def logprobs(self, samples):
        return self._model.logprobs(samples)

    @property
    def name(self) -> str:
        return self._model.name
