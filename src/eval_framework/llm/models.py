"""This is just a default model file with some small models for testing.

Please define your own model file externally and pass it to the eval-framework entrypoint
to use it.
"""

from collections.abc import Sequence
from typing import Any

from eval_framework.constants import RED, RESET
from eval_framework.llm.base import BaseLLM
from eval_framework.llm.huggingface import HFLLM
from eval_framework.llm.vllm import VLLMModel
from eval_framework.shared.types import RawCompletion, RawLoglikelihood
from eval_framework.tasks.base import Sample
from eval_framework.utils import is_extra_installed
from template_formatting.formatter import (
    Message,
)

if is_extra_installed(extra="transformers"):
    from eval_framework.llm.huggingface import Pythia410m, SmolLM135M, Smollm135MInstruct, Qwen3_0_6B  # noqa F401

if is_extra_installed("mistral"):
    from eval_framework.llm.mistral import MagistralVLLM  # noqa F401

if is_extra_installed("vllm"):
    from eval_framework.llm.vllm import Qwen3_0_6B_VLLM, Qwen3_0_6B_VLLM_No_Thinking  # noqa F401


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
        download_path = str(kwargs.pop("download_path", None)) if kwargs.get("download_path") else None
        with self.download_wandb_artifact(
            artifact_name, version, user_supplied_download_path=download_path
        ) as local_artifact_path:
            self.LLM_NAME = local_artifact_path
            self.artifact_name = artifact_name
            self.artifact_version = version
            selected_formatter = self.get_formatter(formatter, formatter_identifier)
            super().__init__(formatter=selected_formatter, **kwargs)

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

        download_path = (
            str(kwargs.pop("download_path", None)) if kwargs.get("download_path") else None
        )  # Remove download_path from kwargs
        with self.download_wandb_artifact(
            artifact_name, version, user_supplied_download_path=download_path
        ) as local_artifact_path:
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
        self._model: BaseLLM
        self.configure_model_backend(artifact_name, version, formatter, backend, kwargs)

    def configure_model_backend(
        self, artifact_name: str, version: str, formatter: str, backend: str, kwargs: Any
    ) -> None:
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

    def generate_from_messages(
        self,
        messages: list[Sequence[Message]],
        stop_sequences: list[str] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> list[RawCompletion]:
        return self._model.generate_from_messages(
            messages=messages, stop_sequences=stop_sequences, max_tokens=max_tokens, temperature=temperature
        )

    def logprobs(self, samples: list[Sample]) -> list[RawLoglikelihood]:
        return self._model.logprobs(samples)

    @property
    def name(self) -> str:
        return self._model.name
