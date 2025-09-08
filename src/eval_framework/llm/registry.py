from collections.abc import Sequence
from typing import Any

from eval_framework.llm.base import BaseLLM
from eval_framework.shared.types import RawCompletion, RawLoglikelihood
from eval_framework.tasks.base import Sample
from eval_framework.utils.constants import RED, RESET
from template_formatting.formatter import Message


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
        self,
        artifact_name: str,
        version: str | None = None,
        formatter: str = "",
        backend: str = "huggingface",
        **kwargs: Any,
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
        assert version, "Version must be specified for registry models"
        self._model: BaseLLM
        self.configure_model_backend(artifact_name, version, formatter, backend, kwargs)

    def configure_model_backend(
        self, artifact_name: str, version: str, formatter: str, backend: str, kwargs: Any
    ) -> None:
        backend = backend.lower()

        match backend:
            case "vllm":
                from eval_framework.llm.vllm import _VLLM_from_wandb_registry

                print(f"{RED}[ Creating VLLM backend for registry model ]{RESET}")
                self._model = _VLLM_from_wandb_registry(
                    artifact_name=artifact_name, version=version, formatter=formatter, **kwargs
                )

            case "huggingface":
                from eval_framework.llm.huggingface import _HFLLM_from_wandb_registry

                print(f"{RED}[ Creating HFLLM backend for registry model ]{RESET}")

                self._model = _HFLLM_from_wandb_registry(
                    artifact_name=artifact_name, version=version, formatter=formatter, **kwargs
                )

            case _:
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
