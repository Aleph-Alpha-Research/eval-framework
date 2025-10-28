from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from eval_framework.shared.types import RawCompletion, RawLoglikelihood
from eval_framework.tasks.base import Sample
from template_formatting.formatter import BaseFormatter, Message


class BaseLLM(ABC):
    @property
    def name(self) -> str:
        """
        This property is used to name the results folder and identify the eval results.
        Overwrite this property in the subclass with e.g. the checkpoint name/huggingface model name."""
        return self.__class__.__name__

    @abstractmethod
    def generate_from_messages(
        self,
        messages: list[Sequence[Message]],
        stop_sequences: list[str] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> list[RawCompletion]:
        """
        stop_sequences and max_tokens are injected by the task if exist. They should be overwritten or
        extended with the properties of the model. This includes but is not limited to the stop tokens
        by the evaluated checkpoint (e.g. <|eot_id|> for an instruction finetuned Llama3.1, <|endoftext|>
        for a pretrained Llama3.1).

        This function is expected to raise errors which are caught and reported when running the eval.
        Please also make sure to raise an error in case of sequence length issues. We expect to always
        raise an error if something impedes the expected completion of a task.

        Important! The completion is expected to be detokenized and to NOT contain special tokens.

        Returns: List[RawCompletion]
        """
        raise NotImplementedError

    @abstractmethod
    def logprobs(self, samples: list[Sample]) -> list[RawLoglikelihood]:
        """
        This function is expected to raise errors which are caught and reported when running the eval.
        Please also make sure to raise an error in case of sequence length issues. We expect to always
        raise an error if something prevents the expected completion of a task.
        """
        raise NotImplementedError

    def generate(
        self,
        samples: list[Sample],
        stop_sequences: list[str] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> list[RawCompletion]:
        messages: list[Sequence[Message]] = [sample.messages for sample in samples]
        return self.generate_from_messages(messages, stop_sequences, max_tokens, temperature)

    def post_process_completion(self, completion: str, sample: Sample) -> str:
        """
        Model-specific post-processing of generated completions.

        Override this method to apply model-specific cleanup or transformations
        (e.g., removing specific artifacts such as reasoning traces, handling special tokens).

        Args:
            completion: The raw completion string from the model
            sample: The sample that was used to generate the completion

        Returns:
            The post-processed completion string
        """
        return completion

    def __del__(self) -> None:
        """
        Method for custom resource cleanup (particularly GPUs)
        """
        pass

    def _get_final_checkpoint(
        self, checkpoint_path: str | Path | None = None, model_name: str | None = None, artifact_name: str | None = None
    ) -> tuple[str | Path | None, str | None]:
        if (num_provided := sum(x is not None for x in [checkpoint_path, model_name, artifact_name])) == 0:
            if not getattr(self, "LLM_NAME", ""):
                raise ValueError("Either LLM_NAME, checkpoint_path, model_name, or artifact_name must be provided.")
            return None, None  # no argument given, so will use the LLM_NAME of the class
        elif num_provided > 1:
            raise ValueError("At most one of `checkpoint_path`, `model_name`, or `artifact_name` must be provided.")

        elif checkpoint_path is not None:
            return checkpoint_path, str(checkpoint_path)

        elif model_name is not None:
            return model_name, model_name

        else:
            from eval_framework.utils.file_ops import WandbFs

            assert artifact_name is not None
            artifact_base, version = artifact_name.split(":", 1) if ":" in artifact_name else (artifact_name, "latest")
            with WandbFs() as wandb_fs:
                self.artifact = wandb_fs.get_artifact(artifact_base, version)  # self.artifact being read in main()
                wandb_fs.download_artifact(self.artifact)
                file_root = wandb_fs.find_hf_checkpoint_root_from_path_list()
                if file_root is None:
                    raise ValueError(f"Could not find HuggingFace checkpoint in artifact {artifact_base}:{version}")
                return file_root, artifact_name

    def _get_final_formatter(
        self,
        formatter: BaseFormatter | None = None,
        formatter_name: str | None = None,
        formatter_kwargs: dict[str, Any] | None = None,
    ) -> BaseFormatter | None:
        if (num_provided := sum(x is not None for x in [formatter, formatter_name])) == 0:
            return None  # none given, so will use the default of the class
        elif num_provided > 1:
            raise ValueError("At most one of `formatter` or `formatter_name` must be provided.")

        if formatter:
            if formatter_kwargs:
                raise ValueError("Cannot provide `formatter_kwargs` when `formatter` is provided.")
            return formatter
        elif formatter_name:
            kwargs = formatter_kwargs or {}
            match formatter_name:
                case "Llama3Formatter":
                    from template_formatting.formatter import Llama3Formatter

                    return Llama3Formatter()
                case "MistralFormatter" | "MagistralFormatter":
                    from eval_framework.llm.mistral import MagistralFormatter

                    return MagistralFormatter(**kwargs)
                case "ConcatFormatter":
                    from template_formatting.formatter import ConcatFormatter

                    return ConcatFormatter()
                case "HFFormatter":
                    from template_formatting.formatter import HFFormatter

                    return HFFormatter(**kwargs)
                case _:
                    raise ValueError(f"Unsupported formatter: {formatter_name}.")
        return None
