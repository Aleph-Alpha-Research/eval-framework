from abc import ABC, abstractmethod
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from pathlib import Path

from eval_framework.shared.types import RawCompletion, RawLoglikelihood
from eval_framework.tasks.base import Sample
from template_formatting.formatter import Message


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
        top_p: float | None = None,
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
        Backends must also cut the completion at the first occurrence of a stop sequence and return the text
        before it; the stop sequence itself must not be part of the completion. Answer extraction relies on
        this and does no stripping of its own.

        Returns: List[RawCompletion]
        """
        raise NotImplementedError

    def generate_from_samples(
        self,
        samples: list[Sample],
        stop_sequences: list[str] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
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
        Backends must also cut the completion at the first occurrence of a stop sequence and return the text
        before it; the stop sequence itself must not be part of the completion. Answer extraction relies on
        this and does no stripping of its own.

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
        top_p: float | None = None,
    ) -> list[RawCompletion]:
        """Generates a model response for each sample.

        Uses 'generate_from_samples' to generate responses if implemented,
        otherwise falls back to 'generate_from_messages'.
        """
        try:
            return self.generate_from_samples(samples, stop_sequences, max_tokens, temperature, top_p)
        except NotImplementedError:
            messages: list[Sequence[Message]] = [sample.messages for sample in samples]
            return self.generate_from_messages(messages, stop_sequences, max_tokens, temperature, top_p)

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

    @contextmanager
    def _get_final_checkpoint(
        self, checkpoint_path: str | Path | None = None, model_name: str | None = None, artifact_name: str | None = None
    ) -> Generator[tuple[str | Path | None, str | None], None, None]:
        if (num_provided := sum(x is not None for x in [checkpoint_path, model_name, artifact_name])) == 0:
            if not getattr(self, "LLM_NAME", ""):
                raise ValueError("Either LLM_NAME, checkpoint_path, model_name, or artifact_name must be provided.")
            yield None, None  # no argument given, so will use the LLM_NAME of the class
        elif num_provided > 1:
            raise ValueError("At most one of `checkpoint_path`, `model_name`, or `artifact_name` must be provided.")

        elif checkpoint_path is not None:
            yield checkpoint_path, str(checkpoint_path)

        elif model_name is not None:
            yield model_name, model_name

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
                yield file_root, artifact_name
