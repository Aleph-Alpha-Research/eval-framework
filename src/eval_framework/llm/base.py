from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Union

from eval_framework.shared.types import RawCompletion, RawLoglikelihood
from eval_framework.tasks.base import Sample
from template_formatting.formatter import Message

if TYPE_CHECKING:
    from template_formatting.formatter import ConcatFormatter, HFFormatter, Llama3Formatter
    from template_formatting.mistral_formatter import MagistralFormatter


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

    def get_formatter(
        self, formatter_name: str, model_identifier: str = ""
    ) -> Union["Llama3Formatter", "MagistralFormatter", "ConcatFormatter", "HFFormatter"]:
        """
        Create formatter instance based on formatter name.

        Args:
            formatter_name: Name of the formatter to create
            model_identifier: Model name/identifier for formatters that need it

        Returns:
            Formatter instance
        """
        match formatter_name:
            case "Llama3Formatter":
                from template_formatting.formatter import Llama3Formatter

                return Llama3Formatter()
            case "MistralFormatter":
                from eval_framework.llm.mistral import MagistralFormatter

                return MagistralFormatter(model_identifier)
            case "ConcatFormatter":
                from template_formatting.formatter import ConcatFormatter

                return ConcatFormatter()
            case "HFFormatter":
                from template_formatting.formatter import HFFormatter

                return HFFormatter(model_identifier)
            case _:
                supported = ["Llama3Formatter", "QwenFormatter", "MistralFormatter", "ConcatFormatter", "HFFormatter"]
                raise ValueError(f"Unsupported formatter: {formatter_name}. Supported formatters: {supported}")
