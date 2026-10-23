from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

from openai.types.chat.completion_create_params import ResponseFormat

from eval_framework.contract import Sample
from eval_framework.shared.types import RawCompletion, RawLoglikelihood
from template_formatting.formatter import Message

if TYPE_CHECKING:
    from eval_framework.llm.retry import RetryPolicy


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
        response_format: ResponseFormat | None = None,
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
        response_format: ResponseFormat | None = None,
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
        response_format: ResponseFormat | None = None,
    ) -> list[RawCompletion]:
        """Generates a model response for each sample.

        Uses 'generate_from_samples' to generate responses if implemented,
        otherwise falls back to 'generate_from_messages'.
        """
        try:
            return self.generate_from_samples(samples, stop_sequences, max_tokens, temperature, top_p, response_format)
        except NotImplementedError:
            messages: list[Sequence[Message]] = [sample.messages for sample in samples]
            return self.generate_from_messages(
                messages, stop_sequences, max_tokens, temperature, top_p, response_format
            )

    def is_retryable(self, exc: Exception) -> bool:
        """Whether ``with_retry()`` repeats a call that raised ``exc``: by default any failure at all."""
        # Ask, don't tell: the wrapper queries the model instead of the model raising a typed error.
        # Deliberate, so that no adapter has to be rewritten around the retry mechanism to work with it.
        return not isinstance(exc, NotImplementedError)

    def with_retry(self, policy: RetryPolicy | None = None) -> BaseLLM:
        """This model with every generation and logprob call repeated while `is_retryable` allows."""
        from eval_framework.llm.retry import RetryingLLM, RetryPolicy

        return RetryingLLM(self, policy or RetryPolicy())

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
