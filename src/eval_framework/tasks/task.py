from abc import ABC, abstractmethod
from collections.abc import Iterable
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from eval_framework.shared.types import BaseMetricContext, Completion
from template_formatting.formatter import Message

if TYPE_CHECKING:
    from eval_framework.llm.base import BaseLLM


class ResponseType(Enum):
    COMPLETION = "completion"
    LOGLIKELIHOODS = "loglikelihoods"


class Sample(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: int
    subject: str
    messages: list[Message]
    ground_truth: str | list[str] | None
    possible_completions: list[str] | None
    context: BaseMetricContext | list[BaseMetricContext] | None = None


class Task(ABC):
    """The contract a caller relies on to run an evaluation"""

    @abstractmethod
    def iterate_samples(self, num_samples: int | None = None) -> Iterable[Sample]: ...

    @abstractmethod
    def generate_completions(
        self,
        llm: "BaseLLM",
        samples: list[Sample],
        stop_sequences: list[str] | None = None,
        max_tokens: int | None = None,
        fail_on_error: bool = True,
    ) -> list[Completion]:
        """Run ``llm`` over ``samples`` and return their completions."""

    @abstractmethod
    def get_metadata(self) -> dict[str, str | list[str]]:
        """Descriptive metadata about the eval for result reporting."""

    @abstractmethod
    def get_response_type(self) -> ResponseType: ...

    @abstractmethod
    def display_name(self) -> str:
        """Human-readable display name. Is allowed to have special characters and whitespaces."""
        ...
