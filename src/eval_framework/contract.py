from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from eval_framework.shared.types import BaseMetricContext, Completion
from template_formatting.formatter import BaseFormatter, Message

if TYPE_CHECKING:
    from eval_framework.llm.base import BaseLLM
    from eval_framework.metrics.base import BaseMetric


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


class Eval(ABC):
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


class Benchmark(ABC):
    """A benchmark is used to provide means of measuring model performance in a domain.

    Benchmark act as factories for `Eval`. They bind all the a prior known information and enrich it
    with the arguments provided at runtime to create concrete instances of `Eval` which are used to
    provide measurements of the models performance.
    """

    @abstractmethod
    def id(self) -> str:
        "Uniquely identifies the benchmark"

    @abstractmethod
    def response_type(self) -> ResponseType:
        """The benchmark's response type"""

    @abstractmethod
    def metrics(self) -> list[type["BaseMetric"]]:
        """The benchmark's metrics"""

    @abstractmethod
    def subjects(self) -> list[Any]:
        """Subjects of the benchmark"""

    @abstractmethod
    def display_name(self) -> str:
        """Human-readable display name. Is allowed to have special characters and whitespaces."""

    @abstractmethod
    def create(
        self,
        num_fewshot: int,
        custom_subjects: list[str] | None,
        custom_hf_revision: str | None,
        user_prompt_suffix: str | None = None,
        seed: int | None = None,
    ) -> Eval: ...

    @abstractmethod
    def markdown_doc(self, formatters: Sequence[BaseFormatter]) -> str:
        """Render the benchmarks's documentation as markdown."""
        ...
