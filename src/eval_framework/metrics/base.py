from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from eval_framework.metrics.aggregators.aggregators import Aggregator
from eval_framework.shared.types import Error

if TYPE_CHECKING:
    from eval_framework.llm.base import BaseLLM

# Returns the run's LLM judge. A provider rather than the judge itself, so metrics can be
# constructed and run without a judge unless they actually call it.
type JudgeProvider = Callable[[], "BaseLLM"]


class MissingJudgeError(Exception):
    """A metric asked for the LLM judge in a run that was configured without one."""


def no_judge() -> "BaseLLM":
    raise MissingJudgeError(
        "This metric scores with an LLM judge, but the run was configured without one (no llm_judge_class set)."
    )


class MetricResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    metric_name: str
    value: float | None
    higher_is_better: bool
    llm_judge_prompt: str | None = None
    llm_judge_response: str | None = None
    code_execution_trace: str | None = None
    error: Error | None = None


class classproperty:
    def __init__(self, method: Any) -> None:
        self.method = method

    def __get__(self, instance: Any, cls: Any) -> Any:
        return self.method(cls)


class BaseMetric[Response](ABC):
    NAME: str
    KEYS: list[str] | None = None
    # The aggregator determines how to aggregate the results of a metric for a single
    # sample over multiple runs (LLM calls). We default to averaging and thus making
    # macro averaging the overall computation default.
    AGGREGATORS: list[Aggregator] = []
    # Responses graded at once. Defaults to 1: e.g. the math metrics time out via signal.alarm(),
    # which is main-thread-only and a single process-wide timer.
    MAX_WORKERS: int = 1

    @classproperty
    def NAMES(cls) -> list[str]:
        if cls.KEYS is None:
            return [cls.NAME]
        return [f"{cls.NAME}/{k}" for k in cls.KEYS]

    @abstractmethod
    def calculate(self, response: Response, *, judge: JudgeProvider = no_judge) -> list[MetricResult]:
        """Score one response. ``judge`` returns the run's LLM judge; calling it in a run configured
        without one raises ``MissingJudgeError``. Metrics that score without a judge ignore it."""
        raise NotImplementedError

    def prepare(self, responses: list[Response]) -> None:
        """Prepare metric before calculating per-response results.
        This is needed for metrics that depend on variables derived from all of the responses."""
