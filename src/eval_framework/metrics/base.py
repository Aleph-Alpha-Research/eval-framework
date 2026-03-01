from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict

from eval_framework.metrics.aggregators.aggregators import Aggregator, IdentifierMean
from eval_framework.shared.types import Error


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
    # The aggregator determines how to aggregate the results of a metrics for a single
    # sample over multiple runs (LLM calls). We default to averaging and thus making
    # macro averaging the overall computatiion default.
    AGGREGATORS: list[Aggregator] = []

    @classproperty
    def NAMES(cls) -> list[str]:
        if cls.KEYS is None:
            return [cls.NAME]
        return [f"{cls.NAME}/{k}" for k in cls.KEYS]

    @abstractmethod
    def calculate(self, response: Response) -> list[MetricResult]:
        raise NotImplementedError
