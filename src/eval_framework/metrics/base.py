from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict

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


Response = TypeVar("Response")


class BaseMetric(ABC, Generic[Response]):
    NAME: str
    KEYS: list[str] | None = None

    @classproperty
    def NAMES(cls) -> list[str]:
        if cls.KEYS is None:
            return [cls.NAME]
        return [f"{cls.NAME}/{k}" for k in cls.KEYS]

    @abstractmethod
    def calculate(self, response: Response) -> list[MetricResult]:
        raise NotImplementedError
