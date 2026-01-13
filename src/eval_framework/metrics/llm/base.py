import functools
import logging
import traceback
from collections.abc import Callable
from typing import Any

from eval_framework.llm.base import BaseLLM
from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.shared.types import Completion, Error

logger = logging.getLogger(__name__)


def safe_metric_calculation(func: Callable) -> Callable:
    """
    Decorator that wraps LLM judge metric calculate methods with exception handling.

    This decorator ensures that exceptions during metric calculation don't crash the
    entire evaluation process. Instead, exceptions are caught and converted to
    MetricResult objects with appropriate error information.
    """

    @functools.wraps(func)
    def wrapper(self: Any, response: Completion) -> list[MetricResult]:
        # Get metric configuration from the class
        metric_names = getattr(self, "NAMES", [self.NAME])
        higher_is_better = getattr(self, "_higher_is_better", True)

        # Handle pre-existing response error
        if response.error is not None:
            logger.debug(f"Skipping {self.NAME} calculation - response already has error: {response.error}")
            return [
                MetricResult(
                    metric_name=name,
                    value=None,
                    higher_is_better=higher_is_better,
                    error=response.error,
                )
                for name in metric_names
            ]

        # Execute the actual calculation with exception handling
        try:
            return func(self, response)
        except Exception as e:
            logger.warning(f"LLM judge metric {self.NAME} failed with {e.__class__.__name__}: {e}")
            error = Error(
                error_class=e.__class__.__name__,
                message=str(e),
                traceback=traceback.format_exc(),
            )
            return [
                MetricResult(
                    metric_name=name,
                    value=None,
                    higher_is_better=higher_is_better,
                    error=error,
                )
                for name in metric_names
            ]

    return wrapper


class BaseLLMJudgeMetric(BaseMetric[Completion]):
    """Base class for LLM-as-judge metrics.

    Attributes:
        _higher_is_better: Override in subclasses where lower values are better (e.g., world knowledge).
                          Used by the safe_metric_calculation decorator for error results.
    """

    _higher_is_better: bool = True

    def __init__(self, llm_judge: BaseLLM, randomize_order: bool = False) -> None:
        self._llm_judge = llm_judge
        self._randomize_order = randomize_order

    def _create_metric_result(
        self,
        metric_name: str,
        higher_is_better: bool,
        value: float | None,
        llm_judge_prompt: str | None = None,
        llm_judge_response: str | None = None,
        code_execution_trace: str | None = None,
        error: Error | None = None,
    ) -> MetricResult:
        """Helper method to create MetricResult with consistent structure."""
        return MetricResult(
            metric_name=metric_name,
            value=value,
            higher_is_better=higher_is_better,
            llm_judge_prompt=llm_judge_prompt,
            llm_judge_response=llm_judge_response,
            code_execution_trace=code_execution_trace,
            error=Error(error_class=error.__class__.__name__, message=str(error), traceback=traceback.format_exc())
            if error
            else None,
        )
