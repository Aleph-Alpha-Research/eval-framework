import traceback

from eval_framework.llm.base import BaseLLM
from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.shared.types import Completion, Error


class BaseLLMJudgeMetric(BaseMetric[Completion]):
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
