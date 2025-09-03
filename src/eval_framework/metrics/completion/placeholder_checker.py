import re

from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.shared.types import BaseMetricContext, Completion, extract_context_metric


class PlaceholderCheckerMetricContext(BaseMetricContext):
    num_placeholders: int


class PlaceholderChecker(BaseMetric[Completion]):
    NAME = "Placeholder Check"

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        context = extract_context_metric(response, PlaceholderCheckerMetricContext)

        assert context.num_placeholders is not None, "Expected 'num_placeholders' in context"
        assert isinstance(context.num_placeholders, int), (
            f"'num_placeholders' has incorrect type: {type(context.num_placeholders)}"
        )

        placeholders = re.findall(r"\[.*?\]", response.completion)
        value = float(len(placeholders) >= context.num_placeholders)
        return [MetricResult(metric_name=self.NAME, value=value, higher_is_better=True, error=response.error)]
