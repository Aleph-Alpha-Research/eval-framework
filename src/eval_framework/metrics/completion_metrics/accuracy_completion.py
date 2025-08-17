from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.shared.types import Completion


class AccuracyCompletion(BaseMetric[Completion]):
    NAME = "Accuracy Completion"

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        ground_truths = response.ground_truth_list
        is_correct = any(response.completion == gt for gt in ground_truths)
        return [
            MetricResult(metric_name=self.NAME, value=float(is_correct), higher_is_better=True, error=response.error)
        ]
