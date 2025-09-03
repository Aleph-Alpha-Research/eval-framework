import ast

from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.shared.types import Completion


class ConcordanceIndex(BaseMetric[Completion]):
    NAME = "ConcordanceIndex"

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        ground_truths = [gt for gt in response.ground_truth_list if gt is not None]
        if not ground_truths:
            return [MetricResult(metric_name=self.NAME, value=0.0, higher_is_better=True, error=response.error)]

        concordance_count = max([calculate_concordance_index(gt, response.completion) for gt in ground_truths])
        return [
            MetricResult(metric_name=self.NAME, value=concordance_count, higher_is_better=True, error=response.error)
        ]


def calculate_concordance_index(
    ground_truth: str,
    completion: str,
) -> float:
    ground_truth_arr = ast.literal_eval(ground_truth)
    completion_arr = ast.literal_eval(completion)

    if len(ground_truth_arr) != len(completion_arr):
        return 0

    concordance_count = 0
    for gt, c in zip(ground_truth_arr, completion_arr):
        concordance_count += 1 if gt == c else 0

    return concordance_count / len(ground_truth_arr)
